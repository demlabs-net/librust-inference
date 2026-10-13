
// ── роутер MoE на устройстве: softmax + top-k (без синхронизации с CPU) ────
// Host-роутинг стоил 16 синхронизаций GPU→CPU на токен (по одной на слой) —
// каждая ждёт всю очередь GPU, и декод MoE упирался именно в это (44 против
// 208 tok/s у llama.cpp). Здесь всё считается кернелом, id уходят прямо в
// mv_id_* буферами — CPU видит только итоговый текст.
// Один threadgroup на токен: E логитов в threadgroup-память, softmax с
// вычитанием максимума, затем K проходов «взять лучший из невыбранных» с
// ничьей в пользу МЕНЬШЕГО индекса (как сортировка на хосте).
kernel void moe_route_topk(
    device const float* LOGITS [[buffer(0)]],   // [T, E]
    device uint* IDS [[buffer(1)]],             // [T, K]
    device float* WTS [[buffer(2)]],            // [T, K]
    constant uint& E [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& NORM [[buffer(5)]],
    constant float& SCALE [[buffer(6)]],
    threadgroup float* sh [[threadgroup(0)]],   // E значений (softmax in place)
    threadgroup float* sv [[threadgroup(1)]],   // 128
    threadgroup uint* si [[threadgroup(2)]],    // 128
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    device const float* lg = LOGITS + (ulong)tgid * E;
    device uint* ids = IDS + (ulong)tgid * K;
    device float* wts = WTS + (ulong)tgid * K;
    // Softmax — дословный kernel_soft_max_4 оркала (nth=32: один simdgroup,
    // float4-раскладка, simd_max/simd_sum-бабочки). Прежний 128-потоковый
    // скалярный вариант отличался деревом суммирования → ±1 ulp в весах
    // слотов → расхождение длинного префилла OLMoE (Ф8.1h).
    device const float4 * lg4 = (device const float4 *) lg;
    const uint ne4 = E / 4u;
    float4 lmax4 = -INFINITY;
    if (E % 4u == 0u) {
        for (uint i = lane; i < ne4; i += 32u) lmax4 = fmax(lmax4, lg4[i]);
        // MAX — как ggml-макрос ((a)>(b)?(a):(b)), НЕ fmax (NaN-семантика)
        #define MM_MAX(a, b) ((a) > (b) ? (a) : (b))
        const float lmax = MM_MAX(MM_MAX(lmax4[0], lmax4[1]), MM_MAX(lmax4[2], lmax4[3]));
        const float m = simd_max(lmax);
        if (lane == 0u) sv[1] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float m0 = sv[1];
        float4 lsum4 = 0.0f;
        for (uint i = lane; i < ne4; i += 32u) {
            const float4 e4 = exp(lg4[i] - m0);
            lsum4 += e4;
            sh[4u*i + 0u] = e4[0]; sh[4u*i + 1u] = e4[1];
            sh[4u*i + 2u] = e4[2]; sh[4u*i + 3u] = e4[3];
        }
        const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
        threadgroup_barrier(mem_flags::mem_none);
        const float sum = simd_sum(lsum);
        if (lane == 0u) sv[2] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = 1.0f / sv[2];
        for (uint i = lane; i < E; i += 128u) sh[i] *= inv;
    } else {
        for (uint i = lane; i < E; i += 128u) sh[i] = lg[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mx = -INFINITY;
        for (uint i = lane; i < E; i += 128u) mx = max(mx, sh[i]);
        sv[lane] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = 64u; st > 0u; st >>= 1u) {
            if (lane < st) sv[lane] = max(sv[lane], sv[lane + st]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float m = sv[0];
        float ssum = 0.0f;
        for (uint i = lane; i < E; i += 128u) {
            const float e = exp(sh[i] - m);
            sh[i] = e;
            ssum += e;
        }
        sv[lane] = ssum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = 64u; st > 0u; st >>= 1u) {
            if (lane < st) sv[lane] += sv[lane + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = (sv[0] > 0.0f) ? 1.0f / sv[0] : 0.0f;
        for (uint i = lane; i < E; i += 128u) sh[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // top-K: K проходов по невыбранным
    for (uint j = 0; j < K; ++j) {
        float bv = -INFINITY;
        uint bi = E;
        for (uint i = lane; i < E; i += 128u) {
            const float v = sh[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        sv[lane] = bv;
        si[lane] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = 64u; st > 0u; st >>= 1u) {
            if (lane < st) {
                const float ov = sv[lane + st];
                const uint oi = si[lane + st];
                if (ov > sv[lane] || (ov == sv[lane] && oi < si[lane])) {
                    sv[lane] = ov;
                    si[lane] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            ids[j] = si[0];
            wts[j] = sv[0];
            if (si[0] < E) sh[si[0]] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // нормировка top-k (clamp 6.1e-5, как build_moe_ffn) и масштаб
    if (NORM == 1u && lane == 0u) {
        float wsum = 0.0f;
        for (uint j = 0; j < K; ++j) wsum += wts[j];
        wsum = max(wsum, 6.103515625e-5f);
        for (uint j = 0; j < K; ++j) wts[j] = wts[j] / wsum;
    }
    if (lane == 0u && SCALE != 0.0f && SCALE != 1.0f) {
        for (uint j = 0; j < K; ++j) wts[j] *= SCALE;
    }
}

// ── matvec по ИДЕНТИФИКАТОРАМ экспертов (MoE, декод) ────────────────────────
// Порт идеи llama.cpp `kernel_mul_mv_id_*`: вместо k отдельных matvec'ов на
// токен (8 экспертов × 3 проекции × 16 слоёв = 384 вызова и столько же
// аллокаций буферов у нас) — ОДИН вызов на проекцию, внутри которого
// перебираются слоты top-k. Строка весов адресуется как
// B + (ids[j] * N + row) * sb_count * bsz.
//
// Два режима (шаблонные флаги):
//   PER_SLOT_IN  = 0 — вход общий: активации [K] (проекции up/gate);
//                = 1 — вход по слотам: [KSLOTS, K] (проекция down, вход —
//                      silu-произведение каждого слота);
//   WSUM         = 0 — выход по слотам БЕЗ весов: C[j * N + n] = dot (up/gate);
//                = 1 — выход суммы с весами роутера: C[n] += w[j] * dot (down).
//
// Порядок суммирования по K — r2-раскладка эталонного mul_mv (хелперы
// mv_q4k_*/mv_q6k_* в common.metal): эталонный kernel_mul_mv_id оборачивает
// ту же impl, что и mul_mat, поэтому mv_id побитово равен по-слотовому
// matvec'у (тест mv_id_matches_per_slot_matvec_bitwise). Пара строк на
// simdgroup (32 лейна на строку), 8 строк на threadgroup, слоты — цикл.
#define MVID_NR 8u
template <int KIND, int PER_SLOT_IN, int WSUM>
static inline void mv_id_impl(device const float* A, device const uint* IDS,
                              device const float* WID, device const uchar* B,
                              device float* C, uint K, uint N, uint KSLOTS,
                              uint tgid, uint sgid, uint lane)
{
    const uint bsz = (KIND == 0) ? 144u : 210u;
    const uint nb = K / 256u;
    const uint first_row = tgid * MVID_NR + sgid * 2u;
    if (first_row >= N) return;
    const bool two = first_row + 1u < N;
    const uint row_bytes = nb * bsz;

    float acc[2] = {0.0f, 0.0f};
    for (uint j = 0; j < KSLOTS; ++j) {
        const uint e = IDS[j];
        // WID читается только в режиме суммы (иначе буфер не привязан)
        const float w = (WSUM == 1) ? WID[j] : 0.0f;
        device const float* a = PER_SLOT_IN ? (A + (ulong)j * K) : A;
        device const uchar* x = B + ((ulong)e * N + first_row) * row_bytes;

        float sumf[2] = {0.f, 0.f};
        if (KIND == 0) {
            const MvQ4KDecomp d = mv_q4k_decomp(lane);
            for (uint ib = d.ix; ib < nb; ib += 4u) {
                MvQ4KY y;
                mv_q4k_y(a, ib, d, y);
                for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
                    mv_q4k_acc(x + (ulong)rr * row_bytes + (ulong)ib * 144u, d, y, sumf[rr]);
                }
            }
        } else {
            const MvQ6KDecomp d = mv_q6k_decomp(lane);
            for (uint ib = d.ix; ib < nb; ib += 2u) {
                float yl[16];
                mv_q6k_y(a, ib, d, yl);
                for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
                    mv_q6k_acc(x + (ulong)rr * row_bytes + (ulong)ib * 210u, d, yl, sumf[rr]);
                }
            }
        }
        for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
            const float v = simd_sum(sumf[rr]);
            if (lane == 0u) {
                if (WSUM == 1) {
                    // Эталон суммирует слоты ДВУМЯ отдельными кернелами
                    // (ggml_mul: p = w·v с округлением в f32, затем ggml_add):
                    // произведение округляется ДО сложения. Контракция в FMA
                    // здесь расходится с эталоном на 1–26 ULP на слой на токен
                    // — на чувствительных моделях это и есть разрыв паритета.
                    #pragma clang fp contract(off)
                    const float p = w * v;
                    acc[rr] = acc[rr] + p;
                } else {
                    C[(ulong)j * N + first_row + rr] = v;
                }
            }
        }
    }
    if (WSUM == 1 && lane == 0u) {
        for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
            C[first_row + rr] = acc[rr];
        }
    }
}

// Обёртки-ядра: явные атрибуты буферов (шаблонные kernel-инстанциации Metal
// требуют typedef decltype и путаются с несколькими параметрами шаблона).
#define MV_ID_KERNEL(NAME, KIND, PSL, WSUM)                                       \
kernel void NAME(                                                                 \
    device const float* A [[buffer(0)]],                                          \
    device const uint* IDS [[buffer(1)]],                                         \
    device const float* WID [[buffer(2)]],                                        \
    device const uchar* B [[buffer(3)]],                                          \
    device float* C [[buffer(4)]],                                                \
    constant uint& K [[buffer(5)]],                                               \
    constant uint& N [[buffer(6)]],                                               \
    constant uint& KS [[buffer(7)]],                                              \
    uint tgid [[threadgroup_position_in_grid]],                                   \
    uint sgid [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]])                                      \
{ mv_id_impl<KIND, PSL, WSUM>(A, IDS, WID, B, C, K, N, KS, tgid, sgid, lane); }

MV_ID_KERNEL(mv_id_q4_k, 0, 0, 0)
MV_ID_KERNEL(mv_id_q4_k_down, 0, 0, 1)
MV_ID_KERNEL(mv_id_q4_k_dpsl, 0, 1, 1)
MV_ID_KERNEL(mv_id_q6_k, 1, 0, 0)
MV_ID_KERNEL(mv_id_q6_k_down, 1, 0, 1)
MV_ID_KERNEL(mv_id_q6_k_dpsl, 1, 1, 1)

// ── mul_mm_id: непрямой sgemm (порт kernel_mul_mm_id_map0 + kernel_mul_mm_id) ──
// Эталон идёт этим путём при has_simdgroup_mm && ne00>=64 && ne21>=32 (число
// токенов ubatch) и K%32==0 (bc_inp=false). Порядок суммирования отличается
// от mv_id ⇒ без этого кернела префилл MoE (OLMoE ≥32 токена) не битуется.
// Каркас — побитово сверенный gemm_nt_q4_k_mm/gemm_nt_q6_k_mm, плюс id-
// косвенность: map0 строит по экспертам списки (token,slot), mm_id грузит
// строки src1 по ids.

typedef struct {
    int32_t  ne02;  // n_expert
    int32_t  ne10;  // (не используется — порядок полей как у оркала)
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;  // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} MmIdMap0Args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
} MmIdArgs;

template <short NE20>
static inline void mm_id_map0_impl(
        constant MmIdMap0Args & args,
        device const char * src2,
        device       char * htpe,
        device       char * hids,
        threadgroup  char * shmem,
        ushort tpitg, ushort ntg) {
    const short ide = tpitg; // expert id
    uint32_t n_all = 0;
    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);
            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*NE20;
            #pragma unroll(NE20)
            for (short i20 = 0; i20 < NE20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }
            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*NE20;
            short sel = 0;
            #pragma unroll(NE20)
            for (short i20 = 0; i20 < NE20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }
            ids_i32[n_all] = (i21 + t)*NE20 + sel - 1;
            n_all += sel > 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) htpe;
    tpe_u32[ide] = n_all;
}

#define MM_ID_MAP0_KERNEL(NAME, NE20)                                             \
kernel void NAME(                                                                 \
    constant MmIdMap0Args & args [[buffer(0)]],                                   \
    device const char * src2 [[buffer(1)]],                                       \
    device       char * htpe [[buffer(2)]],                                       \
    device       char * hids [[buffer(3)]],                                       \
    threadgroup  char * shmem [[threadgroup(0)]],                                 \
    ushort tpitg [[thread_position_in_threadgroup]],                              \
    ushort   ntg [[threads_per_threadgroup]])                                     \
{ mm_id_map0_impl<NE20>(args, src2, htpe, hids, shmem, tpitg, ntg); }

MM_ID_MAP0_KERNEL(mm_id_map0_ne20_1,  1)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_2,  2)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_4,  4)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_5,  5)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_6,  6)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_8,  8)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_10, 10)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_16, 16)
MM_ID_MAP0_KERNEL(mm_id_map0_ne20_22, 22)

// q6_K-деквант для mm-каркаса — дословный dequantize_q6_K (копия загрузчика
// gemm_nt_q6_k_mm; трогать сверенный кернел ради переиспользования не стали).
static inline void q6k_dequantize_mm(device const uchar * xb, short il, thread float4x4 & reg) {
    const uint qlo = 64u * (il / 8) + 32u * ((il / 2) & 1) + 16u * (il & 1);
    const uint qho = 128u + 32u * (il / 8) + 16u * (il & 1);
    device const ushort * ql = (device const ushort *)(xb + qlo);
    device const ushort * qh = (device const ushort *)(xb + qho);
    const float sc = (float)(char)xb[192u + (il % 2) + 2u * (il / 2)];
    const uint i2 = (il / 2) & 3;
    const float d_all = (float)load_half_unaligned(xb + 208u);
    const uint32_t kmask1 = i2 > 1 ? (i2 > 2 ? 0xC0C0C0C0u : 0x30303030u)
                                   : (i2 > 0 ? 0x0C0C0C0Cu : 0x03030303u);
    const uint32_t kmask2 = i2 > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint shr_h = i2 > 2 ? 2u : 0u;
    const uint shl_h = i2 > 1 ? 0u : (i2 > 0 ? 2u : 4u);
    const uint shr_l = i2 > 1 ? 4u : 0u;
    for (short i = 0; i < 4; ++i) {
        const uint32_t lo32 = (uint32_t)ql[2*i] | ((uint32_t)ql[2*i + 1] << 16);
        const uint32_t hi32 = (uint32_t)qh[2*i] | ((uint32_t)qh[2*i + 1] << 16);
        const uint32_t low  = lo32 & kmask2;
        const uint32_t high = hi32 & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 * ((float)(half)(q & 0xFFu)) - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00u)) - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000u)) - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000u)) - ml;
    }
}

template <int KIND> // 0 = Q4_K (144 Б/блок), 1 = Q6_K (210 Б/блок)
static inline void mm_id_impl(
        constant MmIdArgs & args,
        device const char * src0,
        device const char * src1,
        device const uint * htpe,
        device const int  * hids,
        device       char * dst,
        threadgroup  char * shmem,
        uint3 tgpig, ushort tiitg, ushort tiisg, ushort sgitg) {
    constexpr unsigned BLOCK = (KIND == 0) ? 144u : 210u;
    constexpr short  NL      = 16;   // QK_NL для K-квантов

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;          // эксперт
    const int r0 = tgpig.y * 64;     // строки весов
    const int r1 = tgpig.x * 32;     // выбранные токены

    const int neh1 = (int) htpe[im];
    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < 64) ? (short)(args.ne0 - r0) : 64;
    const short nr1 = (    neh1 - r1 < 32) ? (short)(    neh1 - r1) : 32;

    const short lr0 = ((short)tiitg/2) < nr0 ? ((short)tiitg/2) : nr0 - 1;
    const short lr1 = ((short)tiitg/4) < nr1 ? ((short)tiitg/4) : nr1 - 1;
    const short il0 = tiitg % 2;
    short il = il0;

    const int id = hids[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);

    device const uchar * x = (device const uchar *)(src0 + args.nb01*(r0 + lr0) + (uint64_t)im*args.nb02);

    const short iy = 8*(tiitg % 4);
    device const float * y = (device const float *)(src1 + args.nb12*i12 + args.nb11*i11 + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += 32) {
        float4x4 reg;
        if (KIND == 0) {
            q4k_dequantize_mm((device const q4k_block *) x, il, reg);
        } else {
            q6k_dequantize_mm(x, il, reg);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/2)/8;
            const short lx = (tiitg/2)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            // БЕЗ проверки границ (как у оркала в ветке декванта): позиция
            // элемента = блок x + группа il, а не loop_k + 16*il + i.
            *(sa + 64*ib + 8*ly + lx) = (half) reg[i/4][i%4];
        }

        {
            const short sx = (tiitg % 4);
            const short sy = (tiitg/4)/8;
            const short ly = (tiitg/4)%8;
            const short ib = 4*sx + sy;
            device const float4 * y4 = (device const float4 *)(y + loop_k);
            threadgroup half4 * dsh = (threadgroup half4 *)(sb + 64*ib + 8*ly);
            dsh[0] = half4(y4[0]);
            dsh[1] = half4(y4[1]);
        }

        il = (il + 2 < NL) ? (short)(il + 2) : (short)(il % 2);
        x  = (il < 2) ? x + BLOCK : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*64;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*64*(i/4), 64, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id2 = hids[im*args.ne21 + r1 + j];

        const short ide = id2 % args.ne20;
        const short idt = id2 / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + (uint64_t)idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*64;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

#define MM_ID_KERNEL(NAME, KIND)                                                  \
kernel void NAME(                                                                 \
    constant MmIdArgs & args [[buffer(0)]],                                       \
    device const char * src0 [[buffer(1)]],                                       \
    device const char * src1 [[buffer(2)]],                                       \
    device const uint * htpe [[buffer(3)]],                                       \
    device const int  * hids [[buffer(4)]],                                       \
    device       char * dst  [[buffer(5)]],                                       \
    threadgroup  char * shmem [[threadgroup(0)]],                                 \
    uint3   tgpig [[threadgroup_position_in_grid]],                               \
    ushort tiitg [[thread_index_in_threadgroup]],                                 \
    ushort tiisg [[thread_index_in_simdgroup]],                                   \
    ushort sgitg [[simdgroup_index_in_threadgroup]])                              \
{ mm_id_impl<KIND>(args, src0, src1, htpe, hids, dst, shmem, tgpig, tiitg, tiisg, sgitg); }

MM_ID_KERNEL(mm_id_q4_k, 0)
MM_ID_KERNEL(mm_id_q6_k, 1)

// Взвешенная сумма слотов: acc[t,d] += Σ_j w[t,k+j] · src[(t,k+j),d] — тот же
// порядок сложения (j по возрастанию, из нуля), что последовательные
// add_scaled в по-слотовом пути (побитово совпадает с аддитивным узлом оркала).
// Округление как у оркала: ggml_mul (произведение w·src с округлением) и
// ggml_add по слотам — два отдельных кернела. Без pragma clang FMA-контрактит
// s += w*src и теряет промежуточное округление (±1 ulp на 56% выходов).
#pragma clang fp contract(off)
kernel void moe_acc_slots(
    device const float * SRC [[buffer(0)]],
    device const float * W   [[buffer(1)]],
    device       float * ACC [[buffer(2)]],
    constant uint & T  [[buffer(3)]],
    constant uint & KS [[buffer(4)]],
    constant uint & D  [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    const uint td = T * D;
    if (tid >= td) {
        return;
    }
    const uint t  = tid / D;
    const uint dd = tid % D;
    float s = 0.0f;
    for (uint j = 0; j < KS; ++j) {
        s += W[t*KS + j] * SRC[((uint64_t)t*KS + j)*D + dd];
    }
    ACC[tid] += s;
}

// pragma contract(off) выше действует до конца TU — вернуть дефолт (moe.metal
// конкатенируется перед elementwise.metal).
#pragma clang fp contract(fast)
