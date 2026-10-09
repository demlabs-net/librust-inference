
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
    for (uint i = lane; i < E; i += 128u) sh[i] = lg[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // максимум
    float mx = -INFINITY;
    for (uint i = lane; i < E; i += 128u) mx = max(mx, sh[i]);
    sv[lane] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (lane < st) sv[lane] = max(sv[lane], sv[lane + st]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float m = sv[0];
    // exp и сумма
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
// Строки делят threadgroup'ы (16 строк на 128 потоков), слоты — цикл в
// регистрах: гонок за C нет ни в одном режиме.
#define MVID_NR 16u
template <int KIND, int PER_SLOT_IN, int WSUM>
static inline void mv_id_impl(device const float* A, device const uint* IDS,
                              device const float* WID, device const uchar* B,
                              device float* C, uint K, uint N, uint KSLOTS,
                              uint tgid, uint sgid, uint lane)
{
    const uint bsz = (KIND == 0) ? 144u : 210u;
    const uint sb_count = K / 256u;
    const uint r0 = tgid * MVID_NR + sgid * 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint j = 0; j < KSLOTS; ++j) {
        const uint e = IDS[j];
        // WID читается только в режиме суммы (иначе буфер не привязан)
        const float w = (WSUM == 1) ? WID[j] : 0.0f;
        device const float* a = PER_SLOT_IN ? (A + j * K) : A;
        for (uint rr = 0; rr < 4u; ++rr) {
            const uint row = r0 + rr;
            if (row >= N) break;
            device const uchar* rw = B + ((ulong)e * N + row) * sb_count * bsz;
            float part = 0.0f;
            for (uint sb = lane; sb < sb_count; sb += 32u) {
                device const uchar* blk = rw + sb * bsz;
                part += (KIND == 0) ? dot_q4_k(a + sb * 256u, blk)
                                    : dot_q6_k(a + sb * 256u, blk);
            }
            const float v = simd_sum(part);
            if (lane == 0u) {
                if (WSUM == 1) {
                    acc[rr] += w * v;
                } else {
                    C[(ulong)j * N + row] = v;
                }
            }
        }
    }
    if (WSUM == 1 && lane == 0u) {
        for (uint rr = 0; rr < 4u; ++rr) {
            const uint row = r0 + rr;
            if (row < N) C[row] = acc[rr];
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
