
// Decode (m=1) weight-stationary: каждый поток считает 4 выходных столбца
// (float4 аккумулятор, half4-чтения весов) — в ~4× меньше потоков с
// векторными операциями; A[k] — broadcast (кэш). Для m=1 заменяет
// gemm_f16 (grid (1, N/4) вместо (1, N)).
kernel void mv_f16_f4(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n4 = gid.y;
    uint n0 = n4 * 4u;
    if (m >= M || n0 >= N) return;
    float4 acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float a = A[m * K + k];
        device const half4* bp = (device const half4*)(B + k * N + n0);
        half4 b = *bp;
        acc.x += a * (float)b.x;
        acc.y += a * (float)b.y;
        acc.z += a * (float)b.z;
        acc.w += a * (float)b.w;
    }
    C[m * N + n0] = acc.x;
    if (n0 + 1u < N) C[m * N + n0 + 1u] = acc.y;
    if (n0 + 2u < N) C[m * N + n0 + 2u] = acc.z;
    if (n0 + 3u < N) C[m * N + n0 + 3u] = acc.w;
}

// Matvec IQ4_XS — точная редукция ggml `kernel_mul_mv_iq4_xs_f32_impl`
// (правило «паритет первичен»: тот же порядок произведений и то же дерево
// simd_sum, что у llama.cpp — иначе близкие логиты расходятся на жадной
// генерации; ср. mv_q8_0_ks для Q8_0). Строку весов считает ОДИН simdgroup
// из 32 лейнов: лейн (ix = чётность суперблока, it = 0..15 → подблок
// ib = it/2, половина il = it%2) копит float4-частичные суммы в порядке
// ggml, итог — simd_sum. Сетка (32, N, M): n — строка весов, m — строка
// активаций (decode M=1, префилл M>1 — тем же кернелом, тайлового GEMM
// для i-квантов пока нет).
kernel void mv_iq4_xs(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint n = gid.y, m = gid.z;
    if (n >= N) return;
    const uint nb = K / 256u;
    device const uchar* row = B + (ulong)n * nb * 136u;
    device const float* a = A + (ulong)m * K;

    const uint ix = lane / 16u;   // 0/1 — чётность суперблока (шаг 2 по sb)
    const uint it = lane % 16u;
    const uint ib = it / 2u;       // подблок 0..7
    const uint il = it % 2u;       // половина подблока: 8 байт qs

    float sumf = 0.0f;
    for (uint sb = ix; sb < nb; sb += 2u) {
        device const uchar* x = row + sb * 136u;
        // yl[0..3] = y4[0], y4[4], y4[1], y4[5] из ggml: байты 8il+0..3
        // держат элементы ib·32+8il+(0..3) (младшие нибблы) и +(16..19)
        // (старшие); байты 8il+4..7 — элементы +(4..7) и +(20..23).
        device const float* yb = a + (ulong)sb * 256u + ib * 32u + il * 8u;
        const float4 yl0 = {yb[0], yb[1], yb[2], yb[3]};
        const float4 yl1 = {yb[16], yb[17], yb[18], yb[19]};
        const float4 yl2 = {yb[4], yb[5], yb[6], yb[7]};
        const float4 yl3 = {yb[20], yb[21], yb[22], yb[23]};

        device const uchar* qb = x + 8u + 16u * ib + 8u * il;
        const uint q0 = (uint)qb[0] | ((uint)qb[1] << 8) | ((uint)qb[2] << 16) | ((uint)qb[3] << 24);
        const uint q1 = (uint)qb[4] | ((uint)qb[5] << 8) | ((uint)qb[6] << 16) | ((uint)qb[7] << 24);

        float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
        acc1 += yl0 * float4(KV_IQ4[q0 & 0xF], KV_IQ4[(q0 >> 8) & 0xF],
                             KV_IQ4[(q0 >> 16) & 0xF], KV_IQ4[(q0 >> 24) & 0xF]);
        acc2 += yl1 * float4(KV_IQ4[(q0 >> 4) & 0xF], KV_IQ4[(q0 >> 12) & 0xF],
                             KV_IQ4[(q0 >> 20) & 0xF], KV_IQ4[(q0 >> 28) & 0xF]);
        acc1 += yl2 * float4(KV_IQ4[q1 & 0xF], KV_IQ4[(q1 >> 8) & 0xF],
                             KV_IQ4[(q1 >> 16) & 0xF], KV_IQ4[(q1 >> 24) & 0xF]);
        acc2 += yl3 * float4(KV_IQ4[(q1 >> 4) & 0xF], KV_IQ4[(q1 >> 12) & 0xF],
                             KV_IQ4[(q1 >> 20) & 0xF], KV_IQ4[(q1 >> 28) & 0xF]);
        acc1 += acc2;

        const uint scales_h = (uint)x[2] | ((uint)x[3] << 8);
        const int ls = (int)((((x[4 + (ib >> 1)] >> (4u * (ib & 1u))) & 0xF)
                              | (((scales_h >> (2u * ib)) & 3u) << 4))) - 32;
        const float d = (float)load_half_unaligned(x);
        sumf += d * (float)ls * ((acc1.x + acc1.y) + acc1.z + acc1.w);
    }
    sumf = simd_sum(sumf);
    if (lane == 0u) C[(ulong)m * N + n] = sumf;
}

// Декод (m = 1): порт kernel_mul_mv_iq4_xs_f32 эталона (llama.cpp 0.4.0,
// N_R0_IQ4_XS = 2) — ОДИН simdgroup на ПАРУ строк весов: активации читаются
// один раз на пару (4×float4 вместо 16 скалярных на строку), кванты — парой
// u32 (вместо 8 ld8), и две независимые цепи аккумуляции удваивают ILP.
// Арифметика каждого выхода — точь-в-точь mv_iq4_xs (тот же порядок
// слагаемых) → побитово идентично (сверка r2 против mb5 — в тесте
// iq4_xs_matvec_batch_consistency, против хост-декванта — в
// iq4_xs_mv_matches_host_dequant).
// Требует 16-байтового сдвига A и 4-байтового — B (проверяется в диспетчере,
// иначе — прежний mv_iq4_xs). Прежняя попытка векторизовать ОДИН ряд
// (флаг/шаблон ALIGNED) была нейтральна: рычаг не в ширине загрузки сама по
// себе, а в переиспользовании y на два ряда и двух цепях ILP.
kernel void mv_iq4_xs_r2(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint n0 = 2u * tgid.y;
    if (n0 >= N) return;
    const bool two = n0 + 1u < N;
    const uint nb = K / 256u;
    device const uchar* row0 = B + (ulong)n0 * nb * 136u;

    const uint ix = lane / 16u;   // 0/1 — чётность суперблока (шаг 2 по sb)
    const uint it = lane % 16u;
    const uint ib = it / 2u;       // подблок 0..7
    const uint il = it % 2u;       // половина подблока: 8 байт qs

    float sumf[2] = {0.0f, 0.0f};
    for (uint sb = ix; sb < nb; sb += 2u) {
        device const float* yb = A + (ulong)sb * 256u + ib * 32u + il * 8u;
        const float4 yl0 = *(device const float4*)yb;
        const float4 yl1 = *(device const float4*)(yb + 16);
        const float4 yl2 = *(device const float4*)(yb + 4);
        const float4 yl3 = *(device const float4*)(yb + 20);
#pragma unroll
        for (uint rr = 0; rr < 2; ++rr) {
            if (rr == 1u && !two) break;
            device const uchar* x = row0 + (ulong)rr * nb * 136u + sb * 136u;
            device const uint* q4 = (device const uint*)(x + 8u + 16u * ib + 8u * il);
            const uint q0 = q4[0];
            const uint q1 = q4[1];
            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            acc1 += yl0 * float4(KV_IQ4[q0 & 0xF], KV_IQ4[(q0 >> 8) & 0xF],
                                 KV_IQ4[(q0 >> 16) & 0xF], KV_IQ4[(q0 >> 24) & 0xF]);
            acc2 += yl1 * float4(KV_IQ4[(q0 >> 4) & 0xF], KV_IQ4[(q0 >> 12) & 0xF],
                                 KV_IQ4[(q0 >> 20) & 0xF], KV_IQ4[(q0 >> 28) & 0xF]);
            acc1 += yl2 * float4(KV_IQ4[q1 & 0xF], KV_IQ4[(q1 >> 8) & 0xF],
                                 KV_IQ4[(q1 >> 16) & 0xF], KV_IQ4[(q1 >> 24) & 0xF]);
            acc2 += yl3 * float4(KV_IQ4[(q1 >> 4) & 0xF], KV_IQ4[(q1 >> 12) & 0xF],
                                 KV_IQ4[(q1 >> 20) & 0xF], KV_IQ4[(q1 >> 28) & 0xF]);
            acc1 += acc2;
            const uint scales_h = (uint)x[2] | ((uint)x[3] << 8);
            const int ls = (int)((((x[4 + (ib >> 1)] >> (4u * (ib & 1u))) & 0xF)
                                  | (((scales_h >> (2u * ib)) & 3u) << 4))) - 32;
            const float d = (float)load_half_unaligned(x);
            sumf[rr] += d * (float)ls * ((acc1.x + acc1.y) + acc1.z + acc1.w);
        }
    }
#pragma unroll
    for (uint rr = 0; rr < 2; ++rr) {
        const float v = simd_sum(sumf[rr]);
        if (lane == 0u && (rr == 0u || two)) C[n0 + rr] = v;
    }
}

// IQ4_XS matvec с батчем строк активаций (окно верификации спека, 1 < m ≤ 8):
// mv_iq4_xs читает строку весов на КАЖДУЮ строку A (сетка z = M) — при m = 5
// это 5-кратный трафик весов и forward окна дороже трёх шагов декода. Здесь
// один simdgroup на строку весов: квант-блок раскодируется ОДИН раз,
// накопители отдельные на строку, порядок операций для каждого выхода —
// точь-в-точь mv_iq4_xs → побитово тот же результат (паритет первичен).
// ВЕКТОРНЫЕ ЗАГРУЗКИ ПРОВЕРЕНЫ И ОТКАТАНЫ: u32-пары квант-байтов и float4
// активаций (с uniform-флагом выравнивания) ОКАЗАЛИСЬ МЕДЛЕННЕЕ поэлементных
// (m=1: 177 против 190 ГБ/с, m=5: 2.61 против 1.87 мс) — ветка внутри
// раскрученного цикла ломает планирование, а скалярные загрузки и так
// коалесцируются. Кернел bound не загрузками.
template<uint M>
kernel void mv_iq4_xs_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint n = tgid.y;
    if (n >= N) return;
    const uint nb = K / 256u;
    device const uchar* row = B + (ulong)n * nb * 136u;

    const uint ix = lane / 16u;   // 0/1 — чётность суперблока (шаг 2 по sb)
    const uint it = lane % 16u;
    const uint ib = it / 2u;       // подблок 0..7
    const uint il = it % 2u;       // половина подблока: 8 байт qs

    float sumf[M];
#pragma unroll
    for (uint m = 0; m < M; ++m) sumf[m] = 0.0f;
    for (uint sb = ix; sb < nb; sb += 2u) {
        device const uchar* x = row + sb * 136u;
        device const uchar* qb = x + 8u + 16u * ib + 8u * il;
        const uint q0 = (uint)qb[0] | ((uint)qb[1] << 8) | ((uint)qb[2] << 16) | ((uint)qb[3] << 24);
        const uint q1 = (uint)qb[4] | ((uint)qb[5] << 8) | ((uint)qb[6] << 16) | ((uint)qb[7] << 24);
        // деквант — общий для всех строк A (в mv_iq4_xs он перемножен с yl)
        const float4 k0 = float4(KV_IQ4[q0 & 0xF], KV_IQ4[(q0 >> 8) & 0xF],
                                 KV_IQ4[(q0 >> 16) & 0xF], KV_IQ4[(q0 >> 24) & 0xF]);
        const float4 k1 = float4(KV_IQ4[(q0 >> 4) & 0xF], KV_IQ4[(q0 >> 12) & 0xF],
                                 KV_IQ4[(q0 >> 20) & 0xF], KV_IQ4[(q0 >> 28) & 0xF]);
        const float4 k2 = float4(KV_IQ4[q1 & 0xF], KV_IQ4[(q1 >> 8) & 0xF],
                                 KV_IQ4[(q1 >> 16) & 0xF], KV_IQ4[(q1 >> 24) & 0xF]);
        const float4 k3 = float4(KV_IQ4[(q1 >> 4) & 0xF], KV_IQ4[(q1 >> 12) & 0xF],
                                 KV_IQ4[(q1 >> 20) & 0xF], KV_IQ4[(q1 >> 28) & 0xF]);
        const uint scales_h = (uint)x[2] | ((uint)x[3] << 8);
        const int ls = (int)((((x[4 + (ib >> 1)] >> (4u * (ib & 1u))) & 0xF)
                              | (((scales_h >> (2u * ib)) & 3u) << 4))) - 32;
        const float d = (float)load_half_unaligned(x);
        const float dl = d * (float)ls;
#pragma unroll
        for (uint m = 0; m < M; ++m) {
            device const float* yb = A + (ulong)m * K + (ulong)sb * 256u + ib * 32u + il * 8u;
            const float4 yl0 = {yb[0], yb[1], yb[2], yb[3]};
            const float4 yl1 = {yb[16], yb[17], yb[18], yb[19]};
            const float4 yl2 = {yb[4], yb[5], yb[6], yb[7]};
            const float4 yl3 = {yb[20], yb[21], yb[22], yb[23]};
            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            acc1 += yl0 * k0;
            acc2 += yl1 * k1;
            acc1 += yl2 * k2;
            acc2 += yl3 * k3;
            acc1 += acc2;
            sumf[m] += dl * ((acc1.x + acc1.y) + acc1.z + acc1.w);
        }
    }
#pragma unroll
    for (uint m = 0; m < M; ++m) {
        const float v = simd_sum(sumf[m]);
        if (lane == 0u) C[(ulong)m * N + n] = v;
    }
}

// ── Q1_0 / Q2_0 / TQ1_0 / TQ2_0: декод и окна ───────────────────────────
// Точечная функция value_at (common.metal) по позиции внутри блока;
// блоки 128/64/256, байты 18/18/54/66.

template <int FMT>
kernel void mv_tern(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint per = tern_block_elems<FMT>();
    const uint bb = tern_block_bytes<FMT>();
    const uint blocks = K / per;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + ((ulong)n * blocks + b) * bb;
        const uint k0 = b * per;
        for (uint p = 0; p < per; ++p) {
            acc += A[k0 + p] * tern_value_at<FMT>(blk, p);
        }
    }
    C[n] = acc;
}

typedef decltype(mv_tern<0>) mv_tern_t;
template [[host_name("mv_q1_0")]] kernel mv_tern_t mv_tern<0>;
template [[host_name("mv_q2_0")]] kernel mv_tern_t mv_tern<1>;
template [[host_name("mv_tq1_0")]] kernel mv_tern_t mv_tern<2>;
template [[host_name("mv_tq2_0")]] kernel mv_tern_t mv_tern<3>;
template [[host_name("mv_iq2_xxs")]] kernel mv_tern_t mv_tern<4>;
template [[host_name("mv_iq3_xxs")]] kernel mv_tern_t mv_tern<5>;
template [[host_name("mv_iq2_xs")]] kernel mv_tern_t mv_tern<6>;
template [[host_name("mv_iq2_s")]] kernel mv_tern_t mv_tern<7>;
template [[host_name("mv_iq3_s")]] kernel mv_tern_t mv_tern<8>;
template [[host_name("mv_iq1_s")]] kernel mv_tern_t mv_tern<9>;
template [[host_name("mv_iq1_m")]] kernel mv_tern_t mv_tern<10>;

template <int FMT, short M, short NR>
kernel void mv_tern_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint per = tern_block_elems<FMT>();
    const uint bb = tern_block_bytes<FMT>();
    const uint nb = K / per;
    const uint S = per / 32u; // значений на лейн: 128→4, 64→2, 256→8
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * bb;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        const uint p0 = lane * S;
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * bb;
            const ulong k0 = (ulong)ib * per + p0;
#pragma unroll
            for (short m = 0; m < M; ++m) {
                device const float* am = A + (ulong)m * K + k0;
                for (uint i = 0; i < S; ++i) {
                    acc[r][m] += am[i] * tern_value_at<FMT>(blk, p0 + i);
                }
            }
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

typedef decltype(mv_tern_mb<0, 2, 4>) mv_tern_mb4_t;
typedef decltype(mv_tern_mb<0, 2, 8>) mv_tern_mb8_t;
template [[host_name("mv_q1_0_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<0, 2, 4>;
template [[host_name("mv_q1_0_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<0, 3, 4>;
template [[host_name("mv_q1_0_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<0, 4, 4>;
template [[host_name("mv_q1_0_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<0, 5, 8>;
template [[host_name("mv_q1_0_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<0, 6, 8>;
template [[host_name("mv_q2_0_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<1, 2, 4>;
template [[host_name("mv_q2_0_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<1, 3, 4>;
template [[host_name("mv_q2_0_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<1, 4, 4>;
template [[host_name("mv_q2_0_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<1, 5, 8>;
template [[host_name("mv_q2_0_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<1, 6, 8>;
template [[host_name("mv_tq1_0_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<2, 2, 4>;
template [[host_name("mv_tq1_0_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<2, 3, 4>;
template [[host_name("mv_tq1_0_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<2, 4, 4>;
template [[host_name("mv_tq1_0_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<2, 5, 8>;
template [[host_name("mv_tq1_0_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<2, 6, 8>;
template [[host_name("mv_tq2_0_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<3, 2, 4>;
template [[host_name("mv_tq2_0_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<3, 3, 4>;
template [[host_name("mv_tq2_0_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<3, 4, 4>;
template [[host_name("mv_tq2_0_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<3, 5, 8>;
template [[host_name("mv_tq2_0_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<3, 6, 8>;
template [[host_name("mv_iq2_xxs_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<4, 2, 4>;
template [[host_name("mv_iq2_xxs_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<4, 3, 4>;
template [[host_name("mv_iq2_xxs_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<4, 4, 4>;
template [[host_name("mv_iq2_xxs_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<4, 5, 8>;
template [[host_name("mv_iq2_xxs_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<4, 6, 8>;
template [[host_name("mv_iq3_xxs_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<5, 2, 4>;
template [[host_name("mv_iq3_xxs_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<5, 3, 4>;
template [[host_name("mv_iq3_xxs_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<5, 4, 4>;
template [[host_name("mv_iq3_xxs_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<5, 5, 8>;
template [[host_name("mv_iq3_xxs_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<5, 6, 8>;
template [[host_name("mv_iq2_xs_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<6, 2, 4>;
template [[host_name("mv_iq2_xs_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<6, 3, 4>;
template [[host_name("mv_iq2_xs_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<6, 4, 4>;
template [[host_name("mv_iq2_xs_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<6, 5, 8>;
template [[host_name("mv_iq2_xs_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<6, 6, 8>;
template [[host_name("mv_iq2_s_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<7, 2, 4>;
template [[host_name("mv_iq2_s_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<7, 3, 4>;
template [[host_name("mv_iq2_s_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<7, 4, 4>;
template [[host_name("mv_iq2_s_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<7, 5, 8>;
template [[host_name("mv_iq2_s_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<7, 6, 8>;
template [[host_name("mv_iq3_s_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<8, 2, 4>;
template [[host_name("mv_iq3_s_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<8, 3, 4>;
template [[host_name("mv_iq3_s_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<8, 4, 4>;
template [[host_name("mv_iq3_s_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<8, 5, 8>;
template [[host_name("mv_iq3_s_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<8, 6, 8>;
template [[host_name("mv_iq1_s_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<9, 2, 4>;
template [[host_name("mv_iq1_s_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<9, 3, 4>;
template [[host_name("mv_iq1_s_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<9, 4, 4>;
template [[host_name("mv_iq1_s_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<9, 5, 8>;
template [[host_name("mv_iq1_s_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<9, 6, 8>;
template [[host_name("mv_iq1_m_mb2")]] kernel mv_tern_mb4_t mv_tern_mb<10, 2, 4>;
template [[host_name("mv_iq1_m_mb3")]] kernel mv_tern_mb4_t mv_tern_mb<10, 3, 4>;
template [[host_name("mv_iq1_m_mb4")]] kernel mv_tern_mb4_t mv_tern_mb<10, 4, 4>;
template [[host_name("mv_iq1_m_mb5")]] kernel mv_tern_mb8_t mv_tern_mb<10, 5, 8>;
template [[host_name("mv_iq1_m_mb6")]] kernel mv_tern_mb8_t mv_tern_mb<10, 6, 8>;

// ── Q8_1 (40 Б/32) и Q8_K (292 Б/256): декод и окна ─────────────────────
// Q8_1: d f16 @0, s @2, int8 qs[32] @4; Q8_K: f32 d @0, int8 qs[256] @4.
// Деквант весов — d·q (s/bsums — активационные величины, не нужны).
template <bool IS_K>
kernel void mv_q8_1_q8_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint per = IS_K ? 256u : 32u;
    const uint bb = IS_K ? 292u : 36u;
    const uint blocks = K / per;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + ((ulong)n * blocks + b) * bb;
        const float d = IS_K ? as_type<float>(*(device const uint*)(blk))
                             : (float)load_half_unaligned(blk);
        const uint k0 = b * per;
        for (uint j = 0; j < per; ++j) {
            acc += A[k0 + j] * (float)(signed char)blk[4u + j] * d;
        }
    }
    C[n] = acc;
}

typedef decltype(mv_q8_1_q8_k<true>) mv_q8_1_q8_k_t;
template [[host_name("mv_q8_1")]] kernel mv_q8_1_q8_k_t mv_q8_1_q8_k<false>;
template [[host_name("mv_q8_k")]] kernel mv_q8_1_q8_k_t mv_q8_1_q8_k<true>;

template <bool IS_K, short M, short NR>
kernel void mv_q8_1_q8_k_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint per = IS_K ? 256u : 32u;
    const uint bb = IS_K ? 292u : 36u;
    const uint nb = K / per;
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    // Q8_1: 8 лейнов на блок (по 4 значения); Q8_K: 32 лейна на строку
    // (по 8 значений из 256).
    const uint bl = IS_K ? 0u : (lane / 8u);
    const uint li = IS_K ? lane : (lane % 8u);
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * bb;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = bl; ib < nb; ib += (IS_K ? 1u : 4u)) {
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * bb;
            const float d = IS_K ? as_type<float>(*(device const uint*)(blk))
                                 : (float)load_half_unaligned(blk);
            const uint off = IS_K ? (4u + li * 8u) : (4u + li * 4u);
            const uint cnt = IS_K ? 8u : 4u;
            const ulong k0 = (ulong)ib * per + (IS_K ? li * 8u : li * 4u);
#pragma unroll
            for (short m = 0; m < M; ++m) {
                device const float* am = A + (ulong)m * K + k0;
                for (uint i = 0; i < cnt; ++i) {
                    acc[r][m] += am[i] * (float)(signed char)blk[off + i] * d;
                }
            }
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

typedef decltype(mv_q8_1_q8_k_mb<false, 2, 4>) mv_q8_1_mb4_t;
typedef decltype(mv_q8_1_q8_k_mb<false, 2, 8>) mv_q8_1_mb8_t;
typedef decltype(mv_q8_1_q8_k_mb<true, 2, 4>) mv_q8_k_mb4_t;
typedef decltype(mv_q8_1_q8_k_mb<true, 2, 8>) mv_q8_k_mb8_t;
template [[host_name("mv_q8_1_mb2")]] kernel mv_q8_1_mb4_t mv_q8_1_q8_k_mb<false, 2, 4>;
template [[host_name("mv_q8_1_mb3")]] kernel mv_q8_1_mb4_t mv_q8_1_q8_k_mb<false, 3, 4>;
template [[host_name("mv_q8_1_mb4")]] kernel mv_q8_1_mb4_t mv_q8_1_q8_k_mb<false, 4, 4>;
template [[host_name("mv_q8_1_mb5")]] kernel mv_q8_1_mb8_t mv_q8_1_q8_k_mb<false, 5, 8>;
template [[host_name("mv_q8_1_mb6")]] kernel mv_q8_1_mb8_t mv_q8_1_q8_k_mb<false, 6, 8>;
template [[host_name("mv_q8_k_mb2")]] kernel mv_q8_k_mb4_t mv_q8_1_q8_k_mb<true, 2, 4>;
template [[host_name("mv_q8_k_mb3")]] kernel mv_q8_k_mb4_t mv_q8_1_q8_k_mb<true, 3, 4>;
template [[host_name("mv_q8_k_mb4")]] kernel mv_q8_k_mb4_t mv_q8_1_q8_k_mb<true, 4, 4>;
template [[host_name("mv_q8_k_mb5")]] kernel mv_q8_k_mb8_t mv_q8_1_q8_k_mb<true, 5, 8>;
template [[host_name("mv_q8_k_mb6")]] kernel mv_q8_k_mb8_t mv_q8_1_q8_k_mb<true, 6, 8>;

// ── Q4_1 / Q5_1: декод (m=1) и окна (1<m≤8) ─────────────────────────────
// Q4_1: блок 20 Б (d, m f16 + 16 Б нибблов), значение d·q + m;
// Q5_1: блок 24 Б (d, m, qh[4] @4, qs[16] @8), q = ниббл | ((qh>>j)&1)<<4.

template <bool IS_Q5>
kernel void mv_q4_1_q5_1(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 32u;
    const uint bb = IS_Q5 ? 24u : 20u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + ((ulong)n * blocks + b) * bb;
        const float d = (float)load_half_unaligned(blk);
        const float m = (float)load_half_unaligned(blk + 2u);
        const uint qh = IS_Q5 ? ((uint)blk[4] | ((uint)blk[5] << 8) | ((uint)blk[6] << 16) | ((uint)blk[7] << 24)) : 0u;
        device const uchar* qs = blk + (IS_Q5 ? 8u : 4u);
        const uint k0 = b * 32u;
        for (uint j = 0; j < 16u; ++j) {
            uint lo = (uint)(qs[j] & 0x0Fu);
            uint hi = (uint)(qs[j] >> 4u);
            if (IS_Q5) {
                lo |= ((qh >> j) & 1u) << 4;
                hi |= ((qh >> (j + 16u)) & 1u) << 4;
            }
            acc += A[k0 + j] * ((float)lo * d + m);
            acc += A[k0 + j + 16u] * ((float)hi * d + m);
        }
    }
    C[n] = acc;
}

typedef decltype(mv_q4_1_q5_1<true>) mv_q4_1_q5_1_t;
template [[host_name("mv_q4_1")]] kernel mv_q4_1_q5_1_t mv_q4_1_q5_1<false>;
template [[host_name("mv_q5_1")]] kernel mv_q4_1_q5_1_t mv_q4_1_q5_1<true>;

template <bool IS_Q5, short M, short NR>
kernel void mv_q4_1_q5_1_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint nb = K / 32u;
    const uint bb = IS_Q5 ? 24u : 20u;
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    const uint bl = lane / 8u;
    const uint li = lane % 8u;
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * bb;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = bl; ib < nb; ib += 4u) {
        float2 y0[M], y1[M];
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const ulong base = (ulong)m * K + ib * 32u + 2u * li;
            y0[m] = *(device const float2*)(A + base);
            y1[m] = *(device const float2*)(A + base + 16u);
        }
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * bb;
            const float d = (float)load_half_unaligned(blk);
            const float mn = (float)load_half_unaligned(blk + 2u);
            const uint qh = IS_Q5 ? ((uint)blk[4] | ((uint)blk[5] << 8) | ((uint)blk[6] << 16) | ((uint)blk[7] << 24)) : 0u;
            device const uchar* q = blk + (IS_Q5 ? 8u : 4u) + 2u * li;
            const uchar b0 = q[0];
            const uchar b1 = q[1];
            uint q0 = (uint)(b0 & 0x0Fu), q1 = (uint)(b1 & 0x0Fu);
            uint q2 = (uint)(b0 >> 4u), q3 = (uint)(b1 >> 4u);
            if (IS_Q5) {
                const uint j0 = 2u * li;
                q0 |= ((qh >> j0) & 1u) << 4;
                q1 |= ((qh >> (j0 + 1u)) & 1u) << 4;
                q2 |= ((qh >> (j0 + 16u)) & 1u) << 4;
                q3 |= ((qh >> (j0 + 17u)) & 1u) << 4;
            }
            const float w0 = (float)q0 * d + mn;
            const float w1 = (float)q1 * d + mn;
            const float w2 = (float)q2 * d + mn;
            const float w3 = (float)q3 * d + mn;
#pragma unroll
            for (short m = 0; m < M; ++m) {
                acc[r][m] += w0 * y0[m].x + w1 * y0[m].y + w2 * y1[m].x + w3 * y1[m].y;
            }
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

typedef decltype(mv_q4_1_q5_1_mb<false, 2, 4>) mv_q4_1_q5_1_mb4_t;
typedef decltype(mv_q4_1_q5_1_mb<false, 2, 8>) mv_q4_1_q5_1_mb8_t;
typedef decltype(mv_q4_1_q5_1_mb<true, 2, 4>) mv_q5_1_q5_1_mb4_t;
typedef decltype(mv_q4_1_q5_1_mb<true, 2, 8>) mv_q5_1_q5_1_mb8_t;
template [[host_name("mv_q4_1_mb2")]] kernel mv_q4_1_q5_1_mb4_t mv_q4_1_q5_1_mb<false, 2, 4>;
template [[host_name("mv_q4_1_mb3")]] kernel mv_q4_1_q5_1_mb4_t mv_q4_1_q5_1_mb<false, 3, 4>;
template [[host_name("mv_q4_1_mb4")]] kernel mv_q4_1_q5_1_mb4_t mv_q4_1_q5_1_mb<false, 4, 4>;
template [[host_name("mv_q4_1_mb5")]] kernel mv_q4_1_q5_1_mb8_t mv_q4_1_q5_1_mb<false, 5, 8>;
template [[host_name("mv_q4_1_mb6")]] kernel mv_q4_1_q5_1_mb8_t mv_q4_1_q5_1_mb<false, 6, 8>;
template [[host_name("mv_q5_1_mb2")]] kernel mv_q5_1_q5_1_mb4_t mv_q4_1_q5_1_mb<true, 2, 4>;
template [[host_name("mv_q5_1_mb3")]] kernel mv_q5_1_q5_1_mb4_t mv_q4_1_q5_1_mb<true, 3, 4>;
template [[host_name("mv_q5_1_mb4")]] kernel mv_q5_1_q5_1_mb4_t mv_q4_1_q5_1_mb<true, 4, 4>;
template [[host_name("mv_q5_1_mb5")]] kernel mv_q5_1_q5_1_mb8_t mv_q4_1_q5_1_mb<true, 5, 8>;
template [[host_name("mv_q5_1_mb6")]] kernel mv_q5_1_q5_1_mb8_t mv_q4_1_q5_1_mb<true, 6, 8>;

// ── Q2_K / Q3_K: декод (m=1) и окна (1<m≤8) ─────────────────────────────
// Блок 256: Q2_K 84 Б (scales[16], qs[64], d, dmin), Q3_K 110 Б
// (hmask[32], qs[64], scales[12], d). 16 подблоков по 16 значений:
// shift = 2·((s%8)/2), half = s%2 (байты l+16·half), q += 32 на 8 подблоков.
// mb: 32 лейна на строку, 2 лейна на подблок (по 8 байт = 8 значений).

template <bool IS_Q3>
kernel void mv_q2_k_q3_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 256u;
    const uint bb = IS_Q3 ? 110u : 84u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + ((ulong)n * blocks + b) * bb;
        const float d = (float)load_half_unaligned(blk + (IS_Q3 ? 108u : 80u));
        const float dmin = IS_Q3 ? 0.0f : (float)load_half_unaligned(blk + 82u);
        int sc[16];
        if (IS_Q3) q3k_unpack_scales(blk + 96u, sc);
        const uint qs0 = IS_Q3 ? 32u : 16u;
        for (uint s = 0; s < 16u; ++s) {
            float dl, ml;
            if (IS_Q3) {
                dl = d * (float)(sc[s] - 32);
                ml = 0.0f;
            } else {
                const uchar v = blk[s];
                dl = d * (float)(v & 0x0Fu);
                ml = dmin * (float)(v >> 4u);
            }
            const uint shift = 2u * ((s % 8u) / 2u);
            const uint hpart = s % 2u;
            device const uchar* qb = blk + qs0 + (s / 8u) * 32u + hpart * 16u;
            const uchar bit = (uchar)(1u << (s / 2u));
            device const uchar* hb = blk + hpart * 16u;
            const uint k0 = b * 256u + s * 16u;
            for (uint l = 0; l < 16u; ++l) {
                int v = (int)((qb[l] >> shift) & 3u);
                if (IS_Q3 && (hb[l] & bit) == 0u) v -= 4;
                acc += A[k0 + l] * (dl * (float)v - ml);
            }
        }
    }
    C[n] = acc;
}

template <bool IS_Q3, short M>
kernel void mv_q2_k_q3_k_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint n = tgid.y;
    if (n >= N) return;
    const uint blocks = K / 256u;
    const uint bb = IS_Q3 ? 110u : 84u;
    device const uchar* row = B + (ulong)n * blocks * bb;
    // 2 лейна на подблок: s = lane/2, l0 = (lane%2)*8
    const uint s = lane / 2u;
    const uint l0 = (lane % 2u) * 8u;

    float sumf[M];
#pragma unroll
    for (uint m = 0; m < M; ++m) sumf[m] = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = row + b * bb;
        const float d = (float)load_half_unaligned(blk + (IS_Q3 ? 108u : 80u));
        const float dmin = IS_Q3 ? 0.0f : (float)load_half_unaligned(blk + 82u);
        int sc[16];
        if (IS_Q3) q3k_unpack_scales(blk + 96u, sc);
        const uint qs0 = IS_Q3 ? 32u : 16u;
        float dl, ml;
        if (IS_Q3) {
            dl = d * (float)(sc[s] - 32);
            ml = 0.0f;
        } else {
            const uchar v = blk[s];
            dl = d * (float)(v & 0x0Fu);
            ml = dmin * (float)(v >> 4u);
        }
        const uint shift = 2u * ((s % 8u) / 2u);
        const uint hpart = s % 2u;
        device const uchar* qb = blk + qs0 + (s / 8u) * 32u + hpart * 16u + l0;
        const uchar bit = (uchar)(1u << (s / 2u));
        device const uchar* hb = blk + hpart * 16u + l0;
        const ulong k0 = (ulong)b * 256u + s * 16u + l0;
#pragma unroll
        for (uint m = 0; m < M; ++m) {
            device const float* am = A + (ulong)m * K + k0;
            for (uint i = 0; i < 8u; ++i) {
                int v = (int)((qb[i] >> shift) & 3u);
                if (IS_Q3 && (hb[i] & bit) == 0u) v -= 4;
                sumf[m] += am[i] * (dl * (float)v - ml);
            }
        }
    }
#pragma unroll
    for (uint m = 0; m < M; ++m) {
        const float v = simd_sum(sumf[m]);
        if (lane == 0u) C[(ulong)m * N + n] = v;
    }
}

typedef decltype(mv_q2_k_q3_k<true>) mv_q2_k_q3_k_t;
template [[host_name("mv_q2_k")]] kernel mv_q2_k_q3_k_t mv_q2_k_q3_k<false>;
template [[host_name("mv_q3_k")]] kernel mv_q2_k_q3_k_t mv_q2_k_q3_k<true>;
typedef decltype(mv_q2_k_q3_k_mb<true, 2>) mv_q2_k_q3_k_mb_t;
template [[host_name("mv_q2_k_mb2")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<false, 2>;
template [[host_name("mv_q2_k_mb3")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<false, 3>;
template [[host_name("mv_q2_k_mb4")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<false, 4>;
template [[host_name("mv_q2_k_mb5")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<false, 5>;
template [[host_name("mv_q2_k_mb6")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<false, 6>;
template [[host_name("mv_q3_k_mb2")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<true, 2>;
template [[host_name("mv_q3_k_mb3")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<true, 3>;
template [[host_name("mv_q3_k_mb4")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<true, 4>;
template [[host_name("mv_q3_k_mb5")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<true, 5>;
template [[host_name("mv_q3_k_mb6")]] kernel mv_q2_k_q3_k_mb_t mv_q2_k_q3_k_mb<true, 6>;

// ── NVFP4: декод (m=1) и окна (1<m≤8) ───────────────────────────────────
// Блок 36 Б/64: d[4] (UE4M3, по подблоку 16 значений) + qs[32]; байт
// qs[s*8+j] — значения s*16+j (младший ниббл) и s*16+j+8 (старший),
// множитель d_s = ue4m3(d[s]); значения — «удвоенные» E2M1 (KV_FP4).
// Раскладка нибблов внутри подблока — j и j+8 (не j и j+16, как у 32-эл.
// форматов), поэтому блоки 64 нельзя обрабатывать 32-шагами как Q4_0.

kernel void mv_nvfp4(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 64u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + ((ulong)n * blocks + b) * 36u;
        const uint k0 = b * 64u;
        for (uint s = 0; s < 4u; ++s) {
            const float d = ue4m3_to_fp32(blk[s]);
            for (uint j = 0; j < 8u; ++j) {
                const uchar v = blk[4u + s * 8u + j];
                acc += A[k0 + s * 16u + j] * KV_FP4[v & 0x0Fu] * d;
                acc += A[k0 + s * 16u + j + 8u] * KV_FP4[v >> 4u] * d;
            }
        }
    }
    C[n] = acc;
}

// Окна: 32 лейна делят блок 64 на 4 подблока по 8 лейнов; лейн — 1 байт =
// 2 значения (j и j+8). NR строк весов на simdgroup, A — скалярно (сдвиги
// j/j+8 не кратны float2), веса 1× на m строк.
template <short M, short NR>
kernel void mv_nvfp4_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint nb = K / 64u;
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    const uint sub = lane / 8u;   // 0..3 — подблок 16 значений
    const uint j = lane % 8u;     // байт внутри подблока → значения j, j+8
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * 36u;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        const ulong base = (ulong)ib * 64u + sub * 16u + j;
        float y_lo[M], y_hi[M];
#pragma unroll
        for (short m = 0; m < M; ++m) {
            y_lo[m] = A[(ulong)m * K + base];
            y_hi[m] = A[(ulong)m * K + base + 8u];
        }
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * 36u;
            const float d = ue4m3_to_fp32(blk[sub]);
            const uchar v = blk[4u + sub * 8u + j];
            const float w_lo = KV_FP4[v & 0x0Fu] * d;
            const float w_hi = KV_FP4[v >> 4u] * d;
#pragma unroll
            for (short m = 0; m < M; ++m) {
                acc[r][m] += w_lo * y_lo[m] + w_hi * y_hi[m];
            }
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

typedef decltype(mv_nvfp4_mb<2, 4>) mv_nvfp4_mb_t;
template [[host_name("mv_nvfp4_mb2")]] kernel mv_nvfp4_mb_t mv_nvfp4_mb<2, 4>;
template [[host_name("mv_nvfp4_mb3")]] kernel mv_nvfp4_mb_t mv_nvfp4_mb<3, 4>;
template [[host_name("mv_nvfp4_mb4")]] kernel mv_nvfp4_mb_t mv_nvfp4_mb<4, 4>;
template [[host_name("mv_nvfp4_mb5")]] kernel mv_nvfp4_mb_t mv_nvfp4_mb<5, 8>;
template [[host_name("mv_nvfp4_mb6")]] kernel mv_nvfp4_mb_t mv_nvfp4_mb<6, 8>;

// ── IQ4_NL / MXFP4: декод (m=1) и окна (1<m≤8) ──────────────────────────
// Раскладка нибблов — каноническая ggml (байт j: значение j в младшем
// ниббле, j+16 в старшем), как у Q4_0; отличаются только блок и кодбук:
// IQ4_NL — 18 Б (f16 d + 16 Б), значение d·KV_IQ4[nibble];
// MXFP4 — 17 Б (E8M0 e + 16 Б), значение KV_FP4[nibble]·2^(e−128).
// Порядок накопления в mv — последовательный по блокам (как mv_q4_0);
// побитовый порт эталонного mul_mv для этих типов — отдельный заход.

template <bool IS_NL>
kernel void mv_iq4_nl_mxfp4(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 32u;
    const uint blk_bytes = IS_NL ? 18u : 17u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + ((ulong)n * blocks + b) * blk_bytes;
        const float d = IS_NL ? (float)load_half_unaligned(blk)
                              : e8m0_to_fp32_half(blk[0]);
        device const uchar* q = blk + (IS_NL ? 2u : 1u);
        const uint k0 = b * 32u;
        for (uint j = 0; j < 16u; ++j) {
            const uchar v = q[j];
            if (IS_NL) {
                acc += A[k0 + j] * d * KV_IQ4[v & 0x0Fu];
                acc += A[k0 + j + 16u] * d * KV_IQ4[v >> 4u];
            } else {
                acc += A[k0 + j] * KV_FP4[v & 0x0Fu] * d;
                acc += A[k0 + j + 16u] * KV_FP4[v >> 4u] * d;
            }
        }
    }
    C[n] = acc;
}

typedef decltype(mv_iq4_nl_mxfp4<true>) mv_iq4_nl_mxfp4_t;
template [[host_name("mv_iq4_nl")]] kernel mv_iq4_nl_mxfp4_t mv_iq4_nl_mxfp4<true>;
template [[host_name("mv_mxfp4")]] kernel mv_iq4_nl_mxfp4_t mv_iq4_nl_mxfp4<false>;

// Окна: батчевый matvec — раскладка как у mv_q4_0_mb (8 лейнов на блок,
// 2 байта = 4 значения, NR строк весов на simdgroup, A двумя float2).
template <short M, short NR, bool IS_NL>
kernel void mv_iq4_nl_mxfp4_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint nb = K / 32u;
    const uint blk_bytes = IS_NL ? 18u : 17u;
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    const uint bl = lane / 8u;
    const uint li = lane % 8u;
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * blk_bytes;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = bl; ib < nb; ib += 4u) {
        float2 y0[M], y1[M];
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const ulong base = (ulong)m * K + ib * 32u + 2u * li;
            y0[m] = *(device const float2*)(A + base);
            y1[m] = *(device const float2*)(A + base + 16u);
        }
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * blk_bytes;
            const float d = IS_NL ? (float)load_half_unaligned(blk)
                                  : e8m0_to_fp32_half(blk[0]);
            device const uchar* q = blk + (IS_NL ? 2u : 1u) + 2u * li;
            const uchar b0 = q[0];
            const uchar b1 = q[1];
            float w0, w1, w2, w3;
            if (IS_NL) {
                w0 = d * KV_IQ4[b0 & 0x0Fu];
                w1 = d * KV_IQ4[b1 & 0x0Fu];
                w2 = d * KV_IQ4[b0 >> 4u];
                w3 = d * KV_IQ4[b1 >> 4u];
            } else {
                w0 = KV_FP4[b0 & 0x0Fu] * d;
                w1 = KV_FP4[b1 & 0x0Fu] * d;
                w2 = KV_FP4[b0 >> 4u] * d;
                w3 = KV_FP4[b1 >> 4u] * d;
            }
#pragma unroll
            for (short m = 0; m < M; ++m) {
                acc[r][m] += w0 * y0[m].x + w1 * y0[m].y + w2 * y1[m].x + w3 * y1[m].y;
            }
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

typedef decltype(mv_iq4_nl_mxfp4_mb<2, 4, true>) mv_iq4_nl_mxfp4_mb_nl_t;
typedef decltype(mv_iq4_nl_mxfp4_mb<2, 4, false>) mv_iq4_nl_mxfp4_mb_fp_t;
template [[host_name("mv_iq4_nl_mb2")]] kernel mv_iq4_nl_mxfp4_mb_nl_t mv_iq4_nl_mxfp4_mb<2, 4, true>;
template [[host_name("mv_iq4_nl_mb3")]] kernel mv_iq4_nl_mxfp4_mb_nl_t mv_iq4_nl_mxfp4_mb<3, 4, true>;
template [[host_name("mv_iq4_nl_mb4")]] kernel mv_iq4_nl_mxfp4_mb_nl_t mv_iq4_nl_mxfp4_mb<4, 4, true>;
template [[host_name("mv_iq4_nl_mb5")]] kernel mv_iq4_nl_mxfp4_mb_nl_t mv_iq4_nl_mxfp4_mb<5, 8, true>;
template [[host_name("mv_iq4_nl_mb6")]] kernel mv_iq4_nl_mxfp4_mb_nl_t mv_iq4_nl_mxfp4_mb<6, 8, true>;
template [[host_name("mv_mxfp4_mb2")]] kernel mv_iq4_nl_mxfp4_mb_fp_t mv_iq4_nl_mxfp4_mb<2, 4, false>;
template [[host_name("mv_mxfp4_mb3")]] kernel mv_iq4_nl_mxfp4_mb_fp_t mv_iq4_nl_mxfp4_mb<3, 4, false>;
template [[host_name("mv_mxfp4_mb4")]] kernel mv_iq4_nl_mxfp4_mb_fp_t mv_iq4_nl_mxfp4_mb<4, 4, false>;
template [[host_name("mv_mxfp4_mb5")]] kernel mv_iq4_nl_mxfp4_mb_fp_t mv_iq4_nl_mxfp4_mb<5, 8, false>;
template [[host_name("mv_mxfp4_mb6")]] kernel mv_iq4_nl_mxfp4_mb_fp_t mv_iq4_nl_mxfp4_mb<6, 8, false>;

// Q5_K matvec с батчем строк активаций v2 (окно спек-верификации, 1<m≤8) —
// v1 (simdgroup на строку весов, весь 256-суперблок на лейн) дал ×3 к
// mm-тайлу: 1280 скалярных загрузок A на суперблок. Здесь 32 потока на
// строку весов, 16 лейнов ДЕЛЯТ суперблок (c = lane/4 & 3, l0 = 8·(lane%4);
// c = lane/8 давала лишь 2 подблока из 4 — баг ловился тестом
// q5_k_mb_matches_host_dequant), деквант-множители суперблока считаются
// один раз, A читается float4 двумя парами на подблок (8 lo подряд и 8 hi
// подряд, lo/hi одного l разнесены на 32 позиции) — трафик весов 1× на m
// строк (при m≤6; m 7/8 гонит два прохода), как у mv_iq4_xs_mb.
template<uint M>
kernel void mv_q5_k_mb(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint n = tgid.y;
    if (n >= N) return;
    const uint nb = K / 256u;
    device const uchar* row = B + (ulong)n * nb * 176u;

    // 16 лейнов на суперблок: c 0..3 (подблок 64), l0 = 8·(lane%4).
    // c = lane/4: 16 лейнов → c 0..3 × 4 позиции l0 (c = lane/8 давала
    // только 0..1 — два подблока из четырёх, поймано тестом).
    const uint c = (lane / 4u) & 3u;
    const uint l0 = (lane & 3u) * 8u;
    const uint half_sel = lane / 16u; // 0/1 — чётность суперблока (шаг 2)

    float sumf[M];
#pragma unroll
    for (uint m = 0; m < M; ++m) sumf[m] = 0.0f;
    for (uint sb = half_sel; sb < nb; sb += 2u) {
        device const uchar* blk = row + sb * 176u;
        device const uchar* qs = blk + 48u + c * 32u + l0;
        device const uchar* qh = blk + 16u + l0;
        int sc_lo, mn_lo, sc_hi, mn_hi;
        q4k_scale_min(blk + 4, 2u * c + 0u, sc_lo, mn_lo);
        q4k_scale_min(blk + 4, 2u * c + 1u, sc_hi, mn_hi);
        const float d = (float)load_half_unaligned(blk);
        const float dmin = (float)load_half_unaligned(blk + 2u);
        const float dl_lo = d * (float)sc_lo;
        const float ml_lo = dmin * (float)mn_lo;
        const float dl_hi = d * (float)sc_hi;
        const float ml_hi = dmin * (float)mn_hi;
        // 8 значений lo (позиции c*64 + l0..l0+7) и 8 hi (c*64+32+l0..+7).
        // Биты qh — по подблокам c (как u1/u2 <<= 2 в dot_q5_k),
        // маски: lo = 1<<(2c), hi = 1<<(2c+1).
        const uint u_lo = 1u << (2u * c);
        const uint u_hi = 1u << (2u * c + 1u);
        float v_lo[8], v_hi[8];
#pragma unroll
        for (uint i = 0; i < 8; ++i) {
            const uchar b = qs[i];
            const uchar hb = qh[i];
            v_lo[i] = dl_lo * ((float)((uint)(b & 0xFu) + ((hb & u_lo) ? 16u : 0u))) - ml_lo;
            v_hi[i] = dl_hi * ((float)((uint)(b >> 4) + ((hb & u_hi) ? 16u : 0u))) - ml_hi;
        }
        // A: float4 (позиции кратны 4), порядок пар (lo[i], hi[i]) — как в
        // dot_q5_k; внутри пары lo затем hi.
        const uint off_lo = c * 64u + l0;
        const uint off_hi = off_lo + 32u;
#pragma unroll
        for (uint m = 0; m < M; ++m) {
            device const float* am = A + (ulong)m * K + (ulong)sb * 256u;
            const float4 al0 = *(device const float4*)(am + off_lo);
            const float4 al1 = *(device const float4*)(am + off_lo + 4u);
            const float4 ah0 = *(device const float4*)(am + off_hi);
            const float4 ah1 = *(device const float4*)(am + off_hi + 4u);
            const float4 vl0 = {v_lo[0], v_lo[1], v_lo[2], v_lo[3]};
            const float4 vl1 = {v_lo[4], v_lo[5], v_lo[6], v_lo[7]};
            const float4 vh0 = {v_hi[0], v_hi[1], v_hi[2], v_hi[3]};
            const float4 vh1 = {v_hi[4], v_hi[5], v_hi[6], v_hi[7]};
            const float4 acc0 = al0 * vl0 + ah0 * vh0;
            const float4 acc1 = al1 * vl1 + ah1 * vh1;
            sumf[m] += (acc0.x + acc0.y) + (acc0.z + acc0.w)
                     + ((acc1.x + acc1.y) + (acc1.z + acc1.w));
        }
    }
#pragma unroll
    for (uint m = 0; m < M; ++m) {
        const float v = simd_sum(sumf[m]);
        if (lane == 0u) C[(ulong)m * N + n] = v;
    }
}

typedef decltype(mv_q5_k_mb<1>) mv_q5_k_mb_t;
template [[host_name("mv_q5_k_mb2")]] kernel mv_q5_k_mb_t mv_q5_k_mb<2>;
template [[host_name("mv_q5_k_mb3")]] kernel mv_q5_k_mb_t mv_q5_k_mb<3>;
template [[host_name("mv_q5_k_mb4")]] kernel mv_q5_k_mb_t mv_q5_k_mb<4>;
template [[host_name("mv_q5_k_mb5")]] kernel mv_q5_k_mb_t mv_q5_k_mb<5>;
template [[host_name("mv_q5_k_mb6")]] kernel mv_q5_k_mb_t mv_q5_k_mb<6>;

typedef decltype(mv_iq4_xs_mb<1>) mv_iq4_xs_mb_t;
template [[host_name("mv_iq4_xs_mb1")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<1>;
template [[host_name("mv_iq4_xs_mb2")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<2>;
template [[host_name("mv_iq4_xs_mb3")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<3>;
template [[host_name("mv_iq4_xs_mb4")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<4>;
template [[host_name("mv_iq4_xs_mb5")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<5>;
template [[host_name("mv_iq4_xs_mb6")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<6>;

// ── MV-варианты (decode, batch=1): A — вектор [K], thread на выходной n ────

kernel void mv_q4_0(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 18;
        float d = (float)load_half_unaligned(blk);
        device const uchar* q = blk + 2;
        uint k0 = b * 32;
        // Канонический ggml: байт j — веса j и j+16 (см. gemm_nt_q4_0).
        for (uint j = 0; j < 16; ++j) {
            uchar v = q[j];
            acc += A[k0 + j] * ((float)(v & 0xF) - 8.0f) * d;
            acc += A[k0 + j + 16] * ((float)(v >> 4) - 8.0f) * d;
        }
    }
    C[n] = acc;
}

kernel void mv_q8_0(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 34;
        float d = (float)load_half_unaligned(blk);
        device const int8_t* q = (device const int8_t*)(blk + 2);
        uint k0 = b * 32;
        for (uint j = 0; j < 32; ++j) {
            acc += A[k0 + j] * (float)q[j] * d;
        }
    }
    C[n] = acc;
}

kernel void mv_q4_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint superblocks = K / 256;
    float acc = 0.0f;
    for (uint sb = 0; sb < superblocks; ++sb) {
        device const uchar* s = B + (n * superblocks + sb) * 144;
        acc += dot_q4_k(A + sb * 256, s);
    }
    C[n] = acc;
}

kernel void mv_q6_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint superblocks = K / 256;
    float acc = 0.0f;
    for (uint sb = 0; sb < superblocks; ++sb) {
        device const uchar* s = B + (n * superblocks + sb) * 210;
        acc += dot_q6_k(A + sb * 256, s);
    }
    C[n] = acc;
}

// ── K-расщеплённый matvec для K-квантов (Q4_K/Q6_K): декод ─────────────────
// Кернелы выше — «поток на строку весов»: поток читает суперблоки (144/210 Б)
// последовательно и платит латентность памяти на каждый. На MoE (OLMoE:
// N=1024, K=2048 на эксперта) это давало ~17 ГБ/с против ~94 у llama.cpp —
// декод 35 против 208 tok/s; на плотной GLM-4-9B-Q4_K_M 19 против 41 tok/s.
// Здесь строку ведёт simdgroup: лейн ℓ берёт суперблоки ℓ, ℓ+32, … (тем же
// скалярным dot-хелпером, только частично) и складывает через simd_sum.
// 4 строки на threadgroup (по simdgroupу), 128 потоков.
#define MVKS_NR 4u
template <int KIND>  // 0 = Q4_K (144 Б/суперблок), 1 = Q6_K (210 Б)
kernel void mv_ks_t(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint bsz = (KIND == 0) ? 144u : 210u;
    const uint sb_count = K / 256u;
    const uint r = tgid * MVKS_NR + sgid;
    const bool ok = r < N;
    float acc = 0.0f;
    if (ok) {
        device const uchar* row = B + (ulong)r * sb_count * bsz;
        for (uint sb = lane; sb < sb_count; sb += 32u) {
            device const uchar* blk = row + sb * bsz;
            acc += (KIND == 0) ? dot_q4_k(A + sb * 256u, blk)
                               : dot_q6_k(A + sb * 256u, blk);
        }
    }
    const float v = simd_sum(acc);
    if (lane == 0u && ok) C[r] = v;
}

// Q6_K matvec — порт эталонного kernel_mul_mv_q6_K_f32 (llama.cpp 0.4.0):
// simdgroup на ПАРУ строк весов, по 16 лейнов на строку (ix = lane%2 делит
// суперблоки по чётности), y грузится один раз на пару. Порядок суммирования
// по K отличается от mv_q6_k (частичные суммы лейнов + simd_sum) — результат
// НЕ побитовый, приёмка парными прогонами text_parity на всех моделях-приёмке
// (паритет первичен; фоллбэк INFER_Q6K_NO_R2).
kernel void mv_q6_k_r2(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint nb = K / 256u;
    const uint first_row = 2u * tgid.y;
    if (first_row >= N) return;
    const bool two = first_row + 1u < N;
    const ulong row_bytes = (ulong)nb * 210u;

    const MvQ6KDecomp d = mv_q6k_decomp(lane);

    device const uchar* x = B + (ulong)first_row * row_bytes;
    float sumf[2] = {0.f, 0.f};
    for (uint i = d.ix; i < nb; i += 2u) {
        float yl[16];
        mv_q6k_y(A, i, d, yl);
        for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
            mv_q6k_acc(x + (ulong)rr * row_bytes + (ulong)i * 210u, d, yl, sumf[rr]);
        }
    }
    const float v0 = simd_sum(sumf[0]);
    if (lane == 0u) C[first_row] = v0;
    if (two) {
        const float v1 = simd_sum(sumf[1]);
        if (lane == 0u) C[first_row + 1u] = v1;
    }
}

kernel void mv_q5_0(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* s = B + (n * blocks + b) * 22;
        acc += dot_q5_0(A + b * 32, s);
    }
    C[n] = acc;
}

kernel void mv_q5_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint n [[thread_position_in_grid]])
{
    if (n >= N) return;
    const uint superblocks = K / 256;
    float acc = 0.0f;
    for (uint sb = 0; sb < superblocks; ++sb) {
        device const uchar* s = B + (n * superblocks + sb) * 176;
        acc += dot_q5_k(A + sb * 256, s);
    }
    C[n] = acc;
}

// Q5_K matvec — порт эталонного kernel_mul_mv_q5_K_f32 (llama.cpp 0.4.0):
// строку ведут 8 лейнов (tid = lane/4), 4 лейна делят суперблоки (ix = lane%4),
// на simdgroup — ПАРА строк, y грузится один раз на пару. Порядок суммирования
// по K отличается от mv_q5_k — приёмка парными text_parity (паритет первичен,
// фоллбэк INFER_Q5K_NO_R2).
kernel void mv_q5_k_r2(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint nb = K / 256u;
    const uint first_row = 2u * tgid.y;
    if (first_row >= N) return;
    const bool two = first_row + 1u < N;
    const ulong row_bytes = (ulong)nb * 176u;

    const uint tid = lane / 4u;   // 0..7
    const uint ix  = lane % 4u;   // 0..3 — какие суперблоки берёт лейн
    const uint iq  = tid / 4u;    // 0..1 — половина суперблока
    const uint ir  = tid % 4u;    // 0..3 — четверть внутри половины
    const uint l0  = 8u * ir;
    const uint q_off = 32u * iq + l0;
    const uint y_off = 64u * iq + l0;

    const uint hm1 = 1u << (2u * iq);
    const uint hm2 = hm1 << 1;
    const uint hm3 = hm1 << 4;
    const uint hm4 = hm2 << 4;

    device const uchar* x = B + (ulong)first_row * row_bytes;
    float sumf[2] = {0.f, 0.f};

    for (uint i = ix; i < nb; i += 4u) {
        device const uchar* blk = x + (ulong)i * 176u;
        device const float* y1 = A + (ulong)i * 256u + y_off;
        device const float* y2 = y1 + 128u;

        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (uint l = 0; l < 8u; ++l) {
            yl[l + 0] = y1[l +  0]; sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l +  0]; sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }

        for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
            device const uchar* xb = blk + (ulong)rr * row_bytes;
            device const uchar* q1 = xb + 48u + q_off;
            device const uchar* q2 = q1 + 64u;
            device const uchar* qh = xb + 16u + l0;
            device const uchar* sp = xb + 4u + 2u * iq;
            const float dh = (float)load_half_unaligned(xb);
            const float dm = (float)load_half_unaligned(xb + 2u);

            uint16_t sc16[4];
            sc16[0] = (uint16_t)(sp[0] | (sp[1] << 8)) & 0x3f3fu;
            sc16[1] = (uint16_t)(sp[4] | (sp[5] << 8)) & 0x3f3fu;
            sc16[2] = ((uint16_t)(sp[8] | (sp[9] << 8)) & 0x0f0fu) |
                      (((uint16_t)(sp[0] | (sp[1] << 8)) & 0xc0c0u) >> 2);
            sc16[3] = (((uint16_t)(sp[8] | (sp[9] << 8)) >> 4) & 0x0f0fu) |
                      (((uint16_t)(sp[4] | (sp[5] << 8)) & 0xc0c0u) >> 2);
            thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (uint l = 0; l < 8u; ++l) {
                const uint h = qh[l];
                acc1[0] += yl[l + 0] * (float)(q1[l] & 0x0Fu);
                acc1[1] += yl[l + 8] * (float)(q1[l] & 0xF0u);
                acc1[2] += yh[l + 0] * (float)(q2[l] & 0x0Fu);
                acc1[3] += yh[l + 8] * (float)(q2[l] & 0xF0u);
                acc2[0] += (h & hm1) ? yl[l + 0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l + 8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l + 0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l + 8] : 0.f;
            }
            sumf[rr] += dh * (sc8[0] * (acc1[0] + 16.f * acc2[0]) +
                              sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                              sc8[4] * (acc1[2] + 16.f * acc2[2]) +
                              sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                        dm * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                              sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
    }
    const float v0 = simd_sum(sumf[0]);
    if (lane == 0u) C[first_row] = v0;
    if (two) {
        const float v1 = simd_sum(sumf[1]);
        if (lane == 0u) C[first_row + 1u] = v1;
    }
}

// ── Q4_K matvec: порт kernel_mul_mv_q4_K_f32_impl (N_R0=2) ──────────────────
// 32 лейна на строку: ix = lane/8 делит суперблоки с шагом 4, it = lane%8
// (iq = it/4 — половина 128-байтного qs, ir = it%4 — четверть uint16-чанка).
// Пара строк на simdgroup: yl/yh/sumy (активации) грузятся ОДИН раз на пару
// — тот же рычаг, что у mv_q6_k_r2/mv_q5_k_r2 (прежний mv_q4_k: поток на
// строку, 82–164 ГБ/с на формах Q4_K_M). Порядок суммирования по K отличается
// от скалярного dot_q4_k — класс MV_KS, приёмка парными text_parity на
// Q4_K-моделях (GLM-4-9B, OLMoE); фоллбэк INFER_Q4K_NO_R2.
kernel void mv_q4_k_r2(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint nb = K / 256u;
    const uint first_row = 2u * tgid.y;
    if (first_row >= N) return;
    const bool two = first_row + 1u < N;
    const ulong row_bytes = (ulong)nb * 144u;

    const MvQ4KDecomp d = mv_q4k_decomp(lane);

    device const uchar* x = B + (ulong)first_row * row_bytes;
    float sumf[2] = {0.f, 0.f};

    for (uint ib = d.ix; ib < nb; ib += 4u) {
        MvQ4KY y;
        mv_q4k_y(A, ib, d, y);
        for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
            mv_q4k_acc(x + (ulong)rr * row_bytes + (ulong)ib * 144u, d, y, sumf[rr]);
        }
    }
    const float v0 = simd_sum(sumf[0]);
    if (lane == 0u) C[first_row] = v0;
    if (two) {
        const float v1 = simd_sum(sumf[1]);
        if (lane == 0u) C[first_row + 1u] = v1;
    }
}
// Прежний mv_q8_0 — поток на строку: соседние потоки читали строки с шагом
// K·34/32 байт (некоалесцированно), и decode упирался не в полосу памяти.
// Здесь simdgroup ведёт R строк: 4 лейна на Q8_0-блок (по 8 значений),
// 8 блоков за итерацию — лейны читают СОСЕДНИЕ байты одной строки;
// активации (8 float на лейн) переиспользуются на R строк. Итог строки —
// simd_sum. Threadgroup = 2 simdgroup'а (64 потока) → 2·R строк.
#define MV_Q8_R 4u
kernel void mv_q8_0_sg(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = (tgid * 2u + sgid) * MV_Q8_R;
    if (row0 >= N) return;
    const uint nb = K / 32u;
    const uint il = lane % 4u;   // четверть блока: значения il*8 .. il*8+8
    const uint ib0 = lane / 4u;  // 0..7 — блок в пачке из 8
    float acc[MV_Q8_R] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib = ib0; ib < nb; ib += 8u) {
        device const float* a = A + ib * 32u + il * 8u;
        const float x0 = a[0], x1 = a[1], x2 = a[2], x3 = a[3];
        const float x4 = a[4], x5 = a[5], x6 = a[6], x7 = a[7];
        for (uint r = 0u; r < MV_Q8_R; ++r) {
            const uint row = row0 + r;
            if (row >= N) break;
            device const uchar* blk = B + ((ulong)row * nb + ib) * 34u;
            const float d = (float)load_half_unaligned(blk);
            device const char* q = (device const char*)(blk + 2u + il * 8u);
            const float s = x0 * (float)q[0] + x1 * (float)q[1] + x2 * (float)q[2] + x3 * (float)q[3]
                          + x4 * (float)q[4] + x5 * (float)q[5] + x6 * (float)q[6] + x7 * (float)q[7];
            acc[r] += s * d;
        }
    }
    for (uint r = 0u; r < MV_Q8_R; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < N) C[row0 + r] = v;
    }
}

// ── Q8_0 matvec с разбиением K между simdgroup'ами (Ф8.2) ────────────────────
// Схема ggml kernel_mul_mv_q8_0: threadgroup из 4 simdgroup'ов считает 2
// строки, каждый simdgroup — свою четверть блоков K (лейн: 8 значений
// блока), итог — simd_sum + сумма по simdgroup'ам через threadgroup-память.
// Против варианта «simdgroup на 4 строки» — вчетверо больше потоков в
// полёте: на wqkv/wo (6144/4096 строк) полоса памяти недобиралась (120 ГБ/с).
#define MV8_NR0 2u
#define MV8_NSG 4u
kernel void mv_q8_0_ks(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Редукция — ПОБИТОВО как helper_mv_reduce_and_write оркала: 32 слота на
    // строку (зануляет sgitg 0), частичные суммы simd_sum пишут лейн 0 своего
    // simdgroup'а в слот sgitg, финал — simd_sum по 32 лейнам (бабочка:
    // (v0+v2)+(v1+v3), а не последовательная сумма — порядок слагаемых в f32
    // важен для паритета логов, GLM-4-9B ffn_down).
    threadgroup float sh[MV8_NR0 * 32u];
    const uint r0 = tgid * MV8_NR0;
    const uint nb = K / 32u;
    const uint ix = lane / 4u, il = lane % 4u;
    float sumf[MV8_NR0] = {0.0f, 0.0f};
    for (uint ib = sgid * 8u + ix; ib < nb; ib += MV8_NSG * 8u) {
        device const float* yb = A + ib * 32u + il * 8u;
        float yl[8];
        for (uint i = 0; i < 8u; ++i) yl[i] = yb[i];
        for (uint row = 0; row < MV8_NR0; ++row) {
            if (r0 + row >= N) break;
            device const uchar* blk = B + ((ulong)(r0 + row) * nb + ib) * 34u;
            const float d = (float)load_half_unaligned(blk);
            device const char* qs = (device const char*)(blk + 2u + il * 8u);
            float sq = 0.0f;
            for (uint i = 0; i < 8u; ++i) sq += (float)qs[i] * yl[i];
            sumf[row] += sq * d;
        }
    }
    for (uint row = 0; row < MV8_NR0; ++row) {
        if (sgid == 0u) sh[row * 32u + lane] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < MV8_NR0; ++row) {
        if (lane == 0u) sh[row * 32u + sgid] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < MV8_NR0; ++row) {
        if (r0 + row >= N) break;
        const float tot = simd_sum(sh[row * 32u + lane]);
        if (lane == 0u && sgid == 0u) C[r0 + row] = tot;
    }
}

// ── Q8_0 × малый батч M ≤ 8 (окно верификации спека, Ф3) ────────────────────
// Окно спека — forward 1+γ строк. Тайловый GEMM (тайл 32 строки активаций,
// 64 строки весов) при M ≤ 8 упирается в малое число threadgroup'ов и длинную
// K-петлю с барьерами: ≈3.5× matvec'а при любом M. Схема ggml
// kernel_mul_mv_ext (4 строки весов на simdgroup, 8 лейнов на строку) тоже не
// масштабируется: каждые 4 строки весов заново читают все M строк активаций
// (на M=8 трафик активаций в 8× больше трафика весов; у llama.cpp pp8 = 4.8×
// pp1 на этой машине). Здесь все 32 лейна делят K (лейн — 4 значения блока,
// 4 блока за шаг, чтение строки весов — 136 байт подряд), а каждый
// simdgroup ведёт NR строк весов: активации шага читаются один раз на NR
// строк, веса — один раз на M строк активаций.
template <short M, short NR>
kernel void mv_q8_0_mb(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q8_0 [N, K]
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint nb = K / 32u;
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    const uint bl = lane / 8u;          // блок внутри шага из 4 блоков
    const uint el = (lane % 8u) * 4u;   // смещение 4 значений в блоке
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * 34u;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = bl; ib < nb; ib += 4u) {
        float4 y[M];
#pragma unroll
        for (short m = 0; m < M; ++m) {
            y[m] = *(device const float4*)(A + (ulong)m * K + ib * 32u + el);
        }
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * 34u;
            const float d = (float)load_half_unaligned(blk);
            device const char* qs = (device const char*)(blk + 2u + el);
            const float4 w = float4(qs[0], qs[1], qs[2], qs[3]) * d;
#pragma unroll
            for (short m = 0; m < M; ++m) acc[r][m] += dot(w, y[m]);
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

// Набор — по замерам examples/gemm_small_m (сумма матмулов forward'а 8B):
// M=2..4 — NR=4 (26–32 мс при декоде 29); M=5,6 — NR=8 (49–60); M=7,8 —
// два прохода по ≤4 строки (≈64; один проход — 78–165, тайловый — 107).
typedef decltype(mv_q8_0_mb<2, 4>) mv_q8_0_mb_t;
template [[host_name("mv_q8_0_mb2")]] kernel mv_q8_0_mb_t mv_q8_0_mb<2, 4>;
template [[host_name("mv_q8_0_mb3")]] kernel mv_q8_0_mb_t mv_q8_0_mb<3, 4>;
template [[host_name("mv_q8_0_mb4")]] kernel mv_q8_0_mb_t mv_q8_0_mb<4, 4>;
template [[host_name("mv_q8_0_mb5")]] kernel mv_q8_0_mb_t mv_q8_0_mb<5, 8>;
template [[host_name("mv_q8_0_mb6")]] kernel mv_q8_0_mb_t mv_q8_0_mb<6, 8>;

// Q4_0 матвек с батчем строк активаций (окно верификации 1 < m ≤ 8): та же
// раскладка, что у mv_q8_0_mb (32 лейна делят K шагами по 4 блока, NR строк
// весов на simdgroup, веса читаются 1× на m строк активаций), но блок Q4_0 —
// 18 байт (f16 d + 16 байт нибблов). 8 лейнов на блок: лейн берёт 2 байта =
// 4 значения канонической раскладки ggml — младшие нибблы байтов 2l, 2l+1
// (значения 2l, 2l+1) и их старшие пары (значения 2l+16, 2l+17); A читается
// двумя float2 (сдвиги 8·l и 64+8·l байт — 8-байтовое выравнивание).
template <short M, short NR>
kernel void mv_q4_0_mb(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q4_0 [N, K]
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const uint nb = K / 32u;
    const uint r0 = (tgid * nsg + sg) * NR;
    if (r0 >= N) return;
    const uint bl = lane / 8u;        // блок внутри шага из 4 блоков
    const uint li = lane % 8u;        // 2 байта (4 значения) внутри блока
    device const uchar* rows[NR];
    for (short r = 0; r < NR; ++r) {
        rows[r] = B + (ulong)min(r0 + r, N - 1u) * nb * 18u;
    }
    float acc[NR][M];
#pragma unroll
    for (short r = 0; r < NR; ++r)
#pragma unroll
        for (short m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint ib = bl; ib < nb; ib += 4u) {
        float2 y0[M], y1[M];
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const ulong base = (ulong)m * K + ib * 32u + 2u * li;
            y0[m] = *(device const float2*)(A + base);
            y1[m] = *(device const float2*)(A + base + 16u);
        }
#pragma unroll
        for (short r = 0; r < NR; ++r) {
            device const uchar* blk = rows[r] + ib * 18u;
            const float d = (float)load_half_unaligned(blk);
            device const uchar* q = blk + 2u + 2u * li;
            const uchar b0 = q[0];
            const uchar b1 = q[1];
            const float w0 = ((float)(b0 & 0x0Fu) - 8.0f) * d;
            const float w1 = ((float)(b1 & 0x0Fu) - 8.0f) * d;
            const float w2 = ((float)(b0 >> 4u) - 8.0f) * d;
            const float w3 = ((float)(b1 >> 4u) - 8.0f) * d;
#pragma unroll
            for (short m = 0; m < M; ++m) {
                acc[r][m] += w0 * y0[m].x + w1 * y0[m].y + w2 * y1[m].x + w3 * y1[m].y;
            }
        }
    }
#pragma unroll
    for (short r = 0; r < NR; ++r) {
#pragma unroll
        for (short m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0u && r0 + r < N) C[(ulong)m * N + r0 + r] = v;
        }
    }
}

// NR — как у Q8_0 mb (раскладка та же; m 7/8 — два прохода по ≤4 строки).
typedef decltype(mv_q4_0_mb<2, 4>) mv_q4_0_mb_t;
template [[host_name("mv_q4_0_mb2")]] kernel mv_q4_0_mb_t mv_q4_0_mb<2, 4>;
template [[host_name("mv_q4_0_mb3")]] kernel mv_q4_0_mb_t mv_q4_0_mb<3, 4>;
template [[host_name("mv_q4_0_mb4")]] kernel mv_q4_0_mb_t mv_q4_0_mb<4, 4>;
template [[host_name("mv_q4_0_mb5")]] kernel mv_q4_0_mb_t mv_q4_0_mb<5, 8>;
template [[host_name("mv_q4_0_mb6")]] kernel mv_q4_0_mb_t mv_q4_0_mb<6, 8>;

typedef decltype(mv_ks_t<0>) mv_ks_t_t;
template [[host_name("mv_ks_q4_k")]] kernel mv_ks_t_t mv_ks_t<0>;
template [[host_name("mv_ks_q6_k")]] kernel mv_ks_t_t mv_ks_t<1>;

// ════════════════════════════════════════════════════════════════════════════
// mul_mv_ext — малобатчевый matvec (2..8 строк активаций). Порт
// kernel_mul_mv_ext_q4_f32_impl / kernel_mul_mv_ext_q4x4_f32_impl
// (ggml-metal 5266f24da, mul_mv.metal 602..835). Эталон берёт эти кернелы
// вместо classic mul_mv при src1 F32 и K%128==0: Q4_0/Q5_0/Q8_0 — m∈[2,8],
// Q4_K/Q5_K/Q6_K — m∈[4,8] (IQ4_XS в ext-списке нет → classic mul_mv).
// nsg=2; nxpsg = (K%256==0 && m<3) ? 16 : 8; r1ptg: 2→2, 3/6→3, 4/7/8→4,
// 5→5. Порядок суммирования (chpt-чанки с шагом chpt*nxpsg, затем
// simd_shuffle_down-дерево 8/4/2/1) отличается и от classic mul_mv, и от
// mul_mm — поэтому выбор кернела эталоном воспроизводится ТОЧНО (паритет
// первичен). Декванты — дословные dequantize_*_t4/t4x4 из dequantize.h.
// ════════════════════════════════════════════════════════════════════════════

// Раскладки блоков — дословно ggml-common.h (q4_K: q4k_block/q4k_dequantize_mm
// уже в gemm.metal, concat-порядок common→gemm→mul_mv их видит).
typedef struct { half d; uchar qs[16]; } ext_q4_0_block;
typedef struct { half d; uchar qh[4]; uchar qs[16]; } ext_q5_0_block;
typedef struct { half d; char qs[32]; } ext_q8_0_block;
typedef struct { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; } ext_q5k_block;
typedef struct { uchar ql[128]; uchar qh[64]; char scales[16]; half d; } ext_q6k_block;

void ext_deq_f32(device const float4 * src, short il, thread float4 & reg) {
    reg = (float4)(*src);
}
void ext_deq_f16(device const half4 * src, short il, thread float4 & reg) {
    reg = (float4)(*(src));
}
void ext_deq_q4_0(device const ext_q4_0_block * xb, short il, thread float4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}
void ext_deq_q5_0(device const ext_q5_0_block * xb, short il, thread float4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}
void ext_deq_q8_0(device const ext_q8_0_block * xb, short il, thread float4 & reg) {
    device const char4 * qs = (device const char4 *) xb->qs;
    const float d = xb->d;

    reg = (float4) (float4(qs[il]) * d);
}
// dequantize_q5_K (dequantize.h 450) — get_scale_min_k4_just2 = q4k_scale_min_just2 (gemm.metal).
void ext_deq_q5_K(device const ext_q5k_block * xb, short il, thread float4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = q4k_scale_min_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}
// dequantize_q6_K (dequantize.h 473).
void ext_deq_q6_K(device const ext_q6k_block * xb, short il, thread float4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// kernel_mul_mv_ext_q4_f32_impl (mul_mv.metal 602) — дословно, константы FC_*
// фиксированы диспатчем эталона (nsg=2, ne12=1, r2=r3=1 → i12=i13=0).
template<short r1ptg, short NXPSG, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
static inline void ext_q4_impl(
        int ne00, int ne01, int ne11, ulong nb01, ulong nb11, int ne0,
        device const uchar * src0, device const uchar * src1, device float * dst,
        uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG   = 2;
    const short nxpsg = NXPSG;

    const short chpt = 4; // chunks per thread

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;

    const ulong offset0 = (ulong)i01*nb01;
    const ulong offset1 = (ulong)i11*nb11;

    device const q_t * xq = (i01 < ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < ne11) ? (device const float4 *) (src1 + offset1 + ir1*nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < ne11; ++ir1) {
            device float * dst_f32 = dst + (ulong)(i11 + ir1)*ne0;

            if (i01 < ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// kernel_mul_mv_ext_q4x4_f32_impl (mul_mv.metal 705) — дословно.
template<short r1ptg, short NXPSG, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
static inline void ext_q4x4_impl(
        int ne00, int ne01, int ne11, ulong nb01, ulong nb11, int ne0,
        device const uchar * src0, device const uchar * src1, device float * dst,
        uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG   = 2;
    const short nxpsg = NXPSG;

    const short chpt = 1;

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;

    const ulong offset0 = (ulong)i01*nb01;
    const ulong offset1 = (ulong)i11*nb11;

    device const q_t * xq = (i01 < ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4x4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] +=
                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);

            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < ne11; ++ir1) {
            device float * dst_f32 = dst + (ulong)(i11 + ir1)*ne0;

            if (i01 < ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// Диспатчеры (buffer-конвенция движка: 0 = активации A [M,K], 1 = веса B,
// 2 = C [M,N]; скаляры — 3..8).
template<short r1ptg, short NXPSG, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void mv_ext_q4_disp(
        device const uchar* A [[buffer(0)]],
        device const uchar* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant int& ne00 [[buffer(3)]],
        constant int& ne01 [[buffer(4)]],
        constant int& ne11 [[buffer(5)]],
        constant ulong& nb01 [[buffer(6)]],
        constant ulong& nb11 [[buffer(7)]],
        constant int& ne0 [[buffer(8)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    ext_q4_impl<r1ptg, NXPSG, q_t, chpb, deq_t4>(ne00, ne01, ne11, nb01, nb11, ne0, B, A, C, tgpig, tiisg, sgitg);
}

template<short r1ptg, short NXPSG, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void mv_ext_q4x4_disp(
        device const uchar* A [[buffer(0)]],
        device const uchar* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant int& ne00 [[buffer(3)]],
        constant int& ne01 [[buffer(4)]],
        constant int& ne11 [[buffer(5)]],
        constant ulong& nb01 [[buffer(6)]],
        constant ulong& nb11 [[buffer(7)]],
        constant int& ne0 [[buffer(8)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    ext_q4x4_impl<r1ptg, NXPSG, q_t, chpb, deq_t4x4>(ne00, ne01, ne11, nb01, nb11, ne0, B, A, C, tgpig, tiisg, sgitg);
}

// Инстансы: nxpsg=8 — r1ptg 2..5 (для K-квантов достижимы 3..5, лишние
// безвредны); nxpsg=16 — только r1ptg=2 (m=2 && K%256==0). K-кванты — q4x4.
typedef decltype(mv_ext_q4_disp<2, 8, float4,   1, ext_deq_f32>)  mv_ext_f32_t;
template [[host_name("mv_ext_f32_x8_r2")]]  kernel mv_ext_f32_t mv_ext_q4_disp<2, 8, float4,   1, ext_deq_f32>;
template [[host_name("mv_ext_f32_x8_r3")]]  kernel mv_ext_f32_t mv_ext_q4_disp<3, 8, float4,   1, ext_deq_f32>;
template [[host_name("mv_ext_f32_x8_r4")]]  kernel mv_ext_f32_t mv_ext_q4_disp<4, 8, float4,   1, ext_deq_f32>;
template [[host_name("mv_ext_f32_x8_r5")]]  kernel mv_ext_f32_t mv_ext_q4_disp<5, 8, float4,   1, ext_deq_f32>;
template [[host_name("mv_ext_f32_x16_r2")]] kernel mv_ext_f32_t mv_ext_q4_disp<2, 16, float4,  1, ext_deq_f32>;

typedef decltype(mv_ext_q4_disp<2, 8, half4,    1, ext_deq_f16>)  mv_ext_f16_t;
template [[host_name("mv_ext_f16_x8_r2")]]  kernel mv_ext_f16_t mv_ext_q4_disp<2, 8, half4,    1, ext_deq_f16>;
template [[host_name("mv_ext_f16_x8_r3")]]  kernel mv_ext_f16_t mv_ext_q4_disp<3, 8, half4,    1, ext_deq_f16>;
template [[host_name("mv_ext_f16_x8_r4")]]  kernel mv_ext_f16_t mv_ext_q4_disp<4, 8, half4,    1, ext_deq_f16>;
template [[host_name("mv_ext_f16_x8_r5")]]  kernel mv_ext_f16_t mv_ext_q4_disp<5, 8, half4,    1, ext_deq_f16>;
template [[host_name("mv_ext_f16_x16_r2")]] kernel mv_ext_f16_t mv_ext_q4_disp<2, 16, half4,   1, ext_deq_f16>;

typedef decltype(mv_ext_q4_disp<2, 8, ext_q4_0_block, 8, ext_deq_q4_0>) mv_ext_q4_0_t;
template [[host_name("mv_ext_q4_0_x8_r2")]]  kernel mv_ext_q4_0_t mv_ext_q4_disp<2, 8, ext_q4_0_block, 8, ext_deq_q4_0>;
template [[host_name("mv_ext_q4_0_x8_r3")]]  kernel mv_ext_q4_0_t mv_ext_q4_disp<3, 8, ext_q4_0_block, 8, ext_deq_q4_0>;
template [[host_name("mv_ext_q4_0_x8_r4")]]  kernel mv_ext_q4_0_t mv_ext_q4_disp<4, 8, ext_q4_0_block, 8, ext_deq_q4_0>;
template [[host_name("mv_ext_q4_0_x8_r5")]]  kernel mv_ext_q4_0_t mv_ext_q4_disp<5, 8, ext_q4_0_block, 8, ext_deq_q4_0>;
template [[host_name("mv_ext_q4_0_x16_r2")]] kernel mv_ext_q4_0_t mv_ext_q4_disp<2, 16, ext_q4_0_block, 8, ext_deq_q4_0>;

typedef decltype(mv_ext_q4_disp<2, 8, ext_q5_0_block, 8, ext_deq_q5_0>) mv_ext_q5_0_t;
template [[host_name("mv_ext_q5_0_x8_r2")]]  kernel mv_ext_q5_0_t mv_ext_q4_disp<2, 8, ext_q5_0_block, 8, ext_deq_q5_0>;
template [[host_name("mv_ext_q5_0_x8_r3")]]  kernel mv_ext_q5_0_t mv_ext_q4_disp<3, 8, ext_q5_0_block, 8, ext_deq_q5_0>;
template [[host_name("mv_ext_q5_0_x8_r4")]]  kernel mv_ext_q5_0_t mv_ext_q4_disp<4, 8, ext_q5_0_block, 8, ext_deq_q5_0>;
template [[host_name("mv_ext_q5_0_x8_r5")]]  kernel mv_ext_q5_0_t mv_ext_q4_disp<5, 8, ext_q5_0_block, 8, ext_deq_q5_0>;
template [[host_name("mv_ext_q5_0_x16_r2")]] kernel mv_ext_q5_0_t mv_ext_q4_disp<2, 16, ext_q5_0_block, 8, ext_deq_q5_0>;

typedef decltype(mv_ext_q4_disp<2, 8, ext_q8_0_block, 8, ext_deq_q8_0>) mv_ext_q8_0_t;
template [[host_name("mv_ext_q8_0_x8_r2")]]  kernel mv_ext_q8_0_t mv_ext_q4_disp<2, 8, ext_q8_0_block, 8, ext_deq_q8_0>;
template [[host_name("mv_ext_q8_0_x8_r3")]]  kernel mv_ext_q8_0_t mv_ext_q4_disp<3, 8, ext_q8_0_block, 8, ext_deq_q8_0>;
template [[host_name("mv_ext_q8_0_x8_r4")]]  kernel mv_ext_q8_0_t mv_ext_q4_disp<4, 8, ext_q8_0_block, 8, ext_deq_q8_0>;
template [[host_name("mv_ext_q8_0_x8_r5")]]  kernel mv_ext_q8_0_t mv_ext_q4_disp<5, 8, ext_q8_0_block, 8, ext_deq_q8_0>;
template [[host_name("mv_ext_q8_0_x16_r2")]] kernel mv_ext_q8_0_t mv_ext_q4_disp<2, 16, ext_q8_0_block, 8, ext_deq_q8_0>;

typedef decltype(mv_ext_q4x4_disp<3, 8, q4k_block, 16, q4k_dequantize_mm>) mv_ext_q4_k_t;
template [[host_name("mv_ext_q4_k_x8_r3")]] kernel mv_ext_q4_k_t mv_ext_q4x4_disp<3, 8, q4k_block, 16, q4k_dequantize_mm>;
template [[host_name("mv_ext_q4_k_x8_r4")]] kernel mv_ext_q4_k_t mv_ext_q4x4_disp<4, 8, q4k_block, 16, q4k_dequantize_mm>;
template [[host_name("mv_ext_q4_k_x8_r5")]] kernel mv_ext_q4_k_t mv_ext_q4x4_disp<5, 8, q4k_block, 16, q4k_dequantize_mm>;

typedef decltype(mv_ext_q4x4_disp<3, 8, ext_q5k_block, 16, ext_deq_q5_K>) mv_ext_q5_k_t;
template [[host_name("mv_ext_q5_k_x8_r3")]] kernel mv_ext_q5_k_t mv_ext_q4x4_disp<3, 8, ext_q5k_block, 16, ext_deq_q5_K>;
template [[host_name("mv_ext_q5_k_x8_r4")]] kernel mv_ext_q5_k_t mv_ext_q4x4_disp<4, 8, ext_q5k_block, 16, ext_deq_q5_K>;
template [[host_name("mv_ext_q5_k_x8_r5")]] kernel mv_ext_q5_k_t mv_ext_q4x4_disp<5, 8, ext_q5k_block, 16, ext_deq_q5_K>;

typedef decltype(mv_ext_q4x4_disp<3, 8, ext_q6k_block, 16, ext_deq_q6_K>) mv_ext_q6_k_t;
template [[host_name("mv_ext_q6_k_x8_r3")]] kernel mv_ext_q6_k_t mv_ext_q4x4_disp<3, 8, ext_q6k_block, 16, ext_deq_q6_K>;
template [[host_name("mv_ext_q6_k_x8_r4")]] kernel mv_ext_q6_k_t mv_ext_q4x4_disp<4, 8, ext_q6k_block, 16, ext_deq_q6_K>;
template [[host_name("mv_ext_q6_k_x8_r5")]] kernel mv_ext_q6_k_t mv_ext_q4x4_disp<5, 8, ext_q6k_block, 16, ext_deq_q6_K>;

// ── Q5_0 matvec — побитовый порт оркального kernel_mul_mv_q5_0_f32 ──────────
// (mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0=4>, ggml-metal 5266f24da).
// КРИТИЧНО для паритета: кернел воспроизведён ДОСЛОВНО по форме — отдельная
// non-inline `inline` функция dot_y (linkonce_odr, как у оркала), struct-
// чтение qh (align-2 load), тот же вид циклов. Ручной инлайн dot_y в тело
// меняет форму IR и бэкенд Metal принимает ДРУГИЕ решения о контракции
// fmul+fadd в fma — расхождение логов на 1–3 ulp (проверено побитово на
// ffn_down GLM-4-9B: 0/4096 различий только с дословной формой).
// Схема: threadgroup 4 simdgroup'а × 4 строки (r0 = (tgid*4+sgid)*4); K между
// 32 лейнами: ix = lane/2 — номер блока (шаг NQ=16), il = (lane%2)*8 —
// половина блока; yl домножен на недостающие битовые веса (1, 1/256, 1/16,
// 1/4096); итог строки — simd_sum.
struct block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };

inline float dot_y_q5_0(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+16) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+16) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

kernel void mv_q5_0_sg(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = 4;
    constexpr short NW = 32;
    constexpr short NQ = 16;

    const int nb = K/32;

    const int r0 = (tgpig.x*NSG + sgitg)*4;

    device const float        * y = A;

    device const block_q5_0 * ax[4];
    for (int row = 0; row < 4; ++row) {
        ax[row] = ((device const block_q5_0 *) B) + (ulong)(r0 + row)*nb;
    }

    float sumf[4] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    const int ib0 = ix;

    float yl[16];

    device const float * yb = y + ib0*32 + il;

    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        for (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        for (short row = 0; row < 4; row++) {
            sumf[row] += dot_y_q5_0(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += 32 * 16;
    }

    for (int row = 0; row < 4; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < (int)N) {
            C[r0 + row] = tot;
        }
    }
}
