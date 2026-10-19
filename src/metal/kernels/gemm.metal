
// ── GEMM (row-major): C[m,n] = sum_k A[m*K+k] * B[k*N+n] ────────────────────
kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
}

// ── GEMM with B transposed: C[m,n] = sum_k A[m*K+k] * B[n*K+k] ──────────────
// (for y = x @ W^T with W stored [out, in] — avoids materializing W^T)
kernel void gemm_nt_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[n * K + k];
    }
    C[m * N + n] = acc;
}

// gemm_nt_f32 с float4-загрузками: накопление строго последовательное по k
// (x, y, z, w) — ПОБИТОВО идентично скалярному, но вчетверо меньше инструкций
// загрузки и вчетверо больше памяти в полёте на поток. Узкое место скалярной
// версии на маленьких N (GDN beta/alpha [48,5120]: 48 потоков на весь GPU) —
// латентность загрузок, 1.5 ГБ/с; здесь ~4× быстрее. Берётся при K%4==0 и
// 16-байтном выравнивании обоих операндов.
kernel void gemm_nt_f32_v4(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    device const float4* a = (device const float4*)(A + m * K);
    device const float4* b = (device const float4*)(B + n * K);
    const uint k4 = K / 4u;
    float acc = 0.0f;
    for (uint k = 0; k < k4; ++k) {
        const float4 av = a[k];
        const float4 bv = b[k];
        acc += av.x * bv.x;
        acc += av.y * bv.y;
        acc += av.z * bv.z;
        acc += av.w * bv.w;
    }
    C[m * N + n] = acc;
}

// m==1: F32 matvec — simdgroup на строку весов B (Ф8.1). Прежний v4 при m=1
// давал ОДИН поток на выход: n=96 (beta/alpha GDN) — 96 потоков на всю GPU,
// 16 ГБ/с против ~200 доступных (латентность цепочки загрузок на поток).
// Здесь 32 лейна делят K stride-ом по float4, частичные суммы — simd_sum.
// Порядок суммирования изменён относительно v4 (класс MV_KS) — паритет
// подтверждён text_parity на моделях с F32-проекциями; фоллбэк
// INFER_F32_NO_MV=1. Требует K%4==0 и 16-байтного выравнивания A/B.
kernel void gemm_nt_f32_mv(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint n = tgpig.x;
    if (n >= N) return;
    device const float4* a = (device const float4*)A;
    device const float4* b = (device const float4*)(B + (ulong)n * K);
    const uint k4 = K / 4u;
    float acc = 0.0f;
    for (uint k = lane; k < k4; k += 32u) {
        const float4 av = a[k];
        const float4 bv = b[k];
        acc += av.x * bv.x;
        acc += av.y * bv.y;
        acc += av.z * bv.z;
        acc += av.w * bv.w;
    }
    acc = simd_sum(acc);
    if (lane == 0u) C[n] = acc;
}

// ── Quantized GEMM: fp32 activations × fp16/bf16 weights, fp32 accumulate ───
// Same layouts as the f32 kernels; only the B (weight) storage narrows to
// 2 bytes/element. bf16 has no native MSL type on older stacks — the bits
// are passed through and widened with a shift (exact, no rounding).

// C[m,n] = Σ A[m*K+k] · B[n*K+k], B in fp16.
kernel void gemm_nt_f16(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[m * K + k] * (float)B[n * K + k];
    }
    C[m * N + n] = acc;
}
kernel void gemm_nt_bf16(
    device const float* A [[buffer(0)]],
    device const ushort* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[m * K + k] * bf16_to_f32(B, n * K + k);
    }
    C[m * N + n] = acc;
}

// C[m,n] = Σ A[m*K+k] · B[k*N+n], B in fp16.
kernel void gemm_f16(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[m * K + k] * (float)B[k * N + n];
    }
    C[m * N + n] = acc;
}

// C[m,n] = Σ A[m*K+k] · B[k*N+n], B in bf16 (ushort bits).
kernel void gemm_bf16(
    device const float* A [[buffer(0)]],
    device const ushort* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[m * K + k] * bf16_to_f32(B, k * N + n);
    }
    C[m * N + n] = acc;
}

kernel void gemm_nt_q4_0(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 18;
        float d = (float)load_half_unaligned(blk);
        device const uchar* q = blk + 2;
        uint k0 = b * 32;
        // Канонический ggml: байт j — веса j (младший ниббл) и j+16 (старший).
        // Старое чередование (2j, 2j+1) молча читало не те веса на реальных
        // Q4_0 GGUF (эталон dequant_q4_0 канонический с ревью iter-9).
        for (uint j = 0; j < 16; ++j) {
            uchar v = q[j];
            acc += A[m * K + k0 + j] * ((float)(v & 0xF) - 8.0f) * d;
            acc += A[m * K + k0 + j + 16] * ((float)(v >> 4) - 8.0f) * d;
        }
    }
    C[m * N + n] = acc;
}

kernel void gemm_nt_q8_0(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 34;
        float d = (float)load_half_unaligned(blk);
        device const int8_t* q = (device const int8_t*)(blk + 2);
        uint k0 = b * 32;
        for (uint j = 0; j < 32; ++j) {
            acc += A[m * K + k0 + j] * (float)q[j] * d;
        }
    }
    C[m * N + n] = acc;
}

kernel void gemm_nt_q4_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint superblocks = K / 256;
    float acc = 0.0f;
    for (uint sb = 0; sb < superblocks; ++sb) {
        device const uchar* s = B + (n * superblocks + sb) * 144;
        acc += dot_q4_k(A + m * K + sb * 256, s);
    }
    C[m * N + n] = acc;
}

// Q6_K: КАНОНИЧЕСКИЙ ggml block_q6_K (210 Б/256): ql[128] (младшие 4 бита),
// qh[64] (старшие 2 бита), scales[16] (int8 на 16-элементные подблоки),
// d (f16) в КОНЦЕ блока (208..210). y = d·sc·(q − 32).
// Прежний кернел читал кастомный диалект (d в начале, 8 × 8-битных масштабов
// с ниббл-упаковкой) — на реальных GGUF Q6_K-весах Metal декодировал не те
// числа, что CPU-эталон `dequant_q6_k` (тест quant_gemm ловил |Δ| 86..263).
kernel void gemm_nt_q6_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint superblocks = K / 256;
    float acc = 0.0f;
    for (uint sb = 0; sb < superblocks; ++sb) {
        device const uchar* s = B + (n * superblocks + sb) * 210;
        acc += dot_q6_k(A + m * K + sb * 256, s);
    }
    C[m * N + n] = acc;
}

kernel void gemm_nt_q5_0(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* s = B + (n * blocks + b) * 22;
        acc += dot_q5_0(A + m * K + b * 32, s);
    }
    C[m * N + n] = acc;
}

kernel void gemm_nt_q5_k(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint superblocks = K / 256;
    float acc = 0.0f;
    for (uint sb = 0; sb < superblocks; ++sb) {
        device const uchar* s = B + (n * superblocks + sb) * 176;
        acc += dot_q5_k(A + m * K + sb * 256, s);
    }
    C[m * N + n] = acc;
}

// ── Q8_0 batch-GEMM (Ф2.5, prefill m>1) ────────────────────────────────────
// Threadgroup-вариант: BM=8 m-строк × BN=64 n-столбцов на группу; Q8_0-
// блоки загружаются в shared ОДИН раз на 8 строк (bandwidth ×8 против
// поточечного); 512 потоков/группа — 65536 потоков на m=16,n=4096.
// ВАЖНО: размеры shared-массивов — ЛИТЕРАЛЫ (runtime const ломает их).
kernel void gemm_nt_q8_0_batch(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint n0 = gid.x * 64u;
    uint m0 = gid.y * 8u;
    if (n0 >= N) return;
    const uint blocks = K / 32;
    threadgroup float sa[8][32];
    threadgroup float sb[64][32];
    const uint tid = gid.z; // 0..511
    uint i = tid / 64u;     // 0..7 (m-строка в группе)
    uint j = tid % 64u;     // 0..63 (n-столбец в группе)
    float acc = 0.0f;
    for (uint kb = 0; kb < blocks; ++kb) {
        if (tid < 256u) {
            uint ai = tid / 32u, al = tid % 32u;
            uint row = m0 + ai;
            sa[ai][al] = (row < M) ? A[row * K + kb * 32u + al] : 0.0f;
        }
        {
            uint bj = tid % 64u;
            uint col = n0 + bj;
            uint lo = (tid / 64u) * 4u; // 0..28
            if (col < N) {
                device const uchar* blk = B + (col * blocks + kb) * 34;
                float d = (float)load_half_unaligned(blk);
                device const int8_t* q = (device const int8_t*)(blk + 2);
                for (uint t = 0u; t < 4u; ++t) {
                    sb[bj][lo + t] = (float)q[lo + t] * d;
                }
            } else {
                for (uint t = 0u; t < 4u; ++t) { sb[bj][lo + t] = 0.0f; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint l = 0u; l < 32u; ++l) {
            s += sa[i][l] * sb[j][l];
        }
        acc += s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row = m0 + i;
    uint col = n0 + j;
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}



// ── Q8_0 поточечный GEMM, float4 (Ф2.5, prefill m>1) ───────────────────────
// A-чтения float4 (4 FMA на чтение — узкое место было L2-чтения 2:1 к FMA);
// аккумулятор per-block float4 + скалярная сумма по блокам (порядок ближе
// к последовательному). A-буферы свежие (offset 0, MTLBuffer выровнен 16).
kernel void gemm_nt_q8_0_f4(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.x, n = gid.y;
    if (m >= M || n >= N) return;
    const uint blocks = K / 32;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 34;
        float d = (float)load_half_unaligned(blk);
        device const int8_t* q = (device const int8_t*)(blk + 2);
        uint k0 = b * 32;
        device const float* arow = A + m * K + k0;
        float4 acc4 = 0.0f;
        for (uint j4 = 0; j4 < 8; ++j4) {
            float4 a4 = *(device const float4*)(arow + j4 * 4);
            float4 q4;
            q4.x = (float)q[j4 * 4 + 0];
            q4.y = (float)q[j4 * 4 + 1];
            q4.z = (float)q[j4 * 4 + 2];
            q4.w = (float)q[j4 * 4 + 3];
            acc4 += a4 * (q4 * d);
        }
        acc += acc4.x + acc4.y + acc4.z + acc4.w;
    }
    C[m * N + n] = acc;
}

// ── Q8_0 GEMM m>1: n-столбец × 8 m-строк, явные аккумуляторы (Ф2.5) ───────
// Поток (n, m-чанк 8): B-столбец читается 1 раз на 8 строк (было на 1),
// A-строка — 1 чтение на 8 m (av в регистре). acc0..acc7 — ЯВНЫЕ регистры
// (массив acc[8] с runtime-индексом уходит в локальную память!). guard
// для хвостовых m-строк (M не кратно 8).
kernel void gemm_nt_q8_0_m8(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint mb = gid.y * 8u;
    if (n >= N || mb >= M) return;
    const uint blocks = K / 32;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const bool ok0 = mb + 0 < M, ok1 = mb + 1 < M, ok2 = mb + 2 < M, ok3 = mb + 3 < M;
    const bool ok4 = mb + 4 < M, ok5 = mb + 5 < M, ok6 = mb + 6 < M, ok7 = mb + 7 < M;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 34;
        float d = (float)load_half_unaligned(blk);
        device const int8_t* q = (device const int8_t*)(blk + 2);
        uint k0 = b * 32;
        for (uint j = 0; j < 32; ++j) {
            float qv = (float)q[j] * d;
            float a0 = ok0 ? A[(mb + 0) * K + k0 + j] : 0.0f;
            float a1 = ok1 ? A[(mb + 1) * K + k0 + j] : 0.0f;
            float a2 = ok2 ? A[(mb + 2) * K + k0 + j] : 0.0f;
            float a3 = ok3 ? A[(mb + 3) * K + k0 + j] : 0.0f;
            float a4 = ok4 ? A[(mb + 4) * K + k0 + j] : 0.0f;
            float a5 = ok5 ? A[(mb + 5) * K + k0 + j] : 0.0f;
            float a6 = ok6 ? A[(mb + 6) * K + k0 + j] : 0.0f;
            float a7 = ok7 ? A[(mb + 7) * K + k0 + j] : 0.0f;
            acc0 += a0 * qv; acc1 += a1 * qv; acc2 += a2 * qv; acc3 += a3 * qv;
            acc4 += a4 * qv; acc5 += a5 * qv; acc6 += a6 * qv; acc7 += a7 * qv;
        }
    }
    if (ok0) C[(mb + 0) * N + n] = acc0;
    if (ok1) C[(mb + 1) * N + n] = acc1;
    if (ok2) C[(mb + 2) * N + n] = acc2;
    if (ok3) C[(mb + 3) * N + n] = acc3;
    if (ok4) C[(mb + 4) * N + n] = acc4;
    if (ok5) C[(mb + 5) * N + n] = acc5;
    if (ok6) C[(mb + 6) * N + n] = acc6;
    if (ok7) C[(mb + 7) * N + n] = acc7;
}

// ── F16 GEMM m>1: n-столбец × 8 m-строк (Ф2.5, диффузия драфтера) ─────────
// B-столбец (f16) читается 1 раз на 8 строк; half4-чтения; acc0..acc7 —
// явные регистры (массив с runtime-индексом уходит в локальную память).
kernel void gemm_nt_f16_m8(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint mb = gid.y * 8u;
    if (n >= N || mb >= M) return;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const bool ok0 = mb + 0 < M, ok1 = mb + 1 < M, ok2 = mb + 2 < M, ok3 = mb + 3 < M;
    const bool ok4 = mb + 4 < M, ok5 = mb + 5 < M, ok6 = mb + 6 < M, ok7 = mb + 7 < M;
    device const half* brow = B + n * K;
    for (uint k4 = 0; k4 < K / 4; ++k4) {
        half4 b4 = *(device const half4*)(brow + k4 * 4);
        float b0 = (float)b4.x, b1 = (float)b4.y, b2 = (float)b4.z, b3 = (float)b4.w;
        float a0_0 = ok0 ? A[(mb + 0) * K + k4 * 4 + 0] : 0.0f;
        float a0_1 = ok0 ? A[(mb + 0) * K + k4 * 4 + 1] : 0.0f;
        float a0_2 = ok0 ? A[(mb + 0) * K + k4 * 4 + 2] : 0.0f;
        float a0_3 = ok0 ? A[(mb + 0) * K + k4 * 4 + 3] : 0.0f;
        float a1_0 = ok1 ? A[(mb + 1) * K + k4 * 4 + 0] : 0.0f;
        float a1_1 = ok1 ? A[(mb + 1) * K + k4 * 4 + 1] : 0.0f;
        float a1_2 = ok1 ? A[(mb + 1) * K + k4 * 4 + 2] : 0.0f;
        float a1_3 = ok1 ? A[(mb + 1) * K + k4 * 4 + 3] : 0.0f;
        float a2_0 = ok2 ? A[(mb + 2) * K + k4 * 4 + 0] : 0.0f;
        float a2_1 = ok2 ? A[(mb + 2) * K + k4 * 4 + 1] : 0.0f;
        float a2_2 = ok2 ? A[(mb + 2) * K + k4 * 4 + 2] : 0.0f;
        float a2_3 = ok2 ? A[(mb + 2) * K + k4 * 4 + 3] : 0.0f;
        float a3_0 = ok3 ? A[(mb + 3) * K + k4 * 4 + 0] : 0.0f;
        float a3_1 = ok3 ? A[(mb + 3) * K + k4 * 4 + 1] : 0.0f;
        float a3_2 = ok3 ? A[(mb + 3) * K + k4 * 4 + 2] : 0.0f;
        float a3_3 = ok3 ? A[(mb + 3) * K + k4 * 4 + 3] : 0.0f;
        float a4_0 = ok4 ? A[(mb + 4) * K + k4 * 4 + 0] : 0.0f;
        float a4_1 = ok4 ? A[(mb + 4) * K + k4 * 4 + 1] : 0.0f;
        float a4_2 = ok4 ? A[(mb + 4) * K + k4 * 4 + 2] : 0.0f;
        float a4_3 = ok4 ? A[(mb + 4) * K + k4 * 4 + 3] : 0.0f;
        float a5_0 = ok5 ? A[(mb + 5) * K + k4 * 4 + 0] : 0.0f;
        float a5_1 = ok5 ? A[(mb + 5) * K + k4 * 4 + 1] : 0.0f;
        float a5_2 = ok5 ? A[(mb + 5) * K + k4 * 4 + 2] : 0.0f;
        float a5_3 = ok5 ? A[(mb + 5) * K + k4 * 4 + 3] : 0.0f;
        float a6_0 = ok6 ? A[(mb + 6) * K + k4 * 4 + 0] : 0.0f;
        float a6_1 = ok6 ? A[(mb + 6) * K + k4 * 4 + 1] : 0.0f;
        float a6_2 = ok6 ? A[(mb + 6) * K + k4 * 4 + 2] : 0.0f;
        float a6_3 = ok6 ? A[(mb + 6) * K + k4 * 4 + 3] : 0.0f;
        float a7_0 = ok7 ? A[(mb + 7) * K + k4 * 4 + 0] : 0.0f;
        float a7_1 = ok7 ? A[(mb + 7) * K + k4 * 4 + 1] : 0.0f;
        float a7_2 = ok7 ? A[(mb + 7) * K + k4 * 4 + 2] : 0.0f;
        float a7_3 = ok7 ? A[(mb + 7) * K + k4 * 4 + 3] : 0.0f;
        acc0 += a0_0 * b0 + a0_1 * b1 + a0_2 * b2 + a0_3 * b3;
        acc1 += a1_0 * b0 + a1_1 * b1 + a1_2 * b2 + a1_3 * b3;
        acc2 += a2_0 * b0 + a2_1 * b1 + a2_2 * b2 + a2_3 * b3;
        acc3 += a3_0 * b0 + a3_1 * b1 + a3_2 * b2 + a3_3 * b3;
        acc4 += a4_0 * b0 + a4_1 * b1 + a4_2 * b2 + a4_3 * b3;
        acc5 += a5_0 * b0 + a5_1 * b1 + a5_2 * b2 + a5_3 * b3;
        acc6 += a6_0 * b0 + a6_1 * b1 + a6_2 * b2 + a6_3 * b3;
        acc7 += a7_0 * b0 + a7_1 * b1 + a7_2 * b2 + a7_3 * b3;
    }
    if (ok0) C[(mb + 0) * N + n] = acc0;
    if (ok1) C[(mb + 1) * N + n] = acc1;
    if (ok2) C[(mb + 2) * N + n] = acc2;
    if (ok3) C[(mb + 3) * N + n] = acc3;
    if (ok4) C[(mb + 4) * N + n] = acc4;
    if (ok5) C[(mb + 5) * N + n] = acc5;
    if (ok6) C[(mb + 6) * N + n] = acc6;
    if (ok7) C[(mb + 7) * N + n] = acc7;
}

// ── Q8_0 GEMM m>1: m8 + float4 (Ф2.5, w13/w2 доминируют в prefill) ────────
// Поток (n, 8 m-строк): B-столбец 1 раз на 8 строк; float4 по k:
// acc0..acc7 — float4-аккумуляторы (32 регистра), A-чтения float4.
kernel void gemm_nt_q8_0_m8f4(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint mb = gid.y * 8u;
    if (n >= N || mb >= M) return;
    const uint blocks = K / 32;
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float4 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const bool ok0 = mb + 0 < M, ok1 = mb + 1 < M, ok2 = mb + 2 < M, ok3 = mb + 3 < M;
    const bool ok4 = mb + 4 < M, ok5 = mb + 5 < M, ok6 = mb + 6 < M, ok7 = mb + 7 < M;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = B + (n * blocks + b) * 34;
        float d = (float)load_half_unaligned(blk);
        device const int8_t* q = (device const int8_t*)(blk + 2);
        uint k0 = b * 32;
        for (uint j4 = 0; j4 < 8; ++j4) {
            float4 b4;
            b4.x = (float)q[j4 * 4 + 0] * d;
            b4.y = (float)q[j4 * 4 + 1] * d;
            b4.z = (float)q[j4 * 4 + 2] * d;
            b4.w = (float)q[j4 * 4 + 3] * d;
            uint kk = k0 + j4 * 4;
            float4 a0 = ok0 ? *(device const float4*)(A + (mb + 0) * K + kk) : (float4)0.0f;
            float4 a1 = ok1 ? *(device const float4*)(A + (mb + 1) * K + kk) : (float4)0.0f;
            float4 a2 = ok2 ? *(device const float4*)(A + (mb + 2) * K + kk) : (float4)0.0f;
            float4 a3 = ok3 ? *(device const float4*)(A + (mb + 3) * K + kk) : (float4)0.0f;
            float4 a4 = ok4 ? *(device const float4*)(A + (mb + 4) * K + kk) : (float4)0.0f;
            float4 a5 = ok5 ? *(device const float4*)(A + (mb + 5) * K + kk) : (float4)0.0f;
            float4 a6 = ok6 ? *(device const float4*)(A + (mb + 6) * K + kk) : (float4)0.0f;
            float4 a7 = ok7 ? *(device const float4*)(A + (mb + 7) * K + kk) : (float4)0.0f;
            acc0 += a0 * b4; acc1 += a1 * b4; acc2 += a2 * b4; acc3 += a3 * b4;
            acc4 += a4 * b4; acc5 += a5 * b4; acc6 += a6 * b4; acc7 += a7 * b4;
        }
    }
    if (ok0) { float4 v = acc0; C[(mb + 0) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok1) { float4 v = acc1; C[(mb + 1) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok2) { float4 v = acc2; C[(mb + 2) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok3) { float4 v = acc3; C[(mb + 3) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok4) { float4 v = acc4; C[(mb + 4) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok5) { float4 v = acc5; C[(mb + 5) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok6) { float4 v = acc6; C[(mb + 6) * N + n] = v.x + v.y + v.z + v.w; }
    if (ok7) { float4 v = acc7; C[(mb + 7) * N + n] = v.x + v.y + v.z + v.w; }
}
#define QG_BM 32u
#define QG_BN 64u
#define QG_BK 32u
kernel void gemm_nt_q8_0_mma(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    // Активации — f32 (как в ggml mul_mm: веса в half, активации без
    // округления), деквантованные веса — half.
    threadgroup float As[QG_BM * QG_BK];  // [m][k]
    threadgroup half Bs[QG_BN * QG_BK];   // [n][k]
    const uint m0 = tg.y * QG_BM;
    const uint n0 = tg.x * QG_BN;
    const uint nb = K / 32u;
    // simdgroup sg: строки m [ (sg/2)*16, +16 ), столбцы n [ (sg%2)*32, +32 )
    // → фрагменты 2 (m) × 4 (n) по 8×8.
    const uint sm = (sgid / 2u) * 16u;
    const uint sn = (sgid % 2u) * 32u;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = simdgroup_float8x8(0.0f);
    // Загрузка тайлов векторная: поток A — 8 подряд float строки (2×float4),
    // поток B — половина Q8_0-блока строки (масштаб 1 раз, 16 байт 4×char4).
    const uint ar = tid / 4u, ac = (tid % 4u) * 8u;       // A: 32 строки × 4 четверти
    const uint br = tid / 2u, bc = (tid % 2u) * 16u;      // B: 64 строки × 2 половины
    const uint gam = m0 + ar, gbn = n0 + br;
    device const float* arow = A + (ulong)min(gam, M - 1u) * K + ac;
    device const uchar* brow = B + (ulong)min(gbn, N - 1u) * nb * 34u;
    for (uint kb = 0; kb < nb; ++kb) {
        {
            device const packed_float4* ap = (device const packed_float4*)(arow + kb * 32u);
            float4 v0 = float4(ap[0]), v1 = float4(ap[1]);
            if (gam >= M) { v0 = 0.0f; v1 = 0.0f; }
            threadgroup float* dst = As + ar * QG_BK + ac;
            *(threadgroup float4*)(dst) = v0;
            *(threadgroup float4*)(dst + 4) = v1;
        }
        {
            device const uchar* blk = brow + kb * 34u;
            const half d = (gbn < N) ? load_half_unaligned(blk) : 0.0h;
            device const packed_char4* q = (device const packed_char4*)(blk + 2u + bc);
            threadgroup half* dst = Bs + br * QG_BK + bc;
            for (uint c = 0; c < 4; ++c) {
                const char4 v = char4(q[c]);
                *(threadgroup half4*)(dst + c * 4u) = half4(v) * d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < QG_BK; kk += 8u) {
            simdgroup_float8x8 a[2];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 2; ++i)
                simdgroup_load(a[i], As + (sm + i * 8u) * QG_BK + kk, QG_BK);
            // Bs хранится [n][k]; для C = A·Bᵀ нужен фрагмент Bᵀ (k×n) —
            // загрузка с транспонированием.
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(b[j], Bs + (sn + j * 8u) * QG_BK + kk, QG_BK, 0, true);
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Полный тайл — прямо в C; хвостовой — через threadgroup-буфер с
    // проверкой границ.
    if (m0 + QG_BM <= M && n0 + QG_BN <= N) {
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_store(acc[i][j], C + (ulong)(m0 + sm + i * 8u) * N + n0 + sn + j * 8u, N);
        return;
    }
    threadgroup float Cs[QG_BM * QG_BN];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], Cs + (sm + i * 8u) * QG_BN + sn + j * 8u, QG_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < QG_BM * QG_BN; e += 128u) {
        const uint r = e / QG_BN, c = e % QG_BN;
        const uint gm = m0 + r, gn = n0 + c;
        if (gm < M && gn < N) C[(ulong)gm * N + gn] = Cs[e];
    }
}

// ── Q8_0 GEMM m>1 по схеме ggml kernel_mul_mm (Ф8.2b) ────────────────────────
// Тайл 64 (строки весов, N) × 32 (строки активаций, M), K-шаг 32, 128 потоков
// (4 simdgroup'а, каждый — 32×16 выхода = 8 фрагментов 8×8). Разделяемая
// память — блоками 8×8 подряд (веса [k][n], активации [m][k]): simdgroup_load
// со страйдом 8 без конфликтов банков. Активации в half (как у ggml), накопление
// в float. Прежний gemm_nt_q8_0_mma (row-major тайлы 32×32/64×32, f32-
// активации) давал ~5 TFLOPS против ~9 у эталона.
kernel void gemm_nt_q8_0_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q8_0 [N, K]
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    // загрузчики: 2 потока на строку весов (по 16 значений блока),
    // 4 потока на строку активаций (по 8 значений); строки за краем — клэмп
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nb = K / 32u;
    device const uchar* x = B + ((ulong)(r0 + lr0) * nb) * 34u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nb; ++kb) {
        // веса: половина Q8_0-блока (16 значений) → блочная раскладка [k][n]
        {
            device const uchar* blk = x + kb * 34u;
            const float d = (float)load_half_unaligned(blk);
            device const packed_char4* qs = (device const packed_char4*)(blk + 2u);
            float4x4 reg;
            for (short i = 0; i < 4; ++i) reg[i] = float4(char4(qs[4 * il0 + i])) * d;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// ── Q4_0 GEMM m>8 по схеме ggml kernel_mul_mm (префилл Q4_0-моделей) ──
// Тот же тайл 64(N)×32(M)×32(K), те же загрузчик активаций, simdgroup-mma и
// хвостовой staging, что у gemm_nt_q8_0_mm (побайтово совпадает с ggml
// mul_mm); отличается ТОЛЬКО загрузчик весов — у Q4_0 блок 18 Б (f16 d +
// 16 байт нибблов) и каноническая раскладка ggml: значение j — младший
// ниббл байта j, значение j+16 — старший. Два потока на строку весов:
// il0=0 даёт значения 0..15 (младшие нибблы всех 16 байт), il0=1 — 16..31
// (старшие). До этого у Q4_0 mm-тайла не было: m>8 шло скалярным
// gemm_nt_q4_0 (каждый поток читает всю строку весов — префилл 5× медленнее
// Q8_0: 8B-матмулы M=16 442 против 85 мс).
kernel void gemm_nt_q4_0_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q4_0 [N, K]
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    // загрузчики: 2 потока на строку весов (по 16 значений), 4 потока на
    // строку активаций (по 8 значений); строки за краем — клэмп
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nb = K / 32u;
    device const uchar* x = B + ((ulong)(r0 + lr0) * nb) * 18u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nb; ++kb) {
        // веса: 16 значений половины Q4_0-блока → блочная раскладка [k][n]
        {
            device const uchar* blk = x + kb * 18u;
            const float d = (float)load_half_unaligned(blk);
            device const uchar* qs = blk + 2u;
            float4x4 reg;
            for (short i = 0; i < 4; ++i) {
                for (short j = 0; j < 4; ++j) {
                    const uchar b = qs[4 * i + j];
                    const uint nib = il0 == 0 ? (uint)(b & 0x0Fu) : (uint)(b >> 4u);
                    reg[i][j] = ((float)nib - 8.0f) * d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// ── Q4_1 / Q5_1 GEMM m>8 (порт схемы ggml kernel_mul_mm) ─────────────────
// Блок 20/24 Б; il0 выбирает половину 32-шага (низ/верх ниббла), Q5_1
// добавляет 5-й бит из qh (бит i + 16·il0). Значение q·d + m.
template <bool IS_Q5>
kernel void gemm_nt_q4_1_q5_1_mm(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint bb = IS_Q5 ? 24u : 20u;
    const uint nbs = K / 32u;
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * bb;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nbs; ++kb) {
        {
            device const uchar* blk = x + kb * bb;
            const float d = (float)load_half_unaligned(blk);
            const float mn = (float)load_half_unaligned(blk + 2u);
            const uint qh = IS_Q5 ? ((uint)blk[4] | ((uint)blk[5] << 8) | ((uint)blk[6] << 16) | ((uint)blk[7] << 24)) : 0u;
            device const uchar* qs = blk + (IS_Q5 ? 8u : 4u);
            float4x4 reg;
            for (short i = 0; i < 16; ++i) {
                uint q = il0 == 0 ? (uint)(qs[i] & 0x0Fu) : (uint)(qs[i] >> 4u);
                if (IS_Q5) q |= ((qh >> (i + 16u * (uint)il0)) & 1u) << 4;
                reg[i / 4][i % 4] = (float)q * d + mn;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

typedef decltype(gemm_nt_q4_1_q5_1_mm<true>) gemm_nt_q4_1_q5_1_mm_t;
template [[host_name("gemm_nt_q4_1_mm")]] kernel gemm_nt_q4_1_q5_1_mm_t gemm_nt_q4_1_q5_1_mm<false>;
template [[host_name("gemm_nt_q5_1_mm")]] kernel gemm_nt_q4_1_q5_1_mm_t gemm_nt_q4_1_q5_1_mm<true>;

// ── Q8_1 / Q8_K GEMM m>8 ──────────────────────────────────────────────────
// Q8_1: 40 Б/32 (d f16 @0, s f16 @2, int8 qs[32] @4); Q8_K: 292 Б/256
// (f32 d @0, int8 qs[256] @4). Значение — d·q; s/bsums в декванте весов не
// нужны. Общий шаблон: IS_K — Q8_K, иначе Q8_1.
template <bool IS_K>
kernel void gemm_nt_q8_1_q8_k_mm(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint bb = IS_K ? 292u : 36u;
    const uint nbs = K / 32u;
    device const uchar* x = B + (ulong)(r0 + lr0) * (K / (IS_K ? 256u : 32u)) * bb;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nbs; ++kb) {
        {
            // Q8_1: блок на 32 значения; Q8_K: блок на 8 шагов по 32.
            device const uchar* blk = x + (IS_K ? (kb / 8u) * bb : kb * bb);
            const uint inner = IS_K ? (kb % 8u) : 0u;
            const float d = IS_K
                ? as_type<float>(*(device const uint*)(blk))
                : (float)load_half_unaligned(blk);
            device const uchar* qs = blk + (IS_K ? 4u : 4u) + inner * 32u + (uint)il0 * 16u;
            float4x4 reg;
            for (short i = 0; i < 16; ++i) {
                reg[i / 4][i % 4] = (float)(signed char)qs[i] * d;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

typedef decltype(gemm_nt_q8_1_q8_k_mm<true>) gemm_nt_q8_1_q8_k_mm_t;
template [[host_name("gemm_nt_q8_1_mm")]] kernel gemm_nt_q8_1_q8_k_mm_t gemm_nt_q8_1_q8_k_mm<false>;
template [[host_name("gemm_nt_q8_k_mm")]] kernel gemm_nt_q8_1_q8_k_mm_t gemm_nt_q8_1_q8_k_mm<true>;

// ── Q2_K / Q3_K GEMM m>8 (порт схемы ggml kernel_mul_mm) ─────────────────
// Тот же тайл 64(N)×32(M)×32(K) и хвостовой staging, что у q4_0_mm.
// Шаг K=32 = 2 подблока: t = kb%8, s = 2t + il0 (il0 — половина шага),
// shift = 2·(t%4), qs-база = qs0 + (t/4)·32 + il0·16 (16 байт = 16 значений),
// hm-база = il0·16, бит = 1<<(s/2). Q2_K: dl=d·(sc&0xF), ml=dmin·(sc>>4);
// Q3_K: dl=d·(scales[s]−32), значение −4 при нулевом бите hmask.
template <bool IS_Q3>
kernel void gemm_nt_q2_k_q3_k_mm(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint bb = IS_Q3 ? 110u : 84u;
    const uint nbs = K / 32u;
    device const uchar* x = B + (ulong)(r0 + lr0) * (K / 256u) * bb;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nbs; ++kb) {
        {
            device const uchar* blk = x + (kb / 8u) * bb;
            const uint t = kb % 8u;
            const uint s = 2u * t + (uint)il0;
            const float d = (float)load_half_unaligned(blk + (IS_Q3 ? 108u : 80u));
            int sc[16];
            if (IS_Q3) q3k_unpack_scales(blk + 96u, sc);
            float dl, ml;
            if (IS_Q3) {
                dl = d * (float)(sc[s] - 32);
                ml = 0.0f;
            } else {
                const uchar v = blk[s];
                dl = d * (float)(v & 0x0Fu);
                ml = (float)load_half_unaligned(blk + 82u) * (float)(v >> 4u);
            }
            const uint shift = 2u * (t % 4u);
            const uint qs0 = IS_Q3 ? 32u : 16u;
            device const uchar* qb = blk + qs0 + (t / 4u) * 32u + (uint)il0 * 16u;
            const uchar bit = (uchar)(1u << (s / 2u));
            device const uchar* hb = blk + (uint)il0 * 16u;
            float4x4 reg;
            for (short i = 0; i < 16; ++i) {
                int v = (int)((qb[i] >> shift) & 3u);
                if (IS_Q3 && (hb[i] & bit) == 0u) v -= 4;
                reg[i / 4][i % 4] = dl * (float)v - ml;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

typedef decltype(gemm_nt_q2_k_q3_k_mm<true>) gemm_nt_q2_k_q3_k_mm_t;
template [[host_name("gemm_nt_q2_k_mm")]] kernel gemm_nt_q2_k_q3_k_mm_t gemm_nt_q2_k_q3_k_mm<false>;
template [[host_name("gemm_nt_q3_k_mm")]] kernel gemm_nt_q2_k_q3_k_mm_t gemm_nt_q2_k_q3_k_mm<true>;

// ── NVFP4 GEMM m>8 (порт схемы ggml kernel_mul_mm) ───────────────────────
// Блок 36 Б/64: 4 UE4M3-скейла (подблоки по 16) + qs[32]. Шаг тайла по K —
// 32 значения = 2 подблока: kb → блок kb/2, подблок s = 2·(kb%2) + il0
// (il0 — какая 16-половина шага). В подблоке байт j — значения j (низ) и
// j+8 (верх). Остальное (активации, mma, staging) — как у q4_0_mm.
kernel void gemm_nt_nvfp4_mm(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 32u;
    device const uchar* x = B + (ulong)(r0 + lr0) * (K / 64u) * 36u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nbs; ++kb) {
        // веса: 16 значений подблока (il0-половина 32-шага)
        {
            device const uchar* blk = x + (kb / 2u) * 36u;
            const uint sub = 2u * (kb % 2u) + (uint)il0;
            const float d = ue4m3_to_fp32(blk[sub]);
            device const uchar* q = blk + 4u + sub * 8u;
            float4x4 reg;
            for (short i = 0; i < 16; ++i) {
                const uchar v = q[i & 7];
                const uint nib = i < 8 ? (uint)(v & 0x0Fu) : (uint)(v >> 4u);
                reg[i / 4][i % 4] = KV_FP4[nib] * d;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// ── IQ4_NL / MXFP4 GEMM m>8 (порт схемы ggml kernel_mul_mm) ──────────────
// Тот же тайл 64(N)×32(M)×32(K), загрузчик активаций и хвостовой staging,
// что у gemm_nt_q4_0_mm; отличается только загрузчик весов: IQ4_NL — блок
// 18 Б (f16 d + 16 Б нибблов), значение d·KV_IQ4[nibble]; MXFP4 — блок
// 17 Б (E8M0 + 16 Б), значение KV_FP4[nibble]·2^(e−128). Раскладка нибблов
// каноническая (il0=0 — младшие значения 0..15, il0=1 — старшие 16..31).
template <bool IS_NL>
kernel void gemm_nt_iq4_nl_mxfp4_mm(
    device const float* A [[buffer(0)]],
    device const uchar* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 32u;
    const uint blk_bytes = IS_NL ? 18u : 17u;
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * blk_bytes;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint kb = 0; kb < nbs; ++kb) {
        // веса: 16 значений половины блока
        {
            device const uchar* blk = x + kb * blk_bytes;
            const float d = IS_NL ? (float)load_half_unaligned(blk)
                                  : e8m0_to_fp32_half(blk[0]);
            device const uchar* qs = blk + (IS_NL ? 2u : 1u);
            float4x4 reg;
            for (short i = 0; i < 4; ++i) {
                for (short j = 0; j < 4; ++j) {
                    const uchar b = qs[4 * i + j];
                    const uint nib = il0 == 0 ? (uint)(b & 0x0Fu) : (uint)(b >> 4u);
                    reg[i][j] = IS_NL ? d * KV_IQ4[nib] : KV_FP4[nib] * d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

typedef decltype(gemm_nt_iq4_nl_mxfp4_mm<true>) gemm_nt_iq4_nl_mxfp4_mm_t;
template [[host_name("gemm_nt_iq4_nl_mm")]] kernel gemm_nt_iq4_nl_mxfp4_mm_t gemm_nt_iq4_nl_mxfp4_mm<true>;
template [[host_name("gemm_nt_mxfp4_mm")]] kernel gemm_nt_iq4_nl_mxfp4_mm_t gemm_nt_iq4_nl_mxfp4_mm<false>;

// ── IQ4_XS GEMM m>1 по схеме ggml kernel_mul_mm (волна A4: паритет префилла) ──
// Тот же тайл 64(N)×32(M)×32(K) и то же накопление simdgroup_multiply_accumulate,
// что у gemm_nt_q8_0_mm (он побайтово совпадает с ggml mul_mm), — отличается
// только загрузчик весов: QK_NL=256 ⇒ nl=16, два лейна на строку весов (il0)
// проходят 16 подблоков по 16 значений (il = il0, il0+2, …, при обороте x сдвигает-
// ся на следующий 256-суперблок). Деквант — порт dequantize_iq4_xs: dl = d·(ls−32),
// затем dl·kvalues с ОДНИМ округлением в half (как у ggml). Без этого префилл
// (llama.cpp при ne11>8 идёт по mul_mm) расходился с эталоном уже на первом
// токене; matvec (mv_iq4_xs) побайтово верен только при M ≤ 8.
kernel void gemm_nt_iq4_xs_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса IQ4_XS [N, K]
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 256u;                 // число 256-суперблоков
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * 136u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    short il = il0;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint nk = K / 32u;   // шагов по 32
    for (uint kb = 0; kb < nk; ++kb) {
        // деквант 16 значений строки весов (порт dequantize_iq4_xs)
        {
            const uint ib32 = il / 2;          // подблок 0..7
            const uint il2 = il % 2;           // первая/вторая половина
            const uint scales_h = (uint)x[2] | ((uint)x[3] << 8);
            const int ls = (int)((((x[4 + ib32 / 2] >> (4u * (ib32 % 2))) & 0xF)
                                  | (((scales_h >> (2u * ib32)) & 3u) << 4))) - 32;
            const float d = (float)load_half_unaligned(x) * (float)ls;
            device const uint32_t* q4 = (device const uint32_t*)(x + 8) + 4 * ib32;
            float4x4 reg;
            for (short i = 0; i < 4; ++i) {
                const uint aux = (q4[i] >> (4u * il2)) & 0x0f0f0f0f;
                reg[i][0] = d * KV_IQ4[aux & 0xF];
                reg[i][1] = d * KV_IQ4[(aux >> 8) & 0xF];
                reg[i][2] = d * KV_IQ4[(aux >> 16) & 0xF];
                reg[i][3] = d * KV_IQ4[(aux >> 24) & 0xF];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        // продвижение il/x (generic mul_mm): il по подблокам шагом 2, при
        // обороте (il<2) — следующий 256-суперблок
        il = (il + 2 < 16) ? (short)(il + 2) : (short)(il % 2);
        x = (il < 2) ? x + 136u : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// Q5_K-вариант тайлового GEMM (порт kernel_mul_mm_q5_K_f32, тот же sgemm-
// каркас, что у gemm_nt_iq4_xs_mm): скалярный gemm_nt_q5_k на префилле
// (m>8) давал 0.97 TFLOPS против 7 у mm-пути IQ4_XS — gdn.qkv/w_out 22 из
// 23.6 мс GDN-слоя. Деквант — дословный dequantize_q5_K из dequantize.h
// эталона (16 значений на поток, il — тот же generic-ход mul_mm).
kernel void gemm_nt_q5_k_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q5_K [N, K] (176 Б/256)
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 256u;
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * 176u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    short il = il0;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint nk = K / 32u;   // шагов по 32
    for (uint kb = 0; kb < nk; ++kb) {
        // деквант 16 значений строки весов (порт dequantize_q5_K)
        {
            device const uchar* q = x + 48u + 32u * (il / 4) + 16u * (il & 1);
            device const uchar* qh = x + 16u + 16u * (il & 1);
            const uint ul = 1u << (il / 2);
            const uint i3 = il & 3u;
            const uint is = (il / 4u) * 2u;
            device const uchar* scb = x + 4u;
            const uint k2 = i3 / 2u;
            uint sc0, sc1;
            if (is < 4u) {
                sc0 = scb[is + k2] & 63u;
                sc1 = scb[is + 4u + k2] & 63u;
            } else {
                sc0 = (scb[is + 4u + k2] & 0xFu) | ((scb[is - 4u + k2] & 0xC0u) >> 2);
                sc1 = (scb[is + 4u + k2] >> 4) | ((scb[is + k2] & 0xC0u) >> 2);
            }
            const float dh = (float)load_half_unaligned(x);
            const float dmin = (float)load_half_unaligned(x + 2u);
            const float d = i3 < 2u ? dh : dh / 16.f;
            const float dl = d * (float)sc0;
            const float ml = dmin * (float)sc1;
            const uint mask = i3 < 2u ? 0x0Fu : 0xF0u;
            const float qh_val = i3 < 2u ? 16.f : 256.f;
            float4x4 reg;
            for (short i = 0; i < 16; ++i) {
                reg[i / 4][i % 4] =
                    dl * ((float)(q[i] & mask) + ((qh[i] & ul) ? qh_val : 0.f)) - ml;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        il = (il + 2 < 16) ? (short)(il + 2) : (short)(il % 2);
        x = (il < 2) ? x + 176u : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// Q5_0-вариант тайлового GEMM (Ф8.1e, GLM-4-9B: ffn_down в слоях 5,6,8,9,… —
// Q5_0, а оркал при ne00>=64 && ne11>8 берёт kernel_mul_mm без требования
// K%256). Каркас — побитово сверенный gemm_nt_iq4_xs_mm; загрузчик весов —
// ДОСЛОВНЫЙ dequantize_q5_0 из ggml-metal (блок 22 Б: d@0, qh@2, qs@6;
// nl = 32/16 = 2, поток берёт чанк il0 блока kb; рег — float4x4 с ОДНИМ
// округлением в half). Скалярный gemm_nt_q5_0 при m>8 расходится с оркалом
// на ~1e-3 (ffn_out-5 GLM m=12).
struct mm_block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };

kernel void gemm_nt_q5_0_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q5_0 [N, K] (22 Б/32)
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 32u;                  // 32-значных блоков в строке
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * 22u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint nk = K / 32u;
    for (uint kb = 0; kb < nk; ++kb) {
        // деквант 16 значений строки весов — дословный dequantize_q5_0
        {
            device const mm_block_q5_0* xb =
                (device const mm_block_q5_0*)(x + kb * 22u);
            device const uint16_t* qs = ((device const uint16_t*)xb) + 3;
            const float d = xb->d;
            const float md = -16.0h * xb->d;
            const ushort mask = il0 ? (ushort)0x00F0 : (ushort)0x000F;

            const uint32_t qh = *((device const uint32_t*)xb->qh);

            const int x_mv = il0 ? 4 : 0;
            const int gh_mv = il0 ? 12 : 0;
            const int gh_bk = il0 ? 0 : 4;

            float4x4 reg;
            for (short i = 0; i < 8; ++i) {
                const uint8_t xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10;
                const uint8_t xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10;
                const int32_t x0 = ((((uint32_t)qs[i]) & mask) >> x_mv) | xh_0;
                const int32_t x1 = (((((uint32_t)qs[i]) >> 8) & mask) >> x_mv) | xh_1;
                reg[i / 2][2 * (i % 2) + 0] = d * x0 + md;
                reg[i / 2][2 * (i % 2) + 1] = d * x1 + md;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// Q6_K-вариант тайлового GEMM (порт kernel_mul_mm_q6_K_f32, тот же sgemm-
// каркас, что у gemm_nt_q5_k_mm): скалярный gemm_nt_q6_k при m>8 читает веса
// по разу на строку активаций (lm_head 27B [248320,5120]: m=128 — 423 мс в
// бенче). Деквант — дословный dequantize_q6_K из dequantize.h эталона (блок
// 210 Б: ql[128]@0, qh[64]@128, scales[16 int8]@192, d@208; il — тот же
// generic-ход mul_mm по 16-значным группам, x += 210 на обороте).
kernel void gemm_nt_q6_k_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q6_K [N, K] (210 Б/256)
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 256u;
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * 210u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    short il = il0;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint nk = K / 32u;   // шагов по 32
    for (uint kb = 0; kb < nk; ++kb) {
        // деквант 16 значений строки весов (порт dequantize_q6_K)
        {
            const uint qlo = 64u * (il / 8u) + 32u * ((il / 2u) & 1u) + 16u * (il & 1u);
            const uint qho = 128u + 32u * (il / 8u) + 16u * (il & 1u);
            device const ushort* ql = (device const ushort*)(x + qlo);
            device const ushort* qh = (device const ushort*)(x + qho);
            const float sc = (float)(char)x[192u + (il % 2u) + 2u * (il / 2u)];
            const uint i2 = (il / 2u) & 3u;
            const float d_all = (float)load_half_unaligned(x + 208u);
            const uint32_t kmask1 = i2 > 1u ? (i2 > 2u ? 0xC0C0C0C0u : 0x30303030u)
                                            : (i2 > 0u ? 0x0C0C0C0Cu : 0x03030303u);
            const uint32_t kmask2 = i2 > 1u ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
            const float ml = d_all * sc * 32.f;
            const float dl0 = d_all * sc;
            const float dl1 = dl0 / 256.f;
            const float dl2 = dl0 / (256.f * 256.f);
            const float dl3 = dl0 / (256.f * 256.f * 256.f);
            const uint shr_h = i2 > 2u ? 2u : 0u;
            const uint shl_h = i2 > 1u ? 0u : (i2 > 0u ? 2u : 4u);
            const uint shr_l = i2 > 1u ? 4u : 0u;
            float4x4 reg;
            for (short i = 0; i < 4; ++i) {
                const uint32_t lo32 = (uint32_t)ql[2 * i] | ((uint32_t)ql[2 * i + 1] << 16);
                const uint32_t hi32 = (uint32_t)qh[2 * i] | ((uint32_t)qh[2 * i + 1] << 16);
                const uint32_t low = lo32 & kmask2;
                const uint32_t high = hi32 & kmask1;
                const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                reg[i][0] = dl0 * ((float)(half)(q & 0xFFu)) - ml;
                reg[i][1] = dl1 * ((float)(q & 0xFF00u)) - ml;
                reg[i][2] = dl2 * ((float)(q & 0xFF0000u)) - ml;
                reg[i][3] = dl3 * ((float)(q & 0xFF000000u)) - ml;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        il = (il + 2 < 16) ? (short)(il + 2) : (short)(il % 2);
        x = (il < 2) ? x + 210u : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// ── Q4_K × F32 тайловой sgemm (порт kernel_mul_mm_q4_K_f32) ────────────────
// Каркас тот же, что у gemm_nt_q6_k_mm (128 потоков/4 simdgroup'а, тайл
// 64(N)×32(M), шаг K=32, staging в half, simdgroup_multiply_accumulate);
// отличается только деквант — порт dequantize_q4_K + get_scale_min_k4_just2
// из эталонного dequantize.h. Нужен для паритета: при m > 8 эталон считает
// q/k/v-проекции ИМЕННО этим кернелом (use_mm), и его порядок суммирования
// отличается и от matvec'а, и от нашего поэлементного GEMM (на OLMoE —
// разрыв паритета текста p2).
// Точный (дословный) порт ggml dequantize.h: block_q4_K + get_scale_min_k4_just2
// + dequantize_q4_K — чтобы FMA-контракция и порядок операций совпали с
// эталонным кернелом побитово (рукописный порт расходился).
typedef struct {
    half   d;
    half   dmin;
    uchar  scales[12];
    uchar  qs[128];
} q4k_block;

static inline uchar2 q4k_scale_min_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

void q4k_dequantize_mm(device const q4k_block * xb, short il, thread float4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = q4k_scale_min_just2(is, il / 2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

kernel void gemm_nt_q4_k_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const uchar* B [[buffer(1)]],     // веса Q4_K [N, K] (144 Б/256)
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;   // строки весов (N)
    const int r1 = tgpig.x * 32;   // строки активаций (M)
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    const uint nbs = K / 256u;
    device const uchar* x = B + (ulong)(r0 + lr0) * nbs * 144u;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    short il = il0;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint nk = K / 32u;   // шагов по 32
    for (uint kb = 0; kb < nk; ++kb) {
        // деквант 16 значений строки весов — дословный порт dequantize_q4_K
        {
            float4x4 reg;
            q4k_dequantize_mm((device const q4k_block*)x, il, reg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = (half)reg[i / 4][i % 4];
            }
        }
        // активации: 8 значений строки → блочная раскладка [m][k]
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + kb * 32u);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }
        il = (il + 2 < 16) ? (short)(il + 2) : (short)(il % 2);
        x = (il < 2) ? x + 144u : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + 64 <= (int)N && r1 + 32 <= (int)M) {
        device float* Cp = C + (r0 + 32 * (sgitg & 1)) + (ulong)(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], Cp + 8 * (i % 4) + 8 * (ulong)N * (i / 4), N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
        for (short i = 0; i < 8; i++)
            simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tiitg; j < nr1; j += 128) {
            device float* D = C + (ulong)(r1 + j) * N + r0;
            threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
            for (int i = 0; i < nr0; i++) D[i] = Cs[i];
        }
    }
}

// ── F32 × F32 тайловой GEMM (порт kernel_mul_mm_f32_f32) ────────────────────
// Эталон при ne00>=64 && ne11>8 берёт mul_mm и для f32-весов (роутер OLMoE
// gate_inp — dequant_f32, m=префилл). Отличия от квантовых вариантов:
// загрузчик без декванта («no need for dequantization» ветка оркала) с
// проверкой границ loop_k + 16*il + i < K, x — плоские float, шаг 32 на
// итерацию. Порядок суммирования — MMA, не последовательный v4.
kernel void gemm_nt_f32_mm(
    device const float* A [[buffer(0)]],     // активации [M, K]
    device const float* B [[buffer(1)]],     // веса [N, K] f32
    device float* C [[buffer(2)]],           // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)(shmem);          // 64×32 весов
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);   // 32×32 активаций
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;
    const short nr0 = ((int)N - r0 < 64) ? (short)((int)N - r0) : 64;
    const short nr1 = ((int)M - r1 < 32) ? (short)((int)M - r1) : 32;
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short il0 = tiitg % 2;
    device const float* x = B + (ulong)(r0 + lr0) * K;
    const short iy = 8 * (tiitg % 4);
    device const float* y = A + (ulong)(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint nk = K / 32u;
    for (uint kb = 0; kb < nk; ++kb) {
        const uint loop_k = kb * 32u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / 2) / 8;
            const short lx = (tiitg / 2) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                loop_k + 16 * il0 + i < K ? (half)x[loop_k + 16 * il0 + i] : (half)0;
        }
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            device const float4* y4 = (device const float4*)(y + loop_k);
            const float4 v0 = y4[0], v1 = y4[1];
            threadgroup half4* dst = (threadgroup half4*)(sb + 64 * ib + 8 * ly);
            dst[0] = half4(v0);
            dst[1] = half4(v1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* ts = ((threadgroup float*)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
    for (short i = 0; i < 8; i++)
        simdgroup_store(mc[i], ts + 8 * (i % 4) + 8 * 64 * (i / 4), 64, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tiitg; j < nr1; j += 128) {
        device float* D = C + (ulong)(r1 + j) * N + r0;
        threadgroup float* Cs = ((threadgroup float*)shmem) + j * 64;
        for (int i = 0; i < nr0; i++) D[i] = Cs[i];
    }
}
