
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
