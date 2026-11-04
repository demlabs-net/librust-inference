#include <cuda_bf16.h>
// CUDA kernels for the rust-inference CudaBackend (Phase 3).
//
// Compiled to PTX at build time (build.rs, nvcc -arch=compute_86) — one PTX
// binary JITs onto sm_86 (RTX 3090), sm_89 (RTX 4090) and sm_120 (RTX 5090).
// The committed `kernels.ptx` is the source of truth for CI/other hosts.
//
// Determinism: reductions iterate in ascending index order (fixed order —
// the same convention as the CPU reference backend).
//
// NOTE: these are the FIRST-ITERATION kernels (correctness over speed). The
// memory-bound tuning (tiled GEMM, vectorized loads, fused ops, MMA) is the
// Phase 7 work; the op surface is what matters now.

/// Generic elementwise kernel, one thread per element.
/// `op` selects the function (see ElemOp in cuda.rs); `c` is the scalar
/// operand for add_scalar/scale/max_scalar. `y` may alias `out` (in-place).
// Fused SwiGLU gate, row-major per-row gate|up halves — exactly the CPU
// oracle's indexing (cpu.rs silu_mul_fused): with x (rows, 2*ffn) row-major,
// element i of y maps to row i/ffn, col i%ffn and reads
// gate = x[row*2*ffn + col], up = x[row*2*ffn + ffn + col]. The old flat
// y[i] = silu(x[i]) * x[ffn + i] was only correct for row 0 — every row ≥ 1
// read across the row boundary (garbage FFN for all rows > 1).
extern "C" __global__ void silu_mul_fused_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int ffn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int row = i / ffn;
        int col = i % ffn;
        float g = x[row * 2 * ffn + col];
        float up = x[row * 2 * ffn + ffn + col];
        y[i] = (g / (1.0f + expf(-g))) * up;
    }
}

extern "C" __global__ void elementwise_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int n,
    int op,
    float c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i];
    float b = (y != nullptr) ? y[i] : 0.0f;
    float r;
    switch (op) {
        case 0:  r = a + b; break;                    // add
        case 1:  r = a - b; break;                    // sub
        case 2:  r = a * b; break;                    // mul
        case 3:  r = a / b; break;                    // div
        case 4:  r = a + c; break;                    // add_scalar
        case 5:  r = a * c; break;                    // scale
        case 6:  r = sqrtf(a); break;                 // sqrt
        case 7:  r = logf(a); break;                  // log
        case 8:  r = fmaxf(a, c); break;              // max_scalar
        case 9:  r = fmaxf(a, 0.0f); break;           // relu
        case 10: r = a > 0.0f ? a : 0.01f * a; break; // leaky_relu_clip
        case 11: r = 1.0f / (1.0f + expf(-a)); break; // sigmoid
        case 12: r = tanhf(a); break;                 // tanh
        case 13: r = a / (1.0f + expf(-a)); break;    // silu
        case 14: {                                    // gelu_tanh
            float t = tanhf(0.7978845608f * (a + 0.044715f * a * a * a));
            r = 0.5f * a * (1.0f + t);
            break;
        }
        case 15: {                                    // gelu_erf (точный)
            r = 0.5f * a * (1.0f + erff(a * 0.70710678118654752440f));
            break;
        }
        case 16: {                                    // swoosh_l (zipformer icefall):
            // log1p(exp(z)) - 0.08x - 0.035 with z = x-4, stable for z>0.
            float z = a - 4.0f;
            float sp = z > 0.0f ? z + log1pf(expf(-z)) : logf(expf(z) + 1.0f);
            r = sp - 0.08f * a - 0.035f;
            break;
        }
        case 17: {                                    // swoosh_r: x + swoosh_l(x)
            float z = a - 4.0f;
            float sp = z > 0.0f ? z + log1pf(expf(-z)) : logf(expf(z) + 1.0f);
            r = a + sp - 0.08f * a - 0.035f;
            break;
        }
        default: r = a; break;
    }
    out[i] = r;
}

/// y = x + bias[i % cols] — bias broadcast over the last dim.
extern "C" __global__ void add_bias_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int n,
    int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] + bias[i % cols];
}

/// Naive GEMM, one thread per output element. C (M,N) = A (M,K) · B (K,N),
/// row-major. Fixed ascending reduction order (determinism convention).
extern "C" __global__ void gemm_naive_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int k,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;
    int i = idx / n;
    int j = idx % n;
    float acc = 0.0f;
    for (int l = 0; l < k; ++l) {
        acc += a[i * k + l] * b[l * n + j];
    }
    c[idx] = acc;
}

/// C (M,N) = A (M,K) · Bᵀ (N,K) — b is row-major (N,K).
extern "C" __global__ void gemm_nt_naive_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int k,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;
    int i = idx / n;
    int j = idx % n;
    float acc = 0.0f;
    for (int l = 0; l < k; ++l) {
        acc += a[i * k + l] * b[j * k + l];
    }
    c[idx] = acc;
}

// ── norms ────────────────────────────────────────────────────────────────
// One thread per row (rows are small: 128..4096 elems; tuning is Phase 7).

/// y = (x - mean)/sqrt(var + eps) · w + b, row-wise. x (rows, cols).
extern "C" __global__ void layernorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + r * cols;
    float mean = 0.0f;
    for (int j = 0; j < cols; ++j) mean += row[j];
    mean /= cols;
    float var = 0.0f;
    for (int j = 0; j < cols; ++j) { float d = row[j] - mean; var += d * d; }
    var /= cols;
    float inv = rsqrtf(var + eps);
    float* out = y + r * cols;
    for (int j = 0; j < cols; ++j) out[j] = (row[j] - mean) * inv * w[j] + b[j];
}

/// Per-head RMSNorm: x (rows, h·hd), w (hd); each head normalized separately.
// Zipformer BiasNorm (icefall): y[r,j] = x[r,j] * exp(log_scale) /
// sqrt(mean_j (x[r,j] - bias[j])^2), floor 1e-5 on the mean. One thread per
// row: the host oracle sums the row sequentially, and this keeps the CUDA
// result bit-identical (d <= 640, t ~1500 rows — launch-bound, not
// compute-bound).
extern "C" __global__ void bias_norm_zipformer_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float log_scale,
    float* __restrict__ y,
    int t,
    int d)
{
    int row = blockIdx.x;
    if (row >= t) return;
    const float* xr = x + row * d;
    float s = 0.0f;
    for (int j = 0; j < d; ++j) {
        float v = xr[j] - bias[j];
        s += v * v;
    }
    float sc = expf(log_scale) / fmaxf(sqrtf(s / (float)d), 1e-5f);
    float* yr = y + row * d;
    for (int j = 0; j < d; ++j) yr[j] = xr[j] * sc;
}

// Silero VAD complex magnitude по последней оси: X [rows, 2n] (в КАЖДОЙ
// строке сначала re, затем im) → Y [rows, n]:
//   y[r*n + c] = sqrt(x[r*2n + c]^2 + x[r*2n + n + c]^2).
// Прежняя версия складывала половины ВСЕГО тензора (x[i] и x[i+n*rows]) —
// для [1, T, 258] это кадры 0–1 против кадров 2–3 (неверно). PTX в
// репозитории — пересобрать на машине с nvcc; до этого хостовая реализация
// (Backend::magnitude_sqrt по умолчанию) даёт верный результат.
extern "C" __global__ void magnitude_sqrt_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rows) return;
    int r = i / n;
    int c = i % n;
    const float* row = x + (long)r * 2 * n;
    float re = row[c];
    float im = row[n + c];
    y[i] = sqrtf(re * re + im * im);
}

// Silero LSTM cell, fused (plan 7.7): gates = (x·W_ih + b_ih) + (h·W_hh +
// b_hh) with PyTorch i/f/g/o order, then c' = σ(f)·c + σ(i)·tanh(g),
// h' = σ(o)·tanh(c'). One 4·hidden-thread block; the gates land in shared
// memory, the first `hidden` threads do the cell. The host default keeps
// the op-composition (bit-identical to the old path); this kernel only
// changes the summation order (~1 ulp).
extern "C" __global__ void lstm_cell_f32(
    const float* __restrict__ x,
    const float* __restrict__ h,
    const float* __restrict__ c,
    const float* __restrict__ w_ih,
    const float* __restrict__ w_hh,
    const float* __restrict__ b_ih,
    const float* __restrict__ b_hh,
    float* __restrict__ h_new,
    float* __restrict__ c_new,
    int hidden)
{
    extern __shared__ float gates[];
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi < 4 * hidden) {
        const float* row_ih = w_ih + gi * hidden;
        const float* row_hh = w_hh + gi * hidden;
        float acc = b_ih[gi] + b_hh[gi];
        for (int j = 0; j < hidden; ++j) {
            acc += x[j] * row_ih[j] + h[j] * row_hh[j];
        }
        gates[gi] = acc;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        float gi_ = gates[i];
        float gf = gates[hidden + i];
        float gg = gates[2 * hidden + i];
        float go = gates[3 * hidden + i];
        float c2 = (1.0f / (1.0f + expf(-gf))) * c[i]
            + (1.0f / (1.0f + expf(-gi_))) * tanhf(gg);
        float h2 = (1.0f / (1.0f + expf(-go))) * tanhf(c2);
        c_new[i] = c2;
        h_new[i] = h2;
    }
}

// KV compaction shift (plan 1.6): rows [shift..pos) of the (B, cap, D)
// buffer move down to [0..pos-shift) in place. The regions are disjoint by
// construction (shift = pos/2), so the parallel copy is safe.
extern "C" __global__ void kv_shift_f32(
    float* __restrict__ buf,
    int b, int shift, int pos, int d, int cap)
{
    int n = (pos - shift) * d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * n) return;
    int bi = i / n;
    int rem = i % n;
    buf[bi * cap * d + rem] = buf[bi * cap * d + rem + shift * d];
}

extern "C" __global__ void kv_shift_f16(
    __half* __restrict__ buf,
    int b, int shift, int pos, int d, int cap)
{
    int n = (pos - shift) * d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * n) return;
    int bi = i / n;
    int rem = i % n;
    buf[bi * cap * d + rem] = buf[bi * cap * d + rem + shift * d];
}

// q8_0 shift: the i8 data AND the per-32-block f32 scales (d % 32 == 0
// enforced by make_kv_tensor, so shift*d is block-aligned).
extern "C" __global__ void kv_shift_q8(
    signed char* __restrict__ data,
    float* __restrict__ scales,
    int b, int shift, int pos, int d, int cap)
{
    int n = (pos - shift) * d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * n) return;
    int bi = i / n;
    int rem = i % n;
    int base = bi * cap * d;
    data[base + rem] = data[base + rem + shift * d];
    int ns = (pos - shift) * (d / 32);
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < b * ns) {
        int bj = j / ns;
        int rj = j % ns;
        int sbase = bj * cap * (d / 32);
        scales[sbase + rj] = scales[sbase + rj + shift * (d / 32)];
    }
}

// KV re-rotation (plan 1.6): the kept keys were stored rotated for their
// ORIGINAL absolute positions [shift..pos); after the compaction shift they
// must carry the rotations of [0..pos-shift). Every HALVES pair pi of every
// head rotates by -shift * theta^(-2pi/hd). The cos/sin table (hd/2
// entries) is passed in.
// The mrope-text pair filter (mirror of rope_mrope_text_f32): with
// sections the pairs in the "empty" (t) sector carry NO position rotation
// and must not be re-rotated; sect_dims == 0 means the plain rope (all
// pairs rotate).
__device__ inline bool mrope_pair_rotates(int pair, int s0, int s1, int s2, int sect_dims)
{
    if (sect_dims == 0) return true;
    int sector = pair % sect_dims;
    return (sector % 3 == 0 && sector < 3 * s0)
        || (sector % 3 == 1 && sector < 3 * s1)
        || (sector % 3 == 2 && sector < 3 * s2);
}

// The rope pair layout is INTERLEAVED HALVES (mirror of rmsnorm_head_rope
// / rope_mrope_text): pair pi of a head = (k[pi], k[pi + hd/2]) — the
// first hd/2 elements hold the first members, the second half the second.
extern "C" __global__ void kv_rerotate_f32(
    float* __restrict__ k,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int b, int cap, int hd, int nheads, int shift, int pairs_per_head,
    int s0, int s1, int s2, int sect_dims)
{
    int total = b * cap * nheads * pairs_per_head;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int row = i / (nheads * pairs_per_head);
    int rem = i % (nheads * pairs_per_head);
    int hh = rem / pairs_per_head;
    int pi = rem % pairs_per_head;
    if (!mrope_pair_rotates(pi, s0, s1, s2, sect_dims)) return;
    float c = cos_tab[pi];
    float s = sin_tab[pi];
    int hbase = row * (nheads * hd) + hh * hd;
    float x = k[hbase + pi];
    float y = k[hbase + pi + pairs_per_head];
    k[hbase + pi] = x * c - y * s;
    k[hbase + pi + pairs_per_head] = x * s + y * c;
}

extern "C" __global__ void kv_rerotate_f16(
    __half* __restrict__ k,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int b, int cap, int hd, int nheads, int shift, int pairs_per_head,
    int s0, int s1, int s2, int sect_dims)
{
    int total = b * cap * nheads * pairs_per_head;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int row = i / (nheads * pairs_per_head);
    int rem = i % (nheads * pairs_per_head);
    int hh = rem / pairs_per_head;
    int pi = rem % pairs_per_head;
    if (!mrope_pair_rotates(pi, s0, s1, s2, sect_dims)) return;
    float c = cos_tab[pi];
    float s = sin_tab[pi];
    int hbase = row * (nheads * hd) + hh * hd;
    float x = __half2float(k[hbase + pi]);
    float y = __half2float(k[hbase + pi + pairs_per_head]);
    k[hbase + pi] = __float2half_rn(x * c - y * s);
    k[hbase + pi + pairs_per_head] = __float2half_rn(x * s + y * c);
}

// q8_0 re-rotation: one thread per HEAD (the rope pairs cross the 32-block
// halves — the pair pi = (elem pi, elem pi + hd/2)), dequant, rotate,
// re-quantize per 32-block with recomputed scales.
extern "C" __global__ void kv_rerotate_q8(
    signed char* __restrict__ data,
    float* __restrict__ scales,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int b, int cap, int hd, int nheads, int shift,
    int s0, int s1, int s2, int sect_dims)
{
    int blocks_per_row = hd / 32;
    int total = b * cap * nheads;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int row = i / nheads;
    int hh = i % nheads;
    int hbase = row * (nheads * hd) + hh * hd;
    // hd <= 256 for the fixed local array (all LLM head dims are 64/128).
    if (hd > 256) return;
    float tmp[256];
    for (int j = 0; j < hd; ++j) {
        tmp[j] = data[hbase + j]
            * scales[row * (nheads * blocks_per_row) + hh * blocks_per_row + j / 32];
    }
    int pairs = hd / 2;
    for (int pi = 0; pi < pairs; ++pi) {
        if (!mrope_pair_rotates(pi, s0, s1, s2, sect_dims)) continue;
        float c = cos_tab[pi];
        float sn = sin_tab[pi];
        float x = tmp[pi];
        float y = tmp[pi + pairs];
        tmp[pi] = x * c - y * sn;
        tmp[pi + pairs] = x * sn + y * c;
    }
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        float mx = 0.0f;
        for (int j = 0; j < 32; ++j) mx = fmaxf(mx, fabsf(tmp[blk * 32 + j]));
        float ns = mx / 127.0f;
        if (ns < 1e-8f) ns = 1e-8f;
        for (int j = 0; j < 32; ++j) {
            data[hbase + blk * 32 + j] = (signed char)__float2int_rn(tmp[blk * 32 + j] / ns);
        }
        scales[row * (nheads * blocks_per_row) + hh * blocks_per_row + blk] = ns;
    }
}

extern "C" __global__ void rmsnorm_head_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int rows,
    int h,
    int hd,
    float eps)
{
    int r = blockIdx.x;
    int hh = blockIdx.y;
    if (r >= rows || hh >= h) return;
    const float* row = x + r * h * hd + hh * hd;
    float sum = 0.0f;
    for (int j = 0; j < hd; ++j) sum += row[j] * row[j];
    float inv = rsqrtf(sum / hd + eps);
    float* out = y + r * h * hd + hh * hd;
    for (int j = 0; j < hd; ++j) out[j] = row[j] * inv * w[j];
}

/// Per-head LayerNorm: x (rows, h·hd), w/b (hd).
extern "C" __global__ void layernorm_head_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int rows,
    int h,
    int hd,
    float eps)
{
    int r = blockIdx.x;
    int hh = blockIdx.y;
    if (r >= rows || hh >= h) return;
    const float* row = x + r * h * hd + hh * hd;
    float mean = 0.0f;
    for (int j = 0; j < hd; ++j) mean += row[j];
    mean /= hd;
    float var = 0.0f;
    for (int j = 0; j < hd; ++j) { float d = row[j] - mean; var += d * d; }
    var /= hd;
    float inv = rsqrtf(var + eps);
    float* out = y + r * h * hd + hh * hd;
    for (int j = 0; j < hd; ++j) out[j] = (row[j] - mean) * inv * w[j] + b[j];
}

// ── softmax ──────────────────────────────────────────────────────────────
/// Row-wise softmax over the last dim, f32 statistics (f32-compute contract).
extern "C" __global__ void softmax_last_dim_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + r * cols;
    float mx = -INFINITY;
    for (int j = 0; j < cols; ++j) mx = fmaxf(mx, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) sum += expf(row[j] - mx);
    float* out = y + r * cols;
    for (int j = 0; j < cols; ++j) out[j] = expf(row[j] - mx) / sum;
}

// ── slices ───────────────────────────────────────────────────────────────
/// Y (rows, d0) = X (rows, D)[off .. off+d0).
extern "C" __global__ void slice_last_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int d,
    int d0,
    int off)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d0) return;
    int r = i / d0;
    int j = i % d0;
    y[r * d0 + j] = x[r * d + off + j];
}

/// Y (B, T-u, C) = X (B, T, C)[.. T-u]: drop the last u frames.
extern "C" __global__ void slice_time_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int b,
    int t,
    int keep,
    int c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * keep * c) return;
    int bi = i / (keep * c);
    int rem = i % (keep * c);
    y[bi * keep * c + rem] = x[bi * t * c + rem];
}

// ── RoPE ─────────────────────────────────────────────────────────────────
/// Halved RoPE: x (t, hd), pairs (i, i + rope_dim/2), cos/sin (t, rope_dim/2).
extern "C" __global__ void rope_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_buf,
    const float* __restrict__ sin_buf,
    float* __restrict__ y,
    int t,
    int hd,
    int rope_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * hd) return;
    int ti = i / hd;
    int j = i % hd;
    float v = x[i];
    if (j < rope_dim) {
        int half = rope_dim / 2;
        if (j < half) {
            float c = cos_buf[ti * half + j];
            float s = sin_buf[ti * half + j];
            float x1 = x[ti * hd + j + half];
            y[i] = v * c - x1 * s;
        } else {
            int h2 = j - half;
            float c = cos_buf[ti * half + h2];
            float s = sin_buf[ti * half + h2];
            float x0 = x[ti * hd + h2];
            y[i] = x0 * s + v * c;
        }
    } else {
        y[i] = v;
    }
}

/// Interleaved RoPE, per-head (Metal semantics): x (b, t, H·hd), pairs
/// (2i, 2i+1) INSIDE each head; cos/sin [max_seq, hd/2] shared across heads,
/// indexed at absolute (t + start).
extern "C" __global__ void rope_interleaved_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_buf,
    const float* __restrict__ sin_buf,
    float* __restrict__ y,
    int b,
    int t,
    int d,
    int hd,
    int start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * t * d) return;
    int ti = (i / d) % t;
    int dr = i % d; // dim within the whole row
    int hf = hd / 2;
    int pair = (dr % hd) / 2;
    float c = cos_buf[(start + ti) * hf + pair];
    float s = sin_buf[(start + ti) * hf + pair];
    if (dr % 2 == 0) {
        y[i] = x[i] * c - x[i + 1] * s;
    } else {
        y[i] = x[i - 1] * s + x[i] * c;
    }
}

/// Rotate-half RoPE (HF/Qwen3) computed INLINE — no cos/sin tables, no host
/// round-trip. Exact CPU-reference semantics (NEOX halves: pairs (i, i+hd/2),
/// one thread per (t, head, pair); d = h·hd). N (iter-10): the CUDA backend
/// previously ran the trait default = to_cpu_f32 → host sin/cos loop →
/// from_cpu_f32, 2 full syncs × 36 layers per decode token.
extern "C" __global__ void rope_std_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int tq, int d, int hd, int pos, float theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = hd / 2;
    int per_t = (d / hd) * n_pairs;
    int t = i / per_t;
    if (t >= tq) return;
    int rem = i - t * per_t;
    int h = rem / n_pairs;
    int pair = rem - h * n_pairs;
    float p = (float)(pos + t);
    float inv = 1.0f / powf(theta, 2.0f * (float)pair / (float)hd);
    float c = cosf(p * inv);
    float s = sinf(p * inv);
    int base = t * d + h * hd + pair;
    float x0 = x[base];
    float x1 = x[base + n_pairs];
    y[base] = x0 * c - x1 * s;
    y[base + n_pairs] = x1 * c + x0 * s;
}

/// Fused per-head RMSNorm + rotate-half RoPE: Y = rope(rmsnorm_head(X)),
/// one thread per (t, head, pair). Matches the CPU default (rmsnorm_head:
/// y = x·rsqrt(mean(x²)+eps)·w, no mean subtraction) composed with
/// rope_std_f32 — in ONE kernel instead of 3 launches + 2 host round-trips.
extern "C" __global__ void rmsnorm_head_rope_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int tq, int h, int hd, float eps, int pos, float theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = hd / 2;
    int per_t = h * n_pairs;
    int t = i / per_t;
    if (t >= tq) return;
    int rem = i - t * per_t;
    int hh = rem / n_pairs;
    int pair = rem - hh * n_pairs;
    const float* row = x + t * h * hd + hh * hd;
    float sum = 0.0f;
    for (int j = 0; j < hd; ++j) sum += row[j] * row[j];
    float inv = rsqrtf(sum / (float)hd + eps);
    float p = (float)(pos + t);
    float inv_f = 1.0f / powf(theta, 2.0f * (float)pair / (float)hd);
    float c = cosf(p * inv_f);
    float s = sinf(p * inv_f);
    float* out = y + t * h * hd + hh * hd;
    float x0 = row[pair] * inv * w[pair];
    float x1 = row[pair + n_pairs] * inv * w[pair + n_pairs];
    out[pair] = x0 * c - x1 * s;
    out[pair + n_pairs] = x1 * c + x0 * s;
}

/// M-RoPE text (Qwen3-VL IMROPE): halves pairs; the e-section (4th
/// position = 0 for text) is not rotated. Port of the CPU default.
extern "C" __global__ void rope_mrope_text_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int tq, int d, int hd, int pos, float theta,
    int s0, int s1, int s2, int sect_dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = hd / 2;
    int per_t = (d / hd) * n_pairs;
    int t = i / per_t;
    if (t >= tq) return;
    int rem = i - t * per_t;
    int h = rem / n_pairs;
    int pair = rem - h * n_pairs;
    int sector = pair % sect_dims;
    bool is_e = !((sector % 3 == 0 && sector < 3 * s0)
               || (sector % 3 == 1 && sector < 3 * s1)
               || (sector % 3 == 2 && sector < 3 * s2));
    int base = t * d + h * hd + pair;
    if (is_e) {
        y[base] = x[base];
        y[base + n_pairs] = x[base + n_pairs];
        return;
    }
    float p = (float)(pos + t);
    float inv = 1.0f / powf(theta, 2.0f * (float)pair / (float)hd);
    float c = cosf(p * inv);
    float s = sinf(p * inv);
    float x0 = x[base];
    float x1 = x[base + n_pairs];
    y[base] = x0 * c - x1 * s;
    y[base + n_pairs] = x1 * c + x0 * s;
}

/// M-RoPE per-token 4D (Qwen3-VL IMROPE): pos4d [tq·4] = (t, h, w, e) as
/// FLOAT (positions < 2^24 are exact in f32); the per-pair position comes
/// from the category's axis (e = no rotation). Port of the CPU default.
extern "C" __global__ void rope_mrope_f32(
    const float* __restrict__ x,
    const float* __restrict__ pos4d,
    float* __restrict__ y,
    int tq, int d, int hd, float theta,
    int s0, int s1, int s2, int sect_dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = hd / 2;
    int per_t = (d / hd) * n_pairs;
    int t = i / per_t;
    if (t >= tq) return;
    int rem = i - t * per_t;
    int h = rem / n_pairs;
    int pair = rem - h * n_pairs;
    int sector = pair % sect_dims;
    int base = t * d + h * hd + pair;
    float p;
    if (sector % 3 == 0 && sector < 3 * s0) {
        p = pos4d[t * 4];
    } else if (sector % 3 == 1 && sector < 3 * s1) {
        p = pos4d[t * 4 + 1];
    } else if (sector % 3 == 2 && sector < 3 * s2) {
        p = pos4d[t * 4 + 2];
    } else {
        y[base] = x[base];
        y[base + n_pairs] = x[base + n_pairs];
        return;
    }
    float inv = 1.0f / powf(theta, 2.0f * (float)pair / (float)hd);
    float c = cosf(p * inv);
    float s = sinf(p * inv);
    float x0 = x[base];
    float x1 = x[base + n_pairs];
    y[base] = x0 * c - x1 * s;
    y[base + n_pairs] = x1 * c + x0 * s;
}

/// Per-head halved RoPE: x (b, t, h·hd), cos/sin global by t (same rotation
/// for every head, matching the Metal kernel).
extern "C" __global__ void rope_head_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_buf,
    const float* __restrict__ sin_buf,
    float* __restrict__ y,
    int b,
    int t,
    int h,
    int hd,
    int rope_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = h * hd;
    if (i >= b * t * d) return;
    int ti = (i / d) % t;
    int j = i % d;
    int dr = j % hd;
    float v = x[i];
    if (dr < rope_dim) {
        int half = rope_dim / 2;
        if (dr < half) {
            float c = cos_buf[ti * half + dr];
            float s = sin_buf[ti * half + dr];
            float x1 = x[i + half];
            y[i] = v * c - x1 * s;
        } else {
            int h2 = dr - half;
            float c = cos_buf[ti * half + h2];
            float s = sin_buf[ti * half + h2];
            float x0 = x[i - dr + h2]; // element h2 inside the same head
            y[i] = x0 * s + v * c;
        }
    } else {
        y[i] = v;
    }
}

// ── row/vec products, glu, repeat ────────────────────────────────────────
/// y[r, j] = x[r, j] · w[r]
extern "C" __global__ void mul_rows_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int rows,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    y[i] = x[i] * w[i / d];
}

/// y[r, j] = x[r, j] · w[j]
extern "C" __global__ void mul_vec_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int rows,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    y[i] = x[i] * w[i % d];
}

/// glu: x (rows, 2d) → y (rows, d) = a·sigmoid(b); a/b are the two halves
/// of each ROW (x[r][0..d) and x[r][d..2d)).
extern "C" __global__ void glu_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    int r = i / d;
    int c = i % d;
    float a = x[r * 2 * d + c];
    float bb = x[r * 2 * d + d + c];
    y[i] = a / (1.0f + expf(-bb));
}

/// repeat ×2 in time: y[b, 2t + {0,1}, d] = x[b, t, d]
extern "C" __global__ void repeat_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int b,
    int t,
    int t_out,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * t * d) return;
    int bi = i / (t * d);
    int rem = i % (t * d);
    int ti = rem / d;
    int dj = rem % d;
    int o0 = 2 * ti;
    y[(bi * t_out + o0) * d + dj] = x[i];
    if (o0 + 1 < t_out) y[(bi * t_out + o0 + 1) * d + dj] = x[i];
}

/// y = a + scale·b
extern "C" __global__ void add_scaled_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    int n,
    float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + scale * b[i];
}

/// ELEMENTWISE add (the Metal add_bias2d_f32 semantics): y = x + b.
extern "C" __global__ void add_bias2d_f32(
    const float* __restrict__ x,
    const float* __restrict__ b,
    float* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] + b[i];
}

/// snake (DAC): y = x + sin²(a·x)/a — a PER-CHANNEL (Metal semantics):
/// X (B,T,C) flattened to n·c, A (C), Y same layout.
extern "C" __global__ void snake_f32(
    const float* __restrict__ x,
    const float* __restrict__ a,
    float* __restrict__ y,
    int n,
    int c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * c) return;
    int cc = i % c;
    float xv = x[i];
    float av = a[cc];
    // R28: a==0 made sin²(ax)/a NaN — the mathematical limit is x (matches
    // the CPU reference).
    if (av == 0.0f) {
        y[i] = xv;
        return;
    }
    float s = sinf(av * xv);
    y[i] = xv + s * s / av;
}

// ── KV cache, embedding, uniform attention ───────────────────────────────
/// dst (b, cap, d)[pos..pos+tq) = src (b, tq, d)
extern "C" __global__ void kv_update_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int b,
    int tq,
    int pos,
    int d,
    int cap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * tq * d) return;
    int bi = i / (tq * d);
    int rem = i % (tq * d);
    int ti = rem / d;
    int dj = rem % d;
    dst[(bi * cap + pos + ti) * d + dj] = src[i];
}

/// kv_update with f16 storage (KV quantization): src f32 (b, tq, d) →
/// dst half (b, cap, d) at pos. Attention dequantizes inline on read.
extern "C" __global__ void kv_update_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int b,
    int tq,
    int pos,
    int d,
    int cap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * tq * d) return;
    int bi = i / (tq * d);
    int rem = i % (tq * d);
    int ti = rem / d;
    int dj = rem % d;
    dst[(bi * cap + pos + ti) * d + dj] = __float2half_rn(src[i]);
}

/// kv_update copying f16 → f16 (KV grow: the old half cache rows are
/// copied into the grown buffer without precision loss).
extern "C" __global__ void kv_update_f16_f16(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int b,
    int tq,
    int pos,
    int d,
    int cap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * tq * d) return;
    int bi = i / (tq * d);
    int rem = i % (tq * d);
    int ti = rem / d;
    int dj = rem % d;
    dst[(bi * cap + pos + ti) * d + dj] = src[i];
}

/// kv_update copying f16 → f32 (mixed storage during a grow).
extern "C" __global__ void kv_update_f16_f32(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int b,
    int tq,
    int pos,
    int d,
    int cap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * tq * d) return;
    int bi = i / (tq * d);
    int rem = i % (tq * d);
    int ti = rem / d;
    int dj = rem % d;
    dst[(bi * cap + pos + ti) * d + dj] = __half2float(src[i]);
}

/// kv_update with q8_0 storage: src f32 (b, tq, d) → i8 data + per-32-block
/// f32 scales at pos (d must be a multiple of 32). Quantize on write;
/// attention dequantizes inline (kv_read_q8).
extern "C" __global__ void kv_update_q8(
    const float* __restrict__ src,
    signed char* __restrict__ dst,
    float* __restrict__ dsc,
    int b,
    int tq,
    int pos,
    int d,
    int cap)
{
    int nb = d >> 5; // 32-value blocks per row
    int total = b * tq * nb;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int bi = i / (tq * nb);
    int rem = i % (tq * nb);
    int ti = rem / nb;
    int blk = rem % nb;
    const float* s = src + (bi * tq + ti) * d + (blk << 5);
    signed char* dd = dst + (bi * cap + pos + ti) * d + (blk << 5);
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float v = fabsf(s[j]);
        amax = fmaxf(amax, v);
    }
    dsc[(bi * cap + pos + ti) * nb + blk] = amax / 127.0f;
    if (amax == 0.f) {
#pragma unroll
        for (int j = 0; j < 32; ++j) dd[j] = 0;
        return;
    }
    float inv = 127.0f / amax;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        dd[j] = (signed char)__float2int_rn(s[j] * inv);
    }
}

/// kv_update copying q8_0 → q8_0 (KV grow: same format, copy i8 + scales).
extern "C" __global__ void kv_update_q8_q8(
    const signed char* __restrict__ src,
    const float* __restrict__ ssc,
    signed char* __restrict__ dst,
    float* __restrict__ dsc,
    int b,
    int tq,
    int pos,
    int d,
    int cap)
{
    int nb = d >> 5;
    int total = b * tq * d;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int bi = i / (tq * d);
    int rem = i % (tq * d);
    int ti = rem / d;
    int dj = rem % d;
    dst[(bi * cap + pos + ti) * d + dj] = src[(bi * tq + ti) * d + dj];
    // One lane per 32-block copies its scale (same index space).
    if ((dj & 31) == 0) {
        dsc[(bi * cap + pos + ti) * nb + (dj >> 5)] = ssc[(bi * tq + ti) * nb + (dj >> 5)];
    }
}

/// y (len(idx), d) = table (v, d)[idx[i]]
extern "C" __global__ void embedding_f32(
    const unsigned int* __restrict__ idx,
    const float* __restrict__ table,
    float* __restrict__ y,
    int n,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * d) return;
    int r = i / d;
    int j = i % d;
    y[i] = table[idx[r] * d + j];
}

/// Half-precision embedding tables (bf16/f16 — the TTS text/codebook
/// embeddings live in half to halve their VRAM; the OUTPUT stays f32 so
/// the downstream GEMMs are unchanged).
extern "C" __global__ void embedding_bf16(
    const unsigned int* __restrict__ idx,
    const __nv_bfloat16* __restrict__ table,
    float* __restrict__ y,
    int n,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * d) return;
    int r = i / d;
    int j = i % d;
    y[i] = __bfloat162float(table[idx[r] * d + j]);
}

extern "C" __global__ void embedding_f16(
    const unsigned int* __restrict__ idx,
    const __half* __restrict__ table,
    float* __restrict__ y,
    int n,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * d) return;
    int r = i / d;
    int j = i % d;
    y[i] = __half2float(table[idx[r] * d + j]);
}

/// y (b, c) = mean over time of v (b, t, c)
extern "C" __global__ void uniform_attn_f32(
    const float* __restrict__ v,
    float* __restrict__ y,
    int b,
    int t,
    int c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * c) return;
    int bi = i / c;
    int j = i % c;
    float sum = 0.0f;
    for (int ti = 0; ti < t; ++ti) sum += v[(bi * t + ti) * c + j];
    y[i] = sum / t;
}

/// codebook: idx (rows,) = argmin over table rows of L2. One thread per row.
extern "C" __global__ void codebook_search_f32(
    const float* __restrict__ x,
    const float* __restrict__ table,
    unsigned int* __restrict__ idx,
    int rows,
    int v,
    int d)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + r * d;
    float best = INFINITY;
    unsigned int best_i = 0;
    for (int vi = 0; vi < v; ++vi) {
        const float* trow = table + vi * d;
        float acc = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = row[j] - trow[j];
            acc += diff * diff;
        }
        if (acc < best) { best = acc; best_i = vi; }
    }
    idx[r] = best_i;
}

// ── convs ────────────────────────────────────────────────────────────────
/// Dense/grouped 1D conv with left+right padding (matches the Metal kernel):
/// x (b, t, cin), w (cout, cin_g, k); out channel co reads input channels
/// [g·cin_g, (g+1)·cin_g) with g = co / (cout/groups); dense = groups == cin.
/// tout = (t + pad_l + pad_r - dil·(k-1) - 1)/stride + 1.
extern "C" __global__ void conv1d_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int b,
    int t,
    int cin,
    int cout,
    int cin_g,
    int k,
    int stride,
    int pad_l,
    int pad_r,
    int dil,
    int tout,
    int has_bias,
    int groups)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= b * tout * cout) return;
    int bi = i / (tout * cout);
    int rem = i % (tout * cout);
    int to = rem / cout;
    int co = rem % cout;
    float acc = has_bias ? bias[co] : 0.0f;
    int per = cin / groups; // input channels per group (== cin_g)
    int g = co / (cout / groups);
    int ci0 = g * per;
    for (int ci = ci0; ci < ci0 + per; ++ci) {
        for (int kk = 0; kk < k; ++kk) {
            int src = to * stride + kk * dil - pad_l;
            if (src < 0 || src >= t) continue;
            // Weight rows are per output channel (Metal convention): W[co * Cin_g + local_ci].
            acc += x[(bi * t + src) * cin + ci] * w[(co * cin_g + (ci - ci0)) * k + kk];
        }
    }
    y[i] = acc;
}

/// NCHW 2D conv: x (n, ci, h, wdim), w (co, ci, kh, kw); y NCHW
/// (n, co, oh, ow) or time-major (n, oh, co·ow). dil=1; pad on both axes.
extern "C" __global__ void conv2d_nchw_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int n,
    int ci,
    int h,
    int wdim,
    int co,
    int kh,
    int kw,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_f,
    int oh,
    int ow,
    int has_bias,
    int time_major_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * co * oh * ow) return;
    int ni = i / (co * oh * ow);
    int rem = i % (co * oh * ow);
    int coi = rem / (oh * ow);
    int rem2 = rem % (oh * ow);
    int oci = rem2 / ow;
    int owi = rem2 % ow;
    float acc = has_bias ? bias[coi] : 0.0f;
    for (int cii = 0; cii < ci; ++cii) {
        for (int khh = 0; khh < kh; ++khh) {
            for (int kww = 0; kww < kw; ++kww) {
                int hi = oci * stride_t + khh - pad_h;
                int wi = owi * stride_f + kww - pad_w;
                if (hi < 0 || hi >= h || wi < 0 || wi >= wdim) continue;
                acc += x[((ni * ci + cii) * h + hi) * wdim + wi]
                     * w[((coi * ci + cii) * kh + khh) * kw + kww];
            }
        }
    }
    if (time_major_out) {
        // (n, oh, co, ow) — same linear layout as (n, oh, co*ow).
        y[((ni * oh + oci) * co + coi) * ow + owi] = acc;
    } else {
        y[i] = acc;
    }
}

/// MaxPool2d, kernel == stride: x/y (n, c, h, wdim).
extern "C" __global__ void maxpool2d_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int c,
    int h,
    int wdim,
    int kh,
    int kw,
    int oh,
    int ow)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * c * oh * ow) return;
    int ni = i / (c * oh * ow);
    int rem = i % (c * oh * ow);
    int ci = rem / (oh * ow);
    int rem2 = rem % (oh * ow);
    int oi = rem2 / ow;
    int oj = rem2 % ow;
    float best = -INFINITY;
    for (int hi = 0; hi < kh; ++hi) {
        for (int wj = 0; wj < kw; ++wj) {
            float v = x[((ni * c + ci) * h + oi * kh + hi) * wdim + oj * kw + wj];
            best = fmaxf(best, v);
        }
    }
    y[i] = best;
}

/// Transposed 1D conv (stride upsample): w (cin, cout, k), x (b, t, cin),
/// y (b, (t-1)·stride + k, cout) — sparse scatter (matches the Metal kernel).
/// Детерминированная транспонированная свёртка (candle-раскладка W[Cin,Cout,K]):
/// поток на выход, аккумулятор в регистре, фиксированный порядок (k по
/// возрастанию, внутри — ci) — та же схема, что у Metal conv_transpose1d_f32,
/// так что два бэкенда совпадают и побитово. Прежний scatter через atomicAdd
/// давал разные младшие биты на каждом прогоне (TTS-вокодер невоспроизводим,
/// аудит 2026-10-13).
extern "C" __global__ void conv_transpose1d_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    int has_bias,
    float* __restrict__ y,
    int b,
    int t,
    int cin,
    int cout,
    int k,
    int stride,
    int tout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int per = tout * cout;
    if (i >= b * per) return;
    const int bi = i / per;
    const int rem = i % per;
    const int n = rem / cout;
    const int co = rem % cout;
    float acc = has_bias ? bias[co] : 0.0f;
    // k ≡ n (mod stride), k < K, m = (n - k)/stride in [0, T)
    const int k0 = n % stride;
    for (int kk = k0; kk < k; kk += stride) {
        const int m = (n - kk) / stride;
        if (m >= t) continue;
        const float* xrow = x + (bi * t + m) * cin;
        for (int ci = 0; ci < cin; ++ci) {
            acc += xrow[ci] * w[(ci * cout + co) * k + kk];
        }
    }
    y[i] = acc;
}

// ── fused attention ──────────────────────────────────────────────────────
/// f32 → bf16 cast (weights/activations for bf16 GEMMs).
extern "C" __global__ void cast_f32_bf16(
    const float* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16(x[i]);
}

/// f32 → fp16 cast (weights/activations for fp16 GEMMs).
extern "C" __global__ void cast_f32_f16(
    const float* __restrict__ x,
    __half* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half_rn(x[i]);
}

/// Parallel RMSNorm: one 256-thread block per row (block-reduce over cols).
extern "C" __global__ void rmsnorm_v2_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + r * cols;
    __shared__ float red[256];
    float acc = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = row[j];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / cols + eps);
    float* out = y + r * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) out[j] = row[j] * inv * w[j];
}

/// Parallel autoregressive GQA attention: one 256-thread block per
/// (batch, head, query). Scores over the whole KV window are block-reduced
/// (each thread accumulates hd/256 partials), softmax in shared memory, then
/// the weighted V sum is spread over the hd threads. Fixes the 1-thread-per-
/// (b,h,t) kernel that made TTS decode O(tkv·hd) serial per head — the
/// dominant cost of generation (30..186 ms per token at pos 16..273).
/// Same contract as attn_ar_f32: mask is a bias (0 / -inf); tkv <= 4096.
// Type-erased KV readers (each branch is a no-op at runtime unless selected;
// the casts are only type-checked, not executed, in the dead branches).
__device__ inline float kv_read_f32(const float* p, const float* sc, int j) { return p[j]; }
__device__ inline float kv_read_f16(const __half* p, const float* sc, int j) { return __half2float(p[j]); }
__device__ inline float kv_read_q8(const signed char* p, const float* sc, int j) { return (float)p[j] * sc[j >> 5]; }

#define KV_READ(FMT, ptr, sc, j)                                            \
    ((FMT) == 0 ? kv_read_f32((const float*)(ptr), (sc), (j)) :             \
     (FMT) == 1 ? kv_read_f16((const __half*)(ptr), (sc), (j)) :            \
                 kv_read_q8((const signed char*)(ptr), (sc), (j)))

// attn_ar_v4 for all K/V storage pairs: F=0 f32, F=1 f16, F=2 q8_0.
// q/activations/mask stay f32; k/v are read through KV_READ (f16
// dequant inline, q8_0 block-dequant).
//
// Two-kernel warp-chunked attention. The v3 kernel still paid 2
// syncthreads per tk (~300 us at pos 1000 — 36 layers × that reaches the
// 46 ms/token real-time budget at pos ~2000). v4 splits the window into
// CHUNK-sized pieces: one block per (batch, head, query, chunk), each warp
// independently runs the ONLINE softmax over its chunk with warp shuffles
// ONLY — zero block-level syncs in the hot loop — and writes one partial
// (max, sum-exp, weighted-V[32 columns]) per warp. `attn_ar_v4_combine`
// merges the partials. No full-window score array: any tkv ≤ max_seq 32768.
#define ATTN_AR_V4_PARTIAL(NAME, KT, VT, KS, VS)                            \
extern "C" __global__ void NAME(                                           \
    const float* __restrict__ q,                                           \
    const KT* __restrict__ k,                                              \
    const VT* __restrict__ v,                                              \
    const float* __restrict__ mask,                                        \
    float* __restrict__ part,      /* (chunk, warp) slots of 34 f32 */     \
    int b, int tq, int tkv, int n_heads, int kv_heads, int hd, int n_rep,  \
    int has_mask, int nwarps, int nchunks,                                 \
    const float* __restrict__ ksc, const float* __restrict__ vsc)          \
{                                                                          \
    const int CHUNK = 64;                                                  \
    int bi = blockIdx.z / nchunks;                                         \
    int chunk = blockIdx.z % nchunks;                                      \
    int hh = blockIdx.y;                                                   \
    int t = blockIdx.x;                                                    \
    if (bi >= b || hh >= n_heads || t >= tq) return;                       \
    int kvh = hh / n_rep;                                                  \
    int tid = threadIdx.x;                                                 \
    int warp = tid >> 5;                                                   \
    int lane = tid & 31;                                                   \
    /* Q20: warps beyond nwarps must NOT return early — __syncthreads()     \
       below is a BLOCK-wide barrier and a divergent early return is UB     \
       (it can hang the whole pipeline on some hardware). Inactive warps    \
       just contribute nothing: they never write sh_p (their rows are never \
       read) and their partial slot is never written (combine reads only    \
       nwarps slots). */                                                    \
    int active = (warp < nwarps);                                           \
    int c0 = chunk * CHUNK;                                                \
    int cnt = tkv - c0;                                                    \
    if (cnt > CHUNK) cnt = CHUNK;                                          \
    const float* qrow = q + (bi * tq + t) * (n_heads * hd) + hh * hd;      \
    int col = warp * 32 + lane;                                            \
    float scale = rsqrtf((float)hd);                                       \
    /* Pass 1: warp-local partial dots for every tk of the chunk — a       \
       warp's shuffle only sums ITS 32 columns, so the full score needs    \
       all nwarps' partials. Store them to shared, one sync per chunk. */  \
    __shared__ float sh_p[8][64];                                          \
    for (int tk = 0; tk < cnt; ++tk) {                                     \
        float s = 0.0f;                                                    \
        if (active && col < hd) {                                          \
            const KT* krow = k + (bi * tkv + c0 + tk) * (kv_heads * hd) + kvh * hd; \
            const float* kscrow = ksc + ((bi * tkv + c0 + tk) * kv_heads + kvh) * (hd >> 5); \
            s = qrow[col] * KV_READ(KS, krow, kscrow, col);                \
        }                                                                  \
        for (int off = 16; off > 0; off >>= 1) {                           \
            s += __shfl_xor_sync(0xffffffffu, s, off);                     \
        }                                                                  \
        if (lane == 0 && active) sh_p[warp][tk] = s;                       \
    }                                                                      \
    __syncthreads();                                                       \
    /* Pass 2: full score = Σ warp partials, online softmax, warp-local    \
       (all lanes agree on m/sum; each lane owns its column's V sum). */   \
    float m = -INFINITY;                                                   \
    float sum = 0.0f;                                                      \
    float acc = 0.0f;                                                      \
    for (int tk = 0; tk < cnt; ++tk) {                                     \
        float sc = 0.0f;                                                   \
        for (int w = 0; w < nwarps; ++w) sc += sh_p[w][tk];                \
        float score = sc * scale;                                          \
        if (has_mask) score += mask[t * tkv + c0 + tk];                    \
        /* Masked (-inf) positions contribute nothing. Skipping them also  \
           avoids expf(-inf - -inf) = NaN when CONSECUTIVE masked keys     \
           follow an unmasked max (online-softmax rescale degenerates);    \
           the first finite score initializes m. */                        \
        if (score > -INFINITY) {                                           \
            float m_new = fmaxf(m, score);                                 \
            float rescale = __expf(m - m_new);                             \
            sum *= rescale;                                                \
            acc *= rescale;                                                \
            float e = __expf(score - m_new);                               \
            sum += e;                                                      \
            if (col < hd) {                                                \
                const VT* vrow = v + (bi * tkv + c0 + tk) * (kv_heads * hd) + kvh * hd; \
                const float* vscrow = vsc + ((bi * tkv + c0 + tk) * kv_heads + kvh) * (hd >> 5); \
                acc += e * KV_READ(VS, vrow, vscrow, col);                 \
            }                                                              \
            m = m_new;                                                     \
        }                                                                  \
    }                                                                      \
    if (active) {                                                          \
        float* slot = part + ((((bi * tq + t) * n_heads + hh) * nchunks + chunk) * nwarps + warp) * 34; \
        slot[0] = m;                                                       \
        slot[1] = sum;                                                     \
        slot[2 + lane] = acc;                                              \
    }                                                                      \
}

#define ATTN_AR_V4_COMBINE(NAME)                                           \
extern "C" __global__ void NAME(                                           \
    const float* __restrict__ part,      /* (chunk, warp) slots of 34 */   \
    float* __restrict__ y,               /* (b, tq, n_heads·hd) */         \
    int b, int tq, int n_heads, int hd, int nwarps, int nchunks)           \
{                                                                          \
    int bi = blockIdx.z;                                                   \
    int hh = blockIdx.y;                                                   \
    int t = blockIdx.x;                                                    \
    if (bi >= b || hh >= n_heads || t >= tq) return;                       \
    int tid = threadIdx.x;                                                 \
    int base = ((bi * tq + t) * n_heads + hh) * nchunks * nwarps * 34;     \
    int nslot = nchunks * nwarps;                                          \
    /* global max over the partials */                                     \
    float M = -INFINITY;                                                   \
    for (int i = tid; i < nslot; i += 256) M = fmaxf(M, part[base + i * 34]); \
    __shared__ float red[256];                                             \
    red[tid] = M;                                                          \
    __syncthreads();                                                       \
    for (int s = 256 / 2; s > 0; s >>= 1) {                                \
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);             \
        __syncthreads();                                                   \
    }                                                                      \
    M = red[0];                                                            \
    if (tid < hd) {                                                        \
        int col = tid;                                                     \
        int mywarp = col / 32;                                             \
        float l = 0.0f;                                                    \
        float acc = 0.0f;                                                  \
        /* Q25: if EVERY key was masked (or the window is empty), M stays   \
           -inf and __expf(s0[0] - M) = __expf(-inf - -inf) = NaN. Output   \
           zeros instead of NaN. */                                         \
        if (M > -INFINITY) {                                               \
            /* All warps of a chunk see the SAME scores, so their (m, sum) \
               are equal — count each chunk's sum ONCE (warp 0's slot).    \
               The V-sums live per warp: a column takes its acc only from  \
               its own warp's slot. */                                     \
            for (int c = 0; c < nchunks; ++c) {                            \
                const float* s0 = part + base + c * nwarps * 34;           \
                float e = __expf(s0[0] - M);                               \
                l += s0[1] * e;                                            \
                const float* sw = part + base + (c * nwarps + mywarp) * 34; \
                acc += sw[2 + (col & 31)] * e;                             \
            }                                                              \
        }                                                                  \
        y[(bi * tq + t) * (n_heads * hd) + hh * hd + col] = l > 0.0f ? acc / l : 0.0f; \
    }                                                                      \
}

// Name order is {K}{V} — matches the cuda.rs dispatch table
// ((kfmt, vfmt) → kernel name). SUF is the K storage, the suffix is V.
#define ATTN_K(SUF, KT, KS)                                                \
    ATTN_AR_V4_PARTIAL(ATTN_M(attn_ar_v4_, SUF, f32), KT, float, KS, 0)    \
    ATTN_AR_V4_PARTIAL(ATTN_M(attn_ar_v4_, SUF, f16), KT, __half, KS, 1)   \
    ATTN_AR_V4_PARTIAL(ATTN_M(attn_ar_v4_, SUF, q8), KT, signed char, KS, 2)

#define ATTN_M(A, B, C) ATTN_M_I(A, B, C)
#define ATTN_M_I(A, B, C) A##B##C

ATTN_K(f32, float, 0)
ATTN_K(f16, __half, 1)
ATTN_K(q8, signed char, 2)

ATTN_AR_V4_COMBINE(attn_ar_v4_combine)

// ───────── префилл-внимание тайлами (8 запросов × 64 ключа) ─────────
// attn_ar_v4 написан под декод: каждая пара (запрос, голова) перечитывает
// ВЕСЬ KV-кэш из глобальной памяти — на префилле 4.8K это O(K²) трафика
// (~6 с из 9 с TTFT). Здесь блок = 8 запросов ОДНОЙ головы (8 warp'ов),
// K/V-тайл 64 ключей лежит в shared, softmax онлайн в регистрах:
//   score(lane) = Σ_dim q[dim]·Kt[dim][lane]   (q — broadcast из shared)
//   p = exp(s − m), acc[dim] += p·V[key][dim]  (dim = lane + 32e)
// Kt хранится транспонированным с шагом TK+2: и запись при транспозиции,
// и чтение в скалярном произведении идут без банк-конфликтов.
// Динамическая shared: Kt (hd·(TK+2)·2) + Vt (TK·hd·2); Qs/Ps — статика.
#define ATTN_TK 64
#define ATTN_TQ 8
extern "C" __global__ void attn_prefill_tiled_f16(
    const float* __restrict__ q,      // [tq, nh·hd] f32
    const __half* __restrict__ kc,    // [cap, nkv·hd] f16
    const __half* __restrict__ vc,    // [cap, nkv·hd] f16
    const float* __restrict__ mask,   // [tq, tkv] аддитивная или null
    float* __restrict__ y,            // [tq, nh·hd]
    int tq, int tkv, int nh, int nkv, int hd,
    int q_stride, int kv_stride, int has_mask)
{
    extern __shared__ __half smem[];
    __half* Kt = smem;                          // [hd][TK+2]
    __half* Vt = smem + hd * (ATTN_TK + 2);     // [TK][hd]
    __shared__ float Qs[ATTN_TQ * 256];         // строка запроса (broadcast)
    __shared__ float Ps[ATTN_TQ * ATTN_TK];     // вероятности тайла
    const int h = blockIdx.y;
    const int g = h / (nh / nkv);
    const int q0 = blockIdx.x * ATTN_TQ;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int qt = q0 + warp;
    const int nd = hd >> 5;                     // значений на lane (≤ 8)

    if (tid < 0) return;
    for (int i = tid; i < ATTN_TQ * hd; i += blockDim.x) {
        const int w = i / hd, d = i - w * hd;
        const int r = q0 + w;
        Qs[w * 256 + d] = (r < tq) ? q[(size_t)r * q_stride + h * hd + d] : 0.0f;
    }
    __syncthreads();

    float acc[8];
    float m = -INFINITY, l = 0.0f;
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;

    for (int k0 = 0; k0 < tkv; k0 += ATTN_TK) {
        // ── загрузка тайлов: по одному ключу, потоки делят dim ──
        for (int kk = 0; kk < ATTN_TK; ++kk) {
            const int key = k0 + kk;
            for (int d = tid; d < hd; d += blockDim.x) {
                __half kv = __float2half(0.0f), vv = __float2half(0.0f);
                if (key < tkv) {
                    kv = kc[(size_t)key * kv_stride + g * hd + d];
                    vv = vc[(size_t)key * kv_stride + g * hd + d];
                }
                Kt[d * (ATTN_TK + 2) + kk] = kv;
                Vt[kk * hd + d] = vv;
            }
        }
        __syncthreads();
        if (qt < tq) {
            // ── скалярные произведения: lane = ключ (два ключа на lane) ──
            // 4 независимых аккумулятора на ключ: одна цепочка FMA на 4 такта
            // латентности давала 1 FMA/4 такта на поток (ядро упиралось в ILP,
            // а не в полосу shared).
            float s0a = 0.0f, s0b = 0.0f, s0c = 0.0f, s0d = 0.0f;
            float s1a = 0.0f, s1b = 0.0f, s1c = 0.0f, s1d = 0.0f;
            const float* qr = Qs + warp * 256;
            const int lane2 = lane + 32;
            for (int d = 0; d < hd; d += 4) {
                const int d1 = d * (ATTN_TK + 2);
                const int d2 = d1 + (ATTN_TK + 2);
                const int d3 = d2 + (ATTN_TK + 2);
                const int d4 = d3 + (ATTN_TK + 2);
                const float q0 = qr[d], q1 = qr[d + 1], q2 = qr[d + 2], q3 = qr[d + 3];
                s0a += q0 * __half2float(Kt[d1 + lane]);
                s0b += q1 * __half2float(Kt[d2 + lane]);
                s0c += q2 * __half2float(Kt[d3 + lane]);
                s0d += q3 * __half2float(Kt[d4 + lane]);
                s1a += q0 * __half2float(Kt[d1 + lane2]);
                s1b += q1 * __half2float(Kt[d2 + lane2]);
                s1c += q2 * __half2float(Kt[d3 + lane2]);
                s1d += q3 * __half2float(Kt[d4 + lane2]);
            }
            float s0 = (s0a + s0b) + (s0c + s0d);
            float s1 = (s1a + s1b) + (s1c + s1d);
            const int k1 = k0 + lane, k2 = k0 + lane + 32;
            // Масштаб 1/√hd — как у CPU-эталона и ядер attn_ar_v4.
            const float scale = 1.0f / sqrtf((float)hd);
            s0 *= scale;
            s1 *= scale;
            if (k1 >= tkv) s0 = -INFINITY;
            else if (has_mask) s0 += mask[(size_t)qt * tkv + k1];
            if (k2 >= tkv) s1 = -INFINITY;
            else if (has_mask) s1 += mask[(size_t)qt * tkv + k2];
            // ── онлайн-softmax (максимум и сумма — по warp'у) ──
            float mx = fmaxf(s0, s1);
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float m_new = fmaxf(m, mx);
            const float sc = (m > -INFINITY) ? __expf(m - m_new) : 0.0f;
            const float p0 = (s0 > -INFINITY) ? __expf(s0 - m_new) : 0.0f;
            const float p1 = (s1 > -INFINITY) ? __expf(s1 - m_new) : 0.0f;
            float ps = p0 + p1;
            for (int off = 16; off > 0; off >>= 1)
                ps += __shfl_xor_sync(0xffffffff, ps, off);
            Ps[warp * ATTN_TK + lane] = p0;
            Ps[warp * ATTN_TK + lane + 32] = p1;
            l = l * sc + ps;
            m = m_new;
            __syncwarp();
            // ── взвешенная сумма V по своим dim ──
            for (int e = 0; e < nd; ++e) acc[e] *= sc;
            for (int kk = 0; kk < ATTN_TK; ++kk) {
                const float p = Ps[warp * ATTN_TK + kk];
                if (p == 0.0f) continue;
                const __half* vrow = Vt + kk * hd + lane;
                for (int e = 0; e < nd; ++e) acc[e] += p * __half2float(vrow[32 * e]);
            }
            __syncwarp();
        }
        __syncthreads();
    }
    if (tid < 0) return;   // держим все потоки до конца тайла
    if (qt >= tq) return;
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int e = 0; e < nd; ++e)
        y[(size_t)qt * q_stride + h * hd + lane + 32 * e] = acc[e] * inv;
}

// ─────────────────── relative-position attention (NeMo/Zipformer) ───────
// Bidirectional MHSA with relative position scores:
//   ac = (q + u) · kᵀ                          content–content (over dk)
//   bd = (q + v) · pᵀ                          content–position (over dp)
//   mode 0 (GigaAM/NeMo Transformer-XL): bd [T, 2T−1] shifted with the XL
//     pad: shifted[q][k] = bd_pad[(q·(2T−1) + k + T) / (2T)]
//     [(q·(2T−1) + k + T) % (2T)] with bd_pad[r][0] = 0 (P17 — matches
//     gigaam_encoder.py rel_shift: pad(bd,(1,0)) → view(2T,T) → drop
//     row 0 → view(T,2T−1) → [:,:,:,:T]).
//   mode 1 (icefall zipformer): shifted[t][t'] = bd[t][T−1 + t' − t]
//     (no padding, no u/v biases — pass zeros).
// scores = (ac + bd)/√dk; softmax over keys; y = softmax·v (over dk cols).
// When `w0` is non-null the head-0 post-softmax weights [T, T] are written
// there (zipformer nonlin attention).
// One block per (head, query-chunk of Q); scores staged in shared
// (Q·T ≤ 32 KB → Q=4 at T ≤ 2048 — the STT frame cap).
extern "C" __global__ void mhsa_rel_pos_f32(
    const float* __restrict__ q,    // (T, H·dk) content queries
    const float* __restrict__ k,    // (T, H·dk)
    const float* __restrict__ v,    // (T, H·dk)
    const float* __restrict__ qp,   // (T, H·dp) position queries (zipformer p-slice; NeMo: = q)
    const float* __restrict__ p,    // (2T−1, H·dp)
    const float* __restrict__ u,    // (H·dk) pos_bias_u (flattened per head)
    const float* __restrict__ vb,   // (H·dp) pos_bias_v (flattened per head)
    float* __restrict__ y,          // (T, H·dk)
    float* __restrict__ wfull,      // (H, T, T) post-softmax weights or null
    int T,
    int dk,
    int dp,
    int H,
    int mode,
    float scale,
    int has_wfull)                  // O4: wfull pointer may be a dummy —
                                    // never dereference it without this flag
{
    const int Q = 4;
    __shared__ float scs[Q][2048];
    __shared__ float red[256];
    int hh = blockIdx.y;
    int t0 = blockIdx.x * Q;
    int tid = threadIdx.x;
    const float* qh = q + hh * dk;
    const float* kh = k + hh * dk;
    const float* vh = v + hh * dk;
    const float* qph = qp + hh * dp;
    const float* ph = p + hh * dp;
    const float* uh = u + hh * dk;
    const float* vbh = vb + hh * dp;
    // scale passed by the caller: NeMo divides by √dk (its scores use the
    // standard scaled-dot form), zipformer does NOT (the scale is baked into
    // its in_proj weights — icefall RelPositionMultiheadAttentionWeights).
    // ── scores for the chunk's Q query rows ──
    int pairs = Q * T;
    for (int i = tid; i < pairs; i += 256) {
        int ql = i / T;
        int t1 = i % T;
        int qt = t0 + ql;
        if (qt >= T) { scs[ql][t1] = 0.0f; continue; }
        const float* qrow = qh + qt * (H * dk);
        const float* krow = kh + t1 * (H * dk);
        float ac = 0.0f;
        for (int j = 0; j < dk; ++j) {
            ac += (qrow[j] + uh[j]) * krow[j];
        }
        float bd = 0.0f;
        if (mode == 0) {
            // P17: GigaAM rel_shift (gigaam_encoder.py): pad(bd,(1,0)) →
            // view(2T,T) → drop row 0 → view(T,2T−1)[:, :, :T], i.e.
            // pos[q][k] = bd[q][k−q+T−1] with the flat index
            // i = q·(2T−1) + k + T into the padded (2T)-wide matrix.
            // The old code used flat=(t1+1)·T+qt — the transposed index —
            // every (q,k) pair read the wrong p row.
            int flat = qt * (2 * T - 1) + t1 + T;
            int qrow_i = flat / (2 * T);
            int poff = flat % (2 * T);
            if (poff > 0) {
                const float* qr = qh + qrow_i * (H * dk);
                const float* pr = ph + (poff - 1) * (H * dp);
                // min(dk, dp): q rows stride dk, p rows stride dp (see the
                // CPU oracle — the old dp-bound loop read past the q row
                // when dk < dp).
                int jd = dk < dp ? dk : dp;
                for (int j = 0; j < jd; ++j) {
                    bd += (qr[j] + vbh[j]) * pr[j];
                }
            }
        } else {
            int poff = T - 1 + t1 - qt;
            const float* qpr = qph + qt * (H * dp);
            const float* pr = ph + poff * (H * dp);
            for (int j = 0; j < dp; ++j) {
                bd += (qpr[j] + vbh[j]) * pr[j];
            }
        }
        scs[ql][t1] = (ac + bd) * scale;
    }
    __syncthreads();
    // ── per-row softmax + weighted V ──
    for (int ql = 0; ql < Q; ++ql) {
        int qt = t0 + ql;
        if (qt >= T) continue;
        // row max
        float mx = -INFINITY;
        for (int i = tid; i < T; i += 256) mx = fmaxf(mx, scs[ql][i]);
        red[tid] = mx;
        __syncthreads();
        for (int s = 256 / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float row_max = red[0];
        // exp → sum
        float sum = 0.0f;
        for (int i = tid; i < T; i += 256) {
            float e = __expf(scs[ql][i] - row_max);
            scs[ql][i] = e;
            sum += e;
        }
        red[tid] = sum;
        __syncthreads();
        for (int s = 256 / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float inv = 1.0f / red[0];
        // post-softmax weights out (zipformer SelfAttention / NonlinAttention)
        // O4: guarded by an explicit FLAG, not by the pointer: the host used
        // to pass y's address as a dummy "null" — never NULL — so NeMo's
        // wfull=None (mode 0) wrote softmax weights into the OUTPUT buffer y
        // (concurrent blocks race it; OOB past y when T > d).
        if (has_wfull && wfull != NULL) {
            for (int i = tid; i < T; i += 256) {
                wfull[hh * T * T + qt * T + i] = scs[ql][i] * inv;
            }
        }
        // weighted V: thread j owns output column j of this row
        if (tid < dk) {
            float acc = 0.0f;
            const float* vcol = vh + tid;
            const float* srow = scs[ql];
            for (int t1 = 0; t1 < T; ++t1) {
                acc += srow[t1] * vcol[t1 * (H * dk)];
            }
            y[(qt) * (H * dk) + hh * dk + tid] = acc * inv;
        }
        __syncthreads();
    }
}

/// Per-head attention application (zipformer SelfAttention / NonlinAttention):
/// y (T, H·vh) = w (H, T, T) · x (T, H·vh), the post-softmax weights applied
/// over time. One block per head.
extern "C" __global__ void mhsa_apply_weights_f32(
    const float* __restrict__ w,    // (H, T, T) post-softmax weights
    const float* __restrict__ x,    // (T, H·vh)
    float* __restrict__ y,          // (T, H·vh)
    int T,
    int vh,
    int H)
{
    int hh = blockIdx.y;
    int n = T * vh;
    for (int i = threadIdx.x; i < n; i += 256) {
        int t = i / vh;
        int j = i % vh;
        const float* wrow = w + hh * T * T + t * T;
        const float* xcol = x + hh * vh + j;
        float acc = 0.0f;
        for (int t1 = 0; t1 < T; ++t1) {
            acc += wrow[t1] * xcol[t1 * (H * vh)];
        }
        y[t * (H * vh) + hh * vh + j] = acc;
    }
}
/// Autoregressive GQA attention (TTS): q (b, tq, n_heads·hd),
/// k/v (b, tkv, kv_heads·hd), mask (tq, tkv) bias (0 / -inf) or null.
/// One thread per (b, head, tq) — softmax over tkv in f32.
extern "C" __global__ void mhsa_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ sc,
    const float* __restrict__ mask,
    float* __restrict__ y,
    int b,
    int t,
    int h,
    int hd,
    int mode,
    int has_mask)
{
    int bi = blockIdx.x;
    int hh = blockIdx.y;
    int tq = blockIdx.z;
    if (bi >= b || hh >= h || tq >= t) return;
    const float* qrow = q + (bi * t + tq) * (h * hd) + hh * hd;
    const float* vrow0 = v + bi * t * (h * hd) + hh * hd;
    float* scrow = sc + ((bi * h + hh) * t + tq) * t;
    if (mode == 0) {
        float inv = rsqrtf((float)hd);
        for (int tp = 0; tp < t; ++tp) {
            const float* krow = k + (bi * t + tp) * (h * hd) + hh * hd;
            float acc = 0.0f;
            for (int d = 0; d < hd; ++d) acc += qrow[d] * krow[d];
            float m = has_mask ? mask[tq * t + tp] : 0.0f;
            scrow[tp] = acc * inv + m;
        }
    }
    // mode 1: scrow already holds the leader's pre-masked scores.
    float mx = -INFINITY;
    for (int tp = 0; tp < t; ++tp) mx = fmaxf(mx, scrow[tp]);
    float sum = 0.0f;
    for (int tp = 0; tp < t; ++tp) sum += expf(scrow[tp] - mx);
    float* yrow = y + (bi * t + tq) * (h * hd) + hh * hd;
    if (mode == 2) {
        for (int d = 0; d < hd; ++d) {
            float acc = 0.0f;
            for (int tp = 0; tp < t; ++tp) acc += vrow0[tp * (h * hd) + d];
            yrow[d] = acc / t;
        }
        return;
    }
    for (int d = 0; d < hd; ++d) {
        float acc = 0.0f;
        for (int tp = 0; tp < t; ++tp) {
            acc += expf(scrow[tp] - mx) / sum * vrow0[tp * (h * hd) + d];
        }
        yrow[d] = acc;
    }
}

/// Autoregressive GQA attention (TTS): q (b, tq, n_heads·hd),
/// k/v (b, tkv, kv_heads·hd), mask (tq, tkv) bias (0 / -inf) or null.
/// One thread per (b, head, tq) — softmax over tkv in f32.

// ───────────────────────── int8 (Q8_0-style) ─────────────────────────
//
// Symmetric int8 GEMM with per-32-element block scales (llama.cpp Q8_0
// layout), for both weights and activations. 32 elements per scale is ~64x
// finer than per-channel — the precision loss that truncated int8 TTS
// utterances with the old per-channel scheme.
//
// Layouts (all row-major):
//   weights:  W (k, n) i8,  wsc (n, k/32) f32 — wsc[n][kb] = max|W[kb·32.., n]|/127
//   act:      A (m, k) i8,  asc (m, k/32) f32
//   out:      C (m, n) f32
// C[i,j] = Σ_kb asc[i][kb]·wsc[j][kb] · Σ_{k∈kb} A[i,k]·W[k,j]   (i32 per
// 32-block, one fp32 FMA per block — 32 IMADs per dequant).
// k must be a multiple of 32 (all TTS/STT GEMM dims are).

/// Combine the split-k partials: C[r,c] = Σ_s Cpart[(s·m+r)·n + c].
extern "C" __global__ void gemm_i8_reduce(
    const float* __restrict__ Cpart,   // (S·m, n)
    float* __restrict__ C,             // (m, n)
    int n,
    int S,
    int m)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= m * n) return;
    int r = idx / n;
    int c = idx % n;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) acc += Cpart[(s * m + r) * n + c];
    C[r * n + c] = acc;
}

// ─────────────────────── Q6_K GEMM (packed weights) ───────────────────────
// Canonical llama.cpp Q6_K layout, 210 bytes per 256-weight super-block:
//   ql[128]  — low 4 bits (2 weights per byte)
//   qh[64]   — high 2 bits (4 weights per byte)
//   scales[16] — int8 sub-scales (16-elem sub-blocks)
//   d[2]     — f16 super-block scale (bytes 208..210)
// Dequant of weight j (0..255): q = ((ql|qh) & 0x3F) - 32, w = d·sc[j]·q.
// Weights uploaded col-major: block (col, kb) at (col·nsuper + kb)·210 —
// same layout as the quantizer/to_cpu_f32.
__device__ inline float q6k_weight(const unsigned char* blk, int j) {
    int nchunk = j >> 7;          // 0..1 (128-weight halves)
    int lane = j & 31;
    int q = (j >> 5) & 3;         // 0..3 (4 weights share one qh byte)
    int qlb = nchunk * 64 + lane + ((q & 1) << 5);
    int qhb = nchunk * 32 + lane;
    int scb = nchunk * 8 + (lane >> 4) + (q << 1);
    // q1/q2 read the low nibble of their ql byte, q3/q4 the high nibble.
    int low = ((q >> 1) & 1) ? (blk[qlb] >> 4) : (blk[qlb] & 0xF);
    int high = (blk[128 + qhb] >> (q << 1)) & 3;
    int qv = (low | (high << 4)) - 32;
    float d = __half2float(*(const __half*)(blk + 208));
    return d * (float)(signed char)blk[192 + scb] * (float)qv;
}

/// Q5_K canonical llama.cpp layout, 176 bytes per 256-weight super-block:
/// d[2] (f16 super-scale), dmin[2] (f16 min scale), scales[12] (6-bit
/// scale+min per 32-block, get_scale_min_k4), qh[32] (high bits),
/// qs[128] (low 4 bits). Dequant (ggml dequantize_row_q5_K):
///   w = d·sc·q − dmin·m, q = (ql nibble) + (qh bit ? 16 : 0).
__device__ inline float q5k_weight(const unsigned char* blk, int j) {
    int b = j >> 5;          // 0..7 (32-weight blocks)
    int l = j & 31;
    int pair = b >> 1;       // 0..3 (64-weight groups)
    // Per-32-block scales: the odd block of a pair has its OWN scale/min
    // (the encoder packs 8 per-block entries; the CPU decoder reads
    // scale_min_k4(block)). The old `is = pair*2` applied the EVEN block's
    // scale to both halves — blocks 1, 3, 5, 7 dequantized wrong (the
    // q5k fused path only became reachable in wave 6).
    int is = b;
    // get_scale_min_k4(is, scales)
    int sc, m;
    if (is < 4) {
        sc = blk[4 + is] & 63;
        m = blk[4 + is + 4] & 63;
    } else {
        sc = (blk[4 + is + 4] & 0xF) | ((blk[4 + is - 4] >> 6) << 4);
        m = (blk[4 + is + 4] >> 4) | ((blk[4 + is] >> 6) << 4);
    }
    float d = __half2float(*(const __half*)(blk));
    float mn = __half2float(*(const __half*)(blk + 2));
    float dl = d * (float)sc;
    float ml = mn * (float)m;
    int half = b & 1;
    // qs is packed chunk-major (16 + 32·c + l holds values 64c + l in the
    // low nibble and 64c + 32 + l in the high nibble); block pair p maps to
    // chunk p. The old code read blk[16 + l] for every pair — blocks 2..7
    // decoded chunk 0's bytes under their own scales.
    int qsb = 32 * pair;
    int q = half == 0 ? (blk[16 + qsb + l] & 0xF) : (blk[16 + qsb + l] >> 4);
    int u = 1 << (2 * pair + half); // u1/u2 scheme: bit 2·pair (+1 for odd)
    if (blk[144 + l] & u) q += 16;
    return dl * (float)q - ml;
}

/// Canonical Q4_K (GGUF, llama.cpp block_q4_K), 144 bytes per 256-weight
/// super-block: d (f16 @0..2), dmin (f16 @2..4), scales[12] (@4..16 —
/// get_scale_min_k4 6-bit packing of 8 sub-blocks × (scale, min)),
/// qs[128] (@16..144, chunk-major). Dequant (ggml dequantize_row_q4_K):
/// w = d·sc·q − dmin·m, q = 4-bit nibble. N (iter-10): GGUF Q4_K tensors
/// previously ran dequantized to f32 (7x DRAM); this feeds the v2 GEMM.
__device__ inline float q4kc_weight(const unsigned char* blk, int j) {
    int b = j >> 5;   // 0..7 (32-weight sub-blocks)
    int l = j & 31;
    // get_scale_min_k4 (ggml): sub-block b → (scale, min) from the 12
    // packed bytes — verified against the crate's canonical decoder.
    int sc, mn;
    if (b < 4) {
        sc = blk[4 + b] & 63;
        mn = blk[8 + b] & 63;
    } else {
        sc = (blk[4 + b + 4] & 0xF) | ((blk[4 + b - 4] >> 6) << 4);
        mn = (blk[4 + b + 4] >> 4) | ((blk[4 + b] >> 6) << 4);
    }
    // chunk-major qs: byte 16 + 32c + (j&31) holds w[64c + (j&31)] in the
    // low nibble and w[64c + 32 + (j&31)] in the high nibble.
    int c = j >> 6;
    int l32 = j & 31;
    int q = ((j >> 5) & 1) ? (blk[16 + 32 * c + l32] >> 4) : (blk[16 + 32 * c + l32] & 0xF);
    float d = __half2float(*(const __half*)(blk));
    float dmin = __half2float(*(const __half*)(blk + 2));
    return d * (float)sc * (float)q - dmin * (float)mn;
}

/// Q3_K canonical llama.cpp layout, 110 bytes per 256-weight super-block:
/// hmask[32] (non-negative bits), qs[64] (2-bit values), scales[12]
/// (6-bit sub-scales via the aux permutation), d[2] (f16 super-scale).
/// Dequant (ggml dequantize_row_q3_K):
///   w = d·(scs[s]−32)·(q2 − (hm ? 0 : 4)), s = j/16.
__device__ inline float q3k_weight(const unsigned char* blk, int j) {
    int s = j >> 4;
    int l = j & 15;
    // scales[12] → 16 int8 via the ggml aux permutation. R: read BYTE-wise —
    // the old `*(const unsigned int*)(blk + 96)` was misaligned for odd
    // columns (blk = Bs + col·110, 110 ≡ 2 mod 4 → 96+110·col ≡ 2 mod 4);
    // the misaligned shared LDS faulted sporadically with
    // CUDA_ERROR_MISALIGNED_ADDRESS (JIT-dependent — the q3k splitk GEMM
    // failed intermittently while the tiled variant happened to survive).
    unsigned int a0 = (unsigned int)blk[96]
        | ((unsigned int)blk[97] << 8)
        | ((unsigned int)blk[98] << 16)
        | ((unsigned int)blk[99] << 24);
    unsigned int a1 = (unsigned int)blk[100]
        | ((unsigned int)blk[101] << 8)
        | ((unsigned int)blk[102] << 16)
        | ((unsigned int)blk[103] << 24);
    unsigned int tmp = (unsigned int)blk[104]
        | ((unsigned int)blk[105] << 8)
        | ((unsigned int)blk[106] << 16)
        | ((unsigned int)blk[107] << 24);
    const unsigned int k1 = 0x03030303u;
    const unsigned int k2 = 0x0f0f0f0fu;
    unsigned int b0 = (a0 & k2) | (((tmp >> 0) & k1) << 4);
    unsigned int b1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
    unsigned int b2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    unsigned int b3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    int scs = (int)(signed char)((const unsigned char*)(&b0))[s & 3];
    if (s >= 4 && s < 8) scs = (int)(signed char)((const unsigned char*)(&b1))[s & 3];
    if (s >= 8 && s < 12) scs = (int)(signed char)((const unsigned char*)(&b2))[s & 3];
    if (s >= 12) scs = (int)(signed char)((const unsigned char*)(&b3))[s & 3];
    float d = __half2float(*(const __half*)(blk + 108));
    int qbase = (s % 2) * 16 + (s / 8) * 32;
    int shift = ((s / 2) % 4) * 2;
    int hbase = (s % 2) * 16;
    // R (Q3_K): canonical ggml hmask bit is s/2 (8 bits per byte) — the old
    // (s/2)%4 made sub-blocks s and s+8 write the SAME bit, so with mixed
    // signs one sub-block's hmask OR-corrupted the other (decoded as the
    // wrong sign / zero). CPU encode+decode changed in lockstep.
    int mbit = 1 << (s / 2);
    int q2 = (blk[32 + qbase + l] >> shift) & 3;
    int h = (blk[hbase + l] & mbit) ? 0 : 4;
    return d * (float)(scs - 32) * (float)(q2 - h);
}

/// Q4_0 packed-weight dequant, 18 bytes per 32-weight block: f16 d (bytes
/// 0..2) + 16 bytes of 4-bit nibbles. NOTE (iter-10): this is the SPEECH-
/// crate dialect — byte j holds weights (2j, 2j+1) — NOT the canonical ggml
/// pairing (j, j+16) used by the GGUF path. This kernel is fed only by
/// from_cpu_q4k (the speech quantizer's output); GGUF Q4_0 flows dequant→f32
/// and never reaches it. Do NOT "fix" the pairing here without changing the
/// speech quantizer and CudaQ4K::to_cpu together.
__device__ inline float q4k_weight(const unsigned char* blk, int j) {
    float d = __half2float(*(const __half*)(blk));
    int nib = blk[2 + (j >> 1)];
    int q = (j & 1) ? (nib >> 4) : (nib & 0xF);
    return (float)(q - 8) * d;
}

// Q6_K / Q4_0 packed-weight GEMMs. Weights are stored kb-major — block
// (kb, col) at (kb·rows + col)·BLK_BYTES — so one k-block's BN columns are
// ONE contiguous segment; the kernels stage it into shared with vectorized
// (uchar4) cooperative loads (the old per-column scattered byte loads ran
// 70-100x slower than fp16 cuBLAS) and dequantize from shared.
// BN: 128 for Q6_K (26.9 KB tile — 48 KB static-shared limit), 256 for
// Q4_0 (4.6 KB). No activation staging: A is re-read per k-block and
// broadcasts across the block's threads (L2-resident).
#define DEFINE_Q_GEMMS(SUF, WREAD, NSUP_SHIFT, WPB, BLK_BYTES, BN, AT)      \
/* m ≤ 64 (TTS fill/decode): split-k over k-slices. Grid                  \
   (ceil(n/BN)·m, S); each block: one output row × BN columns, per         \
   k-block a cooperative tile load + full WPB-weight dot from shared.      \
   S=1 needs no shared A window (activations read from global) — one       \
   kernel covers every k. `gemm_i8_reduce` combines the partials. The A    \
   operand follows the activation axis (AT = f32/f16/bf16). */             \
extern "C" __global__ void SUF##_splitk(                                    \
    const AT* __restrict__ A,        /* (m, k) */                           \
    const unsigned char* __restrict__ B,/* (k/WPB, n) blocks, kb-major */   \
    float* __restrict__ Cpart,          /* (S·m, n) */                      \
    int k, int n, int S, int m)                                             \
{                                                                           \
    __shared__ unsigned char Bs[BN * BLK_BYTES];                            \
    int tid = threadIdx.x;                                                  \
    int bx = blockIdx.x;                                                    \
    int row = bx % m;                                                       \
    int nb = bx / m;                                                        \
    int col0 = nb * BN;                                                     \
    int ks = k / S;                                                         \
    int nblk_s = ks >> NSUP_SHIFT;                                          \
    int nsup = k >> NSUP_SHIFT;                                             \
    int kb0 = blockIdx.y * nblk_s;                                          \
    const AT* arow = A + (size_t)row * k;                                    \
    float acc = 0.0f;                                                       \
    const int tile_bytes = BN * BLK_BYTES;                                  \
    const int tile_v4 = tile_bytes >> 4;                                    \
    for (int t = 0; t < nblk_s; ++t) {                                      \
        int kb = kb0 + t;                                                   \
        const unsigned char* src = B + (size_t)(kb * n + col0) * BLK_BYTES; \
        /* Fast path: (kb·n + col0)·BLK_BYTES is 16-aligned when n % 8 == 0 \
           (all real TTS/STT n) — uint4 vector loads, fully coalesced.     \
           Otherwise per-column byte loads (never OOB; garbage columns     \
           zeroed — their outputs are masked). */                           \
        if (col0 + BN <= n && (n & 7) == 0) {                               \
            for (int i = tid; i < tile_v4; i += 256) {                      \
                *(uint4*)(Bs + i * 16) = *(const uint4*)(src + i * 16);     \
            }                                                               \
        } else if (tid < BN) {                                              \
            int c = col0 + tid;                                             \
            if (c < n) {                                                    \
                const unsigned char* s = src + (size_t)tid * BLK_BYTES;     \
                unsigned char* d = Bs + tid * BLK_BYTES;                    \
                /* Byte copy: BLK_BYTES is not 16-aligned, the per-column   \
                   source is misaligned — vector loads are not allowed. */  \
                for (int i = 0; i < BLK_BYTES; ++i) d[i] = s[i];            \
            } else {                                                        \
                for (int i = 0; i < BLK_BYTES; ++i) Bs[tid * BLK_BYTES + i] = 0; \
            }                                                               \
        }                                                                   \
        __syncthreads();                                                    \
        if (col0 + tid < n && tid < BN) {                                   \
            const unsigned char* blk = Bs + tid * BLK_BYTES;                \
            /* Modest unroll: full 256-way unrolling of the dequant+FMA     \
               chain spills registers; 8-way keeps the dequant latency      \
               hidden without pressure. */                                  \
            _Pragma("unroll 8")                                             \
            for (int j = 0; j < WPB; ++j) {                                 \
                acc += (float)arow[kb * WPB + j] * WREAD(blk, j);           \
            }                                                               \
        }                                                                   \
        __syncthreads();                                                    \
    }                                                                       \
    if (col0 + tid < n && tid < BN) {                                       \
        Cpart[(blockIdx.y * m + row) * n + col0 + tid] = acc;               \
    }                                                                       \
}                                                                           \
/* m > 64 (STT one-shot frames): block per (32 rows × BN cols), 32 row     \
   accumulators in registers; B staged once per k-block, reused by all     \
   32 rows — B read once per row-chunk, not per row. */                    \
extern "C" __global__ void SUF##_tiled(                                     \
    const AT* __restrict__ A,        /* (m, k) */                           \
    const unsigned char* __restrict__ B,/* (k/WPB, n) blocks, kb-major */   \
    float* __restrict__ Cpart,          /* (S·m, n) — grid.z k-slices */    \
    int m, int k, int n)                                                    \
{                                                                           \
    const int BM = 32;                                                      \
    __shared__ unsigned char Bs[BN * BLK_BYTES];                            \
    int tid = threadIdx.x;                                                  \
    int col0 = blockIdx.x * BN;                                             \
    int r0 = blockIdx.y * BM;                                               \
    int nsup = k >> NSUP_SHIFT;                                             \
    int nkz = (nsup + 15) >> 4;     /* k-split: grid.z=16 multiplies grid */\
    int kb0 = blockIdx.z * nkz;                                             \
    if (kb0 >= nsup) return;                                                \
    int kb1 = nsup < kb0 + nkz ? nsup : kb0 + nkz;                          \
    float acc[BM];                                                          \
    for (int rr = 0; rr < BM; ++rr) acc[rr] = 0.0f;                         \
    const int tile_bytes = BN * BLK_BYTES;                                  \
    const int tile_v4 = tile_bytes >> 4;                                    \
    for (int kb = kb0; kb < kb1; ++kb) {                                    \
        const unsigned char* src = B + (size_t)(kb * n + col0) * BLK_BYTES; \
        if (col0 + BN <= n && (n & 7) == 0) {                               \
            for (int i = tid; i < tile_v4; i += 256) {                      \
                *(uint4*)(Bs + i * 16) = *(const uint4*)(src + i * 16);     \
            }                                                               \
        } else if (tid < BN) {                                              \
            int c = col0 + tid;                                             \
            if (c < n) {                                                    \
                const unsigned char* s = src + (size_t)tid * BLK_BYTES;     \
                unsigned char* d = Bs + tid * BLK_BYTES;                    \
                for (int i = 0; i < BLK_BYTES; ++i) d[i] = s[i];            \
            } else {                                                        \
                for (int i = 0; i < BLK_BYTES; ++i) Bs[tid * BLK_BYTES + i] = 0; \
            }                                                               \
        }                                                                   \
        __syncthreads();                                                    \
        if (col0 + tid < n && tid < BN) {                                   \
            const unsigned char* blk = Bs + tid * BLK_BYTES;                \
            for (int j = 0; j < WPB; ++j) {                                 \
                float w = WREAD(blk, j);                                    \
                const AT* a0 = A + (size_t)r0 * k + kb * WPB + j;           \
                for (int rr = 0; rr < BM; ++rr) {                           \
                    if (r0 + rr < m) acc[rr] += (float)a0[rr * k] * w;      \
                }                                                           \
            }                                                               \
        }                                                                   \
        __syncthreads();                                                    \
    }                                                                       \
    if (col0 + tid >= n || tid >= BN) return;                               \
    for (int rr = 0; rr < BM; ++rr) {                                       \
        if (r0 + rr < m) Cpart[(blockIdx.z * m + r0 + rr) * n + col0 + tid] = acc[rr]; \
    }                                                                       \
}

DEFINE_Q_GEMMS(gemm_q4k, q4k_weight, 5, 32, 18, 256, float)
DEFINE_Q_GEMMS(gemm_q4k_f16, q4k_weight, 5, 32, 18, 256, __half)
DEFINE_Q_GEMMS(gemm_q4k_bf16, q4k_weight, 5, 32, 18, 256, __nv_bfloat16)

// ─────────────────── Q8_0 GEMM splitk (kb-major, configurable A) ────────
// Q8_0 (per-32-block i8 + f32 scale) — the q8_0 m=1 decode was 47-176 us
// (m1 strided reads) or ~55 us (old i8 splitk: activation quantization per
// gemm + byte IMADs) vs 8-12 us cuBLAS bf16. The ACTIVATIONS follow the
// configured dtype (fp32 default; fp16/bf16 cast the A — the plan's
// activation axis). The packed weights are kb-major — block (kb, col) = 32
// contiguous i8 at (kb·n + col)·32 with the per-block scale at (kb·n + col).
// v2 reads DIRECTLY (no shared tile, no syncthreads — the strided 32 B row
// segments share 128 B L1 lines across 4 adjacent k-blocks, so the loads
// stay L1-resident; the shared-tile v1 paid two barriers + a tile store per
// k-block for nothing on the latency-bound m=1 grid).
#define DEFINE_Q8K_SPLITK(SUF, AT)                                           \
extern "C" __global__ void gemm_q8k_splitk_##SUF(                           \
    const AT* __restrict__ A,          /* (m, k) activations */             \
    const signed char* __restrict__ B, /* packed (kb·n + col)·32 i8 */      \
    const float* __restrict__ wsc,     /* (kb·n + col) per-block scales */  \
    float* __restrict__ Cpart,         /* (S·m, n) */                       \
    int k, int n, int S, int m)                                             \
{                                                                           \
    int tid = threadIdx.x;                                                  \
    int bx = blockIdx.x;                                                    \
    int row = bx % m;                                                       \
    int col = (bx / m) * 256 + tid;                                         \
    int ks = k / S;                                                         \
    int kb0 = blockIdx.y * (ks >> 5);                                       \
    float acc = 0.0f;                                                       \
    if (col < n) {                                                          \
        const AT* arow = A + (size_t)row * k;                               \
        _Pragma("unroll 4")                                                 \
        for (int t = 0; t < (ks >> 5); ++t) {                               \
            int kb = kb0 + t;                                               \
            const uint4* bv = (const uint4*)(B + (size_t)(kb * n + col) * 32); \
            float w = wsc[(size_t)kb * n + col];                            \
            const AT* aseg = arow + (size_t)kb * 32;                        \
            float dot = 0.0f;                                               \
            _Pragma("unroll")                                               \
            for (int v = 0; v < 2; ++v) {                                   \
                uint4 ub = __ldg(&bv[v]);                                   \
                const signed char* bh = (const signed char*)&ub;            \
                _Pragma("unroll")                                           \
                for (int u = 0; u < (int)sizeof(AT); ++u) {                 \
                    uint4 ua = __ldg((const uint4*)(aseg + v * 16) + u);    \
                    const AT* ah = (const AT*)&ua;                          \
                    _Pragma("unroll")                                       \
                    for (int j = 0; j < 16 / (int)sizeof(AT); ++j) {        \
                        dot += (float)ah[j] * (float)bh[u * (16 / (int)sizeof(AT)) + j]; \
                    }                                                       \
                }                                                           \
            }                                                               \
            acc += dot * w;                                                 \
        }                                                                   \
        Cpart[(blockIdx.y * m + row) * n + col] = acc;                      \
    }                                                                       \
}

DEFINE_Q8K_SPLITK(f32, float)
DEFINE_Q8K_SPLITK(f16, __half)
DEFINE_Q8K_SPLITK(bf16, __nv_bfloat16)

// ─────────────── Half-weight (fp16/bf16) GEMM splitk ───────────────
// cuBLAS gemm_ex refuses f32-A × half-B (CUBLAS_STATUS_NOT_SUPPORTED) and
// cudarc's safe Gemm forces fp16 output — the half-weight paths (bf16 DAC
// codec, fp16 weights) therefore run this own split-k for every activation
// dtype (the activation axis: fp32 default, fp16/bf16 cast the A rows).
// B is plain (n, k) row-major (no layout change): each thread owns one
// output column and walks its B row in 32-element k-blocks — a row segment
// is 64 B contiguous (2 sectors), and the two k-blocks sharing a 128 B L1
// line make the strided rows L1-resident (no shared tile, no sync). A is
// the same 32-element segment for all 256 threads (L1 broadcast).
#define DEFINE_HALFW_SPLITK(ASUF, BSUF, AT, BT)                               \
extern "C" __global__ void gemm_halfw_splitk_##ASUF##_##BSUF(                 \
    const AT* __restrict__ A,          /* (m, k) row-major activations */     \
    const BT* __restrict__ B,          /* kb-major (kb·n + col)·32 halves */  \
    float* __restrict__ Cpart,         /* (S·m, n) */                         \
    int k, int n, int S, int m)                                               \
{                                                                             \
    int tid = threadIdx.x;                                                    \
    int bx = blockIdx.x;                                                      \
    int row = bx % m;                                                         \
    int col = (bx / m) * 256 + tid;                                           \
    int ks = k / S;                                                           \
    int kb0 = blockIdx.y * (ks >> 5);                                         \
    float acc = 0.0f;                                                         \
    if (col < n) {                                                            \
        const AT* arow = A + (size_t)row * k;                                 \
        _Pragma("unroll 4")                                                   \
        for (int t = 0; t < (ks >> 5); ++t) {                                 \
            int kb = kb0 + t;                                                 \
            const AT* aseg = arow + (size_t)kb * 32;                          \
            const BT* bseg = B + ((size_t)kb * n + col) * 32;                 \
            const uint4* bv = (const uint4*)bseg;                             \
            _Pragma("unroll 4")                                               \
            for (int v = 0; v < 4; ++v) {                                     \
                uint4 ub = __ldg(&bv[v]);                                     \
                const BT* bh = (const BT*)&ub;                                \
                /* A: 8 elements per v — vectorized uint4 loads (broadcast,  \
                   same address for all lanes): 2 uint4 for f32, 1 for half */\
                _Pragma("unroll")                                             \
                for (int u = 0; u < (int)sizeof(AT) / 2; ++u) {               \
                    uint4 ua = __ldg((const uint4*)(aseg + v * 8) + u);       \
                    const AT* ah = (const AT*)&ua;                            \
                    _Pragma("unroll")                                         \
                    for (int j = 0; j < 16 / (int)sizeof(AT); ++j) {          \
                        acc += (float)ah[j] * (float)bh[u * (16 / (int)sizeof(AT)) + j]; \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        Cpart[(blockIdx.y * m + row) * n + col] = acc;                        \
    }                                                                         \
}

DEFINE_HALFW_SPLITK(f32, f16, float, __half)
DEFINE_HALFW_SPLITK(f16, f16, __half, __half)
DEFINE_HALFW_SPLITK(bf16, f16, __nv_bfloat16, __half)
DEFINE_HALFW_SPLITK(f32, bf16, float, __nv_bfloat16)
DEFINE_HALFW_SPLITK(f16, bf16, __half, __nv_bfloat16)
DEFINE_HALFW_SPLITK(bf16, bf16, __nv_bfloat16, __nv_bfloat16)

// ─────────── Half-weight naive GEMM (k % 32 != 0 fallback) ───────────
// The split-k/tiled kernels require k ≡ 0 (mod 32) (kb-major 32-blocks).
// Shapes like the DAC codec's (1, 8, 1024) — raw row-major weights, k<32 —
// used to compute a SILENT ZERO in the half path (ks>>5 = 0 iterations);
// the k%32 fail-fast guard exposed it. This elementary kernel covers any
// k: one thread per output element, row-major B (n, k).
#define DEFINE_HALFW_NAIVE(SUF, BT)                                            \
extern "C" __global__ void gemm_halfw_naive_##SUF(                             \
    const float* __restrict__ A,        /* (m, k) f32 */                      \
    const BT* __restrict__ B,           /* (n, k) row-major raw */            \
    float* __restrict__ C,              /* (m, n) */                          \
    int m, int k, int n)                                                       \
{                                                                              \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= m * n) return;                                                    \
    int r = i / n;                                                             \
    int c = i % n;                                                             \
    const float* a = A + (size_t)r * k;                                        \
    const BT* b = B + (size_t)c * k;                                           \
    float acc = 0.0f;                                                          \
    for (int j = 0; j < k; ++j) acc += a[j] * (float)b[j];                     \
    C[i] = acc;                                                                \
}

DEFINE_HALFW_NAIVE(f16, __half)
DEFINE_HALFW_NAIVE(bf16, __nv_bfloat16)

// ─────────── Half-weight tiled GEMM (m > 1: prefill/fill/STT) ───────────
// The split-k kernels are m=1-shaped (one output row per block); for m>1
// they fall back to S=1 with a long sequential k-loop per row — the prefill
// qkv measured 11x slower than cuBLAS (646 vs 59 us). The tiled kernel is
// the Q6_K tiled structure: one block per (32 rows × 256 cols), the B tile
// staged in shared ONCE per k-block and reused by all 32 rows (B read once
// total), 32 row-accumulators in registers. Both tiles are TRANSPOSED with
// a +1 padding column — consecutive threads touch consecutive banks on
// every access (the unpadded transpose would be 32-way conflicted). A is
// staged in shared per k-block (one coalesced 128 B row per warp); B rows
// are 64 B contiguous segments — the strided rows share 128 B L1 lines
// across adjacent k-blocks (same locality as the split-k direct reads).
#define DEFINE_HALFW_TILED(ASUF, BSUF, AT, BT)                                \
extern "C" __global__ void gemm_halfw_tiled_##ASUF##_##BSUF(                  \
    const AT* __restrict__ A,          /* (m, k) row-major activations */     \
    const BT* __restrict__ B,          /* kb-major (kb·n + col)·32 halves */  \
    float* __restrict__ Cpart,         /* (S·m, n) — grid.z k-slices */        \
    int m, int k, int n)                                                      \
{                                                                             \
    const int BM = 32;                                                        \
    const int BN = 256;                                                       \
    const int BP = BN + 1;                                                    \
    __shared__ BT Bs[32 * BP];          /* T[j][col] — padded transpose */    \
    __shared__ AT As[BM * 33];          /* A-tile: 32 rows × 32 cols + pad */ \
    int tid = threadIdx.x;                                                    \
    int col0 = blockIdx.x * BN;                                               \
    int r0 = blockIdx.y * BM;                                                 \
    int nsup = k >> 5;                                                        \
    int nkz = (nsup + 15) >> 4;     /* k-split: grid.z=16 multiplies grid */  \
    int kb0 = blockIdx.z * nkz;                                               \
    if (kb0 >= nsup) return;                                                  \
    int kb1 = nsup < kb0 + nkz ? nsup : kb0 + nkz;                            \
    float acc[BM];                                                            \
    for (int rr = 0; rr < BM; ++rr) acc[rr] = 0.0f;                           \
    for (int kb = kb0; kb < kb1; ++kb) {                                      \
        if (col0 + tid < n) {                                                 \
            const BT* brow = B + ((size_t)kb * n + col0 + tid) * 32;          \
            const uint4* bv = (const uint4*)brow;                             \
            _Pragma("unroll")                                                 \
            for (int v = 0; v < 4; ++v) {                                     \
                uint4 ub = __ldg(&bv[v]);                                     \
                const BT* bh = (const BT*)&ub;                                \
                _Pragma("unroll")                                             \
                for (int i = 0; i < 8; ++i) {                                 \
                    Bs[(v * 8 + i) * BP + tid] = bh[i];                       \
                }                                                             \
            }                                                                 \
        } else {                                                              \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 32; ++j) Bs[j * BP + tid] = (BT)0;            \
        }                                                                     \
        for (int i = tid; i < BM * 32; i += 256) {                            \
            int rr = i >> 5;                                                  \
            int r = r0 + rr;                                                  \
            As[rr * 33 + (i & 31)] =                                          \
                r < m ? A[(size_t)r * k + kb * 32 + (i & 31)] : (AT)0;        \
        }                                                                     \
        __syncthreads();                                                      \
        if (col0 + tid < n) {                                                 \
            _Pragma("unroll")                                                 \
            for (int rr = 0; rr < BM; ++rr) {                                 \
                float d = 0.0f;                                               \
                _Pragma("unroll")                                             \
                for (int j = 0; j < 32; ++j) {                                \
                    d += (float)As[rr * 33 + j] * (float)Bs[j * BP + tid];    \
                }                                                             \
                acc[rr] += d;                                                 \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (col0 + tid < n) {                                                     \
        for (int rr = 0; rr < BM; ++rr) {                                     \
            if (r0 + rr < m) {                                                \
                Cpart[(blockIdx.z * m + r0 + rr) * n + col0 + tid] = acc[rr]; \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_HALFW_TILED(f32, f16, float, __half)
DEFINE_HALFW_TILED(f16, f16, __half, __half)
DEFINE_HALFW_TILED(bf16, f16, __nv_bfloat16, __half)
DEFINE_HALFW_TILED(f32, bf16, float, __nv_bfloat16)
DEFINE_HALFW_TILED(f16, bf16, __half, __nv_bfloat16)
DEFINE_HALFW_TILED(bf16, bf16, __nv_bfloat16, __nv_bfloat16)

// ─────────────── Q8_0 tiled GEMM (m > 1), kb-major B ───────────────
// Same structure as the half-weight tiled: block per (32 rows × 256 cols),
// B staged in shared once per k-block, 32 row-accumulators. The packed
// weights are kb-major (block (kb, col) = 32 contiguous i8 at
// (kb·n + col)·32) — thread t loads its 32 B row (2 uint4, L1-resident
// across adjacent k-blocks) and the per-block scale wsc(kb, col) directly
// into a register. The scale is factored out of the j-loop (dot * w).
#define DEFINE_Q8K_TILED(SUF, AT)                                             \
extern "C" __global__ void gemm_q8k_tiled_##SUF(                              \
    const AT* __restrict__ A,          /* (m, k) row-major activations */     \
    const signed char* __restrict__ B, /* packed (kb·n + col)·32 i8 */        \
    const float* __restrict__ wsc,     /* (kb·n + col) per-block scales */    \
    float* __restrict__ Cpart,         /* (S·m, n) — grid.z k-slices */        \
    int m, int k, int n)                                                      \
{                                                                             \
    const int BM = 32;                                                        \
    const int BN = 256;                                                       \
    const int BP = BN + 1;                                                    \
    __shared__ signed char Bs[32 * BP];                                       \
    __shared__ AT As[BM * 33];                                                \
    int tid = threadIdx.x;                                                    \
    int col0 = blockIdx.x * BN;                                               \
    int r0 = blockIdx.y * BM;                                                 \
    int nsup = k >> 5;                                                        \
    int nkz = (nsup + 15) >> 4;     /* k-split: grid.z=16 multiplies grid */  \
    int kb0 = blockIdx.z * nkz;                                               \
    if (kb0 >= nsup) return;                                                  \
    int kb1 = nsup < kb0 + nkz ? nsup : kb0 + nkz;                            \
    float acc[BM];                                                            \
    for (int rr = 0; rr < BM; ++rr) acc[rr] = 0.0f;                           \
    for (int kb = kb0; kb < kb1; ++kb) {                                      \
        float w = 0.0f;                                                       \
        if (col0 + tid < n) {                                                 \
            const signed char* brow = B + (size_t)(kb * n + col0 + tid) * 32; \
            const uint4* bv = (const uint4*)brow;                             \
            _Pragma("unroll")                                                 \
            for (int v = 0; v < 2; ++v) {                                     \
                uint4 ub = __ldg(&bv[v]);                                     \
                const signed char* bh = (const signed char*)&ub;              \
                _Pragma("unroll")                                             \
                for (int i = 0; i < 16; ++i) {                                \
                    Bs[(v * 16 + i) * BP + tid] = bh[i];                      \
                }                                                             \
            }                                                                 \
            w = wsc[(size_t)kb * n + col0 + tid];                             \
        } else {                                                              \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 32; ++j) Bs[j * BP + tid] = 0;                \
        }                                                                     \
        for (int i = tid; i < BM * 32; i += 256) {                            \
            int rr = i >> 5;                                                  \
            int r = r0 + rr;                                                  \
            As[rr * 33 + (i & 31)] =                                          \
                r < m ? A[(size_t)r * k + kb * 32 + (i & 31)] : (AT)0;        \
        }                                                                     \
        __syncthreads();                                                      \
        if (col0 + tid < n) {                                                 \
            _Pragma("unroll")                                                 \
            for (int rr = 0; rr < BM; ++rr) {                                 \
                float d = 0.0f;                                               \
                _Pragma("unroll")                                             \
                for (int j = 0; j < 32; ++j) {                                \
                    d += (float)As[rr * 33 + j] * (float)Bs[j * BP + tid];    \
                }                                                             \
                acc[rr] += d * w;                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (col0 + tid < n) {                                                     \
        for (int rr = 0; rr < BM; ++rr) {                                     \
            if (r0 + rr < m) {                                                \
                Cpart[(blockIdx.z * m + r0 + rr) * n + col0 + tid] = acc[rr]; \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_Q8K_TILED(f32, float)
DEFINE_Q8K_TILED(f16, __half)
DEFINE_Q8K_TILED(bf16, __nv_bfloat16)

// ─────────────────── Q6_K GEMM v2 (dequant-amortized) ───────────────────
// The v1 splitk dequantized inline per FMA: 3 shared byte loads + ~12 ALU
// per weight, strided 210 B (gcd(210,32)=2 — 2-way bank conflicts) — 49 us
// for the qkv shape vs 10 us fp16. v2 amortizes: per k-block, 256 threads
// cooperatively dequantize the 32-column tile ONCE into a TRANSPOSED f32
// tile (TILE[j][col] — consecutive threads read consecutive columns, bank
// conflict-free), then the compute is pure FMA. BN=32: Bs 6.7 KB + f32
// tile 32 KB fits the 48 KB static-shared budget. The A operand follows
// the activation axis: the activation tile As is stored AT (f32/f16/bf16 —
// the half widths halve the shared traffic and the (float) conversions).
#define DEFINE_K_SPLITK(NAME, SUF, AT, BLK, WREAD)                             \
extern "C" __global__ void gemm_##NAME##_splitk_v2_##SUF(                      \
    const AT* __restrict__ A,        /* (m, k) — activation axis */          \
    const unsigned char* __restrict__ B,/* packed blocks, (col·nsuper + kb)·BLK */ \
    float* __restrict__ Cpart,          /* (S·m, n) */                       \
    int k,                                                                   \
    int n,                                                                   \
    int S,                                                                   \
    int m)                                                                   \
{                                                                             \
    const int BN = 32;                                                        \
    __shared__ unsigned char Bs[BN * BLK];                                    \
    __shared__ float T[256][BN];                                              \
    __shared__ AT As[256];                                                    \
    __shared__ float part[8][33];                                             \
    int tid = threadIdx.x;                                                    \
    int bx = blockIdx.x;                                                      \
    int row = bx % m;                                                         \
    int nb = bx / m;                                                          \
    int col0 = nb * BN;                                                        \
    int nsup = k >> 8;                                                        \
    int nblk_s = (k / S) >> 8;                                                \
    int kb0 = blockIdx.y * nblk_s;                                            \
    const AT* arow = A + (size_t)row * k;                                     \
    float acc = 0.0f;                                                         \
    for (int t = 0; t < nblk_s; ++t) {                                        \
        int kb = kb0 + t;                                                     \
        /* Activation k-block (shared — the compute's 32 threads would not   \
           hide 256 global loads per thread otherwise). */                    \
        As[tid] = arow[kb * 256 + tid];                                       \
        const unsigned char* src = B + (size_t)(kb * n + col0) * BLK;         \
        /* Flat coalesced tile load (interior: BLK·BN contiguous; edge:      \
           per-column byte copy, garbage columns zeroed — masked later). */  \
        if (col0 + BN <= n && (n & 7) == 0) {                                 \
            for (int i = tid; i < (BN * BLK) / 16; i += 256) {                \
                *(uint4*)(Bs + i * 16) = *(const uint4*)(src + i * 16);       \
            }                                                                 \
        } else if (tid < BN) {                                                \
            int c = col0 + tid;                                               \
            if (c < n) {                                                      \
                const unsigned char* s = src + (size_t)tid * BLK;             \
                unsigned char* d = Bs + tid * BLK;                            \
                for (int i = 0; i < BLK; ++i) d[i] = s[i];                    \
            } else {                                                          \
                for (int i = 0; i < BLK; ++i) Bs[tid * BLK + i] = 0;          \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        /* Dequant pass: thread (col, jg) → TILE[j][col] (transposed —       \
           conflict-free writes and reads). */                                \
        int col = tid & 31;                                                   \
        int jg = tid >> 5;                                                    \
        {                                                                     \
            const unsigned char* blk = Bs + col * BLK;                        \
            for (int r = 0; r < 32; ++r) {                                    \
                int j = jg * 32 + r;                                          \
                T[j][col] = WREAD(blk, j);                                    \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        /* Compute: раньше FMA делали 32 потока из 256 (7/8 блока
           простаивали). Теперь каждый из 8 warp'ов (jg) суммирует свою
           восьмую часть k-блока в part[jg][col], затем потоки tid<BN
           складывают восемь частичных. Порядок суммирования отличается от
           прежнего (f32-шум ~1e-6), точность не страдает. */              \
        {                                                                     \
            const int j0 = jg * 32;                                           \
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;                 \
            for (int j = j0; j < j0 + 32; j += 4) {                           \
                a0 += (float)As[j + 0] * T[j + 0][col];                       \
                a1 += (float)As[j + 1] * T[j + 1][col];                       \
                a2 += (float)As[j + 2] * T[j + 2][col];                       \
                a3 += (float)As[j + 3] * T[j + 3][col];                       \
            }                                                                 \
            part[jg][col] = (a0 + a1) + (a2 + a3);                            \
        }                                                                     \
        __syncthreads();                                                      \
        if (tid < BN && col0 + tid < n) {                                     \
            float p0 = part[0][tid], p1 = part[1][tid], p2 = part[2][tid];    \
            float p3 = part[3][tid], p4 = part[4][tid], p5 = part[5][tid];    \
            float p6 = part[6][tid], p7 = part[7][tid];                       \
            acc += (((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));       \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (tid < BN && col0 + tid < n) {                                         \
        Cpart[(blockIdx.y * m + row) * n + col0 + tid] = acc;                 \
    }                                                                         \
}

DEFINE_K_SPLITK(q6k, f32, float, 210, q6k_weight)
DEFINE_K_SPLITK(q6k, f16, __half, 210, q6k_weight)
DEFINE_K_SPLITK(q6k, bf16, __nv_bfloat16, 210, q6k_weight)
DEFINE_K_SPLITK(q5k, f32, float, 176, q5k_weight)
DEFINE_K_SPLITK(q3k, f32, float, 110, q3k_weight)
// N (iter-10): canonical Q4_K (GGUF) — 144 B/256, the same v2 structure as
// q6k (dequant-amortized f32 tile). Previously GGUF Q4_K was dequantized to
// f32 on the host (7x DRAM traffic); this kernel is fed by from_cpu_q4kc.
DEFINE_K_SPLITK(q4kc, f32, float, 144, q4kc_weight)
DEFINE_K_SPLITK(q4kc, f16, __half, 144, q4kc_weight)
DEFINE_K_SPLITK(q4kc, bf16, __nv_bfloat16, 144, q4kc_weight)

/// K-quant GEMM, m > 1 (TTS prefill chunks + STT one-shot frames): block per
/// (32 rows × 64 cols), 256 threads = 64 columns × 4 rows each. The
/// 32×256 activation tile is staged in shared; the packed B tile is staged
/// too and the dequant is broadcast per column (the 4 row-threads of a
/// column read the same bytes).
// K-quant tiled GEMM (m > 1): A (m,k), B packed (col·nsuper + kb)·BLK,
// Cpart (S·m, n) with grid.z k-slices. NOTE: NO `//` comments inside the
// macro parameter list — the trailing `\` splices lines BEFORE comment
// removal, so an inline comment swallows the next parameter (this broke
// the whole file silently; the runtime ran a stale committed ptx).
#define DEFINE_K_TILED(NAME, BLK, WREAD)                                      \
extern "C" __global__ void gemm_##NAME##_tiled_v3(                            \
    const float* __restrict__ A,                                              \
    const unsigned char* __restrict__ B,                                      \
    float* __restrict__ Cpart,                                                \
    int m,                                                                   \
    int k,                                                                   \
    int n)                                                                   \
{                                                                             \
    const int BM = 32, BN = 64;                                               \
    __shared__ unsigned char Bs[BN * BLK];                                    \
    __shared__ float As[BM * 256];                                            \
    int tid = threadIdx.x;                                                    \
    int col0 = blockIdx.x * BN;                                               \
    int r0 = blockIdx.y * BM;                                                 \
    int nsup = k >> 8;                                                        \
    int nkz = (nsup + 15) >> 4;     /* k-split: grid.z=16 multiplies grid */ \
    int kb0 = blockIdx.z * nkz;                                               \
    if (kb0 >= nsup) return;                                                  \
    int kb1 = nsup < kb0 + nkz ? nsup : kb0 + nkz;                            \
    int col = tid & 63;                                                       \
    int rq = tid >> 6;                                                         \
    float acc[8];                                                             \
    for (int q = 0; q < 8; ++q) acc[q] = 0.0f;                                \
    for (int kb = kb0; kb < kb1; ++kb) {                                      \
        const unsigned char* src = B + (size_t)(kb * n + col0) * BLK;         \
        if (col0 + BN <= n && (n & 7) == 0) {                                 \
            for (int i = tid; i < (BN * BLK) / 16; i += 256) {                \
                *(uint4*)(Bs + i * 16) = *(const uint4*)(src + i * 16);       \
            }                                                                 \
        } else if (tid < BN) {                                                \
            int c = col0 + tid;                                               \
            if (c < n) {                                                      \
                const unsigned char* s = src + (size_t)tid * BLK;             \
                unsigned char* d = Bs + tid * BLK;                            \
                for (int i = 0; i < BLK; ++i) d[i] = s[i];                    \
            } else {                                                          \
                for (int i = 0; i < BLK; ++i) Bs[tid * BLK + i] = 0;          \
            }                                                                 \
        }                                                                     \
        /* Activation tile: 32 rows × 256 j — 4 rows per thread (i = r·256+j, \
           j = i&255, r = i>>8 → thread rq covers rows rq, rq+8, rq+16, rq+24).*/ \
        for (int i = tid; i < BM * 256; i += 256) {                           \
            int j = i & 255;                                                  \
            int rr = i >> 8;                                                  \
            int r = r0 + rr;                                                  \
            As[rr * 256 + j] = (r < m) ? A[(size_t)r * k + kb * 256 + j] : 0.0f;\
        }                                                                     \
        __syncthreads();                                                      \
        if (col0 + col < n) {                                                 \
            const unsigned char* blk = Bs + col * BLK;                        \
            for (int j = 0; j < 256; ++j) {                                   \
                float w = WREAD(blk, j);                                      \
                for (int q = 0; q < 8; ++q) {                                 \
                    int rr = rq + 4 * (q & 3) + 16 * (q >> 2);                \
                    if (r0 + rr < m) acc[q] += As[rr * 256 + j] * w;          \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (col0 + col >= n) return;                                              \
    for (int q = 0; q < 8; ++q) {                                             \
        int rr = rq + 4 * (q & 3) + 16 * (q >> 2);                            \
        if (r0 + rr < m) Cpart[(blockIdx.z * m + r0 + rr) * n + col0 + col] = acc[q]; \
    }                                                                         \
}

DEFINE_K_TILED(q6k, 210, q6k_weight)
DEFINE_K_TILED(q5k, 176, q5k_weight)
DEFINE_K_TILED(q3k, 110, q3k_weight)
DEFINE_K_TILED(q4kc, 144, q4kc_weight)

// ─────────── K-quant tiled GEMM v4 (m > 1, LLM prefill) ───────────
// v3 деquant'ит вес ЗАНОВО на каждую из 4 rq-групп строк одной колонки:
// ~20 ALU на вес при 8 FMA → у 27B prefill шёл на ~2.4 TFLOPS (8.6 мс/токен).
// v4 деquant'ит колонку ОДИН раз в общий f32-тайл T[256][33] и считает
// по нему 32 строки: ~0.6 ALU/вес + 1 FMA, активации читаются из shared
// broadcast'ом (все 32 потока warp'а — один адрес As[r][j]).
// Dynamic shared: T 33.8 KB + As 32×256 f32 32 KB = 65 KB (+ static Bs 4.6 KB)
// — сверх 48 KB, поэтому launcher обязан выставить
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (sm_89 предел 99 KB,
// sm_120 — 227 KB). S=1: результат пишется прямо в C, без reduce-прохода.
#define DEFINE_K_TILED_V4(NAME, BLK, WREAD)                                   \
extern "C" __global__ void gemm_##NAME##_tiled_v4(                            \
    const float* __restrict__ A,                                              \
    const unsigned char* __restrict__ B,                                      \
    float* __restrict__ C,                                                    \
    int m,                                                                    \
    int k,                                                                    \
    int n)                                                                    \
{                                                                             \
    const int BM = 32, BN = 32, BP = 33;                                      \
    extern __shared__ float sm[];                                             \
    float* T = sm;                      /* [256][BP] — dequant-тайл */       \
    float* As = sm + 256 * BP;          /* [BM][256] — активации */           \
    __shared__ unsigned char Bs[BN * BLK];                                    \
    int tid = threadIdx.x;                                                    \
    int col = tid & 31;                                                       \
    int jg = tid >> 5;                  /* warp → группа из 4 строк */        \
    int col0 = blockIdx.x * BN;                                               \
    int r0 = blockIdx.y * BM;                                                 \
    int nsup = k >> 8;                                                        \
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;                     \
    for (int kb = 0; kb < nsup; ++kb) {                                       \
        const unsigned char* src = B + (size_t)(kb * n + col0) * BLK;         \
        if (col0 + BN <= n && (n & 7) == 0) {                                 \
            for (int i = tid; i < (BN * BLK) / 16; i += 256)                  \
                *(uint4*)(Bs + i * 16) = *(const uint4*)(src + i * 16);       \
        } else if (tid < BN) {                                                \
            int c = col0 + tid;                                               \
            if (c < n) {                                                      \
                const unsigned char* s = src + (size_t)tid * BLK;             \
                unsigned char* d = Bs + tid * BLK;                            \
                for (int i = 0; i < BLK; ++i) d[i] = s[i];                    \
            } else {                                                          \
                for (int i = 0; i < BLK; ++i) Bs[tid * BLK + i] = 0;          \
            }                                                                 \
        }                                                                     \
        for (int i = tid; i < BM * 256; i += 256) {                           \
            int j = i & 255;                                                  \
            int rr = i >> 8;                                                  \
            int r = r0 + rr;                                                  \
            As[rr * 256 + j] = (r < m) ? A[(size_t)r * k + kb * 256 + j] : 0.f;\
        }                                                                     \
        __syncthreads();                                                      \
        {                                                                     \
            const unsigned char* blk = Bs + col * BLK;                        \
            for (int r = 0; r < 32; ++r) {                                    \
                int j = jg * 32 + r;                                          \
                T[j * BP + col] = WREAD(blk, j);                              \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
        for (int j = 0; j < 256; ++j) {                                       \
            float w = T[j * BP + col];                                        \
            acc0 += As[jg * 256 + j] * w;                                     \
            acc1 += As[(jg + 8) * 256 + j] * w;                               \
            acc2 += As[(jg + 16) * 256 + j] * w;                              \
            acc3 += As[(jg + 24) * 256 + j] * w;                              \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
    if (col0 + col < n) {                                                     \
        int c = col0 + col;                                                   \
        if (r0 + jg < m)      C[(size_t)(r0 + jg) * n + c] = acc0;            \
        if (r0 + jg + 8 < m)  C[(size_t)(r0 + jg + 8) * n + c] = acc1;        \
        if (r0 + jg + 16 < m) C[(size_t)(r0 + jg + 16) * n + c] = acc2;       \
        if (r0 + jg + 24 < m) C[(size_t)(r0 + jg + 24) * n + c] = acc3;       \
    }                                                                         \
}

DEFINE_K_TILED_V4(q6k, 210, q6k_weight)
DEFINE_K_TILED_V4(q4kc, 144, q4kc_weight)

// ─────── dequant в плотный bf16 (prefill через tensor cores) ───────
// На больших m (prefill-чанки) квантовый GEMM на FMA-ядрах упирается в
// ~25 TFLOPS, а bf16-тензорные ядра дают ×4-8. Поэтому для m ≥ 32 вес
// раскладывается в плотный bf16 (один проход, ~0.1 мс на матрицу 5120×17408)
// и считается cuBLAS gemm_ex (bf16×bf16 → f32 accumulate). Раскладка
// (col·nsuper + kb)·BLK → out[col·k + kb·256 + j].
#define DEFINE_DEQUANT_BF16(NAME, BLK, WREAD)                                 \
extern "C" __global__ void dequant_##NAME##_bf16(                             \
    const unsigned char* __restrict__ B,                                      \
    __nv_bfloat16* __restrict__ out,                                          \
    int n, int k)                                                             \
{                                                                             \
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;         \
    long long total = (long long)n * (long long)k;                            \
    if (idx >= total) return;                                                 \
    int col = (int)(idx / k);                                                 \
    int r = (int)(idx - (long long)col * k);                                  \
    int kb = r >> 8;                                                          \
    int j = r & 255;                                                          \
    out[idx] = (__nv_bfloat16)WREAD(B + (size_t)(kb * n + col) * BLK, j);     \
}

DEFINE_DEQUANT_BF16(q4kc, 144, q4kc_weight)
DEFINE_DEQUANT_BF16(q6k, 210, q6k_weight)

// Q8_0 (наша kb-major упаковка: блок (kb, col) = 32 i8 по (kb·n+col)·32,
// масштаб wsc[kb·n+col]) → плотный bf16 (n, k) row-major.
extern "C" __global__ void dequant_q8k_bf16(const signed char* __restrict__ B,
                                            const float* __restrict__ wsc,
                                            __nv_bfloat16* __restrict__ out,
                                            int n, int k) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n * (long long)k;
    if (idx >= total) return;
    int col = (int)(idx / k);
    int r = (int)(idx - (long long)col * k);
    int kb = r >> 5;
    int j = r & 31;
    size_t blk = (size_t)kb * n + col;
    out[idx] = (__nv_bfloat16)((float)B[blk * 32 + j] * wsc[blk]);
}

// ─────────────────────── im2col (conv → GEMM) ───────────────────────
//
// The DAC codec decoder runs ~10 conv1d + 4 conv-transpose1d per piece
// (~22 GFLOP); the naive per-element conv kernel did that at ~25 GFLOP/s
// (≈0.9 s per 64-token piece — the decode bottleneck). im2col turns every
// conv into a plain GEMM (cuBLAS fp16/bf16), which also lets the conv
// WEIGHTS take DAC_QUANTIZATION like the linear layers do.

/// im2col for a 1D conv (groups=1): X'[t, ci·k + kk] =
/// x[t·stride + kk·dil − pad_l, ci], zero outside. Layout (b, tout, cin·k).
extern "C" __global__ void im2col1d_f32(
    const float* __restrict__ x,   // (b, t, cin)
    float* __restrict__ xc,        // (b, tout, cin·k)
    int b,
    int t,
    int cin,
    int k,
    int stride,
    int pad_l,
    int dil,
    int tout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = cin * k;
    if (i >= b * tout * per) return;
    int bi = i / (tout * per);
    int rem = i % (tout * per);
    int to = rem / per;
    int ck = rem % per;
    int ci = ck / k;
    int kk = ck % k;
    int src = to * stride + kk * dil - pad_l;
    xc[i] = (src >= 0 && src < t) ? x[(bi * t + src) * cin + ci] : 0.0f;
}

/// im2col for a transposed 1D conv (upsample): the output frame to gets
/// contributions from input ti = (to − kk)/stride when divisible — gather:
/// X'[t, ci·k + kk] = x[(t − kk)/stride, ci] or 0.
extern "C" __global__ void im2col_t1d_f32(
    const float* __restrict__ x,   // (b, tin, cin)
    float* __restrict__ xc,        // (b, tout, cin·k)
    int b,
    int tin,
    int cin,
    int k,
    int stride,
    int tout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = cin * k;
    if (i >= b * tout * per) return;
    int bi = i / (tout * per);
    int rem = i % (tout * per);
    int to = rem / per;
    int ck = rem % per;
    int ci = ck / k;
    int kk = ck % k;
    int num = to - kk;
    float v = 0.0f;
    if (num >= 0 && num % stride == 0 && num / stride < tin) {
        v = x[(bi * tin + (num / stride)) * cin + ci];
    }
    xc[i] = v;
}

// DFlash2 two-tap dynamic conv (Ф2.3, спека PR #27342). Grid (t, i).
// ggml-порядок: BASE [2, k, D] (side·k·D + tap·D + i),
// DYN [T, proj] row-major: coeff = dyn[t·proj + r], r = g + n_groups·(tap + k·side).
extern "C" __global__ void dflash2_conv_f32(
    const float* __restrict__ x,
    const float* __restrict__ base,
    const float* __restrict__ dyn,
    float* __restrict__ y,
    int n_tokens, int n_embd, int kernel,
    int group_size, int block_size, int n_groups, int side)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= n_tokens || i >= n_embd) return;
    int blk = t / block_size;
    int pos = t % block_size;
    int g = i / group_size;
    int proj = n_groups * kernel * 2;
    float acc = 0.0f;
    for (int tap = 0; tap < kernel; ++tap) {
        float v = 0.0f;
        if (pos >= tap) {
            int src = blk * block_size + (pos - tap);
            v = x[src * n_embd + i];
        }
        int r = g + n_groups * (tap + kernel * side);
        float coeff = dyn[t * proj + r];
        float b = base[side * kernel * n_embd + tap * n_embd + i];
        acc += (coeff + b) * v;
    }
    y[t * n_embd + i] = acc;
}

// ───────────────── TTS top-k sampling on GPU (N iter-10) ────────────────
// The TTS slow-AR path downloaded the full vocab row every token and did
// the repetition/RAS penalties + bias + softmax + top-k on the host
// (~0.5-5 ms/token, GPU idle). These kernels compute the EXACT same
// pipeline on the GPU and return only the top-k window + the softmax
// denominator, so the host just does the RNG draw over ≤ k candidates.
//
// Semantics mirrored from the CPU reference:
//   val = logits[v];  per-occurrence rep penalty (logit>0 ? /=pen^c :
//   *=pen^c); RAS: last ras_window tokens with count>1 → −ras·(cnt−1);
//   += bias[v] (−inf outside the semantic range); /= temperature;
//   probs = exp(val − max)/sum.

/// Per-vocab occurrence counts from the history (u32 histogram).
extern "C" __global__ void hist_counts_u32(
    const unsigned int* __restrict__ history,
    unsigned int* __restrict__ counts,
    int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h) atomicAdd(&counts[history[i]], 1u);
}

/// Penalize + per-thread local top-k heap + block max/sum-exp. Scratch:
/// heap [256·k] (value,idx pairs), max [1], sum [1].
extern "C" __global__ void tts_topk_penalized_f32(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const unsigned int* __restrict__ counts,
    const unsigned int* __restrict__ history,
    int h, int v, int k,
    float rep_pen, float ras_pen, int ras_window, float temperature,
    float* __restrict__ heap_val,
    unsigned int* __restrict__ heap_idx,
    float* __restrict__ out_max,
    float* __restrict__ out_sum)
{
    const int stride = blockDim.x;
    int n = (v + stride - 1) / stride;
    float hval[64];
    unsigned int hidx[64];
    for (int i = 0; i < k; ++i) { hval[i] = -INFINITY; hidx[i] = 0; }
    float local_max = -INFINITY;
    // Pass 1: penalize + heap insert + local max.
    for (int j = 0; j < n; ++j) {
        int gid = j * stride + threadIdx.x;
        if (gid >= v) continue;
        float val = logits[gid];
        unsigned int c = counts[gid];
        if (c > 0 && rep_pen != 1.0f) {
            float f = powf(rep_pen, (float)c);
            val = val > 0.0f ? val / f : val * f;
        }
        if (ras_pen != 1.0f && h >= 2) {
            int ws = h > ras_window ? h - ras_window : 0;
            unsigned int cnt = 0;
            for (int i = ws; i < h; ++i) if (history[i] == (unsigned int)gid) ++cnt;
            if (cnt > 1) val -= ras_pen * (float)(cnt - 1);
        }
        if (bias != NULL) val += bias[gid];
        val /= temperature;
        if (val > local_max) local_max = val;
        if (val > hval[k - 1]) {
            int pos = k - 1;
            hval[pos] = val; hidx[pos] = (unsigned int)gid;
            while (pos > 0 && hval[pos] > hval[pos - 1]) {
                float tv = hval[pos]; hval[pos] = hval[pos - 1]; hval[pos - 1] = tv;
                unsigned int ti = hidx[pos]; hidx[pos] = hidx[pos - 1]; hidx[pos - 1] = ti;
                --pos;
            }
        }
    }
    // Block max (tree reduce).
    __shared__ float sh[256];
    sh[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sh[threadIdx.x + s] > sh[threadIdx.x])
            sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    float gmax = sh[0];
    // Pass 2: block sum of exp(val − gmax) (recompute val — cheap).
    float local_sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        int gid = j * stride + threadIdx.x;
        if (gid >= v) continue;
        float val = logits[gid];
        unsigned int c = counts[gid];
        if (c > 0 && rep_pen != 1.0f) {
            float f = powf(rep_pen, (float)c);
            val = val > 0.0f ? val / f : val * f;
        }
        if (ras_pen != 1.0f && h >= 2) {
            int ws = h > ras_window ? h - ras_window : 0;
            unsigned int cnt = 0;
            for (int i = ws; i < h; ++i) if (history[i] == (unsigned int)gid) ++cnt;
            if (cnt > 1) val -= ras_pen * (float)(cnt - 1);
        }
        if (bias != NULL) val += bias[gid];
        val /= temperature;
        local_sum += expf(val - gmax);
    }
    sh[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_max[0] = gmax;
        out_sum[0] = sh[0];
    }
    // Dump the per-thread heaps to global scratch (one column per thread).
    for (int i = 0; i < k; ++i) {
        heap_val[threadIdx.x * k + i] = hval[i];
        heap_idx[threadIdx.x * k + i] = hidx[i];
    }
}

/// Merge 256·k heap entries → top-k (single block, k rounds of tree reduce;
/// each thread owns its 32-entry slice with a taken-mask in registers).
extern "C" __global__ void tts_topk_merge_f32(
    const float* __restrict__ heap_val,
    const unsigned int* __restrict__ heap_idx,
    int n,      // 256·k candidates
    int k,
    float* __restrict__ out_val,
    unsigned int* __restrict__ out_idx)
{
    const int per = (n + blockDim.x - 1) / blockDim.x; // ≤ 32 for n ≤ 8192
    float lv[32];
    unsigned int li[32];
    unsigned int taken = 0u; // bitmask of taken slots
    for (int i = 0; i < per; ++i) {
        int c = threadIdx.x + i * blockDim.x;
        if (c < n) {
            lv[i] = heap_val[c];
            li[i] = heap_idx[c];
        } else {
            lv[i] = -INFINITY;
            li[i] = 0;
        }
    }
    __shared__ float sh[256];
    __shared__ unsigned int shi[256];
    for (int r = 0; r < k; ++r) {
        // each thread: best untaken slot
        float best = -INFINITY;
        unsigned int best_i = 0;
        int best_slot = -1;
        for (int i = 0; i < per; ++i) {
            if (!(taken & (1u << i)) && lv[i] > best) {
                best = lv[i];
                best_i = li[i];
                best_slot = i;
            }
        }
        sh[threadIdx.x] = best;
        shi[threadIdx.x] = best_i;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s && sh[threadIdx.x + s] > sh[threadIdx.x]) {
                sh[threadIdx.x] = sh[threadIdx.x + s];
                shi[threadIdx.x] = shi[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_val[r] = sh[0];
            out_idx[r] = shi[0];
        }
        // mark the winner's slot taken (only the owner thread)
        if (best_slot >= 0 && best == sh[0] && best_i == shi[0]) {
            taken |= (1u << best_slot);
        }
        __syncthreads();
    }
}

// ── Row copy without an offset view (CUDA has no nonzero elem_offset views) ──
// copy_tail_f32: dst[0..d) = src[off..off+d) — the read-only row_view.
extern "C" __global__ void copy_tail_f32(const float* __restrict__ src,
                                         float* __restrict__ dst,
                                         int off, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) dst[i] = src[off + i];
}

// copy_into_f32: dst[off..off+d) = src[0..d) — row_write into a batch tensor.
extern "C" __global__ void copy_into_f32(const float* __restrict__ src,
                                         float* __restrict__ dst,
                                         int off, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) dst[off + i] = src[i];
}

// ─────────────── GDN (gated delta net) ───────────────
// Три операции qwen35 до этого исполнялись дефолтными CPU-реализациями
// Backend (to_cpu_f32 → хост-циклы → from_cpu_f32) на КАЖДОМ слое: у 27B
// 48 GDN-слоёв из 64, и профиль дал 97 мс из 142 мс на токен (68 %) только
// на них. Семантика повторяет CPU-эталон (см. backend.rs).

// Гейты: BA [T, 2Hv] → g = softplus(alpha + dt_bias)·a, beta = sigmoid(beta_raw).
extern "C" __global__ void gdn_gates_f32(const float* __restrict__ ba,
                                         const float* __restrict__ dt_bias,
                                         const float* __restrict__ a,
                                         float* __restrict__ g,
                                         float* __restrict__ b,
                                         int n, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int hh = i % h;
    const float x = ba[i] + dt_bias[hh];
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    g[i] = sp * a[hh];
    b[i] = 1.0f / (1.0f + expf(-ba[n + i]));
}

// conv1d (k) + SiLU по каналам + сдвиг свёрточного стейта (k-1).
extern "C" __global__ void gdn_conv_silu_f32(const float* __restrict__ x,
                                             const float* __restrict__ w,
                                             const float* __restrict__ state,
                                             float* __restrict__ y,
                                             float* __restrict__ ns,
                                             int c, int k) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    const float* wch = w + (size_t)ch * k;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const float v = (j < k - 1) ? state[(size_t)ch * (k - 1) + j] : x[ch];
        acc += wch[j] * v;
    }
    y[ch] = acc / (1.0f + expf(-acc));
    float* nsch = ns + (size_t)ch * (k - 1);
    for (int j = 0; j < k - 1; ++j) {
        nsch[j] = (j + 1 < k - 1) ? state[(size_t)ch * (k - 1) + j + 1] : x[ch];
    }
}

// L2-нормировка сегментов длины hd (первый 2·qk выходa conv).
extern "C" __global__ void gdn_l2norm_seg_f32(float* __restrict__ y, float eps,
                                              int hd) {
    extern __shared__ float ssum[];
    const int seg = blockIdx.x;
    float* row = y + (size_t)seg * hd;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += row[i] * row[i];
    ssum[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(ssum[0] + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) row[i] *= inv;
}

// GDN causal depthwise conv для префилла (t>1): y[t,c] = Σ_j w[c,j]·at(c,t+j),
// at(c,p) = p < K−1 ? state[c·(K−1)+p] : x[(p−(K−1))·C + c]. Один поток на
// (t, c). Раньше этой операции не было на CUDA — дефолтный Backend гонял
// 37 МБ на слой через хост (23 мс/слой из 48, весь префилл упирался в это).
extern "C" __global__ void gdn_conv_seq_f32(const float* __restrict__ x,
                                            const float* __restrict__ w,
                                            const float* __restrict__ state,
                                            float* __restrict__ y,
                                            int t, int c, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * c) return;
    int tt = idx / c;
    int ch = idx - tt * c;
    const float* wch = w + (size_t)ch * k;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        int p = tt + j;
        float v = (p < k - 1) ? state[(size_t)ch * (k - 1) + p]
                              : x[(size_t)(p - (k - 1)) * c + ch];
        acc += wch[j] * v;
    }
    y[idx] = acc;
}

// Новый conv-стейт = последние K−1 входов окна.
extern "C" __global__ void gdn_conv_state_f32(const float* __restrict__ x,
                                              const float* __restrict__ state,
                                              float* __restrict__ ns,
                                              int t, int c, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c * (k - 1)) return;
    int ch = idx / (k - 1);
    int j = idx - ch * (k - 1);
    int p = t + j;
    ns[idx] = (p < k - 1) ? state[(size_t)ch * (k - 1) + p]
                          : x[(size_t)(p - (k - 1)) * c + ch];
}

// L2-нормировка out-of-place (префилл): y = x / sqrt(Σx² + eps) по отрезкам
// длины hd. Та же редукция и тот же 1/sqrtf, что у in-place версии.
extern "C" __global__ void gdn_l2norm_out_f32(const float* __restrict__ x,
                                              float* __restrict__ y,
                                              float eps, int hd) {
    extern __shared__ float ssum[];
    const int seg = blockIdx.x;
    const float* row = x + (size_t)seg * hd;
    float* out = y + (size_t)seg * hd;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += row[i] * row[i];
    ssum[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(ssum[0] + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) out[i] = row[i] * inv;
}

// Gated delta rule для S=128: state [Hv,128,128] (транспонирован как в ggml),
// один token-блок 16 токенов в shared один раз на блок. Схема повторяет
// Metal gdn_delta_rule_chunk_f32: s *= e^g; delta = (v − s·k)·β;
// s += k·delta; out = s·q/√S. Строка стейта — warp (lane = 0..31,
// элементы lane + e·32), строки блока — threadIdx.y.
#define GDN_S 128
#define GDN_C 16
#define GDN_ROWS 16
extern "C" __global__ void gdn_delta_rule_chunk_f32(
    const float* __restrict__ Q, const float* __restrict__ Kx,
    const float* __restrict__ V, const float* __restrict__ G,
    const float* __restrict__ B, float* __restrict__ STATE,
    float* __restrict__ OUT, int T, int Hk, int Hv) {
    __shared__ float kt[GDN_C * GDN_S];
    __shared__ float qt[GDN_C * GDN_S];
    __shared__ float gb[2 * GDN_C];
    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 32 + lane;
    const int h = blockIdx.y;
    const int row = blockIdx.x * GDN_ROWS + ty;
    if (h >= Hv || row >= GDN_S) return;
    const int hk = h % Hk;
    const float scale = 1.0f / sqrtf((float)GDN_S);
    float* srow = STATE + ((size_t)h * GDN_S + row) * GDN_S;
    float ls0 = srow[lane], ls1 = srow[lane + 32], ls2 = srow[lane + 64],
          ls3 = srow[lane + 96];
    for (int t0 = 0; t0 < T; t0 += GDN_C) {
        const int c = min(GDN_C, T - t0);
        for (int i = tid; i < GDN_C * GDN_S; i += 32 * GDN_ROWS) {
            const int tt = i / GDN_S, rest = i % GDN_S;
            const int l = rest % 32, ee = rest / 32;
            const int off = tt * GDN_S + l + ee * 32;
            if (tt < c) {
                kt[off] = Kx[((size_t)(t0 + tt) * Hk + hk) * GDN_S + ee * 32 + l];
                qt[off] = Q[((size_t)(t0 + tt) * Hk + hk) * GDN_S + ee * 32 + l];
            } else {
                kt[off] = 0.0f;
                qt[off] = 0.0f;
            }
        }
        if (tid < GDN_C) {
            gb[tid] = (tid < c) ? G[(t0 + tid) * Hv + h] : 0.0f;
            gb[GDN_C + tid] = (tid < c) ? B[(t0 + tid) * Hv + h] : 0.0f;
        }
        __syncthreads();
        for (int tt = 0; tt < c; ++tt) {
            const float eg = expf(gb[tt]);
            const float beta = gb[GDN_C + tt];
            const int tb = tt * GDN_S;
            ls0 *= eg; ls1 *= eg; ls2 *= eg; ls3 *= eg;
            float kv = ls0 * kt[tb + lane] + ls1 * kt[tb + lane + 32] +
                       ls2 * kt[tb + lane + 64] + ls3 * kt[tb + lane + 96];
            for (int off = 16; off > 0; off >>= 1)
                kv += __shfl_down_sync(0xffffffffu, kv, off);
            kv = __shfl_sync(0xffffffffu, kv, 0);
            const float delta =
                (V[((size_t)(t0 + tt) * Hv + h) * GDN_S + row] - kv) * beta;
            ls0 += kt[tb + lane] * delta;
            ls1 += kt[tb + lane + 32] * delta;
            ls2 += kt[tb + lane + 64] * delta;
            ls3 += kt[tb + lane + 96] * delta;
            float y = ls0 * qt[tb + lane] + ls1 * qt[tb + lane + 32] +
                      ls2 * qt[tb + lane + 64] + ls3 * qt[tb + lane + 96];
            for (int off = 16; off > 0; off >>= 1)
                y += __shfl_down_sync(0xffffffffu, y, off);
            if (lane == 0)
                OUT[((size_t)(t0 + tt) * Hv + h) * GDN_S + row] = y * scale;
        }
        __syncthreads();
    }
    srow[lane] = ls0;
    srow[lane + 32] = ls1;
    srow[lane + 64] = ls2;
    srow[lane + 96] = ls3;
}

// Гейтовая RMS-норма GDN: y = x·inv·w·silu(z), inv = 1/√(mean(x²)+eps) по
// отрезкам hd. Один блок на строку (hd ≤ 1024).
extern "C" __global__ void gdn_norm_gated_f32(const float* __restrict__ x,
                                              const float* __restrict__ z,
                                              const float* __restrict__ w,
                                              float* __restrict__ y, int hd,
                                              float eps) {
    extern __shared__ float ssum[];
    const size_t base = (size_t)blockIdx.x * hd;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float v = x[base + i];
        acc += v * v;
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(ssum[0] / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float zz = z[base + i];
        y[base + i] = x[base + i] * inv * w[i] * (zz / (1.0f + expf(-zz)));
    }
}
