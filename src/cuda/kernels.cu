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

/// y = x / sqrt(mean(x²) + eps) · w, row-wise. x (rows, cols), w (cols).
extern "C" __global__ void rmsnorm_f32(
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
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) sum += row[j] * row[j];
    float inv = rsqrtf(sum / cols + eps);
    float* out = y + r * cols;
    for (int j = 0; j < cols; ++j) out[j] = row[j] * inv * w[j];
}

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
extern "C" __global__ void conv_transpose1d_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
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
    if (i >= b * t * cin) return;
    int bi = i / (t * cin);
    int rem = i % (t * cin);
    int ti = rem / cin;
    int ci = rem % cin;
    float xv = x[i];
    for (int co = 0; co < cout; ++co) {
        for (int kk = 0; kk < k; ++kk) {
            int to = ti * stride + kk;
            atomicAdd(&y[(bi * tout + to) * cout + co], xv * w[(ci * cout + co) * k + kk]);
        }
    }
}

/// bias add for transposed conv output (separate pass to avoid atomics on bias).
extern "C" __global__ void conv_transpose_bias_f32(
    float* __restrict__ y,
    const float* __restrict__ bias,
    int n,
    int cout)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += bias[i % cout];
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
extern "C" __global__ void attn_ar_v2_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ mask,
    float* __restrict__ y,
    int b,
    int tq,
    int tkv,
    int n_heads,
    int kv_heads,
    int hd,
    int n_rep,
    int has_mask)
{
    int bi = blockIdx.z;
    int hh = blockIdx.y;
    int t = blockIdx.x;
    if (bi >= b || hh >= n_heads || t >= tq) return;
    int kvh = hh / n_rep;
    const float* qrow = q + (bi * tq + t) * (n_heads * hd) + hh * hd;
    __shared__ float scores[4096];
    __shared__ float exps[4096];
    __shared__ float red[256];
    __shared__ float smx;
    float scale = rsqrtf((float)hd);
    if (tkv > 4096) return;
    if (threadIdx.x == 0) smx = -INFINITY;
    __syncthreads();

    // Phase 1: scores[tk] = q·k[tk] * scale (+mask), block-reduced over hd.
    for (int tk = 0; tk < tkv; ++tk) {
        const float* krow = k + (bi * tkv + tk) * (kv_heads * hd) + kvh * hd;
        float acc = 0.0f;
        for (int j = threadIdx.x; j < hd; j += blockDim.x) acc += qrow[j] * krow[j];
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float sc = red[0] * scale;
            if (has_mask) sc += mask[t * tkv + tk];
            scores[tk] = sc;
            smx = fmaxf(smx, sc);
        }
        __syncthreads();
    }

    // Phase 2: exp/sum (softmax denominator).
    float sum = 0.0f;
    for (int tk = threadIdx.x; tk < tkv; tk += blockDim.x) {
        float e = __expf(scores[tk] - smx);
        exps[tk] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];

    // Phase 3: out[j] = Σ_tk exps[tk]·inv·v[tk][j], j = threadIdx.x.
    float* out = y + (bi * tq + t) * (n_heads * hd) + hh * hd;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        float acc = 0.0f;
        for (int tk = 0; tk < tkv; ++tk) {
            const float* vrow = v + (bi * tkv + tk) * (kv_heads * hd) + kvh * hd;
            acc += exps[tk] * inv * vrow[j];
        }
        out[j] = acc;
    }
}

/// Autoregressive GQA attention (TTS): q (b, tq, n_heads·hd),
/// k/v (b, tkv, kv_heads·hd), mask (tq, tkv) bias (0 / -inf) or null.
/// One thread per (b, head, tq) — softmax over tkv in f32.
extern "C" __global__ void attn_ar_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ mask,
    float* __restrict__ y,
    int b,
    int tq,
    int tkv,
    int n_heads,
    int kv_heads,
    int hd,
    int n_rep,
    int has_mask)
{
    int bi = blockIdx.x;
    int hh = blockIdx.y;
    int t = blockIdx.z;
    if (bi >= b || hh >= n_heads || t >= tq) return;
    int kvh = hh / n_rep;
    const float* qrow = q + (bi * tq + t) * (n_heads * hd) + hh * hd;
    float scale = rsqrtf((float)hd);
    // scores
    float scores[4096];
    if (tkv > 4096) return; // safety: prefill chunks stay under 4096 (Phase 7 removes this)
    float mx = -INFINITY;
    for (int tk = 0; tk < tkv; ++tk) {
        const float* krow = k + (bi * tkv + tk) * (kv_heads * hd) + kvh * hd;
        float acc = 0.0f;
        for (int j = 0; j < hd; ++j) acc += qrow[j] * krow[j];
        acc *= scale;
        if (has_mask) acc += mask[t * tkv + tk];
        scores[tk] = acc;
        mx = fmaxf(mx, acc);
    }
    float sum = 0.0f;
    for (int tk = 0; tk < tkv; ++tk) sum += expf(scores[tk] - mx);
    float* out = y + (bi * tq + t) * (n_heads * hd) + hh * hd;
    for (int j = 0; j < hd; ++j) {
        float acc = 0.0f;
        for (int tk = 0; tk < tkv; ++tk) {
            const float* vrow = v + (bi * tkv + tk) * (kv_heads * hd) + kvh * hd;
            acc += expf(scores[tk] - mx) / sum * vrow[j];
        }
        out[j] = acc;
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

/// Quantize activations f32 (m, k) → i8 + per-32-block scales (m, k/32).
/// One thread per 32-element block; k%32==0 required (callers pad).
extern "C" __global__ void quant_f32_i8_blk(
    const float* __restrict__ x,
    signed char* __restrict__ y,
    float* __restrict__ sc,
    int m,
    int k)
{
    int nblk = k >> 5;
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // over m*nblk blocks
    if (i >= m * nblk) return;
    int r = i / nblk;
    int kb = i % nblk;
    const float* src = x + r * k + kb * 32;
    float mx = 0.0f;
    float v[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        v[j] = src[j];
        mx = fmaxf(mx, fabsf(src[j]));
    }
    float s = mx / 127.0f;
    if (s < 1e-12f) s = 1.0f;
    sc[i] = s;
    signed char* dst = y + r * k + kb * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float q = fminf(127.0f, fmaxf(-128.0f, v[j] / s));
        dst[j] = (signed char)__float2int_rn(q);
    }
}

/// int8 GEMM, m == 1 (TTS decode hot path — one activation row against the
/// weight matrix). B is stored TRANSPOSED (k, n) row-major (weights arrive
/// as (n, k); upload_quant transposes them so warp reads are coalesced).
///
/// Block = 256 threads → 256 columns. Each k-tile (32×256) is staged into
/// shared (padded to 260 → no bank conflicts on read or write): warp w loads
/// k-row kb·32+w — 32 consecutive bytes per warp, one per lane. Every thread
/// then dots the shared tile with its A segment and dequantizes with one
/// fp32 FMA per tile.
extern "C" __global__ void gemm_i8_m1(
    const signed char* __restrict__ A,   // (k)
    const float* __restrict__ asc,       // (k/32)
    const signed char* __restrict__ B,   // (k, n) row-major
    const float* __restrict__ wsc,       // (n, k/32)
    float* __restrict__ C,               // (n)
    int k,
    int n)
{
    __shared__ signed char Bs[32][260];
    int tid = threadIdx.x;
    int col = blockIdx.x * 256 + tid;
    int nblk = k >> 5;
    int w = tid >> 5;
    int lane = tid & 31;
    int col_base = blockIdx.x * 256;

    float acc = 0.0f;
    for (int kb = 0; kb < nblk; ++kb) {
        // Stage the k-tile (32 rows × 256 cols): each warp loads 4 k-rows
        // (w, w+8, w+16, w+24), 8 column segments of 32 bytes each — every
        // load is one full 32-byte coalesced row segment.
#pragma unroll
        for (int r4 = 0; r4 < 4; ++r4) {
            int row = w + r4 * 8;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                int gcol = col_base + s * 32 + lane;
                signed char v = (gcol < n) ? B[(kb * 32 + row) * n + gcol] : 0;
                Bs[row][s * 32 + lane] = v;
            }
        }
        __syncthreads();
        const signed char* a = A + kb * 32;
        int s = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            s += (int)a[j] * (int)Bs[j][tid];
        }
        if (col < n) acc += (float)s * (asc[kb] * wsc[col * nblk + kb]);
        __syncthreads();
    }
    if (col < n) C[col] = acc;
}

/// int8 GEMM, m > 1 — classic tiled kernel. BM=64, BN=64, BK=32; block
/// (16,16) = 256 threads, each thread owns a 4x4 output micro-tile;
/// per-32-block i32 accumulation with one fp32 dequant FMA per k-tile.
/// B is stored transposed (k, n) row-major — warps load consecutive columns.
extern "C" __global__ void gemm_i8_tiled(
    const signed char* __restrict__ A,   // (m, k)
    const float* __restrict__ asc,       // (m, k/32)
    const signed char* __restrict__ B,   // (k, n) row-major
    const float* __restrict__ wsc,       // (n, k/32)
    float* __restrict__ C,               // (m, n)
    int m,
    int k,
    int n)
{
    const int BM = 64, BN = 64, BK = 32;
    __shared__ signed char As[64 * 32];
    __shared__ signed char Bs[32 * 64];
    __shared__ float ascs[64];
    __shared__ float wscs[64];

    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;
    int nblk = k >> 5;
    int tx = threadIdx.x, ty = threadIdx.y;

    int acc_i[4][4] = {};
    float acc_f[4][4] = {};

    for (int kb = 0; kb < nblk; ++kb) {
        // Load A tile (64 x 32) + row scales; B tile (32 x 64) + col scales.
        // A tile: 64 rows × 32 k = 2048 elements, 8 per thread (the (16,16)
        // block covers it via the linear index — tx alone spans only 16 of
        // 32 k elements).
        for (int idx = ty * 16 + tx; idx < BM * BK; idx += 256) {
            int i = idx / BK;
            int j = idx % BK;
            int row = m0 + i;
            As[i * BK + j] = (row < m) ? A[row * k + kb * BK + j] : 0;
        }
        for (int i = ty; i < BM; i += 16) {
            int row = m0 + i;
            ascs[i] = (row < m) ? asc[row * nblk + kb] : 0.0f;
        }
        for (int j = tx; j < BN; j += 16) {
            int col = n0 + j;
            wscs[j] = (col < n) ? wsc[col * nblk + kb] : 0.0f;
        }
        // B tile: 32 k-rows × 64 cols; B is (k, n) row-major — consecutive
        // idx = consecutive columns within a row (coalesced warp loads).
        for (int idx = ty * 16 + tx; idx < BK * BN; idx += 256) {
            int i = idx / BN;
            int j = idx % BN;
            int kk = kb * BK + i;
            int col = n0 + j;
            Bs[i * BN + j] = (col < n) ? B[kk * n + col] : 0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            signed char av[4];
            signed char bv[4];
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) av[rr] = As[(ty * 4 + rr) * BK + i];
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) bv[cc] = Bs[i * BN + tx * 4 + cc];
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    acc_i[rr][cc] += (int)av[rr] * (int)bv[cc];
                }
            }
        }
        // Dequant this k-tile: one fp32 FMA per output element.
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                acc_f[rr][cc] += (float)acc_i[rr][cc] * (ascs[ty * 4 + rr] * wscs[tx * 4 + cc]);
            }
        }
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) acc_i[rr][cc] = 0;
        }
        __syncthreads();
    }
    // Write out (with edge masks).
    for (int rr = 0; rr < 4; ++rr) {
        int row = m0 + ty * 4 + rr;
        if (row >= m) continue;
        for (int cc = 0; cc < 4; ++cc) {
            int col = n0 + tx * 4 + cc;
            if (col < n) C[row * n + col] = acc_f[rr][cc];
        }
    }
}
