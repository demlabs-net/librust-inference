// Metal Shading Language kernels for the Metal engine.
//
// Design rule: ONE kernel call == ONE command-buffer commit (the caller
// batches a whole model step into a single command buffer when needed).
// No im2col materialization: convs are direct.

#include <metal_stdlib>
using namespace metal;

// MSL `tanh` returns NaN for large |x| on some macOS/GPU stacks (observed on
// AGXG15C / macOS 15.7.3 with x = 44.45). Stable form via exp:
// tanh(x) = 1 - 2/(e^(2x)+1); e^(2x) saturates to inf (→ 1) or 0 (→ -1).
float stable_tanh(float x) {
    float e = exp(2.0f * x);
    return 1.0f - 2.0f / (e + 1.0f);
}

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

// C[m,n] = Σ A[m*K+k] · B[n*K+k], B in bf16 (ushort bits).
static inline float bf16_to_f32(device const ushort* p, uint i) {
    return as_type<float>((uint)p[i] << 16);
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

// ── 1D convolution with groups (dense groups=1, depthwise groups=D) ─────────
// X: [B, Cin, T]; W: [Cout, Cin/groups, K]; bias: [Cout] or null
// Y: [B, Cout, T_out], t_out = (T + padL + padR - dil*(K-1) - 1)/stride + 1
// out channel co reads input channels [g*Cin_g, (g+1)*Cin_g) with g = co / Cin_g
kernel void conv1d_groups_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& Cin [[buffer(5)]],
    constant uint& Cout [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& Cin_g [[buffer(8)]],
    constant int& stride [[buffer(9)]],
    constant int& padL [[buffer(10)]],
    constant int& padR [[buffer(11)]],
    constant int& dilation [[buffer(12)]],
    constant uint& t_major [[buffer(13)]],
    constant uint& t_major_out [[buffer(14)]],
    constant uint& dense [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, co = gid.z;
    uint t_out = (T + (uint)padL + (uint)padR - (uint)dilation * (K - 1u) - 1u) / (uint)stride + 1u;
    if (t >= t_out) return;
    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    // dense (groups=1): every out channel reads ALL Cin channels (ci_in = ci,
    // Cin_g == Cin). Grouped (depthwise): out channel co reads channels
    // [g*Cin_g, (g+1)*Cin_g) with g = co / Cin_g.
    for (uint ci = 0; ci < Cin_g; ++ci) {
        uint ci_in = (dense != 0u) ? ci : ((co / Cin_g) * Cin_g + ci);
        for (uint k = 0; k < K; ++k) {
            int t_in = (int)(t * (uint)stride) - padL + (int)k * dilation;
            if (t_in >= 0 && t_in < (int)T) {
                float xv = (t_major != 0u)
                    ? X[(b * T + (uint)t_in) * Cin + ci_in]
                    : X[(b * Cin + ci_in) * T + (uint)t_in];
                acc += xv * W[(co * Cin_g + ci) * K + k];
            }
        }
    }
    if (t_major_out != 0u) {
        Y[(b * t_out + t) * Cout + co] = acc;
    } else {
        Y[(b * Cout + co) * t_out + t] = acc;
    }
}

// ── Direct 2D convolution (subsampling): X [B,Cin,T,F], W [Cout,Cin,kT,kF] ──
// Left-pad on the time axis by padT (no right/freq padding); stride (sT, sF).
// Bias already contains the folded BatchNorm shift.
kernel void conv2d_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& Cin [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& F [[buffer(6)]],
    constant uint& Cout [[buffer(7)]],
    constant uint& kT [[buffer(8)]],
    constant uint& kF [[buffer(9)]],
    constant int& sT [[buffer(10)]],
    constant int& sF [[buffer(11)]],
    constant int& padT [[buffer(12)]],
    constant uint& t_major_out [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, co = gid.y;
    uint t_out = (T + (uint)padT - kT) / (uint)sT + 1u;
    uint f_out = (F - kF) / (uint)sF + 1u;
    uint t = gid.z / f_out;      // packed: z = t*f_out + f
    uint f = gid.z % f_out;
    if (t >= t_out || f >= f_out) return;
    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    for (uint ci = 0; ci < Cin; ++ci) {
        for (uint kt = 0; kt < kT; ++kt) {
            int t_in = (int)(t * (uint)sT) - padT + (int)kt;
            if (t_in < 0) continue;
            for (uint kf = 0; kf < kF; ++kf) {
                int f_in = (int)(f * (uint)sF) + (int)kf;
                acc += X[((b * Cin + ci) * T + (uint)t_in) * F + (uint)f_in]
                     * W[((co * Cin + ci) * kT + kt) * kF + kf];
            }
        }
    }
    if (t_major_out != 0u) {
        // time-major output [B, T_out, Cout*F_out] (flatten-ready for the
        // pre-encode linear): feature index = co*f_out + f
        Y[(b * t_out + t) * (Cout * f_out) + co * f_out + f] = acc;
    } else {
        Y[((b * Cout + co) * t_out + t) * f_out + f] = acc;
    }
}

// ── Softmax over the last dim ───────────────────────────────────────────────
kernel void softmax_last_dim_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    if (r >= rows) return;
    const device float* row = X + r * cols;
    float mx = -INFINITY;
    for (uint c = 0; c < cols; ++c) mx = max(mx, row[c]);
    float sum = 0.0f;
    for (uint c = 0; c < cols; ++c) sum += exp(row[c] - mx);
    device float* out = Y + r * cols;
    for (uint c = 0; c < cols; ++c) out[c] = exp(row[c] - mx) / sum;
}

// ── RMSNorm over the last dim: y = x * rsqrt(mean(x^2) + eps) * w ────────────
kernel void rmsnorm_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    if (r >= rows) return;
    const device float* row = X + r * D;
    float sum = 0.0f;
    for (uint d = 0; d < D; ++d) sum += row[d] * row[d];
    float inv = rsqrt(sum / (float)D + eps);
    device float* out = Y + r * D;
    for (uint d = 0; d < D; ++d) out[d] = row[d] * inv * W[d];
}

// ── LayerNorm over the last dim: y = (x-mean)/sqrt(var+eps)*w + b ────────────
kernel void layernorm_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    if (r >= rows) return;
    const device float* row = X + r * D;
    float mean = 0.0f;
    for (uint d = 0; d < D; ++d) mean += row[d];
    mean /= (float)D;
    float var = 0.0f;
    for (uint d = 0; d < D; ++d) { float c = row[d] - mean; var += c * c; }
    var /= (float)D;
    float inv = rsqrt(var + eps);
    device float* out = Y + r * D;
    for (uint d = 0; d < D; ++d) out[d] = (row[d] - mean) * inv * W[d] + B[d];
}

// ── SiLU ─────────────────────────────────────────────────────────────────────
kernel void silu_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = X[gid];
    Y[gid] = x / (1.0f + exp(-x));
}

// ── ReLU ─────────────────────────────────────────────────────────────────────
kernel void relu_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = X[gid];
    Y[gid] = x > 0.0f ? x : 0.0f;
}

// ── Sigmoid: 1/(1+exp(-x)) — safe for large |x| ──────────────────────────────
kernel void sigmoid_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = X[gid];
    Y[gid] = 1.0f / (1.0f + exp(-x));
}

// ── Elementwise add: Y = A + B ───────────────────────────────────────────────
kernel void add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = A[gid] + B[gid];
}

// ── Elementwise subtract: Y = A - B ──────────────────────────────────────────
kernel void sub_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = A[gid] - B[gid];
}

// ── Elementwise divide: Y = A / B ────────────────────────────────────────────
kernel void div_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = A[gid] / B[gid];
}

// ── Elementwise add scalar: Y = X + c ────────────────────────────────────────
kernel void add_scalar_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& c [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = X[gid] + c;
}

// ── Elementwise sqrt ─────────────────────────────────────────────────────────
kernel void sqrt_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = sqrt(X[gid]);
}

// ── Elementwise log ──────────────────────────────────────────────────────────
kernel void log_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = log(X[gid]);
}

// ── Elementwise max with a scalar floor: Y = max(X, floor) ───────────────────
kernel void max_scalar_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& floor [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = X[gid];
    Y[gid] = x > floor ? x : floor;
}

// ── Clipped leaky ReLU: Y = max(leaky(x, 0.2), -0.4) ─────────────────────────
kernel void leaky_relu_clip_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = X[gid];
    float y = x > 0.0f ? x : 0.2f * x;
    Y[gid] = y > -0.4f ? y : -0.4f;
}

// ── 2D convolution (NCHW): y[n,co,oh,ow] = Σ x[n,ci,ih,iw]·w[co,ci,kh,kw] ────
// X (N,Ci,H,W), W (Co,Ci,KH,KW), bias (Co) or None, Y (N,Co,OH,OW).
kernel void conv2d_nchw_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& Ci [[buffer(5)]],
    constant uint& Hi [[buffer(6)]],
    constant uint& Wi [[buffer(7)]],
    constant uint& Co [[buffer(8)]],
    constant uint& KH [[buffer(9)]],
    constant uint& KW [[buffer(10)]],
    constant uint& OH [[buffer(11)]],
    constant uint& OW [[buffer(12)]],
    constant uint& pad_h [[buffer(13)]],
    constant uint& pad_w [[buffer(14)]],
    constant uint& stride_t [[buffer(15)]],
    constant uint& stride_f [[buffer(16)]],
    constant uint& time_major_out [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, n = gid.y, co = gid.z;
    uint oh = t / OW, ow = t % OW;
    if (oh >= OH || co >= Co) return;
    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    for (uint ci = 0; ci < Ci; ++ci) {
        for (uint kh = 0; kh < KH; ++kh) {
            int ih = (int)(oh * stride_t) - (int)pad_h + (int)kh;
            if (ih < 0 || ih >= (int)Hi) continue;
            for (uint kw = 0; kw < KW; ++kw) {
                int iw = (int)(ow * stride_f) - (int)pad_w + (int)kw;
                if (iw < 0 || iw >= (int)Wi) continue;
                acc += X[((n * Ci + ci) * Hi + (uint)ih) * Wi + (uint)iw]
                     * W[((co * Ci + ci) * KH + kh) * KW + kw];
            }
        }
    }
    if (time_major_out != 0u) {
        // (n, oh, co, ow) — same linear layout as (n, oh, co*ow), the t-one
        // subsampling layout.
        Y[((n * OH + oh) * Co + co) * OW + ow] = acc;
    } else {
        Y[((n * Co + co) * OH + oh) * OW + ow] = acc;
    }
}

// ── MaxPool2d (kernel == stride): y[n,co,oh,ow] = max over (kh,kw) window ────
kernel void maxpool2d_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& Hi [[buffer(4)]],
    constant uint& Wi [[buffer(5)]],
    constant uint& OH [[buffer(6)]],
    constant uint& OW [[buffer(7)]],
    constant uint& KH [[buffer(8)]],
    constant uint& KW [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, n = gid.y, co = gid.z;
    uint oh = t / OW, ow = t % OW;
    if (oh >= OH || co >= C) return;
    float m = -1e30f;
    for (uint kh = 0; kh < KH; ++kh) {
        uint ih = oh * KH + kh;
        for (uint kw = 0; kw < KW; ++kw) {
            uint iw = ow * KW + kw;
            float v = X[((n * C + co) * Hi + ih) * Wi + iw];
            m = v > m ? v : m;
        }
    }
    Y[((n * C + co) * OH + oh) * OW + ow] = m;
}

// ── GLU: gate * sigmoid(value), sigmoid via the tanh identity ───────────────
// X [rows, 2D] → Y [rows, D]
kernel void glu_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x, d = gid.y;
    if (r >= rows || d >= D) return;
    float gate = X[r * 2u * D + d];
    float value = X[r * 2u * D + D + d];
    float sig = 0.5f * tanh(0.5f * value) + 0.5f;
    Y[r * D + d] = gate * sig;
}

// ── Broadcast add (bias): Y[b*n + i] = X[b*n + i] + bias[i] ─────────────────
kernel void add_bias_f32(
    device const float* X [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x, b = gid.y;
    if (i >= n) return;
    Y[b * n + i] = X[b * n + i] + bias[i];
}

// ── Add a per-element bias buffer: Y = X + B (same shape) ───────────────────
// (used to add the chunked-attention mask bias to scores)
kernel void add_bias2d_f32(
    device const float* X [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = X[gid] + B[gid];
}

// ── Scaled residual: Y = A + B*scale ────────────────────────────────────────
kernel void add_scaled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = A[gid] + B[gid] * scale;
}

// ── Scale: Y = X * scale ────────────────────────────────────────────────────
kernel void scale_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = X[gid] * scale;
}

// ── Uniform attention fallback: Y[b,h,t,hd] = mean over kv of V[b,h,kv,hd] ──
kernel void uniform_attn_f32(
    device const float* V [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& KV [[buffer(5)]],
    constant uint& HD [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, h = gid.y;
    uint t = gid.z / HD;        // packed: z = t*HD + d
    uint d = gid.z % HD;
    if (b >= B || h >= H || t >= T || d >= HD) return;
    float acc = 0.0f;
    for (uint kv = 0; kv < KV; ++kv) {
        acc += V[((b * H + h) * KV + kv) * HD + d];
    }
    Y[((b * H + h) * T + t) * HD + d] = acc / (float)KV;
}

// ── Rotary embedding (halved, with pass-through) ────────────────────────────
// X [T, head_dim]; cos/sin [T, half]; rotate the first rope_dim dims only.
kernel void rope_f32(
    device const float* X [[buffer(0)]],
    device const float* cos_buf [[buffer(1)]],
    device const float* sin_buf [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& rope_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x, d = gid.y;
    if (t >= T || d >= head_dim) return;
    uint hd = rope_dim / 2u;
    if (d < rope_dim) {
        if (d < hd) {
            Y[t * head_dim + d] = X[t * head_dim + d] * cos_buf[t * hd + d] - X[t * head_dim + d + hd] * sin_buf[t * hd + d];
        } else {
            uint h = d - hd;
            Y[t * head_dim + d] = X[t * head_dim + h] * sin_buf[t * hd + h] + X[t * head_dim + d] * cos_buf[t * hd + h];
        }
    } else {
        Y[t * head_dim + d] = X[t * head_dim + d];
    }
}

// ── Elementwise multiply: Y = A * B ─────────────────────────────────────────
kernel void mul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    Y[gid] = A[gid] * B[gid];
}

// ── Repeat frames: Y[b, 2t+i, d] = X[b, t, d] (factor 2, truncating write) ──
kernel void repeat_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& T_OUT [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, d = gid.z;
    if (b >= B || t >= T || d >= D) return;
    uint o0 = 2u * t;
    if (o0 < T_OUT) Y[(b * T_OUT + o0) * D + d] = X[(b * T + t) * D + d];
    if (o0 + 1u < T_OUT) Y[(b * T_OUT + o0 + 1u) * D + d] = X[(b * T + t) * D + d];
}

// ── Fused multi-head self-attention (t-one style) ───────────────────────────
// Q/K/V: [B, T, D] time-major (head h occupies dims [h*HD, (h+1)*HD)).
// SC: [B, H, T, T] — leader writes scores; non-leader reads (pre-masked).
// MASK: [T, T] bias (0 / -10000) or null.
// MODE: 0 = leader (compute scores), 1 = non-leader (use SC), 2 = uniform.
// One thread per (b, h, t): loops over kv.
kernel void mhsa_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* SC [[buffer(3)]],
    device const float* MASK [[buffer(4)]],
    device float* Y [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& H [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    constant uint& MODE [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, h = gid.y, t = gid.z;
    if (b >= B || h >= H || t >= T) return;
    uint base = h * HD;
    const device float* qrow = Q + (b * T + t) * (H * HD) + base;
    const device float* vrow0 = V + b * T * (H * HD) + base;

    if (MODE == 0u) {
        // Leader: scores[t'] = dot(q, k[t']) / sqrt(HD) + mask[t][t']
        device float* scrow = SC + ((b * H + h) * T + t) * T;
        float inv = 1.0f / sqrt((float)HD);
        for (uint tp = 0; tp < T; ++tp) {
            const device float* krow = K + (b * T + tp) * (H * HD) + base;
            float acc = 0.0f;
            for (uint d = 0; d < HD; ++d) acc += qrow[d] * krow[d];
            float m = (MASK != nullptr) ? MASK[t * T + tp] : 0.0f;
            scrow[tp] = acc * inv + m;
        }
        // softmax over tp
        float mx = -INFINITY;
        for (uint tp = 0; tp < T; ++tp) mx = max(mx, scrow[tp]);
        float sum = 0.0f;
        for (uint tp = 0; tp < T; ++tp) sum += exp(scrow[tp] - mx);
        // weighted sum of V
        device float* yrow = Y + (b * T + t) * (H * HD) + base;
        for (uint d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (uint tp = 0; tp < T; ++tp) {
                acc += exp(scrow[tp] - mx) / sum * vrow0[tp * (H * HD) + d];
            }
            yrow[d] = acc;
        }
    } else if (MODE == 1u) {
        // Non-leader: reuse cached (pre-masked) scores
        const device float* scrow = SC + ((b * H + h) * T + t) * T;
        float mx = -INFINITY;
        for (uint tp = 0; tp < T; ++tp) mx = max(mx, scrow[tp]);
        float sum = 0.0f;
        for (uint tp = 0; tp < T; ++tp) sum += exp(scrow[tp] - mx);
        device float* yrow = Y + (b * T + t) * (H * HD) + base;
        for (uint d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (uint tp = 0; tp < T; ++tp) {
                acc += exp(scrow[tp] - mx) / sum * vrow0[tp * (H * HD) + d];
            }
            yrow[d] = acc;
        }
    } else {
        // Uniform: mean of V over kv
        device float* yrow = Y + (b * T + t) * (H * HD) + base;
        for (uint d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (uint tp = 0; tp < T; ++tp) acc += vrow0[tp * (H * HD) + d];
            yrow[d] = acc / (float)T;
        }
    }
}

// ── Head-aware rotary embedding (t-one): X [B, T, D], head dims per head ────
// rotate the first rope_dim dims of EACH head (head width HD); pass-through
// the rest. cos/sin [T, rope_dim/2]. offset for keys (streaming) baked into
// the precomputed cos/sin.
kernel void rope_head_f32(
    device const float* X [[buffer(0)]],
    device const float* cos_buf [[buffer(1)]],
    device const float* sin_buf [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& HD [[buffer(7)]],
    constant uint& rope_dim [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint bh = gid.x, t = gid.y, d = gid.z;
    uint D = H * HD;
    if (bh >= B * H || t >= T || d >= D) return;
    uint h = bh % H;
    uint dr = d % HD;
    uint idx = (bh / H * T + t) * D + h * HD + dr;   // absolute element
    if (dr < rope_dim) {
        uint hd = rope_dim / 2u;
        float c = cos_buf[t * hd + (dr % hd)];
        float s = sin_buf[t * hd + (dr % hd)];
        if (dr < hd) {
            Y[idx] = X[idx] * c - X[(bh / H * T + t) * D + h * HD + dr + hd] * s;
        } else {
            uint h2 = dr - hd;
            Y[idx] = X[(bh / H * T + t) * D + h * HD + h2] * s + X[idx] * c;
        }
    } else {
        Y[idx] = X[idx];
    }
}

// ── Head-aware LayerNorm (t-one q_ln/k_ln): X [B, T, D] with H heads of
// width HD; normalize per (row, head): y = (x-mean_h)/sqrt(var_h+eps)*w+b.
kernel void layernorm_head_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant uint& HD [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint r = gid.x, h = gid.y, d = gid.z;
    if (r >= rows || h >= H || d >= HD) return;
    const device float* row = X + r * (H * HD) + h * HD;
    float mean = 0.0f;
    for (uint dd = 0; dd < HD; ++dd) mean += row[dd];
    mean /= (float)HD;
    float var = 0.0f;
    for (uint dd = 0; dd < HD; ++dd) { float c = row[dd] - mean; var += c * c; }
    var /= (float)HD;
    float inv = rsqrt(var + eps);
    Y[r * (H * HD) + h * HD + d] = (row[d] - mean) * inv * W[d] + B[d];
}

// ── Dense 1D convolution (benchmark reference, im2col-free) ────────────────
// X: [B, T, C_in] time-major; W: [C_out, C_in, K]; bias: [C_out] or null
// Y: [B, T_out, C_out]
kernel void conv1d_direct_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& C_in [[buffer(5)]],
    constant uint& C_out [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant int& stride [[buffer(8)]],
    constant int& pad [[buffer(9)]],
    constant int& dilation [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, co = gid.z;
    uint t_out = (T + 2u * (uint)pad - (uint)dilation * (K - 1u) - 1u) / (uint)stride + 1u;
    if (t >= t_out) return;
    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    for (uint ci = 0; ci < C_in; ++ci) {
        for (uint k = 0; k < K; ++k) {
            int t_in = (int)(t * (uint)stride) - pad + (int)k * dilation;
            if (t_in >= 0 && t_in < (int)T) {
                acc += X[(b * T + (uint)t_in) * C_in + ci] * W[(co * C_in + ci) * K + k];
            }
        }
    }
    Y[(b * t_out + t) * C_out + co] = acc;
}

// ── Transposed 1D convolution (DAC upsample/decoder) ────────────────────────
// X: [B, T, Cin] time-major; W: [Cin, Cout, K] (candle conv_transpose layout!
// the stored kernel is (in_channels, out_channels, k)); bias: [Cout] or null
// Y: [B, T_out, Cout], T_out = (T - 1) * stride + K
// y[b, n, co] = sum_{k: (n-k) % stride == 0, 0 <= m = (n-k)/stride < T}
//               x[b, m, ci] · w[ci, co, k]
kernel void conv_transpose1d_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& Cin [[buffer(5)]],
    constant uint& Cout [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& stride [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, n = gid.y, co = gid.z;
    uint t_out = (T - 1u) * stride + K;
    if (n >= t_out) return;
    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    // k ≡ n (mod stride), k < K, m = (n - k)/stride in [0, T)
    uint k0 = n % stride;
    for (uint k = k0; k < K; k += stride) {
        uint m = (n - k) / stride;
        if (m >= T) continue;
        const device float* xrow = X + (b * T + m) * Cin;
        for (uint ci = 0; ci < Cin; ++ci) {
            acc += xrow[ci] * W[(ci * Cout + co) * K + k];
        }
    }
    Y[(b * t_out + n) * Cout + co] = acc;
}

// ── Snake activation (DAC): y = x + sin(a·x)^2 / a, a per-channel ────────────
// X: [B, T, C]; A: [C]; Y: [B, T, C]
kernel void snake_f32(
    device const float* X [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x, c = gid.y;
    if (i >= n) return;
    float x = X[i * C + c];
    float a = A[c];
    Y[i * C + c] = x + sin(a * x) * sin(a * x) / a;
}

// ── GELU (tanh approximation) — DAC ConvNeXt pwconv ─────────────────────────
kernel void gelu_tanh_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= n) return;
    float x = X[i];
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    Y[i] = 0.5f * x * (1.0f + stable_tanh(inner));
}

// ── Embedding lookup (codebook): idx [B, T] uint, table [N, D] → Y [B, T, D]
kernel void embedding_f32(
    device const uint* idx [[buffer(0)]],
    device const float* table [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, d = gid.z;
    if (b >= 1u || t >= T || d >= D) return;   // batch fixed to 1 for codec decode
    uint row = idx[b * T + t];
    if (row >= N) row = N - 1u;                 // clamp — decode never OOB
    Y[(b * T + t) * D + d] = table[row * D + d];
}

// ── RoPE, interleaved pairs per head: pairs (dr, dr+1), dr even ────────────
// X: [B, T, D] with D = H*HD; cos/sin: [max_seq, HD/2]; START = absolute
// position offset (AR generation); Y: [B, T, D]
kernel void rope_interleaved_f32(
    device const float* X [[buffer(0)]],
    device const float* cos_buf [[buffer(1)]],
    device const float* sin_buf [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& HD [[buffer(7)]],
    constant uint& start [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, d = gid.z;
    if (b >= B || t >= T || d >= D) return;
    uint h = d / HD;
    uint dr = d % HD;
    uint hf = HD / 2u;
    uint pair = dr / 2u;
    float c = cos_buf[(start + t) * hf + pair];
    float s = sin_buf[(start + t) * hf + pair];
    uint idx = (b * T + t) * D + h * HD + dr;
    if (dr % 2u == 0u) {
        Y[idx] = X[idx] * c - X[idx + 1u] * s;
    } else {
        Y[idx] = X[idx - 1u] * s + X[idx] * c;
    }
}

// ── Broadcast-multiply by a per-column vector (LayerScale): X (rows, cols) ──
kernel void mul_vec_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x, c = gid.y;
    if (r >= rows || c >= cols) return;
    Y[r * cols + c] = X[r * cols + c] * W[c];
}

// ── tanh (elementwise) — DAC decoder output ────────────────────────────────
kernel void tanh_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= n) return;
    Y[i] = stable_tanh(X[i]);
}

// ── Slice along the last dim (DAC q/k/v split): X (rows, D) → Y (rows, d0),
//    Y[row, d] = X[row, off + d]
kernel void slice_last_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& off [[buffer(4)]],
    constant uint& d0 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x, d = gid.y;
    if (r >= rows || d >= d0) return;
    Y[r * d0 + d] = X[r * D + off + d];
}

// ── Causal unpad along time (DAC decoder transposed convs): ────────────────
// Candle's CausalConvTranspose1d drops the LAST U frames ("remove right
// padding"): X (B, T, C) → Y (B, T-U, C) with Y[b,t,c] = X[b, t, c].
kernel void slice_time_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& U [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, c = gid.z;
    if (b >= B || t >= T - U || c >= C) return;
    Y[((b * (T - U) + t) * C) + c] = X[((b * T + t) * C) + c];
}

// ── Nearest codebook entry (DAC RVQ encode): X (rows, 8), table (N, 8) ──────
// → idx (rows): argmin_N ||x - cb||^2 (L2, exact)
kernel void codebook_search_f32(
    device const float* X [[buffer(0)]],
    device const float* table [[buffer(1)]],
    device uint* idx [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint r = gid;
    if (r >= rows) return;
    const device float* x = X + r * 8u;
    float best = INFINITY;
    uint bi = 0u;
    for (uint n = 0u; n < N; ++n) {
        const device float* cb = table + n * 8u;
        float d = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            float c = x[k] - cb[k];
            d += c * c;
        }
        if (d < best) { best = d; bi = n; }
    }
    idx[r] = bi;
}

// ── KV-cache update: dst (B, cap, D)[b, pos+t, d] = src (B, Tq, D)[b, t, d] ─
kernel void kv_update_f32(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint& B [[buffer(2)]],
    constant uint& Tq [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& cap [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, d = gid.z;
    if (b >= B || t >= Tq || d >= D) return;
    dst[(b * cap + pos + t) * D + d] = src[(b * Tq + t) * D + d];
}

// ── GQA attention with KV cache (AR decode/prefill) ─────────────────────────
// q: (B, Tq, H*HD); k/v: (B, Tkv, KVH*HD); y: (B, Tq, H*HD)
// head h attends kv-head h / n_rep. scores scale = 1/sqrt(HD).
// MASK: (B, Tq, Tkv) or null (generation: causal implicit — q row i sees
// kv rows 0..=i).
kernel void attn_ar_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* MASK [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    constant uint& Tq [[buffer(6)]],
    constant uint& Tkv [[buffer(7)]],
    constant uint& H [[buffer(8)]],
    constant uint& KVH [[buffer(9)]],
    constant uint& HD [[buffer(10)]],
    constant uint& n_rep [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, h = gid.y, t = gid.z;
    if (b >= B || h >= H || t >= Tq) return;
    uint kvh = h / n_rep;
    uint D = H * HD;
    uint KVD = KVH * HD;
    const device float* qrow = Q + (b * Tq + t) * D + h * HD;
    // all cached kv is visible; causality comes from MASK (prefill only)
    uint kv_lim = Tkv;
    float mx = -INFINITY;
    for (uint j = 0u; j < kv_lim; ++j) {
        const device float* krow = K + (b * Tkv + j) * KVD + kvh * HD;
        float acc = 0.0f;
        for (uint d = 0u; d < HD; ++d) acc += qrow[d] * krow[d];
        float m = (MASK != nullptr) ? MASK[(b * Tq + t) * Tkv + j] : 0.0f;
        float s = acc * (1.0f / sqrt((float)HD)) + m; // 1/sqrt(head_dim) — the runtime HD, not a hardcoded 128
        if (s > mx) mx = s;
        // stash raw scores in Y temporarily? no — recompute below
    }
    // recompute sums to avoid extra buffer: two passes over kv
    float sum = 0.0f;
    for (uint j = 0u; j < kv_lim; ++j) {
        const device float* krow = K + (b * Tkv + j) * KVD + kvh * HD;
        float acc = 0.0f;
        for (uint d = 0u; d < HD; ++d) acc += qrow[d] * krow[d];
        float m = (MASK != nullptr) ? MASK[(b * Tq + t) * Tkv + j] : 0.0f;
        float s = acc * 0.08838834764831845f + m;
        sum += exp(s - mx);
    }
    device float* yrow = Y + (b * Tq + t) * D + h * HD;
    for (uint d = 0u; d < HD; ++d) {
        float acc = 0.0f;
        for (uint j = 0u; j < kv_lim; ++j) {
            const device float* vrow = V + (b * Tkv + j) * KVD + kvh * HD;
            const device float* krow = K + (b * Tkv + j) * KVD + kvh * HD;
            float qk = 0.0f;
            for (uint dd = 0u; dd < HD; ++dd) qk += qrow[dd] * krow[dd];
            float m = (MASK != nullptr) ? MASK[(b * Tq + t) * Tkv + j] : 0.0f;
            float s = qk * 0.08838834764831845f + m;
            acc += exp(s - mx) / sum * vrow[d];
        }
        yrow[d] = acc;
    }
}

// ── RMSNorm per head: X (rows, H*HD) → Y, normalized per (row, head) ────────
// W (HD). (QK-norm: q/k normalized per head BEFORE RoPE.)
kernel void rmsnorm_head_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint r = gid.x, h = gid.y, d = gid.z;
    if (r >= rows || h >= H || d >= HD) return;
    const device float* row = X + r * (H * HD) + h * HD;
    float sum = 0.0f;
    for (uint dd = 0u; dd < HD; ++dd) sum += row[dd] * row[dd];
    float inv = rsqrt(sum / (float)HD + eps);
    Y[r * (H * HD) + h * HD + d] = row[d] * inv * W[d];
}

// ── Per-row scalar multiply (embedding mask/scale): X (rows, D), W (rows) ──
kernel void mul_rows_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x, d = gid.y;
    if (r >= rows || d >= D) return;
    Y[r * D + d] = X[r * D + d] * W[r];
}

// ── im2col (conv → GEMM; the DAC codec runs its convs through GEMMs) ──
// Dense 1D conv (groups=1): Xc[t, ci*K + kk] = X[t*stride + kk*dil - pad_l, ci]
// (zero outside). X: (B, T, Cin); Xc: (B, Tout, Cin*K).
kernel void im2col1d_f32(
    device const float* X [[buffer(0)]],
    device float* Xc [[buffer(1)]],
    constant uint& T [[buffer(2)]],
    constant uint& Cin [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant int& stride [[buffer(5)]],
    constant int& pad_l [[buffer(6)]],
    constant int& dil [[buffer(7)]],
    constant uint& Tout [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, ck = gid.z;
    uint ci = ck / K;
    uint kk = ck % K;
    int src = (int)(t * (uint)stride) - pad_l + (int)kk * dil;
    float v = 0.0f;
    if (src >= 0 && src < (int)T) {
        v = X[(b * T + (uint)src) * Cin + ci];
    }
    Xc[(b * Tout + t) * (Cin * K) + ck] = v;
}

// Transposed 1D conv (upsample): Xc[t, ci*K+kk] = X[(t-kk)/stride, ci] when
// (t-kk) % stride == 0 and (t-kk)/stride < Tin, else 0.
// X: (B, Tin, Cin); Xc: (B, Tout, Cin*K).
kernel void im2col_t1d_f32(
    device const float* X [[buffer(0)]],
    device float* Xc [[buffer(1)]],
    constant uint& Tin [[buffer(2)]],
    constant uint& Cin [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    constant uint& Tout [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, ck = gid.z;
    uint ci = ck / K;
    uint kk = ck % K;
    float v = 0.0f;
    if (t >= kk && (t - kk) % stride == 0) {
        uint src = (t - kk) / stride;
        if (src < Tin) {
            v = X[(b * Tin + src) * Cin + ci];
        }
    }
    Xc[(b * Tout + t) * (Cin * K) + ck] = v;
}

// ── Quantized GEMM / MV: A fp32 × B quantized [N, K], fp32 accumulate ───────
// Block layouts (ggml, reimplemented): Q4_0 18 B/32, Q8_0 34 B/32,
// Q4_K 148 B/256, Q6_K 210 B/256. Activations read directly from the block
// bytes; CPU reference: weights_map::dequant_f32 (tests).

static inline half load_half_unaligned(device const uchar* p) {
    // даже смещения блоков (18/34/148/210 кратны 2) — выравнивание ushort ок
    return as_type<half>(*(device const ushort*)p);
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
        for (uint j = 0; j < 16; ++j) {
            uchar v = q[j];
            acc += A[m * K + k0 + j * 2 + 0] * ((float)(v & 0xF) - 8.0f) * d;
            acc += A[m * K + k0 + j * 2 + 1] * ((float)(v >> 4) - 8.0f) * d;
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

// Q4_K: суперблок 256: d(f16), dmin(f16), scales[16] (4-бит sc: байты 0..7,
// 4-бит m: байты 8..15), qs[128]. y = d*dmin + d*sc*(q - 8).
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
        device const uchar* s = B + (n * superblocks + sb) * 148;
        float d = (float)load_half_unaligned(s);
        float dmin = (float)load_half_unaligned(s + 2);
        float min = d * dmin;
        device const uchar* qs = s + 20;
        uint k0 = sb * 256;
        for (uint b = 0; b < 8; ++b) {
            uchar scn = s[4 + b / 2];
            float sc = (float)((b % 2 == 0) ? (scn & 0xF) : (scn >> 4));
            uint kb = k0 + b * 32;
            for (uint j = 0; j < 16; ++j) {
                uchar v = qs[b * 16 + j];
                acc += A[m * K + kb + j * 2 + 0] * (min + d * sc * ((float)(v & 0xF) - 8.0f));
                acc += A[m * K + kb + j * 2 + 1] * (min + d * sc * ((float)(v >> 4) - 8.0f));
            }
        }
    }
    C[m * N + n] = acc;
}

// Q6_K: суперблок 256: d(f16), ql[128] (4 бит), qh[64] (2 бит), scales[16]
// (8 низких + 8 высоких нибблов = 16 × 8-бит). q = ql | (qh<<4); y = d*sc*(q-32).
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
        float d = (float)load_half_unaligned(s);
        device const uchar* ql = s + 2;
        device const uchar* qh = s + 130;
        device const uchar* scales = s + 194;
        uint k0 = sb * 256;
        for (uint b = 0; b < 8; ++b) {
            float sc = (float)((scales[b] & 0xF) | ((scales[b + 8] & 0xF) << 4));
            uint kb = k0 + b * 32;
            for (uint j = 0; j < 32; ++j) {
                uint idx = b * 32 + j;
                uchar lo = (j % 2 == 0) ? (ql[idx / 2] & 0xF) : (ql[idx / 2] >> 4);
                uchar hi = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x3;
                float q = (float)(lo | (hi << 4));
                acc += A[m * K + kb + j] * d * sc * (q - 32.0f);
            }
        }
    }
    C[m * N + n] = acc;
}

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
        for (uint j = 0; j < 16; ++j) {
            uchar v = q[j];
            acc += A[k0 + j * 2 + 0] * ((float)(v & 0xF) - 8.0f) * d;
            acc += A[k0 + j * 2 + 1] * ((float)(v >> 4) - 8.0f) * d;
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
        device const uchar* s = B + (n * superblocks + sb) * 148;
        float d = (float)load_half_unaligned(s);
        float dmin = (float)load_half_unaligned(s + 2);
        float min = d * dmin;
        device const uchar* qs = s + 20;
        uint k0 = sb * 256;
        for (uint b = 0; b < 8; ++b) {
            uchar scn = s[4 + b / 2];
            float sc = (float)((b % 2 == 0) ? (scn & 0xF) : (scn >> 4));
            uint kb = k0 + b * 32;
            for (uint j = 0; j < 16; ++j) {
                uchar v = qs[b * 16 + j];
                acc += A[kb + j * 2 + 0] * (min + d * sc * ((float)(v & 0xF) - 8.0f));
                acc += A[kb + j * 2 + 1] * (min + d * sc * ((float)(v >> 4) - 8.0f));
            }
        }
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
        float d = (float)load_half_unaligned(s);
        device const uchar* ql = s + 2;
        device const uchar* qh = s + 130;
        device const uchar* scales = s + 194;
        uint k0 = sb * 256;
        for (uint b = 0; b < 8; ++b) {
            float sc = (float)((scales[b] & 0xF) | ((scales[b + 8] & 0xF) << 4));
            uint kb = k0 + b * 32;
            for (uint j = 0; j < 32; ++j) {
                uint idx = b * 32 + j;
                uchar lo = (j % 2 == 0) ? (ql[idx / 2] & 0xF) : (ql[idx / 2] >> 4);
                uchar hi = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x3;
                float q = (float)(lo | (hi << 4));
                acc += A[kb + j] * d * sc * (q - 32.0f);
            }
        }
    }
    C[n] = acc;
}

// ── Стандартный rotate-half RoPE (HF/Qwen3) — Ф1.7 perf ─────────────────────
// y[t,h,i] = rotate_half(x) с cos/sin(theta^(-2i/HD)) на абсолютной позиции
// (pos + t). Замена CPU-хелпера rope_standard в qwen3.rs.
kernel void rope_std_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& tq [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& HD [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, i = gid.z;
    if (t >= tq || h >= D / HD || i >= HD / 2) return;
    float p = (float)(pos + t);
    float inv = 1.0 / pow(theta, 2.0 * (float)i / (float)HD);
    float c = cos(p * inv);
    float s = sin(p * inv);
    uint base = (t * D + h * HD + 2 * i);
    float x0 = X[base];
    float x1 = X[base + 1];
    Y[base] = x0 * c - x1 * s;
    Y[base + 1] = x1 * c + x0 * s;
}
