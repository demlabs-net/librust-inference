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

// ── Fused SwiGLU gate: Y[i] = silu(X[i]) * X[row·2·FFN + FFN + i%FFN] ────────
// X: [rows, 2*FFN] (fused gate|up), Y: [rows, FFN]. Заменяет silu + mul
// (2 кернела) — и позволяет слить gate/up-проекции в один GEMM.
kernel void silu_mul_fused_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& ffn [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint row = gid / ffn;
    uint i = gid % ffn;
    float g = X[row * 2u * ffn + i];
    float up = X[row * 2u * ffn + ffn + i];
    Y[gid] = (g / (1.0f + exp(-g))) * up;
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
// Аргмакс по строкам логитов [T, V] + logsumexp строки. ids[t] — ПЕРВЫЙ
// максимум (порядок сравнений, как у CPU-скана greedy_next: строго больше),
// NaN и −∞ → −∞; lse[t] = log Σ exp(x − max) по полному ряду (нужно для
// честных q при сэмплинге). FACTORS — множитель штрафа за повторы на токен
// (1.0 вне истории) или nullptr. Одна threadgroup на строку: выбор черновиков
// и жадная верификация остаются на GPU — вниз идут только id (было: строка
// логитов [1+γ, V] на хост).
kernel void argmax_lse_rows_f32(
    device const float* X [[buffer(0)]],
    device const float* FACTORS [[buffer(1)]],
    device float* IDS [[buffer(2)]],
    device float* LSE [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& V [[buffer(5)]],
    constant uint& HAS_FACTORS [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort ntg [[threads_per_threadgroup]])
{
    if (tgid >= T) return;
    threadgroup float shm[1024];
    threadgroup uint shi[1024];
    threadgroup float shs[1024];
    device const float* row = X + (ulong)tgid * V;
    float m = -INFINITY;
    uint mi = 0u;
    for (uint i = tiitg; i < V; i += ntg) {
        float x = row[i];
        if (HAS_FACTORS != 0u) {
            x = x > 0.0f ? x / FACTORS[i] : x * FACTORS[i];
        }
        if (!isfinite(x)) x = -INFINITY;
        if (x > m) {
            m = x;
            mi = i;
        }
    }
    shm[tiitg] = m;
    shi[tiitg] = mi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg / 2u; s > 0u; s >>= 1) {
        if (tiitg < s) {
            const float o = shm[tiitg + s];
            const uint oi = shi[tiitg + s];
            if (o > shm[tiitg] || (o == shm[tiitg] && oi < shi[tiitg])) {
                shm[tiitg] = o;
                shi[tiitg] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float gmax = shm[0];
    float ssum = 0.0f;
    if (gmax > -INFINITY) {
        for (uint i = tiitg; i < V; i += ntg) {
            float x = row[i];
            if (HAS_FACTORS != 0u) {
                x = x > 0.0f ? x / FACTORS[i] : x * FACTORS[i];
            }
            if (!isfinite(x)) continue;
            ssum += exp(x - gmax);
        }
    }
    shs[tiitg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = ntg / 2u; st > 0u; st >>= 1) {
        if (tiitg < st) {
            shs[tiitg] += shs[tiitg + st];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tiitg == 0u) {
        IDS[tgid] = (float)shi[0];
        LSE[tgid] = gmax > -INFINITY ? log(shs[0]) : -INFINITY;
    }
}

// Построчная копия fp32 со сдвигом/шагом: D[(dst_row0+r)*dst_stride + dst_col0 + c] =
// S[r*cols + c]. Сборка выходов extract-слоёв по блокам префилла и конкат
// фич драфтера (5 слоёв) — на GPU, без скачивания на хост.
kernel void copy_rows_f32(
    device const float* S [[buffer(0)]],
    device float* D [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& dst_row0 [[buffer(4)]],
    constant uint& dst_col0 [[buffer(5)]],
    constant uint& dst_stride [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cols || gid.y >= rows) return;
    D[((ulong)(dst_row0 + gid.y) * dst_stride) + dst_col0 + gid.x] = S[(ulong)gid.y * cols + gid.x];
}

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

// ── f32 → f16 (приведение Q к типу f16-кэша; Ф8.2f) ────────────────────────
kernel void cast_f32_f16(
    device const float* SRC [[buffer(0)]],
    device half* DST [[buffer(1)]],
    constant uint& N4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < N4) ((device half4*)DST)[gid] = half4(((device const float4*)SRC)[gid]);
}

// ── KV-cache update: dst (B, cap, D)[b, pos+t, d] = src (B, Tq, D)[b, t, d] ─
// Запись в f16-кэш (Ф8.2f): значения квантизуются на записи (как в llama.cpp,
// где KV по умолчанию f16) — вдвое меньше памяти под контекст и вдвое меньше
// трафика внимания.
kernel void kv_update_f16_f32(
    device half* dst [[buffer(0)]],
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
    dst[(b * cap + pos + t) * D + d] = (half)src[(b * Tq + t) * D + d];
}

// Копия half→half (рост ёмкости f16-кэша, Ф8.2f).
kernel void kv_copy_f16(
    device half* dst [[buffer(0)]],
    device const half* src [[buffer(1)]],
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
    uint kv_lim = Tkv;
    // scale = 1/sqrt(HD) — НЕ хардкод: Qwen3 hd=128, аудио-энкодер hd=64.
    const float scale = 1.0f / sqrt((float)HD);
    float mx = -INFINITY;
    for (uint j = 0u; j < kv_lim; ++j) {
        const device float* krow = K + (b * Tkv + j) * KVD + kvh * HD;
        float acc = 0.0f;
        for (uint d = 0u; d < HD; ++d) acc += qrow[d] * krow[d];
        float m = (MASK != nullptr) ? MASK[(b * Tq + t) * Tkv + j] : 0.0f;
        float s = acc * scale + m;
        if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (uint j = 0u; j < kv_lim; ++j) {
        const device float* krow = K + (b * Tkv + j) * KVD + kvh * HD;
        float acc = 0.0f;
        for (uint d = 0u; d < HD; ++d) acc += qrow[d] * krow[d];
        float m = (MASK != nullptr) ? MASK[(b * Tq + t) * Tkv + j] : 0.0f;
        float s = acc * scale + m;
        sum += exp(s - mx);
    }
    device float* yrow = Y + (b * Tq + t) * D + h * HD;
    for (uint d = 0u; d < HD; ++d) yrow[d] = 0.0f;
    float inv = 1.0f / sum;
    for (uint j = 0u; j < kv_lim; ++j) {
        const device float* krow = K + (b * Tkv + j) * KVD + kvh * HD;
        const device float* vrow = V + (b * Tkv + j) * KVD + kvh * HD;
        float qk = 0.0f;
        for (uint dd = 0u; dd < HD; ++dd) qk += qrow[dd] * krow[dd];
        float m = (MASK != nullptr) ? MASK[(b * Tq + t) * Tkv + j] : 0.0f;
        float s = qk * scale + m;
        float w = exp(s - mx) * inv;
        for (uint d = 0u; d < HD; ++d) yrow[d] += w * vrow[d];
    }
}

// GQA attention с float4-векторизацией d-цикла (три прохода как в
// attn_ar_f32, но dot и v-взвешивание — float4). Трёхпроходная версия на
// decode-шаге читала KV с DRAM-латентностью серийно в 16 потоках — ~1мс на
// позицию KV (decode деградировал с ростом контекста); float4-чтения дают
// ~40×. Маска -inf обрабатывается как в оригинале (exp(-inf - mx) = 0).
kernel void attn_ar_f32_online(
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
    const float scale = 1.0f / sqrt((float)HD);
    const uint n4 = HD / 4u;
    const device float4* qrow4 = (const device float4*)(Q + (b * Tq + t) * D + h * HD);
    const device float* maskrow = (MASK != nullptr) ? MASK + (b * Tq + t) * Tkv : nullptr;

    float mx = -INFINITY;
    for (uint j = 0u; j < Tkv; ++j) {
        const device float4* krow4 = (const device float4*)(K + (b * Tkv + j) * KVD + kvh * HD);
        float acc = 0.0f;
        for (uint d4 = 0u; d4 < n4; ++d4) {
            float4 q4 = qrow4[d4];
            float4 k4 = krow4[d4];
            acc += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        }
        float maskv = (maskrow != nullptr) ? maskrow[j] : 0.0f;
        float s = acc * scale + maskv;
        if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (uint j = 0u; j < Tkv; ++j) {
        const device float4* krow4 = (const device float4*)(K + (b * Tkv + j) * KVD + kvh * HD);
        float acc = 0.0f;
        for (uint d4 = 0u; d4 < n4; ++d4) {
            float4 q4 = qrow4[d4];
            float4 k4 = krow4[d4];
            acc += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        }
        float maskv = (maskrow != nullptr) ? maskrow[j] : 0.0f;
        float s = acc * scale + maskv;
        sum += exp(s - mx);
    }
    float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    device float4* yrow4 = (device float4*)(Y + (b * Tq + t) * D + h * HD);
    for (uint d4 = 0u; d4 < n4; ++d4) yrow4[d4] = 0.0f;
    for (uint j = 0u; j < Tkv; ++j) {
        const device float4* krow4 = (const device float4*)(K + (b * Tkv + j) * KVD + kvh * HD);
        const device float4* vrow4 = (const device float4*)(V + (b * Tkv + j) * KVD + kvh * HD);
        float qk = 0.0f;
        for (uint d4 = 0u; d4 < n4; ++d4) {
            float4 q4 = qrow4[d4];
            float4 k4 = krow4[d4];
            qk += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        }
        float maskv = (maskrow != nullptr) ? maskrow[j] : 0.0f;
        float s = qk * scale + maskv;
        float w = exp(s - mx) * inv;
        for (uint d4 = 0u; d4 < n4; ++d4) {
            yrow4[d4] += w * vrow4[d4];
        }
    }
}

// GQA attention, полностью распараллеленный по (h, d4): grid.y = H*n4 —
// 512 потоков на (b,t) вместо 16. Каждый поток держит СВОЙ float4 q/k/v;
// полный score получается simd_sum-редукцией по d4-лейнам (n4=32 = один
// simdgroup на голову — broadcast без барьеров). KV читается ОДИН раз
// (не 3 прохода × 16 потоков): ~75× быстрее attn_ar_f32 на decode-шаге.
kernel void attn_ar_f32_fused(
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
    uint b = gid.x;
    uint n4 = HD / 4u;
    // ВСЕГДА 32 лейна на голову: при n4 < 32 (hd=64 → n4=16) неактивные
    // лейны (lane >= n4) дают нулевой acc — simd_sum по 32 лейнам корректен;
    // их y4-запись дублирует активную (одинаковые значения — безопасно).
    // При n4 > 32 (hd=256, Qwen3.5/3.8) лейн ведёт НЕСКОЛЬКО float4-срезов
    // (d4 = lane + 32·c): раньше лейн брал только d4 = lane % n4, и головы
    // шире 128 теряли половину измерений в скоре и в выходе.
    uint h = gid.y / 32u;
    uint lane = gid.y % 32u;
    uint t = gid.z;
    if (b >= B || h >= H || t >= Tq) return;
    uint kvh = h / n_rep;
    uint D = H * HD;
    uint KVD = KVH * HD;
    const float scale = 1.0f / sqrt((float)HD);
    uint nc = (n4 + 31u) / 32u; // float4-срезов на лейн (1 для hd ≤ 128)
    const uint MAXC = 4u;       // hd ≤ 512
    const device float4* qrow4 = (const device float4*)(Q + (b * Tq + t) * D + h * HD);
    float4 q4[MAXC];
    float4 y4[MAXC];
    for (uint c = 0u; c < MAXC; ++c) {
        uint d4 = lane + 32u * c;
        q4[c] = (c < nc && d4 < n4) ? qrow4[d4] : float4(0.0f);
        y4[c] = float4(0.0f);
    }
    const device float* maskrow = (MASK != nullptr) ? MASK + (b * Tq + t) * Tkv : nullptr;

    // online-softmax в ОДНОМ проходе: running max m, ssum, y4-накопление.
    // -inf-маска: skip (continue) ДО exp — нет NaN (exp(-inf - -inf)).
    float m = -INFINITY;
    float ssum_local = 0.0f;
    for (uint j = 0u; j < Tkv; ++j) {
        const device float4* krow4 = (const device float4*)(K + (b * Tkv + j) * KVD + kvh * HD);
        float acc = 0.0f;
        for (uint c = 0u; c < nc; ++c) {
            uint d4 = lane + 32u * c;
            if (d4 < n4) {
                float4 k4 = krow4[d4];
                acc += q4[c].x * k4.x + q4[c].y * k4.y + q4[c].z * k4.z + q4[c].w * k4.w;
            }
        }
        float s = simd_sum(acc) * scale;
        if (maskrow != nullptr) s += maskrow[j];
        if (s == -INFINITY) continue;
        if (s > m) {
            float e = exp(m - s); // < 1: пересчёт старых вкладов
            for (uint c = 0u; c < nc; ++c) y4[c] *= e;
            ssum_local *= e;
            m = s;
        }
        float w = exp(s - m);
        ssum_local += w;
        const device float4* vrow4 = (const device float4*)(V + (b * Tkv + j) * KVD + kvh * HD);
        for (uint c = 0u; c < nc; ++c) {
            uint d4 = lane + 32u * c;
            if (d4 < n4) y4[c] += w * vrow4[d4];
        }
    }
    // ssum_local у каждого лейна УЖЕ полный (w — broadcast из simd_sum) —
    // повторная simd_sum утроила бы его в 32 раза.
    float inv = (ssum_local > 0.0f) ? 1.0f / ssum_local : 0.0f;
    device float4* yrow4 = (device float4*)(Y + (b * Tq + t) * D + h * HD);
    if (n4 < 32u) {
        // узкая голова: неактивные лейны дублируют d4 = lane % n4 (как раньше)
        uint d4 = lane % n4;
        if (lane < n4) yrow4[d4] = y4[0] * inv;
    } else {
        for (uint c = 0u; c < nc; ++c) {
            uint d4 = lane + 32u * c;
            if (d4 < n4) yrow4[d4] = y4[c] * inv;
        }
    }
}

// ── Deepstack (Qwen3-VL): y[t] = x[t] + embd[t+shift] (сдвиг — вклад входных
// эмбеддингов в первые n_deepstack слоёв); t+shift за границей → без вклада.
kernel void deepstack_add_f32(
    device const float* X [[buffer(0)]],
    device const float* E [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& shift [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x, d = gid.y;
    if (t >= T || d >= D) return;
    const uint src = t + shift;
    Y[t * D + d] = X[t * D + d] + ((src < T) ? E[src * D + d] : 0.0f);
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
// Q4_K 144 B/256, Q6_K 210 B/256. Activations read directly from the block
// bytes; CPU reference: weights_map::dequant_f32 (tests).

static inline half load_half_unaligned(device const uchar* p) {
    // даже смещения блоков (18/34/144/210 кратны 2) — выравнивание ushort ок
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

// Canonical Q4_K (GGUF/llama.cpp block_q4_K): 144 B per 256-weight super-block —
// d (f16 @0..2), dmin (f16 @2..4), scales[12] (@4..16, the 6-bit
// get_scale_min_k4 packing of 8 sub-blocks × (scale, min)), qs[128]
// (@16..144, chunk-major). Dequant: w = d·sc·q − dmin·m.
//
// ВАЖНО: здесь была раскладка-«диалект» на 148 Б (qs на смещении 20) с
// формулой min + d·sc·(q−8). Она не совпадала с GGUF: проверка размера в
// from_quantized отвергала КОРРЕКТНЫЕ Q4_K-тензоры (144 Б), а при обходе
// проверки веса читались бы мусором. CUDA-путь и weights_map уже перешли на
// канонические 144 Б — Metal остался позади. Формула и распаковка нибблов
// теперь совпадают с q4kc_weight (kernels.cu) и ggml dequantize_row_q4_K.
static inline void q4k_scale_min(
    device const uchar* scales, uint b, thread int& sc, thread int& mn)
{
    if (b < 4) {
        sc = scales[b] & 63;
        mn = scales[4 + b] & 63;
    } else {
        sc = (scales[b + 4] & 0xF) | ((scales[b - 4] >> 6) << 4);
        mn = (scales[b + 4] >> 4) | ((scales[b] >> 6) << 4);
    }
}

/// Скалярное произведение строки активаций на один Q4_K-суперблок.
static inline float dot_q4_k(device const float* a, device const uchar* blk) {
    float d = (float)load_half_unaligned(blk);
    float dmin = (float)load_half_unaligned(blk + 2);
    device const uchar* qs = blk + 16;
    float acc = 0.0f;
    for (uint b = 0; b < 8; ++b) {
        int sc, mn;
        q4k_scale_min(blk + 4, b, sc, mn);
        float dd = d * (float)sc;
        float mm = dmin * (float)mn;
        // chunk-major: байт qs[32·(b/2) + l] держит w[64·(b/2) + l] в младшем
        // ниббле и w[64·(b/2) + 32 + l] — в старшем.
        device const uchar* qb = qs + 32 * (b >> 1);
        uint base = b * 32;
        if ((b & 1) == 0) {
            for (uint l = 0; l < 32; ++l) {
                acc += a[base + l] * (dd * (float)(qb[l] & 0xF) - mm);
            }
        } else {
            for (uint l = 0; l < 32; ++l) {
                acc += a[base + l] * (dd * (float)(qb[l] >> 4) - mm);
            }
        }
    }
    return acc;
}

// Q6_K (канонический ggml): вклад суперблока 256 в скалярное произведение
// с a[0..256). Раскладка — как в dequantize_row_q6_K: две половины по 128,
// в каждой l = 0..31 даёт 4 элемента (l, l+32, l+64, l+96): низкие нибблы ql,
// старшие 2 бита из qh (по 2 бита на элемент, сдвиги 0/2/4/6), масштаб —
// sch[l/16 + {0,2,4,6}] (int8, 16 элементов на подблок).
static inline float dot_q6_k(device const float* a, device const uchar* s) {
    float d = (float)load_half_unaligned(s + 208);
    device const uchar* ql = s;
    device const uchar* qh = s + 128;
    device const char* sc = (device const char*)(s + 192);
    float acc = 0.0f;
    for (int h2 = 0; h2 < 2; ++h2) {
        device const uchar* qlh = ql + h2 * 64;
        device const uchar* qhh = qh + h2 * 32;
        device const char* sch = sc + h2 * 8;
        int base = h2 * 128;
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            float s1 = (float)sch[is + 0];
            float s2 = (float)sch[is + 2];
            float s3 = (float)sch[is + 4];
            float s4 = (float)sch[is + 6];
            int q1 = (int)((qlh[l + 0] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qlh[l + 0] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
            acc += a[base + l + 0] * d * s1 * (float)q1;
            acc += a[base + l + 32] * d * s2 * (float)q2;
            acc += a[base + l + 64] * d * s3 * (float)q3;
            acc += a[base + l + 96] * d * s4 * (float)q4;
        }
    }
    return acc;
}

// Q5_0 (канонический ggml, 22 Б/32): d(f16) @0, qh[4] (u32 LE, 5-е биты
// элементов j и j+16) @2, qs[16] @6. Парность нибблов — j (младший) и j+16
// (старший); y = ((q | 5-й бит << 4) − 16)·d.
// ── IQ4_XS (i-квант, 136 Б/256): кодбук kvalues_iq4nl ───────────────────────
// Порт ggml `kvalues_iq4nl` и `dequantize_row_iq4_xs`: масштаб подблока —
// 6 бит (ls), реальный множитель d·(ls − 32) (знак возможен!), ниббл — индекс
// в нелинейном кодбуке. Точность чисел важна: значения таблицы совпадают с
// ggml бит-в-бит — иначе паритет с llama.cpp разъедется.
constant float KV_IQ4[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f,
};

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

typedef decltype(mv_iq4_xs_mb<1>) mv_iq4_xs_mb_t;
template [[host_name("mv_iq4_xs_mb1")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<1>;
template [[host_name("mv_iq4_xs_mb2")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<2>;
template [[host_name("mv_iq4_xs_mb3")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<3>;
template [[host_name("mv_iq4_xs_mb4")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<4>;
template [[host_name("mv_iq4_xs_mb5")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<5>;
template [[host_name("mv_iq4_xs_mb6")]] kernel mv_iq4_xs_mb_t mv_iq4_xs_mb<6>;

static inline float dot_q5_0(device const float* a, device const uchar* blk) {
    float d = (float)load_half_unaligned(blk);
    uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
    device const uchar* qs = blk + 6;
    float acc = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        uint lo = (uint)(qs[j] & 0xF) + (((qh >> j) & 1u) << 4);
        uint hi = (uint)(qs[j] >> 4) + (((qh >> (j + 16)) & 1u) << 4);
        acc += a[j] * (((float)lo - 16.0f) * d);
        acc += a[j + 16] * (((float)hi - 16.0f) * d);
    }
    return acc;
}

// Q5_K (канонический ggml, 176 Б/256): d @0, dmin @2, scales[12] @4,
// qh[32] @16 (5-е биты; для чанка c маски m1 = 1<<2c, m2 = 2<<2c),
// qs[128] @48 (низкие 4 бита, чанк-мажор). y = d·sc·q − dmin·m, q ∈ 0..31.
static inline float dot_q5_k(device const float* a, device const uchar* blk) {
    float d = (float)load_half_unaligned(blk);
    float dmin = (float)load_half_unaligned(blk + 2);
    device const uchar* qh = blk + 16;
    device const uchar* qs = blk + 48;
    float acc = 0.0f;
    uint u1 = 1u, u2 = 2u;
    for (uint c = 0; c < 4; ++c) {
        int sc, mn;
        q4k_scale_min(blk + 4, 2 * c + 0, sc, mn);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)mn;
        q4k_scale_min(blk + 4, 2 * c + 1, sc, mn);
        float d2 = d * (float)sc;
        float m2 = dmin * (float)mn;
        uint base = c * 64;
        for (uint l = 0; l < 32; ++l) {
            uchar b = qs[c * 32 + l];
            float lo = (float)((uint)(b & 0xF) + ((qh[l] & u1) ? 16u : 0u));
            float hi = (float)((uint)(b >> 4) + ((qh[l] & u2) ? 16u : 0u));
            acc += a[base + l] * (d1 * lo - m1);
            acc += a[base + 32 + l] * (d2 * hi - m2);
        }
        u1 <<= 2;
        u2 <<= 2;
    }
    return acc;
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

    const uint tid = lane / 2u;   // 0..15 — позиция внутри строки
    const uint ix  = lane % 2u;   // чётность суперблоков
    const uint ip  = tid / 8u;    // половина суперблока
    const uint il  = tid % 8u;
    const uint l0  = 4u * il;
    const uint is  = 8u * ip + l0 / 16u;
    const uint y_off   = 128u * ip + l0;
    const uint q_off_l =  64u * ip + l0;
    const uint q_off_h =  32u * ip + l0;

    device const uchar* x = B + (ulong)first_row * row_bytes;
    float sumf[2] = {0.f, 0.f};
    for (uint i = ix; i < nb; i += 2u) {
        device const float* y = A + (ulong)i * 256u + y_off;
        float yl[16];
        for (uint l = 0; l < 4u; ++l) {
            yl[4u*l + 0] = y[l +  0];
            yl[4u*l + 1] = y[l + 32];
            yl[4u*l + 2] = y[l + 64];
            yl[4u*l + 3] = y[l + 96];
        }
        for (uint rr = 0u; rr < (two ? 2u : 1u); ++rr) {
            device const uchar* xb = x + (ulong)rr * row_bytes + (ulong)i * 210u;
            device const uchar* q1 = xb + q_off_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = xb + 128u + q_off_h;
            device const char*  sc = (device const char*)(xb + 192u) + is;
            const float dh = (float)load_half_unaligned(xb + 208u);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (uint l = 0; l < 4u; ++l) {
                sums[0] += yl[4u*l + 0] * (float)((int8_t)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4u*l + 1] * (float)((int8_t)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4u*l + 2] * (float)((int8_t)((q1[l] >> 4) | ((qh[l] & 0x30)         )) - 32);
                sums[3] += yl[4u*l + 3] * (float)((int8_t)((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)   ) - 32);
            }
            sumf[rr] += dh * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
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

// ── YaRN/линейное скалирование RoPE ─────────────────────────────────────────
// Зеркало ggml rope_yarn/ggml_rope_cache_init (см. src/rope.rs): угол пары k
// на позиции p строится МУЛЬТИПЛИКАТИВНЫМ накоплением theta (p·ts^k — тот же
// порядок перемножений, что у эталона), затем интерполяция частот по коридору
// corr_dims и масштаб cos/sin на mscale. Тождественные параметры
// (ext=0, freq_scale=1, attn=1) — прежний путь pow(theta, 2k/hd).
struct RopeYarn {
    float freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, theta_scale;
};
inline bool rope_yarn_identity(constant RopeYarn& y) {
    return y.ext_factor == 0.0f && y.freq_scale == 1.0f && y.attn_factor == 1.0f;
}
inline void rope_cs(float p, uint k, float theta, uint hd, constant RopeYarn& y,
                    thread float& c, thread float& s) {
    if (rope_yarn_identity(y)) {
        const float fr = 1.0f / pow(theta, 2.0f * (float)k / (float)hd);
        c = cos(p * fr);
        s = sin(p * fr);
        return;
    }
    float theta_acc = p;
    for (uint j = 0u; j < k; ++j) theta_acc *= y.theta_scale;
    const float theta_interp = y.freq_scale * theta_acc;
    float ang = theta_interp;
    float ms = y.attn_factor;
    if (y.ext_factor != 0.0f) {
        const float yy = ((float)k - y.corr_lo) / max(0.001f, y.corr_hi - y.corr_lo);
        const float ramp_mix = (1.0f - min(1.0f, max(0.0f, yy))) * y.ext_factor;
        ang = theta_interp * (1.0f - ramp_mix) + theta_acc * ramp_mix;
        ms *= 1.0f + 0.1f * log(1.0f / y.freq_scale);
    }
    c = cos(ang) * ms;
    s = sin(ang) * ms;
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
    constant RopeYarn& yarn [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, i = gid.z;
    if (t >= tq || h >= D / HD || i >= HD / 2) return;
    float p = (float)(pos + t);
    float c, s;
    rope_cs(p, i, theta, HD, yarn, c, s);
    // NEOX/halves (HF rotate-half): пары (i, i + HD/2)
    uint base = t * D + h * HD + i;
    float x0 = X[base];
    float x1 = X[base + HD / 2];
    Y[base] = x0 * c - x1 * s;
    Y[base + HD / 2] = x1 * c + x0 * s;
}

// Частичный RoPE: вращаются первые NROT компонент каждой головы (частоты
// theta^(-2k/NROT), k — индекс пары), хвост [NROT, HD) копируется.
// PAIRING: 0 — halves (k, k+NROT/2; llama.cpp NEOX, HF rotate_half),
//          1 — adjacent (2k, 2k+1; llama.cpp ROPE_TYPE_NORM — GLM-4/chatglm).
// NROT == HD и PAIRING == 0 — полный поворот (быстрее старый rope_std_f32:
// сетка вдвое меньше).
kernel void rope_std_part_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& tq [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& HD [[buffer(4)]],
    constant uint& NROT [[buffer(5)]],
    constant uint& pos [[buffer(6)]],
    constant float& theta [[buffer(7)]],
    constant uint& PAIRING [[buffer(8)]],
    constant RopeYarn& yarn [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, j = gid.z;
    if (t >= tq || j >= HD) return;
    uint base = t * D + h * HD;
    float v;
    // Частотная база — NROT (не HD): у partial rope пары нумеруются по
    // вращаемому префиксу; yarn-коридор corr_dims тоже в этих индексах.
    if (j >= NROT) {
        v = X[base + j];                       // хвост без вращения
    } else if (PAIRING == 1u) {
        uint k = j >> 1u;                      // пара (2k, 2k+1): соседние
        float p = (float)(pos + t);
        float c, s;
        rope_cs(p, k, theta, NROT, yarn, c, s);
        float x0 = X[base + (k << 1u)];
        float x1 = X[base + (k << 1u) + 1u];
        v = (j & 1u) == 0u ? x0 * c - x1 * s : x0 * s + x1 * c;
    } else {
        uint half_rot = NROT / 2u;             // пара (k, k+NROT/2): половины
        float p = (float)(pos + t);
        if (j < half_rot) {
            float x0 = X[base + j];
            float x1 = X[base + j + half_rot];
            float c, s;
            rope_cs(p, j, theta, NROT, yarn, c, s);
            v = x0 * c - x1 * s;
        } else {
            uint k = j - half_rot;             // частота берётся по k
            float x0 = X[base + k];
            float x1 = X[base + j];
            float c, s;
            rope_cs(p, k, theta, NROT, yarn, c, s);
            v = x0 * s + x1 * c;
        }
    }
    Y[base + j] = v;
}

// ── Embedding with vision-token override (Ф1.11) ───────────────────────────
// Токены id == override_id заменяются строками из VISION (по порядку);
// остальные — обычный lookup. Y: (T, D); VISION: (V, D).
kernel void embedding_override_f32(
    device const uint* idx [[buffer(0)]],
    device const float* table [[buffer(1)]],
    device const float* vision [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& override_id [[buffer(7)]],
    constant uint& V [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.x, t = gid.y, d = gid.z;
    if (b >= 1u || t >= T || d >= D) return;
    uint row = idx[b * T + t];
    if (row == override_id) {
        // порядковый номер vision-токена среди встреченных
        uint vi = 0;
        for (uint tt = 0u; tt < t; ++tt) {
            if (idx[b * T + tt] == override_id) vi++;
        }
        if (vi < V) {
            Y[(b * T + t) * D + d] = vision[vi * D + d];
        } else {
            Y[(b * T + t) * D + d] = 0.0f;
        }
    } else {
        if (row >= N) row = N - 1u;
        Y[(b * T + t) * D + d] = table[row * D + d];
    }
}

// ── Interleaved RoPE (Qwen2/Qwen3-VL IMROPE text): пары (2i, 2i+1) ────────
kernel void rope_mrope_text_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& tq [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& HD [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant uint4& sections [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, i = gid.z;
    if (t >= tq || h >= D / HD || i >= HD / 2) return;
    // M-RoPE text: halves-пары; e-секция (4-я позиция = 0) не ротируется.
    uint sect_dims = sections.x + sections.y + sections.z + sections.w;
    if (sect_dims == 0u) sect_dims = 1u;
    uint sector = i % sect_dims;
    bool is_e = !((sector % 3u == 0u && sector < 3u * sections.x)
               || (sector % 3u == 1u && sector < 3u * sections.y)
               || (sector % 3u == 2u && sector < 3u * sections.z));
    uint base = t * D + h * HD + i;
    uint hhalf = HD / 2;
    if (is_e) {
        Y[base] = X[base];
        Y[base + hhalf] = X[base + hhalf];
        return;
    }
    float p = (float)(pos + t);
    float inv = 1.0 / pow(theta, 2.0 * (float)i / (float)HD);
    float c = cos(p * inv);
    float s = sin(p * inv);
    float x0 = X[base];
    float x1 = X[base + hhalf];
    Y[base] = x0 * c - x1 * s;
    Y[base + hhalf] = x0 * s + x1 * c;
}

// ── M-RoPE per-token 4D (Qwen3-VL IMROPE, картинки): как rope_mrope_text,
// но позиция на токен из буфера POS4D [tq*4] = (t,h,w,e); e-категория не
// ротируется. Позиции в llama.cpp mtmd для картинки: t=pos_0, h=pos_0+i/nx,
// w=pos_0+i%nx, e=0; для текста t=h=w=pos.
kernel void rope_mrope_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& tq [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& HD [[buffer(4)]],
    device const uint* POS4D [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant uint4& sections [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, i = gid.z;
    if (t >= tq || h >= D / HD || i >= HD / 2) return;
    uint sect_dims = sections.x + sections.y + sections.z + sections.w;
    if (sect_dims == 0u) sect_dims = 1u;
    uint sector = i % sect_dims;
    uint pos = 0u;
    bool is_e;
    if (sector % 3u == 0u && sector < 3u * sections.x) {
        pos = POS4D[t * 4u + 0u]; is_e = false;
    } else if (sector % 3u == 1u && sector < 3u * sections.y) {
        pos = POS4D[t * 4u + 1u]; is_e = false;
    } else if (sector % 3u == 2u && sector < 3u * sections.z) {
        pos = POS4D[t * 4u + 2u]; is_e = false;
    } else {
        is_e = true;
    }
    uint base = t * D + h * HD + i;
    uint hhalf = HD / 2;
    if (is_e) {
        Y[base] = X[base];
        Y[base + hhalf] = X[base + hhalf];
        return;
    }
    float p = (float)pos;
    float inv = 1.0 / pow(theta, 2.0 * (float)i / (float)HD);
    float c = cos(p * inv);
    float s = sin(p * inv);
    float x0 = X[base];
    float x1 = X[base + hhalf];
    Y[base] = x0 * c - x1 * s;
    Y[base + hhalf] = x0 * s + x1 * c;
}

// ── GELU (erf, A&S 7.1.26) — vision (clip/Qwen3-VL) ────────────────────────
kernel void gelu_erf_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint i = gid;
    if (i >= n) return;
    float x = X[i];
    float y = x * 0.70710678118654752440f; // x / sqrt(2)
    float ax = fabs(y);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = 0.254829592f * t
               - 0.284496736f * t * t
               + 1.421413741f * t * t * t
               - 1.453152027f * t * t * t * t
               + 1.061405429f * t * t * t * t * t;
    float erf = (y < 0.0f ? -1.0f : 1.0f) * (1.0f - poly * exp(-ax * ax));
    Y[i] = 0.5f * x * (1.0f + erf);
}

// ── Qwen3-VL vision layout ops (all pure GPU, no host round-trips) ─────────
// merge 2×2 + patch_bias + position_embd: conv grid (per×per) → same grid,
// each output block (by,bx) takes 4 neighbour patches in (c0a0,c1a0,c0a1,c1a1)
// order; pos_embd is permuted identically (same src row).
// X/POS: [per², d] patch rows; BIAS: [d]; Y: [per², d].
kernel void merge2x2_add_f32(
    device const float* X [[buffer(0)]],
    device const float* BIAS [[buffer(1)]],
    device const float* POS [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& per [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint blk = gid.x, j = gid.y; // j = token-in-block 0..3
    if (blk >= per * per / 4 || j >= 4) return;
    uint hp = per / 2;
    uint by = blk / hp, bx = blk % hp;
    // src patch row for this position in the block
    uint sy = 2 * by + (j / 2); // j: 0,1 → a0; 2,3 → a1
    uint sx = 2 * bx + (j % 2); // j: 0,2 → c0; 1,3 → c1
    uint src = (sy * per + sx) * d;
    uint dst = (blk * 4 + j) * d;
    for (uint k = 0u; k < d; ++k) {
        Y[dst + k] = X[src + k] + BIAS[k] + POS[src + k];
    }
}

// group4: [n, d] → [n/4, 4·d], 4 consecutive rows concatenated in order.
// X: [n, d]; Y: [n/4, 4d]; out[i*4d + j*d + k] = X[(i*4+j)*d + k].
kernel void group4_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = (n / 4) * 4 * d;
    if (gid >= total) return;
    uint k = gid % d;
    uint rem = gid / d;
    uint j = rem % 4;
    uint i = rem / 4;
    Y[i * 4 * d + j * d + k] = X[(i * 4 + j) * d + k];
}

// concat 4 chunks [nq, proj] → [4·nq, proj] row-interleaved:
// Y[(q*4+j)*proj + p] = chunk_j[q*proj + p]. Y: [4·nq, proj].
kernel void concat4_f32(
    device const float* C0 [[buffer(0)]],
    device const float* C1 [[buffer(1)]],
    device const float* C2 [[buffer(2)]],
    device const float* C3 [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant uint& nq [[buffer(5)]],
    constant uint& proj [[buffer(6)]],
    constant uint& nchunks [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= nq * 4 * proj) return;
    uint p = gid % proj;
    uint rem = gid / proj;
    uint j = rem % 4;
    uint q = rem / 4;
    if (j >= nchunks) return;
    device const float* C = (j == 0) ? C0 : (j == 1) ? C1 : (j == 2) ? C2 : C3;
    Y[gid] = C[q * proj + p];
}


// ── DFlash2 two-tap dynamic conv (Ф2.3, спека PR #27342) ────────────────────
// X [n_tokens, n_embd] (скрытые), BASE [2, kernel, n_embd] (ggml-порядок:
// side·k·D + tap·D + i), DYN [n_tokens, n_groups·kernel·2] (row-major:
// t·proj + r, r = g + n_groups·(tap + k·side)), Y [n_tokens, n_embd].
// y[t,i] = Σ_tap (DYN[t, r(g,tap,side)] + BASE[side,tap,i]) · x[t−tap]
// сдвиг — ВНУТРИ блока (block_size): pos−tap < 0 → 0.
kernel void dflash2_conv_f32(
    device const float* X [[buffer(0)]],
    device const float* BASE [[buffer(1)]],
    device const float* DYN [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& n_tokens [[buffer(4)]],
    constant uint& n_embd [[buffer(5)]],
    constant uint& kern [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    constant uint& block_size [[buffer(8)]],
    constant uint& n_groups [[buffer(9)]],
    constant uint& side [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x, i = gid.y;
    if (t >= n_tokens || i >= n_embd) return;
    uint blk = t / block_size;
    uint pos = t % block_size;
    uint g = i / group_size;
    uint proj = n_groups * kern * 2u;
    float acc = 0.0f;
    for (uint tap = 0u; tap < kern; ++tap) {
        float v = 0.0f;
        if (pos >= tap) {
            uint src = blk * block_size + (pos - tap);
            v = X[src * n_embd + i];
        }
        // ggml coeffs [n_groups, kern, 2, T] в row-major DYN [T, proj]:
        // r = g + n_groups·(tap + kern·side)
        uint r = g + n_groups * (tap + kern * side);
        float coeff = DYN[t * proj + r];
        float b = BASE[side * kern * n_embd + tap * n_embd + i];
        acc += (coeff + b) * v;
    }
    Y[t * n_embd + i] = acc;
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



// debug: barrier-проверка: поток 0 пишет в shared, все читают
kernel void tg_barrier_test(
    device float* C [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    threadgroup float v[32];
    uint tid = gid.z;
    if (tid == 0) { v[0] = 42.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    C[tid] = v[0];
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

// ── Fused per-head RMSNorm + rotate-half RoPE (Ф8) ─────────────────────────
// Y = rope(rmsnorm_head(X)): 4 кернела на слой (qk_norm×2 + rope×2) → 2.
// X (1, T, H·HD), W [HD]; поток (t, h, i) — пара (i, i+HD/2); позиция pos+t.
kernel void rmsnorm_head_rope_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& pos [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, i = gid.z;
    if (t >= T || h >= H || i >= HD / 2) return;
    const device float* row = X + t * (H * HD) + h * HD;
    float sum = 0.0f;
    for (uint dd = 0u; dd < HD; ++dd) sum += row[dd] * row[dd];
    float inv = rsqrt(sum / (float)HD + eps);
    float p = (float)(pos + t);
    float fr = 1.0 / pow(theta, 2.0 * (float)i / (float)HD);
    float c = cos(p * fr);
    float s = sin(p * fr);
    uint base = t * (H * HD) + h * HD + i;
    float x0 = row[i] * inv * W[i];
    float x1 = row[i + HD / 2] * inv * W[i + HD / 2];
    Y[base] = x0 * c - x1 * s;
    Y[base + HD / 2] = x1 * c + x0 * s;
}

// ═══════════════════ Ф9.4: Qwen3.5/3.8 (гибрид Gated DeltaNet) ═══════════════════
// Эталон — llama.cpp src/models/qwen35.cpp + delta-net-base.cpp и CPU-опы ggml
// (ssm_conv, gated_delta_net). Все кернелы — f32, данные row-major.

// Causal depthwise conv1d со стейтом (ggml_ssm_conv на concat[state, x]).
// X [T, C] — вход токенов; W [C, K] — ядро (строка канала подряд, как в GGUF
// ssm_conv1d [K, C] ggml-порядка); STATE [C, K-1] — хвост предыдущих токенов
// (слот j = токен t-(K-1)+j). Y[t, c] = Σ_j W[c, j] · in[t + j], где in —
// конкатенация [STATE (K-1) | X (T)] по времени канала c.
kernel void gdn_conv1d_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* STATE [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x, c = gid.y;
    if (t >= T || c >= C) return;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        // позиция в конкатенации: t + j; < K-1 → из стейта, иначе из X.
        uint p = t + j;
        float v = (p < K - 1) ? STATE[c * (K - 1) + p] : X[(p - (K - 1)) * C + c];
        acc += W[c * K + j] * v;
    }
    Y[t * C + c] = acc;
}

// Сдвиг conv-стейта после T токенов: новый STATE = последние K-1 позиций
// конкатенации [STATE | X]. Пишется в отдельный буфер (in-place пересекается).
kernel void gdn_conv_state_f32(
    device const float* X [[buffer(0)]],
    device const float* STATE [[buffer(1)]],
    device float* NEW [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint j = gid.x, c = gid.y;
    if (j >= K - 1 || c >= C) return;
    // позиция в конкатенации длиной (K-1)+T: хвост начинается с T.
    uint p = T + j;
    NEW[c * (K - 1) + j] = (p < K - 1) ? STATE[c * (K - 1) + p] : X[(p - (K - 1)) * C + c];
}

// L2-нормировка по голове (build_gdn_l2_norm = rms_norm(eps/n) · 1/sqrt(n)):
// y = x / sqrt(Σx² + eps). X/Y [rows, HD], по потоку на строку-голову.
kernel void gdn_l2norm_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& ROWS [[buffer(2)]],
    constant uint& HD [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= ROWS) return;
    device const float* x = X + r * HD;
    float ss = 0.0f;
    for (uint i = 0; i < HD; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrt(ss + eps);
    for (uint i = 0; i < HD; ++i) Y[r * HD + i] = x[i] * inv;
}

// Гейты GDN на токен и v-голову: g = softplus(alpha + dt_bias) · ssm_a,
// beta = sigmoid(beta_raw). ALPHA/BETA [T, H]; G/B — выходы [T, H].
kernel void gdn_gates_f32(
    device const float* ALPHA [[buffer(0)]],
    device const float* BETA [[buffer(1)]],
    device const float* DT_BIAS [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device float* G [[buffer(4)]],
    device float* B [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint h = i % H;
    float a = ALPHA[i] + DT_BIAS[h];
    // softplus с порогом (как ggml: при x > 20 — линейно)
    float sp = a > 20.0f ? a : log(1.0f + exp(a));
    G[i] = sp * A[h];
    B[i] = 1.0f / (1.0f + exp(-BETA[i]));
}

// Рекуррентный gated delta rule (ggml_gated_delta_net, K=1, GDA-скаляр g).
// Q/K [T, Hk, S], V [T, Hv, S], G/B [T, Hv]; STATE [Hv, S, S] хранится
// ТРАНСПОНИРОВАННЫМ как в ggml: s[j*S + i] = S_math[i][j]. Одна threadgroup
// на v-голову, поток на столбец j (S потоков, S ≤ 256). На каждый токен:
//   s *= exp(g); delta_j = (v_j − Σ_i s[j][i]·k_i)·beta; s[j][i] += k_i·delta_j;
//   out_j = (Σ_i s[j][i]·q_i) / sqrt(S).
// Голова q/k для v-головы h — h % Hk (ggml: iq1 = iv1 % neq1).
kernel void gdn_delta_rule_f32(
    device const float* Q [[buffer(0)]],
    device const float* Kx [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* G [[buffer(3)]],
    device const float* B [[buffer(4)]],
    device float* STATE [[buffer(5)]],
    device float* OUT [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& Hk [[buffer(8)]],
    constant uint& Hv [[buffer(9)]],
    constant uint& S [[buffer(10)]],
    uint h [[threadgroup_position_in_grid]],
    uint j [[thread_position_in_threadgroup]])
{
    if (h >= Hv || j >= S) return;
    device float* s = STATE + h * S * S + j * S; // строка j транспонированного стейта
    uint hk = h % Hk;
    float scale = 1.0f / sqrt((float)S);
    for (uint t = 0; t < T; ++t) {
        device const float* q = Q + (t * Hk + hk) * S;
        device const float* k = Kx + (t * Hk + hk) * S;
        float g = exp(G[t * Hv + h]);
        float beta = B[t * Hv + h];
        float kv = 0.0f;
        for (uint i = 0; i < S; ++i) {
            s[i] *= g;
            kv += s[i] * k[i];
        }
        float delta = (V[(t * Hv + h) * S + j] - kv) * beta;
        float o = 0.0f;
        for (uint i = 0; i < S; ++i) {
            s[i] += k[i] * delta;
            o += s[i] * q[i];
        }
        OUT[(t * Hv + h) * S + j] = o * scale;
    }
}

// Гейтовая RMS-норма GDN: y = rmsnorm(x)·w · silu(z), по голове HD.
// X/Z/Y [rows, HD], W [HD]; поток на строку-голову.
kernel void gdn_norm_gated_f32(
    device const float* X [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& ROWS [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= ROWS) return;
    device const float* x = X + r * HD;
    device const float* z = Z + r * HD;
    float ss = 0.0f;
    for (uint i = 0; i < HD; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrt(ss / (float)HD + eps);
    for (uint i = 0; i < HD; ++i) {
        float zz = z[i];
        float silu = zz / (1.0f + exp(-zz));
        Y[r * HD + i] = x[i] * inv * W[i] * silu;
    }
}

// IMROPE (Qwen3.5/3.8/3-VL, rope_type 40) для текста: позиции t=h=w=pos, e=0.
// Ротируются только первые N_ROT измерений головы (частичный rope), пары —
// NEOX (i, i + N_ROT/2). Частота пары i: theta^(-2i/N_ROT); секция пары
// (i % sect_dims) выбирает позицию (t/h/w — все = pos для текста, e = 0).
kernel void rope_imrope_text_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& tq [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& HD [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    constant uint4& sections [[buffer(7)]],
    constant uint& N_ROT [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint t = gid.x, h = gid.y, i = gid.z;
    uint heads = D / HD;
    if (t >= tq || h >= heads || i >= HD) return;
    uint base = t * D + h * HD;
    uint half_rot = N_ROT / 2;
    if (i >= N_ROT) {                    // хвост головы без ротации
        Y[base + i] = X[base + i];
        return;
    }
    if (i >= half_rot) return;           // пару пишет младший индекс
    uint sect_dims = sections.x + sections.y + sections.z + sections.w;
    if (sect_dims == 0u) sect_dims = 1u;
    uint sector = i % sect_dims;
    bool is_e = !((sector % 3u == 1u && sector < 3u * sections.y)
               || (sector % 3u == 2u && sector < 3u * sections.z)
               || (sector % 3u == 0u && sector < 3u * sections.x));
    float p = is_e ? 0.0f : (float)(pos + t);
    float inv = pow(theta, -2.0f * (float)i / (float)N_ROT);
    float c = cos(p * inv);
    float s = sin(p * inv);
    float x0 = X[base + i];
    float x1 = X[base + i + half_rot];
    Y[base + i] = x0 * c - x1 * s;
    Y[base + i + half_rot] = x0 * s + x1 * c;
}

// Эмбеддинги из таблицы в РОДНОМ формате GGUF (Ф9.4): строка idx[t]
// деквантуется на лету — у Qwen3.5 словарь 248K × 2560, в f32 таблица заняла
// бы 2.5 ГБ, а тот же Q8_0-тензор ещё и служит lm_head (tied). FMT: 0 = f16,
// 1 = Q8_0, 2 = Q4_0, 3 = Q4_K, 4 = Q6_K (раскладки — как в dequant_* из
// weights_map и в квант-GEMM выше). Y [T, D].
kernel void embedding_quant_f32(
    device const uint* idx [[buffer(0)]],
    device const uchar* table [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& FMT [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x, d = gid.y;
    if (t >= T || d >= D) return;
    uint row = idx[t];
    if (row >= N) row = N - 1u;
    float v = 0.0f;
    if (FMT == 0u) {
        device const half* p = (device const half*)table;
        v = (float)p[row * D + d];
    } else if (FMT == 1u) {
        device const uchar* blk = table + (row * (D / 32u) + d / 32u) * 34u;
        v = (float)((device const char*)(blk + 2))[d % 32u] * (float)load_half_unaligned(blk);
    } else if (FMT == 2u) {
        // канонический Q4_0: байт j — веса j (младший ниббл) и j+16 (старший)
        device const uchar* blk = table + (row * (D / 32u) + d / 32u) * 18u;
        uint j = d % 32u;
        uchar q = blk[2u + (j % 16u)];
        int nib = (j < 16u) ? (int)(q & 0xF) : (int)(q >> 4);
        v = ((float)nib - 8.0f) * (float)load_half_unaligned(blk);
    } else if (FMT == 3u) {
        device const uchar* blk = table + (row * (D / 256u) + d / 256u) * 144u;
        uint e = d % 256u;
        uint b = e / 32u;
        int sc, mn;
        q4k_scale_min(blk + 4, b, sc, mn);
        // чанк-мажор: байт qs[32·(e/64) + e%32], чётный подблок — младший ниббл
        uchar q = blk[16u + 32u * (e / 64u) + (e % 32u)];
        int nib = ((b & 1u) == 0u) ? (int)(q & 0xF) : (int)(q >> 4);
        v = (float)load_half_unaligned(blk) * (float)sc * (float)nib
            - (float)load_half_unaligned(blk + 2) * (float)mn;
    } else if (FMT == 5u) {
        // Q5_K: d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48
        device const uchar* s = table + (row * (D / 256u) + d / 256u) * 176u;
        uint e = d % 256u;
        uint b = e / 32u;              // подблок 0..7
        uint c = e / 64u;              // чанк 0..3
        uint j = e % 32u;              // индекс внутри 32
        int sc, mn;
        q4k_scale_min(s + 4, b, sc, mn);
        uchar qb = s[48u + c * 32u + j];
        uint bit = ((b & 1u) == 0u) ? (1u << (2u * c)) : (2u << (2u * c));
        uint half5 = ((uint)s[16u + j] & bit) ? 16u : 0u;
        uint q = (uint)((b & 1u) == 0u ? (qb & 0xF) : (qb >> 4)) + half5;
        v = (float)load_half_unaligned(s) * (float)sc * (float)q
            - (float)load_half_unaligned(s + 2) * (float)mn;
    } else if (FMT == 6u) {
        // Q5_0: d @0, qh[4] @2, qs[16] @6; парность j / j+16
        device const uchar* s = table + (row * (D / 32u) + d / 32u) * 22u;
        uint j = d % 32u;
        uint qh = (uint)s[2] | ((uint)s[3] << 8) | ((uint)s[4] << 16) | ((uint)s[5] << 24);
        uchar qb = s[6u + (j % 16u)];
        uint q = (uint)((j < 16u) ? (qb & 0xF) : (qb >> 4)) + (((qh >> j) & 1u) << 4);
        v = ((float)q - 16.0f) * (float)load_half_unaligned(s);
    } else if (FMT == 7u) {
        // IQ4_XS: d @0, scales_h @2, scales_l[4] @4, qs[128] @8; ниббл —
        // индекс кодбука, множитель подблока d·(ls − 32)
        device const uchar* s = table + (row * (D / 256u) + d / 256u) * 136u;
        uint e = d % 256u;
        uint ib = e / 32u, r = e % 32u;
        uint ls = (uint)((s[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0xF)
                | (((((uint)s[2] | ((uint)s[3] << 8)) >> (2u * ib)) & 3u) << 4);
        uchar qb = s[8u + ib * 16u + (r % 16u)];
        uint nib = (r < 16u) ? (uint)(qb & 0xF) : (uint)(qb >> 4);
        v = (float)load_half_unaligned(s) * ((float)(int)ls - 32.0f) * KV_IQ4[nib];
    } else {
        device const uchar* s = table + (row * (D / 256u) + d / 256u) * 210u;
        uint e = d % 256u;
        uint h2 = e / 128u, r = e % 128u, qq = r / 32u, l = r % 32u;
        device const uchar* qlh = s + h2 * 64u;
        uchar qlb = (qq == 0u || qq == 2u) ? qlh[l] : qlh[l + 32u];
        uint lo = (qq < 2u) ? (uint)(qlb & 0xF) : (uint)(qlb >> 4);
        uint hi = ((uint)s[128u + h2 * 32u + l] >> (2u * qq)) & 3u;
        float sc = (float)((device const char*)(s + 192u))[h2 * 8u + l / 16u + 2u * qq];
        v = (float)load_half_unaligned(s + 208) * sc * ((float)(int)(lo | (hi << 4)) - 32.0f);
    }
    Y[t * D + d] = v;
}

// ── Q8_0 matvec (decode m=1), simdgroup-кооперативный (Ф8.2) ────────────────
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

// ── RMSNorm строки, threadgroup-параллельный (Ф8.2) ──────────────────────────
// Прежний rmsnorm_f32 — ОДИН поток на строку (два последовательных прохода
// по D): decode-строка 4096 стоила ~0.6 мс, ×73 нормы на шаг Qwen3-8B —
// больше, чем все матвеки шага. Здесь threadgroup (256 потоков) на строку:
// частичные суммы квадратов → simd_sum → сумма по simdgroup'ам через
// threadgroup-память; нормирование — тем же threadgroup'ом.
kernel void rmsnorm_row_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    device const float* x = X + (ulong)row * D;
    float s = 0.0f;
    for (uint i = tid; i < D; i += tpg) s += x[i] * x[i];
    s = simd_sum(s);
    if (lane == 0u) partial[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsg = (tpg + 31u) / 32u;
    float total = 0.0f;
    for (uint g = 0u; g < nsg; ++g) total += partial[g];
    const float inv = rsqrt(total / (float)D + eps);
    device float* y = Y + (ulong)row * D;
    for (uint i = tid; i < D; i += tpg) y[i] = x[i] * inv * W[i];
}

// ── Per-head RMSNorm (+ RoPE), simdgroup на голову (Ф8.2) ───────────────────
// Прежние rmsnorm_head_f32 / rmsnorm_head_rope_f32 считали сумму квадратов
// головы ЗАНОВО в каждом потоке (HD потоков × HD чтений): O(HD²) на голову.
// Здесь один simdgroup (32 лейна) на (строку, голову): сумма — simd_sum,
// нормирование/ротация — лейны по измерениям.
kernel void rmsnorm_head_sg_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& NH [[buffer(3)]],     // всего голов (строки × H)
    constant uint& HD [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint head = gid / 32u, lane = gid % 32u;
    if (head >= NH) return;
    device const float* x = X + (ulong)head * HD;
    float s = 0.0f;
    for (uint i = lane; i < HD; i += 32u) s += x[i] * x[i];
    const float inv = rsqrt(simd_sum(s) / (float)HD + eps);
    device float* y = Y + (ulong)head * HD;
    for (uint i = lane; i < HD; i += 32u) y[i] = x[i] * inv * W[i];
}

kernel void rmsnorm_head_rope_sg_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant uint& H [[buffer(3)]],
    constant uint& NH [[buffer(4)]],     // T × H
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& pos [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    constant RopeYarn& yarn [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const uint head = gid / 32u, lane = gid % 32u;
    if (head >= NH) return;
    const uint t = head / H;
    device const float* x = X + (ulong)head * HD;
    float s = 0.0f;
    for (uint i = lane; i < HD; i += 32u) s += x[i] * x[i];
    const float inv = rsqrt(simd_sum(s) / (float)HD + eps);
    const float p = (float)(pos + t);
    const uint half_hd = HD / 2u;
    device float* y = Y + (ulong)head * HD;
    for (uint i = lane; i < half_hd; i += 32u) {
        float c, sn;
        rope_cs(p, i, theta, HD, yarn, c, sn);
        const float x0 = x[i] * inv * W[i];
        const float x1 = x[i + half_hd] * inv * W[i + half_hd];
        y[i] = x0 * c - x1 * sn;
        y[i + half_hd] = x1 * c + x0 * sn;
    }
}

// ── Q8_0 GEMM m>1 на матричных блоках simdgroup (Ф8.2, prefill) ──────────────
// C[M,N] = A[M,K] · dequant(B[N,K])ᵀ. Threadgroup (128 потоков = 4
// simdgroup'а) считает тайл 32(M)×64(N); K идёт шагами по 32 (= один
// Q8_0-блок): A-тайл 32×32 и деквантованный B-тайл 64×32 кладутся в
// threadgroup-память (half), затем каждый simdgroup умножает свои 8×8
// фрагменты через simdgroup_multiply_accumulate (аппаратные матричные
// блоки Apple GPU). Прежний m8f4-кернел считал скалярно по 8 строк на поток
// и упирался в ALU (prefill Qwen3-8B ~35 tok/s против 328 у эталона).
#include <metal_simdgroup_matrix>
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

// ── Gated delta rule, simdgroup-параллельный по i (Ф8.2) ─────────────────────
// Прежний gdn_delta_rule_f32: поток на столбец j, и КАЖДЫЙ поток проходил
// всю строку стейта (S=128) дважды на токен последовательно — на prefill
// 128 токенов ×24 слоя это 4.3 мс на слой. Здесь строка j стейта делится
// между 32 лейнами simdgroup'а (по S/32 = 4 элемента в регистрах лейна):
// скалярные произведения — simd_sum, обновление — поэлементно у себя.
// Threadgroup = 4 simdgroup'а → 4 строки j; grid — Hv·S/4 threadgroup'ов.
// Порядок операций на токен тот же, что у ggml gated_delta_net (s *= e^g;
// delta = (v − s·k)·β; s += k·delta; out = s·q / √S).
#define GDN_MAXE 8u   // S ≤ 256
kernel void gdn_delta_rule_sg_f32(
    device const float* Q [[buffer(0)]],
    device const float* Kx [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* G [[buffer(3)]],
    device const float* B [[buffer(4)]],
    device float* STATE [[buffer(5)]],
    device float* OUT [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& Hk [[buffer(8)]],
    constant uint& Hv [[buffer(9)]],
    constant uint& S [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint rowg = tgid * 4u + sgid;          // глобальная строка (h, j)
    const uint h = rowg / S, j = rowg % S;
    if (h >= Hv) return;
    const uint hk = h % Hk;
    const uint ne = S / 32u;                     // элементов строки на лейн
    device float* srow = STATE + ((ulong)h * S + j) * S;
    float s[GDN_MAXE];
    for (uint e = 0; e < ne; ++e) s[e] = srow[lane + e * 32u];
    const float scale = 1.0f / sqrt((float)S);
    for (uint t = 0; t < T; ++t) {
        device const float* q = Q + ((ulong)t * Hk + hk) * S;
        device const float* k = Kx + ((ulong)t * Hk + hk) * S;
        const float eg = exp(G[t * Hv + h]);
        const float beta = B[t * Hv + h];
        float kv = 0.0f;
        for (uint e = 0; e < ne; ++e) {
            s[e] *= eg;
            kv += s[e] * k[lane + e * 32u];
        }
        kv = simd_sum(kv);
        const float delta = (V[((ulong)t * Hv + h) * S + j] - kv) * beta;
        float o = 0.0f;
        for (uint e = 0; e < ne; ++e) {
            s[e] += k[lane + e * 32u] * delta;
            o += s[e] * q[lane + e * 32u];
        }
        o = simd_sum(o);
        if (lane == 0u) OUT[((ulong)t * Hv + h) * S + j] = o * scale;
    }
    for (uint e = 0; e < ne; ++e) srow[lane + e * 32u] = s[e];
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
    threadgroup float sh[MV8_NR0 * MV8_NSG];
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
        const float v = simd_sum(sumf[row]);
        if (lane == 0u) sh[row * MV8_NSG + sgid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u && lane < MV8_NR0 && r0 + lane < N) {
        float t = 0.0f;
        for (uint g = 0; g < MV8_NSG; ++g) t += sh[lane * MV8_NSG + g];
        C[r0 + lane] = t;
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

// ── Decode-attention (Tq=1) с разбиением ключей между simdgroup'ами (Ф8.2b) ──
// attn_ar_f32_fused ведёт голову ОДНИМ simdgroup'ом и проходит все Tkv ключей
// последовательно (simd_sum на каждом ключе): при 16 головах GPU занят 16
// simdgroup'ами — ~120 мкс на слой у Qwen3-0.6B (больше матвеков слоя).
// Здесь голова = threadgroup из NSG simdgroup'ов: simdgroup g берёт ключи
// j ≡ g (mod NSG), ведёт свои (max, sum, acc) online-softmax, в конце
// частичные результаты сливаются через threadgroup-память (как
// flash-decoding). HD ≤ 256, кратно 4; лейн держит HD/128 float4.
#define AD_NSG_MAX 32u
kernel void attn_decode_split_f32(
    device const float* Q [[buffer(0)]],     // [1, H*HD]
    device const float* K [[buffer(1)]],     // [Tkv, KVH*HD]
    device const float* V [[buffer(2)]],
    device float* Y [[buffer(3)]],           // [1, H*HD]
    constant uint& Tkv [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant uint& KVH [[buffer(6)]],
    constant uint& HD [[buffer(7)]],
    constant uint& NSG [[buffer(8)]],           // число разбиений ключей
    threadgroup float* sh [[threadgroup(0)]],   // NSG × (2 + HD)
    uint h [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    if (h >= H) return;
    const uint kvh = h / (H / KVH);
    const uint KD = KVH * HD;
    const uint n4 = HD / 4u;
    const uint nc = (n4 + 31u) / 32u;            // float4 на лейн (≤ 2)
    const float scale = 1.0f / sqrt((float)HD);
    device const float4* q4 = (device const float4*)(Q + h * HD);
    float4 qv[2], acc[2];
    for (uint c = 0; c < 2u; ++c) {
        const uint d4 = lane + 32u * c;
        qv[c] = (c < nc && d4 < n4) ? q4[d4] * scale : float4(0.0f);
        acc[c] = float4(0.0f);
    }
    float m = -INFINITY, s = 0.0f;
    for (uint j = sgitg; j < Tkv; j += NSG) {
        device const float4* k4 = (device const float4*)(K + (ulong)j * KD + kvh * HD);
        float dot = 0.0f;
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) {
                const float4 kk = k4[d4];
                dot += qv[c].x * kk.x + qv[c].y * kk.y + qv[c].z * kk.z + qv[c].w * kk.w;
            }
        }
        const float sc = simd_sum(dot);
        const float mn = max(m, sc);
        const float corr = exp(m - mn);          // m = -inf → 0
        const float p = exp(sc - mn);
        s = s * corr + p;
        device const float4* v4 = (device const float4*)(V + (ulong)j * KD + kvh * HD);
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) acc[c] = acc[c] * corr + p * v4[d4];
        }
        m = mn;
    }
    // слияние частичных результатов simdgroup'ов
    threadgroup float* mine = sh + sgitg * (2u + HD);
    if (lane == 0u) { mine[0] = m; mine[1] = s; }
    for (uint c = 0; c < nc; ++c) {
        const uint d4 = lane + 32u * c;
        if (d4 < n4) *((threadgroup float4*)(mine + 2u) + d4) = acc[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0u) return;
    float gm = -INFINITY;
    for (uint g = 0; g < NSG; ++g) gm = max(gm, sh[g * (2u + HD)]);
    float gs = 0.0f;
    float4 out[2] = {float4(0.0f), float4(0.0f)};
    for (uint g = 0; g < NSG; ++g) {
        threadgroup float* part = sh + g * (2u + HD);
        const float w = (part[0] == -INFINITY) ? 0.0f : exp(part[0] - gm);
        gs += part[1] * w;
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) out[c] += *((threadgroup float4*)(part + 2u) + d4) * w;
        }
    }
    const float inv = (gs > 0.0f) ? 1.0f / gs : 0.0f;
    device float4* y4 = (device float4*)(Y + h * HD);
    for (uint c = 0; c < nc; ++c) {
        const uint d4 = lane + 32u * c;
        if (d4 < n4) y4[d4] = out[c] * inv;
    }
}

// ── Fused qkv-разбор decode-шага (Ф8.2b) ─────────────────────────────────────
// simdgroup на (токен, голову) по всем H + 2·KVH головам qkv-строки:
//   голова < H        → rmsnorm+rope(q) → Q[t, h]
//   H ≤ голова < H+KVH → rmsnorm+rope(k) → KBUF[pos_kv + t, kh]
//   иначе              → копия v        → VBUF[pos_kv + t, vh]
// Вместо slice ×3 + norm/rope ×2 + kv_update ×2 (7 кернелов на слой).
kernel void qkv_norm_rope_kv_f32(
    device const float* QKV [[buffer(0)]],   // [T, (H+2·KVH)·HD]
    device const float* WQ [[buffer(1)]],
    device const float* WK [[buffer(2)]],
    device float* Q [[buffer(3)]],           // [T, H·HD]
    device float* KBUF [[buffer(4)]],        // [cap, KVH·HD]
    device float* VBUF [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& KVH [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    constant uint& rope_pos [[buffer(10)]],
    constant uint& pos_kv [[buffer(11)]],
    constant float& theta [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    constant RopeYarn& yarn [[buffer(14)]],
    uint gid [[thread_position_in_grid]])
{
    const uint NHT = H + 2u * KVH;
    const uint head = gid / 32u, lane = gid % 32u;
    if (head >= T * NHT) return;
    const uint t = head / NHT, hh = head % NHT;
    device const float* x = QKV + (ulong)t * NHT * HD + (ulong)hh * HD;
    if (hh >= H + KVH) {
        device float* y = VBUF + ((ulong)(pos_kv + t) * KVH + (hh - H - KVH)) * HD;
        for (uint i = lane; i < HD; i += 32u) y[i] = x[i];
        return;
    }
    const bool is_q = hh < H;
    device const float* W = is_q ? WQ : WK;
    device float* y = is_q ? Q + ((ulong)t * H + hh) * HD
                           : KBUF + ((ulong)(pos_kv + t) * KVH + (hh - H)) * HD;
    float s = 0.0f;
    for (uint i = lane; i < HD; i += 32u) s += x[i] * x[i];
    const float inv = rsqrt(simd_sum(s) / (float)HD + eps);
    const float p = (float)(rope_pos + t);
    const uint half_hd = HD / 2u;
    for (uint i = lane; i < half_hd; i += 32u) {
        float c, sn;
        rope_cs(p, i, theta, HD, yarn, c, sn);
        const float x0 = x[i] * inv * W[i];
        const float x1 = x[i + half_hd] * inv * W[i + half_hd];
        y[i] = x0 * c - x1 * sn;
        y[i + half_hd] = x1 * c + x0 * sn;
    }
}

// ── Gated delta rule, prefill: блок токенов в threadgroup-памяти (Ф8.2c) ─────
// Прежний sg-кернел читает k/q-строки из глобальной памяти на КАЖДЫЙ токен и
// КАЖДОЙ группой строк: при T=128 это ~0.5 ГБ трафика на слой (упор в полосу).
// Здесь блок из GDN_C токенов грузится в threadgroup-память один раз на
// threadgroup, и все GDN_ROWS строк стейта работают из неё: трафик падает в
// GDN_ROWS раз. Раскладка элементов — как у прежнего кернела (lane + e·32):
// соседние лейны читают соседние слова, банковых конфликтов нет (раскладка
// «4 подряд на лейн» давала 4-кратный конфликт и была медленнее).
// Порядок операций на токен — как в ggml gated_delta_net:
//   s *= e^g; delta = (v − s·k)·β; s += k·delta; out = s·q/√S.
#define GDN_S 128u
#define GDN_E 4u               // элементов строки на лейн (S/32)
#define GDN_C 16u              // токенов в блоке
#define GDN_ROWS 16u           // строк стейта на threadgroup
#define GDN_NTH (32u * GDN_ROWS)
kernel void gdn_delta_rule_chunk_f32(
    device const float* Q [[buffer(0)]],
    device const float* Kx [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* G [[buffer(3)]],
    device const float* B [[buffer(4)]],
    device float* STATE [[buffer(5)]],
    device float* OUT [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& Hk [[buffer(8)]],
    constant uint& Hv [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpt [[thread_position_in_threadgroup]])
{
    // [C][S] с раскладкой lane + e*32; индексация — по (lane, e)
    // Тайлы только для k/q (v читается по одному элементу на строку —
    // широковещательно, трафик ничтожен; тайл v только съедал память).
    threadgroup float kt[GDN_C * GDN_S];
    threadgroup float qt[GDN_C * GDN_S];
    threadgroup float gb[2u * GDN_C];
    const uint lane = tpt.x, ty = tpt.y;
    const uint tid = ty * 32u + lane;
    const uint h = tgpig.y;
    const uint row = tgpig.x * GDN_ROWS + ty;
    if (h >= Hv || row >= GDN_S) return;
    const uint hk = h % Hk;
    const float scale = 1.0f / sqrt((float)GDN_S);
    device float* srow = STATE + ((ulong)h * GDN_S + row) * GDN_S;
    // элементы строки в регистрах лейна: индексы lane + e*32
    float ls0 = srow[lane], ls1 = srow[lane + 32u], ls2 = srow[lane + 64u], ls3 = srow[lane + 96u];
    for (uint t0 = 0; t0 < T; t0 += GDN_C) {
        const uint c = min(GDN_C, T - t0);
        // кооперативная загрузка блока: индекс e = tt·S + (lane + ee·32),
        // кладём так же — соседние лейны пишут соседние слова
        for (uint i = tid; i < GDN_C * GDN_S; i += GDN_NTH) {
            const uint tt = i / GDN_S, rest = i % GDN_S;
            const uint l = rest % 32u, ee = rest / 32u;
            const uint off = tt * GDN_S + l + ee * 32u;
            if (tt < c) {
                kt[off] = Kx[((ulong)(t0 + tt) * Hk + hk) * GDN_S + ee * 32u + l];
                qt[off] = Q[((ulong)(t0 + tt) * Hk + hk) * GDN_S + ee * 32u + l];
            } else {
                kt[off] = 0.0f;
                qt[off] = 0.0f;
            }
        }
        if (tid < GDN_C) {
            const bool ok = tid < c;
            gb[tid] = ok ? G[(t0 + tid) * Hv + h] : 0.0f;
            gb[GDN_C + tid] = ok ? B[(t0 + tid) * Hv + h] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tt = 0; tt < c; ++tt) {
            const float eg = exp(gb[tt]);
            const float beta = gb[GDN_C + tt];
            const uint tb = tt * GDN_S;
            ls0 *= eg; ls1 *= eg; ls2 *= eg; ls3 *= eg;
            float kv = ls0 * kt[tb + lane] + ls1 * kt[tb + lane + 32u]
                     + ls2 * kt[tb + lane + 64u] + ls3 * kt[tb + lane + 96u];
            kv = simd_sum(kv);
            const float delta = (V[((ulong)(t0 + tt) * Hv + h) * GDN_S + row] - kv) * beta;
            ls0 += kt[tb + lane] * delta;
            ls1 += kt[tb + lane + 32u] * delta;
            ls2 += kt[tb + lane + 64u] * delta;
            ls3 += kt[tb + lane + 96u] * delta;
            float y = ls0 * qt[tb + lane] + ls1 * qt[tb + lane + 32u]
                    + ls2 * qt[tb + lane + 64u] + ls3 * qt[tb + lane + 96u];
            y = simd_sum(y);
            if (lane == 0u) OUT[((ulong)(t0 + tt) * Hv + h) * GDN_S + row] = y * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    srow[lane] = ls0;
    srow[lane + 32u] = ls1;
    srow[lane + 64u] = ls2;
    srow[lane + 96u] = ls3;
}

// ── Flash-attention промпта на матричных блоках simdgroup (Ф8.2c) ────────────
// Прежний prefill-кернел: simdgroup на (голову, запрос) и simd_sum на КАЖДЫЙ
// ключ — 128 ключей × 32 головы × 128 запросов пересчётов-редукций на слой.
// Здесь threadgroup (2 simdgroup'а) ведёт 16 запросов одной головы, а блок из
// 32 ключей обрабатывается матричными операциями:
//   S = Q·Kᵀ (8×8-фрагменты) → online-softmax по строкам (маска, коррекция)
//   → O = diag(corr)·O + P·V.
// K и V читаются ПРЯМО из device-памяти фрагментами (simdgroup_load с шагом
// строк) — тайлы K/V в threadgroup-памяти не нужны (лимит 32 КБ: у Qwen3.5
// hd=256 f32-тайлы K/V/Q не помещались). Аккумулятор O — в регистрах,
// коррекция строк — умножением на диагональный фрагмент.
#define FA_BQ 16u      // запросов на threadgroup (2 simdgroup'а × 8)
#define FA_BK 32u      // ключей на шаг
kernel void attn_prefill_mma_f32(
    device const float* Q [[buffer(0)]],     // [Tq, H*HD]
    device const float* K [[buffer(1)]],     // [Tkv, KVH*HD]
    device const float* V [[buffer(2)]],     // [Tkv, KVH*HD]
    device const float* MASK [[buffer(3)]],  // [Tq, Tkv] или nullptr
    device float* Y [[buffer(4)]],           // [Tq, H*HD]
    constant uint& Tq [[buffer(5)]],
    constant uint& Tkv [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& KVH [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    threadgroup float* sh [[threadgroup(0)]],   // [FA_BQ][HD] Q + [FA_BQ][FA_BK] S + [FA_BQ][8] диаг
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpt [[thread_position_in_threadgroup]])
{
    threadgroup float* sQ = sh;
    threadgroup float* sS = sh + FA_BQ * HD;
    threadgroup float* sC = sS + FA_BQ * FA_BK;
    // Threadgroup одномерный (64 = 2×32): номер simdgroup'а — из позиции,
    // не из tpt.y (он для 1-D группы всегда 0 — второй simdgroup тогда считал
    // те же строки 0..7).
    const uint lane = tpt.x % 32u;
    const uint sg = tpt.x / 32u;
    const uint tid = tpt.x;
    const uint h = tgpig.y;
    const uint kvh = h / (H / KVH);
    const uint q0 = tgpig.x * FA_BQ;
    const uint D = H * HD, KD = KVH * HD;
    const float scale = 1.0f / sqrt((float)HD);
    for (uint e = tid; e < FA_BQ * HD; e += 64u) {
        const uint r = e / HD, c = e % HD;
        const uint qi = q0 + r;
        sQ[e] = (qi < Tq) ? Q[(ulong)qi * D + h * HD + c] * scale : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nf = HD / 8u;
    // Фрагменты выхода — в регистрах: циклы по фрагментам разворачиваются с
    // константной границей (32), иначе динамическая индексация o[] уводит
    // массив в локальную память (hd=256 был медленнее прежнего кернела).
    simdgroup_float8x8 o[16];
    #pragma unroll
    for (uint f = 0; f < 32u; ++f) {
        if (f >= nf) break;
        o[f] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    float m_run = -INFINITY, l_run = 0.0f;
    for (uint k0 = 0; k0 < Tkv; k0 += FA_BK) {
        // S[8][32] этого simdgroup'а = Q[8][HD] · K[32][HD]ᵀ
        // Последний неполный блок ключей (Tkv не кратно 32) — отдельной
        // веткой: 8-строчный фрагмент, пересекающий Tkv, читал строки за
        // концом тензора (вне KV-кэша — мусор или NaN за концом MTLBuffer, а
        // 0·NaN в P·V портит весь столбец выхода). Здесь фрагменты после Tkv
        // пропускаются, а пересекающий сдвигается назад к Tkv-8 (его первые
        // столбцы дублируют ключи предыдущего фрагмента — их отсекает
        // softmax). Горячий цикл полных блоков не тронут: ветвление внутри
        // развёрнутых циклов стоило +40 % времени внимания.
        const bool tail = k0 + FA_BK > Tkv;
        if (!tail) {
            for (uint j = 0; j < FA_BK / 8u; ++j) {
                simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
                #pragma unroll
                for (uint kk = 0; kk < 32u; ++kk) {
                    if (kk >= nf) break;
                    simdgroup_float8x8 qa, kb;
                    simdgroup_load(qa, sQ + (sg * 8u) * HD + kk * 8u, HD);
                    simdgroup_load(kb, K + (ulong)(k0 + j * 8u) * KD + kvh * HD + kk * 8u, KD, 0, true);
                    simdgroup_multiply_accumulate(s, qa, kb, s);
                }
                simdgroup_store(s, sS + (sg * 8u) * FA_BK + j * 8u, FA_BK);
            }
        } else {
            for (uint j = 0; j < FA_BK / 8u; ++j) {
                simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
                const uint kf = k0 + j * 8u;
                if (kf < Tkv) {
                    const uint ks = min(kf, Tkv - 8u);
                    #pragma unroll
                    for (uint kk = 0; kk < 32u; ++kk) {
                        if (kk >= nf) break;
                        simdgroup_float8x8 qa, kb;
                        simdgroup_load(qa, sQ + (sg * 8u) * HD + kk * 8u, HD);
                        simdgroup_load(kb, K + (ulong)ks * KD + kvh * HD + kk * 8u, KD, 0, true);
                        simdgroup_multiply_accumulate(s, qa, kb, s);
                    }
                }
                simdgroup_store(s, sS + (sg * 8u) * FA_BK + j * 8u, FA_BK);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // softmax: строка на лейн (строк FA_BQ, лейнов в threadgroup 64)
        if (lane < 8u) {
            const uint row = sg * 8u + lane;      // строка FA_BQ этого лейна
            const uint qi = q0 + row;
            float mx = m_run;
            for (uint j = 0; j < FA_BK; ++j) {
                // ключ столбца: в хвостовом блоке фрагмент мог быть сдвинут
                // к Tkv-8 (столбцы до начала фрагмента kf — дубли)
                const uint kf = k0 + (j & ~7u);
                const uint kj = tail ? min(kf, Tkv - 8u) + (j & 7u) : k0 + j;
                float v = sS[row * FA_BK + j];
                if (kf >= Tkv || kj < kf || kj >= Tkv || qi >= Tq) v = -INFINITY;
                else if (MASK != nullptr) v += MASK[(ulong)qi * Tkv + kj];
                sS[row * FA_BK + j] = v;
                mx = max(mx, v);
            }
            float l = 0.0f;
            for (uint j = 0; j < FA_BK; ++j) {
                const float v = sS[row * FA_BK + j];
                const float p = (v == -INFINITY) ? 0.0f : exp(v - mx);
                sS[row * FA_BK + j] = p;
                l += p;
            }
            const float corr = (m_run == -INFINITY) ? 0.0f : exp(m_run - mx);
            l_run = l_run * corr + l;
            m_run = mx;
            for (uint c = 0; c < 8u; ++c) sC[row * 8u + c] = (c == lane) ? corr : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 cd;
        simdgroup_load(cd, sC + (sg * 8u) * 8u, 8);
        if (!tail) {
            #pragma unroll
            for (uint f = 0; f < 32u; ++f) {
                if (f >= nf) break;
                simdgroup_float8x8 acc;
                simdgroup_multiply(acc, cd, o[f]);
                for (uint j = 0; j < FA_BK / 8u; ++j) {
                    simdgroup_float8x8 pa, vb;
                    simdgroup_load(pa, sS + (sg * 8u) * FA_BK + j * 8u, FA_BK);
                    simdgroup_load(vb, V + (ulong)(k0 + j * 8u) * KD + kvh * HD + f * 8u, KD);
                    simdgroup_multiply_accumulate(acc, pa, vb, acc);
                }
                o[f] = acc;
            }
        } else {
            #pragma unroll
            for (uint f = 0; f < 32u; ++f) {
                if (f >= nf) break;
                simdgroup_float8x8 acc;
                simdgroup_multiply(acc, cd, o[f]);
                for (uint j = 0; j < FA_BK / 8u; ++j) {
                    const uint kf = k0 + j * 8u;
                    if (kf >= Tkv) break;
                    simdgroup_float8x8 pa, vb;
                    simdgroup_load(pa, sS + (sg * 8u) * FA_BK + j * 8u, FA_BK);
                    simdgroup_load(vb, V + (ulong)min(kf, Tkv - 8u) * KD + kvh * HD + f * 8u, KD);
                    simdgroup_multiply_accumulate(acc, pa, vb, acc);
                }
                o[f] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Выход: O — в освободившийся тайл Q, затем нормировка 1/l и запись.
    if (lane < 8u) sC[8u * FA_BK * 0u + (sg * 8u + lane)] = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint f = 0; f < 32u; ++f) {
        if (f >= nf) break;
        simdgroup_store(o[f], sQ + (sg * 8u) * HD + f * 8u, HD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < FA_BQ * HD; e += 64u) {
        const uint r = e / HD, c = e % HD;
        const uint qi = q0 + r;
        if (qi < Tq) Y[(ulong)qi * D + h * HD + c] = sQ[e] * sC[r];
    }
}






// ── Flash-attention промпта: half-тайлы K/V в разделяемой памяти (Ф8.2e) ─────
// Замеры: half-матричные блоки дают 8.4 TFLOPS против 3.5 у float (микробенч),
// а прежний кернел читал K/V-фрагменты из device-памяти КАЖДОЙ подгруппой
// заново (при Tq=512 и контексте 4.6k ~9 ГБ трафика на слой). Здесь блок K и
// V грузится в разделяемую память ОДИН раз на threadgroup и конвертируется в
// half: устройство читается вчетверо реже, блоки — half-скорости.
// Раскладка (hd=128, BQ=32, BK=32, байты): Q half 8 КБ, K half 8 КБ (как есть,
// [key][d]), V half 8 КБ, S float 4 КБ (своя область на подгруппу), P half
// 2 КБ (отдельно: half-раскладка вдвое плотнее float и «поверх S» наезжает на
// чужие строки), диагонали 1 КБ (своя на подгруппу), 1/l 128 Б = 31.1 КБ.
// Выгрузка — через область S, но не более 32 колонок на проход (шаг строки
// области 32: восемь фрагментов за проход залезали в соседние строки — на этом
// падал предыдущий вариант кернела).
#define FA4_BQ 32u
// BK=16: тайлы K/V вдвое меньше (4 КБ каждый при hd=128), вся группа уходит
// в ~16 КБ — на ядро помещаются ДВЕ группы (лимит 32 КБ), что важнее
// удвоенного числа барьеров (замер).
#define FA4_BK 16u
kernel void attn_prefill_fa4_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* MASK [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant uint& Tq [[buffer(5)]],
    constant uint& Tkv [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& KVH [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    threadgroup char* shm [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpt [[thread_position_in_threadgroup]])
{
    threadgroup half* sQh = (threadgroup half*)shm;
    threadgroup half* sKh = sQh + FA4_BQ * HD;
    threadgroup half* sVh = sKh + FA4_BK * HD;
    threadgroup float* sS = (threadgroup float*)(sVh + FA4_BK * HD);
    threadgroup half* sP = (threadgroup half*)(sS + 4u * 8u * FA4_BK);
    threadgroup float* sD = (threadgroup float*)(sP + 4u * 8u * FA4_BK);
    threadgroup float* sL = sD + 4u * 64u;
    const uint lane = tpt.x % 32u;
    const uint sg = tpt.x / 32u;
    const uint tid = tpt.x;
    const uint h = tgpig.y;
    const uint kvh = h / (H / KVH);
    const uint q0 = tgpig.x * FA4_BQ;
    const uint D = H * HD, KD = KVH * HD;
    const uint nf = HD / 8u;
    const float scale = 1.0f / sqrt((float)HD);
    threadgroup float* myS = sS + sg * 8u * FA4_BK;
    threadgroup float* myD = sD + sg * 64u;
    for (uint e = tid; e < FA4_BQ * HD; e += 128u) {
        const uint r = e / HD, c = e % HD;
        const uint qi = q0 + r;
        sQh[e] = (qi < Tq) ? (half)Q[(ulong)qi * D + h * HD + c] : (half)0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qa[16];
    for (uint f = 0; f < nf; ++f)
        simdgroup_load(qa[f], sQh + (sg * 8u) * HD + f * 8u, HD);
    simdgroup_float8x8 o[16];
    for (uint f = 0; f < nf; ++f) o[f] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -INFINITY, l_run = 0.0f;
    for (uint k0 = 0; k0 < Tkv; k0 += FA4_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < FA4_BK * HD; e += 128u) {
            const uint r = e / HD, c = e % HD;
            const uint kj = k0 + r;
            const bool ok = kj < Tkv;
            sKh[e] = ok ? (half)K[(ulong)kj * KD + kvh * HD + c] : (half)0.0h;
            sVh[e] = ok ? (half)V[(ulong)kj * KD + kvh * HD + c] : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < FA4_BK / 8u; ++j) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (uint f = 0; f < nf; ++f) {
                simdgroup_half8x8 kb;
                simdgroup_load(kb, sKh + (j * 8u) * HD + f * 8u, HD, 0, true);
                simdgroup_multiply_accumulate(s, qa[f], kb, s);
            }
            simdgroup_store(s, myS + j * 8u, FA4_BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            const uint qi = q0 + sg * 8u + lane;
            float mx = m_run;
            for (uint j = 0; j < FA4_BK; ++j) {
                const uint kj = k0 + j;
                float v = myS[lane * FA4_BK + j] * scale;
                if (kj >= Tkv || qi >= Tq) v = -INFINITY;
                else if (MASK != nullptr) v += MASK[(ulong)qi * Tkv + kj];
                myS[lane * FA4_BK + j] = v;
                mx = max(mx, v);
            }
            float l = 0.0f;
            for (uint j = 0; j < FA4_BK; ++j) {
                const float v = myS[lane * FA4_BK + j];
                const float p = (v == -INFINITY) ? 0.0f : exp(v - mx);
                sP[(sg * 8u + lane) * FA4_BK + j] = (half)p;
                l += p;
            }
            const float corr = (m_run == -INFINITY) ? 0.0f : exp(m_run - mx);
            l_run = l_run * corr + l;
            m_run = mx;
            for (uint c = 0; c < 8u; ++c) myD[lane * 8u + c] = (c == lane) ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 cd;
        simdgroup_load(cd, myD, 8u);
        for (uint f = 0; f < nf; ++f) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_multiply(acc, cd, o[f]);
            for (uint j = 0; j < FA4_BK / 8u; ++j) {
                simdgroup_half8x8 pa, vb;
                simdgroup_load(pa, sP + (sg * 8u) * FA4_BK + j * 8u, FA4_BK);
                simdgroup_load(vb, sVh + (j * 8u) * HD + f * 8u, HD);
                simdgroup_multiply_accumulate(acc, pa, vb, acc);
            }
            o[f] = acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 8u) sL[sg * 8u + lane] = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8u) {
        const float inv = sL[sg * 8u + lane];
        for (uint c = 0; c < 8u; ++c) myD[lane * 8u + c] = (c == lane) ? inv : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dinv;
    simdgroup_load(dinv, myD, 8u);
    // Выгрузка: не более 4 фрагментов (32 колонки) на проход — шаг строки
    // области S равен FA4_BK, восемь фрагментов ушли бы в соседние строки.
    for (uint f = 0; f < nf; ++f) {
        simdgroup_float8x8 res;
        simdgroup_multiply(res, dinv, o[f]);
        // Порция выгрузки = ровно одна строка области S в колонках:
        // FA4_BK/8 фрагментов по 8 колонок (при BK=32 — четыре фрагмента,
        // при BK=16 — два; больше колонок, чем шаг строки, залезает в
        // соседние строки — на этом падал предыдущий вариант).
        const uint fph = FA4_BK / 8u;
        simdgroup_store(res, myS + (f % fph) * 8u, FA4_BK);
        if (f % fph == fph - 1u || f + 1u == nf) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint fbase = (f / fph) * fph;
            const uint fcount = f - fbase + 1u;   // ≤ fph
            // (строка, фрагмент, столбец) — три измерения: раньше строка
            // считалась как e/8, и при fcount < 8 писались только первые
            // fcount строк (прочие оставались с мусором от маски).
            for (uint e = lane; e < 8u * fcount * 8u; e += 32u) {
                const uint c = e % 8u;
                const uint t1 = e / 8u;
                const uint fi = t1 % fcount;
                const uint r = t1 / fcount;
                const uint qi = q0 + sg * 8u + r;
                if (qi < Tq) Y[(ulong)qi * D + h * HD + (fbase + fi) * 8u + c] = myS[r * FA4_BK + fi * 8u + c];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ── Flash-attention промпта с разбиением ключей (Ф8.2e, сплит-K) ─────────────
// Узкое место длинного промпта — ЛАТЕНТНОСТЬ: один threadgroup последовательно
// проходит все Tkv ключей (декодный кернел ровно эту проблему решает
// разбиением, 12× на длинном контексте). Здесь то же для промпта: threadgroup
// (2 simdgroup'а, 16 запросов одной головы) обрабатывает СВОЙ участок ключей и
// пишет частичный online-softmax (m, l, acc) в глобальный буфер; combine-кернел
// сливает части. Частицы — f32 (точность не хуже неразбитого пути),
// раскладка [Tq, H, SPLIT, 2 + HD]: [0]=m, [1]=l, [2..]=acc.
#define FAP_BQ 16u
kernel void attn_prefill_partial_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* MASK [[buffer(3)]],
    device float* PART [[buffer(4)]],
    constant uint& Tq [[buffer(5)]],
    constant uint& Tkv [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& KVH [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    constant uint& SPLIT [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpt3 [[thread_position_in_threadgroup]])
{
    const uint tpt = tpt3.x;
    const uint lane = tpt % 32u;
    const uint sg = tpt / 32u;
    const uint h = tgpig.y;
    const uint sp = tgpig.z;
    const uint kvh = h / (H / KVH);
    const uint q0 = tgpig.x * FAP_BQ + sg * 8u;
    const uint D = H * HD, KD = KVH * HD;
    const uint nf = HD / 8u;
    const float scale = 1.0f / sqrt((float)HD);
    const uint per = (Tkv + SPLIT - 1u) / SPLIT;
    const uint k0 = sp * per;
    const uint k1 = min(Tkv, k0 + per);
    threadgroup float shS[2u * 8u * 32u];
    threadgroup float shD[2u * 64u];
    threadgroup float shQ[16u * 128u];   // строки Q группы (hd ≤ 128)
    threadgroup float* myS = shS + sg * 256u;
    threadgroup float* myD = shD + sg * 64u;
    // Q — построчно с защитой хвоста: обнуление ЦЕЛОГО 8-строчного
    // фрагмента при q0+8 > Tq теряло последние 1–7 запросов промпта.
    const uint qtg = tgpig.x * FAP_BQ;
    for (uint e = tpt; e < 16u * HD; e += 64u) {
        const uint r = e / HD, c = e % HD;
        const uint qi = qtg + r;
        shQ[e] = (qi < Tq) ? Q[(ulong)qi * D + h * HD + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 qa[16];
    for (uint f = 0; f < nf; ++f) {
        simdgroup_load(qa[f], shQ + (sg * 8u) * HD + f * 8u, HD);
    }
    simdgroup_float8x8 o[16];
    for (uint f = 0; f < nf; ++f) o[f] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -INFINITY, l_run = 0.0f;
    for (uint kb = k0; kb < k1; kb += 32u) {
        // Конец тензора — Tkv (строки за k1 внутри тензора валидны, их
        // отсекает маска kj >= k1); фрагмент, пересекающий Tkv, сдвигается
        // к Tkv-8 — как в attn_prefill_mma_f32.
        const bool tail = kb + 32u > Tkv;
        for (uint j = 0; j < 4u; ++j) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
            const uint kfr = kb + j * 8u;
            if (kfr < Tkv) {
                const uint ks = tail ? min(kfr, Tkv - 8u) : kfr;
                for (uint f = 0; f < nf; ++f) {
                    simdgroup_float8x8 kf;
                    simdgroup_load(kf, K + (ulong)ks * KD + kvh * HD + f * 8u, KD, 0, true);
                    simdgroup_multiply_accumulate(s, qa[f], kf, s);
                }
            }
            simdgroup_store(s, myS + j * 8u, 32u);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            const uint qi = q0 + lane;
            float mx = m_run;
            for (uint j = 0; j < 32u; ++j) {
                const uint kfr = kb + (j & ~7u);
                const uint kj = tail ? min(kfr, Tkv - 8u) + (j & 7u) : kb + j;
                float v = myS[lane * 32u + j] * scale;
                // kj >= k1 — хвост последнего блока участка: диапазоны
                // разбиений не кратны 32, и без этой границы ключи на стыке
                // считались бы ДВАЖДЫ (в своём участке и в соседнем).
                // kj < kfr — дубли сдвинутого хвостового фрагмента.
                if (kfr >= Tkv || kj < kfr || kj >= k1 || kj >= Tkv || qi >= Tq) v = -INFINITY;
                else if (MASK != nullptr) v += MASK[(ulong)qi * Tkv + kj];
                myS[lane * 32u + j] = v;
                mx = max(mx, v);
            }
            float l = 0.0f;
            for (uint j = 0; j < 32u; ++j) {
                const float v = myS[lane * 32u + j];
                const float p = (v == -INFINITY) ? 0.0f : exp(v - mx);
                myS[lane * 32u + j] = p;
                l += p;
            }
            const float corr = (m_run == -INFINITY) ? 0.0f : exp(m_run - mx);
            l_run = l_run * corr + l;
            m_run = mx;
            for (uint c = 0; c < 8u; ++c) myD[lane * 8u + c] = (c == lane) ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 cd;
        simdgroup_load(cd, myD, 8u);
        for (uint f = 0; f < nf; ++f) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_multiply(acc, cd, o[f]);
            for (uint j = 0; j < 4u; ++j) {
                const uint kfr = kb + j * 8u;
                if (kfr >= Tkv) break;
                const uint ks = tail ? min(kfr, Tkv - 8u) : kfr;
                simdgroup_float8x8 pa, vb;
                simdgroup_load(pa, myS + j * 8u, 32u);
                simdgroup_load(vb, V + (ulong)ks * KD + kvh * HD + f * 8u, KD);
                simdgroup_multiply_accumulate(acc, pa, vb, acc);
            }
            o[f] = acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // частичный softmax: [m, l, acc] в [Tq, H, SPLIT, 2+HD]
    if (lane < 8u) {
        const uint qi = q0 + lane;
        if (qi < Tq) {
            device float* rowp = PART + (((ulong)qi * H + h) * SPLIT + sp) * (2u + HD);
            rowp[0] = m_run;
            rowp[1] = l_run;
        }
    }
    for (uint f = 0; f < nf; ++f) {
        simdgroup_store(o[f], myS + (f % 4u) * 8u, 32u);
        if (f % 4u == 3u || f + 1u == nf) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint fbase = (f / 4u) * 4u;
            const uint fcount = f - fbase + 1u;
            for (uint e = lane; e < 8u * fcount * 8u; e += 32u) {
                const uint c = e % 8u;
                const uint t1 = e / 8u;
                const uint fi = t1 % fcount;
                const uint r = t1 / fcount;
                const uint qi = q0 + r;
                if (qi < Tq) {
                    device float* rowp = PART + (((ulong)qi * H + h) * SPLIT + sp) * (2u + HD);
                    rowp[2u + (fbase + fi) * 8u + c] = myS[r * 32u + fi * 8u + c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Слияние частичных softmax: out = Σ acc_s·e^{m_s−M} / Σ l_s·e^{m_s−M}.
// Threadgroup на строку (запрос, голова), лейн ведёт свои float4 колонки.
kernel void attn_prefill_combine_f32(
    device const float* PART [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& TqH [[buffer(2)]],
    constant uint& HD [[buffer(3)]],
    constant uint& SPLIT [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (row >= TqH) return;
    const ulong rs = (ulong)row * SPLIT;
    float gm = -INFINITY;
    for (uint sp = 0; sp < SPLIT; ++sp) gm = max(gm, PART[(rs + sp) * (2u + HD)]);
    float l = 0.0f;
    for (uint sp = 0; sp < SPLIT; ++sp) {
        device const float* rowp = PART + (rs + sp) * (2u + HD);
        const float w = (rowp[0] == -INFINITY) ? 0.0f : exp(rowp[0] - gm);
        l += rowp[1] * w;
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    const uint n4 = HD / 4u;
    for (uint c = lane; c < n4; c += 32u) {
        float4 acc = float4(0.0f);
        for (uint sp = 0; sp < SPLIT; ++sp) {
            device const float* rowp = PART + (rs + sp) * (2u + HD);
            const float w = (rowp[0] == -INFINITY) ? 0.0f : exp(rowp[0] - gm);
            acc += w * *(device const float4*)(rowp + 2u + c * 4u);
        }
        *(device float4*)(Y + (ulong)row * HD + c * 4u) = acc * inv;
    }
}

// ═══════════════ Ф8.2f: f16-KV — кернелы (вдвое меньше памяти и трафика) ══════
// llama.cpp держит KV в f16 по умолчанию; у нас кэш был f32 (память под
// контекст вдвое больше, трафик внимания вдвое выше). Здесь f16-варианты
// трёх кернелов: слитый разбор qkv (пишет half Q/K/V), декодное внимание
// (half4-чтения) и предфилл-внимание (half-фрагменты 8×8, накопление float).

// Слитый qkv: Q (half) + K/V прямо в f16-кэш.
kernel void qkv_norm_rope_kv_f16_f32(
    device const float* QKV [[buffer(0)]],
    device const float* WQ [[buffer(1)]],
    device const float* WK [[buffer(2)]],
    device float* Q [[buffer(3)]],
    device half* KBUF [[buffer(4)]],
    device half* VBUF [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& KVH [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    constant uint& rope_pos [[buffer(10)]],
    constant uint& pos_kv [[buffer(11)]],
    constant float& theta [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    constant RopeYarn& yarn [[buffer(14)]],
    uint gid [[thread_position_in_grid]])
{
    const uint NHT = H + 2u * KVH;
    const uint head = gid / 32u, lane = gid % 32u;
    if (head >= T * NHT) return;
    const uint t = head / NHT, hh = head % NHT;
    device const float* x = QKV + (ulong)t * NHT * HD + (ulong)hh * HD;
    if (hh >= H + KVH) {
        device half* y = VBUF + ((ulong)(pos_kv + t) * KVH + (hh - H - KVH)) * HD;
        for (uint i = lane; i < HD; i += 32u) y[i] = (half)x[i];
        return;
    }
    const bool is_q = hh < H;
    device const float* W = is_q ? WQ : WK;
    // Q — отдельный (f32) буфер, K — половина общей пары указателей: типы
    // разные, поэтому пишем развилкой, а не одним `device half* y`.
    device float* yq = Q + ((ulong)t * H + hh) * HD;
    device half* yk = KBUF + ((ulong)(pos_kv + t) * KVH + (hh - H)) * HD;
    float s = 0.0f;
    for (uint i = lane; i < HD; i += 32u) s += x[i] * x[i];
    const float inv = rsqrt(simd_sum(s) / (float)HD + eps);
    const float p = (float)(rope_pos + t);
    const uint half_hd = HD / 2u;
    for (uint i = lane; i < half_hd; i += 32u) {
        float c, sn;
        rope_cs(p, i, theta, HD, yarn, c, sn);
        const float x0 = x[i] * inv * W[i];
        const float x1 = x[i + half_hd] * inv * W[i + half_hd];
        if (is_q) {
            yq[i] = x0 * c - x1 * sn;
            yq[i + half_hd] = x1 * c + x0 * sn;
        } else {
            yk[i] = (half)(x0 * c - x1 * sn);
            yk[i + half_hd] = (half)(x1 * c + x0 * sn);
        }
    }
}

// Декодное внимание по f16-кэшу: K/V читаются half4 (вдвое меньше трафика).
kernel void attn_decode_split_f16(
    device const float* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& Tkv [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant uint& KVH [[buffer(6)]],
    constant uint& HD [[buffer(7)]],
    constant uint& NSG [[buffer(8)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint h [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    if (h >= H) return;
    const uint kvh = h / (H / KVH);
    const uint KD = KVH * HD;
    const uint n4 = HD / 4u;
    const uint nc = (n4 + 31u) / 32u;
    const float scale = 1.0f / sqrt((float)HD);
    // Q — f32 (как в llama.cpp): округление запроса до half теряло ~11 бит
    // и на близких логитах меняло выбранный токен через десятки шагов.
    device const float4* q4 = (device const float4*)(Q + h * HD);
    float4 qv[2], acc[2];
    for (uint c = 0; c < 2u; ++c) {
        const uint d4 = lane + 32u * c;
        float4 qq = (c < nc && d4 < n4) ? q4[d4] : float4(0.0f);
        qv[c] = qq * scale;
        acc[c] = float4(0.0f);
    }
    float m = -INFINITY, s = 0.0f;
    for (uint j = sgitg; j < Tkv; j += NSG) {
        device const half4* k4 = (device const half4*)(K + (ulong)j * KD + kvh * HD);
        float dot = 0.0f;
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) {
                const float4 kk = float4(k4[d4]);
                dot += qv[c].x * kk.x + qv[c].y * kk.y + qv[c].z * kk.z + qv[c].w * kk.w;
            }
        }
        const float sc = simd_sum(dot);
        const float mn = max(m, sc);
        const float corr = exp(m - mn);
        const float p = exp(sc - mn);
        s = s * corr + p;
        device const half4* v4 = (device const half4*)(V + (ulong)j * KD + kvh * HD);
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) acc[c] = acc[c] * corr + p * float4(v4[d4]);
        }
        m = mn;
    }
    threadgroup float* mine = sh + sgitg * (2u + HD);
    if (lane == 0u) { mine[0] = m; mine[1] = s; }
    for (uint c = 0; c < nc; ++c) {
        const uint d4 = lane + 32u * c;
        if (d4 < n4) *((threadgroup float4*)(mine + 2u) + d4) = acc[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0u) return;
    float gm = -INFINITY;
    for (uint g = 0; g < NSG; ++g) gm = max(gm, sh[g * (2u + HD)]);
    float gs = 0.0f;
    float4 out[2] = {float4(0.0f), float4(0.0f)};
    for (uint g = 0; g < NSG; ++g) {
        threadgroup float* part = sh + g * (2u + HD);
        const float w = (part[0] == -INFINITY) ? 0.0f : exp(part[0] - gm);
        gs += part[1] * w;
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) out[c] += *((threadgroup float4*)(part + 2u) + d4) * w;
        }
    }
    const float inv = (gs > 0.0f) ? 1.0f / gs : 0.0f;
    device float4* y4 = (device float4*)(Y + h * HD);
    for (uint c = 0; c < nc; ++c) {
        const uint d4 = lane + 32u * c;
        if (d4 < n4) y4[d4] = out[c] * inv;
    }
}

// Внимание малого батча запросов по f16-кэшу (окно верификации спека,
// блок драфтера; 2 ≤ Tq ≤ 8): схема attn_decode_split_f16 — ключи кэша
// делятся между NSG simdgroup'ами, частичные (m, s, acc) сводятся в
// threadgroup-памяти; сетка — головы × блоки по QB запросов. Общий путь
// (attn_ar_f16_fused: simdgroup на пару (голова, запрос), весь кэш
// последовательно) на контексте 4k стоил ≈4× декода уже при Tq = 2.
// QB = 1 быстрее всех (examples/verify_cost, 8B, контекст 4k, Tq = 4:
// 1.38× декода против 1.64× при QB = 2 и 2.96× при QB = 4): повторное
// чтение K/V разными threadgroup'ами гасит кэш, а параллелизма больше; при
// QB = 8 аккумуляторы не помещались в регистры. Маска — как у общего пути:
// буфер [Tq, Tkv] (каузальная, SWA, M-RoPE) или её нет — каждый запрос
// видит все ключи.
template <uint QB>
kernel void attn_split_f16_qb(
    device const float* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& Tkv [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant uint& KVH [[buffer(6)]],
    constant uint& HD [[buffer(7)]],
    constant uint& NSG [[buffer(8)]],
    constant uint& Tq [[buffer(9)]],
    device const float* MASK [[buffer(10)]],
    constant uint& HAS_MASK [[buffer(11)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t0 = tgpig.y * QB;
    if (h >= H || t0 >= Tq) return;
    const uint nq = min(QB, Tq - t0);
    const uint kvh = h / (H / KVH);
    const uint D = H * HD;
    const uint KD = KVH * HD;
    const uint n4 = HD / 4u;
    const uint nc = (n4 + 31u) / 32u;
    const float scale = 1.0f / sqrt((float)HD);
    float4 qv[QB][2], acc[QB][2];
    float m[QB], s[QB];
    for (uint i = 0; i < QB; ++i) {
        m[i] = -INFINITY;
        s[i] = 0.0f;
        device const float4* q4 = (device const float4*)(Q + (ulong)(t0 + min(i, nq - 1u)) * D + h * HD);
        for (uint c = 0; c < 2u; ++c) {
            const uint d4 = lane + 32u * c;
            qv[i][c] = (c < nc && d4 < n4) ? q4[d4] * scale : float4(0.0f);
            acc[i][c] = float4(0.0f);
        }
    }
    for (uint j = sgitg; j < Tkv; j += NSG) {
        device const half4* k4 = (device const half4*)(K + (ulong)j * KD + kvh * HD);
        device const half4* v4 = (device const half4*)(V + (ulong)j * KD + kvh * HD);
        float4 kk[2], vv[2];
        for (uint c = 0; c < 2u; ++c) {
            const uint d4 = lane + 32u * c;
            const bool ok = c < nc && d4 < n4;
            kk[c] = ok ? float4(k4[d4]) : float4(0.0f);
            vv[c] = ok ? float4(v4[d4]) : float4(0.0f);
        }
        for (uint i = 0; i < QB; ++i) {
            if (i >= nq) break;
            const float bias = HAS_MASK != 0u ? MASK[(ulong)(t0 + i) * Tkv + j] : 0.0f;
            if (bias == -INFINITY) continue;
            const float sc = simd_sum(dot(qv[i][0], kk[0]) + dot(qv[i][1], kk[1])) + bias;
            const float mn = max(m[i], sc);
            const float corr = exp(m[i] - mn);
            const float p = exp(sc - mn);
            s[i] = s[i] * corr + p;
            acc[i][0] = acc[i][0] * corr + p * vv[0];
            acc[i][1] = acc[i][1] * corr + p * vv[1];
            m[i] = mn;
        }
    }
    // свёртка частичных результатов по simdgroup'ам — по одному запросу
    threadgroup float* mine = sh + sgitg * (2u + HD);
    for (uint i = 0; i < nq; ++i) {
        if (lane == 0u) { mine[0] = m[i]; mine[1] = s[i]; }
        for (uint c = 0; c < nc; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < n4) *((threadgroup float4*)(mine + 2u) + d4) = acc[i][c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0u) {
            float gm = -INFINITY;
            for (uint g = 0; g < NSG; ++g) gm = max(gm, sh[g * (2u + HD)]);
            float gs = 0.0f;
            float4 out[2] = {float4(0.0f), float4(0.0f)};
            for (uint g = 0; g < NSG; ++g) {
                threadgroup float* part = sh + g * (2u + HD);
                const float w = (part[0] == -INFINITY) ? 0.0f : exp(part[0] - gm);
                gs += part[1] * w;
                for (uint c = 0; c < nc; ++c) {
                    const uint d4 = lane + 32u * c;
                    if (d4 < n4) out[c] += *((threadgroup float4*)(part + 2u) + d4) * w;
                }
            }
            const float inv = (gs > 0.0f) ? 1.0f / gs : 0.0f;
            device float4* y4 = (device float4*)(Y + (ulong)(t0 + i) * D + h * HD);
            for (uint c = 0; c < nc; ++c) {
                const uint d4 = lane + 32u * c;
                if (d4 < n4) y4[d4] = out[c] * inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

typedef decltype(attn_split_f16_qb<1>) attn_split_f16_qb_t;
template [[host_name("attn_split_f16_q1")]] kernel attn_split_f16_qb_t attn_split_f16_qb<1>;

// Общий f16-путь внимания (любые Tq, маска): port attn_ar_f32_fused с
// half4-чтением Q/K/V (голова = 32 лейна, online-softmax). Хвостовые
// чанки промпта (Tq < 16) и короткие промпты матричных кернелов не имеют.
kernel void attn_ar_f16_fused(
    device const float* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
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
    uint b = gid.x;
    uint n4 = HD / 4u;
    uint h = gid.y / 32u;
    uint lane = gid.y % 32u;
    uint t = gid.z;
    if (b >= B || h >= H || t >= Tq) return;
    uint kvh = h / n_rep;
    uint D = H * HD;
    uint KVD = KVH * HD;
    const float scale = 1.0f / sqrt((float)HD);
    uint nc = (n4 + 31u) / 32u;
    const uint MAXC = 4u;
    const device float4* qrow4 = (const device float4*)(Q + (b * Tq + t) * D + h * HD);
    float4 q4[MAXC];
    float4 y4[MAXC];
    for (uint c = 0u; c < MAXC; ++c) {
        uint d4 = lane + 32u * c;
        q4[c] = (c < nc && d4 < n4) ? qrow4[d4] : float4(0.0f);
        y4[c] = float4(0.0f);
    }
    const device float* maskrow = (MASK != nullptr) ? MASK + (b * Tq + t) * Tkv : nullptr;
    float m = -INFINITY;
    float ssum_local = 0.0f;
    for (uint j = 0u; j < Tkv; ++j) {
        const device half4* krow4 = (const device half4*)(K + (b * Tkv + j) * KVD + kvh * HD);
        float acc = 0.0f;
        for (uint c = 0u; c < nc; ++c) {
            uint d4 = lane + 32u * c;
            if (d4 < n4) {
                float4 k4 = float4(krow4[d4]);
                acc += q4[c].x * k4.x + q4[c].y * k4.y + q4[c].z * k4.z + q4[c].w * k4.w;
            }
        }
        float s = simd_sum(acc) * scale;
        if (maskrow != nullptr) s += maskrow[j];
        if (s == -INFINITY) continue;
        if (s > m) {
            float e = exp(m - s);
            for (uint c = 0u; c < nc; ++c) y4[c] *= e;
            ssum_local *= e;
            m = s;
        }
        float w = exp(s - m);
        ssum_local += w;
        const device half4* vrow4 = (const device half4*)(V + (b * Tkv + j) * KVD + kvh * HD);
        for (uint c = 0u; c < nc; ++c) {
            uint d4 = lane + 32u * c;
            if (d4 < n4) y4[c] += w * float4(vrow4[d4]);
        }
    }
    float inv = (ssum_local > 0.0f) ? 1.0f / ssum_local : 0.0f;
    device float4* yrow4 = (device float4*)(Y + (b * Tq + t) * D + h * HD);
    if (n4 < 32u) {
        uint d4 = lane % n4;
        if (lane < n4) yrow4[d4] = y4[0] * inv;
    } else {
        for (uint c = 0u; c < nc; ++c) {
            uint d4 = lane + 32u * c;
            if (d4 < n4) yrow4[d4] = y4[c] * inv;
        }
    }
}

// Предфилл-внимание по f16-кэшу: Q/K/V — half-фрагменты 8×8, накопление float.
// NF — число 8-колоночных фрагментов головы (HD/8): 16 для HD ≤ 128,
// 32 для HD = 256 (Qwen3.5). Отдельные инстанциации: при HD=128 массивы
// остаются на 16 фрагментов (регистры — узкое место этого кернела).
template <uint NF>
kernel void attn_prefill_mma_f16_nf(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const float* MASK [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant uint& Tq [[buffer(5)]],
    constant uint& Tkv [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& KVH [[buffer(8)]],
    constant uint& HD [[buffer(9)]],
    threadgroup char* shm [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpt [[thread_position_in_threadgroup]])
{
    threadgroup float* sS = (threadgroup float*)shm;             // 2×(8×32) float
    threadgroup half* sP = (threadgroup half*)(sS + 2u * 8u * 32u);
    threadgroup float* sD = (threadgroup float*)(sP + 2u * 8u * 32u);
    threadgroup float* sL = sD + 2u * 64u;
    threadgroup half* sQh = (threadgroup half*)(sL + 16u);       // 16×HD half
    const uint lane = tpt.x % 32u;
    const uint sg = tpt.x / 32u;
    const uint h = tgpig.y;
    const uint kvh = h / (H / KVH);
    // База БЛОКА строк threadgroup'а (16 строк) — не путать с базой строк
    // simdgroup'а (qtg + sg*8): Q-тайл общий на threadgroup.
    const uint qtg = tgpig.x * 16u;
    const uint D = H * HD, KD = KVH * HD;
    const uint nf = HD / 8u;   // до 32 (HD до 256): фрагменты по 8 значений
    const float scale = 1.0f / sqrt((float)HD);
    threadgroup float* myS = sS + sg * 8u * 32u;
    threadgroup float* myD = sD + sg * 64u;
    // Q — через разделяемую память с построчной защитой: хвостовой блок
    // (Tq не кратно 16) обязан сохранить СВОИ строки, а не обнулить весь
    // 8-строчный фрагмент (иначе последние 1–7 запросов промпта считаются
    // по нулевому Q — было: tq=37, past=11 давало 10 % ошибки).
    for (uint e = tpt.x; e < 16u * HD; e += 64u) {
        const uint r = e / HD, c = e % HD;
        const uint qi = qtg + r;
        sQh[e] = (qi < Tq) ? Q[(ulong)qi * D + h * HD + c] : (half)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qa[NF];
    for (uint f = 0; f < nf; ++f) {
        simdgroup_load(qa[f], sQh + sg * 8u * HD + f * 8u, HD);
    }
    simdgroup_float8x8 o[NF];
    for (uint f = 0; f < nf; ++f) o[f] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -INFINITY, l_run = 0.0f;
    for (uint kb = 0; kb < Tkv; kb += 32u) {
        // Последний неполный блок ключей (Tkv не кратно 32) — отдельной
        // веткой: 8-строчный фрагмент, пересекающий Tkv, читал строки за
        // концом тензора (вне KV-кэша — мусор или NaN за концом MTLBuffer, а
        // 0·NaN в P·V портит весь столбец выхода). Здесь фрагменты после Tkv
        // пропускаются, а пересекающий сдвигается назад к Tkv-8 (его первые
        // столбцы дублируют ключи предыдущего фрагмента — их отсекает
        // softmax). Горячий цикл полных блоков не тронут: ветвление внутри
        // развёрнутых циклов стоило +40 % времени внимания.
        const bool tail = kb + 32u > Tkv;
        if (!tail) {
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
                for (uint f = 0; f < nf; ++f) {
                    simdgroup_half8x8 kf;
                    simdgroup_load(kf, K + (ulong)(kb + j * 8u) * KD + kvh * HD + f * 8u, KD, 0, true);
                    simdgroup_multiply_accumulate(s, qa[f], kf, s);
                }
                simdgroup_store(s, myS + j * 8u, 32u);
            }
        } else {
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.f);
                const uint kfr = kb + j * 8u;
                if (kfr < Tkv) {
                    const uint ks = min(kfr, Tkv - 8u);
                    for (uint f = 0; f < nf; ++f) {
                        simdgroup_half8x8 kf;
                        simdgroup_load(kf, K + (ulong)ks * KD + kvh * HD + f * 8u, KD, 0, true);
                        simdgroup_multiply_accumulate(s, qa[f], kf, s);
                    }
                }
                simdgroup_store(s, myS + j * 8u, 32u);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            const uint qi = qtg + sg * 8u + lane;
            float mx = m_run;
            for (uint j = 0; j < 32u; ++j) {
                const uint kf = kb + (j & ~7u);
                const uint kj = tail ? min(kf, Tkv - 8u) + (j & 7u) : kb + j;
                float v = myS[lane * 32u + j] * scale;
                if (kf >= Tkv || kj < kf || kj >= Tkv || qi >= Tq) v = -INFINITY;
                else if (MASK != nullptr) v += MASK[(ulong)qi * Tkv + kj];
                myS[lane * 32u + j] = v;
                mx = max(mx, v);
            }
            float l = 0.0f;
            for (uint j = 0; j < 32u; ++j) {
                const float v = myS[lane * 32u + j];
                const float p = (v == -INFINITY) ? 0.0f : exp(v - mx);
                sP[(sg * 8u + lane) * 32u + j] = (half)p;
                l += p;
            }
            const float corr = (m_run == -INFINITY) ? 0.0f : exp(m_run - mx);
            l_run = l_run * corr + l;
            m_run = mx;
            for (uint c = 0; c < 8u; ++c) myD[lane * 8u + c] = (c == lane) ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 cd;
        simdgroup_load(cd, myD, 8u);
        if (!tail) {
            for (uint f = 0; f < nf; ++f) {
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
                simdgroup_multiply(acc, cd, o[f]);
                for (uint j = 0; j < 4u; ++j) {
                    simdgroup_half8x8 pa, vb;
                    simdgroup_load(pa, sP + (sg * 8u) * 32u + j * 8u, 32u);
                    simdgroup_load(vb, V + (ulong)(kb + j * 8u) * KD + kvh * HD + f * 8u, KD);
                    simdgroup_multiply_accumulate(acc, pa, vb, acc);
                }
                o[f] = acc;
            }
        } else {
            for (uint f = 0; f < nf; ++f) {
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
                simdgroup_multiply(acc, cd, o[f]);
                for (uint j = 0; j < 4u; ++j) {
                    const uint kfr = kb + j * 8u;
                    if (kfr >= Tkv) break;
                    simdgroup_half8x8 pa, vb;
                    simdgroup_load(pa, sP + (sg * 8u) * 32u + j * 8u, 32u);
                    simdgroup_load(vb, V + (ulong)min(kfr, Tkv - 8u) * KD + kvh * HD + f * 8u, KD);
                    simdgroup_multiply_accumulate(acc, pa, vb, acc);
                }
                o[f] = acc;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 8u) sL[sg * 8u + lane] = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8u) {
        const float inv = sL[sg * 8u + lane];
        for (uint c = 0; c < 8u; ++c) myD[lane * 8u + c] = (c == lane) ? inv : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dinv;
    simdgroup_load(dinv, myD, 8u);
    for (uint f = 0; f < nf; ++f) {
        simdgroup_float8x8 res;
        simdgroup_multiply(res, dinv, o[f]);
        simdgroup_store(res, myS + (f % 4u) * 8u, 32u);
        if (f % 4u == 3u || f + 1u == nf) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint fbase = (f / 4u) * 4u;
            const uint fcount = f - fbase + 1u;
            for (uint e = lane; e < 8u * fcount * 8u; e += 32u) {
                const uint c = e % 8u;
                const uint t1 = e / 8u;
                const uint fi = t1 % fcount;
                const uint r = t1 / fcount;
                // Строка ВСЕГО threadgroup: свой блок строк этого simdgroup'а
                // — sg*8 + r (без sg*8 оба писали строки q0..q0+7: строки
                // 8..15 оставались нулями, «верхние» затирали верные).
                const uint qi = qtg + sg * 8u + r;
                if (qi < Tq) {
                    for (uint cc = 0; cc < 8u; ++cc)
                        Y[(ulong)qi * D + h * HD + (fbase + fi) * 8u + cc] = myS[r * 32u + fi * 8u + cc];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

typedef decltype(attn_prefill_mma_f16_nf<16>) attn_prefill_mma_f16_nf_t;

typedef decltype(mv_ks_t<0>) mv_ks_t_t;
template [[host_name("mv_ks_q4_k")]] kernel mv_ks_t_t mv_ks_t<0>;
template [[host_name("mv_ks_q6_k")]] kernel mv_ks_t_t mv_ks_t<1>;

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
template [[host_name("attn_prefill_mma_f16_nf16")]] kernel attn_prefill_mma_f16_nf_t attn_prefill_mma_f16_nf<16>;
template [[host_name("attn_prefill_mma_f16_nf32")]] kernel attn_prefill_mma_f16_nf_t attn_prefill_mma_f16_nf<32>;
