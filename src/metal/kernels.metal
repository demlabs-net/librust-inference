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
    uint h = gid.y / 32u;
    uint lane = gid.y % 32u;
    uint d4 = lane % n4;
    uint t = gid.z;
    if (b >= B || h >= H || t >= Tq) return;
    uint kvh = h / n_rep;
    uint D = H * HD;
    uint KVD = KVH * HD;
    const float scale = 1.0f / sqrt((float)HD);
    const bool active = (lane < n4);
    const device float4* qrow4 = (const device float4*)(Q + (b * Tq + t) * D + h * HD);
    float4 q4 = qrow4[d4];
    const device float* maskrow = (MASK != nullptr) ? MASK + (b * Tq + t) * Tkv : nullptr;

    // online-softmax в ОДНОМ проходе: running max m, ssum, y4-накопление.
    // -inf-маска: skip (continue) ДО exp — нет NaN (exp(-inf - -inf)).
    float m = -INFINITY;
    float ssum_local = 0.0f;
    float4 y4 = 0.0f;
    for (uint j = 0u; j < Tkv; ++j) {
        const device float4* krow4 = (const device float4*)(K + (b * Tkv + j) * KVD + kvh * HD);
        float4 k4 = krow4[d4];
        float acc = (active) ? (q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w) : 0.0f;
        float s = simd_sum(acc) * scale;
        if (maskrow != nullptr) s += maskrow[j];
        if (s == -INFINITY) continue;
        if (s > m) {
            float e = exp(m - s); // < 1: пересчёт старых вкладов
            y4 *= e;
            ssum_local *= e;
            m = s;
        }
        float w = exp(s - m);
        ssum_local += w;
        const device float4* vrow4 = (const device float4*)(V + (b * Tkv + j) * KVD + kvh * HD);
        y4 += w * vrow4[d4];
    }
    // ssum_local у каждого лейна УЖЕ полный (w — broadcast из simd_sum) —
    // повторная simd_sum утроила бы его в 32 раза.
    float inv = (ssum_local > 0.0f) ? 1.0f / ssum_local : 0.0f;
    device float4* yrow4 = (device float4*)(Y + (b * Tq + t) * D + h * HD);
    yrow4[d4] = y4 * inv;
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
    // NEOX/halves (HF rotate-half): пары (i, i + HD/2)
    uint base = t * D + h * HD + i;
    float x0 = X[base];
    float x1 = X[base + HD / 2];
    Y[base] = x0 * c - x1 * s;
    Y[base + HD / 2] = x1 * c + x0 * s;
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
