
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
