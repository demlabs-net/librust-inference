
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
