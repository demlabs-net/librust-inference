
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
    // Как в эталоне: scale и add — РАЗНЫЕ кернелы ggml (ggml_mul/ggml_scale,
    // затем ggml_add), произведение округляется в f32 ДО сложения. Без
    // отключения контракции clang сливает выражение в FMA — расхождение
    // с эталоном на единицы ULP (на OLMoE это и разрыв паритета текста).
    #pragma clang fp contract(off)
    if (gid >= n) return;
    const float p = B[gid] * scale;
    Y[gid] = A[gid] + p;
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

// ── f32 → f16 (приведение Q к типу f16-кэша; Ф8.2f) ────────────────────────
kernel void cast_f32_f16(
    device const float* SRC [[buffer(0)]],
    device half* DST [[buffer(1)]],
    constant uint& N4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < N4) ((device half4*)DST)[gid] = half4(((device const float4*)SRC)[gid]);
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
