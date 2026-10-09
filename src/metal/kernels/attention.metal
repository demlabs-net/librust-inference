
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
template [[host_name("attn_prefill_mma_f16_nf16")]] kernel attn_prefill_mma_f16_nf_t attn_prefill_mma_f16_nf<16>;
template [[host_name("attn_prefill_mma_f16_nf32")]] kernel attn_prefill_mma_f16_nf_t attn_prefill_mma_f16_nf<32>;
