
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

// ════════════════════════════════════════════════════════════════════════════
// Ф8.1c: побитовый порт эталонного flash-attn VEC-пути (ggml-metal 5266f24da,
// fa.metal kernel_flash_attn_ext_pad / kernel_flash_attn_ext_vec /
// kernel_flash_attn_ext_vec_reduce) для f16 KV, DK=DV, маска f16.
//
// Оркал берёт этот путь при ne01 < 20 && ne00 % 32 == 0
// (ggml_metal_op_flash_attn_ext_use_vec) — то есть и для декода, и для
// короткого префилла. При ne11 < 1024 тюнинг fa_vec_pick даёт BASELINE:
// Q=1, NE=1; C=NCPSG=32; nsg=1; nwg=32 (ggml_metal_op_mul_mat… ops.cpp).
// Машинный код воспроизведён ДОСЛОВНО по живой ветке (уроки Q5_0/контракции):
// dead-ветки (sparse/sinks/bias/scap/квантовый деквант) опущены — их вырезает
// и оркальский компилятор при специализации function-constant'ов; живые
// арифметические выражения, порядок dot/simd_sum/fma/exp и раскладка shmem —
// байт-в-байт как у эталона.
//
// Пайплайн: [pad при ne11%32!=0] → vec (grid (tq, nh, nwg), пишет tmp с
// интерливингом по nwg + S/M в хвосте) → reduce (grid (nrows), сводит 33…
// т.е. nwg частичных онлайн-softmax'ов).
// ════════════════════════════════════════════════════════════════════════════

#define FA_MAXHALF ((half)65504.0f)

// оркальский FOR_UNROLL: полному развороту циклов КРИТИЧЕСКИ важно —
// без него Metal-компилятор даёт ДРУГОЕ округление накоплений (Ф8.1g:
// много-чанговый vec расходился на единицу ulp именно из-за `for` вместо
// `FOR_UNROLL`)
#define FA_FOR(x) _Pragma("clang loop unroll(full)") for (x)

struct fa_vec_args {
    int   ne01;   // число запросов (tq)
    int   ne02;   // число голов Q (nh)
    int   ne03;   // 1
    ulong nb01;   // байт: stride запроса = qd*4
    ulong nb02;   // байт: stride головы = tq*qd*4
    ulong nb03;
    int   ne11;   // длина KV
    int   ne_12_2; // nkv
    int   ne_12_3; // 1
    int   ns10;   // элементы на ключ = nb11/nb10 (у оракула — function constant;
                  // в его раскладке K [D,T,H] это DK, в нашей [T, NKV*D] — NKV*DK)
    ulong nb11;   // байт: stride ключа = dk*2
    ulong nb12;
    ulong nb13;
    int   ns20;   // элементы на ключ у V = nb21/nb20
    ulong nb21;
    ulong nb22;
    ulong nb23;
    int   ne31;   // число строк маски (tq)
    int   ne32;   // 1
    int   ne33;   // 1
    ulong nb31;   // байт: stride строки маски = tkv*2
    ulong nb32;
    ulong nb33;
    int   ne1;    // nh
    int   ne2;    // tq
    int   ne3;    // 1
    float scale;
};

// ── pad: хвостовой чанк C ключей в отдельный буфер (K, V, маска) ─────────────
struct fa_pad_args {
    int   ne11;
    int   ne_12_2;
    int   ne_12_3;
    ulong nb11; ulong nb12; ulong nb13;
    ulong nb21; ulong nb22; ulong nb23;
    int   ne31; int   ne32; int   ne33;
    ulong nb31; ulong nb32; ulong nb33;
};

template <int C>
kernel void fa_pad_k(
    constant fa_pad_args & args [[buffer(0)]],
    device const char * k [[buffer(1)]],
    device const char * v [[buffer(2)]],
    device const char * mask [[buffer(3)]],
    device       char * dst [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiitg [[thread_index_in_threadgroup]],
    ushort3 ntg   [[threads_per_threadgroup]])
{
    // C = FC_flash_attn_ext_pad_ncpsg: 32 для vec-пути, 64 для tile-пути

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            // данные одного ключа одной головы = nb12 байт (наша раскладка
            // interleaved: [T, NKV*D]); stride в pad-буфере остаётся nb11,
            // чтобы vec-кернел читал его тем же NS10 = nb11/2
            for (uint64_t i = tiitg; i < args.nb12; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb22; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    // FC_flash_attn_ext_pad_has_mask == true
    if (i2 < args.ne32 && i3 < args.ne33) {
        for (int ib = i1; ib < args.ne31; ib += C) {
            device const half * mask_src = (device const half *)(mask + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
            device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

            for (int i = tiitg; i < C; i += ntg.x) {
                if (i >= icp) {
                    mask_dst[i] = -FA_MAXHALF;
                } else {
                    mask_dst[i] = mask_src[i];
                }
            }
        }
    }
}

// ── vec: онлайн-softmax по чанкам C=32 ключей, NWG=32 workgroup'ов делят KV ──
// NE/Q — тюнинг fa_vec_pick (при ne11 >= 1024 оркал берёт tuned-инстанциации
// из таблицы ggml-metal-tuning.cpp для своего SKU); NSG — от ne11:
// nsg = 1 при ne11 < 2048, 2 при < 4096, иначе 4 (ggml-metal-ops.cpp).
template <short DK, short DV, short NE, short Q, short NSG, bool HAS_MASK, bool HAS_KVPAD>
kernel void fa_vec(
    constant fa_vec_args & args [[buffer(0)]],
    device const char * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device const char * mask [[buffer(4)]],
    device const char * sinks [[buffer(5)]],
    device const char * pad [[buffer(6)]],
    device       char * dst [[buffer(7)]],
    threadgroup  half * shmem_f16 [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
#define NWG  32
#define NS10 (args.ns10)
#define NS20 (args.ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = ((DK + 127)/128)*128; // PAD2(DK, 128)
    constexpr short PK4 = PK/4;

    constexpr short PV  = ((DV + 127)/128)*128;
    constexpr short PV4 = PV/4;

    constexpr short NW  = 32;
    // NE = fa_vec_baseline_ne(DK, DV): 128→1, 64→2
    constexpr short NL  = NW/NE;
    constexpr short C   = 32;  // OP_FLASH_ATTN_EXT_VEC_NCPSG
    constexpr short SH  = 4*Q*C;

    threadgroup half4  * sq4 = (threadgroup half4  *) (shmem_f16 + 0*PK);
    threadgroup float  * ss  = (threadgroup float  *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK);
    threadgroup float4 * ss4 = (threadgroup float4 *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK);
    threadgroup half   * sm  = (threadgroup half   *) (shmem_f16 +   sgitg*SH + 2*Q*C + Q*NSG*PK);
    // оракул: 2*sgitg*Q*PV (шаг в half; so4 — float4 ⇒ шаг simdgroup = Q*PV4 float4)
    threadgroup float4 * so4 = (threadgroup float4 *) (shmem_f16 + 2*sgitg*Q*PV + Q*NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*Q*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // загрузка Q в shared (f32 → half4 — оркал ОКРУГЛЯЕТ Q до f16 здесь)
    {
        FA_FOR (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            device const float4 * q4 = (device const float4 *) ((device const char *) q + qq*args.nb01);
            if (iq1_q < args.ne01) {
                for (short i = tiisg; i < PK4; i += NW) {
                    if (i < DK4) {
                        sq4[qq*PK4 + i] = (half4) q4[i];
                    } else {
                        sq4[qq*PK4 + i] = (half4) 0.0f;
                    }
                }
            } else {
                for (short i = tiisg; i < PK4; i += NW) {
                    sq4[qq*PK4 + i] = (half4) 0.0f;
                }
            }
        }
    }

    FA_FOR (short qq = 0; qq < Q; ++qq) {
        for (short i = 0; i < DV4/NL; ++i) {
            so4[qq*DV4 + i*NL] = (float4) 0.0f;
        }
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (float4) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q];
        float M[Q];
        FA_FOR (short qq = 0; qq < Q; ++qq) {
            S[qq] = 0.0f;
            M[qq] = -FLT_MAX/2;
        }

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm_base = (device const half *) (mask + iq1*Q*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            device const half * pm[Q];
            FA_FOR (short qq = 0; qq < Q; ++qq) {
                pm[qq] = pm_base + ((iq1*Q + qq) < args.ne01 ? qq*(args.nb31/sizeof(half)) : -iq1*Q*(args.nb31/sizeof(half)));
            }

            if (HAS_KVPAD && ic + C > args.ne11) {
                k    = pad;
                v    = (device const char *) k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = (device const char *) v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!HAS_MASK) {
                    FA_FOR (short qq = 0; qq < Q; ++qq) {
                        if (ic + tiisg >= args.ne11) {
                            sm[qq*C + tiisg] = -FA_MAXHALF;
                        }
                    }
                } else {
                    FA_FOR (short qq = 0; qq < Q; ++qq) {
                        pm[qq] = (device const half *) (mask) +
                            (iq1*Q + qq)*C +
                            (iq2%args.ne32)*(C*args.ne31) +
                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    }
                }

                ic = 0;
            }

            if (HAS_MASK) {
                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) < args.ne01) {
                        sm[qq*C + tiisg] = pm[qq][ic + tiisg];
                    } else {
                        sm[qq*C + tiisg] = -FA_MAXHALF;
                    }
                }
            } else {
                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) >= args.ne01) {
                        sm[qq*C + tiisg] = -FA_MAXHALF;
                    }
                }
            }

            // skip -INF mask
            {
                bool any_finite = false;
                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    if (simd_max(sm[qq*C + tiisg]) > -FA_MAXHALF) {
                        any_finite = true;
                    }
                }
                if (!any_finite) {
                    continue;
                }
            }

            // Q*K^T
            {
                device const half4 * pk4 = (device const half4 *) (k + ic*args.nb11);
                pk4 += ty*NS10/4 + tx;

                float mqk[Q][C/NE];
                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    FA_FOR (short cc = 0; cc < C/NE; ++cc) {
                        mqk[qq][cc] = 0.0f;
                    }
                }

                FA_FOR (short cc = 0; cc < C/NE; ++cc) {
                    FA_FOR (short ii = 0; ii < DK4/NL; ++ii) {
                        const half4 k_elem = pk4[cc*NE*NS10/4 + ii*NL];
                        FA_FOR (short qq = 0; qq < Q; ++qq) {
                            mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                        }
                    }

                    FA_FOR (short qq = 0; qq < Q; ++qq) {
                        if (NE == 1) {
                            mqk[qq][cc] = simd_sum(mqk[qq][cc]);
                        } else {
                            // simdgroup reduce (NE = 4): свёртка по NL-группам
                            if (NE <= 1) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
                            }
                            if (NE <= 2) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  8);
                            }
                            if (NE <= 4) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  4);
                            }
                            if (NE <= 8) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  2);
                            }
                            if (NE <= 16) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  1);
                            }

                            // broadcast
                            mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL*ty);
                        }
                    }
                }

                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    if (HAS_MASK) {
                        ss[qq*C + NE*tx + ty] = fma(mqk[qq][tx], args.scale, (float) sm[qq*C + NE*tx + ty]);
                    } else {
                        mqk[qq][tx] *= args.scale;
                        mqk[qq][tx] += (float) sm[qq*C + NE*tx + ty];
                        ss[qq*C + NE*tx + ty] = mqk[qq][tx];
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    const float m = M[qq];
                    const float s = ss[qq*C + tiisg];

                    M[qq] = simd_max(max(M[qq], s));

                    const float ms = exp(m - M[qq]);
                    const float vs = exp(s - M[qq]);

                    S[qq] = S[qq]*ms + simd_sum(vs);

                    ss[qq*C + tiisg] = vs;

                    // O = diag(ms)*O
                    if ((DV4/NL % NW == 0) || ty == 0) {
                        FA_FOR (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] *= ms;
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                float4 lo[Q][DV4/NL];
                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    FA_FOR (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[qq][ii] = 0.0f;
                    }
                }

                device const half4 * pv4 = (device const half4 *) (v + ic*args.nb21);
                pv4 += ty*NS20/4 + tx;

                FA_FOR (short cc = 0; cc < C/NE; ++cc) {
                    FA_FOR (short ii = 0; ii < DV4/NL; ++ii) {
                        const half4 v_elem = pv4[cc*NE*NS20/4 + ii*NL];
                        FA_FOR (short qq = 0; qq < Q; ++qq) {
                            lo[qq][ii] += float4(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                        }
                    }
                }

                FA_FOR (short qq = 0; qq < Q; ++qq) {
                    FA_FOR (short ii = 0; ii < DV4/NL; ++ii) {
                        if (NE > 1) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
                        }

                        if (NE > 2) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  8);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  8);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  8);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  8);
                        }

                        if (NE > 4) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  4);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  4);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  4);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  4);
                        }

                        if (NE > 8) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  2);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  2);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  2);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  2);
                        }

                        if (NE > 16) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  1);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  1);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  1);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  1);
                        }
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FA_FOR (short qq = 0; qq < Q; ++qq) {
                        FA_FOR (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] += lo[qq][ii];
                        }
                    }
                }
            }
        }

        if (tiisg == 0) {
            FA_FOR (short qq = 0; qq < Q; ++qq) {
                ss[2*qq + 0] = S[qq];
                ss[2*qq + 1] = M[qq];
            }
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce по simdgroup'ам (дословно оркал): NSG == 1 → не выполняется
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FA_FOR (short qq = 0; qq < Q; ++qq) {
                const float S0 = ss[                2*qq + 0];
                const float S1 = ss[r*(SH/2) +      2*qq + 0];

                const float M0 = ss[                2*qq + 1];
                const float M1 = ss[r*(SH/2) +      2*qq + 1];

                const float Mx  = max(M0, M1);

                const float ms0 = exp(M0 - Mx);
                const float ms1 = exp(M1 - Mx);

                const float Sx  = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[2*qq + 0] = Sx;
                    ss[2*qq + 1] = Mx;
                }

                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        FA_FOR (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            if (iq1_q >= args.ne01) {
                continue;
            }

            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1_q*args.ne1;

            const float Sval = NWG == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;

            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[qq*DV4 + i]*Sval;
            }

            if (NWG > 1) {
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = ss[2*qq + 0];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = ss[2*qq + 1];
                }
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

// ── reduce: сведение NWG частичных результатов (онлайн-softmax merge) ────────
template <short DV>
kernel void fa_vec_reduce(
    constant int & nrows_arg [[buffer(0)]],
    device  const char * htmp [[buffer(1)]],
    device        char * dst [[buffer(2)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
#define NWG 32
#define DV4 (DV/4)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)nrows_arg*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV4
}

// Инстанциации vec: baseline (Q=1, NE=baseline_ne, nsg=1) + tuned-конфиги,
// которые fa_vec_pick(M3 Max, F16) реально возвращает при ne11 >= 1024,
// кратно nsg ∈ {1,2,4} (nsg = 1 при ne11<2048, 2 при <4096, иначе 4):
//   dk128: (Q,NE) ∈ {(1,1), (2,2)}
//   dk64:  (Q,NE) ∈ {(1,2), (1,4), (2,4), (4,4)}
#define FA_VEC4(DK, NE, Q, NSG, BASE) \
    template [[host_name(BASE "_mask_pad")]] kernel decltype(fa_vec<DK, DK, NE, Q, NSG, true, true>)   fa_vec<DK, DK, NE, Q, NSG, true, true>; \
    template [[host_name(BASE "_mask")]]     kernel decltype(fa_vec<DK, DK, NE, Q, NSG, true, false>)  fa_vec<DK, DK, NE, Q, NSG, true, false>; \
    template [[host_name(BASE "_pad")]]      kernel decltype(fa_vec<DK, DK, NE, Q, NSG, false, true>)  fa_vec<DK, DK, NE, Q, NSG, false, true>; \
    template [[host_name(BASE)]]             kernel decltype(fa_vec<DK, DK, NE, Q, NSG, false, false>) fa_vec<DK, DK, NE, Q, NSG, false, false>;

FA_VEC4(128, 1, 1, 1, "fa_vec_dk128")
FA_VEC4(128, 1, 1, 2, "fa_vec_dk128_q1ne1_nsg2")
FA_VEC4(128, 1, 1, 4, "fa_vec_dk128_q1ne1_nsg4")
FA_VEC4(128, 2, 2, 1, "fa_vec_dk128_q2ne2")
FA_VEC4(128, 2, 2, 2, "fa_vec_dk128_q2ne2_nsg2")
FA_VEC4(128, 2, 2, 4, "fa_vec_dk128_q2ne2_nsg4")
FA_VEC4(64, 2, 1, 1, "fa_vec_dk64")
FA_VEC4(64, 2, 1, 2, "fa_vec_dk64_q1ne2_nsg2")
FA_VEC4(64, 2, 1, 4, "fa_vec_dk64_q1ne2_nsg4")
FA_VEC4(64, 4, 1, 1, "fa_vec_dk64_q1ne4")
FA_VEC4(64, 4, 1, 2, "fa_vec_dk64_q1ne4_nsg2")
FA_VEC4(64, 4, 1, 4, "fa_vec_dk64_q1ne4_nsg4")
FA_VEC4(64, 4, 2, 1, "fa_vec_dk64_q2ne4")
FA_VEC4(64, 4, 2, 2, "fa_vec_dk64_q2ne4_nsg2")
FA_VEC4(64, 4, 2, 4, "fa_vec_dk64_q2ne4_nsg4")
FA_VEC4(64, 4, 4, 1, "fa_vec_dk64_q4ne4")
FA_VEC4(64, 4, 4, 2, "fa_vec_dk64_q4ne4_nsg2")
FA_VEC4(64, 4, 4, 4, "fa_vec_dk64_q4ne4_nsg4")

template [[host_name("fa_vec_reduce_dv128")]] kernel decltype(fa_vec_reduce<128>) fa_vec_reduce<128>;
template [[host_name("fa_vec_reduce_dv64")]]  kernel decltype(fa_vec_reduce<64>)  fa_vec_reduce<64>;

template [[host_name("fa_vec_pad")]]  kernel decltype(fa_pad_k<32>) fa_pad_k<32>;
template [[host_name("fa_tile_pad")]] kernel decltype(fa_pad_k<64>) fa_pad_k<64>;

// маска f32 → f16 (0/-inf переходят точно; оркал получает маску уже в f16)
kernel void cvt_f32_f16(
    device const float* X [[buffer(0)]],
    device half* Y [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    Y[gid] = (half) X[gid];
}

// ════════════════════════════════════════════════════════════════════════════
// Ф8.1f: tile-путь flash attention (побитовый порт kernel_flash_attn_ext +
// kernel_flash_attn_ext_blk из ggml-metal 0.23.0). Оркал берёт его при
// !(ne01 < 20 && ne00%32==0), т.е. для префилла ne01 >= 20 (и для vec-случаев
// с ne00%32!=0). Параметры для dense-f16 KV: Q=OP_FLASH_ATTN_EXT_NQPSG=8,
// C=OP_FLASH_ATTN_EXT_NCPSG=64, NSG=4 (ne00<512), типы FA_TYPES (half-фрагменты,
// float-аккумуляция), прямое чтение K/V (is_same ветки), без sinks/bias/scap.
// NS10/NS20 — фактический stride ключа из args (в нашей interleaved-раскладке
// NKV*HD). Дословность формы IR обязательна (урок Ф8.1b).
// ════════════════════════════════════════════════════════════════════════════

#define FA_PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define FA_MIN(a, b) ((a) < (b) ? (a) : (b))

struct fa_tile_args {
    int   ne01;   // число запросов (tq)
    int   ne02;   // nh
    int   ne03;   // 1
    ulong nb01;   // байт: stride запроса = qd*4
    ulong nb02;   // байт: stride головы Q = hd*4
    ulong nb03;
    int   ne11;   // длина KV
    int   ne_12_2; // nkv
    int   ne_12_3; // 1
    int   ns10;   // элементы на ключ = nb11/2
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int   ns20;   // элементы на ключ V = nb21/2
    ulong nb21;
    ulong nb22;
    ulong nb23;
    int   ne31;   // строк маски (tq)
    int   ne32;   // 1
    int   ne33;   // 1
    ulong nb31;   // байт: stride строки маски = tkv*2
    ulong nb32;
    ulong nb33;
    int   ne1;    // nh
    int   ne2;    // tq
    int   ne3;    // 1
    float scale;
};

struct fa_blk_args {
    int   ne01;   // tq
    int   ne30;   // tkv (ширина маски)
    int   ne31;   // tq
    int   ne32;   // 1
    int   ne33;   // 1
    ulong nb31;
    ulong nb32;
    ulong nb33;
};

// скан блоков маски: 0 = весь -INF (пропустить), 1 = проверить, 2 = все нули
kernel void fa_tile_blk(
    constant fa_blk_args & args [[buffer(0)]],
    device const char * mask [[buffer(1)]],
    device       char * dst [[buffer(2)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int32_t Q = 8;  // FC_flash_attn_ext_blk_nqptg
    const int32_t C = 64; // FC_flash_attn_ext_blk_ncpsg

    constexpr short NW = 32;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  FA_MAXHALF;
        half mmax = -FA_MAXHALF;

        FA_FOR (short j = 0; j < Q; ++j) {
            FA_FOR (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -FA_MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

template <short DK, short DV, bool HAS_MASK, bool HAS_KVPAD, bool BC_MASK>
kernel void fa_tile(
    constant fa_tile_args & args [[buffer(0)]],
    device const char * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device const char * mask [[buffer(4)]],
    device const char * sinks [[buffer(5)]],
    device const char * pad [[buffer(6)]],
    device const char * blk [[buffer(7)]],
    device       char * dst [[buffer(8)]],
    threadgroup  half * shmem_f16 [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
#define NS10 (args.ns10)
#define NS20 (args.ns20)
#define NSG 4
#define Q 8
#define C 64

    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = FA_PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = 32;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup half  * sq  = (threadgroup half  *) (shmem_f16 + 0*T);
    threadgroup half4 * sq4 = (threadgroup half4 *) (shmem_f16 + 0*T);
    threadgroup float * so  = (threadgroup float *) (shmem_f16 + 0*T + Q*DK);
    threadgroup float4* so4 = (threadgroup float4*) (shmem_f16 + 0*T + Q*DK);
    threadgroup float * ss  = (threadgroup float *) (shmem_f16 + Q*T);
    threadgroup float2* ss2 = (threadgroup float2*) (shmem_f16 + Q*T);

    // маска в half2 лежит во второй половине строки ss (дословно оракалу)
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FA_FOR (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory
    FA_FOR (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (half4) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // zero out
    FA_FOR (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        const float slope = 1.0f;
        (void) slope;

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (HAS_KVPAD && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!HAS_MASK) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FA_FOR (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -FA_MAXHALF;
                            }
                        }
                    }
                } else {
                    FA_FOR (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (HAS_MASK) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FA_FOR (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FA_FOR (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (BC_MASK) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-FA_MAXHALF, -FA_MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FA_FOR (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            // Q*K^T — прямое чтение f16 KV (is_same-ветка оракала)
            {
                device      const half * pk = (device const half *) (k + ic*args.nb11);
                threadgroup const half * pq = sq;
                threadgroup       float* ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FA_FOR (short cc = 0; cc < NC; ++cc) {
                    simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

                    simdgroup_half8x8 mk[2];
                    simdgroup_half8x8 mq[2];

                    // note: too much unroll can tank the performance for large heads
                    #pragma unroll (FA_MIN(DK8/2, 4*NSG))
                    for (short i = 0; i < DK8/2; ++i) {
                        simdgroup_barrier(mem_flags::mem_none);

                        simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                        simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                        simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                        simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                        simdgroup_barrier(mem_flags::mem_none);

                        simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                        simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            FA_FOR (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (blk_cur != 2) {
                    s2 += float2(sm2[j*SH + tiisg]);
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FA_FOR (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V — прямое чтение f16 V
            {
                static_assert(PV8 % NSG == 0, "");

                constexpr short NO = PV8/NSG;

                simdgroup_float8x8 lo[NO];

                {
                    auto sot = so + 8*sgitg;

                    FA_FOR (short ii = 0; ii < NO; ++ii) {
                        simdgroup_load(lo[ii], sot, PV, 0, false);

                        sot += 8*NSG;
                    }
                }

                {
                    device const half * pv = (device const half *) (v + ic*args.nb21);

                    pv += 8*sgitg;

                    if (DV <= 64) {
                        FA_FOR (short cc = 0; cc < C/8; ++cc) {
                            simdgroup_float8x8 vs;
                            simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                            FA_FOR (short ii = 0; ii < NO/2; ++ii) {
                                simdgroup_half8x8 mv[2];

                                simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                            }

                            pv  += 8*NS20;
                        }
                    } else {
                        constexpr short NC = (C/8)/2;

                        FA_FOR (short cc = 0; cc < NC; ++cc) {
                            simdgroup_float8x8 vs[2];

                            simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                            simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                            FA_FOR (short ii = 0; ii < NO/2; ++ii) {
                                simdgroup_half8x8 mv[4];

                                simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                            }

                            pv  += 2*8*NS20;
                        }
                    }
                }

                {
                    auto sot = so + 8*sgitg;

                    FA_FOR (short ii = 0; ii < NO; ++ii) {
                        simdgroup_store(lo[ii], sot, PV, 0, false);

                        sot += 8*NSG;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // store to global memory
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FA_FOR (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
#undef NSG
#undef Q
#undef C
}

template [[host_name("fa_tile_dk128_mask")]]      kernel decltype(fa_tile<128, 128, true,  false, false>) fa_tile<128, 128, true,  false, false>;
template [[host_name("fa_tile_dk128_mask_pad")]]  kernel decltype(fa_tile<128, 128, true,  true,  false>) fa_tile<128, 128, true,  true,  false>;
template [[host_name("fa_tile_dk128_mask_bcm")]]      kernel decltype(fa_tile<128, 128, true,  false, true>) fa_tile<128, 128, true,  false, true>;
template [[host_name("fa_tile_dk128_mask_pad_bcm")]]  kernel decltype(fa_tile<128, 128, true,  true,  true>) fa_tile<128, 128, true,  true,  true>;
template [[host_name("fa_tile_dk64_mask")]]       kernel decltype(fa_tile<64, 64, true,  false, false>) fa_tile<64, 64, true,  false, false>;
template [[host_name("fa_tile_dk64_mask_pad")]]   kernel decltype(fa_tile<64, 64, true,  true,  false>) fa_tile<64, 64, true,  true,  false>;
template [[host_name("fa_tile_dk64_mask_bcm")]]       kernel decltype(fa_tile<64, 64, true,  false, true>) fa_tile<64, 64, true,  false, true>;
template [[host_name("fa_tile_dk64_mask_pad_bcm")]]   kernel decltype(fa_tile<64, 64, true,  true,  true>) fa_tile<64, 64, true,  true,  true>;
