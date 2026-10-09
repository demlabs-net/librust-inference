
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

// Fused GDN-хвост decode-шага (t=1): conv → silu → split → l2norm за один
// проход вместо четырёх кернелов и трёх копий slice_last. Арифметика каждого
// выхода побитово равна цепочке gdn_conv1d_f32 + silu_f32 + slice_last +
// gdn_l2norm_f32: conv копит j=0..K-1 в том же порядке, silu — x/(1+exp(-x)),
// Σx² по голове — последовательно i=0..HD-1 (копит поток 0, барьер, затем
// все HD потоков пишут y = x·inv). Одна threadgroup на голову: головы —
// отрезки длины HD поперёк [q | k | v], v-головы нормируются единичным
// множителем (инвариант прежнего вызова: l2norm применялся только к q и k).
kernel void gdn_conv_silu_split_norm_f32(
    device const float* X [[buffer(0)]],      // [C] — qkv текущего токена
    device const float* W [[buffer(1)]],      // [C, K]
    device const float* STATE [[buffer(2)]],  // [C, K-1]
    device float* Y [[buffer(3)]],            // [C] = [q | k | v]
    device float* NEW [[buffer(4)]],          // [C, K-1] — новый стейт
    constant uint& C [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& HD [[buffer(7)]],          // длина головы l2norm
    constant uint& QK [[buffer(8)]],          // Hk·HD — граница q/k
    constant float& eps [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]])
{
    const uint base = head * HD;
    if (base >= C) return;
    const bool normed = base < 2u * QK;
    threadgroup float sv[1024];
    threadgroup float sinv;

    if (i < HD) {
        const uint c = base + i;
        float acc = 0.0f;
        for (uint j = 0; j < K; ++j) {
            float v = (j < K - 1) ? STATE[c * (K - 1) + j] : X[c];
            acc += W[c * K + j] * v;
        }
        sv[i] = acc / (1.0f + exp(-acc));
        // Сдвиг стейта при T=1: новый = [старый[1..], x].
        for (uint j = 0; j < K - 1; ++j) {
            NEW[c * (K - 1) + j] = (j + 1 < K - 1) ? STATE[c * (K - 1) + j + 1] : X[c];
        }
    }
    if (normed) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i == 0u) {
            float ss = 0.0f;
            for (uint j = 0; j < HD; ++j) ss += sv[j] * sv[j];
            sinv = 1.0f / sqrt(ss + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < HD) Y[base + i] = sv[i] * (normed ? sinv : 1.0f);
}

// Гейты GDN на токен и v-голову: g = softplus(alpha + dt_bias) · ssm_a,
// beta = sigmoid(beta_raw). ALPHA/BETA [T, H]; G/B — выходы [T, H].
// BA [T, 2H]: alpha = BA[t·2H + h], beta = BA[t·2H + H + h] (одна GEMM на
// обе проекции — см. Gdn::w_ba).
kernel void gdn_gates_f32(
    device const float* BA [[buffer(0)]],
    device const float* DT_BIAS [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device float* G [[buffer(4)]],
    device float* B [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint t = i / H, h = i % H;
    device const float* row = BA + (ulong)t * 2u * H;
    float a = row[h] + DT_BIAS[h];
    // softplus с порогом (как ggml: при x > 20 — линейно)
    float sp = a > 20.0f ? a : log(1.0f + exp(a));
    G[i] = sp * A[h];
    B[i] = 1.0f / (1.0f + exp(-row[H + h]));
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
// Одна строка на simdgroup: lane 0 последовательно копит Σx² (тот же порядок
// сложения, что был у скалярной одно-поточной версии — паритет первичен),
// барьер, затем все 32 лейна пишут строку. Прежняя версия «один поток на
// строку» при декоде (rows=1) отдавала всю строку 6144 одному потоку:
// 1.2 мс/слой на 27B-гибриде вместо ~0.03.
kernel void gdn_norm_gated_f32(
    device const float* X [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& ROWS [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (r >= ROWS) return;
    device const float* x = X + (ulong)r * HD;
    device const float* z = Z + (ulong)r * HD;
    threadgroup float s_inv;
    if (lane == 0u) {
        float ss = 0.0f;
        for (uint i = 0; i < HD; ++i) ss += x[i] * x[i];
        s_inv = 1.0f / sqrt(ss / (float)HD + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = s_inv;
    for (uint i = lane; i < HD; i += 32u) {
        float zz = z[i];
        float silu = zz / (1.0f + exp(-zz));
        Y[(ulong)r * HD + i] = x[i] * inv * W[i] * silu;
    }
}

// Параллельная версия гейтовой RMS-нормы (Ф8.1): Σx² — по всем 32 лейнам
// (stride-32, тот же шаг, что у фазы записи), частичные суммы — simd_sum.
// Прежняя версия копила HD=6144 элементов ОДНИМ лейном последовательно: на
// декоде 27B (rows=1) это 0.16 мс/слой — латентность глобальных загрузок
// одного потока. Порядок суммирования изменён относительно прежней цепочки
// (класс MV_KS) — паритет подтверждён text_parity на моделях с GDN
// (27B-гибрид, Qwen3.5-4B); фоллбэк INFER_NG_SERIAL=1.
kernel void gdn_norm_gated_par_f32(
    device const float* X [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& ROWS [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (r >= ROWS) return;
    device const float* x = X + (ulong)r * HD;
    device const float* z = Z + (ulong)r * HD;
    float ss = 0.0f;
    for (uint i = lane; i < HD; i += 32u) ss += x[i] * x[i];
    ss = simd_sum(ss);
    const float inv = 1.0f / sqrt(ss / (float)HD + eps);
    for (uint i = lane; i < HD; i += 32u) {
        float zz = z[i];
        float silu = zz / (1.0f + exp(-zz));
        Y[(ulong)r * HD + i] = x[i] * inv * W[i] * silu;
    }
}

// То же, что gdn_norm_gated_par_f32, но строку обрабатывает threadgroup из
// 4 simdgroup (128 потоков): при декоде 27B rows=1, и одной simdgroup мало —
// 0.04 мс/слой против ~0.01. Σx²: частичная сумма simdgroup'а (simd_sum),
// затем четыре частичных складываются в фиксированном порядке s0+s1+s2+s3 —
// детерминировано. Фоллбэк INFER_NG_SERIAL=1 (прежний последовательный).
kernel void gdn_norm_gated_par4_f32(
    device const float* X [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& ROWS [[buffer(4)]],
    constant uint& HD [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint r [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    if (r >= ROWS) return;
    device const float* x = X + (ulong)r * HD;
    device const float* z = Z + (ulong)r * HD;
    threadgroup float sgs[4];
    float ss = 0.0f;
    for (uint i = tid; i < HD; i += 128u) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0u) sgs[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sqrt(((sgs[0] + sgs[1]) + (sgs[2] + sgs[3])) / (float)HD + eps);
    for (uint i = tid; i < HD; i += 128u) {
        float zz = z[i];
        float silu = zz / (1.0f + exp(-zz));
        Y[(ulong)r * HD + i] = x[i] * inv * W[i] * silu;
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
