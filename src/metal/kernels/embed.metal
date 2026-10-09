
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
