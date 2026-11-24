
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
