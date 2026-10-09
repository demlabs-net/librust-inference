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

// C[m,n] = Σ A[m*K+k] · B[n*K+k], B in bf16 (ushort bits).
static inline float bf16_to_f32(device const ushort* p, uint i) {
    return as_type<float>((uint)p[i] << 16);
}

// ── Quantized GEMM / MV: A fp32 × B quantized [N, K], fp32 accumulate ───────
// Block layouts (ggml, reimplemented): Q4_0 18 B/32, Q8_0 34 B/32,
// Q4_K 144 B/256, Q6_K 210 B/256. Activations read directly from the block
// bytes; CPU reference: weights_map::dequant_f32 (tests).

static inline half load_half_unaligned(device const uchar* p) {
    // даже смещения блоков (18/34/144/210 кратны 2) — выравнивание ushort ок
    return as_type<half>(*(device const ushort*)p);
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

// ── Q8_0 GEMM m>1 на матричных блоках simdgroup (Ф8.2, prefill) ──────────────
// C[M,N] = A[M,K] · dequant(B[N,K])ᵀ. Threadgroup (128 потоков = 4
// simdgroup'а) считает тайл 32(M)×64(N); K идёт шагами по 32 (= один
// Q8_0-блок): A-тайл 32×32 и деквантованный B-тайл 64×32 кладутся в
// threadgroup-память (half), затем каждый simdgroup умножает свои 8×8
// фрагменты через simdgroup_multiply_accumulate (аппаратные матричные
// блоки Apple GPU). Прежний m8f4-кернел считал скалярно по 8 строк на поток
// и упирался в ALU (prefill Qwen3-8B ~35 tok/s против 328 у эталона).
#include <metal_simdgroup_matrix>
