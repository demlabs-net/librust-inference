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

// ── Q1_0 / Q2_0 / TQ1_0 / TQ2_0: точечное значение по позиции в блоке ────
// FMT: 0 — Q1_0 (18 Б/128, биты → ±d), 1 — Q2_0 (18 Б/64, 2 бита →
// (q−1)·d), 2 — TQ1_0 (54 Б/256: qs[48] по 3^n + qh[4]), 3 — TQ2_0
// (66 Б/256: 2 бита). Порядок TQ1_0 — как в dequantize_row_tq1_0:
// 32 байта × 5 (m-major), затем 16 байт × 5, затем qh[4] × 4.
template <int FMT>
static inline uint tern_block_elems() {
    return FMT == 0 ? 128u : (FMT == 1 ? 64u : 256u);
}
template <int FMT>
static inline uint tern_block_bytes() {
    return FMT == 2 ? 54u : (FMT == 3 ? 66u : 18u);
}
template <int FMT>
static inline float tern_value_at(device const uchar* blk, uint p) {
    if (FMT == 0) {
        const float d = (float)load_half_unaligned(blk);
        return ((blk[2u + p / 8u] >> (p % 8u)) & 1u) ? d : -d;
    } else if (FMT == 1) {
        const float d = (float)load_half_unaligned(blk);
        const uint q = (blk[2u + p / 4u] >> (2u * (p % 4u))) & 3u;
        return ((float)q - 1.0f) * d;
    } else if (FMT == 2) {
        const float d = (float)load_half_unaligned(blk + 52u);
        uchar byte;
        uint n;
        if (p < 160u) {
            n = p / 32u;
            byte = blk[p % 32u];
        } else if (p < 240u) {
            n = (p - 160u) / 16u;
            byte = blk[32u + (p - 160u) % 16u];
        } else {
            n = (p - 240u) / 4u;
            byte = blk[48u + (p - 240u) % 4u];
        }
        const uint pw = n == 0u ? 1u : (n == 1u ? 3u : (n == 2u ? 9u : (n == 3u ? 27u : 81u)));
        // Эталон: uint8_t q = byte·3^n (переполнение mod 256), xi = (q·3)>>8.
        const uint q = ((uint)byte * pw) & 0xFFu;
        const int xi = (int)((q * 3u) >> 8);
        return (float)(xi - 1) * d;
    } else {
        const float d = (float)load_half_unaligned(blk + 64u);
        const uint chunk = p / 128u;
        const uint l = (p % 128u) / 32u;
        const uint m = p % 32u;
        const uint q = (blk[chunk * 32u + m] >> (2u * l)) & 3u;
        return ((float)q - 1.0f) * d;
    }
}

// ── Q2_K / Q3_K ───────────────────────────────────────────────────────────
// Q2_K: scales[16] @0 (4 бита scale | 4 бита min), qs[64] @16, d @80, dmin @82.
// Q3_K: hmask[32] @0, qs[64] @32, scales[12] @96, d @108; 6-битные scale со
// знаком собираются из 12 байт (старшие 4 бита — в scales[8..12]).
// Подблок s (16 значений): shift = 2·((s%8)/2), half = s%2 (байты l+16·half),
// q += 32 на каждые 8 подблоков.
static inline void q3k_unpack_scales(device const uchar* s12, thread int sc[16]) {
    uint aux[4];
    for (int i = 0; i < 3; ++i) {
        aux[i] = (uint)s12[4 * i] | ((uint)s12[4 * i + 1] << 8)
               | ((uint)s12[4 * i + 2] << 16) | ((uint)s12[4 * i + 3] << 24);
    }
    const uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
    for (int i = 0; i < 16; ++i) {
        sc[i] = (int)(signed char)((aux[i / 4] >> (8 * (i % 4))) & 0xFF);
    }
}

// ── MXFP4/NVFP4 (OCP E2M1) ────────────────────────────────────────────────
// kvalues_fp4 ggml: «удвоенные» значения E2M1 (0,1,2,3,4,6,8,12,0,-1,…,-12);
// масштаб компенсирует удвоение (E8M0_TO_FP32_HALF / ue4m3_to_fp32·0.5).
constant float KV_FP4[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

/// UE4M3-скейл NVFP4 → float (порт ggml_ue4m3_to_fp32): unsigned, 4 бита
/// экспоненты (bias 7), 3 мантиссы; 0 и 0x7F — ноль; значение·0.5
/// (компенсация «удвоенных» kvalues_fp4).
static inline float ue4m3_to_fp32(uchar x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    const int exp = (int)((x >> 3u) & 0xFu);
    const float man = (float)(x & 0x7u);
    const float raw = exp == 0 ? ldexp(man, -9) : ldexp(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
}

/// E8M0-скейл MXFP4 → float (порт ggml_e8m0_to_fp32_half): d = 2^(e−128);
/// при e<2 — субнормальные степени (биты 0x00200000 << e).
static inline float e8m0_to_fp32_half(uchar e) {
    const uint bits = e < 2u ? (0x00200000u << e) : (((uint)e - 1u) << 23);
    return as_type<float>(bits);
}

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
// на позиции p: theta = p·powf(base, (-1/n_dims)·2k) — одним powf, как в
// оркальном Metal-кернеле, затем интерполяция частот по коридору corr_dims и
// масштаб cos/sin на mscale. Тождественные параметры (ext=0, freq_scale=1,
// attn=1) — прямой cos/sin от theta.
struct RopeYarn {
    float freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, theta_scale;
};
inline bool rope_yarn_identity(constant RopeYarn& y) {
    return y.ext_factor == 0.0f && y.freq_scale == 1.0f && y.attn_factor == 1.0f;
}
inline void rope_cs(float p, uint k, float theta, uint hd, constant RopeYarn& y,
                    thread float& c, thread float& s) {
    // Оркальный kernel_rope_norm/neox (ggml-metal 0.23.0) считает theta
    // ОДНИМ powf: theta = pos · powf(base, (-1/n_dims)·iw), iw = 2k. Ни
    // 1/powf, ни итеративное накопление (то было зеркалом CPU-кэша
    // ggml_rope_cache_init — на Metal он не выполняется): округление
    // отличается на ulp и ломает побитовый паритет Q/K при pos>0 (GLM m=12).
    const float inv_ndims = -1.0f / (float)hd;
    const float theta_acc = p * pow(theta, inv_ndims * (2.0f * (float)k));
    if (rope_yarn_identity(y)) {
        c = cos(theta_acc);
        s = sin(theta_acc);
        return;
    }
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

// ── Матвеки K-квантов: раскладка r2 и per-block combine (порт mul_mv) ───────
// Общие части kernel_mul_mv_q4_K_f32_impl / kernel_mul_mv_q6_K_f32_impl
// (0.4.0): активационный регистр лейна на суперблок + вклад суперблока в
// частичную сумму строки. Используются и plain-матвеками (mv_q4_k_r2,
// mv_q6_k_r2), и id-кернелами MoE (mv_id_*) — так mul_mat_id считает ровно
// тот же порядок по K, что эталонный mul_mat (его mv_id оборачивает ту же
// impl), и побитово совпадает с по-слотовым matvec.

// Q4_K: 32 лейна на строку — ix = lane/8 (фазы суперблоков, шаг 4),
// it = lane%8 → iq (половина qs/scales), ir (четверть uint16-чанка).
struct MvQ4KDecomp { uint ix, iq, ir; };
static inline MvQ4KDecomp mv_q4k_decomp(uint lane) {
    const uint ix = lane / 8u, it = lane % 8u;
    return MvQ4KDecomp{ix, it / 4u, it % 4u};
}
struct MvQ4KY { float yl[16], yh[16]; float4 sumy; };
static inline void mv_q4k_y(device const float* a, uint ib, thread const MvQ4KDecomp& d,
                            thread MvQ4KY& o) {
    device const float* y4 = a + (ulong)ib * 256u + 64u * d.iq + 8u * d.ir;
    o.sumy = {0.f, 0.f, 0.f, 0.f};
    for (uint l = 0; l < 8u; ++l) {
        o.yl[l + 0] = y4[l +   0]; o.sumy[0] += o.yl[l + 0];
        o.yl[l + 8] = y4[l +  32]; o.sumy[1] += o.yl[l + 8];
        o.yh[l + 0] = y4[l + 128]; o.sumy[2] += o.yh[l + 0];
        o.yh[l + 8] = y4[l + 160]; o.sumy[3] += o.yh[l + 8];
    }
}
// Вклад суперблока (144 Б) в частичную сумму строки данным лейном.
// ВАЖНО: накопление — ОДНИМ выражением `s += dh*(...) - dm*(...)`, как в
// эталонном kernel_mul_mv_q4_K_f32_impl. Прежняя форма (return + `sumf +=`
// в вызывающем коде) давала ДРУГУЮ FMA-контракцию clang: промежуточные acc
// в быстром кернеле на ~2 порядка крупнее финальной суммы (массовая
// компенсация), и разница контракции — до ~3e-5 на скаляр при финальных
// значениях ~2e-4. Это и был разрыв паритета на OLMoE.
static inline void mv_q4k_acc(device const uchar* xb, thread const MvQ4KDecomp& d,
                              thread const MvQ4KY& y, thread float& s) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    device const uchar* sp = xb + 4u + 2u * d.iq;
    uint16_t sc16[4];
    const uint16_t s0 = (uint16_t)(sp[0] | (sp[1] << 8));
    const uint16_t s2 = (uint16_t)(sp[4] | (sp[5] << 8));
    const uint16_t s4 = (uint16_t)(sp[8] | (sp[9] << 8));
    sc16[0] = s0 & kmask1;
    sc16[1] = s2 & kmask1;
    sc16[2] = (s4 & kmask2) | ((s0 & kmask3) >> 2);
    sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    device const uchar* q1 = xb + 16u + 32u * d.iq + 8u * d.ir;
    device const uchar* q2 = q1 + 64u;
    const float dm = (float)load_half_unaligned(xb + 2u);
    const float dh = (float)load_half_unaligned(xb);

    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
    for (uint l = 0; l < 4u; ++l) {
        const uint16_t a = (uint16_t)(q1[2 * l] | (q1[2 * l + 1] << 8));
        const uint16_t b = (uint16_t)(q2[2 * l] | (q2[2 * l + 1] << 8));
        acc1[0] += y.yl[2 * l + 0] * (float)(a & 0x000Fu);
        acc1[1] += y.yl[2 * l + 1] * (float)(a & 0x0F00u);
        acc1[2] += y.yl[2 * l + 8] * (float)(a & 0x00F0u);
        acc1[3] += y.yl[2 * l + 9] * (float)(a & 0xF000u);
        acc2[0] += y.yh[2 * l + 0] * (float)(b & 0x000Fu);
        acc2[1] += y.yh[2 * l + 1] * (float)(b & 0x0F00u);
        acc2[2] += y.yh[2 * l + 8] * (float)(b & 0x00F0u);
        acc2[3] += y.yh[2 * l + 9] * (float)(b & 0xF000u);
    }
    s += dh * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
               (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
               (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
               (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
         dm * (y.sumy[0] * sc8[2] + y.sumy[1] * sc8[3] +
               y.sumy[2] * sc8[6] + y.sumy[3] * sc8[7]);
}

// Q6_K: 16 лейнов на строку (tid = lane/2 — позиция, ix = lane%2 — чётность
// суперблоков), ip/il делят суперблок.
struct MvQ6KDecomp { uint ix, y_off, q_off_l, q_off_h, is; };
static inline MvQ6KDecomp mv_q6k_decomp(uint lane) {
    const uint tid = lane / 2u, ix = lane % 2u;
    const uint ip = tid / 8u, il = tid % 8u;
    const uint l0 = 4u * il;
    return MvQ6KDecomp{ix, 128u * ip + l0, 64u * ip + l0, 32u * ip + l0,
                       8u * ip + l0 / 16u};
}
static inline void mv_q6k_y(device const float* a, uint i, thread const MvQ6KDecomp& d,
                            thread float* yl) {
    device const float* y = a + (ulong)i * 256u + d.y_off;
    for (uint l = 0; l < 4u; ++l) {
        yl[4u * l + 0] = y[l +  0];
        yl[4u * l + 1] = y[l + 32];
        yl[4u * l + 2] = y[l + 64];
        yl[4u * l + 3] = y[l + 96];
    }
}
// Накопление — одним выражением, как эталонный kernel_mul_mv_q6_K_f32_impl
// (`sumf[row] += dh[0] * (...)`): см. комментарий у mv_q4k_acc про контракцию.
static inline void mv_q6k_acc(device const uchar* xb, thread const MvQ6KDecomp& d,
                              thread const float* yl, thread float& s) {
    device const uchar* q1 = xb + d.q_off_l;
    device const uchar* q2 = q1 + 32;
    device const uchar* qh = xb + 128u + d.q_off_h;
    device const char*  sc = (device const char*)(xb + 192u) + d.is;
    const float dh = (float)load_half_unaligned(xb + 208u);
    float4 sums = {0.f, 0.f, 0.f, 0.f};
    for (uint l = 0; l < 4u; ++l) {
        sums[0] += yl[4u * l + 0] * (float)((int8_t)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
        sums[1] += yl[4u * l + 1] * (float)((int8_t)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
        sums[2] += yl[4u * l + 2] * (float)((int8_t)((q1[l] >> 4) | ((qh[l] & 0x30)         )) - 32);
        sums[3] += yl[4u * l + 3] * (float)((int8_t)((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)   ) - 32);
    }
    s += dh * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
}
