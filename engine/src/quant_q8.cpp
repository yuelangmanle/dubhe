// quant_q8.cpp — 量化激活 + 整数点积内核
//
// 为什么需要这条路径：
//   f32 路径每个权重元素都要"位解包 + 转 float + FMA"，实测只有 3.5~6 Gelem/s
//   （M4 内存带宽 ~120 GB/s，远未饱和）——纯算力瓶颈。把激活也量化成 int8 后，
//   可以用 ARM 的 SDOT（一条指令 16 次乘加），理论上有 4~8 倍提升。
//
// 数值约定：
//   权重 W ≈ d_w · q_w（K-quant 另有 -dmin·m 偏置）
//   激活 x ≈ d_a · q_a（每 32 个元素一块）
//   ⇒ Σ x·W = d_a·d_w · Σ q_a·q_w        （整数量化器里 Σ q_a·q_w 用 SDOT 算）
#include "internal.h"

#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define DUBHE_NEON 1
#endif

namespace dubhe {

namespace {

void quantize_row_q8_0(const float* x, block_q8_0* y, int64_t n) {
    for (int64_t i = 0; i < n / 32; i++) {
        const float* xb = x + i * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) {
            const float a = std::fabs(xb[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < 32; j++) {
            int v = (int)std::lround(xb[j] * id);
            v = v < -128 ? -128 : (v > 127 ? 127 : v);
            y[i].qs[j] = (int8_t)v;
        }
    }
}

}  // namespace

bool act_q8_supported(uint32_t wtype) {
    switch (wtype) {
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1:
        case T_Q8_0: case T_Q4_K: case T_Q6_K:
            return true;
        // TODO(M1): Q5_K 的整数内核尚未实现，先走 f32 路径，避免静默算错
        default:
            return false;
    }
}

size_t act_q8_bytes(uint32_t wtype, int64_t n) {
    (void)wtype;
    return (size_t)(n / 32) * sizeof(block_q8_0);
}

void quantize_act(uint32_t wtype, const float* x, void* dst, int64_t n) {
    (void)wtype;
    quantize_row_q8_0(x, (block_q8_0*)dst, n);
}

// ---------------------------------------------------------------- 内核
#if defined(DUBHE_NEON)

namespace {

inline int32_t hsum_s32(int32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_s32(v);
#else
    int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(s, s), 0);
#endif
}

inline int32x4_t dot16(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpadalq_s16(vpadalq_s16(acc, p0), p1);
#endif
}

inline int32_t sum32(const block_q8_0* a) {
    int32x4_t s = vpadalq_s16(vdupq_n_s32(0), vpaddlq_s8(vld1q_s8(a->qs)));
    s = vpadalq_s16(s, vpaddlq_s8(vld1q_s8(a->qs + 16)));
    return hsum_s32(s);
}

// 把一个 32 位 qh 的低 16 位 / 高 16 位分别摊成 16 个 lane 的 bit4
inline void spread_qh(uint32_t qh, int8x16_t sh, uint8x16_t one, uint8x16_t* lo, uint8x16_t* hi) {
    const uint8x16_t lo_src = vcombine_u8(vdup_n_u8((uint8_t)(qh & 0xFF)),
                                          vdup_n_u8((uint8_t)((qh >> 8) & 0xFF)));
    const uint8x16_t hi_src = vcombine_u8(vdup_n_u8((uint8_t)((qh >> 16) & 0xFF)),
                                          vdup_n_u8((uint8_t)((qh >> 24) & 0xFF)));
    *lo = vshlq_n_u8(vandq_u8(vshlq_u8(lo_src, sh), one), 4);
    *hi = vshlq_n_u8(vandq_u8(vshlq_u8(hi_src, sh), one), 4);
}

}  // namespace

float vec_dot_q8(uint32_t wtype, const void* xq, const void* row, int64_t n) {
    const block_q8_0* a = (const block_q8_0*)xq;
    float sum = 0.f;

    switch (wtype) {
        case T_Q8_0: {
            const block_q8_0* b = (const block_q8_0*)row;
            for (int64_t i = 0; i < n / 32; i++) {
                int32x4_t c = vdupq_n_s32(0);
                c = dot16(c, vld1q_s8(a[i].qs), vld1q_s8(b[i].qs));
                c = dot16(c, vld1q_s8(a[i].qs + 16), vld1q_s8(b[i].qs + 16));
                sum += fp16_to_fp32(a[i].d) * fp16_to_fp32(b[i].d) * (float)hsum_s32(c);
            }
            return sum;
        }
        case T_Q4_0: {
            const block_q4_0* b = (const block_q4_0*)row;
            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t s8 = vdupq_n_s8(8);
            for (int64_t i = 0; i < n / 32; i++) {
                const uint8x16_t q = vld1q_u8(b[i].qs);
                const int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4b)), s8);
                const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), s8);
                int32x4_t c = vdupq_n_s32(0);
                c = dot16(c, q0, vld1q_s8(a[i].qs));
                c = dot16(c, q1, vld1q_s8(a[i].qs + 16));
                sum += fp16_to_fp32(a[i].d) * fp16_to_fp32(b[i].d) * (float)hsum_s32(c);
            }
            return sum;
        }
        case T_Q5_0: {
            const block_q5_0* b = (const block_q5_0*)row;
            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const uint8x16_t one = vdupq_n_u8(1);
            const int8x16_t s16 = vdupq_n_s8(16);
            const int8x16_t sh = {0, -1, -2, -3, -4, -5, -6, -7,
                                  0, -1, -2, -3, -4, -5, -6, -7};
            for (int64_t i = 0; i < n / 32; i++) {
                uint32_t qh;
                std::memcpy(&qh, b[i].qh, 4);
                uint8x16_t lo_bit, hi_bit;
                spread_qh(qh, sh, one, &lo_bit, &hi_bit);
                const uint8x16_t q = vld1q_u8(b[i].qs);
                const int8x16_t q0 =
                    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q, m4b), lo_bit)), s16);
                const int8x16_t q1 =
                    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q, 4), hi_bit)), s16);
                int32x4_t c = vdupq_n_s32(0);
                c = dot16(c, q0, vld1q_s8(a[i].qs));
                c = dot16(c, q1, vld1q_s8(a[i].qs + 16));
                sum += fp16_to_fp32(a[i].d) * fp16_to_fp32(b[i].d) * (float)hsum_s32(c);
            }
            return sum;
        }
        case T_Q4_1: {
            const block_q4_1* b = (const block_q4_1*)row;
            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            for (int64_t i = 0; i < n / 32; i++) {
                const uint8x16_t q = vld1q_u8(b[i].qs);
                const int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(q, m4b));
                const int8x16_t q1 = vreinterpretq_s8_u8(vshrq_n_u8(q, 4));
                int32x4_t c = vdupq_n_s32(0);
                c = dot16(c, q0, vld1q_s8(a[i].qs));
                c = dot16(c, q1, vld1q_s8(a[i].qs + 16));
                const float da = fp16_to_fp32(a[i].d);
                sum += da * (fp16_to_fp32(b[i].d) * (float)hsum_s32(c) +
                             fp16_to_fp32(b[i].m) * (float)sum32(&a[i]));
            }
            return sum;
        }
        case T_Q5_1: {
            const block_q5_1* b = (const block_q5_1*)row;
            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const uint8x16_t one = vdupq_n_u8(1);
            const int8x16_t sh = {0, -1, -2, -3, -4, -5, -6, -7,
                                  0, -1, -2, -3, -4, -5, -6, -7};
            for (int64_t i = 0; i < n / 32; i++) {
                uint32_t qh;
                std::memcpy(&qh, b[i].qh, 4);
                uint8x16_t lo_bit, hi_bit;
                spread_qh(qh, sh, one, &lo_bit, &hi_bit);
                const uint8x16_t q = vld1q_u8(b[i].qs);
                const int8x16_t q0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q, m4b), lo_bit));
                const int8x16_t q1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q, 4), hi_bit));
                int32x4_t c = vdupq_n_s32(0);
                c = dot16(c, q0, vld1q_s8(a[i].qs));
                c = dot16(c, q1, vld1q_s8(a[i].qs + 16));
                const float da = fp16_to_fp32(a[i].d);
                sum += da * (fp16_to_fp32(b[i].d) * (float)hsum_s32(c) +
                             fp16_to_fp32(b[i].m) * (float)sum32(&a[i]));
            }
            return sum;
        }
        case T_Q6_K: {
            // 布局：256 个元素 = 2 个 128 组；每组内 q1..q4 各 32 个，
            // 每 32 个又按 16-lane 用两个不同的 scale（sc[0..1] / sc[2..3] …）。
            const block_q6_K* b = (const block_q6_K*)row;
            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const uint8x16_t m2b = vdupq_n_u8(0x03);
            const int8x16_t s32 = vdupq_n_s8(32);
            for (int64_t i = 0; i < n / 256; i++) {
                const block_q8_0* ag = a + i * 8;
                float bsum = 0.f;
                for (int nb = 0; nb < 2; nb++) {
                    const uint8_t* ql = b[i].ql + nb * 64;
                    const uint8_t* qh = b[i].qh + nb * 32;
                    const int8_t* sc = b[i].scales + nb * 8;
                    const block_q8_0* ab = ag + nb * 4;

                    // 每个 qg 的 32 个元素用同一个激活块，两个 16-lane 各用一个 scale
                    int32x4_t c[4];
                    for (int k = 0; k < 4; k++) c[k] = vdupq_n_s32(0);
                    for (int h = 0; h < 2; h++) {
                        const uint8x16_t qlv = vld1q_u8(ql + h * 16);
                        const uint8x16_t qhv = vld1q_u8(qh + h * 16);
                        const uint8x16_t qlv32 = vld1q_u8(ql + 32 + h * 16);
                        const int8x16_t q1 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(qlv, m4b), vshlq_n_u8(vandq_u8(qhv, m2b), 4))), s32);
                        const int8x16_t q2 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(qlv32, m4b),
                                vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), m2b), 4))), s32);
                        const int8x16_t q3 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8(qlv, 4),
                                vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), m2b), 4))), s32);
                        const int8x16_t q4 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8(qlv32, 4),
                                vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), m2b), 4))), s32);

                        const int8x16_t a1 = vld1q_s8(ab[0].qs + h * 16);
                        const int8x16_t a2 = vld1q_s8(ab[1].qs + h * 16);
                        const int8x16_t a3 = vld1q_s8(ab[2].qs + h * 16);
                        const int8x16_t a4 = vld1q_s8(ab[3].qs + h * 16);

                        c[0] = vmlaq_n_s32(c[0], dot16(vdupq_n_s32(0), q1, a1), sc[h + 0]);
                        c[1] = vmlaq_n_s32(c[1], dot16(vdupq_n_s32(0), q2, a2), sc[h + 2]);
                        c[2] = vmlaq_n_s32(c[2], dot16(vdupq_n_s32(0), q3, a3), sc[h + 4]);
                        c[3] = vmlaq_n_s32(c[3], dot16(vdupq_n_s32(0), q4, a4), sc[h + 6]);
                    }
                    for (int k = 0; k < 4; k++)
                        bsum += fp16_to_fp32(ab[k].d) * (float)hsum_s32(c[k]);
                }
                sum += fp16_to_fp32(b[i].d) * bsum;
            }
            return sum;
        }
        case T_Q4_K: {
            // 256 个元素 = 4 个 64 组；每组低半字节 32 个 + 高半字节 32 个，
            // 各自一个 (scale, min)，正好各占一个 32 元激活块。
            const block_q4_K* b = (const block_q4_K*)row;
            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const float dm = fp16_to_fp32(b[i].dmin);
                const block_q8_0* ag = a + i * 8;
                float bsum = 0.f;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(2 * j + 0, b[i].scales, &sc1, &m1);
                    get_scale_min_k4(2 * j + 1, b[i].scales, &sc2, &m2);
                    const uint8x16_t qa = vld1q_u8(b[i].qs + j * 32);
                    const uint8x16_t qb = vld1q_u8(b[i].qs + j * 32 + 16);
                    const int8x16_t lo_a = vreinterpretq_s8_u8(vandq_u8(qa, m4b));
                    const int8x16_t lo_b = vreinterpretq_s8_u8(vandq_u8(qb, m4b));
                    const int8x16_t hi_a = vreinterpretq_s8_u8(vshrq_n_u8(qa, 4));
                    const int8x16_t hi_b = vreinterpretq_s8_u8(vshrq_n_u8(qb, 4));

                    const block_q8_0* a0 = ag + 2 * j + 0;
                    const block_q8_0* a1 = ag + 2 * j + 1;
                    int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0);
                    c0 = dot16(c0, lo_a, vld1q_s8(a0->qs));
                    c0 = dot16(c0, lo_b, vld1q_s8(a0->qs + 16));
                    c1 = dot16(c1, hi_a, vld1q_s8(a1->qs));
                    c1 = dot16(c1, hi_b, vld1q_s8(a1->qs + 16));

                    const float da0 = fp16_to_fp32(a0->d);
                    const float da1 = fp16_to_fp32(a1->d);
                    bsum += da0 * (d * (float)sc1 * (float)hsum_s32(c0) -
                                   dm * (float)m1 * (float)sum32(a0));
                    bsum += da1 * (d * (float)sc2 * (float)hsum_s32(c1) -
                                   dm * (float)m2 * (float)sum32(a1));
                }
                sum += bsum;
            }
            return sum;
        }
        default:
            return 0.f;
    }
}

#else  // 非 NEON 平台暂时走 f32 路径

float vec_dot_q8(uint32_t, const void*, const void*, int64_t) { return 0.f; }

#endif

}  // namespace dubhe
