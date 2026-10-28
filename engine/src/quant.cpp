// quant.cpp — 反量化与点积
//
// 布局与 llama.cpp 的 ggml 完全一致（这是 GGUF 的事实标准）。
// 只实现本项目需要的类型；其余在加载期直接拒绝，不做"静默降级"。
#include "internal.h"

namespace dubhe {

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---------------------------------------------------------------- dequant
void dequant_row(uint32_t type, const void* src, float* dst, int64_t n) {
    switch (type) {
        case T_F32:
            std::memcpy(dst, src, (size_t)n * 4);
            return;
        case T_F16: {
            const uint16_t* s = (const uint16_t*)src;
            for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_fp32(s[i]);
            return;
        }
        case T_BF16: {
            const uint16_t* s = (const uint16_t*)src;
            for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_fp32(s[i]);
            return;
        }
        case T_Q4_0: {
            const block_q4_0* b = (const block_q4_0*)src;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                for (int j = 0; j < 16; j++) {
                    dst[i * 32 + j]      = ((b[i].qs[j] & 0x0F) - 8) * d;
                    dst[i * 32 + j + 16] = ((b[i].qs[j] >> 4) - 8) * d;
                }
            }
            return;
        }
        case T_Q5_0: {
            const block_q5_0* b = (const block_q5_0*)src;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const uint32_t qh = (uint32_t)b[i].qh[0] | ((uint32_t)b[i].qh[1] << 8) |
                                    ((uint32_t)b[i].qh[2] << 16) | ((uint32_t)b[i].qh[3] << 24);
                float* y = dst + i * 32;
                for (int j = 0; j < 16; j++) {
                    const uint8_t xh0 = (uint8_t)(((qh >> j) & 1) << 4);
                    const uint8_t xh1 = (uint8_t)(((qh >> (j + 16)) & 1) << 4);
                    y[j]      = (float)((int)((b[i].qs[j] & 0x0F) | xh0) - 16) * d;
                    y[j + 16] = (float)((int)((b[i].qs[j] >> 4) | xh1) - 16) * d;
                }
            }
            return;
        }
        case T_Q5_1: {
            const block_q5_1* b = (const block_q5_1*)src;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const float m = fp16_to_fp32(b[i].m);
                const uint32_t qh = (uint32_t)b[i].qh[0] | ((uint32_t)b[i].qh[1] << 8) |
                                    ((uint32_t)b[i].qh[2] << 16) | ((uint32_t)b[i].qh[3] << 24);
                float* y = dst + i * 32;
                for (int j = 0; j < 16; j++) {
                    const uint8_t xh0 = (uint8_t)(((qh >> j) & 1) << 4);
                    const uint8_t xh1 = (uint8_t)(((qh >> (j + 16)) & 1) << 4);
                    y[j]      = (float)((int)((b[i].qs[j] & 0x0F) | xh0)) * d + m;
                    y[j + 16] = (float)((int)((b[i].qs[j] >> 4) | xh1)) * d + m;
                }
            }
            return;
        }
        case T_Q4_1: {
            const block_q4_1* b = (const block_q4_1*)src;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const float m = fp16_to_fp32(b[i].m);
                float* y = dst + i * 32;
                for (int j = 0; j < 16; j++) {
                    y[j]      = (float)(b[i].qs[j] & 0x0F) * d + m;
                    y[j + 16] = (float)(b[i].qs[j] >> 4) * d + m;
                }
            }
            return;
        }
        case T_Q8_0: {
            const block_q8_0* b = (const block_q8_0*)src;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                for (int j = 0; j < 32; j++) dst[i * 32 + j] = b[i].qs[j] * d;
            }
            return;
        }
        case T_Q4_K: {
            const block_q4_K* b = (const block_q4_K*)src;
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
                const uint8_t* q = b[i].qs;
                float* y = dst + i * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                    const float d1 = d * sc, m1 = mn * m;
                    get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                    const float d2 = d * sc, m2 = mn * m;
                    for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0x0F) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    is += 2;
                }
            }
            return;
        }
        case T_Q5_K: {
            const block_q5_K* b = (const block_q5_K*)src;
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
                const uint8_t* ql = b[i].qs;
                const uint8_t* qh = b[i].qh;
                float* y = dst + i * 256;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                    const float d1 = d * sc, m1 = mn * m;
                    get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                    const float d2 = d * sc, m2 = mn * m;
                    for (int l = 0; l < 32; l++)
                        *y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++)
                        *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                    ql += 32;
                    is += 2;
                    u1 = (uint8_t)(u1 << 2);
                    u2 = (uint8_t)(u2 << 2);
                }
            }
            return;
        }
        case T_Q6_K: {
            const block_q6_K* b = (const block_q6_K*)src;
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const uint8_t* ql = b[i].ql;
                const uint8_t* qh = b[i].qh;
                const int8_t*  sc = b[i].scales;
                float* y = dst + i * 256;
                for (int nb = 0; nb < 2; nb++) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l + 0]  = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    y += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return;
        }
        default:
            for (int64_t i = 0; i < n; i++) dst[i] = 0.f;
            return;
    }
}

// ---------------------------------------------------------------- dot
float vec_dot(uint32_t type, const float* x, const void* row, int64_t n) {
    float sum = 0.f;
    switch (type) {
        case T_F32: {
            const float* w = (const float*)row;
            for (int64_t i = 0; i < n; i++) sum += x[i] * w[i];
            return sum;
        }
        case T_F16: {
            const uint16_t* w = (const uint16_t*)row;
            for (int64_t i = 0; i < n; i++) sum += x[i] * fp16_to_fp32(w[i]);
            return sum;
        }
        case T_BF16: {
            const uint16_t* w = (const uint16_t*)row;
            for (int64_t i = 0; i < n; i++) sum += x[i] * bf16_to_fp32(w[i]);
            return sum;
        }
        case T_Q4_0: {
            const block_q4_0* b = (const block_q4_0*)row;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                float s = 0.f;
                for (int j = 0; j < 16; j++) {
                    s += x[i * 32 + j] * (float)((b[i].qs[j] & 0x0F) - 8);
                    s += x[i * 32 + j + 16] * (float)((b[i].qs[j] >> 4) - 8);
                }
                sum += d * s;
            }
            return sum;
        }
        case T_Q5_0: {
            const block_q5_0* b = (const block_q5_0*)row;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const uint32_t qh = (uint32_t)b[i].qh[0] | ((uint32_t)b[i].qh[1] << 8) |
                                    ((uint32_t)b[i].qh[2] << 16) | ((uint32_t)b[i].qh[3] << 24);
                const float* xb = x + i * 32;
                float s = 0.f;
                for (int j = 0; j < 16; j++) {
                    const int q0 = (int)((b[i].qs[j] & 0x0F) | (uint8_t)(((qh >> j) & 1) << 4)) - 16;
                    const int q1 = (int)((b[i].qs[j] >> 4) | (uint8_t)(((qh >> (j + 16)) & 1) << 4)) - 16;
                    s += xb[j] * (float)q0;
                    s += xb[j + 16] * (float)q1;
                }
                sum += d * s;
            }
            return sum;
        }
        case T_Q5_1: {
            const block_q5_1* b = (const block_q5_1*)row;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const float m = fp16_to_fp32(b[i].m);
                const uint32_t qh = (uint32_t)b[i].qh[0] | ((uint32_t)b[i].qh[1] << 8) |
                                    ((uint32_t)b[i].qh[2] << 16) | ((uint32_t)b[i].qh[3] << 24);
                const float* xb = x + i * 32;
                float s = 0.f, sx = 0.f;
                for (int j = 0; j < 16; j++) {
                    const int q0 = (int)((b[i].qs[j] & 0x0F) | (uint8_t)(((qh >> j) & 1) << 4));
                    const int q1 = (int)((b[i].qs[j] >> 4) | (uint8_t)(((qh >> (j + 16)) & 1) << 4));
                    s += xb[j] * (float)q0;
                    s += xb[j + 16] * (float)q1;
                    sx += xb[j] + xb[j + 16];
                }
                sum += d * s + m * sx;
            }
            return sum;
        }
        case T_Q4_1: {
            const block_q4_1* b = (const block_q4_1*)row;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const float m = fp16_to_fp32(b[i].m);
                const float* xb = x + i * 32;
                float s = 0.f, sx = 0.f;
                for (int j = 0; j < 16; j++) {
                    s += xb[j] * (float)(b[i].qs[j] & 0x0F);
                    s += xb[j + 16] * (float)(b[i].qs[j] >> 4);
                    sx += xb[j] + xb[j + 16];
                }
                sum += d * s + m * sx;
            }
            return sum;
        }
        case T_Q8_0: {
            const block_q8_0* b = (const block_q8_0*)row;
            for (int64_t i = 0; i < n / 32; i++) {
                const float d = fp16_to_fp32(b[i].d);
                float s = 0.f;
                for (int j = 0; j < 32; j++) s += x[i * 32 + j] * (float)b[i].qs[j];
                sum += d * s;
            }
            return sum;
        }
        case T_Q4_K: {
            const block_q4_K* b = (const block_q4_K*)row;
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
                const uint8_t* q = b[i].qs;
                const float* xb = x + i * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                    const float d1 = d * sc, m1 = mn * m;
                    get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                    const float d2 = d * sc, m2 = mn * m;
                    float s1 = 0.f, s2 = 0.f, sx1 = 0.f, sx2 = 0.f;
                    for (int l = 0; l < 32; l++) {
                        s1  += xb[l] * (float)(q[l] & 0x0F);
                        sx1 += xb[l];
                        s2  += xb[l + 32] * (float)(q[l] >> 4);
                        sx2 += xb[l + 32];
                    }
                    sum += d1 * s1 - m1 * sx1 + d2 * s2 - m2 * sx2;
                    q += 32;
                    xb += 64;
                    is += 2;
                }
            }
            return sum;
        }
        case T_Q5_K: {
            const block_q5_K* b = (const block_q5_K*)row;
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
                const uint8_t* ql = b[i].qs;
                const uint8_t* qh = b[i].qh;
                const float* xb = x + i * 256;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                    const float d1 = d * sc, m1 = mn * m;
                    get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                    const float d2 = d * sc, m2 = mn * m;
                    float s1 = 0.f, s2 = 0.f, sx1 = 0.f, sx2 = 0.f;
                    for (int l = 0; l < 32; l++) {
                        s1  += xb[l] * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0));
                        sx1 += xb[l];
                        s2  += xb[l + 32] * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0));
                        sx2 += xb[l + 32];
                    }
                    sum += d1 * s1 - m1 * sx1 + d2 * s2 - m2 * sx2;
                    ql += 32;
                    xb += 64;
                    is += 2;
                    u1 = (uint8_t)(u1 << 2);
                    u2 = (uint8_t)(u2 << 2);
                }
            }
            return sum;
        }
        case T_Q6_K: {
            const block_q6_K* b = (const block_q6_K*)row;
            for (int64_t i = 0; i < n / 256; i++) {
                const float d = fp16_to_fp32(b[i].d);
                const uint8_t* ql = b[i].ql;
                const uint8_t* qh = b[i].qh;
                const int8_t*  sc = b[i].scales;
                const float* xb = x + i * 256;
                for (int nb = 0; nb < 2; nb++) {
                    float s = 0.f;
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        s += xb[l + 0]  * (float)sc[is + 0] * (float)q1;
                        s += xb[l + 32] * (float)sc[is + 2] * (float)q2;
                        s += xb[l + 64] * (float)sc[is + 4] * (float)q3;
                        s += xb[l + 96] * (float)sc[is + 6] * (float)q4;
                    }
                    sum += d * s;
                    xb += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return sum;
        }
        default:
            return 0.f;
    }
}

}  // namespace dubhe
