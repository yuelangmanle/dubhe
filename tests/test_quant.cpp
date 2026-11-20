// test_quant.cpp — 量化内核数值自检
//
// 对每个量化类型：造随机权重行 → 用 dequant_row 反量化得到"真值" →
// 与 vec_dot 的融合点积结果对比。任何偏差说明该内核的位解包有错。
#include "../engine/src/internal.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace dubhe;

namespace {

int g_fail = 0;

// 把一行的 scale 字段写成合理值，避免 inf/nan
void sane_scales(uint32_t type, uint8_t* p, int n) {
    const int bs = type_block_size(type);
    const int bb = type_block_bytes(type);
    const uint16_t one = 0x2C00;   // ~0.0625
    const uint16_t zero = 0x0000;
    for (int b = 0; b < n / bs; b++) {
        uint8_t* q = p + (size_t)b * bb;
        switch (type) {
            case T_Q4_0: case T_Q5_0: case T_Q8_0:
                std::memcpy(q, &one, 2);
                break;
            case T_Q4_1: case T_Q5_1:
                std::memcpy(q, &one, 2);
                std::memcpy(q + 2, &zero, 2);
                break;
            case T_Q4_K: case T_Q5_K:
                std::memcpy(q, &one, 2);
                std::memcpy(q + 2, &one, 2);
                // 保留随机的 6-bit scale/min，覆盖每个子块的独立系数。
                break;
            case T_Q6_K:
                std::memcpy(q + 208, &one, 2);
                for (int k = 0; k < 16; k++) q[192 + k] = 1;
                break;
            default:
                break;
        }
    }
}

void check(uint32_t type, int n, int iters) {
    std::mt19937 rng(1234 + (unsigned)type);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255);

    const size_t rb = type_row_bytes(type, n);
    std::vector<uint8_t> row(rb);
    std::vector<float> x((size_t)n), ref((size_t)n);

    double worst = 0;
    double worst_rel = 0;
    for (int it = 0; it < iters; it++) {
        for (auto& b : row) b = (uint8_t)byte(rng);
        if (type == T_F32) {
            float* f = (float*)row.data();
            for (int i = 0; i < n; i++) f[i] = u(rng);
        } else if (type == T_F16) {
            for (int i = 0; i < n; i++) {
                const uint16_t h = fp32_to_fp16(u(rng));
                std::memcpy(row.data() + i * 2, &h, 2);
            }
        } else if (type == T_BF16) {
            // 把指数位限制在 [-8, 8] 附近，避免随机位产生 inf/nan
            for (int i = 0; i < n; i++) {
                uint16_t h = (uint16_t)(row[i * 2] | (row[i * 2 + 1] << 8));
                h = (uint16_t)((h & 0x807Fu) | (((uint16_t)(127 + (h % 17) - 8)) << 7));
                std::memcpy(row.data() + i * 2, &h, 2);
            }
        }
        sane_scales(type, row.data(), n);
        for (auto& v : x) v = u(rng);

        dequant_row(type, row.data(), ref.data(), n);
        double ref_dot = 0, norm = 0;
        for (int i = 0; i < n; i++) {
            ref_dot += (double)x[i] * ref[i];
            norm += std::fabs((double)x[i] * ref[i]);
        }
        const double got = vec_dot(type, x.data(), row.data(), n);
        const double err = std::fabs(got - ref_dot);
        const double rel = norm > 0 ? err / norm : err;
        if (!std::isfinite(rel)) {
            std::printf("%-6s n=%-5d ✗ 点积非有限值（第 %d 轮）\n", type_name(type), n, it + 1);
            g_fail++;
            return;
        }
        if (err > worst) worst = err;
        if (rel > worst_rel) worst_rel = rel;
    }
    const bool ok = worst_rel < 1e-4;
    if (!ok) g_fail++;
    std::printf("%-6s n=%-5d 最大绝对误差 %.3e  最大相对误差 %.3e  %s\n", type_name(type), n,
                worst, worst_rel, ok ? "✓" : "✗ 不一致");
}

// 对比"整数点积路径"与"f32 参考路径"。
// 激活被量化成 int8，必然有量化噪声，所以容差比内核自检宽；
// 但位解包写错会产生 O(1) 的偏差，仍能被抓住。
void check_q8(uint32_t type, int n, int iters) {
    std::mt19937 rng(777 + (unsigned)type);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255);

    const size_t rb = type_row_bytes(type, n);
    std::vector<uint8_t> row(rb);
    std::vector<uint8_t> aq(act_q8_bytes(type, n));
    std::vector<float> x((size_t)n), qx((size_t)n), dq((size_t)n);

    double worst_rel = 0;
    double worst_arithmetic = 0;
    for (int it = 0; it < iters; it++) {
        for (auto& b : row) b = (uint8_t)byte(rng);
        sane_scales(type, row.data(), n);
        for (auto& v : x) v = u(rng);
        quantize_act(type, x.data(), aq.data(), n);

        const double ref = vec_dot(type, x.data(), row.data(), n);
        const double got = vec_dot_q8(type, aq.data(), row.data(), n);
        const double err = std::fabs(got - ref);
        dequant_row(T_Q8_0, aq.data(), qx.data(), n);
        const double arithmetic_err = std::fabs(got - vec_dot(type, qx.data(), row.data(), n));
        // 参考值的量级（避免 ref 接近 0 时相对误差爆炸）
        double scale = 0;
        dequant_row(type, row.data(), dq.data(), n);
        for (int i = 0; i < n; i++) scale += std::fabs((double)x[i] * dq[i]);
        const double rel = scale > 0 ? err / scale : err;
        const double arithmetic_rel = scale > 0 ? arithmetic_err / scale : arithmetic_err;
        if (!std::isfinite(rel) || !std::isfinite(arithmetic_rel)) {
            std::printf("%-6s n=%-5d ✗ 整数点积非有限值（第 %d 轮）\n", type_name(type), n, it + 1);
            g_fail++;
            return;
        }
        if (rel > worst_rel) worst_rel = rel;
        if (arithmetic_rel > worst_arithmetic) worst_arithmetic = arithmetic_rel;
    }
    const bool ok = worst_rel < 5e-3 && worst_arithmetic < 1e-5;
    if (!ok) g_fail++;
    std::printf("%-6s n=%-5d 整数路径相对误差 %.3e（同激活 %.3e） %s\n",
                type_name(type), n, worst_rel, worst_arithmetic,
                ok ? "✓" : "✗ 与 f32 路径不一致");
}

}  // namespace

int main() {
    const int sizes[] = {256, 512, 896, 1024, 4864};
    const uint32_t types[] = {T_F32, T_F16, T_BF16, T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1,
                              T_Q8_0, T_Q4_K, T_Q5_K, T_Q6_K};
    std::printf("=== 内核位解包自检（对照 dequant + f32 点积）===\n");
    for (uint32_t t : types) {
        if (!type_supported(t)) { std::printf("%-6s 未实现，跳过\n", type_name(t)); continue; }
        for (int n : sizes) {
            if (n % type_block_size(t) != 0) continue;
            check(t, n, 20);
        }
    }

    std::printf("\n=== 整数点积路径自检（对照 f32 路径）===\n");
    for (uint32_t t : types) {
        const bool supported = act_q8_supported(t);
#if defined(__ARM_NEON)
        if (t == T_Q5_K && !supported) g_fail++;
#else
        if (supported) g_fail++;
#endif
        if (!supported) continue;
        for (int n : sizes) {
            if (n % type_block_size(t) != 0 || n % 32 != 0) continue;
            check_q8(t, n, 20);
        }
    }

    std::printf("\n%s\n", g_fail == 0 ? "全部通过" : "存在失败项");
    return g_fail == 0 ? 0 : 1;
}
