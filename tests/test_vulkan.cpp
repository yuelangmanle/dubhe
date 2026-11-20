#include "internal.h"
#include "vulkan_matvec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace dubhe;

static bool check(int rows, int cols) {
    const size_t stride = type_row_bytes(T_Q4_K, cols);
    std::vector<uint8_t> weights(stride * rows);
    std::vector<float> x(cols), got(rows);
    std::mt19937 rng(42 + cols + rows);
    for (auto& byte : weights) byte = static_cast<uint8_t>(rng());
    for (size_t offset = 0; offset < weights.size(); offset += 144) {
        const uint16_t scale = fp32_to_fp16(0.0625f + (rng() % 32) / 1024.f);
        const uint16_t minimum = fp32_to_fp16(0.125f + (rng() % 32) / 1024.f);
        std::memcpy(weights.data() + offset, &scale, sizeof(scale));
        std::memcpy(weights.data() + offset + 2, &minimum, sizeof(minimum));
    }
    for (float& value : x) value = (static_cast<int>(rng() % 4097) - 2048) / 2048.f;

    if (!vulkan::matvec_q4_k(weights.data(), x.data(), got.data(), rows, cols)) {
        std::fprintf(stderr, "Vulkan 矩阵运算失败（%d × %d）\n", rows, cols);
        return false;
    }
    double worst = 0;
    for (int row = 0; row < rows; ++row) {
        const float ref = vec_dot(T_Q4_K, x.data(), weights.data() + row * stride, cols);
        const double diff = std::fabs(static_cast<double>(got[row]) - ref);
        const double scaled_diff = diff / std::max(1.0, std::fabs(static_cast<double>(ref)));
        if (!std::isfinite(scaled_diff) || scaled_diff > 5e-4) {
            std::fprintf(stderr, "%d × %d 第 %d 行：CPU %.8g，Vulkan %.8g\n", rows, cols, row, ref, got[row]);
            return false;
        }
        worst = std::max(worst, scaled_diff);
    }
    std::printf("Vulkan Q4_K %d × %d 通过，最大相对误差 %.2e\n", rows, cols, worst);
    return true;
}

int main() {
    const char* device = vulkan::device_name();
    if (!device) {
        std::fprintf(stderr, "没有可用的 Vulkan 计算设备\n");
        return 1;
    }
    std::printf("Vulkan 设备 %s\n", device);
    float output = 42.f;
    if (vulkan::supports(1, 255) || vulkan::matvec_q4_k(&output, &output, &output, 1, 255) || output != 42.f) {
        std::fprintf(stderr, "Vulkan 应跳过非整块 Q4_K 输入\n");
        return 1;
    }
    return check(7, 256) && check(513, 1536) && check(65537, 256) && check(23, 512) ? 0 : 1;
}
