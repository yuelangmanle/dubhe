#pragma once

namespace dubhe::vulkan {

// 仅可选 Android Vulkan 构建提供这些函数。没有设备时返回 nullptr。
const char* device_name();
bool supports(int rows, int columns);

// GGUF Q4_K 行连续存储，输入和输出均为 f32。失败时不改写输出。
bool matvec_q4_k(const void* weights, const float* input, float* output, int rows, int columns);

}  // namespace dubhe::vulkan
