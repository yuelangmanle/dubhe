// vit_gpu_stub.cpp — 非 Apple 平台上的空实现
//
// 视觉塔的 GPU matmul 只有 Metal 一档。其它平台（Android / Linux）一律返回
// false，`matmul_qt` 会照原样走 CPU 路径 —— 功能不变，只是慢。
#include "vit_gpu.h"

namespace dubhe {

void vit_bind(const void*, size_t) {}
bool vit_gemm_gpu(const void*, uint32_t, int64_t, int64_t,
                  const float*, int64_t, float*, int64_t) { return false; }
bool vit_attn_gpu(const float*, const float*, const float*, int64_t,
                  int, int, int, float*) { return false; }
void vit_gpu_release() {}
bool vit_gpu_active() { return false; }

}  // namespace dubhe
