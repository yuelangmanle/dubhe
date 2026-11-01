// vit_gpu.h — 视觉塔的 GPU 分块 matmul
//
// 为什么单独一个模块：视觉塔的权重是**独立的 mmproj GGUF**，和语言模型的
// 权重不在同一个 mmap 里；而 metal.mm 的权重缓冲（g_w）是单例，绑了语言
// 模型就不能再绑视觉塔。与其把 metal.mm 改成多实例（那是并发批处理那一轮
// 的事），不如让视觉塔自己持有一份 Metal 状态 —— 它只需要"一个权重缓冲 +
// 一组 mm3_* 流水线"，着色器源码还是同一份（metal_kernels.h）。
//
// 为什么只搬 matmul：视觉塔 5.06 s 里 95% 是 `matmul_qt`，而 CPU 上它已经
// 跑到 62 GFLOP/s（6 核 NEON，接近 CPU 上限），GPU 上是 2 TFLOP/s 量级。
// 剩下的 rmsnorm / RoPE / 双向注意力 / 深度卷积加起来不到 5% 的浮点量，
// 先留在 CPU，省得把整条链路重写一遍。
#pragma once

#include <cstddef>
#include <cstdint>

namespace dubhe {

// 把 mmproj 的 mmap 登记为视觉塔的权重缓冲。很轻 —— 不建 Metal 设备、
// 不编译着色器，真正的初始化推迟到第一次 vit_gemm_gpu 调用，
// 这样纯文本场景不会多付 ~200 ms 的着色器编译时间。
void vit_bind(const void* base, size_t size);

// 分块 matmul：Y[t][o] = Σ_i X[t][i] · W[o][i]
//   wdata    : 权重张量首地址（必须在 vit_bind 登记的范围内）
//   type     : GGUF 张量类型（只支持 F16 / Q8_0 / Q4_K / Q6_K）
//   n_in     : 行宽，必须是 256 的倍数（mm3_* 的 K 分块要求）
//   x_stride : X 的相邻 token 间隔；只有 patchify 用得到，那种情况回退 CPU
// 返回 true 表示 Y 已写好；false 表示这一档不可用，调用方走 CPU。
bool vit_gemm_gpu(const void* wdata, uint32_t type, int64_t n_in, int64_t n_out,
                  const float* X, int64_t n_tok, float* Y, int64_t x_stride);

// 释放 Metal 资源（模型卸载时调）
void vit_gpu_release();

// 诊断：是否已经在 GPU 上跑过
bool vit_gpu_active();

}  // namespace dubhe
