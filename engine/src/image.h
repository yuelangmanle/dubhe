// image.h — 图像解码与缩放
//
// 为什么不自己写 PNG/JPEG 解码器：这个项目坚持"零外部依赖"，但图像格式是
// 唯一一处我认为不值得自己写的地方 —— PNG 要 inflate、JPEG 要 Huffman +
// IDCT + 色度上采样，加起来几千行，而且极易在边界情况上出错（渐进式 JPEG、
// CMYK、EXIF 旋转、16 位 PNG）。这些错误的表现是"模型看错了图"，极难定位。
//
// 所以：macOS 走系统的 ImageIO（PNG/JPEG/HEIC/WebP/TIFF 全部免费支持，
// 还顺带处理了 EXIF 方向）；其他平台走 image_stub.cpp 里报错，等 Android
// 阶段再接 stb_image 或 libjpeg-turbo。接口在两边完全一致。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dubhe {

// 解码到 RGB8（HWC，3 字节/像素，已按 EXIF 方向摆正）。
bool load_image_rgb(const char* path, int* w, int* h, std::vector<uint8_t>* rgb,
                    std::string* err);

// 双三次缩放。用的是 Catmull-Rom（a = -0.5），与 PIL 的 BICUBIC 一致 ——
// 预处理必须和参考实现对齐，否则同一张图在两边会切出不同的像素，
// 视觉塔的输出差异会被误判成"引擎算错了"。
void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

}  // namespace dubhe
