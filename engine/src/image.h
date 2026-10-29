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

// 视频抽帧。按 fps 均匀取帧，最多 max_frames 帧；所有帧统一缩放到**第一帧**的
// 尺寸（视觉塔要求同一批 patch 形状一致，而手机视频的尺寸偶尔会中途变化）。
// frames 里是连续排列的 RGB8，每帧 w*h*3 字节。
// max_side 限制长边像素（视频原始 1080p 一帧就是 8160 个 patch，必须先缩）。
//
// 抽帧而不是"每帧都算"：LFM3-VL 的 temporal_patch_size = 2，两帧合成一个
// patch，而 24/30 fps 的相邻帧差异极小 —— 按 fps 抽稀到 1–2 fps 通常不损失
// 语义，却把 token 数直接砍掉一个数量级。
bool load_video_frames_rgb(const char* path, double fps, int max_frames, int max_side,
                           int* w, int* h, std::vector<std::vector<uint8_t>>* frames,
                           std::string* err);

// 双三次缩放。用的是 Catmull-Rom（a = -0.5），与 PIL 的 BICUBIC 一致 ——
// 预处理必须和参考实现对齐，否则同一张图在两边会切出不同的像素，
// 视觉塔的输出差异会被误判成"引擎算错了"。
void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

}  // namespace dubhe
