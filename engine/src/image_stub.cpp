// image_stub.cpp — 非 Apple 平台的占位实现
//
// 保留接口一致，让引擎在 Linux / Android 上照常编译（只是不能读图）。
// Android 阶段接 stb_image 或 libjpeg-turbo 时，把这里的两个函数换成真实现即可，
// 上层（mmproj / CLI / HTTP）一行都不用改。
#include "image.h"

#include <cstring>

namespace dubhe {

bool load_image_rgb(const char* path, int* w, int* h, std::vector<uint8_t>* rgb,
                    std::string* err) {
    (void)path; (void)w; (void)h; (void)rgb;
    if (err) *err = "本平台的图像解码尚未接入（macOS 走 ImageIO，其他平台待补 stb_image）";
    return false;
}

bool load_video_frames_rgb(const char* path, double fps, int max_frames, int max_side,
                           int* w, int* h, std::vector<std::vector<uint8_t>>* frames,
                           std::string* err) {
    (void)path; (void)fps; (void)max_frames; (void)max_side; (void)w; (void)h; (void)frames;
    if (err) *err = "本平台的视频抽帧尚未接入（macOS 走 AVFoundation，其他平台待补）";
    return false;
}

void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    if (sw == dw && sh == dh) { std::memcpy(dst, src, (size_t)sw * sh * 3); return; }
    // 最近邻兜底，保证不崩
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++) {
            const int sx = x * sw / dw, sy = y * sh / dh;
            for (int c = 0; c < 3; c++)
                dst[((size_t)y * dw + x) * 3 + c] = src[((size_t)sy * sw + sx) * 3 + c];
        }
}

}  // namespace dubhe
