// image.mm — macOS 上用 ImageIO 解码，缩放自己写；视频走 AVFoundation 抽帧
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>

#include <cmath>
#include <cstring>

#include "image.h"

namespace dubhe {
namespace {

// Catmull-Rom 权重（a = -0.5），与 PIL / torchvision 的 BICUBIC 同一族。
// 注意 PIL 的 BICUBIC 实际用的是 a = -0.5 的卷积核，采样点按
// (dst + 0.5) * scale - 0.5 对齐 —— 这两点都要一致，否则缩放出图会偏半像素。
inline double cubic_w(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x <= 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

}  // namespace

bool load_image_rgb(const char* path, int* w, int* h, std::vector<uint8_t>* rgb,
                    std::string* err) {
    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:path];
        NSURL* url = [NSURL fileURLWithPath:p];
        CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
        if (!src) {
            if (err) *err = std::string("打不开图像文件: ") + path;
            return false;
        }
        // CreateThumbnailAtIndex 会应用 EXIF 方向；kCGImageSourceCreateThumbnailWithTransform
        CGImageRef img = CGImageSourceCreateThumbnailAtIndex(
            src, 0,
            (__bridge CFDictionaryRef) @{
                (id)kCGImageSourceCreateThumbnailFromImageAlways : @YES,
                (id)kCGImageSourceCreateThumbnailWithTransform : @YES,
                (id)kCGImageSourceThumbnailMaxPixelSize : @(8192),
            });
        if (!img) {
            CFRelease(src);
            if (err) *err = "无法解码该图像格式";
            return false;
        }
        const size_t iw = CGImageGetWidth(img), ih = CGImageGetHeight(img);
        if (iw == 0 || ih == 0 || iw > 16384 || ih > 16384) {
            CGImageRelease(img);
            CFRelease(src);
            if (err) *err = "图像尺寸非法";
            return false;
        }
        std::vector<uint8_t> buf((size_t)iw * ih * 4);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(buf.data(), iw, ih, 8, iw * 4, cs,
                                                 (CGBitmapInfo)kCGImageAlphaNoneSkipLast |
                                                 (CGBitmapInfo)kCGBitmapByteOrderDefault);
        CGColorSpaceRelease(cs);
        if (!ctx) {
            CGImageRelease(img);
            CFRelease(src);
            if (err) *err = "创建位图上下文失败";
            return false;
        }
        CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)iw, (CGFloat)ih), img);
        CGContextRelease(ctx);
        CGImageRelease(img);
        CFRelease(src);

        rgb->resize((size_t)iw * ih * 3);
        for (size_t i = 0; i < (size_t)iw * ih; i++) {
            (*rgb)[i * 3 + 0] = buf[i * 4 + 0];
            (*rgb)[i * 3 + 1] = buf[i * 4 + 1];
            (*rgb)[i * 3 + 2] = buf[i * 4 + 2];
        }
        *w = (int)iw;
        *h = (int)ih;
        return true;
    }
}

// 把一张 CGImage 转成 RGB8（HWC）。sw/sh 是目标尺寸，0 表示原尺寸。
static bool cgimage_to_rgb(CGImageRef img, int* w, int* h, std::vector<uint8_t>* rgb) {
    const size_t iw = CGImageGetWidth(img), ih = CGImageGetHeight(img);
    if (iw == 0 || ih == 0 || iw > 16384 || ih > 16384) return false;
    std::vector<uint8_t> buf((size_t)iw * ih * 4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(buf.data(), iw, ih, 8, iw * 4, cs,
                                             (CGBitmapInfo)kCGImageAlphaNoneSkipLast |
                                             (CGBitmapInfo)kCGBitmapByteOrderDefault);
    CGColorSpaceRelease(cs);
    if (!ctx) return false;
    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)iw, (CGFloat)ih), img);
    CGContextRelease(ctx);
    rgb->resize((size_t)iw * ih * 3);
    for (size_t i = 0; i < (size_t)iw * ih; i++) {
        (*rgb)[i * 3 + 0] = buf[i * 4 + 0];
        (*rgb)[i * 3 + 1] = buf[i * 4 + 1];
        (*rgb)[i * 3 + 2] = buf[i * 4 + 2];
    }
    *w = (int)iw;
    *h = (int)ih;
    return true;
}

bool load_video_frames_rgb(const char* path, double fps, int max_frames, int max_side,
                           int* w, int* h, std::vector<std::vector<uint8_t>>* frames,
                           std::string* err) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset* asset = [AVURLAsset URLAssetWithURL:url options:nil];
        if (!asset) { if (err) *err = std::string("打不开视频: ") + path; return false; }

        const double secs = CMTimeGetSeconds(asset.duration);
        if (!(secs > 0.0)) {
            if (err) *err = "读不出视频时长（文件损坏或不是视频）";
            return false;
        }
        if (fps <= 0.0) fps = 1.0;
        int count = (int)std::ceil(secs * fps);
        if (count < 2) count = 2;          // temporal_patch_size = 2，至少两帧
        if (count > max_frames) count = max_frames;
        if (max_frames < 2) count = 2;

        AVAssetImageGenerator* gen = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
        // 手机拍的视频带旋转元数据，不 applied 的话整段会躺倒
        gen.appliesPreferredTrackTransform = YES;
        // 只要求"接近这个时间点"，不强求精确 —— 精确解码要逐帧 seek，慢一个数量级
        gen.requestedTimeToleranceBefore = CMTimeMake(1, 30);
        gen.requestedTimeToleranceAfter  = CMTimeMake(1, 30);
        if (max_side > 0)
            gen.maximumSize = CGSizeMake((CGFloat)max_side, (CGFloat)max_side);

        frames->clear();
        frames->reserve((size_t)count);
        for (int i = 0; i < count; i++) {
            double t = (count <= 1) ? 0.0 : (double)i / fps;
            if (t > secs - 0.02) t = secs - 0.02;
            if (t < 0.0) t = 0.0;
            CGImageRef img = [gen copyCGImageAtTime:CMTimeMakeWithSeconds(t, 600)
                                         actualTime:nullptr
                                              error:nil];
            if (!img) continue;
            std::vector<uint8_t> rgb;
            int fw = 0, fh = 0;
            const bool ok = cgimage_to_rgb(img, &fw, &fh, &rgb);
            CGImageRelease(img);
            if (!ok) continue;
            if (frames->empty()) {
                *w = fw;
                *h = fh;
                frames->push_back(std::move(rgb));
            } else {
                // 尺寸不一致就缩到第一帧的尺寸 —— 视觉塔要求所有 patch 同形
                if (fw != *w || fh != *h) {
                    std::vector<uint8_t> fixed((size_t)(*w) * (*h) * 3);
                    resize_bicubic(rgb.data(), fw, fh, fixed.data(), *w, *h);
                    frames->push_back(std::move(fixed));
                } else {
                    frames->push_back(std::move(rgb));
                }
            }
        }
        if (frames->size() < 2) {
            if (err) *err = "抽帧失败（解出的帧不足 2 张）";
            return false;
        }
        // 奇数帧补一帧（复制最后一帧），凑成整数个 temporal patch
        if (frames->size() % 2) frames->push_back(frames->back());
        return true;
    }
}

void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {    if (sw == dw && sh == dh) {
        std::memcpy(dst, src, (size_t)sw * sh * 3);
        return;
    }
    const double sx = (double)sw / dw, sy = (double)sh / dh;

    // 先横向再纵向，中间缓冲用 float，避免两次 8 位量化把误差放大。
    std::vector<float> mid((size_t)dw * sh * 3);
    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < dw; x++) {
            const double cx = (x + 0.5) * sx - 0.5;
            const int x0 = (int)std::floor(cx);
            double ws[4];
            double sum = 0.0;
            for (int t = 0; t < 4; t++) { ws[t] = cubic_w(cx - (x0 - 1 + t)); sum += ws[t]; }
            if (sum == 0.0) { ws[1] = 1.0; sum = 1.0; }
            for (int ch = 0; ch < 3; ch++) {
                double acc = 0.0;
                for (int t = 0; t < 4; t++) {
                    int xi = x0 - 1 + t;
                    if (xi < 0) xi = 0;
                    if (xi >= sw) xi = sw - 1;
                    acc += ws[t] * (double)src[((size_t)y * sw + xi) * 3 + ch];
                }
                mid[((size_t)y * dw + x) * 3 + ch] = (float)(acc / sum);
            }
        }
    }
    for (int y = 0; y < dh; y++) {
        const double cy = (y + 0.5) * sy - 0.5;
        const int y0 = (int)std::floor(cy);
        double ws[4];
        double sum = 0.0;
        for (int t = 0; t < 4; t++) { ws[t] = cubic_w(cy - (y0 - 1 + t)); sum += ws[t]; }
        if (sum == 0.0) { ws[1] = 1.0; sum = 1.0; }
        for (int x = 0; x < dw; x++) {
            for (int ch = 0; ch < 3; ch++) {
                double acc = 0.0;
                for (int t = 0; t < 4; t++) {
                    int yi = y0 - 1 + t;
                    if (yi < 0) yi = 0;
                    if (yi >= sh) yi = sh - 1;
                    acc += ws[t] * (double)mid[((size_t)yi * dw + x) * 3 + ch];
                }
                double v = acc / sum;
                v = v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);
                dst[((size_t)y * dw + x) * 3 + ch] = (uint8_t)std::lround(v);
            }
        }
    }
}

}  // namespace dubhe
