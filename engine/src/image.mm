// image.mm — macOS 上用 ImageIO 解码，缩放自己写
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>

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

void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    if (sw == dw && sh == dh) {
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
