// image.mm — macOS 上用 ImageIO 解码，缩放自己写；视频走 AVFoundation 抽帧
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>

#include <cmath>
#include <cstring>

#include "image.h"
#include "image_resize.h"

namespace dubhe {

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

void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    resize_bicubic_rgb(src, sw, sh, dst, dw, dh);
}

}  // namespace dubhe
