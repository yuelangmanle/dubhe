// vlprobe.cpp — 视觉塔探针
//
// 用途：把"预处理"和"视觉塔"两个阶段拆开验证。
//   vlprobe mmproj.gguf image.jpg --dump /tmp/vl
// 会把 packed pixels 和最终嵌入都写成裸 f32，方便和 numpy 参考实现逐层对拍。
//
// 为什么要 dump 中间结果：视觉塔有 14 层、两种算子、三处容易写反的索引
// （轴向 RoPE 的 cos/sin 拼接、merge-block 的 token 顺序、门控卷积的栅格重排）。
// 只比对最终输出的话，错了只知道"错了"，不知道错在哪一层。
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "image.h"
#include "mmproj.h"

using namespace dubhe;

static bool write_f32(const std::string& path, const float* p, size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(p, 4, n, f);
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "用法: vlprobe <mmproj.gguf> <image> [--dump 前缀] [--size 边对齐]\n");
        return 2;
    }
    const char* mm = argv[1];
    const char* img = argv[2];
    std::string dump;
    for (int i = 3; i + 1 < argc; i++)
        if (std::strcmp(argv[i], "--dump") == 0) dump = argv[i + 1];

    std::string err;
    MmProj mp;
    if (!mp.load(mm, &err)) {
        std::fprintf(stderr, "加载 mmproj 失败: %s\n", err.c_str());
        return 1;
    }
    std::string desc;
    mp.describe(&desc);
    std::fputs(desc.c_str(), stdout);

    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    if (!load_image_rgb(img, &w, &h, &rgb, &err)) {
        std::fprintf(stderr, "读图失败: %s\n", err.c_str());
        return 1;
    }
    const auto& c = mp.config();
    int oh = 0, ow = 0;
    mp.smart_resize(h, w, &oh, &ow);
    std::printf("原图 %d×%d → smart_resize %d×%d（patch 网格 %d×%d）\n", w, h, ow, oh,
                ow / c.patch, oh / c.patch);

    std::vector<uint8_t> resized((size_t)ow * oh * 3);
    resize_bicubic(rgb.data(), w, h, resized.data(), ow, oh);

    const int gh = oh / c.patch, gw = ow / c.patch;
    const int64_t np = (int64_t)gh * gw;
    std::vector<float> pixels((size_t)(np * c.patch_dim()));
    if (!mp.pack_pixels(resized.data(), ow, oh, pixels.data())) {
        std::fprintf(stderr, "pack_pixels 失败（尺寸不合法）\n");
        return 1;
    }

    const int64_t nt = np / (c.merge * c.merge);
    std::vector<float> out((size_t)(nt * c.out_dim));
    // 逐层 dump：文件编号和 HF 的 hidden_states 对齐（HF 的 [0] 是嵌入输出，
    // [i+1] 是第 i 层之后），这样 diff 脚本不用做偏移换算。
    if (!dump.empty())
        mp.on_layer = [&](int li, const float* x, int64_t n) {
            char p[512];
            std::snprintf(p, sizeof p, "%s.layer%02d.f32", dump.c_str(), li + 1);
            write_f32(p, x, (size_t)(n * c.hidden));
        };
    auto t0 = std::chrono::steady_clock::now();
    if (!mp.encode(pixels.data(), gh, gw, out.data(), &err)) {
        std::fprintf(stderr, "视觉塔前向失败: %s\n", err.c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 统计量：只看最终输出的话，视觉塔算错和图片太怪很难区分，所以把
    // 均值和方差都打出来，和参考实现一比就知道是不是同一个东西。
    double s = 0.0, ss = 0.0, mn = 1e30, mx = -1e30;
    for (float v : out) { s += v; ss += (double)v * v; mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); }
    const double n = (double)out.size();
    const double mean = s / n;
    std::printf("视觉塔: %lld patch → %lld token × %d 维，耗时 %.0f ms\n",
                (long long)np, (long long)nt, c.out_dim, ms);
    std::printf("输出统计: mean %.6f  std %.6f  min %.4f  max %.4f\n",
                mean, std::sqrt(ss / n - mean * mean), mn, mx);
    std::printf("前 8 个值:");
    for (int i = 0; i < 8 && i < (int)out.size(); i++) std::printf(" %.5f", out[(size_t)i]);
    std::printf("\n");

    if (!dump.empty()) {
        write_f32(dump + ".pixels.f32", pixels.data(), pixels.size());
        write_f32(dump + ".out.f32", out.data(), out.size());
        if (FILE* f = std::fopen((dump + ".meta.txt").c_str(), "w")) {
            std::fprintf(f, "%d %d\n", gh, gw);
            std::fclose(f);
        }
        std::printf("已 dump: %s.{pixels,out}.f32 + meta.txt\n", dump.c_str());
    }
    return 0;
}
