// mvbench.mm — 单独量 mv_* 内核（解码 matvec）的吞吐与正确性
//
// 为什么单独做：改一次内核就要重新加载 1.1 GB 模型才能看到数字，
// 一次迭代 40 秒，而且模型加载本身的噪声盖住了 5% 级的差别。
// 这个工具只跑一个内核，秒级出数，还能顺便和 CPU 参考实现对拍。
//
// 用法：
//   make mvbench && ./build/mvbench                    # 默认 Q4_K 896x4864
//   ./build/mvbench q4_k 4864 896 20                   # 类型 n_out n_in 轮数
//   DUBHE_MV_R=4 ./build/mvbench q4_k 4864 896 20      # 换 MV_R
#include "metal_kernels.h"

#include "../engine/src/internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace dubhe;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char* tname = (argc > 1) ? argv[1] : "q4_k";
    const int n_out = (argc > 2) ? atoi(argv[2]) : 4864;
    const int n_in  = (argc > 3) ? atoi(argv[3]) : 896;
    const int reps  = (argc > 4) ? atoi(argv[4]) : 20;

    uint32_t type = T_Q4_K;
    const char* kname = "mv_q4_k";
    if (!strcmp(tname, "q6_k")) { type = T_Q6_K; kname = "mv_q6_k"; }
    else if (!strcmp(tname, "f16")) { type = T_F16; kname = "mv_f16"; }
    else if (!strcmp(tname, "q8_0")) { type = T_Q8_0; kname = "mv_q8_0"; }

    const size_t rbytes = type_row_bytes(type, n_in);
    const size_t wbytes = rbytes * (size_t)n_out;
    std::printf("类型 %s  形状 %d×%d  权重 %.1f MiB\n",
                tname, n_out, n_in, (double)wbytes / 1048576.0);

    int mv_rows = 2;
    if (const char* s = getenv("DUBHE_MV_R")) {
        const int v = atoi(s);
        if (v >= 1 && v <= 16) mv_rows = v;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("没有 Metal 设备\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    // 和 metal.mm 一样：把 MV_R 注入到 MSL 源码最前面
    std::string src = "#define MV_R " + std::to_string(mv_rows) + "\n" + kDubheMSL;
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                           options:[MTLCompileOptions new]
                                             error:&e];
    if (!lib) {
        std::printf("编译失败: %s\n", e ? [[e localizedDescription] UTF8String] : "?");
        return 1;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kname]];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) { std::printf("pipeline 失败\n"); return 1; }

    // 随机权重（Q4_K 的块头 d/dmin 用小的有限值，避免 f16 溢出成 inf）
    std::mt19937 rng(7);
    std::vector<uint8_t> host(wbytes);
    for (size_t i = 0; i < wbytes; i++) host[i] = (uint8_t)(rng() & 0xFF);
    if (type == T_Q4_K) {
        // 块头 2 字节 d + 2 字节 dmin 必须是合法 f16：塞 0x2800 ≈ 0.0625
        for (size_t b = 0; b + 144 <= wbytes; b += 144) {
            host[b + 0] = 0x00; host[b + 1] = 0x28;
            host[b + 2] = 0x00; host[b + 3] = 0x2C;
        }
    }

    std::vector<float> xv((size_t)n_in);
    for (auto& v : xv) v = (float)((int)(rng() % 2001) - 1000) / 1000.0f;

    id<MTLBuffer> xbuf = [dev newBufferWithBytes:xv.data() length:sizeof(float) * n_in
                                         options:MTLResourceStorageModeShared];
    // REP 份**不同**的权重缓冲：如果只用一份重复派发，后面几次会命中 L2，
    // 量出来的是缓存带宽而不是显存带宽。真实前向里每层权重都不一样。
    const int REP = 8;
    id<MTLBuffer> wbufs[8];
    id<MTLBuffer> ybufs[8];
    for (int i = 0; i < REP; i++) {
        wbufs[i] = [dev newBufferWithBytes:host.data() length:wbytes
                                   options:MTLResourceStorageModeShared];
        ybufs[i] = [dev newBufferWithLength:sizeof(float) * n_out
                                    options:MTLResourceStorageModeShared];
    }

    uint32_t nin = (uint32_t)n_in, nout = (uint32_t)n_out;
    uint64_t woff = 0;
    const int per_tg = 4 * mv_rows;
    const NSUInteger tgc = (NSUInteger)((n_out + per_tg - 1) / per_tg);

    // 一次 commit 里连续派发 REP 次：单次派发的固定开销（约 100 µs 量级）
    // 会把 9 MiB 这种小张量的带宽压得面目全非，必须摊掉。
    double best = 1e30;
    for (int r = 0; r < reps + 3; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:ps];
        [en setBuffer:xbuf offset:0 atIndex:1];
        [en setBytes:&nin length:4 atIndex:3];
        [en setBytes:&woff length:8 atIndex:4];
        [en setBytes:&nout length:4 atIndex:5];
        for (int i = 0; i < REP; i++) {
            [en setBuffer:wbufs[i] offset:0 atIndex:0];
            [en setBuffer:ybufs[i] offset:0 atIndex:2];
            [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [en endEncoding];
        const double t0 = now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        const double ms = (now_ms() - t0) / (double)REP;
        if (r >= 3 && ms < best) best = ms;
    }

    // CPU 参考：用引擎自己的反量化路径算一遍（只算前 8 行，够判断对错）
    const int chk = std::min(n_out, 8);
    std::vector<float> ref((size_t)chk, 0.f);
    for (int r = 0; r < chk; r++) {
        const uint8_t* row = host.data() + (size_t)r * rbytes;
        double acc = 0.0;
        if (type == T_Q4_K) {
            const int nb = n_in / 256;
            for (int b = 0; b < nb; b++) {
                const uint8_t* blk = row + (size_t)b * 144;
                const float d = fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
                const float dm = fp16_to_fp32((uint16_t)(blk[2] | (blk[3] << 8)));
                // 按内核的单元编号走：o=1..8 是数据单元，每个覆盖 32 个元素
                // （低半字节 → sub-block 2k，高半字节 → sub-block 2k+1）
                for (int o = 1; o <= 8; o++) {
                    const int k = (o - 1) / 2, g = (o - 1) % 2;
                    int sc0, mn0, sc1, mn1;
                    for (int j = 2 * k; j <= 2 * k + 1; j++) {
                        int sc, mn;
                        if (j < 4) { sc = blk[4 + j] & 63; mn = blk[8 + j] & 63; }
                        else {
                            sc = (blk[8 + j] & 0x0F) | ((blk[j] >> 6) << 4);
                            mn = (blk[8 + j] >> 4) | ((blk[4 + j] >> 6) << 4);
                        }
                        if (j == 2 * k) { sc0 = sc; mn0 = mn; }
                        else { sc1 = sc; mn1 = mn; }
                    }
                    const int e_lo = b * 256 + k * 64 + g * 16;
                    const int e_hi = e_lo + 32;
                    for (int i = 0; i < 16; i++) {
                        const int byte = blk[o * 16 + i];
                        acc += (double)(d * (float)sc0 * (float)(byte & 0x0F) - dm * (float)mn0)
                             * (double)xv[(size_t)(e_lo + i)];
                        acc += (double)(d * (float)sc1 * (float)(byte >> 4) - dm * (float)mn1)
                             * (double)xv[(size_t)(e_hi + i)];
                    }
                }
            }
        } else {
            acc = 0.0;  // 其它类型不做参考校验
        }
        ref[(size_t)r] = (float)acc;
    }

    // 诊断：把第 0 行按"单元"拆开的偏和打出来，方便看内核漏了哪几段
    if (getenv("MVBENCH_DUMP") && type == T_Q4_K) {
        const int nb0 = n_in / 256;
        std::printf("row0 单元偏和（o=1..8，块 0..%d）:\n", nb0 - 1);
        for (int b = 0; b < std::min(nb0, 2); b++) {
            const uint8_t* blk = host.data() + (size_t)b * 144;
            const float d = fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            const float dm = fp16_to_fp32((uint16_t)(blk[2] | (blk[3] << 8)));
            for (int o = 1; o <= 8; o++) {
                const int k = (o - 1) / 2, g = (o - 1) % 2;
                int sc0 = 0, mn0 = 0, sc1 = 0, mn1 = 0;
                for (int j = 2 * k; j <= 2 * k + 1; j++) {
                    int sc, mn;
                    if (j < 4) { sc = blk[4 + j] & 63; mn = blk[8 + j] & 63; }
                    else {
                        sc = (blk[8 + j] & 0x0F) | ((blk[j] >> 6) << 4);
                        mn = (blk[8 + j] >> 4) | ((blk[4 + j] >> 6) << 4);
                    }
                    if (j == 2 * k) { sc0 = sc; mn0 = mn; } else { sc1 = sc; mn1 = mn; }
                }
                const int e_lo = b * 256 + k * 64 + g * 16;
                double acc2 = 0.0;
                for (int i = 0; i < 16; i++) {
                    const int byte = blk[o * 16 + i];
                    acc2 += (double)(d * (float)sc0 * (float)(byte & 0x0F) - dm * (float)mn0)
                          * (double)xv[(size_t)(e_lo + i)];
                    acc2 += (double)(d * (float)sc1 * (float)(byte >> 4) - dm * (float)mn1)
                          * (double)xv[(size_t)(e_lo + 32 + i)];
                }
                std::printf("  b=%d o=%d boff=%3d xlo=%3d xhi=%3d 偏和=%10.4f\n",
                            b, o, o * 16, k * 64 + g * 16, k * 64 + g * 16 + 32, acc2);
            }
        }
    }

    const float* y = (const float*)[ybufs[0] contents];
    double maxd = 0.0, maxr = 0.0;
    if (type == T_Q4_K) {
        for (int r = 0; r < chk; r++) {
            maxd = std::max(maxd, (double)std::fabs((double)y[r] - (double)ref[(size_t)r]));
            maxr = std::max(maxr, (double)std::fabs((double)ref[(size_t)r]));
        }
    }

    const double gbs = (double)wbytes / (best * 1e6);
    std::printf("MV_R=%d  threadgroup=%d 组 × 128 线程\n", mv_rows, (int)tgc);
    std::printf("耗时 %.3f ms  →  **%.1f GB/s**\n", best, gbs);
    if (type == T_Q4_K) {
        for (int r = 0; r < std::min(chk, 4); r++)
            std::printf("  row %d: gpu=%.6f  ref=%.6f  Δ=%.3e\n",
                        r, y[r], ref[(size_t)r], (double)y[r] - (double)ref[(size_t)r]);
        std::printf("对拍 rel=%.2e  max|Δ|=%.3e  (max|ref|=%.3e)  %s\n",
                    maxd / (maxr + 1e-6), maxd, maxr,
                    (maxd / (maxr + 1e-6) < 1e-4) ? "OK" : "**不一致**");
    }
    return 0;
}
