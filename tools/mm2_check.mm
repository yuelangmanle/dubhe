// mm2_check.mm — 二维分块 matmul（mm2_*）与旧 mm_* 内核逐元素对拍
//
// 两个内核吃同一份权重字节和同一份激活，只有累加顺序不同（f32 下差异
// 应在 1e-5 量级）。任何超过 1e-3 的偏差都说明分块版的内核写错了。
//
// 用法：./build/mm2_check
#include "metal_kernels.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using dubhe_msl::kBmax;
using dubhe_msl::kMmRows;
using dubhe_msl::kMmSgPerTg;

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLLibrary> g_lib;
static int g_fail = 0;

static id<MTLComputePipelineState> pipe(const char* n) {
    id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:n]];
    if (!f) { std::printf("  缺少函数 %s\n", n); g_fail++; return nil; }
    NSError* e = nil;
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) {
        std::printf("  pipeline %s 失败: %s\n", n, e ? [[e localizedDescription] UTF8String] : "?");
        g_fail++;
    }
    return p;
}

static id<MTLBuffer> buf(size_t bytes) {
    return [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// 造一份"合法但随机"的权重字节。块头里的 d/dm 写成固定的 f16 值，
// 避免随机字节解出 inf/nan 把误差判断搅乱。
static void fill_w(std::vector<uint8_t>& w, size_t block_bytes, size_t nblocks, int header) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& x : w) x = (uint8_t)d(rng);
    if (header >= 0) {
        // header = d/dm 的字节偏移；写入 f16 的 0.0625 (0x2C00) 与 0.03125 (0x2800)
        for (size_t b = 0; b < nblocks; b++) {
            uint8_t* p = w.data() + b * block_bytes + header;
            p[0] = 0x00; p[1] = 0x2C;   // d  = 0.0625
            p[2] = 0x00; p[3] = 0x28;   // dm = 0.03125
        }
    }
}

static float f16(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

// Q4_K 的 CPU 参考解量化（布局：d(2) dm(2) scales(12) qs(128)）
static void deq_q4k_row(const uint8_t* q, int nb, float* out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = q + (size_t)b * 144;
        const float d  = f16((uint16_t)(blk[0] | (blk[1] << 8)));
        const float dm = f16((uint16_t)(blk[2] | (blk[3] << 8)));
        const uint8_t* sc = blk + 4;
        const uint8_t* qs = blk + 16;
        for (int s = 0; s < 8; s++) {
            int s1, m1;
            if (s < 4) { s1 = sc[s] & 63; m1 = sc[s + 4] & 63; }
            else { s1 = (sc[s + 4] & 0x0F) | ((sc[s - 4] >> 6) << 4);
                   m1 = (sc[s + 4] >> 4)    | ((sc[s] >> 6) << 4); }
            for (int i = 0; i < 32; i++) {
                const uint8_t byte = qs[(size_t)(s >> 1) * 32 + i];
                const int v = (s & 1) ? (byte >> 4) : (byte & 0x0F);
                out[(size_t)b * 256 + s * 32 + i] = d * s1 * v - dm * m1;
            }
        }
    }
}

static void cmp(const char* name, const std::vector<float>& ref, const std::vector<float>& got) {
    double mx = 0, scale = 0;
    int at = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        scale = std::max(scale, std::fabs((double)ref[i]));
        if (d > mx) { mx = d; at = (int)i; }
    }
    const double rel = scale > 0 ? mx / scale : mx;
    const bool ok = rel < 1e-3;
    if (!ok) g_fail++;
    std::printf("  %-10s %s  rel=%.3e  max=%.3e @%d  ref=%.6f got=%.6f\n",
                name, ok ? "✓" : "✗ 不一致", rel, mx, at, ref[at], got[at]);
}

struct Case {
    const char* oldk;
    const char* newk;
    int block_bytes;   // 每 block 字节
    int block_elems;   // 每 block 元素
    int header;        // d/dm 所在偏移，-1 表示没有
};

int main(int argc, char** argv) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { std::printf("没有 Metal 设备\n"); return 1; }
        g_q = [g_dev newCommandQueue];
        NSError* e = nil;
        g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kDubheMSL]
                                    options:[MTLCompileOptions new]
                                      error:&e];
        if (!g_lib) {
            std::printf("MSL 编译失败: %s\n", e ? [[e localizedDescription] UTF8String] : "?");
            return 1;
        }

        const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
        const char* g_kernel = (argc > 2) ? argv[2] : "mm2_q4_k";
        if (bench) {
            // 裸内核吞吐。LFM3-1.7B 的真实形状，nrow = BMAX。
            // 用 GFLOP/s 判断离 ALU 上限（实测 fp32 峰值 2.6 TFLOP/s）还有多远。
            struct Shape { int n_in, n_out; const char* tag; };
            const Shape shapes[] = {
                {2048, 2048, "q_proj / o_proj / attn_gate"},
                {2048,  512, "k_proj / v_proj"},
                {2048, 6144, "conv in_proj"},
                {6144, 2048, "conv out_proj"},
                {2048, 7680, "ffn gate / up"},
                {7680, 2048, "ffn down"},
            };
            const int nrow = kBmax;
            double total_macs = 0, total_ms = 0;
            for (const auto& s : shapes) {
                const size_t nblocks_row = (size_t)(s.n_in / 256);
                const size_t wbytes = nblocks_row * (size_t)s.n_out * 144;
                std::vector<uint8_t> W(wbytes);
                fill_w(W, 144, nblocks_row * (size_t)s.n_out, 0);
                std::vector<float> XT((size_t)s.n_in * kBmax, 0.5f);
                id<MTLBuffer> bw = buf(wbytes);
                std::memcpy([bw contents], W.data(), wbytes);
                id<MTLBuffer> bxt = buf(XT.size() * 4);
                std::memcpy([bxt contents], XT.data(), XT.size() * 4);
                id<MTLBuffer> by = buf((size_t)s.n_out * kBmax * 4);

                id<MTLComputePipelineState> p = pipe(g_kernel);
                if (false) {
                    // maxTotalThreadsPerThreadgroup 会被寄存器占用率压下来：
                    // 它等于 min(1024, 32768/寄存器数)。看到 256 就说明
                    // 每线程约 128 个寄存器 → 每个 core 只能驻留一个 threadgroup。
                    std::printf("  [mm2_q4_k] maxTotalThreadsPerThreadgroup = %lu, "
                                "静态 threadgroup 内存 = %lu 字节\n",
                                (unsigned long)p.maxTotalThreadsPerThreadgroup,
                                (unsigned long)p.staticThreadgroupMemoryLength);
                }
                uint32_t nin = (uint32_t)s.n_in, nout = (uint32_t)s.n_out, nr = (uint32_t)nrow;
                uint64_t woff = 0;
                const bool is_old = std::strncmp(g_kernel, "mm_", 3) == 0;
                const int rows_per_tg = is_old ? 4 : kMmRows;
                const int nthreads = is_old ? 128 : kMmSgPerTg * 32;
                const NSUInteger tgc = (NSUInteger)((s.n_out + rows_per_tg - 1) / rows_per_tg);

                // 一次 commit 里编 INNER 次派发再除以 INNER。单次 commit+wait
                // 在本机约 200 µs，比内核本身还长，不摊掉就全是噪声。
                const int INNER = 20;
                auto run = [&](int times) {
                    id<MTLCommandBuffer> cb = [g_q commandBuffer];
                    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                    [en setComputePipelineState:p];
                    [en setBuffer:bw offset:0 atIndex:0];
                    [en setBuffer:bxt offset:0 atIndex:1];
                    [en setBuffer:by offset:0 atIndex:2];
                    [en setBytes:&nin length:4 atIndex:3];
                    [en setBytes:&woff length:8 atIndex:4];
                    [en setBytes:&nout length:4 atIndex:5];
                    [en setBytes:&nr length:4 atIndex:6];
                    for (int t = 0; t < times; t++) {
                        [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(nthreads, 1, 1)];
                    }
                    [en endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                };
                run(INNER);
                const int REP = 5;
                double best = 1e30;
                for (int i = 0; i < REP; i++) {
                    auto t0 = std::chrono::steady_clock::now();
                    run(INNER);
                    auto t1 = std::chrono::steady_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / INNER;
                    if (ms < best) best = ms;
                }
                const double macs = (double)s.n_in * s.n_out * nrow;
                std::printf("  %-28s in=%5d out=%5d  %7.2f ms  %6.1f GFLOP/s\n",
                            s.tag, s.n_in, s.n_out, best, macs * 2 / best / 1e6);
                total_macs += macs;
                total_ms += best;
            }
            std::printf("  %-28s %26.2f ms  %6.1f GFLOP/s (单层合计)\n", "一层",
                        total_ms, total_macs * 2 / total_ms / 1e6);
            return 0;
        }

        const int n_in = 2048, n_out = 64, nrow = 8;
        std::printf("n_in=%d n_out=%d nrow=%d kBmax=%d kMmRows=%d\n", n_in, n_out, nrow, kBmax, kMmRows);

        // 激活按 [n_in][kBmax]，与 k_trans 的输出布局一致
        std::vector<float> XT((size_t)n_in * kBmax);
        {
            std::mt19937 rng(99);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (size_t e = 0; e < (size_t)n_in; e++)
                for (int j = 0; j < kBmax; j++)
                    XT[e * kBmax + j] = (j < nrow) ? u(rng) : 0.f;
        }
        id<MTLBuffer> bxt = buf(XT.size() * 4);
        std::memcpy([bxt contents], XT.data(), XT.size() * 4);

        const Case cases[] = {
            {"mm_f32",  "mm2_f32",  4,   1,   -1},
            {"mm_f16",  "mm2_f16",  2,   1,   -1},
            {"mm_q8_0", "mm2_q8_0", 34,  32,  0},
            {"mm_q4_k", "mm2_q4_k", 144, 256, 0},
            {"mm_q6_k", "mm2_q6_k", 210, 256, 208},
        };

        for (const auto& c : cases) {
            const size_t nblocks_row = (size_t)(n_in / c.block_elems);
            const size_t wbytes = nblocks_row * (size_t)n_out * (size_t)c.block_bytes;
            std::vector<uint8_t> W(wbytes);
            fill_w(W, (size_t)c.block_bytes, nblocks_row * (size_t)n_out, c.header);
            // f32/f16 没有块头，随机字节会解出 nan，必须填真实数值
            if (c.block_bytes == 4) {
                float* p = (float*)W.data();
                std::mt19937 rng(7);
                std::uniform_real_distribution<float> u(-1.f, 1.f);
                for (size_t i = 0; i < wbytes / 4; i++) p[i] = u(rng);
            } else if (c.block_bytes == 2) {
                __fp16* p = (__fp16*)W.data();
                std::mt19937 rng(7);
                std::uniform_real_distribution<float> u(-1.f, 1.f);
                for (size_t i = 0; i < wbytes / 2; i++) p[i] = (__fp16)u(rng);
            }

            id<MTLBuffer> bw = buf(wbytes);
            std::memcpy([bw contents], W.data(), wbytes);
            id<MTLBuffer> yold = buf((size_t)n_out * kBmax * 4);
            id<MTLBuffer> ynew = buf((size_t)n_out * kBmax * 4);
            std::memset([yold contents], 0, (size_t)n_out * kBmax * 4);
            std::memset([ynew contents], 0, (size_t)n_out * kBmax * 4);

            uint32_t nin = n_in, nout = n_out, nr = nrow;
            uint64_t woff = 0;

            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];

            id<MTLComputePipelineState> po = pipe(c.oldk);
            id<MTLComputePipelineState> pn = pipe(c.newk);
            if (!po || !pn) { [en endEncoding]; [cb commit]; continue; }

            // 旧内核：一个 simdgroup 一行
            [en setComputePipelineState:po];
            [en setBuffer:bw offset:0 atIndex:0];
            [en setBuffer:bxt offset:0 atIndex:1];
            [en setBuffer:yold offset:0 atIndex:2];
            [en setBytes:&nin length:4 atIndex:3];
            [en setBytes:&woff length:8 atIndex:4];
            [en setBytes:&nout length:4 atIndex:5];
            [en setBytes:&nr length:4 atIndex:6];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_out / 4), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            // 新内核：一个 threadgroup kMmRows 行
            [en setComputePipelineState:pn];
            [en setBuffer:bw offset:0 atIndex:0];
            [en setBuffer:bxt offset:0 atIndex:1];
            [en setBuffer:ynew offset:0 atIndex:2];
            [en setBytes:&nin length:4 atIndex:3];
            [en setBytes:&woff length:8 atIndex:4];
            [en setBytes:&nout length:4 atIndex:5];
            [en setBytes:&nr length:4 atIndex:6];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_out + kMmRows - 1) / kMmRows), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kMmSgPerTg * 32, 1, 1)];
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            std::vector<float> a((size_t)n_out * kBmax), b((size_t)n_out * kBmax);
            std::memcpy(a.data(), [yold contents], a.size() * 4);
            std::memcpy(b.data(), [ynew contents], b.size() * 4);
            cmp(c.newk, a, b);

            // Q4_K 额外和 CPU 参考对一遍，确定是哪一侧错
            if (std::strcmp(c.newk, "mm2_q4_k") == 0) {
                std::vector<float> ref((size_t)n_out * kBmax, 0.f);
                std::vector<float> row((size_t)n_in);
                for (int r = 0; r < n_out; r++) {
                    deq_q4k_row(W.data() + (size_t)r * nblocks_row * 144,
                                (int)nblocks_row, row.data());
                    for (int j = 0; j < kBmax; j++) {
                        double acc = 0;
                        for (int e = 0; e < n_in; e++)
                            acc += (double)row[e] * XT[(size_t)e * kBmax + j];
                        ref[(size_t)j * n_out + r] = (float)acc;
                    }
                }
                cmp("old vs cpu", ref, a);
                cmp("new vs cpu", ref, b);
            }
        }

        std::printf("\n%s\n", g_fail == 0 ? "全部一致" : "有失败项");
        return g_fail == 0 ? 0 : 1;
    }
}
