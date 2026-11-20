// gpu_optest.mm — 融合算子逐个对拍（Metal 实现 vs C 参考实现）
//
// 为什么需要它：整个前向是 410 个 dispatch 编进一个 command buffer，
// 只要有一个算子写错，症状就是"输出乱码"，从外面完全看不出是哪个。
// 这个工具把每个算子单独拉出来比，几秒钟就能定位。
//
// 用法：make optest && ./build/optest
#include "metal_kernels.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLLibrary> g_lib;
static int g_fail = 0;

static id<MTLComputePipelineState> pipe(const char* name) {
    id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        std::fprintf(stderr, "  缺少函数 %s\n", name);
        std::exit(1);
    }
    NSError* e = nil;
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) {
        std::fprintf(stderr, "  pipeline %s 失败: %s\n", name, e ? e.localizedDescription.UTF8String : "?");
        std::exit(1);
    }
    return p;
}

static id<MTLBuffer> buf(size_t bytes) {
    id<MTLBuffer> buffer = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
        std::fprintf(stderr, "  无法分配 %zu 字节的 Metal 缓冲区\n", bytes);
        std::exit(1);
    }
    return buffer;
}

static bool run(id<MTLCommandBuffer> cb, const char* name) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusCompleted) return true;
    NSError* e = cb.error;
    std::fprintf(stderr, "  %s 的 Metal 命令执行失败: %s\n", name,
                 e ? e.localizedDescription.UTF8String : "无可用的命令缓冲区");
    return false;
}

static void report(const char* name, const float* ref, const float* got, int n, double tolerance = 1e-4) {
    double mx = 0;
    int at = 0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            std::printf("  %-12s ✗ 非有限值 @%d  ref=%.6f got=%.6f\n", name, i, ref[i], got[i]);
            g_fail++;
            return;
        }
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        if (d > mx) { mx = d; at = i; }
    }
    const bool ok = mx < tolerance;
    if (!ok) g_fail++;
    std::printf("  %-12s %s  maxerr=%.3e @%d  ref=%.6f got=%.6f\n",
                name, ok ? "✓" : "✗ 失败", mx, at, ref[at], got[at]);
}

static void fill(std::vector<float>& v, unsigned seed = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& x : v) x = d(rng);
}

int main() {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { std::fprintf(stderr, "没有 Metal 设备\n"); return 77; }
        g_q = [g_dev newCommandQueue];
        if (!g_q) { std::fprintf(stderr, "无法创建 Metal 命令队列\n"); return 1; }
        NSError* e = nil;
        g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kDubheMSL]
                                    options:[MTLCompileOptions new]
                                      error:&e];
        if (!g_lib) {
            std::printf("MSL 编译失败: %s\n", e ? [[e localizedDescription] UTF8String] : "?");
            return 1;
        }
        std::printf("设备 %s\n", [[g_dev name] UTF8String]);

        // ---------------------------------------------------------- rmsnorm
        for (int rows : {1, 3}) {
            const int n = 896;
            std::printf("[k_rmsnorm] n=%d rows=%d\n", n, rows);
            std::vector<float> x((size_t)n * rows), w(n), ref((size_t)n * rows);
            fill(x, 7); fill(w, 8);
            const float eps = 1e-5f;
            for (int r = 0; r < rows; r++) {
                const float* xr = x.data() + (size_t)r * n;
                float* o = ref.data() + (size_t)r * n;
                double ss = 0;
                for (int i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
                const float sc = (float)(1.0 / std::sqrt(ss / n + eps));
                for (int i = 0; i < n; i++) o[i] = xr[i] * sc * w[i];
            }
            id<MTLBuffer> bx = buf(x.size() * 4), bw = buf(w.size() * 4), bo = buf(ref.size() * 4);
            std::memcpy([bx contents], x.data(), x.size() * 4);
            std::memcpy([bw contents], w.data(), w.size() * 4);
            uint32_t nn = (uint32_t)n;
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pipe("k_rmsnorm")];
            [en setBuffer:bx offset:0 atIndex:0];
            [en setBuffer:bw offset:0 atIndex:1];
            [en setBuffer:bo offset:0 atIndex:2];
            [en setBytes:&nn length:4 atIndex:3];
            [en setBytes:&eps length:4 atIndex:4];
            uint64_t zero = 0;
            [en setBytes:&zero length:8 atIndex:5];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [en endEncoding];
            if (!run(cb, "k_rmsnorm")) return 1;
            report("rmsnorm", ref.data(), (float*)[bo contents], (int)ref.size());
        }

        // ---------------------------------------------------------- add / bias / silu / copy
        for (int rows : {1, 3}) {
            const int n = 896;
            std::printf("[k_add] n=%d rows=%d\n", n, rows);
            std::vector<float> a((size_t)n * rows), b((size_t)n * rows);
            fill(a, 11); fill(b, 12);
            std::vector<float> ref(a);
            for (size_t i = 0; i < ref.size(); i++) ref[i] += b[i];
            id<MTLBuffer> ba = buf(a.size() * 4), bb = buf(b.size() * 4);
            std::memcpy([ba contents], a.data(), a.size() * 4);
            std::memcpy([bb contents], b.data(), b.size() * 4);
            uint32_t nn = (uint32_t)n;
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pipe("k_add")];
            [en setBuffer:ba offset:0 atIndex:0];
            [en setBuffer:bb offset:0 atIndex:1];
            [en setBytes:&nn length:4 atIndex:2];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n, (NSUInteger)rows, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [en endEncoding];
            if (!run(cb, "k_add")) return 1;
            report("add", ref.data(), (float*)[ba contents], (int)ref.size());

            std::printf("[k_bias] n=%d rows=%d\n", n, rows);
            std::vector<float> y((size_t)n * rows), bias(n);
            fill(y, 13); fill(bias, 14);
            std::vector<float> ref2(y);
            for (int r = 0; r < rows; r++)
                for (int i = 0; i < n; i++) ref2[(size_t)r * n + i] += bias[i];
            id<MTLBuffer> by = buf(y.size() * 4), bbias = buf(bias.size() * 4);
            std::memcpy([by contents], y.data(), y.size() * 4);
            std::memcpy([bbias contents], bias.data(), bias.size() * 4);
            cb = [g_q commandBuffer];
            en = [cb computeCommandEncoder];
            [en setComputePipelineState:pipe("k_bias")];
            [en setBuffer:by offset:0 atIndex:0];
            [en setBuffer:bbias offset:0 atIndex:1];
            uint64_t boff = 0;
            [en setBytes:&boff length:8 atIndex:2];
            [en setBytes:&nn length:4 atIndex:3];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n, (NSUInteger)rows, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [en endEncoding];
            if (!run(cb, "k_bias")) return 1;
            report("bias", ref2.data(), (float*)[by contents], (int)ref2.size());
        }

        {
            const int n = 4864;
            std::printf("[k_silu_mul] n=%d rows=2\n", n);
            std::vector<float> g((size_t)n * 2), u((size_t)n * 2), ref((size_t)n * 2);
            fill(g, 21); fill(u, 22);
            for (size_t i = 0; i < ref.size(); i++)
                ref[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
            id<MTLBuffer> bg = buf(g.size() * 4), bu = buf(u.size() * 4), bo = buf(ref.size() * 4);
            std::memcpy([bg contents], g.data(), g.size() * 4);
            std::memcpy([bu contents], u.data(), u.size() * 4);
            uint32_t nn = (uint32_t)n;
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pipe("k_silu_mul")];
            [en setBuffer:bo offset:0 atIndex:0];
            [en setBuffer:bg offset:0 atIndex:1];
            [en setBuffer:bu offset:0 atIndex:2];
            [en setBytes:&nn length:4 atIndex:3];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n, 2, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [en endEncoding];
            if (!run(cb, "k_silu_mul")) return 1;
            report("silu_mul", ref.data(), (float*)[bo contents], (int)ref.size());
        }

        // ---------------------------------------------------------- rope
        //
        // 注意 buffer(9) 的 `angles`：Round 6 给 k_rope 加了 partial RoPE
        // （zero-tail，只有前 angles 个频率真的转），但这里当时没跟着加，
        // 于是这个用例一直在拿**野值**当 angles 用 —— 表现是 rope.q /
        // rope.kcache 常年报失败，而引擎本身是对的。加参数时一定要回头
        // 看所有直接调内核的工具。
        //
        // 两个用例各测一次：angles = hd/2 = 32（全量 RoPE）和 16
        // （partial，尾部 16 个频率必须原样透传）。
        for (int angles : {32, 16}) for (int rows : {1, 3}) {
            const int n_head = 14, n_kvh = 2, hd = 64, pos0 = 5;
            const int q_dim = n_head * hd, k_dim = n_kvh * hd;
            std::printf("[k_rope] hd=%d rows=%d angles=%d pos0=%d\n", hd, rows, angles, pos0);
            std::vector<float> q((size_t)q_dim * rows), k((size_t)k_dim * rows);
            std::vector<float> refq, refk((size_t)k_dim * rows);
            fill(q, 31); fill(k, 32);
            refq = q;
            refk = k;
            const float theta = 10000.f;
            auto rope_one = [&](float* v, int nrow_head, int row) {
                for (int h = 0; h < nrow_head; h++) {
                    float* base = v + (size_t)h * hd;
                    for (int i = 0; i < hd / 2; i++) {
                        if (i >= angles) continue;   // zero-tail：不转，原样透传
                        const float freq = 1.f / std::pow(theta, 2.f * i / hd);
                        const float a = (pos0 + row) * freq;
                        const float c = std::cos(a), s = std::sin(a);
                        const float x0 = base[i], x1 = base[i + hd / 2];
                        base[i] = x0 * c - x1 * s;
                        base[i + hd / 2] = x0 * s + x1 * c;
                    }
                }
            };
            for (int r = 0; r < rows; r++) {
                rope_one(refq.data() + (size_t)r * q_dim, n_head, r);
                rope_one(refk.data() + (size_t)r * k_dim, n_kvh, r);
            }
            // KV cache：3 层 × 16 槽 × k_dim
            const int cap = 16, layer = 1;
            const size_t cache_bytes = (size_t)3 * cap * k_dim * sizeof(_Float16);
            const size_t start = ((size_t)layer * cap + pos0) * k_dim;
            const uint64_t koff = start * sizeof(_Float16);
            id<MTLBuffer> bq = buf(q.size() * 4), bk = buf(k.size() * 4), bkc = buf(cache_bytes);
            std::memcpy([bq contents], q.data(), q.size() * 4);
            std::memcpy([bk contents], k.data(), k.size() * 4);
            std::memset([bkc contents], 0, cache_bytes);
            uint32_t a = n_head, b = n_kvh, c = hd, p0 = pos0;
            float th = theta;
            uint32_t ang = (uint32_t)angles;
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pipe("k_rope")];
            [en setBuffer:bq offset:0 atIndex:0];
            [en setBuffer:bk offset:0 atIndex:1];
            [en setBuffer:bkc offset:0 atIndex:2];
            [en setBytes:&koff length:8 atIndex:3];
            [en setBytes:&a length:4 atIndex:4];
            [en setBytes:&b length:4 atIndex:5];
            [en setBytes:&c length:4 atIndex:6];
            [en setBytes:&p0 length:4 atIndex:7];
            [en setBytes:&th length:4 atIndex:8];
            [en setBytes:&ang length:4 atIndex:9];
            [en dispatchThreads:MTLSizeMake((NSUInteger)((n_head + n_kvh) * (hd / 2)), (NSUInteger)rows, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [en endEncoding];
            if (!run(cb, "k_rope")) return 1;
            report("rope.q", refq.data(), (float*)[bq contents], (int)refq.size());
            const auto* kcp = static_cast<const _Float16*>([bkc contents]) + start;
            const std::vector<float> gotk(kcp, kcp + refk.size());
            // K 的结果写入 fp16，容差包含最后一次舍入。
            report("rope.kcache", refk.data(), gotk.data(), (int)gotk.size(), 5e-4);
        }

        // ---------------------------------------------------------- attn（split-K）
        struct AttentionCase { uint32_t window, wrap; };
        const AttentionCase cases[] = {{0, 0}, {32, 0}, {0, 31}};
        for (const auto& mode : cases) for (int rows : {1, 3}) for (int nsp : {1, 4}) {
            const int n_head = 14, n_kvh = 2, hd = 64, pos0 = 200;
            const int kv_dim = n_kvh * hd, q_dim = n_head * hd;
            const int cap = 256;
            std::printf("[k_attn_split] hd=%d rows=%d nsp=%d pos0=%d win=%u kvwrap=%u\n",
                        hd, rows, nsp, pos0, mode.window, mode.wrap);
            const int nlen0 = pos0 + 1;
            std::vector<float> q((size_t)q_dim * rows);
            std::vector<float> kc((size_t)cap * kv_dim), vc((size_t)cap * kv_dim);
            fill(q, 41); fill(kc, 42); fill(vc, 43);
            const std::vector<_Float16> khalf(kc.begin(), kc.end()), vhalf(vc.begin(), vc.end());
            std::vector<float> ref((size_t)q_dim * rows, 0.f);
            const float scale = 1.f / std::sqrt((float)hd);
            const int mul = n_head / n_kvh;
            auto cache_row = [&](int t) {
                return (size_t)(mode.wrap ? ((uint32_t)t & mode.wrap) : (uint32_t)t) * kv_dim;
            };
            for (int r = 0; r < rows; r++) {
                const int nlen = nlen0 + r;
                const int start = mode.window && nlen > (int)mode.window ? nlen - (int)mode.window : 0;
                for (int h = 0; h < n_head; h++) {
                    const int kvh = h / mul;
                    const float* qh = q.data() + (size_t)r * q_dim + (size_t)h * hd;
                    std::vector<double> sc(nlen - start);
                    double mx = -1e30;
                    for (int t = start; t < nlen; t++) {
                        const _Float16* key = khalf.data() + cache_row(t) + kvh * hd;
                        double s = 0;
                        for (int d = 0; d < hd; d++) s += (double)qh[d] * key[d];
                        sc[t - start] = s * scale;
                        if (s * scale > mx) mx = s * scale;
                    }
                    double sum = 0;
                    for (double& score : sc) { score = std::exp(score - mx); sum += score; }
                    float* o = ref.data() + (size_t)r * q_dim + (size_t)h * hd;
                    for (int d = 0; d < hd; d++) {
                        double acc = 0;
                        for (int t = start; t < nlen; t++) {
                            acc += sc[t - start] * vhalf[cache_row(t) + kvh * hd + d];
                        }
                        o[d] = (float)(acc / sum);
                    }
                }
            }
            id<MTLBuffer> bq = buf(q.size() * 4), bkc = buf(khalf.size() * sizeof(_Float16));
            id<MTLBuffer> bvc = buf(vhalf.size() * sizeof(_Float16)), bo = buf(ref.size() * 4);
            id<MTLBuffer> bp = buf((size_t)rows * nsp * n_head * (hd + 2) * 4);
            std::memcpy([bq contents], q.data(), q.size() * 4);
            std::memcpy([bkc contents], khalf.data(), khalf.size() * sizeof(_Float16));
            std::memcpy([bvc contents], vhalf.data(), vhalf.size() * sizeof(_Float16));
            uint32_t a = n_head, b = n_kvh, c = hd, n0 = (uint32_t)nlen0, kvd = kv_dim;
            uint32_t nspu = (uint32_t)nsp;
            float sc2 = scale;
            uint64_t coff = 0;
            id<MTLCommandBuffer> cb = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pipe("k_attn_split")];
            [en setBuffer:bq offset:0 atIndex:0];
            [en setBuffer:bkc offset:0 atIndex:1];
            [en setBuffer:bvc offset:0 atIndex:2];
            [en setBuffer:bp offset:0 atIndex:3];
            [en setBytes:&coff length:8 atIndex:4];
            [en setBytes:&a length:4 atIndex:5];
            [en setBytes:&b length:4 atIndex:6];
            [en setBytes:&c length:4 atIndex:7];
            [en setBytes:&n0 length:4 atIndex:8];
            [en setBytes:&kvd length:4 atIndex:9];
            [en setBytes:&sc2 length:4 atIndex:10];
            [en setBytes:&nspu length:4 atIndex:11];
            [en setBytes:&mode.window length:sizeof(mode.window) atIndex:12];
            [en setBytes:&mode.wrap length:sizeof(mode.wrap) atIndex:13];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                 (NSUInteger)rows, (NSUInteger)nsp)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [en setComputePipelineState:pipe("k_attn_comb")];
            [en setBuffer:bp offset:0 atIndex:0];
            [en setBuffer:bo offset:0 atIndex:1];
            [en setBytes:&a length:4 atIndex:2];
            [en setBytes:&c length:4 atIndex:3];
            [en setBytes:&nspu length:4 atIndex:4];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                 (NSUInteger)rows, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [en endEncoding];
            if (!run(cb, "k_attn_split/k_attn_comb")) return 1;
            report("attn", ref.data(), (float*)[bo contents], (int)ref.size());
        }

        std::printf("\n%s\n", g_fail == 0 ? "全部通过" : "有失败项");
        return g_fail == 0 ? 0 : 1;
    }
}
