// mma_probe.mm — 判断 Apple GPU 的 simdgroup_matrix 是硬件矩阵单元还是标量模拟
//
// 为什么必须先做这个实验：prefill 的 GEMM 有两种可能的写法，
//   A) 标量/向量 FMA：每指令 1 个 MAC/lane（float4 是 4 个 MAC / 4 拍）
//   B) simdgroup_matrix：每指令 8×8×8 = 512 个 MAC / simdgroup
// 如果 B 是硬件实现的，它的 MAC/cycle 会明显高于 A，值得为它重写整个
// prefill 路径（要把权重量化成 half 搬进 threadgroup memory）。
// 如果 B 只是标量展开，那重写毫无意义，应该走"二维分块 + 共享激活"。
//
// 用法：./build/mma_probe
#include <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <chrono>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// 每线程每轮 4 条 float4 FMA = 16 个 MAC
kernel void bench_fma(device float* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint tid [[thread_position_in_grid]])
{
    float4 a = float4(1.0000001f, 1.0000002f, 1.0000003f, 1.0000004f);
    float4 b = float4(0.9999999f);
    float4 c0 = float4(0.f), c1 = float4(0.f), c2 = float4(0.f), c3 = float4(0.f);
    for (uint i = 0; i < iters; i++) {
        c0 = fma(a, b, c0);
        c1 = fma(a, b, c1);
        c2 = fma(a, b, c2);
        c3 = fma(a, b, c3);
    }
    out[tid] = c0.x + c1.x + c2.x + c3.x;
}

// fp16 版本：每线程每轮 4 条 half4 FMA = 16 个 MAC。
// 若它明显快于 float4，说明 ALU 对 half 是双发射的 —— 那权重就该解成
// half 并在 half 上累加（需要分段累加控制误差）。
kernel void bench_hfma(device float* out [[buffer(0)]],
                       constant uint& iters [[buffer(1)]],
                       uint tid [[thread_position_in_grid]])
{
    half4 a = half4(1.001h, 1.002h, 1.003h, 1.004h);
    half4 b = half4(0.999h);
    half4 c0 = half4(0.h), c1 = half4(0.h), c2 = half4(0.h), c3 = half4(0.h);
    for (uint i = 0; i < iters; i++) {
        c0 = fma(a, b, c0);
        c1 = fma(a, b, c1);
        c2 = fma(a, b, c2);
        c3 = fma(a, b, c3);
    }
    out[tid] = float(c0.x + c1.x + c2.x + c3.x);
}

// 每 simdgroup 每轮 2 条 8x8x8 MMA = 1024 个 MAC
kernel void bench_mma(device float* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint sgid [[simdgroup_index_in_threadgroup]])
{
    threadgroup half sa[64];
    threadgroup half sb[64];
    if (tid < 64) { sa[tid] = half(float(tid) * 0.001f); sb[tid] = half(1.0f); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 a, b;
    simdgroup_load(a, sa, 8);
    simdgroup_load(b, sb, 8);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint i = 0; i < iters; i++) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
    }
    threadgroup float sink[64];
    simdgroup_store(c0, sink, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) out[sgid] = sink[0] + sink[7];
}
)MSL";

static id<MTLComputePipelineState> mk(id<MTLDevice> dev, id<MTLLibrary> lib, const char* n) {
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:n]];
    if (!f) { std::printf("缺少 %s\n", n); return nil; }
    NSError* e = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) std::printf("pipeline %s: %s\n", n, e ? [[e localizedDescription] UTF8String] : "?");
    return p;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("没有 Metal 设备\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:[MTLCompileOptions new]
                                                 error:&e];
        if (!lib) {
            std::printf("MSL 编译失败: %s\n", e ? [[e localizedDescription] UTF8String] : "?");
            return 1;
        }
        std::printf("设备 %s  (%lu 核)\n", [[dev name] UTF8String],
                    (unsigned long)[dev maxThreadsPerThreadgroup].width);

        const int TG = 4096;          // 4096 个线程 = 1024 个 simdgroup
        const int ITERS = 20000;
        id<MTLBuffer> out = [dev newBufferWithLength:TG * 4 options:MTLResourceStorageModeShared];
        uint32_t iters = ITERS;

        struct Case { const char* name; const char* fn; int macs_per_sg_per_iter; int sg_per_tg; };
        Case cases[] = {
            {"float4 FMA",  "bench_fma", 16 * 32, 4},   // 每线程 16 MAC × 32 lane
            {"half4 FMA",   "bench_hfma", 16 * 32, 4},
            {"8x8x8 MMA",   "bench_mma", 1024,    4},
        };

        for (auto& c : cases) {
            id<MTLComputePipelineState> p = mk(dev, lib, c.fn);
            if (!p) continue;
            // 预热
            for (int rep = 0; rep < 2; rep++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setComputePipelineState:p];
                [en setBuffer:out offset:0 atIndex:0];
                [en setBytes:&iters length:4 atIndex:1];
                [en dispatchThreadgroups:MTLSizeMake(TG / 128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [en endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            const int REP = 5;
            double best = 1e30;
            for (int rep = 0; rep < REP; rep++) {
                auto t0 = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setComputePipelineState:p];
                [en setBuffer:out offset:0 atIndex:0];
                [en setBytes:&iters length:4 atIndex:1];
                [en dispatchThreadgroups:MTLSizeMake(TG / 128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [en endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                auto t1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best) best = ms;
            }
            const double sg = (double)TG / 32.0;
            const double macs = sg * (double)ITERS * (double)c.macs_per_sg_per_iter;
            const double tflops = macs * 2.0 / (best * 1e-3) / 1e12;
            std::printf("  %-12s %8.2f ms   %7.2f TFLOP/s (fp32 等效)\n", c.name, best, tflops);
        }
        return 0;
    }
}
