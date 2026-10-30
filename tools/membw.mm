// membw.mm — 这台 GPU 的"只读带宽天花板"探针
//
// 为什么需要它
// ------------
// decode 是带宽受限的：每 token 要把整个权重读一遍。所以"当前 GB/s 离天花板
// 还有多远"决定了继续优化内核还值不值。之前几轮反复在这上面犯错：
// 拿 F16 的 82 GB/s 当参照，就以为 Q4_K 的 69 GB/s 还有 20% 空间 ——
// 但两者的**访问形状**不同（F16 每 lane 一条 uint4 连续读，Q4_K 一个 warp
// 拆成 4 个块、每块 8 条 lane 读 128 字节）。形状不同，天花板就不同。
//
// 这个探针把三件事分别量出来：
//   stream   连续读：lane 步长 = 1 条 uint4，warp 覆盖 512 字节连续区
//   block4   仿 mv_q4_k：一个 warp 分 4 组，每组 8 条 lane 覆盖 128 字节
//   block1   一个 warp 只读一个块区（128 字节 × 32 lane 全重叠）
//
// 三个数的**最大值**才是"随便怎么优化都到不了"的那条线。如果 mv_q4_k 已经
// 贴着它，就说明该换方向（并发、更小的模型、更低的量化），而不是继续抠指令。
//
// 用法：
//   make membw && ./build/membw [字节数] [重复轮数]
#include <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// 连续读：每个 lane 每次读一条 uint4，warp 内 32 lane 覆盖 512 字节。
kernel void k_stream(device const uint4* p [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     constant ulong& nvec   [[buffer(2)]],
                     constant ulong& stride [[buffer(3)]],
                     uint tid [[thread_position_in_grid]])
{
    // grid-stride：让 occupancy 由硬件决定，不由网格大小决定。
    uint4 acc = uint4(0);
    for (ulong i = (ulong)tid; i < nvec; i += stride) {
        const uint4 v = p[i];
        acc ^= v;
    }
    // 必须把结果写出去，否则整个循环会被优化掉。
    if ((acc.x | acc.y | acc.z | acc.w) == 0xFFFFFFFFu) out[tid & 1023u] = 1.0f;
}

// 仿 mv_q4_k：一个 warp 分 4 个"块"，每块 8 条 lane，每条 lane 读一条 uint4。
// 一个 144 字节的块只用到 128 字节数据区（块头 16 字节不读），所以地址按
// 128 字节块 + 16 字节偏移算，和真实内核一致。
kernel void k_block4(device const uchar* p [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     constant ulong& nblk   [[buffer(2)]],
                     constant ulong& nw     [[buffer(3)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]])
{
    const ulong wid  = (ulong)tgid * 4u + sg;         // 全局 warp 号
    const uint  blk  = lane >> 3;                     // 0..3
    const uint  m    = lane & 7u;                     // 0..7
    uint4 acc = uint4(0);
    for (ulong b = wid; b < nblk; b += nw) {
        const uchar* q = p + b * 144u + 16u + m * 16u;
        acc ^= *(device const uint4*)q;
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xFFFFFFFFu) out[blk] = 1.0f;
}

// 一个 warp 读整行（n_in=2048 → 8 块 × 144 = 1152 字节 = 72 个 16 字节单元）。
// 单元按 lane 连续切：lane 读 u = it*32 + lane，u 覆盖 0..71 中的 0..31 / 32..63 /
// 64..71。这样一条 warp 指令读的是 **512 字节连续且 512 对齐** 的区间 ——
// 块头（u%9==0）不再挖洞。这是"改布局"方案的可行性预演。
kernel void k_row(device const uchar* p [[buffer(0)]],
                  device float* out      [[buffer(1)]],
                  constant ulong& nrow   [[buffer(2)]],
                  constant ulong& nw     [[buffer(3)]],
                  uint tgid [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg   [[simdgroup_index_in_threadgroup]])
{
    const ulong wid = (ulong)tgid * 4u + sg;
    uint4 acc = uint4(0);
    for (ulong r = wid; r < nrow; r += nw) {
        const uchar* base = p + r * 1152u;
        #pragma unroll
        for (uint it = 0; it < 3u; it++) {
            const uint u = it * 32u + lane;
            if (u < 72u) acc ^= *(device const uint4*)(base + u * 16u);
        }
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xFFFFFFFFu) out[lane & 31u] = 1.0f;
}

// 一个 warp 只读 128 字节（32 lane 全重叠在同一块）。用来验证
// "warp 内地址重复"是否会被内存系统合并。
kernel void k_block1(device const uchar* p [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     constant ulong& nblk   [[buffer(2)]],
                     constant ulong& nw     [[buffer(3)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]])
{
    const ulong wid = (ulong)tgid * 4u + sg;
    uint4 acc = uint4(0);
    for (ulong b = wid; b < nblk; b += nw) {
        const uchar* q = p + b * 144u + 16u + (lane & 7u) * 16u;
        acc ^= *(device const uint4*)q;
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xFFFFFFFFu) out[lane & 31u] = 1.0f;
}
)MSL";

static double now_ms() {
    return (double)CFAbsoluteTimeGetCurrent() * 1000.0;
}

int main(int argc, char** argv) {
    const size_t bytes = (argc > 1) ? (size_t)strtoull(argv[1], nullptr, 10)
                                    : (size_t)1073741824ull;
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("没有 Metal 设备\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:[MTLCompileOptions new]
                                             error:&e];
    if (!lib) {
        std::printf("编译失败: %s\n", e ? [[e localizedDescription] UTF8String] : "?");
        return 1;
    }

    id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!buf) { std::printf("分配 %zu 字节失败\n", bytes); return 1; }
    // 用一条 blob 内核把页填上，避免第一次读付 page-in 的代价
    std::vector<uint8_t> tmp(std::min<size_t>(bytes, 1u << 20), 0xA5);
    id<MTLBuffer> up = [dev newBufferWithBytes:tmp.data() length:tmp.size()
                                       options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        for (size_t o = 0; o < bytes; o += tmp.size())
            [bl copyFromBuffer:up sourceOffset:0 toBuffer:buf destinationOffset:o
                        size:std::min(tmp.size(), bytes - o)];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    id<MTLBuffer> out = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];

    struct Case { const char* name; const char* fn; bool is_vec; };
    const Case cases[] = {
        {"stream",  "k_stream",  true},
        {"block4",  "k_block4",  false},
        {"block1",  "k_block1",  false},
        {"row",     "k_row",     false},
    };

    const uint64_t nvec = (uint64_t)(bytes / 16ull);
    const uint64_t nblk = (uint64_t)(bytes / 144ull);
    std::printf("缓冲 %.2f GiB，%d 轮，报告最小值\n\n", (double)bytes / 1073741824.0, reps);
    std::printf("| 形状 | 有效读量 | 耗时 | 带宽 |\n|---|---:|---:|---:|\n");

    for (const Case& c : cases) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:c.fn]];
        id<MTLComputePipelineState> ps =
            [dev newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) { std::printf("| %s | pipeline 失败 |\n", c.name); continue; }

        // 派发网格：让每个 core 有足够多的组。128 线程/组，组数取 640。
        const NSUInteger tgs = 640;

        // 有效读量：k_stream 每条 lane 读 16 字节，覆盖整个缓冲；
        // block4/block1 只读每个 144 字节块的 128 字节数据区。
        const uint64_t nrow = (uint64_t)(bytes / 1152ull);
        const double read_bytes = c.is_vec ? (double)bytes
                              : (c.fn[2] == 'r' ? (double)nrow * 1152.0
                                                : (double)nblk * 128.0);

        double best = 1e30;
        for (int r = 0; r < reps + 2; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:ps];
            [en setBuffer:buf offset:0 atIndex:0];
            [en setBuffer:out offset:0 atIndex:1];
            if (c.is_vec) {
                const uint64_t stride = (uint64_t)tgs * 128u;
                [en setBytes:&nvec length:8 atIndex:2];
                [en setBytes:&stride length:8 atIndex:3];
                [en dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                const uint64_t nw = (uint64_t)tgs * 4u;
                const uint64_t cnt = (c.fn[2] == 'r') ? nrow : nblk;
                [en setBytes:&cnt length:8 atIndex:2];
                [en setBytes:&nw length:8 atIndex:3];
                [en dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            [en endEncoding];
            const double t0 = now_ms();
            [cb commit];
            [cb waitUntilCompleted];
            const double ms = now_ms() - t0;
            if (r < 2) continue;  // 预热
            if (ms < best) best = ms;
        }
        const double gbs = read_bytes / (best * 1e6);
        std::printf("| %s | %.2f GiB | %.2f ms | **%.1f GB/s** |\n",
                    c.name, read_bytes / 1073741824.0, best, gbs);
    }
    return 0;
}
