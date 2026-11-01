// vit_gpu.mm — 视觉塔的 GPU 分块 matmul（Metal）
//
// 设计
// ----
// 视觉塔的每一次 matmul 都是同一种形状：Y[t][o] = Σ_i X[t][i]·W[o][i]，
// t 是 patch / token 数（最大 1024+），o 是输出行数。这和语言模型 prefill
// 的 gemm 是**同一个形状**，所以直接复用第三代的 mm3_* 内核：
//   * 一个 threadgroup 算 64 个输出行 × 32 个 token
//   * simdgroup_matrix 8×8×8，权重反量化进 threadgroup memory
// 只需要自己做两件事：把激活转置成 [n_in][32]，以及按 32 token 一批派发。
//
// 为什么不用 metal.mm 的 gemm：那个走的是 g_w（语言模型的 mmap），单例。
// 这里自己持有一份权重缓冲，指到 mmproj 的 mmap 上，零拷贝。
//
// 拷贝成本
// --------
// X / Y 是调用方（mmproj.cpp）的 std::vector，不是 MTLBuffer，所以每次调用
// 要 memcpy 进出。一次前向约 100 次 matmul，合计 ~1.4 GB 的 memcpy，
// 按 50 GB/s 算 ~28 ms —— 相对 5 s 的基线可以忽略，换来的是整条链路不用改
// 数据结构。真要再快一档，就把激活也搬进 MTLBuffer（那要改 mmproj.cpp 的
// 全部中间变量，留到有需要时再做）。
#include "vit_gpu.h"

#include "internal.h"
#include "metal_kernels.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dubhe {
namespace {

// 视觉塔专用的一小段 MSL，拼在 kDubheMSL 后面一起编译。
//
// 为什么不用现成的 k_trans：k_trans 一次只写一段（`dst[e*stride+j]`），而
// 视觉塔有 32 个 token 块要转置。分 32 次写同一块缓冲，Metal 的资源依赖
// 追踪会把相邻派发全部串行化 —— 实测正是这个让 matmul 只有 440 GFLOP/s
// （独立基准 2.19 TFLOP/s）。这里一次写完，并且排成 chunk-major
// `[chunk][n_in][32]`，每块的起点 = chunk·n_in·128 字节，天然满足
// setBuffer:offset: 的 256 字节对齐要求，块之间再无任何依赖。
static const char* kVitMSL = R"MSL(
kernel void vit_trans_chunks(device const float* src [[buffer(0)]],
                             device float* dst       [[buffer(1)]],
                             constant uint& n        [[buffer(2)]],
                             constant uint& n_tok    [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const uint k = gid.x;    // 元素下标（行内第几个）
    const uint c = gid.y;    // token 块号
    if (k >= n) return;
    device float* d = dst + ((ulong)c * n + k) * 32u;
    const uint t0 = c * 32u;
    for (uint m = 0; m < 32u; m++)
        d[m] = (t0 + m < n_tok) ? src[(ulong)(t0 + m) * n + k] : 0.0f;
}
)MSL";

constexpr int kTok  = dubhe_msl::kMmaBm;   // 一个 threadgroup 吃多少 token（32）
constexpr int kRows = dubhe_msl::kMmaBn;   // 一个 threadgroup 算多少输出行（64）

id<MTLDevice>       g_dev  = nil;
id<MTLCommandQueue> g_q    = nil;
id<MTLLibrary>      g_lib  = nil;
id<MTLBuffer>       g_w    = nil;    // mmproj 的 mmap（零拷贝）
const uint8_t*      g_base = nullptr;
size_t              g_wsize = 0;

id<MTLBuffer> g_bx  = nil;   // 激活 [n_tok][n_in]（稠密）
id<MTLBuffer> g_by  = nil;   // 输出 [n_tok][n_out]
id<MTLBuffer> g_bxt = nil;   // 转置后的激活 [n_in][kTok]
size_t g_bx_cap = 0, g_by_cap = 0, g_bxt_cap = 0;

id<MTLComputePipelineState> g_pipe_f16 = nil, g_pipe_q8 = nil;
id<MTLComputePipelineState> g_pipe_q4  = nil, g_pipe_q6 = nil;
id<MTLComputePipelineState> g_pipe_tr  = nil;

bool g_ready = false;
bool g_dead  = false;     // 初始化失败过就别每次重试
bool g_used  = false;
std::string g_err;

id<MTLComputePipelineState> make_pipe(const char* name) {
    id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { g_err = std::string("找不到内核 ") + name; return nil; }
    NSError* e = nil;
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { g_err = std::string(name) + ": " + (e ? [[e localizedDescription] UTF8String] : "?"); }
    return p;
}

bool ensure(id<MTLBuffer> __strong* buf, size_t* cap, size_t need) {
    if (*buf && *cap >= need) return true;
    size_t n = *cap ? *cap : (1u << 20);
    while (n < need) n *= 2;
    *buf = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!*buf) { g_err = "分配 Metal 缓冲失败"; return false; }
    *cap = n;
    return true;
}

bool init_gpu() {
    if (g_ready) return true;
    if (g_dead)  return false;
    if (!g_base) { g_err = "权重尚未登记"; g_dead = true; return false; }

    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { g_err = "没有 Metal 设备"; g_dead = true; return false; }
    g_q = [g_dev newCommandQueue];
    if (!g_q) { g_err = "创建命令队列失败"; g_dead = true; return false; }

    NSError* e = nil;
    std::string src = std::string(kDubheMSL) + kVitMSL;
    g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                options:[MTLCompileOptions new]
                                  error:&e];
    if (!g_lib) {
        g_err = std::string("视觉塔着色器编译失败: ") +
                (e ? [[e localizedDescription] UTF8String] : "?");
        g_dead = true;
        return false;
    }

    g_pipe_f16 = make_pipe("mm3_f16");
    g_pipe_q8  = make_pipe("mm3_q8_0");
    g_pipe_q4  = make_pipe("mm3_q4_k");
    g_pipe_q6  = make_pipe("mm3_q6_k");
    g_pipe_tr  = make_pipe("vit_trans_chunks");
    if (!g_pipe_tr) { g_dead = true; return false; }

    // mmap 起始地址是页对齐的，可以直接零拷贝包成 MTLBuffer。
    g_w = [g_dev newBufferWithBytesNoCopy:(void*)g_base
                                   length:g_wsize
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
    if (!g_w) { g_err = "包装权重 mmap 失败"; g_dead = true; return false; }

    if (std::getenv("DUBHE_VIT_DEBUG"))
        std::fprintf(stderr, "[vit] GPU 就绪：%s，权重 %.1f MB\n",
                     [[g_dev name] UTF8String], (double)g_wsize / 1048576.0);
    g_ready = true;
    return true;
}

}  // namespace

void vit_bind(const void* base, size_t size) {
    if (g_base == base && g_wsize == size) return;
    vit_gpu_release();
    g_base  = (const uint8_t*)base;
    g_wsize = size;
    g_dead  = false;
}

bool vit_gemm_gpu(const void* wdata, uint32_t type, int64_t n_in, int64_t n_out,
                  const float* X, int64_t n_tok, float* Y, int64_t x_stride) {
    // 关掉的口子：DUBHE_VIT_CPU=1 强制走 CPU，用来 A/B 和回归对拍。
    static const bool force_cpu = std::getenv("DUBHE_VIT_CPU") != nullptr;
    if (force_cpu || g_dead) return false;

    if (!g_base || !wdata || !X || !Y) return false;
    if (n_in <= 0 || n_out <= 0 || n_tok <= 0) return false;
    // 带跨度的只有 patchify（每行只取输入的一段），那一步占总量不到 1%，
    // 留给 CPU 更省事 —— GPU 侧的转置内核假定行跨度就是 n_in。
    if (x_stride != n_in) return false;
    // mm3_* 的 K 分块依赖完整的 256 元素块（Q4_K/Q6_K 的块跨度、F16 的对齐）。
    if ((n_in % 256) != 0) return false;

    const uint8_t* wd = (const uint8_t*)wdata;
    if (wd < g_base || wd >= g_base + g_wsize) return false;

    if (!init_gpu()) return false;

    id<MTLComputePipelineState> p = nil;
    switch (type) {
        case T_F16:  p = g_pipe_f16; break;
        case T_Q8_0: p = g_pipe_q8;  break;
        case T_Q4_K: p = g_pipe_q4;  break;
        case T_Q6_K: p = g_pipe_q6;  break;
        default: return false;
    }
    if (!p) return false;

    // 转置缓冲按 chunk-major 排：[chunk][n_in][32]，stride 恒为 32。
    const int64_t n_chunk = (n_tok + kTok - 1) / kTok;
    const size_t xbytes = (size_t)n_tok * (size_t)n_in * 4;
    const size_t ybytes = (size_t)n_tok * (size_t)n_out * 4;
    const size_t tbytes = (size_t)n_chunk * (size_t)n_in * kTok * 4;
    if (!ensure(&g_bx, &g_bx_cap, xbytes)) return false;
    if (!ensure(&g_by, &g_by_cap, ybytes)) return false;
    if (!ensure(&g_bxt, &g_bxt_cap, tbytes)) return false;

    // X 是稠密的 [n_tok][n_in]，一次搬完。
    std::memcpy([g_bx contents], X, xbytes);

    uint32_t nin = (uint32_t)n_in, nout = (uint32_t)n_out;
    uint32_t nr  = (uint32_t)n_tok;
    const uint64_t woff = (uint64_t)(wd - g_base);
    const NSUInteger tgc = (NSUInteger)((n_out + kRows - 1) / kRows);

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];

    // 整段转置**一次**做完。块之间从此只读同一份结果，互不依赖，GPU 可以
    // 把后面的 32 个 matmul 派发叠起来跑（见 kVitMSL 上的注释）。
    [en setComputePipelineState:g_pipe_tr];
    [en setBuffer:g_bx offset:0 atIndex:0];
    [en setBuffer:g_bxt offset:0 atIndex:1];
    [en setBytes:&nin length:4 atIndex:2];
    [en setBytes:&nr length:4 atIndex:3];
    [en dispatchThreads:MTLSizeMake((NSUInteger)n_in, (NSUInteger)n_chunk, 1)
  threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];

    const uint32_t stride = (uint32_t)kTok;
    for (int64_t c = 0; c < n_chunk; c++) {
        const uint32_t take = (uint32_t)std::min<int64_t>(kTok, n_tok - c * kTok);
        const uint64_t yoff = (uint64_t)c * (uint64_t)kTok * (uint64_t)n_out;  // 元素为单位

        [en setComputePipelineState:p];
        [en setBuffer:g_w offset:0 atIndex:0];
        // 第 c 块起点 = c·n_in·32 个 float = c·n_in·128 字节，256 字节对齐成立。
        [en setBuffer:g_bxt offset:(NSUInteger)((size_t)c * (size_t)n_in * kTok * 4) atIndex:1];
        [en setBuffer:g_by offset:0 atIndex:2];
        [en setBytes:&nin length:4 atIndex:3];
        [en setBytes:&woff length:8 atIndex:4];
        [en setBytes:&nout length:4 atIndex:5];
        [en setBytes:&take length:4 atIndex:6];
        [en setBytes:&stride length:4 atIndex:7];
        [en setBytes:&yoff length:8 atIndex:8];
        [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        g_err = "视觉塔 matmul 执行失败";
        if (std::getenv("DUBHE_VIT_DEBUG"))
            std::fprintf(stderr, "[vit] %s\n", g_err.c_str());
        return false;
    }

    std::memcpy(Y, [g_by contents], ybytes);
    g_used = true;
    return true;
}

bool vit_gpu_active() { return g_used; }

void vit_gpu_release() {
    g_bxt = nil; g_by = nil; g_bx = nil; g_w = nil;
    g_bx_cap = g_by_cap = g_bxt_cap = 0;
    g_pipe_f16 = g_pipe_q8 = g_pipe_q4 = g_pipe_q6 = g_pipe_tr = nil;
    g_lib = nil; g_q = nil; g_dev = nil;
    g_ready = false;
    g_used  = false;
    g_base  = nullptr;
    g_wsize = 0;
}

}  // namespace dubhe
