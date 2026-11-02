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
#include <chrono>
#include <cmath>
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

// ---------------------------------------------------------------- 双向 GQA 注意力
//
// 分工：一个 threadgroup = 4 个 simdgroup，**一个 simdgroup 负责一个 (head, query) 行**。
// lane 认领行内元素：HD=64 时每条 lane 拿 2 个（lane 和 lane+32）。
//
// 为什么不用 flash-attention 那套 MMA 分块：这里是条件最简单的一档 ——
// 无因果掩码（全 N² 都要算）、无 KV cache、无分页、fp32 进 fp32 出、
// 一次性算完。没有 mask 就不需要跨 dispatch 的在线状态，也就没有寄存器
// 压力；而 211 GFLOP 在 1.79 TFLOP/s 的 lane 吞吐下约 120 ms，MMA 的
// 复杂度换不来多少收益。先把"10 倍"拿到手，再看要不要上 MMA。
//
// 用**在线 softmax**（running max + 回缩）而不是先存 N 个分数：
// N=3840 时一行分数就是 15 KB，塞进 threadgroup memory 会把 occupancy 打死，
// 而放回 global 是 945 MB。
//
// `s > m` 这个分支在整个 warp 上是一致的（s 由 simd_sum 广播），所以没有发散。
//
// NT 必须是编译期常量：qv[]/acc[] 用运行时的 nt 索引会被编译器放进 local
// memory（等于每次都走一次内存），所以按 HD/32 特化成三档。
#define VIT_ATTN_R 4u   /* 一个 simdgroup 同时算几行 */

#define VIT_ATTN_KERNEL(NAME, NT)                                              \
kernel void NAME(device const float* q     [[buffer(0)]],                      \
                 device const float* k     [[buffer(1)]],                      \
                 device const float* v     [[buffer(2)]],                      \
                 device float* o           [[buffer(3)]],                      \
                 constant uint& N          [[buffer(4)]],                      \
                 constant uint& H          [[buffer(5)]],                      \
                 constant uint& KV         [[buffer(6)]],                      \
                 constant uint& HD         [[buffer(7)]],                      \
                 constant float& scale     [[buffer(8)]],                      \
                 uint tg   [[threadgroup_position_in_grid]],                   \
                 uint lane [[thread_index_in_simdgroup]],                      \
                 uint sgi  [[simdgroup_index_in_threadgroup]])                 \
{                                                                              \
    const uint rows_per_tg = 4u * VIT_ATTN_R;   /* 4 个 simdgroup */           \
    const uint nblk = (N + rows_per_tg - 1u) / rows_per_tg;                    \
    const uint head = tg / nblk;                                               \
    const uint row0 = (tg % nblk) * rows_per_tg + sgi * VIT_ATTN_R;            \
    if (row0 >= N || head >= H) return;                                        \
    const uint nr = min(VIT_ATTN_R, N - row0);                                 \
                                                                               \
    const uint qd  = H * HD;                                                   \
    const uint kd  = KV * HD;                                                  \
    const uint kvh = head / (H / KV);                                          \
    device const float* kb = k + (ulong)kvh * HD;                              \
    device const float* vb = v + (ulong)kvh * HD;                              \
                                                                               \
    float qv[VIT_ATTN_R][NT];                                                  \
    float acc[VIT_ATTN_R][NT];                                                 \
    float m[VIT_ATTN_R];                                                       \
    float l[VIT_ATTN_R];                                                       \
    _Pragma("unroll")                                                          \
    for (uint r = 0; r < VIT_ATTN_R; r++) {                                    \
        device const float* qr = q + (ulong)(row0 + r) * qd + head * HD;       \
        _Pragma("unroll")                                                      \
        for (uint t = 0; t < NT; t++) {                                        \
            qv[r][t]  = qr[t * 32u + lane];                                    \
            acc[r][t] = 0.0f;                                                  \
        }                                                                      \
        m[r] = -INFINITY;                                                      \
        l[r] = 0.0f;                                                           \
    }                                                                          \
                                                                               \
    for (uint j = 0; j < N; j++) {                                             \
        device const float* kj = kb + (ulong)j * kd;                           \
        float w[VIT_ATTN_R];                                                   \
        _Pragma("unroll")                                                      \
        for (uint r = 0; r < VIT_ATTN_R; r++) {                                \
            float p = 0.0f;                                                    \
            _Pragma("unroll")                                                  \
            for (uint t = 0; t < NT; t++) p += qv[r][t] * kj[t * 32u + lane];  \
            const float s = simd_sum(p) * scale;                               \
            /* w 是"当前元素"的权重，c 是"把已有累加回缩"的系数 —— 两者不同。 */\
            /* 第一版把 c 当 w 用了：首次迭代 m=-inf 于是 c=exp(-inf)=0，     */\
            /* 等于每行都把自己的第一个元素乘了 0。症状是"看着对但差 10%"。  */\
            if (s > m[r]) {                                                    \
                const float c = exp(m[r] - s);                                 \
                l[r] = l[r] * c + 1.0f;                                        \
                _Pragma("unroll")                                              \
                for (uint t = 0; t < NT; t++) acc[r][t] *= c;                  \
                m[r] = s;                                                      \
                w[r] = 1.0f;                                                   \
            } else {                                                           \
                w[r] = exp(s - m[r]);                                          \
                l[r] += w[r];                                                  \
            }                                                                  \
        }                                                                      \
        device const float* vj = vb + (ulong)j * kd;                           \
        _Pragma("unroll")                                                      \
        for (uint r = 0; r < VIT_ATTN_R; r++) {                                \
            _Pragma("unroll")                                                  \
            for (uint t = 0; t < NT; t++) acc[r][t] += w[r] * vj[t * 32u + lane]; \
        }                                                                      \
    }                                                                          \
                                                                               \
    for (uint r = 0; r < nr; r++) {                                            \
        const float inv = 1.0f / l[r];                                         \
        device float* orow = o + (ulong)(row0 + r) * qd + head * HD;           \
        _Pragma("unroll")                                                      \
        for (uint t = 0; t < NT; t++) orow[t * 32u + lane] = acc[r][t] * inv;  \
    }                                                                          \
}

VIT_ATTN_KERNEL(vit_attn1, 1)   // HD = 32
VIT_ATTN_KERNEL(vit_attn2, 2)   // HD = 64
VIT_ATTN_KERNEL(vit_attn4, 4)   // HD = 128
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

// 注意力专用（q/k/v/o 形状与 matmul 不同，分开分配，避免互相扩容踩踏）
id<MTLBuffer> g_bq = nil, g_bk = nil, g_bv = nil, g_bo = nil;
size_t g_bq_cap = 0, g_bk_cap = 0, g_bv_cap = 0, g_bo_cap = 0;

id<MTLComputePipelineState> g_pipe_f16 = nil, g_pipe_q8 = nil;
id<MTLComputePipelineState> g_pipe_q4  = nil, g_pipe_q6 = nil;
id<MTLComputePipelineState> g_pipe_tr  = nil;
id<MTLComputePipelineState> g_pipe_attn1 = nil, g_pipe_attn2 = nil,
                            g_pipe_attn4 = nil;

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
    g_pipe_attn1 = make_pipe("vit_attn1");
    g_pipe_attn2 = make_pipe("vit_attn2");
    g_pipe_attn4 = make_pipe("vit_attn4");
    // 注意力流水线缺了不算致命：调用方会落回 CPU 版（只是慢 10 倍），
    // 而 matmul 那几条缺了就等于整个视觉塔回 CPU。分开判。
    if (!g_pipe_attn1 && !g_pipe_attn2 && !g_pipe_attn4)
        std::fprintf(stderr, "[vit] 注意力内核不可用，双向注意力走 CPU\n");

    // mmap 起始地址是页对齐的，可以直接零拷贝包成 MTLBuffer。
    //
    // 长度必须**向上**取整到页，不能直接把文件大小递进去：
    // 实测两个 mmproj（VL 2B / Audio 2B）的文件大小分别是
    // 203031872 % 4096 = 1344、337862080 % 4096 = 3520 —— 都不是页整数倍。
    // Metal 的零拷贝包装按页登记 vm 区间，长度取整方向一旦是向下，
    // 尾部那一页里的张量就整块落在缓冲之外（和 metal.mm 里
    // bind_weights 那个 bug 同源，那边已经修过，这里一并堵上）。
    //
    // 向上取整是安全的：mmap 的映射本来就覆盖到页边界，超出 EOF 的字节
    // 读出来是 0，不会缺页。g_wsize 本身保持精确值不动 —— 它还要兼作
    // 张量地址的越界检查上界（见 vit_gemm_gpu 里的 `wd >= g_base + g_wsize`）。
    const size_t page = 4096;
    if (((uintptr_t)g_base & (page - 1)) != 0) {
        g_err = "mmproj 映射基址未页对齐，无法零拷贝";
        g_dead = true;
        return false;
    }
    const size_t wlen = (g_wsize + page - 1) & ~(page - 1);
    g_w = [g_dev newBufferWithBytesNoCopy:(void*)g_base
                                   length:wlen
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

// 双向 GQA 注意力。一层一次 dispatch，一次同步。
//
// 拷贝成本：N=3840 时 Q/O 各 15.7 MB、K/V 各 7.9 MB，一层 47 MB，
// 7 个注意力层合计 330 MB，按 50 GB/s 算 6.6 ms —— 相对要省的 4.2 秒可以忽略。
// （真要再省，得把整个塔的激活都搬进 MTLBuffer，那要改 mmproj.cpp 的全部
//  中间变量，等这一版量出收益之后再说。）
bool vit_attn_gpu(const float* Q, const float* K, const float* V, int64_t N,
                  int H, int KV, int HD, float* O) {
    // 两级开关：DUBHE_VIT_CPU 关掉整个视觉塔 GPU；DUBHE_VIT_ATTN_CPU 只关注意力。
    // 后者是为了把"注意力这一项"和"matmul 那一路"的误差分开量 —— 两个都开
    // 的时候对拍只看到总误差，分不清是谁贡献的。
    static const bool force_cpu = std::getenv("DUBHE_VIT_CPU") != nullptr;
    static const bool attn_cpu  = std::getenv("DUBHE_VIT_ATTN_CPU") != nullptr;
    if (force_cpu || attn_cpu || g_dead) return false;
    if (!Q || !K || !V || !O || N <= 0) return false;
    if (H <= 0 || KV <= 0 || (H % KV) != 0) return false;
    // 内核按 HD/32 特化成 1/2/4 三档；其它 head_dim 落回 CPU
    if (HD != 32 && HD != 64 && HD != 128) return false;
    if (N > 0x7fffffff) return false;

    if (!init_gpu()) return false;

    id<MTLComputePipelineState> p = (HD == 32)  ? g_pipe_attn1
                                  : (HD == 64)  ? g_pipe_attn2
                                                : g_pipe_attn4;
    if (!p) return false;

    const size_t qbytes = (size_t)N * (size_t)H * (size_t)HD * 4u;
    const size_t kbytes = (size_t)N * (size_t)KV * (size_t)HD * 4u;
    if (!ensure(&g_bq, &g_bq_cap, qbytes)) return false;
    if (!ensure(&g_bk, &g_bk_cap, kbytes)) return false;
    if (!ensure(&g_bv, &g_bv_cap, kbytes)) return false;
    if (!ensure(&g_bo, &g_bo_cap, qbytes)) return false;

    const bool trace = std::getenv("DUBHE_VIT_TRACE") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    std::memcpy([g_bq contents], Q, qbytes);
    std::memcpy([g_bk contents], K, kbytes);
    std::memcpy([g_bv contents], V, kbytes);

    uint32_t n = (uint32_t)N, h = (uint32_t)H, kv = (uint32_t)KV, hd = (uint32_t)HD;
    float sc = 1.0f / std::sqrt((float)HD);

    const uint32_t nblk = (n + 15u) / 16u;  // 与 MSL 里的 rows_per_tg = 4*VIT_ATTN_R 一致

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:p];
    [en setBuffer:g_bq offset:0 atIndex:0];
    [en setBuffer:g_bk offset:0 atIndex:1];
    [en setBuffer:g_bv offset:0 atIndex:2];
    [en setBuffer:g_bo offset:0 atIndex:3];
    [en setBytes:&n  length:4 atIndex:4];
    [en setBytes:&h  length:4 atIndex:5];
    [en setBytes:&kv length:4 atIndex:6];
    [en setBytes:&hd length:4 atIndex:7];
    [en setBytes:&sc length:4 atIndex:8];
    [en dispatchThreadgroups:MTLSizeMake((NSUInteger)nblk * (NSUInteger)h, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) {
        g_err = "视觉塔注意力执行失败";
        if (std::getenv("DUBHE_VIT_DEBUG")) std::fprintf(stderr, "[vit] %s\n", g_err.c_str());
        return false;
    }

    std::memcpy(O, [g_bo contents], qbytes);
    if (trace) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[vit] attn N=%lld H=%d KV=%d HD=%d  %.1f ms\n",
                     (long long)N, H, KV, HD, ms);
    }
    g_used = true;
    return true;
}

void vit_gpu_release() {
    g_bxt = nil; g_by = nil; g_bx = nil; g_w = nil;
    g_bq = g_bk = g_bv = g_bo = nil;
    g_bx_cap = g_by_cap = g_bxt_cap = 0;
    g_bq_cap = g_bk_cap = g_bv_cap = g_bo_cap = 0;
    g_pipe_f16 = g_pipe_q8 = g_pipe_q4 = g_pipe_q6 = g_pipe_tr = nil;
    g_pipe_attn1 = g_pipe_attn2 = g_pipe_attn4 = nil;
    g_lib = nil; g_q = nil; g_dev = nil;
    g_ready = false;
    g_used  = false;
    g_base  = nullptr;
    g_wsize = 0;
}

}  // namespace dubhe
