// metal.mm — Metal GPU 后端
//
// 三个关键设计：
//   1. 着色器在**运行时**编译（newLibraryWithSource:）。这样不需要离线
//      Metal Toolchain（`xcodebuild -downloadComponent MetalToolchain`），
//      任何装了 Xcode CLT 的机器都能跑。代价是加载时多 ~200ms。
//   2. 权重零拷贝：整个 GGUF 的 mmap 区域直接包成一个 MTLBuffer
//      （MTLResourceStorageModeShared + Apple 统一内存）。每张量用 woff
//      字节偏移定位，绕开 setBuffer:offset: 的 256 字节对齐限制。
//      → 内存占用与 CPU 路径完全相同，不翻倍。
//   3. 一个线程算一行。B=1 解码时 n_out 通常 896~151936，远大于 GPU 核数，
//      这是最自然的并行粒度，而且每行的累加顺序与 CPU 基准路径一致，
//      所以 GPU 结果应当与 CPU 逐位接近（相对误差 ~1e-6）。
//
// 非 Apple 平台不编译本文件，Model::matvec 会一直走 CPU 路径。
#include "internal.h"
#include "metal.h"
#include "metal_kernels.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace dubhe {
namespace metal {

namespace {

// 与着色器里的 SG_PER_TG 必须一致：一个 threadgroup 装 4 个 simdgroup。
constexpr int kSgPerTg = 4;
// 与着色器里的 MV_R 必须一致：解码时一个 simdgroup 同时算几行。
// 解码是延迟受限的，一个 simdgroup 只算一行时 outstanding load 太少，
// 带宽上不去（Q4_K 实测只有 39 GB/s，而 Q8_0 能到 67 GB/s）。
//
// 这个数在 init() 里被 DUBHE_MV_R 覆盖，并在编译着色器前作为 `#define MV_R`
// 注入到 MSL 源码最前面 —— 两边必须同一个值，所以只有这一个来源。
int g_mv_rows = 2;

// ---- 二维分块 matmul（prefill）的派发参数，必须与着色器一致
constexpr int kMmSgPerTg = 8;    // 256 线程/threadgroup
constexpr int kMmRows    = 32;   // MMR：每 threadgroup 的输出行数
constexpr int kMmTileK   = 256;  // TILE_K：K 方向分片

enum KernelKind {
    K_F32 = 0, K_F16, K_Q4_0, K_Q5_0, K_Q8_0, K_Q4_K, K_Q6_K,
    // 新格式一律**追加在末尾**：所有表都是按这个下标平行索引的，
    // 插在中间会让老下标整体错位（症状是格式张冠李戴，不报错）。
    K_Q5_1, K_Q5_K, K_COUNT
};

const char* const kNames[K_COUNT] = {
    "mv_f32", "mv_f16", "mv_q4_0", "mv_q5_0", "mv_q8_0", "mv_q4_k", "mv_q6_k",
    "mv_q5_1", "mv_q5_k"
};

// 解码 matvec 的候选实现（第 2 版）。空串 = 这一档没有候选，直接用 kNames。
//
// 为什么单开一张表而不是直接替换：新内核必须能在**同一个二进制**里 A/B，
// 否则又要回到"跨轮比较绝对数字"的老坑（同一份二进制跨轮能差 1.5x）。
// 用 DUBHE_MV_V2=0 关掉，默认开。
const char* const kNames2[K_COUNT] = {
    "", "", "mv_q4_0_v2", "mv_q5_0_v2", "", "", "", "", ""
};

// 分块 matmul 的函数名。Q4_0/Q5_0 没有分块版本（空串），派发时回退到 mm_*。
const char* const kM2Names[K_COUNT] = {
    "mm2_f32", "mm2_f16", "", "", "mm2_q8_0", "mm2_q4_k", "mm2_q6_k",
    "mm2_q5_1", "mm2_q5_k"
};

// MMA 版分块 matmul（第 3 代）。覆盖 F16 / Q8_0 / Q4_K / Q6_K / Q5_1 / Q5_K，
// 其余回退 mm2_*。空串表示"没有这一档"，派发时跳过。
const char* const kM3Names[K_COUNT] = {
    "", "mm3_f16", "", "", "mm3_q8_0", "mm3_q4_k", "mm3_q6_k",
    "mm3_q5_1", "mm3_q5_k"
};

// 与 MSL 的 MMA_BN / MMA_BM / MMA_NK 一致（全部取自 metal_kernels.h 的唯一真源）
constexpr int kMmaBn = dubhe_msl::kMmaBn;
constexpr int kMmaBm = dubhe_msl::kMmaBm;
constexpr int kMmaNk = dubhe_msl::kMmaNk;

id<MTLDevice>       g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLLibrary>      g_lib = nil;
id<MTLBuffer>       g_w = nil;        // 整个 mmap（零拷贝）
id<MTLBuffer>       g_x = nil;        // 激活（可复用，按需扩容）
id<MTLBuffer>       g_y = nil;        // 输出（可复用，按需扩容）
size_t              g_x_cap = 0;
size_t              g_y_cap = 0;
bool                g_ready = false;
std::string         g_err;
id<MTLComputePipelineState> g_pipe[K_COUNT] = {nil};
id<MTLComputePipelineState> g_pipe2[K_COUNT] = {nil};

int kind_of(uint32_t t) {
    switch (t) {
        case T_F32:  return K_F32;
        case T_F16:  return K_F16;
        case T_Q4_0: return K_Q4_0;
        case T_Q5_0: return K_Q5_0;
        case T_Q5_1: return K_Q5_1;
        case T_Q8_0: return K_Q8_0;
        case T_Q4_K: return K_Q4_K;
        case T_Q5_K: return K_Q5_K;
        case T_Q6_K: return K_Q6_K;
        default:     return -1;
    }
}

bool ensure_buf(id<MTLBuffer> __strong* buf, size_t* cap, size_t need) {
    if (*buf && *cap >= need) return true;
    size_t n = *cap ? *cap : 4096;
    while (n < need) n *= 2;
    *buf = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!*buf) return false;
    *cap = n;
    return true;
}

void fwd_release();   // 定义在下面（同一个匿名命名空间）

}  // namespace

// ---------------------------------------------------------------- 生命周期

bool available() {
    static int cached = -1;
    if (cached < 0) cached = MTLCreateSystemDefaultDevice() != nil ? 1 : 0;
    return cached == 1;
}

bool init(std::string* err) {
    if (g_ready) return true;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) {
        if (err) *err = "没有可用的 Metal 设备";
        return false;
    }
    g_queue = [g_dev newCommandQueue];
    if (!g_queue) {
        if (err) *err = "创建 Metal 命令队列失败";
        return false;
    }

    // 默认就是 fast math（MTLCompileOptions 的 fastMathEnabled 默认 YES），
    // 这里不显式设置以免踩到 macOS 15 的弃用告警。
    MTLCompileOptions* opt = [MTLCompileOptions new];
    NSError* e = nil;

    // MV_R 是解码 matvec 的行数（一个 simdgroup 同时算几行）。它决定寄存器
    // 压力和 outstanding load 深度，最优值随量化格式和 GPU 代数变，所以做成
    // 运行期注入：着色器源码最前面拼一行 `#define MV_R n`，C++ 侧读同一个
    // g_mv_rows 算派发网格。两边**必须**一致，否则行号错位 —— 结果是错的
    // 而不是崩的，这种 bug 最贵。
    if (const char* s = getenv("DUBHE_MV_R")) {
        const int v = atoi(s);
        if (v >= 1 && v <= 16) g_mv_rows = v;
    }
    std::string src = "#define MV_R " + std::to_string(g_mv_rows) + "\n" + kDubheMSL;
    g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                options:opt
                                  error:&e];
    if (!g_lib) {
        g_err = e ? [[e localizedDescription] UTF8String] : "MSL 编译失败";
        // 这个错误以前只塞进 g_err 就返回了，调用方静默退回 CPU ——
        // 症状是"莫名其妙慢十倍"，查了很久。必须打出来。
        std::fprintf(stderr, "\n[metal] MSL 编译失败: %s\n\n", g_err.c_str());
        if (err) *err = "MSL 编译失败: " + g_err;
        return false;
    }

    // 第 2 版解码内核：有就编，编不出来就退回第 1 版（不是致命错误）。
    // 默认开，DUBHE_MV_V2=0 关掉用于 A/B。
    const bool want_v2 = getenv("DUBHE_MV_V2") == nullptr ||
                         atoi(getenv("DUBHE_MV_V2")) != 0;
    for (int i = 0; i < K_COUNT; i++) {
        id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kNames[i]]];
        if (!fn) {
            if (err) *err = std::string("着色器缺少函数 ") + kNames[i];
            return false;
        }
        NSError* pe = nil;
        g_pipe[i] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!g_pipe[i]) {
            g_err = pe ? [[pe localizedDescription] UTF8String] : "pipeline 创建失败";
            if (err) *err = std::string("pipeline ") + kNames[i] + " 失败: " + g_err;
            return false;
        }
        if (want_v2 && kNames2[i][0] != '\0') {
            id<MTLFunction> fn2 = [g_lib newFunctionWithName:
                [NSString stringWithUTF8String:kNames2[i]]];
            if (fn2) {
                NSError* pe2 = nil;
                g_pipe2[i] = [g_dev newComputePipelineStateWithFunction:fn2 error:&pe2];
                if (!g_pipe2[i])
                    std::fprintf(stderr, "  %s pipeline 失败，退回 %s\n",
                                 kNames2[i], kNames[i]);
            } else {
                std::fprintf(stderr, "  着色器缺少 %s，退回 %s\n", kNames2[i], kNames[i]);
            }
        }
    }
    g_ready = true;
    return true;
}

// 只解绑权重与前向缓冲。device / queue / library 保留——它们是进程级
// 共享资源，卸载一个模型不应该把别的模型一起废掉。
void unbind_weights() {
    fwd_release();
    g_w = nil;
    g_x = nil;
    g_y = nil;
    g_x_cap = g_y_cap = 0;
}

void shutdown() {
    fwd_release();
    for (int i = 0; i < K_COUNT; i++) { g_pipe[i] = nil; g_pipe2[i] = nil; }
    g_w = nil;
    g_x = nil;
    g_y = nil;
    g_x_cap = g_y_cap = 0;
    g_lib = nil;
    g_queue = nil;
    g_dev = nil;
    g_ready = false;
}

const char* device_name() {
    if (!g_dev) return "";
    return [[g_dev name] UTF8String];
}

// ---------------------------------------------------------------- 权重绑定

// base 必须是页对齐的 mmap 起始地址；size 为映射长度。
bool bind_weights(const void* base, size_t size, std::string* err) {
    if (!g_ready) { if (err) *err = "Metal 未初始化"; return false; }
    if (!base || size == 0) { if (err) *err = "权重地址无效"; return false; }

    // 必须页对齐，且长度取整到页（Metal 会校验）
    //
    // 取整方向只能是**向上**。原先是向下取整，等于把文件最后一页里的张量
    // 整块切掉 —— 而 GGUF 的 `output_norm.weight` 恰恰经常就落在那里。
    // 症状极其隐蔽：那个张量的 GPU 侧读出来全是 0，于是最后一层 rmsnorm
    // 输出恒 0、logits 恒 0、argmax 恒 0（融合校验报 rel=1.0e+00 才发现）。
    // 实测 Qwen2.5-0.5B 的 q5_0 文件：49,0471,776 + 3584 = 文件尾，
    // 向下取整后这块整个在缓冲之外；q4_0 文件只被切掉最后 1888 字节，
    // 前半段还对，所以校验勉强通过 —— 是个一直在的静默精度损失。
    //
    // 向上取整是安全的：mmap 的映射长度本来就按页上取整，最后一页里超出
    // EOF 的字节读出来是 0，不会缺页。
    const size_t page = 4096;
    if (((uintptr_t)base & (page - 1)) != 0) {
        if (err) *err = "mmap 基址未页对齐，无法零拷贝";
        return false;
    }
    const size_t len = (size + page - 1) & ~(page - 1);
    if (len == 0) { if (err) *err = "映射长度不足一页"; return false; }

    g_w = [g_dev newBufferWithBytesNoCopy:(void*)base
                                   length:len
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
    if (!g_w) {
        if (err) *err = "权重零拷贝绑定失败（文件过大或地址未对齐）";
        return false;
    }
    return true;
}

bool supports(uint32_t type, int n_in) {
    const int k = kind_of(type);
    if (k < 0) return false;
    // 量化格式按整块切分，n_in 必须是块大小的整数倍
    const int bs = type_block_size(type);
    if (bs > 1 && (n_in % bs) != 0) return false;
    return g_ready && g_w != nil;
}

// ---------------------------------------------------------------- 矩阵乘

// x: [n_in] f32；y: [n_out] f32；woff: 张量在 mmap 内的字节偏移。
bool mfvec(uint32_t type, const float* x, float* y, int n_out, int n_in, uint64_t woff) {
    if (!supports(type, n_in) || n_out <= 0) return false;
    const int k = kind_of(type);

    if (!ensure_buf(&g_x, &g_x_cap, sizeof(float) * (size_t)n_in)) return false;
    if (!ensure_buf(&g_y, &g_y_cap, sizeof(float) * (size_t)n_out)) return false;

    std::memcpy([g_x contents], x, sizeof(float) * (size_t)n_in);

    uint32_t nin = (uint32_t)n_in;
    uint32_t nout = (uint32_t)n_out;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:(g_pipe2[k] ? g_pipe2[k] : g_pipe[k])];
    [enc setBuffer:g_w offset:0 atIndex:0];
    [enc setBuffer:g_x offset:0 atIndex:1];
    [enc setBuffer:g_y offset:0 atIndex:2];
    [enc setBytes:&nin length:sizeof(nin) atIndex:3];
    [enc setBytes:&woff length:sizeof(woff) atIndex:4];
    [enc setBytes:&nout length:sizeof(nout) atIndex:5];

    // 一个 threadgroup = 4 个 simdgroup，每个 simdgroup 算 g_mv_rows 行。
    // 必须按 threadgroup 派发并自己判越界：dispatchThreads 的非均匀裁剪
    // 只在 thread 粒度生效，而这里内核用的是 threadgroup_position_in_grid。
    const NSUInteger per_tg = (NSUInteger)(kSgPerTg * g_mv_rows);
    const NSUInteger tgc = (NSUInteger)((n_out + per_tg - 1) / per_tg);
    [enc dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        g_err = cb.error ? [[cb.error localizedDescription] UTF8String] : "命令缓冲执行失败";
        return false;
    }
    std::memcpy(y, [g_y contents], sizeof(float) * (size_t)n_out);
    return true;
}

const char* last_error() { return g_err.c_str(); }

// ================================================================ 全 GPU 前向
//
// 两种模式共用一套缓冲与一套命令流：
//   nrow == 1  → 解码。走 mv_*（matvec），只有第 0 行算 logits。
//   nrow  > 1  → prefill。走 mm_*（matmul），权重只读一遍喂 nrow 个 token，
//                且只有**最后一行**算 logits（out_w 占权重流量的 36%，
//                prefill 时其余行不需要它）。
//
// 为什么 prefill 要批处理：解码时每 token 必须把全部权重读一遍，带宽就是
// 天花板（实测 GPU 有 92% 的时间在等权重）。批处理把同一遍读取摊到 nrow
// 个 token 上，权重算术强度 ×nrow。这是 TTFT 从"秒级"掉到"百毫秒级"的原因。

namespace {

// 注意：不要再写一个字面量 8。MSL 是字符串，BMAX 对 C++ 不可见，所以
// 唯一真源放在 metal_kernels.h 的 namespace dubhe_msl 里。这里直接从那里
// 取，改 BMAX 时两边不会漂移（漂移的症状是结果错乱而不是编译错误）。
//
// 这里有两个批大小，别再混：
//   kBmax    = 8  —— mm2_* / mm_* 内核一次吃几个 token（受寄存器限制）
//   kBatchMax= 32 —— 一次前向最多提交几个 token（受 mm3_* 的 MMA_BM 限制）
// 两者不一致，所以超过 8 行而该张量没有 MMA 内核时，gemm 里要按 8 行分批。
constexpr int kBmax     = dubhe_msl::kBmax;
constexpr int kBatchMax = dubhe_msl::kMmaBm;
// 注意力 key 区间最多切几段（split-K）。
// 段数决定长上下文的并行度：解码时一个 simdgroup 算一个 head 的一段 key，
// 并行度 = (n_head/4) × nsp 个 threadgroup。n_head=16 时 nsp=8 只有 32 个
// threadgroup（128 warp），对 10 个核来说延迟掩盖不够；提到 32 是 128 个
// threadgroup（512 warp）。代价只是 combine 阶段多读几个 partial。
constexpr int kAttnSplits = 64;

enum FwdKernel {
    F_RMSNORM = 0, F_ADD, F_BIAS, F_SILU, F_ROPE, F_COPY, F_COPY_H,
    F_ATTN, F_ATTN_SPLIT, F_ATTN_COMB, F_TRANS,
    F_HEADNORM, F_GATE_MUL, F_MUL, F_CONV, F_CONV_SHIFT, F_COUNT
};
const char* const kFwdNames[F_COUNT] = {
    "k_rmsnorm", "k_add", "k_bias", "k_silu_mul", "k_rope", "k_copy", "k_copy_h",
    "k_attn", "k_attn_split", "k_attn_comb", "k_trans",
    "k_headnorm", "k_gate_mul", "k_mul", "k_conv", "k_conv_shift"
};
id<MTLComputePipelineState> g_fpipe[F_COUNT] = {nil};

// 批处理矩阵乘内核。顺序与 KernelKind 完全一致，所以可以直接用 kind_of()
// 的下标同时索引 g_pipe[]（解码）与 g_mpipe[]（prefill）。
const char* const kMmNames[K_COUNT] = {
    "mm_f32", "mm_f16", "mm_q4_0", "mm_q5_0", "mm_q8_0", "mm_q4_k", "mm_q6_k",
    "mm_q5_1", "mm_q5_k"
};
id<MTLComputePipelineState> g_mpipe[K_COUNT] = {nil};
// 二维分块版本（prefill 主路径）。与 g_mpipe 并行存在，格式不支持时为 nil。
id<MTLComputePipelineState> g_m2pipe[K_COUNT] = {nil};
// MMA 版本（prefill 第 3 代）。nil 表示该格式没有 MMA 内核，回退 m2。
id<MTLComputePipelineState> g_m3pipe[K_COUNT] = {nil};

Spec         g_spec;
bool         g_fwd_ready = false;

// 前向用到的所有激活缓冲。按 kBatchMax 行分配，解码时只用第 0 行。
id<MTLBuffer> g_bx = nil, g_bxb = nil, g_bq = nil, g_bk = nil, g_bv = nil;
id<MTLBuffer> g_batt = nil, g_bgate = nil, g_bup = nil, g_bffn = nil, g_blogits = nil;
id<MTLBuffer> g_bkc = nil, g_bvc = nil;
id<MTLBuffer> g_bfinal = nil;   // 最后一行的 norm 输出（避免原地 aliasing）
id<MTLBuffer> g_bxt = nil;      // 转置后的激活 [n_in][BMAX]（prefill 用）
id<MTLBuffer> g_bpart = nil;    // 注意力的 split-K 局部结果
// ---- LFM3 ShortConv 专用
id<MTLBuffer> g_bconv = nil;    // in_proj 的 B/C/x 输出，3*n_embd
id<MTLBuffer> g_bconvh = nil;   // hidden = B⊙X，n_embd
id<MTLBuffer> g_bconvy = nil;   // y = C⊙conv，n_embd
id<MTLBuffer> g_bconvst = nil;  // 各 conv 层的历史缓冲 [n_conv][K-1][n_embd]
id<MTLBuffer> g_bgateatt = nil; // 逐 head 的注意力输出门，n_head

id<MTLBuffer> make_buf(size_t bytes) {
    return [g_dev newBufferWithLength:(bytes ? bytes : 256) options:MTLResourceStorageModeShared];
}

void fwd_release() {
    g_fwd_ready = false;
    g_bx = g_bxb = g_bq = g_bk = g_bv = nil;
    g_batt = g_bgate = g_bup = g_bffn = g_blogits = nil;
    g_bkc = g_bvc = nil;
    g_bfinal = nil;
    g_bxt = nil;
    g_bpart = nil;
    g_bconv = g_bconvh = g_bconvy = g_bconvst = g_bgateatt = nil;
    for (int i = 0; i < F_COUNT; i++) g_fpipe[i] = nil;
    for (int i = 0; i < K_COUNT; i++) g_mpipe[i] = nil;
}

}  // namespace

bool fwd_init(const Spec& s, std::string* err) {
    if (!g_ready) { if (err) *err = "Metal 未初始化"; return false; }
    g_fwd_ready = false;

    // 融合内核不支持的情况直接拒绝，让调用方退回 CPU
    if (s.hd > 128) { if (err) *err = "head_dim > 128，注意力内核未覆盖"; return false; }
    if (s.n_head % s.n_head_kv != 0) { if (err) *err = "n_head 不是 n_head_kv 的整数倍"; return false; }
    if (s.n_conv > 0 && (s.conv_l_cache < 2 || s.conv_l_cache > 32)) {
        if (err) *err = "ShortConv 核宽超出范围";
        return false;
    }
    for (int l = 0; l < s.n_layer; l++) {
        const LayerSpec& L = s.layers[(size_t)l];
        if (L.type[metal::S_ATTN_NORM] != T_F32 ||
            L.type[metal::S_FFN_NORM] != T_F32) {
            if (err) *err = "norm 权重不是 F32，融合内核未覆盖";
            return false;
        }
        // conv 层没有 q/k/v，其余层必须有
        if (L.kind == G_CONV) {
            if (L.ex_off[metal::X_CONV_IN] == ~0ull || L.ex_off[metal::X_CONV_OUT] == ~0ull) {
                if (err) *err = "ShortConv 张量缺失";
                return false;
            }
        }
    }

    for (int i = 0; i < F_COUNT; i++) {
        if (g_fpipe[i]) continue;
        id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kFwdNames[i]]];
        if (!fn) { if (err) *err = std::string("着色器缺少函数 ") + kFwdNames[i]; return false; }
        NSError* pe = nil;
        g_fpipe[i] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!g_fpipe[i]) {
            g_err = pe ? [[pe localizedDescription] UTF8String] : "pipeline 创建失败";
            if (err) *err = std::string("pipeline ") + kFwdNames[i] + " 失败: " + g_err;
            return false;
        }
    }
    for (int i = 0; i < K_COUNT; i++) {
        if (g_mpipe[i] || !kMmNames[i][0]) continue;
        id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kMmNames[i]]];
        if (!fn) { if (err) *err = std::string("着色器缺少函数 ") + kMmNames[i]; return false; }
        NSError* pe = nil;
        g_mpipe[i] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!g_mpipe[i]) {
            g_err = pe ? [[pe localizedDescription] UTF8String] : "pipeline 创建失败";
            if (err) *err = std::string("pipeline ") + kMmNames[i] + " 失败: " + g_err;
            return false;
        }
    }
    for (int i = 0; i < K_COUNT; i++) {
        if (g_m2pipe[i] || !kM2Names[i][0]) continue;
        id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kM2Names[i]]];
        if (!fn) { if (err) *err = std::string("着色器缺少函数 ") + kM2Names[i]; return false; }
        NSError* pe = nil;
        g_m2pipe[i] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!g_m2pipe[i]) {
            g_err = pe ? [[pe localizedDescription] UTF8String] : "pipeline 创建失败";
            if (err) *err = std::string("pipeline ") + kM2Names[i] + " 失败: " + g_err;
            return false;
        }
    }
    // MMA 版：失败不该让整个前向起不来（老 GPU 可能不支持 simdgroup_matrix），
    // 记一条日志然后回退 mm2_* 即可。
    for (int i = 0; i < K_COUNT; i++) {
        if (g_m3pipe[i] || !kM3Names[i][0]) continue;
        id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kM3Names[i]]];
        if (!fn) { std::fprintf(stderr, "  MMA 内核缺失 %s，回退 mm2\n", kM3Names[i]); continue; }
        NSError* pe = nil;
        g_m3pipe[i] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!g_m3pipe[i]) {
            std::fprintf(stderr, "  MMA pipeline %s 失败（%s），回退 mm2\n", kM3Names[i],
                         pe ? [[pe localizedDescription] UTF8String] : "未知");
        }
    }

    g_spec = s;

    const size_t B = (size_t)kBatchMax;
    const size_t n_embd = (size_t)s.n_embd, n_ff = (size_t)s.n_ff, n_vocab = (size_t)s.n_vocab;
    const size_t kv_dim = (size_t)s.n_head_kv * (size_t)s.hd;
    // KV cache 用 fp16。长上下文时 KV 会超过权重的流量（见 k_attn 的注释），
    // 这里减半直接换来一倍的上下文吞吐。
    const size_t kvb = (size_t)s.n_layer * (size_t)s.kv_cap * kv_dim * 2u;

    g_bx      = make_buf(B * n_embd * 4);
    g_bxb     = make_buf(B * n_embd * 4);
    g_bq      = make_buf(B * (size_t)s.n_head * (size_t)s.hd * 4);
    g_bk      = make_buf(B * kv_dim * 4);
    g_bv      = make_buf(B * kv_dim * 4);
    g_batt    = make_buf(B * n_embd * 4);
    g_bgate   = make_buf(B * n_ff * 4);
    g_bup     = make_buf(B * n_ff * 4);
    g_bffn    = make_buf(B * n_ff * 4);
    // logits 按 B 行分配：批量模式下每一行都要出 logits（B 个序列各取各的
    // 下一个 token），单序列解码仍然只用第 0 行。B×n_vocab×4 在 32×128000
    // 时是 16 MB，相对 1 GB 权重可以忽略。
    g_blogits = make_buf(B * n_vocab * 4);
    g_bfinal  = make_buf(B * n_embd * 4);
    g_bxt     = make_buf(B * (size_t)(s.n_ff > s.n_embd ? s.n_ff : s.n_embd) * 4);
    g_bpart   = make_buf(B * (size_t)kAttnSplits * (size_t)s.n_head * ((size_t)s.hd + 2) * 4);
    g_bkc     = make_buf(kvb);
    g_bvc     = make_buf(kvb);

    // LFM3 的 ShortConv 缓冲。llama 系 n_conv == 0，这几个只占 256 字节。
    const size_t nconv = (size_t)std::max(0, s.n_conv);
    const size_t lcache = (size_t)std::max(1, s.conv_l_cache);
    g_bconv    = make_buf(B * 3 * n_embd * 4);
    g_bconvh   = make_buf(B * n_embd * 4);
    g_bconvy   = make_buf(B * n_embd * 4);
    // 卷积滚动状态按 B 份分配：批量模式下每个序列有自己的历史，混用会串台。
    g_bconvst  = make_buf(B * nconv * (lcache > 1 ? lcache - 1 : 1) * n_embd * 4);
    g_bgateatt = make_buf(B * (size_t)s.n_head * 4);

    if (!g_bx || !g_bxb || !g_bq || !g_bk || !g_bv || !g_batt || !g_bgate || !g_bup ||
        !g_bffn || !g_blogits || !g_bfinal || !g_bxt || !g_bpart || !g_bkc || !g_bvc ||
        !g_bconv || !g_bconvh || !g_bconvy || !g_bconvst || !g_bgateatt) {
        if (err) *err = "前向缓冲分配失败";
        return false;
    }
    std::memset([g_bkc contents], 0, kvb);
    std::memset([g_bvc contents], 0, kvb);
    if (g_bconvst) {
        std::memset([g_bconvst contents], 0,
                    nconv * (lcache > 1 ? lcache - 1 : 1) * n_embd * 4);
    }

    g_fwd_ready = true;
    return true;
}

bool fwd_grow_kv(int new_cap, std::string* err) {
    if (!g_fwd_ready) return false;
    if (new_cap <= g_spec.kv_cap) return true;

    const size_t kv_dim = (size_t)g_spec.n_head_kv * (size_t)g_spec.hd;
    const size_t old_row = (size_t)g_spec.kv_cap * kv_dim;
    const size_t new_row = (size_t)new_cap * kv_dim;
    const size_t bytes = (size_t)g_spec.n_layer * new_row * 2u;   // fp16

    id<MTLBuffer> nk = make_buf(bytes), nv = make_buf(bytes);
    if (!nk || !nv) { if (err) *err = "KV cache 扩容分配失败"; return false; }
    std::memset([nk contents], 0, bytes);
    std::memset([nv contents], 0, bytes);

    // 统一内存：直接 CPU memcpy，按层重排（单位是 half，所以指针用 uint16_t）
    uint16_t* dk = (uint16_t*)[nk contents];
    uint16_t* dv = (uint16_t*)[nv contents];
    const uint16_t* sk = (const uint16_t*)[g_bkc contents];
    const uint16_t* sv = (const uint16_t*)[g_bvc contents];
    for (int l = 0; l < g_spec.n_layer; l++) {
        std::memcpy(dk + (size_t)l * new_row, sk + (size_t)l * old_row, old_row * 2u);
        std::memcpy(dv + (size_t)l * new_row, sv + (size_t)l * old_row, old_row * 2u);
    }
    g_bkc = nk;
    g_bvc = nv;
    g_spec.kv_cap = new_cap;
    return true;
}

// 诊断开关：DUBHE_ATTN_WRAP=N 让注意力把回看的 KV 位置按位与回绕到 N 个槽位。
// 指令数、访存次数、dispatch 形状全部不变，只有工作集从几十 MB 缩到几 KB。
// 用来回答一个决定路线的问题：长上下文 decode 是被**带宽**卡住，还是被
// **指令数**卡住？前者值得做 Q8 KV，后者只能上 MMA。结果错误，只用于计时。
static const uint32_t g_attn_wrap = []() -> uint32_t {
    const char* s = getenv("DUBHE_ATTN_WRAP");
    if (!s) return 0u;
    const int v = atoi(s);
    if (v <= 0) return 0u;
    uint32_t n = 1;
    while (n < (uint32_t)v) n <<= 1;
    return n - 1;                       // 传 mask：内核里用 t & kvwrap
}();

// x0: [nrow][n_embd] 已反量化的 embedding；pos0: 首行在序列中的位置。
//
// rpos/rslab/rslot 非空时进入**多序列模式**：第 r 行属于一个独立的序列，
// 有自己的位置（rpos[r]）、自己的 KV 起始槽位（rslab[r]）、自己的 ShortConv
// 历史（rslot[r]）。这时 logits 要按 [nrow][n_vocab] 给出。
//
// 为什么只按行拆「位置相关」的算子、而不是把整个内核改成吃数组：
// 权重 matvec 是唯一值得摊薄的东西（一次读 1.02 GB 权重产出 nrow 个 token），
// 而 RoPE / v 写入 / 注意力都是每行几微秒的小算子。把它们按行各派发一次，
// 用的是现成内核 + 缓冲偏移，**一个着色器都不用改**，单序列路径完全不受影响。
static bool fwd_impl(const float* x0, int nrow, int pos0, uint64_t slab0, uint64_t cslot,
                     const int* rpos, const uint64_t* rslab, const uint64_t* rslot,
                     float* logits, std::string* err) {
    if (!g_fwd_ready) { if (err) *err = "GPU 前向未初始化"; return false; }
    if (nrow <= 0 || nrow > kBatchMax) { if (err) *err = "批大小越界"; return false; }
    const bool multi = (rpos != nullptr && rslab != nullptr);
    if (multi) {
        for (int r = 0; r < nrow; r++)
            if (rpos[r] < 0 || (int64_t)rslab[r] + rpos[r] + 1 > (int64_t)g_spec.kv_cap) {
                if (err) *err = "KV cache 越界";
                return false;
            }
    } else if ((int64_t)slab0 + pos0 + nrow > (int64_t)g_spec.kv_cap) {
        if (err) *err = "KV cache 越界";
        return false;
    }

    const int n_embd = g_spec.n_embd, n_ff = g_spec.n_ff;
    const int n_head = g_spec.n_head, n_kvh = g_spec.n_head_kv, hd = g_spec.hd;
    const int kv_dim = n_kvh * hd;
    const int q_dim = n_head * hd;
    const size_t kv_row = (size_t)g_spec.kv_cap * (size_t)kv_dim;

    // 逐行取值。单序列时退化成原来的 pos0 + row / 槽位 0，代码路径完全一样。
    const auto row_pos  = [&](int r) -> int      { return multi ? rpos[r] : pos0 + r; };
    const auto row_slab = [&](int r) -> uint64_t { return multi ? rslab[r] : slab0; };
    const auto row_slot = [&](int r) -> uint64_t { return (rslot && multi) ? rslot[r] : cslot; };

    std::memcpy([g_bx contents], x0, sizeof(float) * (size_t)n_embd * (size_t)nrow);

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    // 诊断用：把"CPU 编码"和"提交后等 GPU"分开计时。两者之和远大于
    // GPU 忙时时，说明瓶颈在 CPU 侧，改内核没用。
    static const bool want_split = getenv("DUBHE_GPU_SPLIT") != nullptr;
    const auto t_enc0 = std::chrono::steady_clock::now();

    // 诊断：DUBHE_SKIP=<位掩码> 按组跳过，用来给 decode 的时间做归因。
    //
    // 微基准量不出「真实前向里这一组占多少时间」：同一个内核在 6 个 dispatch
    // 的 command buffer 里和在 435 个的里面表现完全不同（并行度、DVFS、L2
    // 复用全变了）。唯一可靠的归因是**在真实前向里把它拿掉**，看时间掉多少。
    //
    //   bit0 = 注意力 q/k/v        bit1 = 注意力 out
    //   bit2 = FFN gate+up         bit3 = FFN down
    //   bit4 = lm_head             bit5 = 全部逐元素算子
    //   bit6 = 注意力内核（含 RoPE / v 写入 / 拆分合并）
    //
    // 拿掉之后输出必然是乱的，只用来计时 —— 而且必须同时设
    // DUBHE_NO_FUSED_CHECK=1，否则融合校验会把整条路径踢回 CPU，
    // 量到的是 CPU 的 10 tok/s（踩过）。
    //
    // 之前试过的另一条路是「插 N 个空 dispatch 看斜率」，结果每个空 dispatch
    // 折合 ~2 ms，明显不合理 —— 那个探针写的是 g_bxb，Metal 的资源依赖追踪
    // 每次都要把整块缓冲的 hazard 处理一遍，量到的是它自己引入的刷新，不是
    // 派发成本。删掉了，别再走这条路。
    static const int kSkip = []() {
        const char* s = getenv("DUBHE_SKIP");
        return s ? atoi(s) : 0;
    }();
    const bool skip_qkv  = (kSkip & 1) != 0;
    const bool skip_attn = (kSkip & 2) != 0;
    const bool skip_ffn1 = (kSkip & 4) != 0;
    const bool skip_ffn2 = (kSkip & 8) != 0;
    const bool skip_lmh  = (kSkip & 16) != 0;
    const bool skip_ew   = (kSkip & 32) != 0;   // 逐元素（norm / add / silu / bias）
    const bool skip_attk = (kSkip & 64) != 0;   // 注意力内核本体

    // 逐元素算子：2D 网格（行内下标 × 行号）。
    // 注意：每个算子必须自己绑全部 buffer —— 早先版本把它们收进一个只
    // setPipeline 的公共 lambda，结果 k_add 读到的是上一次 gemm 留下的
    // 绑定，残差被加进了权重 mmap 里，症状是"输出等于输入 embedding"。
    auto add = [&](id<MTLBuffer> d, id<MTLBuffer> s) {
        if (skip_ew) return;
        uint32_t n = (uint32_t)n_embd;
        [en setComputePipelineState:g_fpipe[F_ADD]];
        [en setBuffer:d offset:0 atIndex:0];
        [en setBuffer:s offset:0 atIndex:1];
        [en setBytes:&n length:4 atIndex:2];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, (NSUInteger)nrow, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto silu_mul = [&]() {
        if (skip_ew) return;
        uint32_t n = (uint32_t)n_ff;
        [en setComputePipelineState:g_fpipe[F_SILU]];
        [en setBuffer:g_bffn offset:0 atIndex:0];
        [en setBuffer:g_bgate offset:0 atIndex:1];
        [en setBuffer:g_bup offset:0 atIndex:2];
        [en setBytes:&n length:4 atIndex:3];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_ff, (NSUInteger)nrow, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    // 矩阵乘：rows 行激活 → rows 行输出。rows == 1 走 matvec，否则走 matmul。
    //
    // 三代 prefill 内核的 token 摊薄倍数不一样：mm3_* 吃 kMmaBm=32 个 token，
    // mm2_* 和 mm_* 吃 kBmax=8 个。所以**先定走哪一代、再按那一代的步长转置**。
    // 顺序不能反 —— 转置写下的步长和内核读的步长必须同一套，否则结果是乱的
    // 而不是报错，这种 bug 最难查。
    auto gemm = [&](uint32_t type, id<MTLBuffer> xin, id<MTLBuffer> yout, uint64_t woff,
                    int n_out, int n_in, int rows) {
        const int k = kind_of(type);
        if (k < 0) return;
        uint32_t nin = (uint32_t)n_in, nout = (uint32_t)n_out, nr = (uint32_t)rows;

        // prefill 走二维分块内核：一个 threadgroup 算 kMmRows 行，
        // 激活切片搬进 threadgroup memory 由组内 8 个 simdgroup 共享。
        // 需要 n_in 是 TILE_K 的整数倍（分块解码依赖完整的 256 元素块）。
        // DUBHE_NO_MM2=1 强制回退到旧内核，用于 A/B 定位。
        static const bool no_mm2 = getenv("DUBHE_NO_MM2") != nullptr;
        const bool tiled = rows > 1 && !no_mm2 && g_m2pipe[k] && (n_in % kMmTileK) == 0;

        // 第 3 代：MMA 版。条件比 mm2 更严 —— 行跨度按 256 元素整块算，
        // 所以 n_in 必须是 256 的整数倍。DUBHE_NO_MMA=1 用于 A/B。
        static const bool no_mma = getenv("DUBHE_NO_MMA") != nullptr;
        const bool mma = rows > 1 && !no_mma && g_m3pipe[k] && (n_in % 256) == 0;

        // 把 xin 里第 j0 行起的 stride 行激活转置进 g_bxt（步长 stride）。
        // 转置本身只有 n_in×stride 个 float，代价可忽略。
        auto transpose = [&](int j0, int stride) {
            uint32_t nn = (uint32_t)n_in, jj = (uint32_t)j0, st = (uint32_t)stride;
            [en setComputePipelineState:g_fpipe[F_TRANS]];
            [en setBuffer:xin offset:0 atIndex:0];
            [en setBuffer:g_bxt offset:0 atIndex:1];
            [en setBytes:&nn length:4 atIndex:2];
            [en setBytes:&nr length:4 atIndex:3];
            [en setBytes:&jj length:4 atIndex:4];
            [en setBytes:&st length:4 atIndex:5];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n_in, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        if (mma) {
            const uint32_t xs = (uint32_t)kMmaBm;
            const uint64_t yo = 0;
            transpose(0, kMmaBm);
            [en setComputePipelineState:g_m3pipe[k]];
            [en setBuffer:g_w offset:0 atIndex:0];
            [en setBuffer:g_bxt offset:0 atIndex:1];
            [en setBuffer:yout offset:0 atIndex:2];
            [en setBytes:&nin length:sizeof(nin) atIndex:3];
            [en setBytes:&woff length:sizeof(woff) atIndex:4];
            [en setBytes:&nout length:sizeof(nout) atIndex:5];
            [en setBytes:&nr length:sizeof(nr) atIndex:6];
            [en setBytes:&xs length:sizeof(xs) atIndex:7];
            [en setBytes:&yo length:sizeof(yo) atIndex:8];
            const NSUInteger tgc = (NSUInteger)((n_out + kMmaBn - 1) / kMmaBn);
            [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }

        if (rows > 1) {
            // mm2_* / mm_* 只吃 kBmax 个 token。rows 超过就按 kBmax 分批派发，
            // 每批的转置和输出行都用 yo 平移 —— 权重只读一遍，成本不变。
            for (int c = 0; c < rows; c += kBmax) {
                const int take = (rows - c < kBmax) ? (rows - c) : kBmax;
                const uint32_t nrc = (uint32_t)take;
                const uint64_t yo = (uint64_t)c * (uint64_t)n_out;
                transpose(c, kBmax);
                [en setComputePipelineState:tiled ? g_m2pipe[k] : g_mpipe[k]];
                [en setBuffer:g_w offset:0 atIndex:0];
                [en setBuffer:g_bxt offset:0 atIndex:1];
                [en setBuffer:yout offset:0 atIndex:2];
                [en setBytes:&nin length:sizeof(nin) atIndex:3];
                [en setBytes:&woff length:sizeof(woff) atIndex:4];
                [en setBytes:&nout length:sizeof(nout) atIndex:5];
                [en setBytes:&nrc length:sizeof(nrc) atIndex:6];
                [en setBytes:&yo length:sizeof(yo) atIndex:7];
                if (tiled) {
                    const NSUInteger tgc = (NSUInteger)((n_out + kMmRows - 1) / kMmRows);
                    [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kMmSgPerTg * 32, 1, 1)];
                } else {
                    const NSUInteger tgc = (NSUInteger)((n_out + kSgPerTg - 1) / kSgPerTg);
                    [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }
            return;
        }

        // 单 token：matvec。一个 simdgroup 算 g_mv_rows 行。
        [en setComputePipelineState:(g_pipe2[k] ? g_pipe2[k] : g_pipe[k])];
        [en setBuffer:g_w offset:0 atIndex:0];
        [en setBuffer:xin offset:0 atIndex:1];
        [en setBuffer:yout offset:0 atIndex:2];
        [en setBytes:&nin length:sizeof(nin) atIndex:3];
        [en setBytes:&woff length:sizeof(woff) atIndex:4];
        [en setBytes:&nout length:sizeof(nout) atIndex:5];
        const int per_tg = kSgPerTg * g_mv_rows;
        const NSUInteger tgc = (NSUInteger)((n_out + per_tg - 1) / per_tg);
        [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };

    auto rmsnorm = [&](id<MTLBuffer> xin, uint64_t woff, id<MTLBuffer> yout, int rows) {
        if (skip_ew) return;
        uint32_t n = (uint32_t)n_embd;
        float eps = g_spec.rms_eps;
        [en setComputePipelineState:g_fpipe[F_RMSNORM]];
        [en setBuffer:xin offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBuffer:yout offset:0 atIndex:2];
        [en setBytes:&n length:4 atIndex:3];
        [en setBytes:&eps length:4 atIndex:4];
        [en setBytes:&woff length:8 atIndex:5];
        [en dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto bias = [&](id<MTLBuffer> y, uint64_t boff, int n) {
        if (skip_ew || boff == ~0ull) return;
        uint32_t nn = (uint32_t)n;
        [en setComputePipelineState:g_fpipe[F_BIAS]];
        [en setBuffer:y offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBytes:&boff length:8 atIndex:2];
        [en setBytes:&nn length:4 atIndex:3];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n, (NSUInteger)nrow, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    // ---- LFM3 专用算子

    // 每 head 的 QK RMSNorm：一个 threadgroup 一个 head。
    // nh 是这张张量的 head 数——q 是 n_head，k 只有 n_head_kv，两者不能混用。
    auto headnorm = [&](id<MTLBuffer> x, uint64_t woff, int nh, int rows) {
        if (skip_ew) return;
        uint32_t nhv = (uint32_t)nh, d = (uint32_t)hd;
        float eps = g_spec.rms_eps;
        [en setComputePipelineState:g_fpipe[F_HEADNORM]];
        [en setBuffer:x offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBuffer:x offset:0 atIndex:2];   // 原地
        [en setBytes:&nhv length:4 atIndex:3];
        [en setBytes:&d length:4 atIndex:4];
        [en setBytes:&eps length:4 atIndex:5];
        [en setBytes:&woff length:8 atIndex:6];
        [en dispatchThreadgroups:MTLSizeMake((NSUInteger)(nh * rows), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };

    // 逐 head 的 sigmoid 输出门
    auto gate_mul = [&](int rows) {
        if (skip_ew) return;
        uint32_t nh = (uint32_t)n_head, d = (uint32_t)hd;
        [en setComputePipelineState:g_fpipe[F_GATE_MUL]];
        [en setBuffer:g_batt offset:0 atIndex:0];
        [en setBuffer:g_bgateatt offset:0 atIndex:1];
        [en setBytes:&nh length:4 atIndex:2];
        [en setBytes:&d length:4 atIndex:3];
        [en dispatchThreads:MTLSizeMake((NSUInteger)q_dim, (NSUInteger)rows, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    // dst[r][i] = a[ao + r*sa + i] * b[bo + r*sb + i]，dst 的行跨度固定是 n
    auto mul = [&](id<MTLBuffer> dst, id<MTLBuffer> a, uint64_t ao, uint64_t sa,
                   id<MTLBuffer> b, uint64_t bo, uint64_t sb, int n) {
        uint32_t nn = (uint32_t)n;
        [en setComputePipelineState:g_fpipe[F_MUL]];
        [en setBuffer:dst offset:0 atIndex:0];
        [en setBuffer:a offset:0 atIndex:1];
        [en setBuffer:b offset:0 atIndex:2];
        [en setBytes:&nn length:4 atIndex:3];
        [en setBytes:&ao length:8 atIndex:4];
        [en setBytes:&bo length:8 atIndex:5];
        [en setBytes:&sa length:8 atIndex:6];
        [en setBytes:&sb length:8 atIndex:7];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n, (NSUInteger)nrow, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    // ShortConv 的卷积 + 历史滚动。
    // 一行一个 dispatch，多行时按行序连续派发——同一 command buffer 内的
    // compute dispatch 顺序执行，所以 state 能逐行正确递推。
    // hbase 是这一批行在 g_bconvh / g_bconvy 里的起始字节偏移。
    // 多序列模式下每行各派发一次（rows=1、hbase=r*n_embd*4），因为每行的
    // 卷积滚动状态槽位不同；单序列时整批一次派发，和原来一致。
    auto conv_step_at = [&](uint64_t slot_off, uint64_t kwoff, uint64_t hbase, int rows) {
        if (skip_ew) return;
        const uint32_t n = (uint32_t)n_embd;
        const uint32_t K = (uint32_t)g_spec.conv_l_cache;
        for (int r = 0; r < rows; r++) {
            const uint64_t hoff = hbase + (uint64_t)r * (uint64_t)n_embd * sizeof(float);
            const uint64_t yoff = hoff;
            [en setComputePipelineState:g_fpipe[F_CONV]];
            [en setBuffer:g_bconvh offset:0 atIndex:0];
            [en setBuffer:g_bconvy offset:0 atIndex:1];
            [en setBuffer:g_w offset:0 atIndex:2];
            [en setBuffer:g_bconvst offset:0 atIndex:3];
            [en setBytes:&n length:4 atIndex:4];
            [en setBytes:&K length:4 atIndex:5];
            [en setBytes:&kwoff length:8 atIndex:6];
            [en setBytes:&slot_off length:8 atIndex:7];
            [en setBytes:&hoff length:8 atIndex:8];
            [en setBytes:&yoff length:8 atIndex:9];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [en setComputePipelineState:g_fpipe[F_CONV_SHIFT]];
            [en setBuffer:g_bconvh offset:0 atIndex:0];
            [en setBuffer:g_bconvst offset:0 atIndex:1];
            [en setBytes:&n length:4 atIndex:2];
            [en setBytes:&K length:4 atIndex:3];
            [en setBytes:&slot_off length:8 atIndex:4];
            [en setBytes:&hoff length:8 atIndex:5];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };
    auto conv_step = [&](uint64_t slot_off, uint64_t kwoff, int rows) {
        conv_step_at(slot_off, kwoff, 0, rows);
    };

    for (int l = 0; l < g_spec.n_layer; l++) {
        const LayerSpec& L = g_spec.layers[(size_t)l];
        const uint64_t coff = (uint64_t)l * kv_row * 2u;                 // fp16

        rmsnorm(g_bx, L.off[metal::S_ATTN_NORM], g_bxb, nrow);

        if (L.kind == G_CONV) {
            // ---- LFM3 ShortConv 子层
            // in_proj 一次算出 [B|C|X]（各 n_embd），hidden = B⊙X，
            // 因果卷积，y = C⊙conv，最后 out_proj 落到 g_bxb（与 o_proj 同位置）。
            const int n3 = 3 * n_embd;
            gemm(L.ex_type[metal::X_CONV_IN], g_bxb, g_bconv,
                 L.ex_off[metal::X_CONV_IN], n3, n_embd, nrow);
            // 每行 3*n_embd：B 在 [0,n)，C 在 [n,2n)，X 在 [2n,3n)
            const uint64_t rowstride = (uint64_t)3 * n_embd * 4;
            mul(g_bconvh, g_bconv, 0, rowstride,
                g_bconv, (uint64_t)2 * n_embd * 4, rowstride, n_embd);
            // 卷积状态槽位：单序列是 [conv_slot][K-1][n_embd]，
            // 多序列是 [conv_slot][B][K-1][n_embd] —— 每个序列一份历史。
            {
                const uint64_t st = (uint64_t)(g_spec.conv_l_cache > 1 ? g_spec.conv_l_cache - 1 : 1) *
                                    (uint64_t)n_embd * 4u;
                if (multi) {
                    for (int r = 0; r < nrow; r++)
                        conv_step_at(((uint64_t)L.conv_slot * (uint64_t)kBatchMax + row_slot(r)) * st,
                                     L.ex_off[metal::X_CONV_K],
                                     (uint64_t)r * (uint64_t)n_embd * 4u, 1);
                } else {
                    conv_step(((uint64_t)L.conv_slot * (uint64_t)kBatchMax + row_slot(0)) * st,
                              L.ex_off[metal::X_CONV_K], nrow);
                }
            }
            mul(g_bconvy, g_bconvy, 0, (uint64_t)n_embd * 4,
                g_bconv, (uint64_t)n_embd * 4, rowstride, n_embd);
            gemm(L.ex_type[metal::X_CONV_OUT], g_bconvy, g_bxb,
                 L.ex_off[metal::X_CONV_OUT], n_embd, n_embd, nrow);
        } else {
            if (!skip_qkv) {
                gemm(L.type[metal::S_Q], g_bxb, g_bq, L.off[metal::S_Q], q_dim,  n_embd, nrow);
                gemm(L.type[metal::S_K], g_bxb, g_bk, L.off[metal::S_K], kv_dim, n_embd, nrow);
                gemm(L.type[metal::S_V], g_bxb, g_bv, L.off[metal::S_V], kv_dim, n_embd, nrow);
            }

            bias(g_bq, L.bias[0], q_dim);
            bias(g_bk, L.bias[1], kv_dim);
            bias(g_bv, L.bias[2], kv_dim);

            // LFM3：QK RMSNorm 在 RoPE 之前
            if (L.ex_off[metal::X_QNORM] != ~0ull)
                headnorm(g_bq, L.ex_off[metal::X_QNORM], n_head, nrow);
            if (L.ex_off[metal::X_KNORM] != ~0ull)
                headnorm(g_bk, L.ex_off[metal::X_KNORM], n_kvh, nrow);

            // RoPE：q 就地，k 旋转后直接落到 cache 的 pos 槽
            //
            // 多序列模式下每行各派发一次：内核用的是标量 pos0 和标量 koff，
            // 行号从 gid.y 取。所以把 q/k 的缓冲按行平移、把 koff 换成该行
            // 序列的槽位、pos0 换成该行的位置，网格的 y 维压到 1 —— 内核
            // 一个字节都不用改。q_dim/kv_dim 都是 2048 字节的倍数，
            // setBuffer 的 256 字节对齐自然满足。
            {
                uint32_t a = (uint32_t)n_head, b = (uint32_t)n_kvh, c = (uint32_t)hd;
                float th = L.rope_theta_l;
                uint32_t ang = (L.rope_angles_l > 0 && L.rope_angles_l < hd / 2)
                                   ? (uint32_t)L.rope_angles_l : (uint32_t)(hd / 2);
                const uint32_t tot = (a + b) * (c / 2);
                const int npass = multi ? nrow : 1;
                for (int r = 0; r < npass; r++) {
                    uint32_t p0 = (uint32_t)row_pos(r);
                    uint64_t ko = coff + (uint64_t)(row_slab(r) + (uint64_t)row_pos(r)) *
                                             (uint64_t)kv_dim * 2u;
                    [en setComputePipelineState:g_fpipe[F_ROPE]];
                    [en setBuffer:g_bq offset:(NSUInteger)((size_t)r * (size_t)q_dim * 4) atIndex:0];
                    [en setBuffer:g_bk offset:(NSUInteger)((size_t)r * (size_t)kv_dim * 4) atIndex:1];
                    [en setBuffer:g_bkc offset:0 atIndex:2];
                    [en setBytes:&ko length:8 atIndex:3];
                [en setBytes:&a length:4 atIndex:4];
                [en setBytes:&b length:4 atIndex:5];
                [en setBytes:&c length:4 atIndex:6];
                [en setBytes:&p0 length:4 atIndex:7];
                [en setBytes:&th length:4 atIndex:8];
                [en setBytes:&ang length:4 atIndex:9];
                    [en dispatchThreads:MTLSizeMake(tot, (NSUInteger)(multi ? 1 : nrow), 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }

            // v → cache（fp16，所以用 k_copy_h 而不是 k_copy）
            {
                uint32_t n = (uint32_t)kv_dim;
                uint64_t zero = 0;
                const int npass = multi ? nrow : 1;
                for (int r = 0; r < npass; r++) {
                    uint64_t vo = coff + (uint64_t)(row_slab(r) + (uint64_t)row_pos(r)) *
                                             (uint64_t)kv_dim * 2u;
                    [en setComputePipelineState:g_fpipe[F_COPY_H]];
                    [en setBuffer:g_bvc offset:0 atIndex:0];
                    [en setBytes:&vo length:8 atIndex:1];
                    [en setBuffer:g_bv offset:(NSUInteger)((size_t)r * (size_t)kv_dim * 4) atIndex:2];
                    [en setBytes:&zero length:8 atIndex:3];
                    [en setBytes:&n length:4 atIndex:4];
                    [en dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, (NSUInteger)(multi ? 1 : nrow), 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }

            // 注意力（行 r 的可见长度 = pos0 + r + 1）
            //
            // split-K：按上下文长度决定切几段。段数太少时只有 n_head 个
            // simdgroup，长上下文会被串行延迟拖死（实测 631 token 时注意力
            // 独占 18 ms/token）。段数随长度增长，直到把 SIMD 单元喂满。
            {
                uint32_t a = (uint32_t)n_head, b = (uint32_t)n_kvh, c = (uint32_t)hd;
                uint32_t nlen0 = (uint32_t)(pos0 + 1);
                uint32_t kvd = (uint32_t)kv_dim;
                float sc = 1.0f / std::sqrt((float)hd);
                // 滑窗层：每行只能看最近 n_swa 个位置
                uint32_t win = 0;
                if (L.kind == G_ATTN_SWA && g_spec.n_swa > 0) win = (uint32_t)g_spec.n_swa;
                // 多序列模式下各行位置不同，切段数按**最长的那一行**定。
                int nlen_max = pos0 + nrow;
                if (multi) {
                    nlen_max = 1;
                    for (int r = 0; r < nrow; r++) nlen_max = std::max(nlen_max, row_pos(r) + 1);
                }
                nlen_max = std::min(nlen_max, win > 0 ? (int)win : nlen_max);
                // 切几段（split-K）。
                //
                // 这里的分母决定了短上下文时的并行度：一个 threadgroup 只
                // 吃 4 个头，n_head=14 时整张网格只有 4 个 threadgroup = 16
                // 个 warp，而 M4 有 10 个 GPU 核 —— 每个核摊不到 2 个 warp，
                // 而内核在位置上是一条**带 simd_sum 屏障的串行链**，延迟完全
                // 盖不住。实测 64 token 上下文时注意力独占 1.13 ms/token
                // （decode 的 13%），全部是并行度不足。
                //
                // 所以分母不能只按"上下文多长"来定，还得按"网格够不够满"。
                //
                // 6 是交替采样量出来的（同一进程外的 12 次对比，div=6 六轮
                // 全胜）：中位 decode 124.7 → 135.7 tok/s，+8.8%。原来的
                // 96 在 64 token 上下文下只切 1 段，等于没切。
                // DUBHE_ATTN_DIV 留了个口子继续扫。
                static const int attn_div = []() {
                    const char* s = getenv("DUBHE_ATTN_DIV");
                    const int v = s ? atoi(s) : 6;
                    return v >= 1 && v <= 512 ? v : 6;
                }();
                uint32_t nsp = (uint32_t)std::min(kAttnSplits,
                                                  std::max(1, nlen_max / attn_div));

                if (skip_attk) {
                    // 诊断开关：跳过注意力本体（保留 RoPE 与 v 写入）
                } else {
                    // 多序列模式：每行各派发一次。行号在内核里取 tgid.y，
                    // 所以把 y 压到 1、把 q/out 按行平移、把 coff 换成该行
                    // 序列的槽位、nlen0 换成该行的位置即可。
                    //
                    // partial 缓冲的行距固定按 kAttnSplits 算（而不是运行时的
                    // nsp）：kAttnSplits*n_head*(hd+2)*4 在 hd=128 时是 532480
                    // 字节 = 256 的整数倍，正好满足 setBuffer 的对齐要求，而
                    // 按 nsp 算会在 nsp=3 这类奇数上踩到非对齐。
                    const size_t part_row = (size_t)kAttnSplits * (size_t)n_head *
                                            ((size_t)hd + 2u) * 4u;
                    const int npass = multi ? nrow : 1;
                    for (int r = 0; r < npass; r++) {
                        const uint64_t co = coff + (uint64_t)row_slab(r) * (uint64_t)kv_dim * 2u;
                        uint32_t nl = multi ? (uint32_t)(row_pos(r) + 1) : nlen0;
                        const NSUInteger qo = (NSUInteger)((size_t)r * (size_t)q_dim * 4);
                        const NSUInteger ao = (NSUInteger)((size_t)r * (size_t)q_dim * 4);
                        const NSUInteger po = (NSUInteger)((size_t)r * part_row);
                        if (nsp == 1) {
                            // 短上下文：融合版一步到位，省掉 partial 缓冲与 combine
                            [en setComputePipelineState:g_fpipe[F_ATTN]];
                            [en setBuffer:g_bq offset:qo atIndex:0];
                            [en setBuffer:g_bkc offset:0 atIndex:1];
                            [en setBuffer:g_bvc offset:0 atIndex:2];
                            [en setBuffer:g_batt offset:ao atIndex:3];
                            [en setBytes:&co length:8 atIndex:4];
                            [en setBytes:&a length:4 atIndex:5];
                            [en setBytes:&b length:4 atIndex:6];
                            [en setBytes:&c length:4 atIndex:7];
                            [en setBytes:&nl length:4 atIndex:8];
                            [en setBytes:&kvd length:4 atIndex:9];
                            [en setBytes:&sc length:4 atIndex:10];
                            [en setBytes:&win length:4 atIndex:11];
                            uint32_t wrapv = g_attn_wrap;
                            [en setBytes:&wrapv length:4 atIndex:12];
                            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                                 (NSUInteger)(multi ? 1 : nrow), 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        } else {
                            [en setComputePipelineState:g_fpipe[F_ATTN_SPLIT]];
                            [en setBuffer:g_bq offset:qo atIndex:0];
                            [en setBuffer:g_bkc offset:0 atIndex:1];
                            [en setBuffer:g_bvc offset:0 atIndex:2];
                            [en setBuffer:g_bpart offset:po atIndex:3];
                            [en setBytes:&co length:8 atIndex:4];
                            [en setBytes:&a length:4 atIndex:5];
                            [en setBytes:&b length:4 atIndex:6];
                            [en setBytes:&c length:4 atIndex:7];
                            [en setBytes:&nl length:4 atIndex:8];
                            [en setBytes:&kvd length:4 atIndex:9];
                            [en setBytes:&sc length:4 atIndex:10];
                            [en setBytes:&nsp length:4 atIndex:11];
                            [en setBytes:&win length:4 atIndex:12];
                            uint32_t wrapv = g_attn_wrap;
                            [en setBytes:&wrapv length:4 atIndex:13];
                            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                                 (NSUInteger)(multi ? 1 : nrow),
                                                                 (NSUInteger)nsp)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                            [en setComputePipelineState:g_fpipe[F_ATTN_COMB]];
                            [en setBuffer:g_bpart offset:po atIndex:0];
                            [en setBuffer:g_batt offset:ao atIndex:1];
                            [en setBytes:&a length:4 atIndex:2];
                            [en setBytes:&c length:4 atIndex:3];
                            [en setBytes:&nsp length:4 atIndex:4];
                            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                                 (NSUInteger)(multi ? 1 : nrow), 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                    }
                }
            }

            // LFM3：逐 head 的 sigmoid 输出门，在 o_proj 之前
            if (L.ex_off[metal::X_AGATE] != ~0ull) {
                gemm(L.ex_type[metal::X_AGATE], g_bxb, g_bgateatt,
                     L.ex_off[metal::X_AGATE], n_head, n_embd, nrow);
                gate_mul(nrow);
            }

            // 输出投影 → g_bxb（与 conv 路径统一落点）
            if (!skip_attn)
                gemm(L.type[metal::S_OUT], g_batt, g_bxb, L.off[metal::S_OUT], n_embd, n_embd, nrow);
        }

        // sandwich norm：子层输出先归一化再加残差
        if (L.ex_off[metal::X_POST_ATTN] != ~0ull)
            rmsnorm(g_bxb, L.ex_off[metal::X_POST_ATTN], g_bxb, nrow);
        add(g_bx, g_bxb);

        // FFN
        rmsnorm(g_bx, L.off[metal::S_FFN_NORM], g_bxb, nrow);
        if (!skip_ffn1) {
            gemm(L.type[metal::S_GATE], g_bxb, g_bgate, L.off[metal::S_GATE], n_ff, n_embd, nrow);
            gemm(L.type[metal::S_UP],   g_bxb, g_bup,   L.off[metal::S_UP],   n_ff, n_embd, nrow);
        }
        silu_mul();
        if (!skip_ffn2)
            gemm(L.type[metal::S_DOWN], g_bffn, g_bxb, L.off[metal::S_DOWN], n_embd, n_ff, nrow);
        if (L.ex_off[metal::X_POST_FFN] != ~0ull)
            rmsnorm(g_bxb, L.ex_off[metal::X_POST_FFN], g_bxb, nrow);
        add(g_bx, g_bxb);
    }

    // 单序列/prefill：只有最后一行需要 logits。先把这一行搬到 xb 的第 0 行，
    // 再 norm + out_w，省掉 nrow-1 次 out_w —— 它是全模型最大的单个矩阵乘
    // （占权重流量 21%）。
    //
    // 多序列：每一行都要 logits（B 个序列各取各的下一个 token），所以
    // 老老实实对 nrow 行做 out_w。这正是批处理赚的那一笔：权重读一遍，
    // 出 B 份 logits。
    const int nlm = multi ? nrow : 1;
    if (!multi) {
        uint32_t n = (uint32_t)n_embd;
        uint64_t doff = 0;
        uint64_t soff = (uint64_t)(nrow - 1) * (uint64_t)n_embd * sizeof(float);
        [en setComputePipelineState:g_fpipe[F_COPY]];
        [en setBuffer:g_bxb offset:0 atIndex:0];
        [en setBytes:&doff length:8 atIndex:1];
        [en setBuffer:g_bx offset:0 atIndex:2];
        [en setBytes:&soff length:8 atIndex:3];
        [en setBytes:&n length:4 atIndex:4];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        rmsnorm(g_bxb, g_spec.off_out_norm, g_bfinal, 1);
    } else {
        rmsnorm(g_bx, g_spec.off_out_norm, g_bfinal, nrow);
    }
    if (!skip_lmh)
        gemm(g_spec.type_out_w, g_bfinal, g_blogits, g_spec.off_out_w,
             g_spec.n_vocab, n_embd, nlm);

    [en endEncoding];
    const auto t_enc1 = std::chrono::steady_clock::now();
    [cb commit];
    [cb waitUntilCompleted];
    if (want_split) {
        const auto t_end = std::chrono::steady_clock::now();
        static double enc = 0, wait = 0;
        static long   cnt = 0;
        enc  += std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
        wait += std::chrono::duration<double, std::milli>(t_end - t_enc1).count();
        if (++cnt % 32 == 0)
            std::fprintf(stderr, "[split] 编码 %.1f ms / 提交后等 %.1f ms（%ld 次调用）\n",
                         enc, wait, cnt);
    }

    // 诊断：DUBHE_GPU_TIME=1 时打印"GPU 真正忙"的累计时间。若它明显小于
    // 墙上时间，说明瓶颈在 CPU 侧编码 / 提交，而不是 GPU 算力。
    static const bool want_time = getenv("DUBHE_GPU_TIME") != nullptr;
    if (want_time && cb.status == MTLCommandBufferStatusCompleted) {
        static double busy = 0;
        static long   ntok = 0;
        static auto   t0   = std::chrono::steady_clock::now();
        busy += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        ntok += nrow;
        if ((ntok % 64) < nrow) {
            const double wall =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[gpu] %ld token, GPU 忙 %.1f ms / 墙 %.1f ms（占 %.0f%%）\n",
                         ntok, busy, wall, wall > 0 ? busy * 100.0 / wall : 0.0);
        }
    }

    if (cb.status == MTLCommandBufferStatusError) {
        g_err = cb.error ? [[cb.error localizedDescription] UTF8String] : "GPU 前向执行失败";
        if (err) *err = g_err;
        g_fwd_ready = false;
        return false;
    }
    std::memcpy(logits, [g_blogits contents],
                sizeof(float) * (size_t)g_spec.n_vocab * (size_t)nlm);
    return true;
}

bool fwd_batch(const float* x0, int nrow, int pos0, float* logits, std::string* err) {
    return fwd_impl(x0, nrow, pos0, 0, 0, nullptr, nullptr, nullptr, logits, err);
}

bool fwd_batch_at(const float* x0, int nrow, int pos0, uint64_t slab0, uint64_t cslot,
                  float* logits, std::string* err) {
    return fwd_impl(x0, nrow, pos0, slab0, cslot, nullptr, nullptr, nullptr, logits, err);
}

bool fwd_multi(const float* x0, int nrow, const int* rpos, const uint64_t* rslab,
               const uint64_t* rslot, float* logits, std::string* err) {
    if (!rpos || !rslab) { if (err) *err = "多序列前向缺少 rpos/rslab"; return false; }
    return fwd_impl(x0, nrow, 0, 0, 0, rpos, rslab, rslot, logits, err);
}

bool fwd_token(const float* x0, int pos, float* logits, std::string* err) {
    return fwd_batch(x0, 1, pos, logits, err);
}

int fwd_batch_max() { return kBatchMax; }

bool fwd_copy_x(float* dst, int n) {
    if (!g_fwd_ready || !g_bx) return false;
    const int m = std::min(n, g_spec.n_embd);
    std::memcpy(dst, [g_bx contents], sizeof(float) * (size_t)m);
    return true;
}

void fwd_reset_conv() {
    if (!g_fwd_ready || !g_bconvst || g_spec.n_conv <= 0) return;
    const size_t lcache = (size_t)std::max(1, g_spec.conv_l_cache);
    // 必须清**整块**缓冲：多序列模式下槽位下标是 conv_slot*kBatchMax + 序列号，
    // 会落到 n_conv*(lcache-1)*n_embd 之外。只清前一段的话，被复用的槽位里
    // 留着上一条序列的卷积历史，输出看着通顺但内容串台 —— 融合校验能抓到，
    // 但它报的是"rel=0.66"，看不出根因在这儿。
    const size_t bytes = (size_t)kBatchMax * (size_t)g_spec.n_conv *
                         (lcache > 1 ? lcache - 1 : 1) * (size_t)g_spec.n_embd * sizeof(float);
    std::memset([g_bconvst contents], 0, bytes);
}

// ---------------------------------------------------------------- Qwen3.5
//
// 每 token 一次提交、一次同步。32 层全部编进同一个 command buffer。
// 逐算子提交实测 4.0 tok/s（248 次同步），所以这里的全部意义就是"一次"。

namespace {

enum Q35Kernel {
    Q_PREP = 0, Q_CONV, Q_L2NORM, Q_DELTA, Q_GRMS, Q_ROPE, Q_SPLIT, Q_GATEDIM, Q_COUNT
};
const char* const kQ35Names[Q_COUNT] = {
    "k_ssm_prep", "k_ssm_conv", "k_l2norm_heads", "k_delta_rule",
    "k_gated_rmsnorm", "k_rope_mrope", "k_split_qg", "k_gate_mul_dim"
};

Q35Spec g_q35;
bool g_q35_ready = false;
id<MTLComputePipelineState> g_q35pipe[Q_COUNT] = {nil};

id<MTLBuffer> g_q35_x = nil, g_q35_xb = nil, g_q35_qkv = nil, g_q35_y = nil;
id<MTLBuffer> g_q35_qk = nil, g_q35_z = nil, g_q35_beta = nil, g_q35_alpha = nil;
id<MTLBuffer> g_q35_o = nil, g_q35_att = nil, g_q35_qg = nil, g_q35_q = nil;
id<MTLBuffer> g_q35_gate = nil, g_q35_k = nil, g_q35_v = nil;
id<MTLBuffer> g_q35_ffn = nil, g_q35_up = nil, g_q35_ffn2 = nil, g_q35_down = nil;
id<MTLBuffer> g_q35_logits = nil;
id<MTLBuffer> g_q35_state = nil, g_q35_conv = nil, g_q35_kc = nil, g_q35_vc = nil;

void q35_release() {
    g_q35_ready = false;
    g_q35_x = g_q35_xb = g_q35_qkv = g_q35_y = nil;
    g_q35_qk = g_q35_z = g_q35_beta = g_q35_alpha = nil;
    g_q35_o = g_q35_att = g_q35_qg = g_q35_q = nil;
    g_q35_gate = g_q35_k = g_q35_v = nil;
    g_q35_ffn = g_q35_up = g_q35_ffn2 = g_q35_down = nil;
    g_q35_logits = nil;
    g_q35_state = g_q35_conv = g_q35_kc = g_q35_vc = nil;
    for (int i = 0; i < Q_COUNT; i++) g_q35pipe[i] = nil;
}

}  // namespace

bool q35_init(const Q35Spec& s, std::string* err) {
    if (!g_ready) { if (err) *err = "Metal 未初始化"; return false; }
    q35_release();
    g_q35 = s;

    for (int i = 0; i < Q_COUNT; i++) {
        id<MTLFunction> fn = [g_lib newFunctionWithName:
                              [NSString stringWithUTF8String:kQ35Names[i]]];
        if (!fn) { if (err) *err = std::string("着色器缺少 ") + kQ35Names[i]; return false; }
        NSError* pe = nil;
        g_q35pipe[i] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!g_q35pipe[i]) {
            if (err) *err = std::string("pipeline 失败: ") + kQ35Names[i];
            return false;
        }
    }

    // q35 复用的通用内核（rmsnorm / add / silu / k→f16 KV / headnorm / attn）。
    // 这些平时由 fwd_init 创建；q35 走自己的初始化路径，必须在这里补上，
    // 否则 setComputePipelineState:nil 会直接段错误（踩过）。
    {
        const std::pair<int, const char*> common[] = {
            {F_RMSNORM, "k_rmsnorm"}, {F_ADD, "k_add"}, {F_SILU, "k_silu_mul"},
            {F_COPY_H, "k_copy_h"}, {F_HEADNORM, "k_headnorm"}, {F_ATTN, "k_attn"},
        };
        for (auto& p : common) {
            id<MTLFunction> fn = [g_lib newFunctionWithName:
                                  [NSString stringWithUTF8String:p.second]];
            if (!fn) { if (err) *err = std::string("着色器缺少 ") + p.second; return false; }
            NSError* pe = nil;
            g_fpipe[p.first] = [g_dev newComputePipelineStateWithFunction:fn error:&pe];
            if (!g_fpipe[p.first]) {
                if (err) *err = std::string("pipeline 失败: ") + p.second;
                return false;
            }
        }
    }

    const size_t F = sizeof(float);
    const int cdim = 2 * s.ssm_kdim + s.ssm_vdim;
    const int q_dim = s.n_head * s.hd;
    const int kv_dim = s.n_head_kv * s.hd;
    const int gate_sz = std::max(s.n_ff, q_dim);

    g_q35_x      = make_buf((size_t)s.n_embd * F);
    g_q35_xb     = make_buf((size_t)s.n_embd * F);
    g_q35_qkv    = make_buf((size_t)cdim * F);
    g_q35_y      = make_buf((size_t)cdim * F);
    g_q35_qk     = make_buf((size_t)s.ssm_kdim * 2 * F);
    g_q35_z      = make_buf((size_t)s.ssm_vdim * F);
    g_q35_beta   = make_buf((size_t)s.ssm_nv * F);
    g_q35_alpha  = make_buf((size_t)s.ssm_nv * F);
    g_q35_o      = make_buf((size_t)s.ssm_vdim * F);
    g_q35_att    = make_buf((size_t)s.n_embd * F);
    g_q35_qg     = make_buf((size_t)2 * q_dim * F);
    g_q35_q      = make_buf((size_t)q_dim * F);
    g_q35_gate   = make_buf((size_t)gate_sz * F);
    g_q35_k      = make_buf((size_t)kv_dim * F);
    g_q35_v      = make_buf((size_t)kv_dim * F);
    g_q35_ffn    = make_buf((size_t)s.n_ff * F);
    g_q35_up     = make_buf((size_t)s.n_ff * F);
    g_q35_ffn2   = make_buf((size_t)s.n_ff * F);
    g_q35_down   = make_buf((size_t)s.n_embd * F);
    g_q35_logits = make_buf((size_t)s.n_vocab * F);
    g_q35_state  = make_buf((size_t)s.n_delta * s.ssm_nv * s.ssm_S * s.ssm_S * F);
    g_q35_conv   = make_buf((size_t)s.n_delta * (size_t)(s.ssm_conv - 1) * cdim * F);
    const size_t kvb = (size_t)s.n_kv_layer * (size_t)s.kv_cap * (size_t)kv_dim * 2;
    g_q35_kc = make_buf(kvb);
    g_q35_vc = make_buf(kvb);

    if (!g_q35_x || !g_q35_xb || !g_q35_qkv || !g_q35_y || !g_q35_qk || !g_q35_z ||
        !g_q35_beta || !g_q35_alpha || !g_q35_o || !g_q35_att || !g_q35_qg ||
        !g_q35_q || !g_q35_gate || !g_q35_k || !g_q35_v || !g_q35_ffn ||
        !g_q35_up || !g_q35_ffn2 || !g_q35_down || !g_q35_logits ||
        !g_q35_state || !g_q35_conv || !g_q35_kc || !g_q35_vc) {
        if (err) *err = "Qwen3.5 缓冲分配失败（内存不足）";
        return false;
    }
    std::memset([g_q35_state contents], 0, [g_q35_state length]);
    std::memset([g_q35_conv contents], 0, [g_q35_conv length]);
    std::memset([g_q35_kc contents], 0, [g_q35_kc length]);
    std::memset([g_q35_vc contents], 0, [g_q35_vc length]);

    g_q35_ready = true;
    return true;
}

bool q35_token(const float* x0, int pos, float* logits, std::string* err) {
    if (std::getenv("DUBHE_Q35_DEBUG"))
        std::fprintf(stderr, "[q35] enter pos=%d ready=%d cap=%d\n",
                     pos, (int)g_q35_ready, g_q35.kv_cap);
    if (!g_q35_ready) { if (err) *err = "Qwen3.5 Metal 未初始化"; return false; }
    const Q35Spec& s = g_q35;
    if (pos < 0 || pos >= s.kv_cap) { if (err) *err = "Qwen3.5 上下文已满"; return false; }

    const int n_embd = s.n_embd, n_ff = s.n_ff;
    const int hd = s.hd, nh = s.n_head, nkv = s.n_head_kv;
    const int kv_dim = nkv * hd, q_dim = nh * hd;
    const int S = s.ssm_S, nk = s.ssm_nk, nv = s.ssm_nv;
    const int kdim = s.ssm_kdim, vdim = s.ssm_vdim;
    const int cdim = 2 * kdim + vdim;
    const int ck = s.ssm_conv;

    std::memcpy([g_q35_x contents], x0, sizeof(float) * (size_t)n_embd);

    const bool g_dump = std::getenv("DUBHE_Q35_DUMP") != nullptr;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];

    // 单 token matvec：和 fwd_impl 的 gemm(rows=1) 同一条路径
    auto mv = [&](uint32_t type, id<MTLBuffer> xin, id<MTLBuffer> yout,
                  uint64_t woff, int n_out, int n_in) {
        const int k = kind_of(type);
        if (k < 0) return;
        uint32_t nin = (uint32_t)n_in, nout = (uint32_t)n_out;
        [en setComputePipelineState:(g_pipe2[k] ? g_pipe2[k] : g_pipe[k])];
        [en setBuffer:g_w offset:0 atIndex:0];
        [en setBuffer:xin offset:0 atIndex:1];
        [en setBuffer:yout offset:0 atIndex:2];
        [en setBytes:&nin length:4 atIndex:3];
        [en setBytes:&woff length:8 atIndex:4];
        [en setBytes:&nout length:4 atIndex:5];
        const NSUInteger per_tg = (NSUInteger)(kSgPerTg * g_mv_rows);
        const NSUInteger tgc = (NSUInteger)((n_out + (int)per_tg - 1) / (int)per_tg);
        [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    auto rms = [&](id<MTLBuffer> xin, uint64_t woff, id<MTLBuffer> yout) {
        uint32_t n = (uint32_t)n_embd;
        float eps = s.rms_eps;
        [en setComputePipelineState:g_fpipe[F_RMSNORM]];
        [en setBuffer:xin offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBuffer:yout offset:0 atIndex:2];
        [en setBytes:&n length:4 atIndex:3];
        [en setBytes:&eps length:4 atIndex:4];
        [en setBytes:&woff length:8 atIndex:5];
        [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto hnorm = [&](id<MTLBuffer> x, uint64_t woff, int heads) {
        uint32_t nhv = (uint32_t)heads, d = (uint32_t)hd;
        float eps = s.rms_eps;
        [en setComputePipelineState:g_fpipe[F_HEADNORM]];
        [en setBuffer:x offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBuffer:x offset:0 atIndex:2];
        [en setBytes:&nhv length:4 atIndex:3];
        [en setBytes:&d length:4 atIndex:4];
        [en setBytes:&eps length:4 atIndex:5];
        [en setBytes:&woff length:8 atIndex:6];
        [en dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    auto add = [&](id<MTLBuffer> d, id<MTLBuffer> src) {
        uint32_t n = (uint32_t)n_embd;
        [en setComputePipelineState:g_fpipe[F_ADD]];
        [en setBuffer:d offset:0 atIndex:0];
        [en setBuffer:src offset:0 atIndex:1];
        [en setBytes:&n length:4 atIndex:2];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    // 子步骤指纹（要和 CPU 的 [cpu-attn] 对拍）。每调用一次就同步一次，
    // 只有 DUBHE_Q35_DUMP 打开且落在第 3 层时才走。
    auto sdump = [&](int l, int pos_, const char* tag, id<MTLBuffer> b, int n) {
        if (!g_dump || l != 3) return;
        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* p = (const float*)[b contents];
        float mx = 0.f;
        int bad = 0;
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(p[i])) bad++;
            else mx = std::max(mx, std::fabs(p[i]));
        }
        std::fprintf(stderr, "[gpu-attn] L%02d %s=%.5f bad=%d\n", l, tag, mx, bad);
        cb = [g_queue commandBuffer];
        en = [cb computeCommandEncoder];
    };

    for (int l = 0; l < s.n_layer; l++) {
        const Q35Layer& L = s.layers[(size_t)l];
        if (std::getenv("DUBHE_Q35_DEBUG") && l < 3)
            std::fprintf(stderr, "[q35] layer %d kind=%d slot=%d kv=%d\n",
                         l, L.kind, L.ssm_slot, L.kv_slot);
        rms(g_q35_x, L.off_attn_norm, g_q35_xb);

        if (L.kind == 0) {
            // ---- Gated DeltaNet
            auto step = [&](const char* t) {
                if (std::getenv("DUBHE_Q35_DEBUG") && l == 0 && pos == 0)
                    std::fprintf(stderr, "[q35]   %s\n", t);
            };
            step("mv qkv");
            mv(L.ty_qkv,   g_q35_xb, g_q35_qkv,   L.off_qkv,   cdim, n_embd);
            step("mv z");
            mv(L.ty_z,     g_q35_xb, g_q35_z,     L.off_z,     vdim, n_embd);
            step("mv beta");
            mv(L.ty_beta,  g_q35_xb, g_q35_beta,  L.off_beta,  nv,   n_embd);
            step("mv alpha");
            mv(L.ty_alpha, g_q35_xb, g_q35_alpha, L.off_alpha, nv,   n_embd);
            step("prep");

            {
                uint32_t nvv = (uint32_t)nv;
                [en setComputePipelineState:g_q35pipe[Q_PREP]];
                [en setBuffer:g_q35_beta offset:0 atIndex:0];
                [en setBuffer:g_q35_alpha offset:0 atIndex:1];
                [en setBuffer:g_w offset:0 atIndex:2];
                [en setBytes:&L.off_dt length:8 atIndex:3];
                [en setBytes:&L.off_a length:8 atIndex:4];
                [en setBytes:&nvv length:4 atIndex:5];
                [en dispatchThreads:MTLSizeMake((NSUInteger)nv, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            step("prep done");
            {
                uint64_t soff = (uint64_t)L.ssm_slot * (uint64_t)(ck - 1) * (uint64_t)cdim * 4u;
                uint64_t koff = L.off_conv;
                uint32_t cd = (uint32_t)cdim, kk = (uint32_t)ck;
                [en setComputePipelineState:g_q35pipe[Q_CONV]];
                [en setBuffer:g_q35_conv offset:(NSUInteger)soff atIndex:0];
                [en setBuffer:g_q35_qkv offset:0 atIndex:1];
                [en setBuffer:g_w offset:0 atIndex:2];
                [en setBytes:&koff length:8 atIndex:3];
                [en setBytes:&cd length:4 atIndex:4];
                [en setBytes:&kk length:4 atIndex:5];
                [en setBuffer:g_q35_y offset:0 atIndex:6];
                [en dispatchThreads:MTLSizeMake((NSUInteger)cdim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            step("conv done");
            {
                uint32_t ss = (uint32_t)S, heads = (uint32_t)(nk * 2);
                float eps = s.rms_eps;
                [en setComputePipelineState:g_q35pipe[Q_L2NORM]];
                [en setBuffer:g_q35_y offset:0 atIndex:0];
                [en setBuffer:g_q35_qk offset:0 atIndex:1];
                [en setBytes:&ss length:4 atIndex:2];
                [en setBytes:&eps length:4 atIndex:3];
                [en dispatchThreads:MTLSizeMake((NSUInteger)heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            step("l2norm done");
            {
                uint64_t soff = (uint64_t)L.ssm_slot * (uint64_t)nv * S * S * 4u;
                uint32_t ss = (uint32_t)S, nkk = (uint32_t)nk;
                [en setComputePipelineState:g_q35pipe[Q_DELTA]];
                [en setBuffer:g_q35_state offset:(NSUInteger)soff atIndex:0];
                [en setBuffer:g_q35_qk offset:0 atIndex:1];
                [en setBuffer:g_q35_qk offset:(NSUInteger)((size_t)kdim * 4) atIndex:2];
                [en setBuffer:g_q35_y offset:(NSUInteger)((size_t)2 * kdim * 4) atIndex:3];
                [en setBuffer:g_q35_alpha offset:0 atIndex:4];
                [en setBuffer:g_q35_beta offset:0 atIndex:5];
                [en setBuffer:g_q35_o offset:0 atIndex:6];
                [en setBytes:&ss length:4 atIndex:7];
                [en setBytes:&nkk length:4 atIndex:8];
                [en dispatchThreads:MTLSizeMake((NSUInteger)S, (NSUInteger)nv, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            step("delta done");
            {
                uint32_t ss = (uint32_t)S;
                float eps = s.rms_eps;
                [en setComputePipelineState:g_q35pipe[Q_GRMS]];
                [en setBuffer:g_q35_o offset:0 atIndex:0];
                [en setBuffer:g_q35_z offset:0 atIndex:1];
                [en setBuffer:g_w offset:0 atIndex:2];
                [en setBytes:&L.off_norm length:8 atIndex:3];
                [en setBytes:&ss length:4 atIndex:4];
                [en setBytes:&eps length:4 atIndex:5];
                [en dispatchThreads:MTLSizeMake((NSUInteger)nv, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            step("grms done");
            mv(L.ty_sout, g_q35_o, g_q35_att, L.off_sout, n_embd, vdim);
            step("sout done");
        } else {
            // ---- gated full attention
            mv(L.ty_q, g_q35_xb, g_q35_qg, L.off_q, 2 * q_dim, n_embd);
            mv(L.ty_k, g_q35_xb, g_q35_k,  L.off_k, kv_dim,   n_embd);
            mv(L.ty_v, g_q35_xb, g_q35_v,  L.off_v, kv_dim,   n_embd);
            {
                uint32_t d = (uint32_t)hd;
                [en setComputePipelineState:g_q35pipe[Q_SPLIT]];
                [en setBuffer:g_q35_qg offset:0 atIndex:0];
                [en setBuffer:g_q35_q offset:0 atIndex:1];
                [en setBuffer:g_q35_gate offset:0 atIndex:2];
                [en setBytes:&d length:4 atIndex:3];
                [en dispatchThreads:MTLSizeMake((NSUInteger)hd, (NSUInteger)nh, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            hnorm(g_q35_q, L.off_qn, nh);
            hnorm(g_q35_k, L.off_kn, nkv);

            const uint64_t coff = (uint64_t)L.kv_slot * (uint64_t)s.kv_cap *
                                  (uint64_t)kv_dim * 2u;
            const uint64_t koff = coff + (uint64_t)pos * (uint64_t)kv_dim * 2u;
            {
                uint32_t a = (uint32_t)nh, b = (uint32_t)nkv, c = (uint32_t)hd;
                uint32_t p0 = (uint32_t)pos;
                float th = s.rope_theta;
                uint32_t ang = (uint32_t)s.rope_angles;
                uint32_t nrot = (uint32_t)(s.rope_angles * 2);
                uint32_t s0 = (uint32_t)s.rope_sections[0], s1 = (uint32_t)s.rope_sections[1];
                uint32_t s2 = (uint32_t)s.rope_sections[2], s3 = (uint32_t)s.rope_sections[3];
                if (std::getenv("DUBHE_NO_MROPE")) s0 = s1 = s2 = s3 = 0;
                const NSUInteger tot = (NSUInteger)((nh + nkv) * (s.rope_angles));
                [en setComputePipelineState:g_q35pipe[Q_ROPE]];
                [en setBuffer:g_q35_q offset:0 atIndex:0];
                [en setBuffer:g_q35_k offset:0 atIndex:1];
                [en setBuffer:g_q35_kc offset:0 atIndex:2];
                [en setBytes:&koff length:8 atIndex:3];
                [en setBytes:&a length:4 atIndex:4];
                [en setBytes:&b length:4 atIndex:5];
                [en setBytes:&c length:4 atIndex:6];
                [en setBytes:&p0 length:4 atIndex:7];
                [en setBytes:&th length:4 atIndex:8];
                [en setBytes:&ang length:4 atIndex:9];
                [en setBytes:&nrot length:4 atIndex:10];
                [en setBytes:&s0 length:4 atIndex:11];
                [en setBytes:&s1 length:4 atIndex:12];
                [en setBytes:&s2 length:4 atIndex:13];
                [en setBytes:&s3 length:4 atIndex:14];
                [en dispatchThreads:MTLSizeMake(tot, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            sdump(l, pos, "q", g_q35_q, q_dim);
            {
                uint64_t voff = koff, zero = 0;
                uint32_t n = (uint32_t)kv_dim;
                [en setComputePipelineState:g_fpipe[F_COPY_H]];
                [en setBuffer:g_q35_vc offset:0 atIndex:0];
                [en setBytes:&voff length:8 atIndex:1];
                [en setBuffer:g_q35_v offset:0 atIndex:2];
                [en setBytes:&zero length:8 atIndex:3];
                [en setBytes:&n length:4 atIndex:4];
                [en dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            {
                uint32_t a = (uint32_t)nh, b = (uint32_t)nkv, c = (uint32_t)hd;
                uint32_t nlen = (uint32_t)(pos + 1);
                uint32_t kvd = (uint32_t)kv_dim;
                float sc = 1.0f / std::sqrt((float)hd);
                uint32_t win = 0, wrap = 0;
                [en setComputePipelineState:g_fpipe[F_ATTN]];
                [en setBuffer:g_q35_q offset:0 atIndex:0];
                [en setBuffer:g_q35_kc offset:0 atIndex:1];
                [en setBuffer:g_q35_vc offset:0 atIndex:2];
                [en setBuffer:g_q35_att offset:0 atIndex:3];
                [en setBytes:&coff length:8 atIndex:4];
                [en setBytes:&a length:4 atIndex:5];
                [en setBytes:&b length:4 atIndex:6];
                [en setBytes:&c length:4 atIndex:7];
                [en setBytes:&nlen length:4 atIndex:8];
                [en setBytes:&kvd length:4 atIndex:9];
                [en setBytes:&sc length:4 atIndex:10];
                [en setBytes:&win length:4 atIndex:11];
                [en setBytes:&wrap length:4 atIndex:12];
                [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((nh + 3) / 4), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            sdump(l, pos, "attn", g_q35_att, q_dim);
            {
                uint32_t n = (uint32_t)q_dim;
                [en setComputePipelineState:g_q35pipe[Q_GATEDIM]];
                [en setBuffer:g_q35_att offset:0 atIndex:0];
                [en setBuffer:g_q35_gate offset:0 atIndex:1];
                [en setBytes:&n length:4 atIndex:2];
                [en dispatchThreads:MTLSizeMake((NSUInteger)q_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            sdump(l, pos, "gated", g_q35_att, q_dim);
            mv(L.ty_o, g_q35_att, g_q35_down, L.off_o, n_embd, q_dim);
            sdump(l, pos, "oproj", g_q35_down, n_embd);
        }

        add(g_q35_x, L.kind == 0 ? g_q35_att : g_q35_down);

        // ---- FFN（SwiGLU）
        rms(g_q35_x, L.off_ffn_norm, g_q35_xb);
        mv(L.ty_gate, g_q35_xb, g_q35_up,   L.off_gate, n_ff, n_embd);
        mv(L.ty_up,   g_q35_xb, g_q35_ffn,  L.off_up,   n_ff, n_embd);
        {
            uint32_t n = (uint32_t)n_ff;
            [en setComputePipelineState:g_fpipe[F_SILU]];
            [en setBuffer:g_q35_ffn2 offset:0 atIndex:0];
            [en setBuffer:g_q35_up offset:0 atIndex:1];
            [en setBuffer:g_q35_ffn offset:0 atIndex:2];
            [en setBytes:&n length:4 atIndex:3];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n_ff, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        mv(L.ty_down, g_q35_ffn2, g_q35_down, L.off_down, n_embd, n_ff);
        add(g_q35_x, g_q35_down);

        // 逐层指纹：和 CPU 的 `[dbg] L%02d ... rms=` 对拍用。每层一次同步，
        // 只用于定位分叉点，正常路径不走这里。
        if (g_dump) {
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const float* xp = (const float*)[g_q35_x contents];
            double ss = 0.0;
            float mx = 0.f;
            for (int i = 0; i < n_embd; i++) {
                const float v = xp[i];
                ss += (double)v * v;
                mx = std::max(mx, std::fabs(v));
            }
            std::fprintf(stderr, "[gpu] L%02d kind=%d rms=%.4f max=%.3f\n",
                         l, L.kind, std::sqrt(ss / n_embd), mx);
            cb = [g_queue commandBuffer];
            en = [cb computeCommandEncoder];
        }
    }

    rms(g_q35_x, s.off_out_norm, g_q35_xb);
    mv(s.ty_out_w, g_q35_xb, g_q35_logits, s.off_out_w, s.n_vocab, n_embd);

    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        if (err) *err = cb.error ? [[cb.error localizedDescription] UTF8String]
                                 : "Qwen3.5 命令缓冲执行失败";
        return false;
    }
    std::memcpy(logits, [g_q35_logits contents], sizeof(float) * (size_t)s.n_vocab);
    if (std::getenv("DUBHE_Q35_DEBUG")) {
        const float* lg = (const float*)logits;
        int am = 0; float mx = -1e30f; int nbad = 0;
        for (int i = 0; i < s.n_vocab; i++) {
            if (!std::isfinite(lg[i])) { nbad++; continue; }
            if (lg[i] > mx) { mx = lg[i]; am = i; }
        }
        std::fprintf(stderr, "[q35] pos=%d argmax=%d max=%.4f bad=%d\n", pos, am, mx, nbad);
    }
    return true;
}

void q35_reset() {
    if (!g_q35_ready) return;
    std::memset([g_q35_state contents], 0, [g_q35_state length]);
    std::memset([g_q35_conv contents], 0, [g_q35_conv length]);
    std::memset([g_q35_kc contents], 0, [g_q35_kc length]);
    std::memset([g_q35_vc contents], 0, [g_q35_vc length]);
}

}  // namespace metal
}  // namespace dubhe
