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
    K_F32 = 0, K_F16, K_Q4_0, K_Q5_0, K_Q8_0, K_Q4_K, K_Q6_K, K_COUNT
};

const char* const kNames[K_COUNT] = {
    "mv_f32", "mv_f16", "mv_q4_0", "mv_q5_0", "mv_q8_0", "mv_q4_k", "mv_q6_k"
};

// 分块 matmul 的函数名。Q4_0/Q5_0 没有分块版本（空串），派发时回退到 mm_*。
const char* const kM2Names[K_COUNT] = {
    "mm2_f32", "mm2_f16", "", "", "mm2_q8_0", "mm2_q4_k", "mm2_q6_k"
};

// MMA 版分块 matmul（第 3 代）。覆盖 F16 / Q4_K / Q6_K，其余回退 mm2_*。
// 空串表示"没有这一档"，派发时跳过。
const char* const kM3Names[K_COUNT] = {
    "", "mm3_f16", "", "", "mm3_q8_0", "mm3_q4_k", "mm3_q6_k"
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

int kind_of(uint32_t t) {
    switch (t) {
        case T_F32:  return K_F32;
        case T_F16:  return K_F16;
        case T_Q4_0: return K_Q4_0;
        case T_Q5_0: return K_Q5_0;
        case T_Q8_0: return K_Q8_0;
        case T_Q4_K: return K_Q4_K;
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
    for (int i = 0; i < K_COUNT; i++) g_pipe[i] = nil;
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
    const size_t page = 4096;
    if (((uintptr_t)base & (page - 1)) != 0) {
        if (err) *err = "mmap 基址未页对齐，无法零拷贝";
        return false;
    }
    const size_t len = size & ~(page - 1);
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
    [enc setComputePipelineState:g_pipe[k]];
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
    "mm_f32", "mm_f16", "mm_q4_0", "mm_q5_0", "mm_q8_0", "mm_q4_k", "mm_q6_k"
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
        if (g_mpipe[i]) continue;
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
    g_blogits = make_buf(n_vocab * 4);
    g_bfinal  = make_buf(n_embd * 4);
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
    g_bconvst  = make_buf(nconv * (lcache > 1 ? lcache - 1 : 1) * n_embd * 4);
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

// x0: [nrow][n_embd] 已反量化的 embedding；pos0: 首行在序列中的位置。
bool fwd_batch(const float* x0, int nrow, int pos0, float* logits, std::string* err) {
    if (!g_fwd_ready) { if (err) *err = "GPU 前向未初始化"; return false; }
    if (nrow <= 0 || nrow > kBatchMax) { if (err) *err = "批大小越界"; return false; }
    if (pos0 + nrow > g_spec.kv_cap) { if (err) *err = "KV cache 越界"; return false; }

    const int n_embd = g_spec.n_embd, n_ff = g_spec.n_ff;
    const int n_head = g_spec.n_head, n_kvh = g_spec.n_head_kv, hd = g_spec.hd;
    const int kv_dim = n_kvh * hd;
    const int q_dim = n_head * hd;
    const size_t kv_row = (size_t)g_spec.kv_cap * (size_t)kv_dim;

    std::memcpy([g_bx contents], x0, sizeof(float) * (size_t)n_embd * (size_t)nrow);

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    // 诊断用：把"CPU 编码"和"提交后等 GPU"分开计时。两者之和远大于
    // GPU 忙时时，说明瓶颈在 CPU 侧，改内核没用。
    static const bool want_split = getenv("DUBHE_GPU_SPLIT") != nullptr;
    const auto t_enc0 = std::chrono::steady_clock::now();

    // 逐元素算子：2D 网格（行内下标 × 行号）。
    // 注意：每个算子必须自己绑全部 buffer —— 早先版本把它们收进一个只
    // setPipeline 的公共 lambda，结果 k_add 读到的是上一次 gemm 留下的
    // 绑定，残差被加进了权重 mmap 里，症状是"输出等于输入 embedding"。
    auto add = [&](id<MTLBuffer> d, id<MTLBuffer> s) {
        uint32_t n = (uint32_t)n_embd;
        [en setComputePipelineState:g_fpipe[F_ADD]];
        [en setBuffer:d offset:0 atIndex:0];
        [en setBuffer:s offset:0 atIndex:1];
        [en setBytes:&n length:4 atIndex:2];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, (NSUInteger)nrow, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto silu_mul = [&]() {
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
        [en setComputePipelineState:g_pipe[k]];
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
        if (boff == ~0ull) return;
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
    auto conv_step = [&](uint64_t slot_off, uint64_t kwoff, int rows) {
        const uint32_t n = (uint32_t)n_embd;
        const uint32_t K = (uint32_t)g_spec.conv_l_cache;
        for (int r = 0; r < rows; r++) {
            const uint64_t hoff = (uint64_t)r * (uint64_t)n_embd * sizeof(float);
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

    for (int l = 0; l < g_spec.n_layer; l++) {
        const LayerSpec& L = g_spec.layers[(size_t)l];
        const uint64_t coff = (uint64_t)l * kv_row * 2u;                 // fp16
        const uint64_t kpos = coff + (uint64_t)pos0 * (uint64_t)kv_dim * 2u;

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
            conv_step((uint64_t)L.conv_slot * (uint64_t)(g_spec.conv_l_cache - 1) *
                          (uint64_t)n_embd * 4,
                      L.ex_off[metal::X_CONV_K], nrow);
            mul(g_bconvy, g_bconvy, 0, (uint64_t)n_embd * 4,
                g_bconv, (uint64_t)n_embd * 4, rowstride, n_embd);
            gemm(L.ex_type[metal::X_CONV_OUT], g_bconvy, g_bxb,
                 L.ex_off[metal::X_CONV_OUT], n_embd, n_embd, nrow);
        } else {
            gemm(L.type[metal::S_Q], g_bxb, g_bq, L.off[metal::S_Q], q_dim,  n_embd, nrow);
            gemm(L.type[metal::S_K], g_bxb, g_bk, L.off[metal::S_K], kv_dim, n_embd, nrow);
            gemm(L.type[metal::S_V], g_bxb, g_bv, L.off[metal::S_V], kv_dim, n_embd, nrow);

            bias(g_bq, L.bias[0], q_dim);
            bias(g_bk, L.bias[1], kv_dim);
            bias(g_bv, L.bias[2], kv_dim);

            // LFM3：QK RMSNorm 在 RoPE 之前
            if (L.ex_off[metal::X_QNORM] != ~0ull)
                headnorm(g_bq, L.ex_off[metal::X_QNORM], n_head, nrow);
            if (L.ex_off[metal::X_KNORM] != ~0ull)
                headnorm(g_bk, L.ex_off[metal::X_KNORM], n_kvh, nrow);

            // RoPE：q 就地，k 旋转后直接落到 cache 的 pos 槽
            {
                uint32_t a = (uint32_t)n_head, b = (uint32_t)n_kvh, c = (uint32_t)hd;
                uint32_t p0 = (uint32_t)pos0;
                float th = L.rope_theta_l;
                uint32_t ang = (L.rope_angles_l > 0 && L.rope_angles_l < hd / 2)
                                   ? (uint32_t)L.rope_angles_l : (uint32_t)(hd / 2);
                [en setComputePipelineState:g_fpipe[F_ROPE]];
                [en setBuffer:g_bq offset:0 atIndex:0];
                [en setBuffer:g_bk offset:0 atIndex:1];
                [en setBuffer:g_bkc offset:0 atIndex:2];
                [en setBytes:&kpos length:8 atIndex:3];
                [en setBytes:&a length:4 atIndex:4];
                [en setBytes:&b length:4 atIndex:5];
                [en setBytes:&c length:4 atIndex:6];
                [en setBytes:&p0 length:4 atIndex:7];
                [en setBytes:&th length:4 atIndex:8];
                [en setBytes:&ang length:4 atIndex:9];
                const uint32_t tot = (a + b) * (c / 2);
                [en dispatchThreads:MTLSizeMake(tot, (NSUInteger)nrow, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // v → cache（fp16，所以用 k_copy_h 而不是 k_copy）
            {
                uint32_t n = (uint32_t)kv_dim;
                uint64_t zero = 0;
                [en setComputePipelineState:g_fpipe[F_COPY_H]];
                [en setBuffer:g_bvc offset:0 atIndex:0];
                [en setBytes:&kpos length:8 atIndex:1];
                [en setBuffer:g_bv offset:0 atIndex:2];
                [en setBytes:&zero length:8 atIndex:3];
                [en setBytes:&n length:4 atIndex:4];
                [en dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, (NSUInteger)nrow, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
                const int nlen_max = std::min(pos0 + nrow,
                                              win > 0 ? (int)win : (pos0 + nrow));
                uint32_t nsp = (uint32_t)std::min(kAttnSplits, std::max(1, nlen_max / 96));

                if (nsp == 1) {
                    // 短上下文：融合版一步到位，省掉 partial 缓冲与 combine
                    [en setComputePipelineState:g_fpipe[F_ATTN]];
                    [en setBuffer:g_bq offset:0 atIndex:0];
                    [en setBuffer:g_bkc offset:0 atIndex:1];
                    [en setBuffer:g_bvc offset:0 atIndex:2];
                    [en setBuffer:g_batt offset:0 atIndex:3];
                    [en setBytes:&coff length:8 atIndex:4];
                    [en setBytes:&a length:4 atIndex:5];
                    [en setBytes:&b length:4 atIndex:6];
                    [en setBytes:&c length:4 atIndex:7];
                    [en setBytes:&nlen0 length:4 atIndex:8];
                    [en setBytes:&kvd length:4 atIndex:9];
                    [en setBytes:&sc length:4 atIndex:10];
                    [en setBytes:&win length:4 atIndex:11];
                    [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                         (NSUInteger)nrow, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    [en setComputePipelineState:g_fpipe[F_ATTN_SPLIT]];
                    [en setBuffer:g_bq offset:0 atIndex:0];
                    [en setBuffer:g_bkc offset:0 atIndex:1];
                    [en setBuffer:g_bvc offset:0 atIndex:2];
                    [en setBuffer:g_bpart offset:0 atIndex:3];
                    [en setBytes:&coff length:8 atIndex:4];
                    [en setBytes:&a length:4 atIndex:5];
                    [en setBytes:&b length:4 atIndex:6];
                    [en setBytes:&c length:4 atIndex:7];
                    [en setBytes:&nlen0 length:4 atIndex:8];
                    [en setBytes:&kvd length:4 atIndex:9];
                    [en setBytes:&sc length:4 atIndex:10];
                    [en setBytes:&nsp length:4 atIndex:11];
                    [en setBytes:&win length:4 atIndex:12];
                    [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                         (NSUInteger)nrow, (NSUInteger)nsp)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [en setComputePipelineState:g_fpipe[F_ATTN_COMB]];
                    [en setBuffer:g_bpart offset:0 atIndex:0];
                    [en setBuffer:g_batt offset:0 atIndex:1];
                    [en setBytes:&a length:4 atIndex:2];
                    [en setBytes:&c length:4 atIndex:3];
                    [en setBytes:&nsp length:4 atIndex:4];
                    [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4),
                                                         (NSUInteger)nrow, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }

            // LFM3：逐 head 的 sigmoid 输出门，在 o_proj 之前
            if (L.ex_off[metal::X_AGATE] != ~0ull) {
                gemm(L.ex_type[metal::X_AGATE], g_bxb, g_bgateatt,
                     L.ex_off[metal::X_AGATE], n_head, n_embd, nrow);
                gate_mul(nrow);
            }

            // 输出投影 → g_bxb（与 conv 路径统一落点）
            gemm(L.type[metal::S_OUT], g_batt, g_bxb, L.off[metal::S_OUT], n_embd, n_embd, nrow);
        }

        // sandwich norm：子层输出先归一化再加残差
        if (L.ex_off[metal::X_POST_ATTN] != ~0ull)
            rmsnorm(g_bxb, L.ex_off[metal::X_POST_ATTN], g_bxb, nrow);
        add(g_bx, g_bxb);

        // FFN
        rmsnorm(g_bx, L.off[metal::S_FFN_NORM], g_bxb, nrow);
        gemm(L.type[metal::S_GATE], g_bxb, g_bgate, L.off[metal::S_GATE], n_ff, n_embd, nrow);
        gemm(L.type[metal::S_UP],   g_bxb, g_bup,   L.off[metal::S_UP],   n_ff, n_embd, nrow);
        silu_mul();
        gemm(L.type[metal::S_DOWN], g_bffn, g_bxb, L.off[metal::S_DOWN], n_embd, n_ff, nrow);
        if (L.ex_off[metal::X_POST_FFN] != ~0ull)
            rmsnorm(g_bxb, L.ex_off[metal::X_POST_FFN], g_bxb, nrow);
        add(g_bx, g_bxb);
    }

    // 只有最后一行需要 logits：先把这一行搬到 xb 的第 0 行，再 norm + out_w。
    // 省掉 nrow-1 次 out_w —— 它是全模型最大的单个矩阵乘（占权重流量 36%）。
    {
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
    }
    rmsnorm(g_bxb, g_spec.off_out_norm, g_bfinal, 1);
    gemm(g_spec.type_out_w, g_bfinal, g_blogits, g_spec.off_out_w, g_spec.n_vocab, n_embd, 1);

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
    std::memcpy(logits, [g_blogits contents], sizeof(float) * (size_t)g_spec.n_vocab);
    return true;
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
    const size_t bytes = (size_t)g_spec.n_conv * (lcache > 1 ? lcache - 1 : 1) *
                         (size_t)g_spec.n_embd * sizeof(float);
    std::memset([g_bconvst contents], 0, bytes);
}

}  // namespace metal
}  // namespace dubhe
