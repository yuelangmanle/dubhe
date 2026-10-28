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
#include <cstring>

namespace dubhe {
namespace metal {

namespace {

enum KernelKind {
    K_F32 = 0, K_F16, K_Q4_0, K_Q5_0, K_Q8_0, K_Q4_K, K_Q6_K, K_COUNT
};

const char* const kNames[K_COUNT] = {
    "mv_f32", "mv_f16", "mv_q4_0", "mv_q5_0", "mv_q8_0", "mv_q4_k", "mv_q6_k"
};

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
    g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kDubheMSL]
                                options:opt
                                  error:&e];
    if (!g_lib) {
        g_err = e ? [[e localizedDescription] UTF8String] : "MSL 编译失败";
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
    // f32/f16 内核用 float4 展开，要求 n_in 是 4 的倍数
    if ((k == K_F32 || k == K_F16) && (n_in % 4) != 0) return false;
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
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pipe[k]];
    [enc setBuffer:g_w offset:0 atIndex:0];
    [enc setBuffer:g_x offset:0 atIndex:1];
    [enc setBuffer:g_y offset:0 atIndex:2];
    [enc setBytes:&nin length:sizeof(nin) atIndex:3];
    [enc setBytes:&woff length:sizeof(woff) atIndex:4];

    // 一个线程一行；每个线程的工作量很大（几百到几千次乘加），
    // 线程组给小一点能让更多行同时驻留，掩盖访存延迟。
    NSUInteger tg = std::min<NSUInteger>(g_pipe[k].maxTotalThreadsPerThreadgroup, 32);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_out, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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

namespace {

enum FwdKernel {
    F_RMSNORM = 0, F_ADD, F_BIAS, F_SILU, F_ROPE, F_COPY, F_ATTN, F_COUNT
};
const char* const kFwdNames[F_COUNT] = {
    "k_rmsnorm", "k_add", "k_bias", "k_silu_mul", "k_rope", "k_copy", "k_attn"
};
id<MTLComputePipelineState> g_fpipe[F_COUNT] = {nil};

Spec         g_spec;
bool         g_fwd_ready = false;

// 前向用到的所有激活缓冲
id<MTLBuffer> g_bx = nil, g_bxb = nil, g_bq = nil, g_bk = nil, g_bv = nil;
id<MTLBuffer> g_batt = nil, g_bgate = nil, g_bup = nil, g_bffn = nil, g_blogits = nil;
id<MTLBuffer> g_bkc = nil, g_bvc = nil;

id<MTLBuffer> make_buf(size_t bytes) {
    return [g_dev newBufferWithLength:(bytes ? bytes : 256) options:MTLResourceStorageModeShared];
}

void fwd_release() {
    g_fwd_ready = false;
    g_bx = g_bxb = g_bq = g_bk = g_bv = nil;
    g_batt = g_bgate = g_bup = g_bffn = g_blogits = nil;
    g_bkc = g_bvc = nil;
    for (int i = 0; i < F_COUNT; i++) g_fpipe[i] = nil;
}

}  // namespace

bool fwd_init(const Spec& s, std::string* err) {
    if (!g_ready) { if (err) *err = "Metal 未初始化"; return false; }
    g_fwd_ready = false;

    // 融合内核不支持的情况直接拒绝，让调用方退回 CPU
    if (s.hd > 128) { if (err) *err = "head_dim > 128，注意力内核未覆盖"; return false; }
    if (s.n_head % s.n_head_kv != 0) { if (err) *err = "n_head 不是 n_head_kv 的整数倍"; return false; }
    for (int l = 0; l < s.n_layer; l++) {
        if (s.layers[(size_t)l].type[metal::S_ATTN_NORM] != T_F32 ||
            s.layers[(size_t)l].type[metal::S_FFN_NORM] != T_F32) {
            if (err) *err = "norm 权重不是 F32，融合内核未覆盖";
            return false;
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

    g_spec = s;

    const size_t n_embd = (size_t)s.n_embd, n_ff = (size_t)s.n_ff, n_vocab = (size_t)s.n_vocab;
    const size_t kv_dim = (size_t)s.n_head_kv * (size_t)s.hd;
    const size_t kvb = (size_t)s.n_layer * (size_t)s.kv_cap * kv_dim * sizeof(float);

    g_bx      = make_buf(n_embd * 4);
    g_bxb     = make_buf(n_embd * 4);
    g_bq      = make_buf((size_t)s.n_head * (size_t)s.hd * 4);
    g_bk      = make_buf(kv_dim * 4);
    g_bv      = make_buf(kv_dim * 4);
    g_batt    = make_buf(n_embd * 4);
    g_bgate   = make_buf(n_ff * 4);
    g_bup     = make_buf(n_ff * 4);
    g_bffn    = make_buf(n_ff * 4);
    g_blogits = make_buf(n_vocab * 4);
    g_bkc     = make_buf(kvb);
    g_bvc     = make_buf(kvb);

    if (!g_bx || !g_bxb || !g_bq || !g_bk || !g_bv || !g_batt || !g_bgate || !g_bup ||
        !g_bffn || !g_blogits || !g_bkc || !g_bvc) {
        if (err) *err = "前向缓冲分配失败";
        return false;
    }
    std::memset([g_bkc contents], 0, kvb);
    std::memset([g_bvc contents], 0, kvb);

    g_fwd_ready = true;
    return true;
}

bool fwd_grow_kv(int new_cap, std::string* err) {
    if (!g_fwd_ready) return false;
    if (new_cap <= g_spec.kv_cap) return true;

    const size_t kv_dim = (size_t)g_spec.n_head_kv * (size_t)g_spec.hd;
    const size_t old_row = (size_t)g_spec.kv_cap * kv_dim;
    const size_t new_row = (size_t)new_cap * kv_dim;
    const size_t bytes = (size_t)g_spec.n_layer * new_row * sizeof(float);

    id<MTLBuffer> nk = make_buf(bytes), nv = make_buf(bytes);
    if (!nk || !nv) { if (err) *err = "KV cache 扩容分配失败"; return false; }
    std::memset([nk contents], 0, bytes);
    std::memset([nv contents], 0, bytes);

    // 统一内存：直接 CPU memcpy，按层重排
    float* dk = (float*)[nk contents];
    float* dv = (float*)[nv contents];
    const float* sk = (const float*)[g_bkc contents];
    const float* sv = (const float*)[g_bvc contents];
    for (int l = 0; l < g_spec.n_layer; l++) {
        std::memcpy(dk + (size_t)l * new_row, sk + (size_t)l * old_row, old_row * sizeof(float));
        std::memcpy(dv + (size_t)l * new_row, sv + (size_t)l * old_row, old_row * sizeof(float));
    }
    g_bkc = nk;
    g_bvc = nv;
    g_spec.kv_cap = new_cap;
    return true;
}

bool fwd_token(const float* x0, int pos, float* logits, std::string* err) {
    if (!g_fwd_ready) { if (err) *err = "GPU 前向未初始化"; return false; }
    if (pos >= g_spec.kv_cap) { if (err) *err = "KV cache 越界"; return false; }

    const int n_embd = g_spec.n_embd, n_ff = g_spec.n_ff;
    const int n_head = g_spec.n_head, n_kvh = g_spec.n_head_kv, hd = g_spec.hd;
    const int kv_dim = n_kvh * hd;
    const size_t kv_row = (size_t)g_spec.kv_cap * (size_t)kv_dim;

    std::memcpy([g_bx contents], x0, sizeof(float) * (size_t)n_embd);

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];

    auto elem = [&](id<MTLComputePipelineState> p, uint32_t n) {
        [en setComputePipelineState:p];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto mv = [&](uint32_t type, id<MTLBuffer> xin, id<MTLBuffer> yout, uint64_t woff,
                  int n_out, int n_in) -> bool {
        const int k = kind_of(type);
        if (k < 0) return false;
        uint32_t nin = (uint32_t)n_in;
        [en setComputePipelineState:g_pipe[k]];
        [en setBuffer:g_w offset:0 atIndex:0];
        [en setBuffer:xin offset:0 atIndex:1];
        [en setBuffer:yout offset:0 atIndex:2];
        [en setBytes:&nin length:sizeof(nin) atIndex:3];
        [en setBytes:&woff length:sizeof(woff) atIndex:4];
        NSUInteger tg = std::min<NSUInteger>(g_pipe[k].maxTotalThreadsPerThreadgroup, 32);
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_out, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        return true;
    };

    auto rmsnorm = [&](id<MTLBuffer> xin, uint64_t woff, id<MTLBuffer> yout) {
        uint32_t n = (uint32_t)n_embd;
        float eps = g_spec.rms_eps;
        [en setComputePipelineState:g_fpipe[F_RMSNORM]];
        [en setBuffer:xin offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBuffer:yout offset:0 atIndex:2];
        [en setBytes:&n length:4 atIndex:3];
        [en setBytes:&eps length:4 atIndex:4];
        [en setBytes:&woff length:8 atIndex:5];
        [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto bias = [&](id<MTLBuffer> y, uint64_t boff, int n) {
        if (boff == ~0ull) return;
        uint32_t nn = (uint32_t)n;
        [en setComputePipelineState:g_fpipe[F_BIAS]];
        [en setBuffer:y offset:0 atIndex:0];
        [en setBuffer:g_w offset:0 atIndex:1];
        [en setBytes:&boff length:8 atIndex:2];
        [en setBytes:&nn length:4 atIndex:3];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto add = [&](id<MTLBuffer> d, id<MTLBuffer> s) {
        uint32_t n = (uint32_t)n_embd;
        [en setComputePipelineState:g_fpipe[F_ADD]];
        [en setBuffer:d offset:0 atIndex:0];
        [en setBuffer:s offset:0 atIndex:1];
        [en setBytes:&n length:4 atIndex:2];
        [en dispatchThreads:MTLSizeMake((NSUInteger)n_embd, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    for (int l = 0; l < g_spec.n_layer; l++) {
        const LayerSpec& L = g_spec.layers[(size_t)l];
        const uint64_t coff = (uint64_t)l * kv_row * sizeof(float);
        const uint64_t kpos = coff + (uint64_t)pos * (uint64_t)kv_dim * sizeof(float);

        rmsnorm(g_bx, L.off[metal::S_ATTN_NORM], g_bxb);
        mv(L.type[metal::S_Q], g_bxb, g_bq,   L.off[metal::S_Q], n_head * hd, n_embd);
        mv(L.type[metal::S_K], g_bxb, g_bk,   L.off[metal::S_K], kv_dim, n_embd);
        mv(L.type[metal::S_V], g_bxb, g_bv,   L.off[metal::S_V], kv_dim, n_embd);

        bias(g_bq, L.bias[0], n_head * hd);
        bias(g_bk, L.bias[1], kv_dim);
        bias(g_bv, L.bias[2], kv_dim);

        // RoPE：q 就地，k 旋转后直接落到 cache 的 pos 槽
        {
            uint32_t a = (uint32_t)n_head, b = (uint32_t)n_kvh, c = (uint32_t)hd;
            uint32_t p = (uint32_t)pos;
            float th = g_spec.rope_theta;
            [en setComputePipelineState:g_fpipe[F_ROPE]];
            [en setBuffer:g_bq offset:0 atIndex:0];
            [en setBuffer:g_bk offset:0 atIndex:1];
            [en setBuffer:g_bkc offset:0 atIndex:2];
            [en setBytes:&kpos length:8 atIndex:3];
            [en setBytes:&a length:4 atIndex:4];
            [en setBytes:&b length:4 atIndex:5];
            [en setBytes:&c length:4 atIndex:6];
            [en setBytes:&p length:4 atIndex:7];
            [en setBytes:&th length:4 atIndex:8];
            const uint32_t tot = (a + b) * (c / 2);
            [en dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // v → cache
        {
            uint32_t n = (uint32_t)kv_dim;
            [en setComputePipelineState:g_fpipe[F_COPY]];
            [en setBuffer:g_bvc offset:0 atIndex:0];
            [en setBytes:&kpos length:8 atIndex:1];
            [en setBuffer:g_bv offset:0 atIndex:2];
            uint64_t zero = 0;
            [en setBytes:&zero length:8 atIndex:3];
            [en setBytes:&n length:4 atIndex:4];
            [en dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // 注意力
        {
            uint32_t a = (uint32_t)n_head, b = (uint32_t)n_kvh, c = (uint32_t)hd;
            uint32_t nlen = (uint32_t)(pos + 1);
            uint32_t kvd = (uint32_t)kv_dim;
            float sc = 1.0f / std::sqrt((float)hd);
            [en setComputePipelineState:g_fpipe[F_ATTN]];
            [en setBuffer:g_bq offset:0 atIndex:0];
            [en setBuffer:g_bkc offset:0 atIndex:1];
            [en setBuffer:g_bvc offset:0 atIndex:2];
            [en setBuffer:g_batt offset:0 atIndex:3];
            [en setBytes:&coff length:8 atIndex:4];
            [en setBytes:&a length:4 atIndex:5];
            [en setBytes:&b length:4 atIndex:6];
            [en setBytes:&c length:4 atIndex:7];
            [en setBytes:&nlen length:4 atIndex:8];
            [en setBytes:&kvd length:4 atIndex:9];
            [en setBytes:&sc length:4 atIndex:10];
            [en dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_head + 3) / 4), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        // 输出投影（复用 xb）+ 残差
        mv(L.type[metal::S_OUT], g_batt, g_bxb, L.off[metal::S_OUT], n_embd, n_embd);
        add(g_bx, g_bxb);

        // FFN
        rmsnorm(g_bx, L.off[metal::S_FFN_NORM], g_bxb);
        mv(L.type[metal::S_GATE], g_bxb, g_bgate, L.off[metal::S_GATE], n_ff, n_embd);
        mv(L.type[metal::S_UP],   g_bxb, g_bup,   L.off[metal::S_UP],   n_ff, n_embd);
        {
            uint32_t n = (uint32_t)n_ff;
            [en setComputePipelineState:g_fpipe[F_SILU]];
            [en setBuffer:g_bffn offset:0 atIndex:0];
            [en setBuffer:g_bgate offset:0 atIndex:1];
            [en setBuffer:g_bup offset:0 atIndex:2];
            [en setBytes:&n length:4 atIndex:3];
            [en dispatchThreads:MTLSizeMake((NSUInteger)n_ff, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        mv(L.type[metal::S_DOWN], g_bffn, g_bxb, L.off[metal::S_DOWN], n_embd, n_ff);
        add(g_bx, g_bxb);
    }

    rmsnorm(g_bx, g_spec.off_out_norm, g_bxb);
    mv(g_spec.type_out_w, g_bxb, g_blogits, g_spec.off_out_w, g_spec.n_vocab, n_embd);

    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        g_err = cb.error ? [[cb.error localizedDescription] UTF8String] : "GPU 前向执行失败";
        if (err) *err = g_err;
        g_fwd_ready = false;
        return false;
    }
    std::memcpy(logits, [g_blogits contents], sizeof(float) * (size_t)g_spec.n_vocab);
    return true;
}

}  // namespace metal
}  // namespace dubhe
