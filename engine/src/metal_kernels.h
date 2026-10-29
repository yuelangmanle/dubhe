// metal_kernels.h — Metal 着色器源码（运行时编译，不依赖离线 Metal Toolchain）
//
// 设计要点：
//   1. 权重在 GPU 侧只有一个 MTLBuffer，直接包住 mmap 的文件页（统一内存，零拷贝）。
//      每个张量用 woff（字节偏移）定位，避免 setBuffer:offset: 的 256 字节对齐限制。
//   2. **一个 simdgroup 算一行**（不是一线程一行）。32 个 lane 分摊行内元素，
//      最后 simd_sum 归约。B=1 解码时这是关键：ffn_down 只有 896 行，
//      "一线程一行"只发出 896 个线程，10 核 GPU 一半算力在睡觉，而且每个
//      线程背着一串串行依赖的 load，延迟完全盖不住。改成 simdgroup 之后
//      并行度 ×32（28672 线程），且同一时刻 32 个 lane 读的是行内相邻数据。
//   3. 权重按 GGUF 原始量化格式直读，不解包到内存。
//   4. **prefill 用批处理内核（mm_*）**：一个 simdgroup 负责一行输出，块只解一次
//      就乘 BMAX 个激活。解码时权重带宽是天花板（每 token 要把 0.4 GB 权重全读
//      一遍），批处理把同一遍读取摊到 8 个 token 上，算术强度直接 ×8。
//
// lane 分工的两种形态（决定了解码代码怎么写）：
//   * 32 元块（Q4_0/Q5_0/Q8_0）：一个块正好 32 个元素 → lane 就是块内下标。
//     低 16 个元素取低半字节，高 16 个取高半字节。
//   * 256 元块（Q4_K/Q6_K）：块内再分 8（或 16）个 32（或 16）元素的小段，
//     lane 在每个小段里各取一个元素 → 天然对应量化格式的 scale 布局。
#pragma once

static const char* kDubheMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// 一个 threadgroup 装 4 个 simdgroup → 4 行。128 线程是苹果 GPU 上
// 占用率与调度粒度的折中（M 系列每个 core 可驻留多个 128 线程的组）。
#define SG_PER_TG 4

// prefill 批大小上限。8 是寄存器压力与权重复用率的平衡点：
// 每个 lane 8 个累加器，加上解码用的临时量，仍在 255 个寄存器内。
#define BMAX 8

// ---------------------------------------------------------------- 工具
static inline float f16_to_f32(ushort h) { return float(as_type<half>(h)); }

static inline float rd_f16(device const uchar* p) {
    ushort h = ushort(p[0]) | (ushort(p[1]) << 8);
    return f16_to_f32(h);
}

// 按字节偏移取 float 指针（绕开 setBuffer:offset: 的 256 字节对齐限制）
static inline device float* fat(device float* b, ulong off) {
    return (device float*)((device uchar*)b + off);
}
static inline device const float* fatc(device const float* b, ulong off) {
    return (device const float*)((device const uchar*)b + off);
}

// ================================================================ matvec（解码，B=1）
//
// 统一签名（第 5 个参数 n_out 用于边界判断，因为按 threadgroup 派发时
// 硬件不会自动裁掉越界的 threadgroup）：
//   w     : 整个 mmap
//   x     : 激活 [n_in] f32
//   y     : 输出 [n_out] f32
//   woff  : 本张量在 mmap 内的字节偏移

// ---------------------------------------------------------------- F32 / F16
kernel void mv_f32(device const uchar* w      [[buffer(0)]],
                   device const float* x      [[buffer(1)]],
                   device float* y            [[buffer(2)]],
                   constant uint& n_in        [[buffer(3)]],
                   constant ulong& woff       [[buffer(4)]],
                   constant uint& n_out       [[buffer(5)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    device const float* p = (device const float*)(w + woff) + (ulong)row * n_in;
    float acc = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) acc += p[i] * x[i];
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

kernel void mv_f16(device const uchar* w      [[buffer(0)]],
                   device const float* x      [[buffer(1)]],
                   device float* y            [[buffer(2)]],
                   constant uint& n_in        [[buffer(3)]],
                   constant ulong& woff       [[buffer(4)]],
                   constant uint& n_out       [[buffer(5)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    device const ushort* p = (device const ushort*)(w + woff) + (ulong)row * n_in;
    float acc = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) acc += f16_to_f32(p[i]) * x[i];
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- Q4_0（18 字节 / 32 元）
kernel void mv_q4_0(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 18u;
    // lane 0..15 取低半字节（元素 0..15），lane 16..31 取高半字节（元素 16..31）
    const uint idx = lane & 15u;
    const bool hi  = lane >= 16u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 18u;
        const uint byte = uint(q[2u + idx]);
        const int v = hi ? int(byte >> 4) : int(byte & 0x0Fu);
        acc += rd_f16(q) * float(v - 8) * x[(b << 5) + lane];
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- Q5_0（22 字节 / 32 元）
kernel void mv_q5_0(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 22u;
    const uint idx = lane & 15u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 22u;
        const uint qh = uint(q[2]) | (uint(q[3]) << 8) | (uint(q[4]) << 16) | (uint(q[5]) << 24);
        const int bit5 = int((qh >> lane) & 1u) << 4;
        const uint byte = uint(q[6u + idx]);
        const int v = (lane < 16u) ? (int(byte & 0x0Fu) | bit5)
                                   : (int(byte >> 4)    | bit5);
        acc += rd_f16(q) * float(v - 16) * x[(b << 5) + lane];
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- Q8_0（34 字节 / 32 元）
kernel void mv_q8_0(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 34u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 34u;
        const int v = int(((device const char*)(q + 2u))[lane]);
        acc += rd_f16(q) * float(v) * x[(b << 5) + lane];
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- Q4_K（144 字节 / 256 元）
static inline void k4_scale_min(int j, device const uchar* q, thread uchar& d, thread uchar& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)    | ((q[j] >> 6) << 4);
    }
}

kernel void mv_q4_k(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 8;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 144u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q  = p + b * 144u;
        const float d  = rd_f16(q);
        const float dm = rd_f16(q + 2u);
        device const uchar* sc = q + 4u;
        device const uchar* qs = q + 16u;
        device const float* xb = x + (b << 8);
        // 8 个小段 × 32 元素：lane 在每个小段里各取一个元素。
        // 关键简化：把 min 项折进逐元素 fma —— sum(d*sc*v - dm*m) * x
        //   = sum (d*sc*v - dm*m) * x，省掉单独的 x 求和。
        for (uint s = 0; s < 8u; s++) {
            uchar s1, m1;
            k4_scale_min(int(s), sc, s1, m1);
            const uint byte = uint(qs[(s >> 1) * 32u + lane]);
            const int v = (s & 1u) ? int(byte >> 4) : int(byte & 0x0Fu);
            acc += (d * float(s1) * float(v) - dm * float(m1)) * xb[(s << 5) + lane];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- Q6_K（210 字节 / 256 元）
kernel void mv_q6_k(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 8;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 210u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q  = p + b * 210u;
        device const uchar* ql = q;
        device const uchar* qh = q + 128u;
        device const char*  sc = (device const char*)(q + 192u);
        const float d = rd_f16(q + 208u);
        device const float* xb = x + (b << 8);
        const uint si = lane >> 4;    // scale 粒度是 16 元素
        // 256 元素按 32 一段分 8 段，lane 在每段取一个 → 元素 lane + 32k
        for (uint k = 0; k < 8u; k++) {
            const uint h = k >> 2;                  // 前半 / 后半 128
            const uint g = k & 3u;                  // 段内第几组
            const uint li = h * 64u + lane + (g & 1u) * 32u;
            const uint lo = (g < 2u) ? (uint(ql[li]) & 0x0Fu) : (uint(ql[li]) >> 4);
            const uint hi2 = (uint(qh[h * 32u + lane]) >> (2u * g)) & 3u;
            const int v = int(lo | (hi2 << 4)) - 32;
            const float s = float(sc[h * 8u + si + g * 2u]);
            acc += d * s * float(v) * xb[(k << 5) + lane];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ================================================================ matmul（prefill，B=nrow≤8）
//
// 为什么值：解码时每个 token 必须把全部权重读一遍，带宽就是天花板。
// 批处理把"读一遍权重"摊到 nrow 个 token 上：每解出一个权重值，
// 立刻和 nrow 个激活相乘（nrow 次 fma）。
//
// 关键布局决定：激活按 **转置** 存放，XT[e][j]（e 是行内元素下标，j 是批内
// 第几行），步长恒为 BMAX。这样"一个权重值 × 8 个激活"退化成两条 float4
// 加载 —— 而不是 8 条标量加载。实测这一步把 prefill 又提速一倍多：
// 每元素从「1 权重 + 8 标量 X + 8 fma」变成「1 权重 + 2 float4 + 8 fma」。
//
// XT 由 k_trans 生成，并且对 j >= nrow 的位置填 0，于是内核里可以完全
// 去掉 `if (j < nrow)` 的分支。

// 一条权重值 wv 同时更新 8 个累加器
#define FMA8(wv)                                                       \
    do {                                                               \
        const float4 a0 = xp[0];                                       \
        const float4 a1 = xp[1];                                       \
        acc[0] += (wv) * a0.x;  acc[1] += (wv) * a0.y;                 \
        acc[2] += (wv) * a0.z;  acc[3] += (wv) * a0.w;                 \
        acc[4] += (wv) * a1.x;  acc[5] += (wv) * a1.y;                 \
        acc[6] += (wv) * a1.z;  acc[7] += (wv) * a1.w;                 \
    } while (0)

// 归约并写回 [j][n_out]
#define RED8()                                                         \
    do {                                                               \
        for (int j = 0; j < BMAX; j++) {                               \
            const float s = simd_sum(acc[j]);                          \
            if (lane == 0u) Y[(ulong)j * n_out + row] = s;             \
        }                                                              \
    } while (0)

kernel void mm_f32(device const uchar* w      [[buffer(0)]],
                   device const float* X      [[buffer(1)]],
                   device float* Y            [[buffer(2)]],
                   constant uint& n_in        [[buffer(3)]],
                   constant ulong& woff       [[buffer(4)]],
                   constant uint& n_out       [[buffer(5)]],
                   constant uint& nrow        [[buffer(6)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    device const float* p = (device const float*)(w + woff) + (ulong)row * n_in;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) {
        device const float4* xp = (device const float4*)(X + (ulong)i * BMAX);
        FMA8(p[i]);
    }
    RED8();
}

kernel void mm_f16(device const uchar* w      [[buffer(0)]],
                   device const float* X      [[buffer(1)]],
                   device float* Y            [[buffer(2)]],
                   constant uint& n_in        [[buffer(3)]],
                   constant ulong& woff       [[buffer(4)]],
                   constant uint& n_out       [[buffer(5)]],
                   constant uint& nrow        [[buffer(6)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    device const ushort* p = (device const ushort*)(w + woff) + (ulong)row * n_in;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) {
        device const float4* xp = (device const float4*)(X + (ulong)i * BMAX);
        FMA8(f16_to_f32(p[i]));
    }
    RED8();
}

kernel void mm_q4_0(device const uchar* w     [[buffer(0)]],
                    device const float* X     [[buffer(1)]],
                    device float* Y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    constant uint& nrow       [[buffer(6)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 18u;
    const uint idx = lane & 15u;
    const bool hi  = lane >= 16u;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 18u;
        const uint byte = uint(q[2u + idx]);
        const int v = hi ? int(byte >> 4) : int(byte & 0x0Fu);
        const uint e = (b << 5) + lane;
        device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
        FMA8(rd_f16(q) * float(v - 8));
    }
    RED8();
}

kernel void mm_q5_0(device const uchar* w     [[buffer(0)]],
                    device const float* X     [[buffer(1)]],
                    device float* Y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    constant uint& nrow       [[buffer(6)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 22u;
    const uint idx = lane & 15u;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 22u;
        const uint qh = uint(q[2]) | (uint(q[3]) << 8) | (uint(q[4]) << 16) | (uint(q[5]) << 24);
        const int bit5 = int((qh >> lane) & 1u) << 4;
        const uint byte = uint(q[6u + idx]);
        const int v = (lane < 16u) ? (int(byte & 0x0Fu) | bit5)
                                   : (int(byte >> 4)    | bit5);
        const uint e = (b << 5) + lane;
        device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
        FMA8(rd_f16(q) * float(v - 16));
    }
    RED8();
}

kernel void mm_q8_0(device const uchar* w     [[buffer(0)]],
                    device const float* X     [[buffer(1)]],
                    device float* Y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    constant uint& nrow       [[buffer(6)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 34u;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 34u;
        const int v = int(((device const char*)(q + 2u))[lane]);
        const uint e = (b << 5) + lane;
        device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
        FMA8(rd_f16(q) * float(v));
    }
    RED8();
}

kernel void mm_q4_k(device const uchar* w     [[buffer(0)]],
                    device const float* X     [[buffer(1)]],
                    device float* Y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    constant uint& nrow       [[buffer(6)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 8;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 144u;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q  = p + b * 144u;
        const float d  = rd_f16(q);
        const float dm = rd_f16(q + 2u);
        device const uchar* sc = q + 4u;
        device const uchar* qs = q + 16u;
        for (uint s = 0; s < 8u; s++) {
            uchar s1, m1;
            k4_scale_min(int(s), sc, s1, m1);
            const uint byte = uint(qs[(s >> 1) * 32u + lane]);
            const int v = (s & 1u) ? int(byte >> 4) : int(byte & 0x0Fu);
            const uint e = (b << 8) + (s << 5) + lane;
            device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
            FMA8(d * float(s1) * float(v) - dm * float(m1));
        }
    }
    RED8();
}

kernel void mm_q6_k(device const uchar* w     [[buffer(0)]],
                    device const float* X     [[buffer(1)]],
                    device float* Y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    constant uint& n_out      [[buffer(5)]],
                    constant uint& nrow       [[buffer(6)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    const uint nb = n_in >> 8;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 210u;
    const uint si = lane >> 4;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q  = p + b * 210u;
        device const uchar* ql = q;
        device const uchar* qh = q + 128u;
        device const char*  sc = (device const char*)(q + 192u);
        const float d = rd_f16(q + 208u);
        for (uint k = 0; k < 8u; k++) {
            const uint h = k >> 2;
            const uint g = k & 3u;
            const uint li = h * 64u + lane + (g & 1u) * 32u;
            const uint lo = (g < 2u) ? (uint(ql[li]) & 0x0Fu) : (uint(ql[li]) >> 4);
            const uint hi2 = (uint(qh[h * 32u + lane]) >> (2u * g)) & 3u;
            const int v = int(lo | (hi2 << 4)) - 32;
            const uint e = (b << 8) + (k << 5) + lane;
            device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
            FMA8(d * float(sc[h * 8u + si + g * 2u]) * float(v));
        }
    }
    RED8();
}

// ================================================================ 融合算子
//
// 为什么要这些：一次 command buffer 的 commit+wait 在本机实测要 200 µs。
// 如果每个 matvec 都同步一次，0.5B 模型单 token 有 169 次同步 = 34 ms，
// 把 GPU 的算力全埋掉了。所以整层前向的所有算子必须编进同一个 command
// buffer，CPU 只在最外层同步一次。
//
// 逐元素算子统一用 uint2 网格：gid.x = 行内下标，gid.y = 批内第几行。
// 解码时只派发 gid.y = 0 一层，批处理时派发 nrow 层——同一份代码覆盖两种模式。

// 激活转置：[nrow][n] → [n][BMAX]，j >= nrow 的位置填 0。
// 填 0 是为了让 matmul 内核可以无条件做满 8 次 fma，去掉所有分支。
kernel void k_trans(device const float* src [[buffer(0)]],
                    device float* dst       [[buffer(1)]],
                    constant uint& n        [[buffer(2)]],
                    constant uint& nrow     [[buffer(3)]],
                    uint e [[thread_position_in_grid]])
{
    if (e >= n) return;
    for (uint j = 0; j < BMAX; j++)
        dst[(ulong)e * BMAX + j] = (j < nrow) ? src[(ulong)j * n + e] : 0.0f;
}

// RMSNorm：out = x * rsqrt(mean(x^2)+eps) * w   （w 为 f32，位于权重 mmap 内）
// 一个 threadgroup 处理一行，行号由 threadgroup_position_in_grid 给出。
kernel void k_rmsnorm(device const float* x [[buffer(0)]],
                      device const uchar* w [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      constant uint& n      [[buffer(3)]],
                      constant float& eps   [[buffer(4)]],
                      constant ulong& woff  [[buffer(5)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]])
{
    device const float* xr = x + (ulong)row * n;
    device float* orow = out + (ulong)row * n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += tgs) ss += xr[i] * xr[i];
    threadgroup float sm[32];
    ss = simd_sum(ss);
    if ((tid & 31u) == 0u) sm[tid >> 5] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float v = (tid < (tgs >> 5)) ? sm[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0u) sm[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(sm[0] / float(n) + eps);
    device const float* wf = (device const float*)(w + woff);
    for (uint i = tid; i < n; i += tgs) orow[i] = xr[i] * scale * wf[i];
}

// d += s
kernel void k_add(device float* d       [[buffer(0)]],
                  device const float* s [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n) {
        const ulong o = (ulong)gid.y * n + gid.x;
        d[o] += s[o];
    }
}

// y += b（b 是 mmap 里的 f32 bias，用字节偏移定位；bias 对所有行相同）
kernel void k_bias(device float* y          [[buffer(0)]],
                   device const uchar* b    [[buffer(1)]],
                   constant ulong& boff     [[buffer(2)]],
                   constant uint& n         [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n) y[(ulong)gid.y * n + gid.x] += ((device const float*)(b + boff))[gid.x];
}

// o = silu(g) * u
kernel void k_silu_mul(device float* o        [[buffer(0)]],
                       device const float* g  [[buffer(1)]],
                       device const float* u  [[buffer(2)]],
                       constant uint& n       [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n) {
        const ulong i = (ulong)gid.y * n + gid.x;
        const float v = g[i];
        o[i] = (v / (1.0f + exp(-v))) * u[i];
    }
}

// RoPE：q 就地旋转；k 旋转后直接写进 KV cache 的 pos 槽。
// pos = pos0 + gid.y，koff 已包含层的偏移与 pos0 的偏移。
kernel void k_rope(device float* q          [[buffer(0)]],
                   device const float* k    [[buffer(1)]],
                   device float* kc         [[buffer(2)]],
                   constant ulong& koff     [[buffer(3)]],
                   constant uint& n_head    [[buffer(4)]],
                   constant uint& n_kvh     [[buffer(5)]],
                   constant uint& hd        [[buffer(6)]],
                   constant uint& pos0      [[buffer(7)]],
                   constant float& theta    [[buffer(8)]],
                   constant uint& angles    [[buffer(9)]],
                   uint2 gid [[thread_position_in_grid]])
{
    const uint hh = hd >> 1;   // 注意：half 是 MSL 内建的 16 位浮点类型名，不能用作变量
    if (gid.x >= (n_head + n_kvh) * hh) return;
    const uint row = gid.y;
    const uint h = gid.x / hh;
    const uint i = gid.x - h * hh;
    // partial RoPE（zero-tail）：只有前 angles 个角度真的转，其余频率取 0，
    // cos=1/sin=0 退化成恒等拷贝。这样 k 仍然会照常落到 cache 的正确槽位。
    const float freq = (i < angles) ? (1.0f / pow(theta, float(2u * i) / float(hd))) : 0.0f;
    const float a = float(pos0 + row) * freq;
    const float c = cos(a), s = sin(a);
    const uint qdim = n_head * hd, kdim = n_kvh * hd;
    if (h < n_head) {
        device float* v = q + (ulong)row * qdim + h * hd;
        const float x0 = v[i], x1 = v[i + hh];
        v[i]      = x0 * c - x1 * s;
        v[i + hh] = x0 * s + x1 * c;
    } else {
        const uint kh = h - n_head;
        device const float* src = k + (ulong)row * kdim + kh * hd;
        device float* dst = fat(kc, koff) + (ulong)row * kdim + kh * hd;
        const float x0 = src[i], x1 = src[i + hh];
        dst[i]      = x0 * c - x1 * s;
        dst[i + hh] = x0 * s + x1 * c;
    }
}

// d[doff + row*n + i] = s[soff + row*n + i]
kernel void k_copy(device float* d          [[buffer(0)]],
                   constant ulong& doff     [[buffer(1)]],
                   device const float* s    [[buffer(2)]],
                   constant ulong& soff     [[buffer(3)]],
                   constant uint& n         [[buffer(4)]],
                   uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n) {
        const ulong o = (ulong)gid.y * n + gid.x;
        fat(d, doff)[o] = fatc(s, soff)[o];
    }
}

// ---------------------------------------------------------------- GQA 因果注意力
//
// 问题：解码时只有 n_head 个 head（0.5B 是 14 个），一个 head 一个 simdgroup
// 就只有 14 个 simdgroup = 448 线程，而 M4 有 40 个 SIMD 单元。于是每个
// SIMD 单元上只挂一个 simdgroup，完全没有延迟掩盖：每次 key 迭代里的
// simd_sum / exp 都是串行依赖，全暴露在关键路径上。
// 实测 631 token 上下文时注意力吃掉 18 ms/token（短上下文只要 12 ms/token）。
//
// 解法（flash-decoding 的 split-K）：把 key 区间切成 nsp 段，每段一个
// simdgroup 并行算局部 (m, l, acc)，再用一个很轻的 combine 内核按
// exp(m_i - M) 归并。并行度 ×nsp，串行长度 ÷nsp。
//
// 空区间（k0 >= k1）写 m=-1e30 / l=0 / acc=0，combine 里 exp(-1e30 - M) = 0，
// 自然不参与归并——批处理时每行长度不同，靠这个机制统一。

// 短上下文（nsp == 1）走这个融合版：直接写输出，不经过 partial 缓冲，
// 也少一次 combine dispatch。split 版在 nsp == 1 时是纯粹的额外开销。
kernel void k_attn(device const float* q  [[buffer(0)]],
                   device const float* kc [[buffer(1)]],
                   device const float* vc [[buffer(2)]],
                   device float* out      [[buffer(3)]],
                   constant ulong& coff   [[buffer(4)]],
                   constant uint& n_head  [[buffer(5)]],
                   constant uint& n_kvh   [[buffer(6)]],
                   constant uint& hd      [[buffer(7)]],
                   constant uint& nlen0   [[buffer(8)]],
                   constant uint& kv_dim  [[buffer(9)]],
                   constant float& scale  [[buffer(10)]],
                   constant uint& win     [[buffer(11)]],
                   uint2 tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row  = tgid.y;
    const uint head = tgid.x * 4u + sg;
    if (head >= n_head) return;
    const uint mul = n_head / n_kvh;
    const uint kvh = head / mul;
    device const float* qh = q + ((ulong)row * n_head + head) * hd;
    device const float* kb = fatc(kc, coff) + kvh * hd;
    device const float* vb = fatc(vc, coff) + kvh * hd;
    const uint nlen = nlen0 + row;
    // 滑窗：只回看最近 win 个位置（win == 0 表示全量）
    const uint t0 = (win > 0u && nlen > win) ? (nlen - win) : 0u;

    float m = -1e30f, l = 0.0f;
    float acc[4];
    for (int j = 0; j < 4; j++) acc[j] = 0.0f;

    for (uint t = t0; t < nlen; t++) {
        device const float* kk = kb + (ulong)t * kv_dim;
        float partial = 0.0f;
        for (uint d = lane; d < hd; d += 32u) partial += qh[d] * kk[d];
        const float s = simd_sum(partial) * scale;

        float w;
        if (s > m) {
            const float c = exp(m - s);
            l *= c;
            int jj = 0;
            for (uint d = lane; d < hd; d += 32u, jj++) acc[jj] *= c;
            m = s;
            w = 1.0f;
        } else {
            w = exp(s - m);
        }
        l += w;

        device const float* vv = vb + (ulong)t * kv_dim;
        int j = 0;
        for (uint d = lane; d < hd; d += 32u, j++) acc[j] += w * vv[d];
    }

    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    device float* o = out + ((ulong)row * n_head + head) * hd;
    int j = 0;
    for (uint d = lane; d < hd; d += 32u, j++) o[d] = acc[j] * inv;
}

// 局部（未归并）结果布局：[row][split][head][hd + 2]，最后两个是 m 与 l
kernel void k_attn_split(device const float* q  [[buffer(0)]],
                         device const float* kc [[buffer(1)]],
                         device const float* vc [[buffer(2)]],
                         device float* part     [[buffer(3)]],
                         constant ulong& coff   [[buffer(4)]],
                         constant uint& n_head  [[buffer(5)]],
                         constant uint& n_kvh   [[buffer(6)]],
                         constant uint& hd      [[buffer(7)]],
                         constant uint& nlen0   [[buffer(8)]],
                         constant uint& kv_dim  [[buffer(9)]],
                         constant float& scale  [[buffer(10)]],
                         constant uint& nsp     [[buffer(11)]],
                         constant uint& win     [[buffer(12)]],
                         uint3 tgid [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row  = tgid.y;
    const uint sp   = tgid.z;
    const uint head = tgid.x * 4u + sg;
    if (head >= n_head) return;

    const uint nlen = nlen0 + row;
    const uint chunk = (nlen + nsp - 1u) / nsp;
    // 滑窗：本段的下界不能早于 t0。段被完全裁掉时 l 保持 0，
    // combine 阶段按 exp 权重 0 处理，不影响结果。
    const uint t0 = (win > 0u && nlen > win) ? (nlen - win) : 0u;
    uint k0 = sp * chunk;
    if (k0 < t0) k0 = t0;
    uint k1 = sp * chunk + chunk;
    if (k1 > nlen) k1 = nlen;

    const uint W = hd + 2u;
    device float* p = part + (((ulong)row * nsp + sp) * n_head + head) * W;

    float m = -1e30f, l = 0.0f;
    float acc[4];
    for (int j = 0; j < 4; j++) acc[j] = 0.0f;

    if (k0 < k1) {
        const uint mul = n_head / n_kvh;
        const uint kvh = head / mul;
        device const float* qh = q + ((ulong)row * n_head + head) * hd;
        device const float* kb = fatc(kc, coff) + kvh * hd;
        device const float* vb = fatc(vc, coff) + kvh * hd;

        for (uint t = k0; t < k1; t++) {
            device const float* kk = kb + (ulong)t * kv_dim;
            float partial = 0.0f;
            for (uint d = lane; d < hd; d += 32u) partial += qh[d] * kk[d];
            const float s = simd_sum(partial) * scale;

            float w;
            if (s > m) {
                // 在线 softmax：最大值变大时，除了缩放分母 l，还必须把已经
                // 累加的 acc 一起按 exp(m_old - m_new) 缩放，否则历史 key
                // 的贡献会被高估——flash-attention 最经典的坑。
                const float c = exp(m - s);
                l *= c;
                int jj = 0;
                for (uint d = lane; d < hd; d += 32u, jj++) acc[jj] *= c;
                m = s;
                w = 1.0f;
            } else {
                w = exp(s - m);
            }
            l += w;

            device const float* vv = vb + (ulong)t * kv_dim;
            int j = 0;
            for (uint d = lane; d < hd; d += 32u, j++) acc[j] += w * vv[d];
        }
    }

    int j = 0;
    for (uint d = lane; d < hd; d += 32u, j++) p[d] = acc[j];
    if (lane == 0u) { p[hd] = m; p[hd + 1u] = l; }
}

// 归并 nsp 个局部结果。nsp == 1 时它只是把 acc/l 归一化后写出。
kernel void k_attn_comb(device const float* part [[buffer(0)]],
                        device float* out        [[buffer(1)]],
                        constant uint& n_head    [[buffer(2)]],
                        constant uint& hd        [[buffer(3)]],
                        constant uint& nsp       [[buffer(4)]],
                        uint3 tgid [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row  = tgid.y;
    const uint head = tgid.x * 4u + sg;
    if (head >= n_head) return;

    const uint W = hd + 2u;
    float m = -1e30f;
    for (uint s = 0; s < nsp; s++) {
        m = max(m, part[(((ulong)row * nsp + s) * n_head + head) * W + hd]);
    }

    float l = 0.0f;
    float acc[4];
    for (int j = 0; j < 4; j++) acc[j] = 0.0f;
    for (uint s = 0; s < nsp; s++) {
        device const float* p = part + (((ulong)row * nsp + s) * n_head + head) * W;
        const float ls = p[hd + 1u];
        const float c = (ls > 0.0f) ? exp(p[hd] - m) : 0.0f;
        l += ls * c;
        int j = 0;
        for (uint d = lane; d < hd; d += 32u, j++) acc[j] += p[d] * c;
    }

    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    device float* o = out + ((ulong)row * n_head + head) * hd;
    int j = 0;
    for (uint d = lane; d < hd; d += 32u, j++) o[d] = acc[j] * inv;
}

// ================================================================ LFM3 专用
//
// LFM3 在 llama 那套子层之外还多四件事：每 head 的 QK RMSNorm、
// 逐 head 的 sigmoid 输出门、门控短卷积、sandwich norm。
// sandwich norm 复用 k_rmsnorm（只是权重偏移不同），所以这里只补三个 kernel
// 和一个带偏移的乘法。

// 每 head 独立 RMSNorm。一个 threadgroup 处理一个 head，
// 权重只有 hd 长、所有 head 共用。原地（out == x）安全：
// 归约全部做完并过了 barrier 才开始写。
//
// 网格用一维：threadgroup id 展开成 (row, head)。Metal 要求同一个 kernel 里
// 的网格属性要么全是标量、要么全是同宽度向量——threads_per_threadgroup 是
// 标量，所以这里不能写成 uint2 的二维网格。
kernel void k_headnorm(device const float* x [[buffer(0)]],
                       device const uchar* w [[buffer(1)]],
                       device float* out     [[buffer(2)]],
                       constant uint& n_head [[buffer(3)]],
                       constant uint& hd     [[buffer(4)]],
                       constant float& eps   [[buffer(5)]],
                       constant ulong& woff  [[buffer(6)]],
                       uint tg   [[threadgroup_position_in_grid]],
                       uint tid  [[thread_position_in_threadgroup]],
                       uint tgs  [[threads_per_threadgroup]])
{
    const uint head = tg % n_head;
    const uint row  = tg / n_head;
    device const float* xr = x + ((ulong)row * n_head + head) * hd;
    device float* orow = out + ((ulong)row * n_head + head) * hd;
    float ss = 0.0f;
    for (uint i = tid; i < hd; i += tgs) ss += xr[i] * xr[i];
    threadgroup float sm[32];
    ss = simd_sum(ss);
    if ((tid & 31u) == 0u) sm[tid >> 5] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float v = (tid < (tgs >> 5)) ? sm[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0u) sm[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(sm[0] / float(hd) + eps);
    device const float* wf = (device const float*)(w + woff);
    for (uint i = tid; i < hd; i += tgs) orow[i] = xr[i] * scale * wf[i];
}

// 逐 head 的 sigmoid 输出门：att[head] *= sigmoid(gate[head])
kernel void k_gate_mul(device float* att         [[buffer(0)]],
                       device const float* g     [[buffer(1)]],
                       constant uint& n_head     [[buffer(2)]],
                       constant uint& hd         [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= n_head * hd) return;
    const uint h = gid.x / hd;
    const float s = 1.0f / (1.0f + exp(-g[(ulong)gid.y * n_head + h]));
    att[(ulong)gid.y * n_head * hd + gid.x] *= s;
}

// 带偏移的逐元素乘法：dst[r][i] = a[ao + r*n + i] * b[bo + r*n + i]
// ShortConv 用它做 hidden = B ⊙ X 和 y = C ⊙ conv_out，省掉两次额外搬运。
// ao/bo 是**字节**偏移（和其余 kernel 的 woff/coff 一致），所以必须过 fatc，
// 不能直接写 a[ao + o]——那是按元素下标算的，会把偏移放大 4 倍。
// sa/sb 是 a/b 的**行**跨度（字节）。g_bconv 每行是 3*n_embd 而不是 n_embd，
// 把行跨度写死成 n 会让 row 0 正确、row≥1 全错——prefill 时表现为
// "nrow=1 精确、nrow=2 就开始崩"。
kernel void k_mul(device float* dst        [[buffer(0)]],
                  device const float* a    [[buffer(1)]],
                  device const float* b    [[buffer(2)]],
                  constant uint& n         [[buffer(3)]],
                  constant ulong& ao       [[buffer(4)]],
                  constant ulong& bo       [[buffer(5)]],
                  constant ulong& sa       [[buffer(6)]],
                  constant ulong& sb       [[buffer(7)]],
                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= n) return;
    const ulong row = (ulong)gid.y;
    dst[row * n + gid.x] = fatc(a, ao + row * sa)[gid.x] * fatc(b, bo + row * sb)[gid.x];
}

// LFM3 ShortConv 的深度可分离因果卷积。
//
//   state 存 h(t-K+1) .. h(t-1)（K-1 个），state[0] 最旧；
//   y[c] = Σ_k W[c*K+k] * buf[k][c]，buf = [state..., h(t)]
//
// 卷积**先算**、state **后滚**（见 k_conv_shift）。反过来的话核会整体偏移
// 一位，而且 h(t) 会被 W[K-2] 和 W[K-1] 各算一次——CPU 参考路径踩过这个坑。
// 一次 dispatch 只处理一行；多行时主机侧按行序连续派发，
// 同一 command buffer 内的 compute dispatch 顺序执行，state 能逐行正确递推。
kernel void k_conv(device const float* h        [[buffer(0)]],
                   device float* y              [[buffer(1)]],
                   device const uchar* w        [[buffer(2)]],
                   device float* state          [[buffer(3)]],
                   constant uint& n             [[buffer(4)]],
                   constant uint& K             [[buffer(5)]],
                   constant ulong& kwoff        [[buffer(6)]],
                   constant ulong& soff         [[buffer(7)]],
                   constant ulong& hoff         [[buffer(8)]],
                   constant ulong& yoff         [[buffer(9)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    const uint c = gid;
    device const float* wi = (device const float*)(w + kwoff) + (ulong)c * K;
    device const float* st = fatc(state, soff);
    device const float* hh = fatc(h, hoff);
    float acc = 0.0f;
    for (uint k = 0; k + 1u < K; k++) acc += wi[k] * st[(ulong)k * n + c];
    acc += wi[K - 1u] * hh[c];
    fat(y, yoff)[c] = acc;   // 多行时必须按行落，否则所有行都写进第 0 行
}

// state 滚动：丢掉最旧的一格，把 h(t) 放到末尾
kernel void k_conv_shift(device const float* h    [[buffer(0)]],
                         device float* state       [[buffer(1)]],
                         constant uint& n          [[buffer(2)]],
                         constant uint& K          [[buffer(3)]],
                         constant ulong& soff      [[buffer(4)]],
                         constant ulong& hoff      [[buffer(5)]],
                         uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    device float* st = fat(state, soff);
    device const float* hh = fatc(h, hoff);
    for (uint k = 0; k + 2u < K; k++) st[(ulong)k * n + gid] = st[(ulong)(k + 1u) * n + gid];
    st[(ulong)(K - 2u) * n + gid] = hh[gid];
}
)MSL";
