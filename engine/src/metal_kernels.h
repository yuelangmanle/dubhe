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
#include <metal_simdgroup_matrix>
using namespace metal;

// 一个 threadgroup 装 4 个 simdgroup → 4 行。128 线程是苹果 GPU 上
// 占用率与调度粒度的折中（M 系列每个 core 可驻留多个 128 线程的组）。
#define SG_PER_TG 4

// prefill 批大小上限。8 是寄存器压力与权重复用率的平衡点：
// 每个 lane 8 个累加器，加上解码用的临时量，仍在 255 个寄存器内。
#define BMAX 8

// 解码时一个 simdgroup 同时处理几行。
//
// 为什么不是 1：解码是**延迟受限**而不是带宽受限。一行一个 simdgroup 时，
// 每行的块循环是一条"取一个权重字节 → 一条 fma"的依赖链，单条链的
// 未完成 load 数被调度器限制住，带宽上不去。实测 Q4_K（1.05 GB）和
// Q8_0（1.76 GB）decode 都是 26 ms/token —— 前者只有 39 GB/s，后者 67 GB/s，
// 说明 Q4_K 根本没到带宽墙，是 MLP（memory level parallelism）不够。
//
// 改成一次算 MV_R 行：x 只读一遍（复用），MV_R 条独立的 load 链同时在飞，
// 同样的块循环喂饱 MV_R 倍的 outstanding load。MV_R=4 时寄存器仍然够用。
#define MV_R 2
// MSL 的程序级变量必须落在 constant 地址空间，`constexpr uint` 会直接编译失败
// （program scope variable must reside in constant address space）。
// 这里用宏：循环里写成 `r < MVR`，展开后是 `r < 4`，不会有 uint/int 比较告警。
#define MVR MV_R

// ---------------------------------------------------------------- 工具
static inline float f16_to_f32(ushort h) { return float(as_type<half>(h)); }

static inline float rd_f16(device const uchar* p) {
    ushort h = ushort(p[0]) | (ushort(p[1]) << 8);
    return f16_to_f32(h);
}

// 非对齐的 16 字节读。
//
// 为什么需要：Q4_K 的块跨度是 144（16 的倍数），所以反量化能直接
// `*(device const uint4*)` 一次读 16 字节。Q6_K 的块跨度是 210 ——
// 只保证 2 字节对齐，`uint4` 会触发对齐错误，只能退化成 16 条字节 load。
// 而实测（§9.15）字节 load 的**条数**就是 Q6_K 慢 4.4 倍的唯一原因。
//
// 做法：退到 4 字节边界读 20 字节，再按 (addr & 3) 把中间那 16 字节拼出来。
// 因为跨度的奇偶性，addr & 3 只会是 0 或 2，所以是"要不要右移 16 位"的二选一。
// 用 select 而不是 if —— 同一个 warp 里不同行的奇偶性可能不同，分支会发散。
//
// 代价：5 条 load 换 16 个元素（0.31 条/元素），比逐字节的 2 条/元素少 6 倍。
// a==0 时多读的那一个 word 落在块内（ql 之后就是 qh），是合法内存。
static inline uint4 ld16s(device const uchar* addr) {
    const uint a = uint((ulong)addr & 3ul);
    device const uint* p = (device const uint*)(addr - a);
    const uint w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3], w4 = p[4];
    const bool z = (a == 0u);
    return uint4(z ? w0 : ((w0 >> 16) | (w1 << 16)),
                 z ? w1 : ((w1 >> 16) | (w2 << 16)),
                 z ? w2 : ((w2 >> 16) | (w3 << 16)),
                 z ? w3 : ((w3 >> 16) | (w4 << 16)));
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    device const float* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = (device const float*)(w + woff) + (ulong)(row0 + (r < nok ? r : 0u)) * n_in;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) {
        const float xv = x[i];
        for (uint r = 0; r < MVR; r++) acc[r] += p[r][i] * xv;
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    device const ushort* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = (device const ushort*)(w + woff) + (ulong)(row0 + (r < nok ? r : 0u)) * n_in;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) {
        const float xv = x[i];
        for (uint r = 0; r < MVR; r++) acc[r] += f16_to_f32(p[r][i]) * xv;
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    const uint nb = n_in >> 5;
    const ulong rbytes = (ulong)nb * 18u;
    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;
    // lane 0..15 取低半字节（元素 0..15），lane 16..31 取高半字节（元素 16..31）
    const uint idx = lane & 15u;
    const bool hi  = lane >= 16u;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const float xv = x[(b << 5) + lane];
        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + b * 18u;
            const uint byte = uint(q[2u + idx]);
            const int v = hi ? int(byte >> 4) : int(byte & 0x0Fu);
            acc[r] += rd_f16(q) * float(v - 8) * xv;
        }
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    const uint nb = n_in >> 5;
    const ulong rbytes = (ulong)nb * 22u;
    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;
    const uint idx = lane & 15u;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const float xv = x[(b << 5) + lane];
        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + b * 22u;
            const uint qh = uint(q[2]) | (uint(q[3]) << 8) | (uint(q[4]) << 16) | (uint(q[5]) << 24);
            const int bit5 = int((qh >> lane) & 1u) << 4;
            const uint byte = uint(q[6u + idx]);
            const int v = (lane < 16u) ? (int(byte & 0x0Fu) | bit5)
                                       : (int(byte >> 4)    | bit5);
            acc[r] += rd_f16(q) * float(v - 16) * xv;
        }
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    const uint nb = n_in >> 5;
    const ulong rbytes = (ulong)nb * 34u;
    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const float xv = x[(b << 5) + lane];
        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + b * 34u;
            const int v = int(((device const char*)(q + 2u))[lane]);
            acc[r] += rd_f16(q) * float(v) * xv;
        }
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    const uint nb = n_in >> 8;
    const ulong rbytes = (ulong)nb * 144u;
    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    // 分工方式决定一切。
    //
    // 旧写法：一个 lane 在**每个** sub-block 里取一个元素 → 一个块要 8 次
    // k4_scale_min（每次约 10 条指令，带分支和 3 次字节读）。一个 lane 每块
    // 只算 8 个元素，却付了 80 条指令的账 → **10 条指令/元素**。
    // Q8_0 那边是 1 条指令/元素，于是 Q4_K 的 decode 只有 14.5 GB/s，
    // 而 Q8_0 能到 58 GB/s —— 差的不是带宽，是 ALU。
    //
    // 新写法：一个 lane **固定负责一个 sub-block 的 8 个元素**。
    //   s = lane/4  → 第几个 sub-block（0..7）
    //   h = lane%4  → sub-block 内第几个 8 元素组
    // 于是每块只需 1 次 k4_scale_min，8 个元素 → **约 3 条指令/元素**。
    //
    // 字节布局（GGUF Q4_K 的 qs 是 64 元素交错，不是"每个 sub-block 16 字节"）：
    //   64 元素块 k 占字节 [16+32k, 16+32k+32)，
    //   元素 64k+i        → 第 i 字节的**低**半字节
    //   元素 64k+32+i     → 第 i 字节的**高**半字节
    // 所以 sub-block 为偶数取低半字节、奇数取高半字节，两者读的是同一段 8 字节。
    const uint sb   = lane >> 2;              // 0..7
    const uint h    = lane & 3u;              // 0..3
    const uint chunk = sb >> 1;               // 64 元素块号 0..3
    const bool take_hi = (sb & 1u) != 0u;
    const uint boff = 16u + (chunk << 5) + (h << 3);   // 本 lane 的 8 字节起点
    const uint xoff = (sb << 5) + (h << 3);            // 本 lane 的 8 元素起点

    for (uint b = 0; b < nb; b++) {
        device const float* xb = x + (b << 8) + xoff;
        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + b * 144u;
            uchar s1, m1;
            k4_scale_min(int(sb), q + 4u, s1, m1);
            const float sc = rd_f16(q)      * float(s1);
            const float mn = rd_f16(q + 2u) * float(m1);
            // 8 个字节一次读进来（uint2 = 8 字节，地址 8 字节对齐：
            // 块跨度 144 与 boff = 16+32k+8h 都是 8 的倍数，GGUF 张量基址
            // 按 32 对齐，所以成立）。逐个 uchar 读是 8 条 load 指令，
            // 一条 uint2 就够 —— 解码时 load 指令数才是真正的天花板。
            const uint2 raw = *(device const uint2*)(q + boff);
            // min 项折进逐元素 fma：sum(d*sc*v - dm*m)*x 省掉单独的 x 求和
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                const uint word = (i < 4u) ? raw.x : raw.y;
                const uint byte = (word >> ((i & 3u) << 3)) & 0xFFu;
                const int v = take_hi ? int(byte >> 4) : int(byte & 0x0Fu);
                acc[r] += (sc * float(v) - mn) * xb[i];
            }
        }
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    const uint nb = n_in >> 8;
    const ulong rbytes = (ulong)nb * 210u;
    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    // 与 mv_q4_k 同一个思路：让一个 lane **固定负责一个 16 元素 scale 组里的
    // 8 个元素**，而不是"每个 32 元素段各取一个"。scale 粒度是 16 元素，
    // 32 个 lane / 16 组 = 每组 2 个 lane，正好各 8 个元素 → 每块只需 1 次
    // scale 读，旧写法要 8 次。
    //
    // Q6_K 的字节布局（ql 128B 低 4 位 + qh 64B 高 2 位 + 16 个 int8 scale + d）：
    //   元素 e 的 ql 下标 = (e & 63) + ((e >> 7) << 6)
    //   ql 取高半字节当且仅当 (e & 64) != 0
    //   元素 e 的 qh 下标 = ((e >> 7) << 5) + (e & 31)，取第 2*((e>>5)&3) 位起的 2 位
    // 因为 8 整除 16、16 整除 32，一个 lane 的 8 个元素必然落在同一个 q = e>>5 里，
    // 所以上面这些"下标/移位"全部是**每块算一次**的标量。
    const uint g16 = lane >> 1;               // 16 元素组号 0..15
    const uint q   = g16 >> 1;                // 32 元素段号 0..7
    const uint r0  = ((g16 & 1u) << 4) + ((lane & 1u) << 3);   // 段内起点
    const uint ql_off = 32u * (q & 1u) + 64u * (q >> 2) + r0;
    const uint qh_off = 32u * (q >> 2) + r0;
    const bool take_hi = (q & 2u) != 0u;
    const uint vh = 2u * (q & 3u);
    const uint xoff = (q << 5) + r0;

    for (uint b = 0; b < nb; b++) {
        device const float* xb = x + (b << 8) + xoff;
        for (uint r = 0; r < MVR; r++) {
            device const uchar* q_ = p[r] + b * 210u;
            device const uchar* ql = q_ + ql_off;
            device const uchar* qh = q_ + 128u + qh_off;
            const float sc = rd_f16(q_ + 208u) * float(((device const char*)(q_ + 192u))[g16]);
            // Q6_K 的块跨度是 210 字节（≡ 2 mod 4），奇数块拿不到 4 字节对齐，
            // 所以只能按 ushort 向量化：8 个字节 4 条 load，而不是 8 条 uchar。
            // 解码时 load 指令数就是天花板，这一条把 ql/qh 的 load 数砍了一半。
            ushort qlw[4], qhw[4];
            #pragma unroll
            for (uint i = 0; i < 4u; i++) {
                qlw[i] = ((device const ushort*)ql)[i];
                qhw[i] = ((device const ushort*)qh)[i];
            }
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                const uint lb = (uint(qlw[i >> 1]) >> ((i & 1u) << 3)) & 0xFFu;
                const uint hb = (uint(qhw[i >> 1]) >> ((i & 1u) << 3)) & 0xFFu;
                const uint lo = take_hi ? (lb >> 4) : (lb & 0x0Fu);
                const uint hi2 = (hb >> vh) & 3u;
                const int v = int(lo | (hi2 << 4)) - 32;
                acc[r] += sc * float(v) * xb[i];
            }
        }
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
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
                   constant ulong& yoff       [[buffer(7)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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
                   constant ulong& yoff       [[buffer(7)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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
                    constant ulong& yoff      [[buffer(7)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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
                    constant ulong& yoff      [[buffer(7)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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
                    constant ulong& yoff      [[buffer(7)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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
                    constant ulong& yoff      [[buffer(7)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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
                    constant ulong& yoff      [[buffer(7)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgid * SG_PER_TG + sg;
    if (row >= n_out) return;
    Y += yoff;   // 分批派发时指向本批第 0 个 token 的起始行
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

// ================================================================ 二维分块 matmul
//
// 上面 mm_* 系列一个 simdgroup 只算**一行**输出，于是同一块激活被 n_out 个
// simdgroup 各读一遍。ffn_up 的激活只有 64 KB，却被读 7680 次，折合每个
// batch 491 MB 的激活流量（全 prefill 约 5.1 TB）。这些流量大部分命中 L2，
// 但 L2 带宽也是有限的：按 600 GB/s 算正好 8 秒 —— 与实测 prefill 里
// GEMM 占的 8.1 秒吻合。所以方向是"减少同一份激活的重复读取"。
//
// 新结构（mm2_*）：
//   * 一个 threadgroup = MM_SG_PER_TG 个 simdgroup × MM_SG 行 = MMR(32) 行输出。
//     组内 8 个 simdgroup 读的是同一片激活 → 激活流量除以 32。
//   * K 方向按 TILE_K(256) 分片，一个片正好是一个 Q4_K/Q6_K 块。
//   * lane 固定负责 8 个**连续**元素（e0 = lane*8）。因为 8 整除 16/32/256，
//     一个 lane 的 8 个元素必然落在同一个 scale 组里 → scale 解包、ql/qh
//     下标、移位量全部退化成"每块算一次"的标量（旧写法每块要算 8 次）。
//
// 激活**不经过 threadgroup memory**，直接读 device 内存。这一点违反直觉，
// 但实测如此：每个 lane 只读自己那 8 个元素的激活，从不碰别人的数据，所以
// threadgroup memory 唯一的作用只是"把搬运工作摊给 256 个线程"。去掉它、
// 靠 L1/L2 兜住同组内 8 个 simdgroup 的重复读，裸内核 560 → 653 GFLOP/s，
// 静态 threadgroup 内存归零、占用率反而上升。
//
// 本轮试过但**更慢**的方案（都实测过，别再走一遍）：
//   * threadgroup memory 显式 staging + 每片一次 barrier：560 GFLOP/s
//   * 上面的 staging 再加双缓冲（tgmem 16 KB）：518 GFLOP/s —— tgmem 翻倍
//     把每 core 的 threadgroup 数从 4 压到 2，占用率损失大于延迟收益
//   * BMAX 16（权重复用翻倍）：203 GFLOP/s —— 累加器翻倍直接溢出寄存器
//   * MM_SG 8（MMR=64，更少 threadgroup）：181 GFLOP/s —— 同上
//   * simdgroup_matrix：M4 上 8x8x8 MMA 只有 2.45 TFLOP/s，**低于** float4
//     FMA 的 2.84 —— 苹果是标量模拟，不是硬件矩阵单元（见 tools/mma_probe.mm）
//   * fp16 FMA：3.11 vs 2.62 TFLOP/s，只有 1.19x，不值得换精度
//
// 约束：n_in 必须是 TILE_K 的整数倍（分块解码依赖完整的 256 元素块），
// 否则派发侧回退到旧的 mm_* 内核。
#define MM_SG_PER_TG 8                  // 每 threadgroup 的 simdgroup 数（256 线程）
#define MM_SG        4                  // 每 simdgroup 算几行
#define MMR          (MM_SG_PER_TG * MM_SG)   // 每 threadgroup 的输出行数 = 32
#define TILE_K       256

// 激活按 float4 读，所以 BMAX 必须是 4 的倍数；
// lane 负责 8 个连续元素，所以一片 K 至少要 32×8 = 256 个元素。
static_assert(BMAX % 4 == 0, "BMAX 必须是 4 的倍数");
static_assert(TILE_K >= 256, "TILE_K 必须 >= 32 lane × 8 元素");

// ================================================================ 权重预取
//
// 为什么要把"读"和"算"拆成两半
// ----------------------------
// 一片 K（TILE_K=256 元素）只有 8×MM_SG×BMAX = 256 个 fma 的活儿，
// 折合每 lane 约 256 拍。而一次没命中的 DRAM 往返要 600 拍以上。
// 原来的写法是"读权重 → 立刻用"，于是每个 b 循环开头都要干等一次往返，
// 实测 issue 效率只有 33%（693 GFLOP/s / 2.1 TFLOP/s 的可达值）。
//
// 把下一片的原始字提前读进寄存器，当前片算的时候加载就在飞 —— 延迟被
// 完全隐藏，代价是每行多占 2–8 个寄存器。
//
// 寄存器的红线在 64 附近：acc(MM_SG×BMAX=32) + wv(MM_SG×8=32) 已经 64，
// 再往上就越过 64 这条线，occupancy 掉一半。实测 MM_SG=6（96 个）直接
// 从 672 掉到 236 GFLOP/s。所以预取只能压得很省：只预取"真正大块的那段
// 负载"，scale/d 这些小字段留在 CV 里现读（同一 simdgroup 的 32 个 lane
// 读的是同一 16 字节，一条事务就够，基本都命中 L1）。

struct RawQ4K { uint2 q; };            // qs 8 字节
struct RawQ6K { ushort ql[4]; ushort qh[4]; };  // ql 8 + qh 8
struct RawQ8  { ushort q[4]; };        // 8 字节
struct RawF16 { uint4 q; };            // 16 字节
struct RawF32 { uint4 q; };            // 只预取前 16 字节，后 16 字节在 CV 里补

static inline RawQ4K ld_q4k(device const uchar* q, uint lane) {
    RawQ4K r;
    r.q = *(device const uint2*)(q + 16u + ((lane >> 3) << 5) + ((lane & 3u) << 3));
    return r;
}

static inline RawQ6K ld_q6k(device const uchar* q, uint lane) {
    const uint qq = lane >> 2;
    const uint r0 = (lane & 3u) << 3;
    device const ushort* qlp =
        (device const ushort*)(q + 32u * (qq & 1u) + 64u * (qq >> 2) + r0);
    device const ushort* qhp =
        (device const ushort*)(q + 128u + 32u * (qq >> 2) + r0);
    RawQ6K r;
#pragma unroll
    for (uint i = 0; i < 4u; i++) { r.ql[i] = qlp[i]; r.qh[i] = qhp[i]; }
    return r;
}

static inline RawQ8 ld_q8_0(device const uchar* q, uint lane) {
    // 块跨度 34 ≡ 2 (mod 4)，起始地址只有 2 字节对齐 → 只能按 ushort 读，
    // 用 char4/uint2 会踩未对齐（这个坑在 mm2_q8_0 第一版上踩过）。
    device const ushort* up = (device const ushort*)
        (q + (lane >> 2) * 34u + 2u + ((lane & 3u) << 3));
    RawQ8 r;
#pragma unroll
    for (uint i = 0; i < 4u; i++) r.q[i] = up[i];
    return r;
}

static inline RawF16 ld_f16(device const uchar* q, uint lane) {
    RawF16 r;
    r.q = *(device const uint4*)(q + ((lane << 3) << 1));
    return r;
}

static inline RawF32 ld_f32(device const uchar* q, uint lane) {
    RawF32 r;
    r.q = *(device const uint4*)(q + ((lane << 3) << 2));
    return r;
}

// ---------------------------------------------------------------- 换算
// 约定：wv[0..7] 是本 lane 那 8 个元素的解量化结果（f32）。

static inline void cv_q4k(device const uchar* q, RawQ4K raw, uint lane,
                          thread float* wv) {
    const float d_  = rd_f16(q);
    const float dm_ = rd_f16(q + 2u);
    uchar s1_, m1_;
    k4_scale_min(int(lane >> 2), q + 4u, s1_, m1_);
    const float sc_ = d_ * float(s1_);
    const float mn_ = dm_ * float(m1_);
    const uint sh_ = ((lane >> 2) & 1u) << 2;
#pragma unroll
    for (uint i_ = 0; i_ < 8u; i_++) {
        const uint word_ = (i_ < 4u) ? raw.q.x : raw.q.y;
        const uint v_ = (word_ >> (((i_ & 3u) << 3) + sh_)) & 0x0Fu;
        wv[i_] = fma(sc_, float(v_), -mn_);
    }
}

static inline void cv_q6k(device const uchar* q, RawQ6K raw, uint lane,
                          thread float* wv) {
    const uint qq_ = lane >> 2;
    const float d_ = rd_f16(q + 208u) * float(((device const char*)(q + 192u))[lane >> 1]);
    const uint sh_ = ((qq_ & 2u) != 0u) << 2;
    const uint vh_ = 2u * (qq_ & 3u);
#pragma unroll
    for (uint i_ = 0; i_ < 8u; i_++) {
        const uint lb_ = (uint(raw.ql[i_ >> 1]) >> (((i_ & 1u) << 3) + sh_)) & 0x0Fu;
        const uint hb_ = (uint(raw.qh[i_ >> 1]) >> ((i_ & 1u) << 3)) & 0xFFu;
        const int v_ = int(lb_ | (((hb_ >> vh_) & 3u) << 4)) - 32;
        wv[i_] = d_ * float(v_);
    }
}

static inline void cv_q8_0(device const uchar* q, RawQ8 raw, uint lane,
                           thread float* wv) {
    const float d_ = rd_f16(q + (lane >> 2) * 34u);
#pragma unroll
    for (uint i_ = 0; i_ < 4u; i_++) {
        const uint w_ = uint(raw.q[i_]);
        wv[i_ * 2u]     = d_ * float(int((char)(w_ & 0xFFu)));
        wv[i_ * 2u + 1] = d_ * float(int((char)(w_ >> 8)));
    }
}

static inline void cv_f16(device const uchar* q, RawF16 raw, uint lane,
                          thread float* wv) {
    (void)q; (void)lane;
    const float2 a_ = float2(as_type<half2>(raw.q.x));
    const float2 b_ = float2(as_type<half2>(raw.q.y));
    const float2 c_ = float2(as_type<half2>(raw.q.z));
    const float2 e_ = float2(as_type<half2>(raw.q.w));
    wv[0] = a_.x; wv[1] = a_.y; wv[2] = b_.x; wv[3] = b_.y;
    wv[4] = c_.x; wv[5] = c_.y; wv[6] = e_.x; wv[7] = e_.y;
}

static inline void cv_f32(device const uchar* q, RawF32 raw, uint lane,
                          thread float* wv) {
    (void)lane;
    // 后半段 16 字节在预取阶段没读，这里补上。F32 权重是 f16 的两倍、
    // Q4_K 的八倍，本来就不是 prefill 的主力格式，不值得为它多占寄存器。
    const float4 a_ = as_type<float4>(raw.q);
    const float4 b_ = *(device const float4*)(q + ((lane << 3) << 2) + 16u);
    wv[0] = a_.x; wv[1] = a_.y; wv[2] = a_.z; wv[3] = a_.w;
    wv[4] = b_.x; wv[5] = b_.y; wv[6] = b_.z; wv[7] = b_.w;
}

// ---------------------------------------------------------------- 内核主体
//   RB = 单行字节跨度（p[r] 用）
//   TB = 每片 K 的字节跨度（块循环里 b*TB 用）。注意 TB 与 RB 不是一回事：
//        256 元素一片对 F32 是 1024 字节，对 Q4_K 是 144 字节；把它们
//        混成一个量会让 b*RB 直接跳出整行（这个 bug 让第一版结果全错）。
#define MM2_KERNEL(NAME, RT, LD, CV, RB, TB)                                  \
kernel void NAME(device const uchar* w      [[buffer(0)]],                     \
                 device const float* XT     [[buffer(1)]],                     \
                 device float* Y            [[buffer(2)]],                     \
                 constant uint& n_in        [[buffer(3)]],                     \
                 constant ulong& woff       [[buffer(4)]],                     \
                 constant uint& n_out       [[buffer(5)]],                     \
                 constant uint& nrow        [[buffer(6)]],                     \
                 constant ulong& yoff       [[buffer(7)]],                     \
                 uint tgid [[threadgroup_position_in_grid]],                    \
                 uint tid  [[thread_position_in_threadgroup]],                  \
                 uint lane [[thread_index_in_simdgroup]],                       \
                 uint sg   [[simdgroup_index_in_threadgroup]])                  \
{                                                                              \
    (void)0;                                                                   \
    Y += yoff;                                                                 \
    const uint row0 = tgid * MMR;                                              \
    const uint nb   = n_in / TILE_K;                                           \
    const uint e0   = lane << 3;                                               \
                                                                               \
    uint row[MM_SG];                                                           \
    device const uchar* p[MM_SG];                                              \
    float acc[MM_SG][BMAX];                                                    \
    RT rw[MM_SG];                                                              \
    _Pragma("unroll")                                                          \
    for (uint r = 0; r < MM_SG; r++) {                                         \
        row[r] = row0 + sg * MM_SG + r;                                        \
        p[r] = w + woff + (ulong)min(row[r], n_out - 1u) * (ulong)(RB);        \
        _Pragma("unroll")                                                      \
        for (uint j = 0; j < BMAX; j++) acc[r][j] = 0.0f;                      \
        rw[r] = LD(p[r], lane);                                                \
    }                                                                          \
                                                                               \
    for (uint b = 0; b < nb; b++) {                                            \
                                                                               \
        /* 先发下一片的加载，再算当前片 —— 这一行的位置就是全部收益所在。 */    \
        /* 用 min() 夹住下标而不是三元判断：三元可能让加载变成条件执行。实测    \
           两者在端到端上打平（机器噪声内），但夹下标少一条分支、代码更直白，   \
           所以留这个。夹住之后末片会被多读一次，读的是合法内存，值不用。 */    \
        RT rn[MM_SG];                                                          \
        const ulong bnext = (ulong)min(b + 1u, nb - 1u);                       \
        _Pragma("unroll")                                                      \
        for (uint r = 0; r < MM_SG; r++)                                       \
            rn[r] = LD(p[r] + bnext * (ulong)(TB), lane);                      \
                                                                               \
        float wv[MM_SG][8];                                                    \
        _Pragma("unroll")                                                      \
        for (uint r = 0; r < MM_SG; r++)                                       \
            CV(p[r] + (ulong)b * (ulong)(TB), rw[r], lane, wv[r]);              \
                                                                               \
        /* 8 个元素：先把这一格的 BMAX 个激活读进寄存器，再摊给 MM_SG 行。       \
           读一次激活做 MM_SG×BMAX 次 fma —— 这就是分块的全部收益。 */          \
        _Pragma("unroll")                                                      \
        for (uint i = 0; i < 8u; i++) {                                        \
            const device float4* xq_ =                                         \
                (const device float4*)(XT + (ulong)b * (TILE_K * BMAX) + (e0 + i) * BMAX); \
            float4 xa_[BMAX / 4];                                              \
            _Pragma("unroll")                                                  \
            for (uint t = 0; t < BMAX / 4u; t++) xa_[t] = xq_[t];               \
            _Pragma("unroll")                                                  \
            for (uint r = 0; r < MM_SG; r++) {                                 \
                _Pragma("unroll")                                              \
                for (uint t = 0; t < BMAX / 4u; t++) {                         \
                    acc[r][t*4+0] += wv[r][i] * xa_[t].x;                      \
                    acc[r][t*4+1] += wv[r][i] * xa_[t].y;                      \
                    acc[r][t*4+2] += wv[r][i] * xa_[t].z;                      \
                    acc[r][t*4+3] += wv[r][i] * xa_[t].w;                      \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (uint r = 0; r < MM_SG; r++) rw[r] = rn[r];                        \
    }                                                                          \
                                                                               \
    _Pragma("unroll")                                                          \
    for (uint r = 0; r < MM_SG; r++) {                                         \
        _Pragma("unroll")                                                      \
        for (uint j = 0; j < BMAX; j++) {                                      \
            const float s_ = simd_sum(acc[r][j]);                              \
            if (lane == 0u && row[r] < n_out)                                  \
                Y[(ulong)j * n_out + row[r]] = s_;                             \
        }                                                                      \
    }                                                                          \
}

MM2_KERNEL(mm2_f32,  RawF32, ld_f32,  cv_f32,  (n_in * 4u),         (TILE_K * 4u))
MM2_KERNEL(mm2_f16,  RawF16, ld_f16,  cv_f16,  (n_in * 2u),         (TILE_K * 2u))
MM2_KERNEL(mm2_q8_0, RawQ8,  ld_q8_0, cv_q8_0, ((n_in >> 5) * 34u), ((TILE_K >> 5) * 34u))
MM2_KERNEL(mm2_q4_k, RawQ4K, ld_q4k,  cv_q4k,  ((n_in >> 8) * 144u), 144u)
MM2_KERNEL(mm2_q6_k, RawQ6K, ld_q6k,  cv_q6k,  ((n_in >> 8) * 210u), 210u)

// ================================================================ MMA 分块 matmul（prefill，第 3 代）
//
// 为什么推倒重写 mm2_*
// -------------------
// mm2 把反量化结果留在**寄存器**里：wv[MM_SG][8] 占 32 个，累加器
// acc[MM_SG][BMAX] 又占 32 个，一共 64+ —— 正好压在寄存器红线上，occupancy
// 被砍一半。真实模型实测只有 680 GFLOP/s，是本机裸 FMA 峰值 3.1 TFLOP/s 的 22%。
//
// 当时的结论是"再复用更多就净亏"，于是所有"提高复用率"的方向（MM_SG↑、
// BMAX↑、threadgroup 暂存）都被否掉了。**顺序反了**：应该先换掉累加器的
// 表示方式，再谈复用。换成 MMA 之后：
//
//   * 累加器是 simdgroup_float8x8，一个 8x8 块只占 2 个寄存器/lane
//   * 反量化结果写 threadgroup memory（half），不占寄存器
//   * 一条 MMA 指令 = 8x8x8 = 512 次乘加 = 16 条 fma/lane
//
// 同一台机器上 llama.cpp（走 simdgroup_float8x8）跑 882 tok/s、MLX 840，
// 我们 184。差距不在硬件也不在指令集，就在这个结构上。
//
// 分块
// ----
//   每个 threadgroup：BN=64 个输出行 × BM=32 个 token，K 方向每次 32 元素。
//   4 个 simdgroup 各管 16 个输出行（2 个 8x8 累加块）× 4 个 token 块。
//   128 线程，每行 2 个线程，每个线程反量化 16 个元素 ——
//   正好是 Q4_K 一个 32 元子块的一半，子块的 scale/min 只算一次。
//
// 为什么 BM 从 8 提到 32
// ---------------------
// mm2 时代 BMAX=8 是被**寄存器**逼出来的：acc[MM_SG][8] 占 32 个，wv[MM_SG][8]
// 再占 32 个，64 就是红线。换 MMA 之后累加器只要 2 个寄存器/lane/8x8 块，
// 于是"摊薄倍数"第一次可以单独调。权重反量化的代价是每 256 元素块算一次
// scale/min，摊到 8 个 token 上是 1/8，摊到 32 个上是 1/32 —— 而 prefill
// 的真实瓶颈正是这个摊薄倍数（把反量化向量化后零增益，就是这个结论的反证）。
//
// threadgroup 内存：权重 64×32 half = 4 KB，激活 32×32 half = 2 KB，
// 输出暂存 32×64 float = 8 KB，合计 14 KB —— 每 core 还能驻留 2 个。
//
// 激活存成 [k][m]（k 行主序、m 连续），让每个线程一次 float4 写四个 token；
// 读的时候给 simdgroup_load 打开 transpose，语义等价于 [m][k]。
// 反过来存 [m][k] 就要 4 次标量 half 写，白扔一条向量指令。

#define MMA_BN 64
#define MMA_BM 32
#define MMA_NK 32

kernel void mm3_q4_k(device const uchar* w      [[buffer(0)]],
                     device const float* XT     [[buffer(1)]],   // [n_in][xstride] 已转置
                     device float* Y            [[buffer(2)]],   // [xstride][n_out]
                     constant uint& n_in        [[buffer(3)]],
                     constant ulong& woff       [[buffer(4)]],
                     constant uint& n_out       [[buffer(5)]],
                     constant uint& nrow        [[buffer(6)]],
                     constant uint& xstride     [[buffer(7)]],
                     constant ulong& yoff       [[buffer(8)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid  [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]])
{
    // threadgroup 内存重叠使用：ws/xs 只在 K 循环里活着，cs 只在循环结束后
    // 活着，两者永不同时存在。合起来 14 KB → 8 KB —— 每 core 能多驻留一个
    // 组，而 prefill 恰恰是**延迟受限**的（反量化的 load 在等 device memory），
    // 多一组就多一倍 latency hiding。8 KB 是按 cs（32×64 float）算的。
    threadgroup half smem[MMA_BM * MMA_BN * 2];        // 4096 halfs = 8 KB
    threadgroup half* ws = smem;                       // 权重：反量化后的 [n][k]
    threadgroup half* xs = smem + MMA_BN * MMA_NK;     // 激活：[k][m]

    const uint row0 = tgid * MMA_BN;
    const uint nb   = n_in / MMA_NK;
    const ulong rbytes = (ulong)(n_in >> 8) * 144u;   // Q4_K：每行 144 字节/256 元素

    // 本线程负责第 lr 行的第 16*half16 .. +16 个元素
    const uint lr     = tid >> 1;          // 0..63
    const uint half16 = tid & 1u;          // 0 / 1
    const uint grow   = row0 + lr;
    device const uchar* prow = w + woff + (ulong)min(grow, n_out - 1u) * rbytes;

    // 本 simdgroup 负责 16 个输出行 × 32 个 token = 2×4 个 8x8 累加块
    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2u; i++)
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint b = 0; b < nb; b++) {
        const uint k0 = b * MMA_NK;

        // ---- 反量化 16 个元素进 ws[lr][16*half16 ..]
        {
            const uint e0    = k0 + (half16 << 4);   // 全局元素下标
            const uint inblk = e0 & 255u;            // 块内下标
            const uint chunk = inblk >> 6;           // 0..3
            const uint inch  = inblk & 63u;          // 0/16/32/48
            const bool hi    = inch >= 32u;          // 取高半字节还是低半字节
            const uint sb    = inblk >> 5;           // 子块 0..7
            device const uchar* q = prow + (ulong)(e0 >> 8) * 144u;

            uchar s1, m1;
            k4_scale_min(int(sb), q + 4u, s1, m1);
            const float sc = rd_f16(q)      * float(s1);
            const float mn = rd_f16(q + 2u) * float(m1);

            // 16 个连续字节：chunk 内偏移 inch&31 起，全部同一种半字节。
            // 一次 uint4 读进来（16 字节，8 字节对齐成立：块跨度 144、
            // 偏移 16+32k+16h 都是 8 的倍数），再按 nibble 摊开。
            // 逐个 uchar 读是 16 条 load 指令，这里只要 1 条。
            const uint4 raw = *(device const uint4*)(q + 16u + (chunk << 5) + (inch & 31u));
            const uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            const uint sh = hi ? 4u : 0u;
            const uint4 o4 = uint4(lr * MMA_NK + (half16 << 4));

            half4 h4[4];
            #pragma unroll
            for (uint t = 0; t < 4u; t++) {
                #pragma unroll
                for (uint j = 0; j < 4u; j++) {
                    const uint byte = (words[t] >> (j << 3)) & 0xFFu;
                    const uint v = (byte >> sh) & 0x0Fu;
                    h4[t][j] = (half)fma(sc, float(v), -mn);
                }
            }
            // half4 直接存，避免 16 次标量 half 写
            threadgroup half4* dst = (threadgroup half4*)(ws + (o4.x & ~3u));
            #pragma unroll
            for (uint t = 0; t < 4u; t++) dst[t] = h4[t];
        }

        // ---- 激活搬进 xs：32 token × 32 元素 = 1024 个，128 线程每人 8 个。
        // 一次 float4 读四个 token（m 在 XT 里连续），一次 half4 写进 xs。
        // 越界 token 已由 k_trans 填 0，这里不必再判。
        #pragma unroll
        for (uint t = 0; t < 2u; t++) {
            const uint i4 = tid * 2u + t;       // 0..255
            const uint m0 = (i4 & 7u) << 2;     // 0,4,...,28
            const uint k  = i4 >> 3;            // 0..31
            const float4 v = *(device const float4*)(XT + (ulong)(k0 + k) * xstride + m0);
            *(threadgroup half4*)(xs + k * MMA_BM + m0) =
                half4((half)v.x, (half)v.y, (half)v.z, (half)v.w);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 4 个 K 子步 × 2 个行块 × 4 个 token 块 = 32 条 MMA
        // ma 从 xs 取：xs 是 [k][m]，所以 origin 的 x 是 token 偏移、y 是 K 偏移，
        // 再开 transpose 把 [k][m] 读成 [m][k]。
        #pragma unroll
        for (uint kk = 0; kk < 4u; kk++) {
            simdgroup_half8x8 ma[4];
            #pragma unroll
            for (uint j = 0; j < 4u; j++)
                simdgroup_load(ma[j], xs, MMA_BM, ulong2(j << 3, kk << 3), true);

            simdgroup_half8x8 mb[2];
            #pragma unroll
            for (uint i = 0; i < 2u; i++)
                simdgroup_load(mb[i], ws, MMA_NK, ulong2(kk << 3, (sg << 4) + (i << 3)), true);

            #pragma unroll
            for (uint i = 0; i < 2u; i++) {
                #pragma unroll
                for (uint j = 0; j < 4u; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[j], mb[i], acc[i][j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- 写回：先落到 threadgroup，再按边界拷到 Y（n_out 不一定是 8 的倍数）
    // 循环末尾那道 barrier 已经保证 ws 不再被读，这里可以安全地把它当 cs 用。
    threadgroup float* cs = (threadgroup float*)smem;
    #pragma unroll
    for (uint i = 0; i < 2u; i++) {
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            simdgroup_store(acc[i][j], cs, MMA_BN,
                            ulong2((sg << 4) + (i << 3), j << 3), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint t = 0; t < 16u; t++) {
        const uint idx = tid * 16u + t;
        const uint m = idx >> 6;          // 0..31
        const uint n = idx & 63u;         // 0..63
        if (m < nrow && row0 + n < n_out)
            Y[yoff + (ulong)m * n_out + row0 + n] = cs[m * MMA_BN + n];
    }
}

// ---------------------------------------------------------------- Q6_K
//
// Q6_K 块：256 元素 / 210 字节 —— ql[128]、qh[64]、sc[16]（int8）、d(fp16)@208。
//
// 6 bit 的拼法是 ggml 定的顺序，别自己发明：
//   256 块分成两个 128 的半区 h = inblk >> 7，每半区 4 组 × 32 元素，
//   组号 l0 = (inblk & 127) >> 5。元素值 q = 低 4 位 | 高 2 位 << 4：
//     * 低 4 位：l0 为偶数取 ql[l]，奇数取 ql[l + 32]；
//                l0 < 2 取低半字节，l0 >= 2 取高半字节。
//     * 高 2 位：qh[l] 右移 l0*2 位。
//   一个 32 元组的 scale 下标 = h*8 + l0*2 + (j >> 4)，j 是组内下标。
//
// 为什么这个内核比 Q4_K 那个更值：Q4_K_M 里 ffn_down / attn_v 是 Q6_K，
// 约占权重的 23%。没有这个内核时它们走"32 行拆 4 批"的老路 —— 权重被
// 读 4 遍，等于把 23% 的流量放大成 92%。
//
// 当前状态：**没启用**（metal.mm 的 kM3Names 里是空串）。结果正确，但慢。
// -------------------------------------------------------------------
// 实测（1.7B Q4_K_M，5 token 提示词，交错 3 轮一致）：
//   只开 Q4_K 的 MMA   prefill  55 ms
//   再开 Q6_K 的 MMA   prefill 242 ms   ← 慢 4.4 倍，而输出逐字一致
//
// 二分过程（每一步都重新编译、交错测）：
//   1. 反量化砍到只剩一条字节 load          → 242 ms（没变）
//   2. 反量化砍到零访存                     →  52 ms（恢复）
//   3. 地址从 q[qlo+lane] 改成 warp 内统一的 q[qlo] → 241 ms（没变）
//
// 结论：不是合并访问的问题，是**每条 load 本身的发射/延迟成本**。
// Q6_K 每元素要 2 条字节 load；Q4_K 是每 16 元素 1 条 uint4 —— 差 32 倍。
// 6 bit 的打包方式（低 4 位在 ql、高 2 位在 qh）天生不适合逐元素抽取。
//
// 救回来的唯一办法是**对齐修正后的 uint4 读**：块跨度 210 只保证 2 字节
// 对齐，所以要先读 q-2 处的 uint4，再按 (q&3) 移位拼出 16 字节。收益
// 上限约 prefill 1.5x（Q6_K 权重被读 4 遍 ≈ 多花 18 ms）。留作下一步。
//
// 线程分工（保持现状，等向量化之后再回来调）
// ------------------------------------------
// lane 就是 32 元组内的元素下标：一个 warp 一次处理一行，32 条 lane 读
// ql[lane]、qh[lane] 共 32 个连续字节；4 个 warp 并行 4 行，16 轮走完
// 64 行。d 和 scale 是 warp 内广播。
kernel void mm3_q6_k(device const uchar* w      [[buffer(0)]],
                     device const float* XT     [[buffer(1)]],
                     device float* Y            [[buffer(2)]],
                     constant uint& n_in        [[buffer(3)]],
                     constant ulong& woff       [[buffer(4)]],
                     constant uint& n_out       [[buffer(5)]],
                     constant uint& nrow        [[buffer(6)]],
                     constant uint& xstride     [[buffer(7)]],
                     constant ulong& yoff       [[buffer(8)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid  [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]])
{
    // 与 Q4_K 版同样的重叠布局：14 KB → 8 KB
    threadgroup half smem[MMA_BM * MMA_BN * 2];
    threadgroup half* ws = smem;
    threadgroup half* xs = smem + MMA_BN * MMA_NK;

    const uint row0 = tgid * MMA_BN;
    const uint nb   = n_in / MMA_NK;
    const ulong rbytes = (ulong)(n_in >> 8) * 210u;

    const uint lr     = tid >> 1;          // 0..63
    const uint half16 = tid & 1u;          // 0 / 1
    device const uchar* prow =
        w + woff + (ulong)min(row0 + lr, n_out - 1u) * rbytes;

    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2u; i++)
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint b = 0; b < nb; b++) {
        const uint k0 = b * MMA_NK;

        // ---- 反量化 16 个元素进 ws[lr][16*half16 ..]
        {
            const uint inblk = k0 & 255u;             // 块内偏移
            const uint h128  = inblk >> 7;            // 半区 0/1
            const uint l0    = (inblk & 127u) >> 5;   // 组号 0..3
            const uint hif   = (l0 >= 2u) ? 4u : 0u;  // ql 取高半字节还是低半字节
            const uint sh    = (l0 >> 1) << 1;        // qh 的移位 0/2/4/6
            const uint qlo   = h128 * 64u + ((l0 & 1u) ? 32u : 0u) + (half16 << 4);
            const uint qho   = 128u + h128 * 32u + (half16 << 4);
            device const uchar* q = prow + (ulong)(k0 >> 8) * 210u;

            const float d  = rd_f16(q + 208u);
            const float sc = d * float(int(
                ((device const char*)(q + 192u))[h128 * 8u + l0 * 2u + half16]));

            const uint4 qlv = ld16s(q + qlo);
            const uint4 qhv = ld16s(q + qho);
            threadgroup half* dst = ws + lr * MMA_NK + (half16 << 4);

            #pragma unroll
            for (uint i = 0; i < 16u; i++) {
                const uint qb = (qlv[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
                const uint qh = (qhv[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
                const uint v  = ((qb >> hif) & 0x0Fu) | (((qh >> sh) & 3u) << 4);
                dst[i] = (half)(sc * float(int(v) - 32));
            }
        }

        // ---- 激活搬进 xs（与 Q4_K 版相同）
        #pragma unroll
        for (uint t = 0; t < 2u; t++) {
            const uint i4 = tid * 2u + t;
            const uint m0 = (i4 & 7u) << 2;
            const uint k  = i4 >> 3;
            const float4 v = *(device const float4*)(XT + (ulong)(k0 + k) * xstride + m0);
            *(threadgroup half4*)(xs + k * MMA_BM + m0) =
                half4((half)v.x, (half)v.y, (half)v.z, (half)v.w);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 4u; kk++) {
            simdgroup_half8x8 ma[4];
            #pragma unroll
            for (uint j = 0; j < 4u; j++)
                simdgroup_load(ma[j], xs, MMA_BM, ulong2(j << 3, kk << 3), true);

            simdgroup_half8x8 mb[2];
            #pragma unroll
            for (uint i = 0; i < 2u; i++)
                simdgroup_load(mb[i], ws, MMA_NK, ulong2(kk << 3, (sg << 4) + (i << 3)), true);

            #pragma unroll
            for (uint i = 0; i < 2u; i++) {
                #pragma unroll
                for (uint j = 0; j < 4u; j++)
                    simdgroup_multiply_accumulate(acc[i][j], ma[j], mb[i], acc[i][j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 循环末尾的 barrier 保证 ws/xs 已不再被读，这里把同一块内存当 cs 用
    threadgroup float* cs = (threadgroup float*)smem;
    #pragma unroll
    for (uint i = 0; i < 2u; i++) {
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            simdgroup_store(acc[i][j], cs, MMA_BN,
                            ulong2((sg << 4) + (i << 3), j << 3), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint t = 0; t < 16u; t++) {
        const uint idx = tid * 16u + t;
        const uint m = idx >> 6;
        const uint n = idx & 63u;
        if (m < nrow && row0 + n < n_out)
            Y[yoff + (ulong)m * n_out + row0 + n] = cs[m * MMA_BN + n];
    }
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

// 激活转置：[nrow][n] → [n][stride]，src 从第 j0 行开始取，越界位置填 0。
// 填 0 是为了让 matmul 内核可以无条件做满 stride 次 fma，去掉所有分支。
//
// 为什么 stride 要变成运行期参数：两代内核的 token 摊薄倍数不一样。
// mm2_* 一次吃 BMAX=8 个 token，mm3_* 吃 MMA_BM=32 个 —— 同一个 g_bxt
// 缓冲要用两种布局，所以步长和行偏移都不能再写死。
kernel void k_trans(device const float* src [[buffer(0)]],
                    device float* dst       [[buffer(1)]],
                    constant uint& n        [[buffer(2)]],
                    constant uint& nrow     [[buffer(3)]],
                    constant uint& j0       [[buffer(4)]],
                    constant uint& stride   [[buffer(5)]],
                    uint e [[thread_position_in_grid]])
{
    if (e >= n) return;
    for (uint j = 0; j < stride; j++)
        dst[(ulong)e * stride + j] = (j0 + j < nrow) ? src[(ulong)(j0 + j) * n + e] : 0.0f;
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

// ---------------------------------------------------------------- C++ 侧常量
//
// 上面 MSL 是字符串，里面的 #define 对 C++ 不可见。这里给出**唯一真源**：
// metal.mm 的派发参数和 tools 里的对拍工具都从这里取，避免两边漂移。
namespace dubhe_msl {

constexpr int kBmax      = 8;    // 与 MSL 的 BMAX 一致（mm2_* 的 token 摊薄）
constexpr int kMmaBm     = 32;   // 与 MSL 的 MMA_BM 一致（mm3_* 的 token 摊薄）
constexpr int kMmaBn     = 64;   // 与 MSL 的 MMA_BN 一致
constexpr int kMmaNk     = 32;   // 与 MSL 的 MMA_NK 一致
constexpr int kSgPerTg   = 4;    // 与 MSL 的 SG_PER_TG 一致（matvec 用）
constexpr int kMvRows    = 2;    // 与 MSL 的 MV_R 一致
constexpr int kMmSgPerTg = 8;    // 与 MSL 的 MM_SG_PER_TG 一致
constexpr int kMmRows    = 32;   // 与 MSL 的 MMR 一致
constexpr int kMmTileK   = 256;  // 与 MSL 的 TILE_K 一致

}  // namespace dubhe_msl
