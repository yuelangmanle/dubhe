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
//
// MV_R 现在由 C++ 侧在编译着色器前注入（见 metal.mm 的 g_mv_rows），
// 所以这里的 2 只是"没注入时的兜底"。改这个数**不会**改变行为 ——
// 要调就得用 DUBHE_MV_R 环境变量或模型级 autotune。
#ifndef MV_R
#define MV_R 2
#endif
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

    // 一条 load 换 4 次乘加。
    //
    // 原来写 `p[r][i]`，一个 lane 一次只读 4 字节 —— 每 4 字节付一条 load 指令。
    // Q8_0 是一条 load 换 34 字节，两者差 8.5 倍指令数。实测 F16 解码只有
    // 18.2 GB/s（Q8_0 是 66.0），差的正是这个。
    // 改成 float4：32 lane × 4 = 128 个元素一轮。
    const uint nb = n_in >> 7;
    const uint lane4 = lane << 2;
    for (uint b = 0; b < nb; b++) {
        const uint i0 = (b << 7) + lane4;
        const float4 xv = *(device const float4*)(x + i0);
        #pragma unroll
        for (uint r = 0; r < MVR; r++) {
            const float4 wv = *(device const float4*)(p[r] + i0);
            acc[r] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    }
    for (uint i = (nb << 7) + lane; i < n_in; i += 32u) {
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
    device const half* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = (device const half*)(w + woff) + (ulong)(row0 + (r < nok ? r : 0u)) * n_in;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    // 一条 load 换 8 次乘加（half4 = 8 字节，两次 half4 = 16 字节 / 8 个元素）。
    // 原来是 `p[r][i]` 逐元素读 —— 2 字节付一条 load 指令，比 Q8_0 多 17 倍。
    // 实测 F16 解码 18.2 GB/s 就是这么来的（Q8_0 是 66.0）。
    // 32 lane × 8 = 256 个元素一轮。
    const uint nb = n_in >> 8;
    const uint lane8 = lane << 3;
    for (uint b = 0; b < nb; b++) {
        const uint i0 = (b << 8) + lane8;
        const float4 xa = *(device const float4*)(x + i0);
        const float4 xb = *(device const float4*)(x + i0 + 4u);
        #pragma unroll
        for (uint r = 0; r < MVR; r++) {
            const half4 wa = *(device const half4*)(p[r] + i0);
            const half4 wb = *(device const half4*)(p[r] + i0 + 4u);
            acc[r] += float(wa.x) * xa.x + float(wa.y) * xa.y +
                      float(wa.z) * xa.z + float(wa.w) * xa.w +
                      float(wb.x) * xb.x + float(wb.y) * xb.y +
                      float(wb.z) * xb.z + float(wb.w) * xb.w;
        }
    }
    for (uint i = (nb << 8) + lane; i < n_in; i += 32u) {
        const float xv = x[i];
        for (uint r = 0; r < MVR; r++) acc[r] += float(p[r][i]) * xv;
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

// ------------------------------------------------- Q5_0 解码 v2：lane 认领**块**
//
// 为什么要重写：上面那版是"lane = 块内元素下标"，32 条 lane 一起吃掉
// 一个 22 字节的块。一个 warp 指令周期只搬运 22 字节，而 Q5_0 每 22 字节
// 要 4 条 load（d 2 字节、qh 4 字节、qs 1 字节、x 1 个 float），于是
// **3.7 字节/指令**。
//
// 这不是理论推演，是同进程实测出来的相关性：
//   Q8_0  8.5 字节/指令 → 83 GB/s（lm_head，144 MB）
//   Q5_0  3.7 字节/指令 → 38 GB/s（ffn gate+up，144 MB）
//   比值 2.3x vs 2.2x —— 斜率几乎就是 1，说明**是发射端卡住，不是显存**。
//
// 改法：一条 lane 认领一整块（32 个元素），一个 warp 一次吃掉 32 块 = 704
// 字节。nibble 区用 ld16s 读 16 字节（22 的块跨度只保证 2 字节对齐，
// 不能用 uint4，见 ld16s 的注释）。
//
// 代价是 x 变成"每条 lane 要一整段 32 个 float"。所以：
//   * 用 float4 读进寄存器（8 条指令换 128 字节），
//   * 在 MVR 行之间复用（同一块的 x 对所有行都一样）。
// 综合下来每行约 40 条指令，上面那版是 168 条。
//
// n_in=896 时 nb=28 < 32，会有 4 条 lane 空转；这是可接受的，因为
// 真正省掉的是 28 次循环迭代。
kernel void mv_q5_0_v2(device const uchar* w     [[buffer(0)]],
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

    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    for (uint b = lane; b < nb; b += 32u) {
        // 这一块的 x 切片 [b*32, b*32+32)，32 个 float 全进寄存器
        const ulong xb = (ulong)b * 32u;
        float xl[32];
        #pragma unroll
        for (uint t = 0; t < 8u; t++) {
            const float4 v = *(device const float4*)(x + xb + (t << 2));
            xl[t * 4u + 0u] = v.x;
            xl[t * 4u + 1u] = v.y;
            xl[t * 4u + 2u] = v.z;
            xl[t * 4u + 3u] = v.w;
        }

        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + (ulong)b * 22u;
            const uint4 qs = ld16s(q + 6u);          // 16 字节 nibble 区
            const uint h0 = uint(q[2]) | (uint(q[3]) << 8);   // 元素 0..15 的高位
            const uint h1 = uint(q[4]) | (uint(q[5]) << 8);   // 元素 16..31 的高位
            const uint qw[4] = {qs.x, qs.y, qs.z, qs.w};
            float s = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16u; i++) {
                const uint byte = (qw[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
                const uint lo = (byte & 0x0Fu) | (((h0 >> i) & 1u) << 4);
                const uint hi = (byte >> 4)    | (((h1 >> i) & 1u) << 4);
                s += float(int(lo) - 16) * xl[i];
                s += float(int(hi) - 16) * xl[16u + i];
            }
            acc[r] += rd_f16(q) * s;
        }
    }

    // 归约：lane L 只累加了第 L, L+32, ... 块，所以 simd_sum 正好是整行的和
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
}

// ------------------------------------------------- Q4_0 解码 v2：lane 认领**块**
//
// 和 mv_q5_0_v2 是同一个形状，只是块小一圈、少一组高位：
//   Q5_0  24 字节 = d(2) + qh(4) + qs(16)，v = d*(q - 16)
//   Q4_0  18 字节 = d(2) +          qs(16)，v = d*(q -  8)
//
// 为什么非补不可：Q4_0 是 llama.cpp 生态里下载量最大的格式，而它此前只有
// 老写法（lane = 块内元素下标，一个 warp 一个周期只搬 18 字节、要发 4 条
// load → 4.5 字节/指令）。同机实测 Q4_0 解码 111 tok/s，比 Q5_0 的 161
// 还低 —— 权重更小反而更慢，只可能是发射端卡住，不是显存。
//
// 18 = 2×9，块跨度只保证 2 字节对齐，qs 在 q+2 处也不能用 uint4，
// 同样走 ld16s 退 4 字节再拼。
kernel void mv_q4_0_v2(device const uchar* w     [[buffer(0)]],
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

    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    for (uint b = lane; b < nb; b += 32u) {
        const ulong xb = (ulong)b * 32u;
        float xl[32];
        #pragma unroll
        for (uint t = 0; t < 8u; t++) {
            const float4 v = *(device const float4*)(x + xb + (t << 2));
            xl[t * 4u + 0u] = v.x;
            xl[t * 4u + 1u] = v.y;
            xl[t * 4u + 2u] = v.z;
            xl[t * 4u + 3u] = v.w;
        }

        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + (ulong)b * 18u;
            const uint4 qs = ld16s(q + 2u);
            const uint qw[4] = {qs.x, qs.y, qs.z, qs.w};
            float s = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16u; i++) {
                const uint byte = (qw[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
                s += float(int(byte & 0x0Fu) - 8) * xl[i];
                s += float(int(byte >> 4)    - 8) * xl[16u + i];
            }
            acc[r] += rd_f16(q) * s;
        }
    }

    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
}

// ---------------------------------------------------------------- Q5_1（24 字节 / 32 元）
//
// 和 Q5_0 只差一处：多一个 m，解量化是 d*q + m 而不是 d*(q-16)。
// 看着更简单，但 m 项要乘 sum(x)，所以还是得把 32 个 x 累一遍 —— 好在它们
// 本来就在寄存器里，而 sum(x) 对 MVR 行是共用的，提到行循环外只算一次。
//
// 为什么照 v2（一条 lane 认领一整块）而不是照 mv_q5_0 的旧写法：旧写法是
// 3.7 字节/指令的发射受限版本，Q5_1 块更大（24 vs 22）只会更差。
//
// 这个格式为什么值得单开一档：Qwen 官方放出来的 "q5_k_m" 文件里，绝大多数
// 张量其实是 Q5_1（不是 Q5_K），所以它是"能跑起来"和"跑不动"的分界线。
kernel void mv_q5_1(device const uchar* w     [[buffer(0)]],
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
    const ulong rbytes = (ulong)nb * 24u;

    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;

    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    for (uint b = lane; b < nb; b += 32u) {
        const ulong xb = (ulong)b * 32u;
        float xl[32];
        #pragma unroll
        for (uint t = 0; t < 8u; t++) {
            const float4 v = *(device const float4*)(x + xb + (t << 2));
            xl[t * 4u + 0u] = v.x;
            xl[t * 4u + 1u] = v.y;
            xl[t * 4u + 2u] = v.z;
            xl[t * 4u + 3u] = v.w;
        }
        float sx = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 32u; i++) sx += xl[i];

        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + (ulong)b * 24u;
            // 块跨度 24 = 8×3，q+8 只保证 8 字节对齐 → 用 ld16s
            const uint4 qs = ld16s(q + 8u);
            const uint qh = uint(q[4]) | (uint(q[5]) << 8) |
                            (uint(q[6]) << 16) | (uint(q[7]) << 24);
            const uint qw[4] = {qs.x, qs.y, qs.z, qs.w};
            float s = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16u; i++) {
                const uint byte = (qw[i >> 2] >> ((i & 3u) << 3)) & 0xFFu;
                const uint lo = (byte & 0x0Fu) | (((qh >> i) & 1u) << 4);
                const uint hi = (byte >> 4)    | (((qh >> (16u + i)) & 1u) << 4);
                s += float(lo) * xl[i];
                s += float(hi) * xl[16u + i];
            }
            acc[r] += fma(rd_f16(q), s, rd_f16(q + 2u) * sx);
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

// 同一个映射，但整块头已经作为 uint4 读进寄存器了。
static inline uint hdr_byte(uint4 h, uint i) {
    const uint w = (i < 4u) ? h.x : (i < 8u) ? h.y : (i < 12u) ? h.z : h.w;
    return (w >> ((i & 3u) << 3)) & 0xFFu;
}

// 注意偏移：k4_scale_min(j, q, ...) 传进来的 q 是**块头第 4 字节**（scales 起点），
// 而 hdr_byte 的下标从**块头第 0 字节**算 —— 所以下面每个下标都要 +4。
// 第一次写这里时照抄了 k4_scale_min 的下标、忘了这 4 个字节，
// 结果 scale/min 全部错位 → 融合校验 rel=1.3e+01 → 静默回退 CPU（16 tok/s）。
static inline void k4_scale_min_h(uint j, uint4 h, thread uint& s, thread uint& m) {
    if (j < 4u) {
        s = hdr_byte(h, j + 4u) & 63u;
        m = hdr_byte(h, j + 8u) & 63u;
    } else {
        s = (hdr_byte(h, j + 8u) & 0x0Fu) | ((hdr_byte(h, j) >> 6) << 4);
        m = (hdr_byte(h, j + 8u) >> 4)    | ((hdr_byte(h, j + 4u) >> 6) << 4);
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
    // 历史：最早一个 lane 在每个 sub-block 里取一个元素，一个块要算 8 次
    // k4_scale_min，10 条指令/元素，decode 只有 14.5 GB/s。
    // 后来改成"一个 lane 固定管一个 sub-block 的 8 个元素"，1 次 k4_scale_min
    // 摊到 8 个元素，约 3 条指令/元素，提到 59 GB/s。
    //
    // 但 59 GB/s 仍然低于 F16 的 83 GB/s，而两者的 MVR / 线程组形状完全一样。
    // 差别只在**一条 load 覆盖多少元素**：
    //   mv_f16   一条 16 字节 load  → 8 个元素，每 8 元素 3 条 load
    //   mv_q4_k  一条 8 字节 uint2  → 8 个元素，外加 8 条块头字节读
    // 于是同样的元素量，Q4_K 要多发 3 倍以上的 load 指令。
    //
    // 这一版让每个 lane 用**一条 uint4** 管 32 个元素：
    //   blk = lane/8  → warp 内第几个块（一个 warp 覆盖 4 个块）
    //   m   = lane%8  → 块内第几个 16 字节组（一个块正好 8 组）
    //   k   = m/2     → 64 元素块号 0..3
    //   g   = m%2     → 组内前半 / 后半
    // 一个 16 字节组覆盖 32 个元素：低半字节 = 元素 64k+16g+i（sub-block 2k），
    // 高半字节 = 元素 64k+32+16g+i（sub-block 2k+1）。所以只需 2 次
    // k4_scale_min，摊到 32 个元素。load 指令数：每 32 元素 1 条 uint4 + 8 条
    // 块头字节读，而旧版要 4 条 uint2 + 32 条字节读。
    //
    // 尾部：nb 不一定是 4 的倍数。用 min() 夹住块下标避免越界，再用 msk 把
    // 多出来的块贡献清零 —— 分支-free，代价是每条 lane 两次乘法。
    //
    // ==== Round 17：两条被否掉的路，别重走 ====
    //
    // (1) 换成"整行 16 字节单元按 lane 连续吃"（512 字节对齐、块头不挖洞）。
    //     tools/membw.mm 量出纯读带宽能从 67~73 提到 95~99 GB/s，看着很值。
    //     真实 decode 却是 **0.80x**（61.0 → 49.0 tok/s，5 轮交错配对，
    //     两边的轮内极差都只有 1.02x，不是噪声）。
    //     原因：144 = 9×16，一行 nb*9 个单元，warp 的 32 条 lane 除不尽。
    //     nb=8 时 72 个单元要 3 趟（96 个 lane 槽），24 个槽空转，
    //     等于把 ALU 和 x 读取抬高了 50%。而下面这个映射的 32 个**数据**单元
    //     正好等于 32 条 lane，利用率本来就是满的。
    //     教训：纯读探针只量内存，不量 lane 利用率；解码的约束是后者。
    //     （顺带踩到一个坑：新映射里 o==0 的 lane 踩在块头上，msk 必须把它
    //      一起挡掉。旧映射没有"块头 lane"这个角色，所以以前不会遇到。）
    //
    // (2) 想让一条 lane 一次吃 MVB 个块、把 x 的读指令摊薄。
    //     **不成立**：x 是按块索引的（x[b*256 + ...]），跨块根本不是同一段。
    //     x 只能在 MVR 个行之间复用，而那个复用已经做了。
    //
    // 所以这个映射就停在这里：它的上限是 membw 里 block4 的 72 GB/s，
    // 实测 68 GB/s（94%）。要再往上必须改**权重布局**（把块头拆成独立数组，
    // 数据区就是 8 个连续单元 × 4 块 = 32 条 lane 全满、且 512 字节对齐），
    // 代价是加载时重排 + 一份额外内存。那是另一件事，不在这一轮。
    const uint blk  = lane >> 3;             // 0..3
    const uint m    = lane & 7u;             // 0..7
    const uint k    = m >> 1;                // 0..3
    const uint g    = m & 1u;                // 0..1
    const uint boff = 16u + (k << 5) + (g << 4);
    const uint xlo  = (k << 6) + (g << 4);
    const uint xhi  = xlo + 32u;
    const uint sb0  = k << 1;

    const uint nbi = (nb + 3u) >> 2;
    for (uint bi = 0; bi < nbi; bi++) {
        const uint braw = (bi << 2) + blk;
        const bool ok   = (braw < nb);
        const uint b    = ok ? braw : (nb - 1u);
        const float msk = ok ? 1.0f : 0.0f;
        const ulong xb  = (ulong)b * 256u;

        float xl[16], xh[16];
        #pragma unroll
        for (uint t = 0; t < 4u; t++) {
            const float4 v = *(device const float4*)(x + xb + xlo + (t << 2));
            xl[t*4u+0] = v.x; xl[t*4u+1] = v.y; xl[t*4u+2] = v.z; xl[t*4u+3] = v.w;
            const float4 w = *(device const float4*)(x + xb + xhi + (t << 2));
            xh[t*4u+0] = w.x; xh[t*4u+1] = w.y; xh[t*4u+2] = w.z; xh[t*4u+3] = w.w;
        }

        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + (ulong)b * 144u;
            // 块头**不要**改成一条 uint4 —— 两种形状都试过，都更慢：
            //   全 warp 广播（32 lane 读同一块头）：56.2 vs 59.4
            //   8 lane 一组（warp 内 4 组各读各的）：61.3 vs 63.2
            // 字节读虽然条数多，但全部命中 L1，比 16 字节向量读更划算。
            uchar s0, n0, s1, n1;
            k4_scale_min(int(sb0), q + 4u, s0, n0);
            k4_scale_min(int(sb0 + 1u), q + 4u, s1, n1);
            const float dl = rd_f16(q);
            const float dm = rd_f16(q + 2u);
            const float scl = dl * float(s0) * msk;
            const float mnl = dm * float(n0) * msk;
            const float sch = dl * float(s1) * msk;
            const float mnh = dm * float(n1) * msk;
            const uint4 raw = *(device const uint4*)(q + boff);
            float a = 0.0f;
            // min 项折进逐元素 fma：sum(d*sc*v - dm*m)*x 省掉单独的 x 求和
            #pragma unroll
            for (uint i = 0; i < 16u; i++) {
                const uint word = (i < 4u) ? raw.x : (i < 8u) ? raw.y
                                : (i < 12u) ? raw.z : raw.w;
                const uint byte = (word >> ((i & 3u) << 3)) & 0xFFu;
                a += (scl * float(byte & 0x0Fu) - mnl) * xl[i];
                a += (sch * float(byte >> 4) - mnh) * xh[i];
            }
            acc[r] += a;
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

// ---------------------------------------------------------------- Q5_K（176 字节 / 256 元）
//
// 和 Q4_K 的唯一区别：每个元素还有第 5 位，存在 32 字节的 qh 里。
//
// 位序不是"元素 i 用 qh 的第 i 位"（那样 32 字节正好 256 位，看着很像），
// 而是把 8 个子块的第 5 位**交错**在同一个字节里：
//     元素 32s + t  →  qh[t] 的第 s 位      (s = 子块 0..7, t = 子块内 0..31)
// 这个排法让一条 lane 管 32 个元素（跨子块 2k / 2k+1）时，两半用的是**同一批**
// qh 字节（16g .. 16g+15），只是取的位不同（2k / 2k+1）—— 所以只多一条
// uint4 读，而不是两条。位下标恰好等于子块下标，取位时不用再算。
//
// 映射、尾块处理、块头为什么不做成向量读，全部沿用 mv_q4_k 的结论
// （那里试过的两条死路，别在这里重走）。
kernel void mv_q5_k(device const uchar* w     [[buffer(0)]],
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
    const ulong rbytes = (ulong)nb * 176u;
    device const uchar* p[MV_R];
    for (uint r = 0; r < MVR; r++)
        p[r] = w + woff + (ulong)(row0 + (r < nok ? r : 0u)) * rbytes;
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    const uint blk  = lane >> 3;             // 0..3
    const uint m    = lane & 7u;             // 0..7
    const uint k    = m >> 1;                // 0..3
    const uint g    = m & 1u;                // 0..1
    const uint boff = 48u + (k << 5) + (g << 4);   // qs 起点（Q4_K 是 16）
    const uint hoff = 16u + (g << 4);              // qh 起点
    const uint xlo  = (k << 6) + (g << 4);
    const uint xhi  = xlo + 32u;
    const uint sb0  = k << 1;                      // 低半字节 → 子块 2k

    const uint nbi = (nb + 3u) >> 2;
    for (uint bi = 0; bi < nbi; bi++) {
        const uint braw = (bi << 2) + blk;
        const bool ok   = (braw < nb);
        const uint b    = ok ? braw : (nb - 1u);
        const float msk = ok ? 1.0f : 0.0f;
        const ulong xb  = (ulong)b * 256u;

        float xl[16], xh[16];
        #pragma unroll
        for (uint t = 0; t < 4u; t++) {
            const float4 v = *(device const float4*)(x + xb + xlo + (t << 2));
            xl[t*4u+0] = v.x; xl[t*4u+1] = v.y; xl[t*4u+2] = v.z; xl[t*4u+3] = v.w;
            const float4 w = *(device const float4*)(x + xb + xhi + (t << 2));
            xh[t*4u+0] = w.x; xh[t*4u+1] = w.y; xh[t*4u+2] = w.z; xh[t*4u+3] = w.w;
        }

        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = p[r] + (ulong)b * 176u;
            uchar s0, n0, s1, n1;
            k4_scale_min(int(sb0), q + 4u, s0, n0);
            k4_scale_min(int(sb0 + 1u), q + 4u, s1, n1);
            const float dl = rd_f16(q);
            const float dm = rd_f16(q + 2u);
            const float scl = dl * float(s0) * msk;
            const float mnl = dm * float(n0) * msk;
            const float sch = dl * float(s1) * msk;
            const float mnh = dm * float(n1) * msk;
            const uint4 raw = *(device const uint4*)(q + boff);
            // qh：一条 uint4 覆盖 16 个字节，正好是本 lane 那 16 个元素
            const uint4 hraw = *(device const uint4*)(q + hoff);
            float a = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16u; i++) {
                const uint sh  = (i & 3u) << 3;
                const uint word = (i < 4u) ? raw.x : (i < 8u) ? raw.y
                                : (i < 12u) ? raw.z : raw.w;
                const uint hword = (i < 4u) ? hraw.x : (i < 8u) ? hraw.y
                                 : (i < 12u) ? hraw.z : hraw.w;
                const uint byte = (word >> sh) & 0xFFu;
                const uint hbyte = (hword >> sh) & 0xFFu;
                const uint lo = (byte & 0x0Fu) | (((hbyte >> sb0) & 1u) << 4);
                const uint hi = (byte >> 4)    | (((hbyte >> (sb0 + 1u)) & 1u) << 4);
                a += (scl * float(lo) - mnl) * xl[i];
                a += (sch * float(hi) - mnh) * xh[i];
            }
            acc[r] += a;
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

// ---------------------------------------------------------------- Q5_1 / Q5_K 的单行版
//
// 为什么需要这一代：mm2_* 按 TILE_K=256 切片，要求 n_in % 256 == 0；
// mm3_* 也一样。而 Qwen2.5 的 n_embd = 896（896 = 3.5×256），**不满足**，
// 于是 attn_q/k/v/o 与 ffn_gate/up 这几档（n_in=896）在 rows>1 时根本没有
// 可用的分块内核 —— 只有这一代能吃任意 n_in。
//
// 缺了它们的症状不是"慢"而是**直接崩**：gemm 里 g_mpipe[k] 是 nil，
// setComputePipelineState:nil 当场炸。这也是给新格式加内核时必须一起补齐
// 的一档，不能只写 mm2/mm3。
kernel void mm_q5_1(device const uchar* w     [[buffer(0)]],
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
    Y += yoff;
    const uint nb = n_in >> 5;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 24u;
    const uint idx = lane & 15u;
    const bool hi  = lane >= 16u;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 24u;
        const uint qh = uint(q[4]) | (uint(q[5]) << 8) |
                        (uint(q[6]) << 16) | (uint(q[7]) << 24);
        const int bit5 = int((qh >> lane) & 1u) << 4;
        const uint byte = uint(q[8u + idx]);
        const int v = hi ? (int(byte >> 4) | bit5) : (int(byte & 0x0Fu) | bit5);
        const uint e = (b << 5) + lane;
        device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
        FMA8(fma(rd_f16(q), float(v), rd_f16(q + 2u)));
    }
    RED8();
}

kernel void mm_q5_k(device const uchar* w     [[buffer(0)]],
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
    Y += yoff;
    const uint nb = n_in >> 8;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 176u;
    float acc[BMAX];
    for (int j = 0; j < BMAX; j++) acc[j] = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q  = p + b * 176u;
        device const uchar* qh = q + 16u;
        const float dl = rd_f16(q);
        const float dm = rd_f16(q + 2u);
        // 8 个子块各一轮，lane 覆盖子块内 32 个元素
        for (uint k = 0; k < 8u; k++) {
            const uint chunk = k >> 1;
            const uint hi    = k & 1u;
            const uint byte  = uint(q[48u + (chunk << 5) + lane]);
            const uint lo    = hi ? (byte >> 4) : (byte & 0x0Fu);
            const uint hb    = (uint(qh[lane]) >> k) & 1u;
            uchar sc_, m_;
            k4_scale_min(int(k), q + 4u, sc_, m_);
            const int v = int(lo | (hb << 4));
            const uint e = (b << 8) + (k << 5) + lane;
            device const float4* xp = (device const float4*)(X + (ulong)e * BMAX);
            FMA8(fma(dl * float(sc_), float(v), -dm * float(m_)));
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
struct RawQ5K { uint2 q; uint2 h; };   // qs 8 字节 + qh 8 字节
struct RawQ51 { uint2 q; uint h; };    // qs 8 字节 + qh 4 字节
struct RawQ6K { ushort ql[4]; ushort qh[4]; };  // ql 8 + qh 8
struct RawQ8  { ushort q[4]; };        // 8 字节
struct RawF16 { uint4 q; };            // 16 字节
struct RawF32 { uint4 q; };            // 只预取前 16 字节，后 16 字节在 CV 里补

static inline RawQ4K ld_q4k(device const uchar* q, uint lane) {
    RawQ4K r;
    r.q = *(device const uint2*)(q + 16u + ((lane >> 3) << 5) + ((lane & 3u) << 3));
    return r;
}

// Q5_K：qs 在 48（Q4_K 是 16），qh 在 16。两者都是 8 字节对齐
// （块跨度 176 = 22×8，48/16/32 的偏移都是 8 的倍数），所以 uint2 安全。
static inline RawQ5K ld_q5k(device const uchar* q, uint lane) {
    RawQ5K r;
    const uint o = (lane & 3u) << 3;
    r.q = *(device const uint2*)(q + 48u + ((lane >> 3) << 5) + o);
    r.h = *(device const uint2*)(q + 16u + o);
    return r;
}

// Q5_1：块跨度 24，qh 只有 4 字节（32 个元素各 1 位）。
// 一条 lane 管 8 个元素 = 子块内的 (lane&3)*8..+7，高位全在同一个 uint32 里。
// 注意块偏移 (lane>>2)*24 不能漏 —— mm2 的 lane 同时承担"片内第几块"和
// "块内第几组"，漏了它所有 lane 都去读第 0 块（结果错但不越界，最难查）。
static inline RawQ51 ld_q51(device const uchar* q, uint lane) {
    RawQ51 r;
    const device uchar* b = q + (ulong)(lane >> 2) * 24u;
    r.q = *(device const uint2*)(b + 8u + ((lane & 1u) << 3));
    r.h = *(device const uint*)(b + 4u);
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

// Q5_K：子块下标 sb = 2·(lane>>3) + bit2，而**第 5 位的位号恰好等于 sb**，
// 所以 nibble 的选择位和 qh 的位号是同一个值，一行算完两处都用。
static inline void cv_q5k(device const uchar* q, RawQ5K raw, uint lane,
                          thread float* wv) {
    const float d_  = rd_f16(q);
    const float dm_ = rd_f16(q + 2u);
    const uint sb_  = ((lane >> 3) << 1) + ((lane >> 2) & 1u);
    uchar s1_, m1_;
    k4_scale_min(int(sb_), q + 4u, s1_, m1_);
    const float sc_ = d_ * float(s1_);
    const float mn_ = dm_ * float(m1_);
    const uint sh_  = ((lane >> 2) & 1u) << 2;
#pragma unroll
    for (uint i_ = 0; i_ < 8u; i_++) {
        const uint bit_ = (i_ & 3u) << 3;
        const uint word_ = (i_ < 4u) ? raw.q.x : raw.q.y;
        const uint hword_ = (i_ < 4u) ? raw.h.x : raw.h.y;
        const uint v_ = ((word_ >> (bit_ + sh_)) & 0x0Fu)
                      | (((hword_ >> bit_) >> sb_ & 1u) << 4);
        wv[i_] = fma(sc_, float(v_), -mn_);
    }
}

// Q5_1：d·q + m，没有 -16 的偏移。高位在同一个 uint32 里，
// 位号 = (lane&3)*8 + i，正好是 ld 里那个 8 字节组的下标。
static inline void cv_q51(device const uchar* q, RawQ51 raw, uint lane,
                          thread float* wv) {
    const float d_ = rd_f16(q);
    const float m_ = rd_f16(q + 2u);
    const uint hi_ = (lane & 3u) << 3;
    const uint sh_ = ((lane >> 1) & 1u) << 2;
#pragma unroll
    for (uint i_ = 0; i_ < 8u; i_++) {
        const uint bit_ = (i_ & 3u) << 3;
        const uint word_ = (i_ < 4u) ? raw.q.x : raw.q.y;
        const uint v_ = ((word_ >> (bit_ + sh_)) & 0x0Fu)
                      | (((raw.h >> (hi_ + i_)) & 1u) << 4);
        wv[i_] = fma(d_, float(v_), m_);
    }
}

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
                          thread float* wv) {    const uint qq_ = lane >> 2;
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
// 注意 TB 是"一片 TILE_K=256 元素"的字节跨度，不是"一个块"的：
// Q5_1 的块只有 32 元素，一片里有 8 个块 → 8×24=192。
MM2_KERNEL(mm2_q5_k, RawQ5K, ld_q5k,  cv_q5k,  ((n_in >> 8) * 176u), 176u)
MM2_KERNEL(mm2_q5_1, RawQ51, ld_q51,  cv_q51,  ((n_in >> 5) * 24u), ((TILE_K >> 5) * 24u))

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

// ---------------------------------------------------------------- Q5_K
//
// 和 Q4_K 只差"第 5 位"。mm3 里本线程管的是**同一个子块内的 16 个连续元素**
// （MMA_NK=32 且 e0 是 16 的倍数，16 个元素不会跨子块也不会跨 64 元素段），
// 所以 qh 只需要一条 16 字节读，位号 = 子块下标，和 nibble 的选择位是同一个。
kernel void mm3_q5_k(device const uchar* w      [[buffer(0)]],
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
    threadgroup half smem[MMA_BM * MMA_BN * 2];
    threadgroup half* ws = smem;
    threadgroup half* xs = smem + MMA_BN * MMA_NK;

    const uint row0 = tgid * MMA_BN;
    const uint nb   = n_in / MMA_NK;
    const ulong rbytes = (ulong)(n_in >> 8) * 176u;

    const uint lr     = tid >> 1;
    const uint half16 = tid & 1u;
    const uint grow   = row0 + lr;
    device const uchar* prow = w + woff + (ulong)min(grow, n_out - 1u) * rbytes;

    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2u; i++)
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint b = 0; b < nb; b++) {
        const uint k0 = b * MMA_NK;

        {
            const uint e0    = k0 + (half16 << 4);
            const uint inblk = e0 & 255u;
            const uint chunk = inblk >> 6;           // 0..3
            const uint inch  = inblk & 63u;          // 0/16/32/48
            const bool hi    = inch >= 32u;
            const uint sb    = inblk >> 5;           // 子块 0..7 == 第 5 位的位号
            device const uchar* q = prow + (ulong)(e0 >> 8) * 176u;

            uchar s1, m1;
            k4_scale_min(int(sb), q + 4u, s1, m1);
            const float sc = rd_f16(q)      * float(s1);
            const float mn = rd_f16(q + 2u) * float(m1);

            // qs 在 48（Q4_K 是 16），qh 在 16；两者偏移都是 16 的倍数，
            // 块跨度 176 = 11×16，所以 uint4 对齐成立。
            const uint4 raw = *(device const uint4*)(q + 48u + (chunk << 5) + (inch & 31u));
            const uint4 hraw = *(device const uint4*)(q + 16u + (inch & 31u));
            const uint words[4]  = { raw.x, raw.y, raw.z, raw.w };
            const uint hwords[4] = { hraw.x, hraw.y, hraw.z, hraw.w };
            const uint sh = hi ? 4u : 0u;
            const uint hb = sb;
            const uint4 o4 = uint4(lr * MMA_NK + (half16 << 4));

            half4 h4[4];
            #pragma unroll
            for (uint t = 0; t < 4u; t++) {
                #pragma unroll
                for (uint j = 0; j < 4u; j++) {
                    const uint bit = j << 3;
                    const uint byte = (words[t] >> bit) & 0xFFu;
                    const uint v = ((byte >> sh) & 0x0Fu)
                                 | (((hwords[t] >> bit) >> hb & 1u) << 4);
                    h4[t][j] = (half)fma(sc, float(v), -mn);
                }
            }
            threadgroup half4* dst = (threadgroup half4*)(ws + (o4.x & ~3u));
            #pragma unroll
            for (uint t = 0; t < 4u; t++) dst[t] = h4[t];
        }

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

// ---------------------------------------------------------------- Q5_1
//
// MMA_NK=32 正好等于一个 Q5_1 块，于是"本线程那 16 个元素"就是块的
// 低半字节或高半字节 —— half16 直接就是 hi。这是所有格式里最省事的一档。
kernel void mm3_q5_1(device const uchar* w      [[buffer(0)]],
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
    threadgroup half smem[MMA_BM * MMA_BN * 2];
    threadgroup half* ws = smem;
    threadgroup half* xs = smem + MMA_BN * MMA_NK;

    const uint row0 = tgid * MMA_BN;
    const uint nb   = n_in / MMA_NK;
    const ulong rbytes = (ulong)(n_in >> 5) * 24u;

    const uint lr     = tid >> 1;
    const uint half16 = tid & 1u;
    const uint grow   = row0 + lr;
    device const uchar* prow = w + woff + (ulong)min(grow, n_out - 1u) * rbytes;

    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2u; i++)
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint b = 0; b < nb; b++) {
        const uint k0 = b * MMA_NK;

        {
            const uint e0 = k0 + (half16 << 4);
            device const uchar* q = prow + (ulong)(e0 >> 5) * 24u;

            const float d = rd_f16(q);
            const float m = rd_f16(q + 2u);
            const uint qh = *(device const uint*)(q + 4u);
            // 块跨度 24 只保证 8 字节对齐 → 用 ld16s 读 nibble 区
            const uint4 raw = ld16s(q + 8u);
            const uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            const uint sh = half16 << 2;
            const uint hb = half16 << 4;

            half4 h4[4];
            #pragma unroll
            for (uint t = 0; t < 4u; t++) {
                #pragma unroll
                for (uint j = 0; j < 4u; j++) {
                    const uint bit = j << 3;
                    const uint byte = (words[t] >> bit) & 0xFFu;
                    const uint v = ((byte >> sh) & 0x0Fu)
                                 | (((qh >> (hb + (t << 2) + j)) & 1u) << 4);
                    h4[t][j] = (half)fma(d, float(v), m);
                }
            }
            threadgroup half4* dst = (threadgroup half4*)(ws + ((lr * MMA_NK + (half16 << 4)) & ~3u));
            #pragma unroll
            for (uint t = 0; t < 4u; t++) dst[t] = h4[t];
        }

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

// ---------------------------------------------------------------- F16
//
// F16 的 MMA 内核是最省事的一档：权重本来就是 half，反量化那一步整个不存在，
// 只要把 16 个 half 从 device 搬进 threadgroup 的 ws。
//
// 这条路径为什么值钱
// ------------------
// F16 之前走的是 mm2_f16（非 MMA）。1.7B 模型 prefill 只有 54 tok/s，
// 折合 2×1.7e9×54 ≈ 184 GFLOP/s；而 Q4_K 的 MMA 路径有 1.7 TFLOP/s。
// 差 9 倍，全在"没有用上 simdgroup_matrix"——和格式无关，和 4bit 也无关。
//
// 搬运用 uint4（一次 8 个 half）而不是逐元素：16 个 half = 32 字节 = 2 条
// uint4。对齐成立：n_in 是 256 的倍数 → 行跨度为 512 字节的倍数；k0 是 32 的
// 倍数、half16*16 个元素正好 32 字节 → 偏移恒为 32 字节对齐。
//
// 这里刻意没有走"simdgroup_load 直接从 device 读"：那样省掉 ws 和一道 barrier，
// 但每个 simdgroup 各自读 8 行 × 16 字节，跨 cache line 更碎。先照 proven 的
// 结构抄一遍，量完再决定要不要动。
kernel void mm3_f16(device const uchar* w      [[buffer(0)]],
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
    threadgroup half smem[MMA_BM * MMA_BN * 2];        // 4096 halfs = 8 KB
    threadgroup half* ws = smem;                       // 权重 [n][k]
    threadgroup half* xs = smem + MMA_BN * MMA_NK;     // 激活 [k][m]

    const uint row0 = tgid * MMA_BN;
    const uint nb   = n_in / MMA_NK;
    const ulong rbytes = (ulong)n_in * 2u;             // 每行 n_in 个 half

    const uint lr     = tid >> 1;          // 0..63
    const uint half16 = tid & 1u;          // 0 / 1
    const uint grow   = row0 + lr;
    device const uchar* prow = w + woff + (ulong)min(grow, n_out - 1u) * rbytes;

    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2u; i++)
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint b = 0; b < nb; b++) {
        const uint k0 = b * MMA_NK;

        // ---- 16 个 half 直接搬进 ws：2 条 uint4 读 + 2 条 uint4 写
        {
            device const uint4* src =
                (device const uint4*)(prow + (ulong)((k0 + (half16 << 4)) << 1));
            threadgroup uint4* dst =
                (threadgroup uint4*)(ws + lr * MMA_NK + (half16 << 4));
            dst[0] = src[0];
            dst[1] = src[1];
        }

        // ---- 激活搬进 xs：与 mm3_q4_k 完全一致（一次 float4 读四个 token）
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

// ---------------------------------------------------------------- Q8_0
//
// Q8_0 块：32 元素 / 34 字节 —— d(fp16)@0、qs[32] int8@2。v[i] = d * qs[i]。
//
// 一个线程负责 16 个元素 = 半个块，所以 scale 每块只读一次。
// 块跨度 34 ≡ 2 (mod 4)，起始地址只有 2 字节对齐 → 只能按 ushort 读
// （8 条换 16 字节）。这是格式决定的，不是没优化：34 字节的块头让
// uint4 永远落不到 16 字节边界上，ld16s 那种"退 4 字节再拼"实测在
// Q6_K 上比 ushort 还慢 1.5%。
//
// Q8_0 之前走 mm2_q8_0：1.7B 上 prefill 184.6 tok/s，折合
// 2×1.7e9×184.6 ≈ 628 GFLOP/s —— 只有 Q4_K MMA 路径的 1/3，
// 而 Q8_0 每元素的解包指令数比 Q4_K 还少。差的全在没用 MMA。
kernel void mm3_q8_0(device const uchar* w      [[buffer(0)]],
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
    threadgroup half smem[MMA_BM * MMA_BN * 2];
    threadgroup half* ws = smem;
    threadgroup half* xs = smem + MMA_BN * MMA_NK;

    const uint row0 = tgid * MMA_BN;
    const uint nb   = n_in / MMA_NK;
    const ulong rbytes = (ulong)(n_in >> 5) * 34u;     // 每行 (n_in/32) 个块

    const uint lr     = tid >> 1;
    const uint half16 = tid & 1u;
    const uint grow   = row0 + lr;
    device const uchar* prow = w + woff + (ulong)min(grow, n_out - 1u) * rbytes;

    simdgroup_float8x8 acc[2][4];
    #pragma unroll
    for (uint i = 0; i < 2u; i++)
        #pragma unroll
        for (uint j = 0; j < 4u; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint b = 0; b < nb; b++) {
        const uint k0 = b * MMA_NK;

        // ---- 16 个元素：半个 Q8_0 块，scale 只算一次
        {
            const uint e0  = k0 + (half16 << 4);
            const uint blk = e0 >> 5;          // 32 元素块号
            const uint inb = e0 & 31u;         // 0 / 16
            device const uchar* q = prow + (ulong)blk * 34u;
            const float d = f16_to_f32(*(device const ushort*)q);
            device const ushort* qs = (device const ushort*)(q + 2u + inb);

            half4 h4[4];
            #pragma unroll
            for (uint t = 0; t < 4u; t++) {
                const uint w0 = uint(qs[t * 2u]);
                const uint w1 = uint(qs[t * 2u + 1u]);
                h4[t][0] = (half)(d * float(int((char)(w0 & 0xFFu))));
                h4[t][1] = (half)(d * float(int((char)(w0 >> 8))));
                h4[t][2] = (half)(d * float(int((char)(w1 & 0xFFu))));
                h4[t][3] = (half)(d * float(int((char)(w1 >> 8))));
            }
            threadgroup half4* dst = (threadgroup half4*)(ws + lr * MMA_NK + (half16 << 4));
            #pragma unroll
            for (uint t = 0; t < 4u; t++) dst[t] = h4[t];
        }

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
// 这个内核排查时骗过整整一轮，记在这里
// --------------------------------------
// 第一版把 qh 的移位写成 `(l0 >> 1) << 1`（得到 0,0,2,2），正确是 `l0 << 1`
// （0,2,4,6）。半个块的权重因此全错 —— 于是 GPU 前向结果对不上 CPU，
// 引擎**正确地退回了 CPU**。症状是"慢 21 倍"，而不是任何报错。
//
// 更坑的是"贪心输出与关闭时逐字一致"这个验证：两边其实都在 CPU 上跑，
// 所以它什么都没验证。正确的验证是 `dubhe inspect` 那一行的融合校验
// （拿 CPU 前向做基准），或 `DUBHE_GPU_TIME=1` 看 `[gpu]` 忙时行有没有出现。
//
// 改对一个比特之后：prefill 418 → 508 tok/s（1.24x），decode 不变。
//
// 读法：块跨度 210 只保证 2 字节对齐，用不了 uint4，走 ld16s 做对齐修正。
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
            const uint sh    = l0 << 1;               // qh 的移位 = l0*2 → 0/2/4/6
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
                   device half* kc          [[buffer(2)]],
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
        // KV cache 存 fp16：长上下文时 KV 流量会超过权重，见 k_attn 的注释
        device half* dst = (device half*)((device uchar*)kc + koff)
                         + (ulong)row * kdim + kh * hd;
        const float x0 = src[i], x1 = src[i + hh];
        dst[i]      = half(x0 * c - x1 * s);
        dst[i + hh] = half(x0 * s + x1 * c);
    }
}

// d[doff + row*n + i] = (half)s[soff + row*n + i]
// 只用于把 V 写进 fp16 的 KV cache。K 走 k_rope，因为要就地旋转。
kernel void k_copy_h(device half* d          [[buffer(0)]],
                     constant ulong& doff    [[buffer(1)]],
                     device const float* s   [[buffer(2)]],
                     constant ulong& soff    [[buffer(3)]],
                     constant uint& n        [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < n) {
        const ulong o = (ulong)gid.y * n + gid.x;
        device half* dd = (device half*)((device uchar*)d + doff);
        dd[o] = half(fatc(s, soff)[o]);
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
                   device const half* kc  [[buffer(1)]],
                   device const half* vc  [[buffer(2)]],
                   device float* out      [[buffer(3)]],
                   constant ulong& coff   [[buffer(4)]],
                   constant uint& n_head  [[buffer(5)]],
                   constant uint& n_kvh   [[buffer(6)]],
                   constant uint& hd      [[buffer(7)]],
                   constant uint& nlen0   [[buffer(8)]],
                   constant uint& kv_dim  [[buffer(9)]],
                   constant float& scale  [[buffer(10)]],
                   constant uint& win     [[buffer(11)]],
                   constant uint& kvwrap  [[buffer(12)]],
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
    // KV cache 是 fp16。
    //
    // 为什么必须这么干：KV 是 f32 时，24506 token 上下文下 KV 要 2.41 GB，
    // 比 1.10 GB 的权重还大一倍，decode 直接从 52.9 掉到 16.9 tok/s，
    // 而且是纯带宽受限（3.5 GB/token ÷ 59 ms ≈ 59 GB/s，没到算力瓶颈）。
    // 换 fp16 后同一场景的 KV 减半到 1.2 GB。
    device const half* kb = (device const half*)((device uchar*)kc + coff) + kvh * hd;
    device const half* vb = (device const half*)((device uchar*)vc + coff) + kvh * hd;
    const uint nlen = nlen0 + row;
    // 滑窗：只回看最近 win 个位置（win == 0 表示全量）
    const uint t0 = (win > 0u && nlen > win) ? (nlen - win) : 0u;

    float m = -1e30f, l = 0.0f;
    float acc[4];
    for (int j = 0; j < 4; j++) acc[j] = 0.0f;

    for (uint t = t0; t < nlen; t++) {
        // kvwrap 是诊断开关（非 0 时按位与回绕到一小段 KV）：指令数与访存
        // 次数都不变，只把工作集缩小到能进 cache。用来区分"注意力是被带宽
        // 卡住"还是"被指令数卡住"——结果错误，只用于计时。
        const uint tt = kvwrap ? (t & kvwrap) : t;
        device const half* kk = kb + (ulong)tt * kv_dim;
        float partial = 0.0f;
        for (uint d = lane; d < hd; d += 32u) partial += qh[d] * float(kk[d]);
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

        device const half* vv = vb + (ulong)tt * kv_dim;
        int j = 0;
        for (uint d = lane; d < hd; d += 32u, j++) acc[j] += w * float(vv[d]);
    }

    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    device float* o = out + ((ulong)row * n_head + head) * hd;
    int j = 0;
    for (uint d = lane; d < hd; d += 32u, j++) o[d] = acc[j] * inv;
}

// 局部（未归并）结果布局：[row][split][head][hd + 2]，最后两个是 m 与 l
kernel void k_attn_split(device const float* q  [[buffer(0)]],
                         device const half* kc  [[buffer(1)]],
                         device const half* vc  [[buffer(2)]],
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
                         constant uint& kvwrap  [[buffer(13)]],
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
        device const half* kb = (device const half*)((device uchar*)kc + coff) + kvh * hd;
        device const half* vb = (device const half*)((device uchar*)vc + coff) + kvh * hd;

        for (uint t = k0; t < k1; t++) {
            const uint tt = kvwrap ? (t & kvwrap) : t;
            device const half* kk = kb + (ulong)tt * kv_dim;
            float partial = 0.0f;
            for (uint d = lane; d < hd; d += 32u) partial += qh[d] * float(kk[d]);
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

            device const half* vv = vb + (ulong)tt * kv_dim;
            int j = 0;
            for (uint d = lane; d < hd; d += 32u, j++) acc[j] += w * float(vv[d]);
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
