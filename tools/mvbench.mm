// mvbench.mm — 解码 matvec 内核的吞吐 / 正确性 / 布局 A/B
//
// 为什么单独做：改一次内核就要重新加载 1.1 GB 模型才能看到数字，一次迭代
// 40 秒，而且模型加载本身的噪声盖住了 5% 级的差别。这个工具只跑内核，
// 秒级出数，还能顺便和 CPU 参考实现对拍。
//
// 为什么必须**同进程交错**：这个文件早期版本把两种内核分开跑，同一份二进制
// 跨轮能差 1.5x（DVFS + 本机负载），绝对数字完全不可信。现在两种内核在
// 同一个进程里逐轮交替，只报**比值**。
//
// 布局 A/B 是什么：
//   GGUF 的 Q4_K 块是 144 字节 = 16 字节块头 + 128 字节数据。于是 warp 的
//   一条 load 只能是"4 段 128 字节、每段前面挖 16 字节的洞"，纯读天花板
//   74.8 GB/s（见 tools/membw.mm）。
//   把块头抽到独立数组之后，一行的数据区就是 nb×128 字节连续，4 个块 =
//   512 字节连续且对齐，纯读天花板 100 GB/s —— 而且 lane 映射完全不用改
//   （lane 还是 blk=lane>>3 / m=lane&7，32 个数据单元正好 32 条 lane）。
//   总字节数不变，只是把数据重排了一遍。
//
// 用法：
//   make mvbench
//   ./build/mvbench                      # 默认 q4_k 8192×2048
//   ./build/mvbench q4_k 2048 8192 10    # 类型 n_out n_in 轮数
//   DUBHE_MV_R=4 ./build/mvbench         # 换 MV_R
#include "metal_kernels.h"

#include "../engine/src/internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace dubhe;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 重排布局的实验内核。放在这里而不是 metal_kernels.h：还没证明它更快之前，
// 不该把实验代码混进出货的着色器。
static const char* kPackedMSL = R"MSL(
// 重排后的 Q4_K matvec。数据区（每行 nb×128 字节连续）+ 块头区（每块 16 字节）
// 分成两个数组，其余和 mv_q4_k 一模一样。
kernel void mv_q4_k_packed(device const uchar* w   [[buffer(0)]],
                           device const float* x   [[buffer(1)]],
                           device float* y         [[buffer(2)]],
                           constant uint& n_in     [[buffer(3)]],
                           constant ulong& woff    [[buffer(4)]],  // 数据区起点
                           constant uint& n_out    [[buffer(5)]],
                           constant ulong& hoff    [[buffer(6)]],  // 块头区起点
                           constant uint& flags    [[buffer(7)]],  // 诊断用，见下
                           uint tgid [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = (tgid * SG_PER_TG + sg) * MVR;
    if (row0 >= n_out) return;
    const uint nok = min(uint(MVR), n_out - row0);
    const uint nb = n_in >> 8;
    const ulong drow = (ulong)nb * 128u;
    const ulong hrow = (ulong)nb * 16u;

    device const uchar* pd[MV_R];
    device const uchar* ph[MV_R];
    for (uint r = 0; r < MVR; r++) {
        const uint rr = row0 + (r < nok ? r : 0u);
        pd[r] = w + woff + (ulong)rr * drow;
        ph[r] = w + hoff + (ulong)rr * hrow;
    }
    float acc[MV_R];
    for (uint r = 0; r < MVR; r++) acc[r] = 0.0f;

    // flags 是**诊断开关**，只在 mvbench 里用，出货路径永远传 0：
    //   bit0 = 跳过 x 读取（用常数 1.0 代替）→ 量 x 读取占多少时间
    //   bit1 = 跳过 fma（直接累加原始字节）→ 量 ALU 占多少时间
    // 两者都置位时，内核只剩"读权重"，应当接近 membw 的纯读天花板。
    const bool skip_x = (flags & 1u) != 0u;
    const bool skip_alu = (flags & 2u) != 0u;

    const uint blk = lane >> 3;              // 0..3
    const uint m   = lane & 7u;              // 0..7
    const uint k   = m >> 1;
    const uint g   = m & 1u;
    const uint xlo = (k << 6) + (g << 4);
    const uint xhi = xlo + 32u;
    const uint sb0 = k << 1;

    const uint nbi = (nb + 3u) >> 2;
    for (uint bi = 0; bi < nbi; bi++) {
        const uint braw = (bi << 2) + blk;
        const bool ok   = (braw < nb);
        const uint b    = ok ? braw : (nb - 1u);
        const float msk = ok ? 1.0f : 0.0f;
        const ulong xb  = (ulong)b * 256u;

        float xl[16], xh[16];
        if (skip_x) {
            #pragma unroll
            for (uint t = 0; t < 16u; t++) { xl[t] = 1.0f; xh[t] = 1.0f; }
        } else {
            #pragma unroll
            for (uint t = 0; t < 4u; t++) {
                const float4 v = *(device const float4*)(x + xb + xlo + (t << 2));
                xl[t*4u+0] = v.x; xl[t*4u+1] = v.y; xl[t*4u+2] = v.z; xl[t*4u+3] = v.w;
                const float4 u = *(device const float4*)(x + xb + xhi + (t << 2));
                xh[t*4u+0] = u.x; xh[t*4u+1] = u.y; xh[t*4u+2] = u.z; xh[t*4u+3] = u.w;
            }
        }

        for (uint r = 0; r < MVR; r++) {
            device const uchar* q = ph[r] + (ulong)b * 16u;
            uchar s0, n0, s1, n1;
            k4_scale_min(int(sb0), q + 4u, s0, n0);
            k4_scale_min(int(sb0 + 1u), q + 4u, s1, n1);
            const float dl = rd_f16(q);
            const float dm = rd_f16(q + 2u);
            const float scl = dl * float(s0) * msk;
            const float mnl = dm * float(n0) * msk;
            const float sch = dl * float(s1) * msk;
            const float mnh = dm * float(n1) * msk;
            const uint4 raw = *(device const uint4*)(pd[r] + (ulong)b * 128u + ((ulong)m << 4));
            float a = 0.0f;
            if (skip_alu) {
                // 只把字节加起来：保留权重读取，去掉解包和 fma
                a = float(raw.x ^ raw.y ^ raw.z ^ raw.w);
            } else {
                #pragma unroll
                for (uint i = 0; i < 16u; i++) {
                    const uint word = (i < 4u) ? raw.x : (i < 8u) ? raw.y
                                    : (i < 12u) ? raw.z : raw.w;
                    const uint byte = (word >> ((i & 3u) << 3)) & 0xFFu;
                    a += (scl * float(byte & 0x0Fu) - mnl) * xl[i];
                    a += (sch * float(byte >> 4) - mnh) * xh[i];
                }
            }
            acc[r] += a;
        }
    }
    for (uint r = 0; r < MVR; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && r < nok) y[row0 + r] = s;
    }
}
)MSL";

// 把 GGUF 原始布局重排成 (数据区 | 块头区)。总字节数不变。
// 原始：row r 的 block b 在 r*nb*144 + b*144，块头 16 字节在前。
// 重排：数据在 r*nb*128 + b*128 + m*16（m=0..7），块头在 data_end + (r*nb+b)*16。
static void repack_q4_k(const std::vector<uint8_t>& src, int n_out, int n_in,
                        std::vector<uint8_t>& dst) {
    const int nb = n_in / 256;
    const size_t drow = (size_t)nb * 128u;
    const size_t hoff = (size_t)n_out * drow;
    dst.assign(src.size(), 0);
    for (int r = 0; r < n_out; r++) {
        for (int b = 0; b < nb; b++) {
            const uint8_t* s = src.data() + (size_t)r * (size_t)nb * 144u + (size_t)b * 144u;
            // 块头
            std::memcpy(dst.data() + hoff + ((size_t)r * nb + b) * 16u, s, 16u);
            // 数据区（原始字节 16..143）
            std::memcpy(dst.data() + (size_t)r * drow + (size_t)b * 128u, s + 16, 128u);
        }
    }
}

int main(int argc, char** argv) {
    // 关掉 stdout 缓冲：这个工具会故意跑出非法参数/崩溃来量边界，一旦
    // 段错误，块缓冲里攒的输出全丢，只剩一个 exit=139，等于什么都没测。
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* tname = (argc > 1) ? argv[1] : "q4_k";
    const int n_out = (argc > 2) ? atoi(argv[2]) : 8192;
    const int n_in  = (argc > 3) ? atoi(argv[3]) : 2048;
    const int reps  = (argc > 4) ? atoi(argv[4]) : 12;

    uint32_t type = T_Q4_K;
    const char* kname = "mv_q4_k";
    if (!strcmp(tname, "q6_k")) { type = T_Q6_K; kname = "mv_q6_k"; }
    else if (!strcmp(tname, "f16")) { type = T_F16; kname = "mv_f16"; }
    else if (!strcmp(tname, "q8_0")) { type = T_Q8_0; kname = "mv_q8_0"; }
    // q5_0 必须单列：本机 qwen2.5-0.5b 的 Q4_K_M 里 55% 的字节是 Q5_0
    // （ffn 的 gate/up/down 全是 Q5_0，不是 Q4_K）。不测它等于没测。
    else if (!strcmp(tname, "q5_0")) { type = T_Q5_0; kname = "mv_q5_0"; }
    else if (!strcmp(tname, "q4_0")) { type = T_Q4_0; kname = "mv_q4_0"; }
    const bool packed_ok = (type == T_Q4_K) && (n_in % 256 == 0);

    // 候选内核：和现有内核**同签名、同权重**，只是换了 lane 分工。
    // q4_k 的候选是重排布局（需要额外两块 buffer），q5_0 的候选是 v2
    // （lane 认领整块），后者不需要重排，所以能把两者放进同一套对拍里。
    //
    // q5_0 v2 的生产路径只有 qwen2.5-0.5b 一个模型覆盖，而它的 n_embd=896
    // 只够 nb=28 < 32 —— 也就是 v2 里 `b += 32` 的**多轮循环一次都没跑到**。
    // 这里用任意 n_in 造数据，把那条路补上。
    const char* kname2 = (type == T_Q5_0) ? "mv_q5_0_v2" : nullptr;
    const bool has_v2 = packed_ok || kname2 != nullptr;

    const size_t rbytes = type_row_bytes(type, n_in);
    const size_t wbytes = rbytes * (size_t)n_out;
    std::printf("类型 %s  形状 %d×%d  权重 %.1f MiB\n",
                tname, n_out, n_in, (double)wbytes / 1048576.0);

    int mv_rows = 2;
    if (const char* s = getenv("DUBHE_MV_R")) {
        const int v = atoi(s);
        if (v >= 1 && v <= 16) mv_rows = v;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("没有 Metal 设备\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    // 和 metal.mm 一样：把 MV_R 注入到 MSL 源码最前面
    std::string src = "#define MV_R " + std::to_string(mv_rows) + "\n" + kDubheMSL + kPackedMSL;
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                           options:[MTLCompileOptions new]
                                             error:&e];
    if (!lib) {
        std::printf("编译失败: %s\n", e ? [[e localizedDescription] UTF8String] : "?");
        return 1;
    }
    auto pipeline = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) { std::printf("缺少函数 %s\n", name); return nil; }
        return [dev newComputePipelineStateWithFunction:fn error:&e];
    };
    id<MTLComputePipelineState> ps_old = pipeline(kname);
    id<MTLComputePipelineState> ps_new = packed_ok ? pipeline("mv_q4_k_packed")
                                      : (kname2 ? pipeline(kname2) : nil);
    if (!ps_old) { std::printf("pipeline 失败\n"); return 1; }
    if (has_v2 && !ps_new) { std::printf("候选内核 pipeline 失败\n"); return 1; }

    // 随机权重（Q4_K 的块头 d/dmin 用小的有限值，避免 f16 溢出成 inf）
    std::mt19937 rng(7);
    std::vector<uint8_t> host(wbytes);
    for (size_t i = 0; i < wbytes; i++) host[i] = (uint8_t)(rng() & 0xFF);
    if (type == T_Q4_K) {
        for (size_t b = 0; b + 144 <= wbytes; b += 144) {
            host[b + 0] = 0x00; host[b + 1] = 0x28;   // d    ≈ 0.0625
            host[b + 2] = 0x00; host[b + 3] = 0x2C;   // dmin ≈ 0.125
        }
    }
    std::vector<uint8_t> host_packed;
    if (packed_ok) repack_q4_k(host, n_out, n_in, host_packed);

    std::vector<float> xv((size_t)n_in);
    for (auto& v : xv) v = (float)((int)(rng() % 2001) - 1000) / 1000.0f;

    id<MTLBuffer> xbuf = [dev newBufferWithBytes:xv.data() length:sizeof(float) * n_in
                                         options:MTLResourceStorageModeShared];

    // REP 份**不同**的权重缓冲：如果只用一份重复派发，后面几次会命中 L2，
    // 量出来的是缓存带宽而不是显存带宽。真实前向里每层权重都不一样。
    const int REP = 6;
    id<MTLBuffer> w_old[6], w_new[6], y_old[6], y_new[6];
    for (int i = 0; i < REP; i++) {
        w_old[i] = [dev newBufferWithBytes:host.data() length:wbytes
                                   options:MTLResourceStorageModeShared];
        y_old[i] = [dev newBufferWithLength:sizeof(float) * n_out
                                    options:MTLResourceStorageModeShared];
        if (has_v2) {
            // 重排布局要换成重排后的权重；v2 只是换 lane 分工，权重一模一样
            w_new[i] = [dev newBufferWithBytes:(packed_ok ? host_packed : host).data()
                                       length:wbytes
                                      options:MTLResourceStorageModeShared];
            y_new[i] = [dev newBufferWithLength:sizeof(float) * n_out
                                        options:MTLResourceStorageModeShared];
        }
    }

    uint32_t nin = (uint32_t)n_in, nout = (uint32_t)n_out;
    uint64_t woff = 0;
    const uint64_t hoff = packed_ok ? (uint64_t)((size_t)n_out * (size_t)(n_in / 256) * 128u) : 0;
    const int per_tg = 4 * mv_rows;
    const NSUInteger tgc = (NSUInteger)((n_out + per_tg - 1) / per_tg);

    // 一个"轮次"= 两种内核各派发 REP 次（每次换一份权重），一次 commit。
    // 两种内核在**同一个 command buffer** 里交替，环境漂移对两边同时生效。
    //
    // MVB_PASSES=P 让每轮在 REP 份权重上再循环 P 遍，等于把 dispatch 数乘 P。
    // 目的是分离「固定开销」和「带宽」：同一个形状，N=6 与 N=6P 两点连线，
    // 斜率就是单个 dispatch 的真实代价，截距就是一次 commit 的启动开销。
    // 权重份数（REP=6）不变，所以 P>1 时后面的遍会命中缓存 —— 小形状下
    // 正好用来量**纯固定开销**（缓存里没有带宽成本）。
    static const int kPasses = []() {
        const char* s = getenv("MVB_PASSES");
        const int v = s ? atoi(s) : 1;
        return v >= 1 && v <= 64 ? v : 1;
    }();
    const int kPerRound = REP * kPasses;

    uint32_t flags = 0;
    auto run_round = [&](bool use_new) -> double {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:use_new ? ps_new : ps_old];
        [en setBuffer:xbuf offset:0 atIndex:1];
        [en setBytes:&nin length:4 atIndex:3];
        [en setBytes:&woff length:8 atIndex:4];
        [en setBytes:&nout length:4 atIndex:5];
        // 只有重排布局的候选内核多要两块参数；v2 的签名和现有内核完全一致，
        // 多设 buffer 会踩到「绑定数超过函数声明」的校验。
        if (use_new && packed_ok) {
            [en setBytes:&hoff length:8 atIndex:6];
            [en setBytes:&flags length:4 atIndex:7];
        }
        for (int p = 0; p < kPasses; p++) {
            for (int i = 0; i < REP; i++) {
                [en setBuffer:(use_new ? w_new[i] : w_old[i]) offset:0 atIndex:0];
                [en setBuffer:(use_new ? y_new[i] : y_old[i]) offset:0 atIndex:2];
                [en dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }
        [en endEncoding];
        const double t0 = now_ms();
        [cb commit];
        [cb waitUntilCompleted];
        return (now_ms() - t0) / (double)kPerRound;
    };

    // 诊断矩阵：现有布局 / 重排布局 / 重排但跳过 x / 重排但跳过 fma / 重排全跳。
    // 全部在同一个进程里逐轮交替，只报比值 —— 这个工具早期版本分开跑，
    // 同一份二进制跨轮能差 1.5x，绝对数字没有意义。
    //
    // 「重排布局」这两项**必须**跟 packed_ok 绑在一起。早期版本把
    // {"重排布局", true, 0} 无条件塞进 cases，非 Q4_K 的类型（或 n_in 不是
    // 256 倍数）ps_new 是 nil、w_new 是野指针，第二轮直接段错误 —— 症状是
    // 「exit=139，一行结果都没有」，看着像内核写错了，其实是表里多了一行。
    struct Case { const char* name; bool use_new; uint32_t f; };
    std::vector<Case> cases;
    cases.push_back({"现有布局", false, 0});
    if (packed_ok) {
        cases.push_back({"重排布局", true, 0});
        cases.push_back({"重排·无x读", true, 1});
        cases.push_back({"重排·无ALU", true, 2});
        cases.push_back({"重排·只读权", true, 3});
    } else if (kname2) {
        cases.push_back({"v2（lane 认领整块）", true, 0});
    }
    std::vector<double> best(cases.size(), 1e30);
    for (int r = 0; r < reps + 3; r++) {
        for (size_t i = 0; i < cases.size(); i++) {
            flags = cases[i].f;
            const double t = run_round(cases[i].use_new);
            if (r >= 3) best[i] = std::min(best[i], t);
        }
    }
    const double base_ms = best[0];

    // CPU 参考（只算前 8 行）：按单元编号走，和内核的映射一一对应
    const int chk = std::min(n_out, 8);
    std::vector<float> ref((size_t)chk, 0.f);
    for (int r = 0; r < chk; r++) {
        if (type != T_Q4_K) break;
        const uint8_t* row = host.data() + (size_t)r * rbytes;
        double acc = 0.0;
        const int nb = n_in / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t* blk = row + (size_t)b * 144;
            const float d = fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            const float dm = fp16_to_fp32((uint16_t)(blk[2] | (blk[3] << 8)));
            for (int o = 1; o <= 8; o++) {
                const int k = (o - 1) / 2, g = (o - 1) % 2;
                int sc0 = 0, mn0 = 0, sc1 = 0, mn1 = 0;
                for (int j = 2 * k; j <= 2 * k + 1; j++) {
                    int sc, mn;
                    if (j < 4) { sc = blk[4 + j] & 63; mn = blk[8 + j] & 63; }
                    else {
                        sc = (blk[8 + j] & 0x0F) | ((blk[j] >> 6) << 4);
                        mn = (blk[8 + j] >> 4) | ((blk[4 + j] >> 6) << 4);
                    }
                    if (j == 2 * k) { sc0 = sc; mn0 = mn; } else { sc1 = sc; mn1 = mn; }
                }
                const int e_lo = b * 256 + k * 64 + g * 16;
                for (int i = 0; i < 16; i++) {
                    const int byte = blk[o * 16 + i];
                    acc += (double)(d * (float)sc0 * (float)(byte & 0x0F) - dm * (float)mn0)
                         * (double)xv[(size_t)(e_lo + i)];
                    acc += (double)(d * (float)sc1 * (float)(byte >> 4) - dm * (float)mn1)
                         * (double)xv[(size_t)(e_lo + 32 + i)];
                }
            }
        }
        ref[(size_t)r] = (float)acc;
    }

    auto check = [&](const char* tag, id<MTLBuffer> yb) {
        const float* y = (const float*)[yb contents];
        double maxd = 0.0, maxr = 0.0;
        for (int r = 0; r < chk; r++) {
            maxd = std::max(maxd, (double)std::fabs((double)y[r] - (double)ref[(size_t)r]));
            maxr = std::max(maxr, (double)std::fabs((double)ref[(size_t)r]));
        }
        const double rel = maxd / (maxr + 1e-6);
        std::printf("  %-8s 对拍 rel=%.2e  %s\n", tag, rel, rel < 1e-4 ? "OK" : "**不一致**");
        return rel < 1e-4;
    };

    std::printf("\nMV_R=%d  threadgroup=%d 组 × 128 线程  每轮 %d 个 dispatch"
                "（PASSES=%d）\n\n", mv_rows, (int)tgc, kPerRound, kPasses);
    std::printf("| 内核 | 每 dispatch | 带宽 | 相对现有 |\n|---|---:|---:|---:|\n");
    for (size_t i = 0; i < cases.size(); i++) {
        const double gbs = (double)wbytes / (best[i] * 1e6);
        const double us = best[i] * 1000.0;
        if (i == 0)
            std::printf("| **%s** | %.2f µs | %.1f GB/s | 1.00x |\n",
                        cases[i].name, us, gbs);
        else
            std::printf("| %s | %.2f µs | %.1f GB/s | **%.3fx** |\n",
                        cases[i].name, us, gbs, base_ms / best[i]);
    }
    std::printf("\n");
    // 对拍只覆盖 Q4_K：CPU 参考实现是按 144 字节块写的，别的类型跑它会
    // 拿零当参考值，报出一个假的「不一致」。
    bool ok_new = true;
    if (type == T_Q4_K) {
        ok_new = check("现有", y_old[0]) && ok_new;
        if (packed_ok) ok_new = check("重排", y_new[0]) && ok_new;
    } else if (kname2) {
        // 没有 CPU 参考时，拿**已在生产里跑通**的 v1 当基准。这条覆盖到了
        // nb > 32 的多轮循环 —— 生产模型 n_embd=896 只有 nb=28，跑不到那里。
        const float* a = (const float*)[y_old[0] contents];
        const float* b = (const float*)[y_new[0] contents];
        double maxd = 0.0, maxr = 0.0;
        for (int r = 0; r < n_out; r++) {
            maxd = std::max(maxd, (double)std::fabs((double)a[r] - (double)b[r]));
            maxr = std::max(maxr, (double)std::fabs((double)a[r]));
        }
        const double rel = maxd / (maxr + 1e-6);
        std::printf("  %-8s 对拍 rel=%.2e  %s\n", "v2",
                    rel, rel < 1e-4 ? "OK" : "**不一致**");
        ok_new = ok_new && rel < 1e-4;
    }
    return ok_new ? 0 : 1;
}
