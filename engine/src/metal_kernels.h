// metal_kernels.h — Metal 着色器源码（运行时编译，不依赖离线 Metal Toolchain）
//
// 设计要点：
//   1. 权重在 GPU 侧只有一个 MTLBuffer，直接包住 mmap 的文件页（统一内存，零拷贝）。
//      每个张量用 woff（字节偏移）定位，避免 setBuffer:offset: 的 256 字节对齐限制。
//   2. 一个线程算一行（一个输出神经元）。B=1 解码时这是最自然的并行粒度：
//      n_out 通常 896~151936，远大于 GPU 核数。
//   3. 权重按 GGUF 原始量化格式直读，不解包到内存。
#pragma once

static const char* kDubheMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- 工具
static inline float f16_to_f32(ushort h) { return float(as_type<half>(h)); }

static inline float rd_f16(device const uchar* p) {
    ushort h = ushort(p[0]) | (ushort(p[1]) << 8);
    return f16_to_f32(h);
}

// 一个 32 元块的 float 点积，用 float4 一次算 4 个
// ---------------------------------------------------------------- F32 / F16
kernel void mv_f32(device const uchar* w      [[buffer(0)]],
                   device const float* x      [[buffer(1)]],
                   device float* y            [[buffer(2)]],
                   constant uint& n_in        [[buffer(3)]],
                   constant ulong& woff       [[buffer(4)]],
                   uint row                   [[thread_position_in_grid]])
{
    device const float* p = (device const float*)(w + woff) + (ulong)row * n_in;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < n_in; i += 4) {
        acc += *(device const float4*)(p + i) * *(device const float4*)(x + i);
    }
    y[row] = acc.x + acc.y + acc.z + acc.w;
}

kernel void mv_f16(device const uchar* w      [[buffer(0)]],
                   device const float* x      [[buffer(1)]],
                   device float* y            [[buffer(2)]],
                   constant uint& n_in        [[buffer(3)]],
                   constant ulong& woff       [[buffer(4)]],
                   uint row                   [[thread_position_in_grid]])
{
    device const ushort* p = (device const ushort*)(w + woff) + (ulong)row * n_in;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < n_in; i += 4) {
        acc += float4(f16_to_f32(p[i]), f16_to_f32(p[i+1]), f16_to_f32(p[i+2]), f16_to_f32(p[i+3]))
             * *(device const float4*)(x + i);
    }
    y[row] = acc.x + acc.y + acc.z + acc.w;
}

// ---------------------------------------------------------------- Q4_0（18 字节 / 32 元）
kernel void mv_q4_0(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    uint row                  [[thread_position_in_grid]])
{
    const uint nb = n_in / 32;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 18;
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 18;
        const float d = rd_f16(q);
        device const float* xb = x + b * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; j++) {
            s += xb[j]      * float(int(q[2 + j] & 0x0F) - 8);
            s += xb[j + 16] * float(int(q[2 + j] >> 4)   - 8);
        }
        sum += d * s;
    }
    y[row] = sum;
}

// ---------------------------------------------------------------- Q5_0（22 字节 / 32 元）
kernel void mv_q5_0(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    uint row                  [[thread_position_in_grid]])
{
    const uint nb = n_in / 32;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 22;
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 22;
        const float d = rd_f16(q);
        const uint qh = uint(q[2]) | (uint(q[3]) << 8) | (uint(q[4]) << 16) | (uint(q[5]) << 24);
        device const uchar* qs = q + 6;
        device const float* xb = x + b * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; j++) {
            const int lo = (int(qs[j]) & 0x0F) | (int((qh >> j) & 1u) << 4);
            const int hi = (int(qs[j]) >> 4)   | (int((qh >> (j + 16)) & 1u) << 4);
            s += xb[j]      * float(lo - 16);
            s += xb[j + 16] * float(hi - 16);
        }
        sum += d * s;
    }
    y[row] = sum;
}

// ---------------------------------------------------------------- Q8_0（34 字节 / 32 元）
kernel void mv_q8_0(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    uint row                  [[thread_position_in_grid]])
{
    const uint nb = n_in / 32;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 34;
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 34;
        const float d = rd_f16(q);
        device const char* qs = (device const char*)(q + 2);
        device const float* xb = x + b * 32;
        float s = 0.0f;
        for (int j = 0; j < 32; j++) s += xb[j] * float(qs[j]);
        sum += d * s;
    }
    y[row] = sum;
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
                    uint row                  [[thread_position_in_grid]])
{
    const uint nb = n_in / 256;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 144;
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 144;
        const float d  = rd_f16(q);
        const float dm = rd_f16(q + 2);
        device const uchar* sc = q + 4;
        device const uchar* qs = q + 16;
        device const float* xb = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            k4_scale_min(is + 0, sc, s1, m1);
            k4_scale_min(is + 1, sc, s2, m2);
            const float d1 = d * float(s1), e1 = dm * float(m1);
            const float d2 = d * float(s2), e2 = dm * float(m2);
            device const uchar* qq = qs + (j / 64) * 32;
            float a1 = 0.0f, a2 = 0.0f, sx1 = 0.0f, sx2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                a1  += xb[j + l]      * float(int(qq[l] & 0x0F));
                sx1 += xb[j + l];
                a2  += xb[j + 32 + l] * float(int(qq[l] >> 4));
                sx2 += xb[j + 32 + l];
            }
            sum += d1 * a1 - e1 * sx1 + d2 * a2 - e2 * sx2;
            is += 2;
        }
    }
    y[row] = sum;
}

// ---------------------------------------------------------------- Q6_K（210 字节 / 256 元）
kernel void mv_q6_k(device const uchar* w     [[buffer(0)]],
                    device const float* x     [[buffer(1)]],
                    device float* y           [[buffer(2)]],
                    constant uint& n_in       [[buffer(3)]],
                    constant ulong& woff      [[buffer(4)]],
                    uint row                  [[thread_position_in_grid]])
{
    const uint nb = n_in / 256;
    device const uchar* p = w + woff + (ulong)row * (ulong)nb * 210;
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const uchar* q = p + b * 210;
        device const uchar* ql = q;
        device const uchar* qh = q + 128;
        device const char*  sc = (device const char*)(q + 192);
        const float d = rd_f16(q + 208);
        device const float* xb = x + b * 256;
        for (int nb2 = 0; nb2 < 2; nb2++) {
            device const uchar* l = ql + nb2 * 64;
            device const uchar* h = qh + nb2 * 32;
            device const char*  s = sc + nb2 * 8;
            device const float* xx = xb + nb2 * 128;
            float acc = 0.0f;
            for (int i = 0; i < 32; i++) {
                const int is = i / 16;
                const int q1 = (int(l[i]      & 0x0F) | (int((h[i] >> 0) & 3) << 4)) - 32;
                const int q2 = (int(l[i + 32] & 0x0F) | (int((h[i] >> 2) & 3) << 4)) - 32;
                const int q3 = (int(l[i]      >> 4)   | (int((h[i] >> 4) & 3) << 4)) - 32;
                const int q4 = (int(l[i + 32] >> 4)   | (int((h[i] >> 6) & 3) << 4)) - 32;
                acc += xx[i]      * float(s[is + 0]) * float(q1);
                acc += xx[i + 32] * float(s[is + 2]) * float(q2);
                acc += xx[i + 64] * float(s[is + 4]) * float(q3);
                acc += xx[i + 96] * float(s[is + 6]) * float(q4);
            }
            sum += d * acc;
        }
    }
    y[row] = sum;
}

// ================================================================ 融合算子
//
// 为什么要这些：一次 command buffer 的 commit+wait 在本机实测要 200 µs。
// 如果每个 matvec 都同步一次，0.5B 模型单 token 有 169 次同步 = 34 ms，
// 把 GPU 的算力全埋掉了。所以整层前向的所有算子必须编进同一个 command
// buffer，CPU 只在最外层同步一次。
//
// 下面的算子都是逐元素/规约级别的小东西，放在 GPU 上纯粹是为了"不打断
// 命令流"，不是为了它们本身快。

// 按字节偏移取 float 指针（绕开 setBuffer:offset: 的 256 字节对齐限制）
static inline device float* fat(device float* b, ulong off) {
    return (device float*)((device uchar*)b + off);
}
static inline device const float* fatc(device const float* b, ulong off) {
    return (device const float*)((device const uchar*)b + off);
}

// RMSNorm：out = x * rsqrt(mean(x^2)+eps) * w   （w 为 f32，位于权重 mmap 内）
kernel void k_rmsnorm(device const float* x [[buffer(0)]],
                      device const uchar* w [[buffer(1)]],
                      device float* out     [[buffer(2)]],
                      constant uint& n      [[buffer(3)]],
                      constant float& eps   [[buffer(4)]],
                      constant ulong& woff  [[buffer(5)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]])
{
    float ss = 0.0f;
    for (uint i = tid; i < n; i += tgs) ss += x[i] * x[i];
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
    for (uint i = tid; i < n; i += tgs) out[i] = x[i] * scale * wf[i];
}

// d += s
kernel void k_add(device float* d       [[buffer(0)]],
                  device const float* s [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint i [[thread_position_in_grid]])
{
    if (i < n) d[i] += s[i];
}

// y += b（b 是 mmap 里的 f32 bias，用字节偏移定位）
kernel void k_bias(device float* y          [[buffer(0)]],
                   device const uchar* b    [[buffer(1)]],
                   constant ulong& boff     [[buffer(2)]],
                   constant uint& n         [[buffer(3)]],
                   uint i [[thread_position_in_grid]])
{
    if (i < n) y[i] += ((device const float*)(b + boff))[i];
}

// o = silu(g) * u
kernel void k_silu_mul(device float* o        [[buffer(0)]],
                       device const float* g  [[buffer(1)]],
                       device const float* u  [[buffer(2)]],
                       constant uint& n       [[buffer(3)]],
                       uint i [[thread_position_in_grid]])
{
    if (i < n) {
        const float v = g[i];
        o[i] = (v / (1.0f + exp(-v))) * u[i];
    }
}

// 原地 RoPE：q 就地旋转；k 旋转后直接写进 KV cache 的 pos 槽
kernel void k_rope(device float* q          [[buffer(0)]],
                   device const float* k    [[buffer(1)]],
                   device float* kc         [[buffer(2)]],
                   constant ulong& koff     [[buffer(3)]],
                   constant uint& n_head    [[buffer(4)]],
                   constant uint& n_kvh     [[buffer(5)]],
                   constant uint& hd        [[buffer(6)]],
                   constant uint& pos       [[buffer(7)]],
                   constant float& theta    [[buffer(8)]],
                   uint gid [[thread_position_in_grid]])
{
    const uint hh = hd >> 1;   // 注意：half 是 MSL 内建的 16 位浮点类型名，不能用作变量
    if (gid >= (n_head + n_kvh) * hh) return;
    const uint h = gid / hh;
    const uint i = gid - h * hh;
    const float freq = 1.0f / pow(theta, float(2u * i) / float(hd));
    const float a = float(pos) * freq;
    const float c = cos(a), s = sin(a);
    if (h < n_head) {
        device float* v = q + h * hd;
        const float x0 = v[i], x1 = v[i + hh];
        v[i]      = x0 * c - x1 * s;
        v[i + hh] = x0 * s + x1 * c;
    } else {
        const uint kh = h - n_head;
        device const float* src = k + kh * hd;
        device float* dst = fat(kc, koff) + kh * hd;
        const float x0 = src[i], x1 = src[i + hh];
        dst[i]      = x0 * c - x1 * s;
        dst[i + hh] = x0 * s + x1 * c;
    }
}

// d[doff..] = s[soff..]
kernel void k_copy(device float* d          [[buffer(0)]],
                   constant ulong& doff     [[buffer(1)]],
                   device const float* s    [[buffer(2)]],
                   constant ulong& soff     [[buffer(3)]],
                   constant uint& n         [[buffer(4)]],
                   uint i [[thread_position_in_grid]])
{
    if (i < n) fat(d, doff)[i] = fatc(s, soff)[i];
}

// GQA 因果注意力：一个 simdgroup 负责一个 head，在线 softmax（不落 scores 数组）。
// 一个 threadgroup 装 4 个 simdgroup。要求 hd <= 128（acc[4]）。
kernel void k_attn(device const float* q  [[buffer(0)]],
                   device const float* kc [[buffer(1)]],
                   device const float* vc [[buffer(2)]],
                   device float* out      [[buffer(3)]],
                   constant ulong& coff   [[buffer(4)]],
                   constant uint& n_head  [[buffer(5)]],
                   constant uint& n_kvh   [[buffer(6)]],
                   constant uint& hd      [[buffer(7)]],
                   constant uint& nlen    [[buffer(8)]],
                   constant uint& kv_dim  [[buffer(9)]],
                   constant float& scale  [[buffer(10)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint head = tgid * 4u + sg;
    if (head >= n_head) return;
    const uint mul = n_head / n_kvh;
    const uint kvh = head / mul;
    device const float* qh = q + head * hd;
    device const float* kb = fatc(kc, coff) + kvh * hd;
    device const float* vb = fatc(vc, coff) + kvh * hd;

    float m = -1e30f, l = 0.0f;
    float acc[4];
    for (int j = 0; j < 4; j++) acc[j] = 0.0f;

    for (uint t = 0; t < nlen; t++) {
        device const float* kk = kb + (ulong)t * kv_dim;
        float partial = 0.0f;
        for (uint d = lane; d < hd; d += 32u) partial += qh[d] * kk[d];
        const float s = simd_sum(partial) * scale;

        float w;
        if (s > m) {
            // 在线 softmax：最大值变大时，除了缩放分母 l，还必须把已经累加
            // 的 acc 一起按 exp(m_old - m_new) 缩放，否则历史 key 的贡献
            // 会被高估——这是 flash-attention 最经典的坑。
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

    const float inv = 1.0f / l;
    device float* o = out + head * hd;
    int j = 0;
    for (uint d = lane; d < hd; d += 32u, j++) o[d] = acc[j] * inv;
}
)MSL";
