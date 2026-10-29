// mm_math.h — 多模态塔共用的数值核函数
//
// 视觉塔（mmproj.cpp）和语音塔（audiomm.cpp）的骨架是同一套：量化 matmul、
// NEON 点积、RMSNorm/LayerNorm、SiLU/GELU、softmax 加权求和。这些函数的
// 调优过程写在 mmproj.cpp 的历史里（点积 NEON 化 3 倍、matmul 不整块反量化
// 省掉 74% 的权重流量），语音塔直接复用，不要各写一份 —— 两份实现意味着
// 两份性能差异和两倍的出错面。
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "mmproj.h"   // MmTensor

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace dubhe {
namespace mm {

constexpr int kThreads = 8;

// 分阶段计时，DUBHE_MM_TRACE=1 时在进程退出时打印。塔的性能问题太容易猜错
// （我先后猜过"反量化流量"和"缓存未命中"，都不是主因），所以留个开关，
// 让数据说话。
struct Trace {
    std::atomic<int64_t> ns_mm{0}, ns_deq{0};
    std::atomic<int64_t> n_calls{0}, n_elems{0};
    bool on = std::getenv("DUBHE_MM_TRACE") != nullptr;
    ~Trace() {
        if (!on) return;
        std::fprintf(stderr,
                     "[mm] matmul %.0f ms / %lld 次 / %.1f M 参数次；反量化 %.0f ms\n",
                     ns_mm.load() / 1e6, (long long)n_calls.load(), n_elems.load() / 1e6,
                     ns_deq.load() / 1e6);
    }
};

inline Trace& trace() {
    static Trace t;
    return t;
}

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// ACT2FN["gelu"] 是 erf 版，不是 tanh 近似。两者差 1e-3 量级，
// 在投影器最后一层会被放大成可见的输出差异，所以必须用 erf。
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.70710678f)); }

// 点积。row 是刚反量化出来的 f32 行，x 是某个 token 的激活。
//
// 为什么手写 NEON：同一段循环，标量版单线程 6.2 GFLOP/s，NEON 版 18.6，
// 8 线程 69.7 —— 视觉塔 90% 的时间都在这里，值这三倍。
inline float dot_f32(const float* row, const float* x, int64_t n) {
    int64_t i = 0;
    float s = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(row + i),      vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vld1q_f32(row + i + 4),  vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(row + i + 8),  vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12));
    }
    s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i + 4 <= n; i += 4)
        s += vaddvq_f32(vmulq_f32(vld1q_f32(row + i), vld1q_f32(x + i)));
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (; i + 4 <= n; i += 4) {
        s0 += row[i] * x[i];
        s1 += row[i + 1] * x[i + 1];
        s2 += row[i + 2] * x[i + 2];
        s3 += row[i + 3] * x[i + 3];
    }
    s = (s0 + s1) + (s2 + s3);
#endif
    for (; i < n; i++) s += row[i] * x[i];
    return s;
}

// 同上，但 W 是**量化**张量，不整块反量化。
//
// 为什么值得单独写一个：塔的权重比激活大两个数量级，先 deq_all 成 f32 再乘，
// 等于每次前向把权重搬两遍 —— 全部浪费在搬运上。这里改成每行反量化到栈上的
// 行缓冲（n_in 最多 3072，12 KB，正好留在 L1），乘完就扔，权重只读它本来的
// 大小。反量化的成本也被摊到 n_tok 个 token 上（循环是行在外、token 在内），
// 所以 n_tok 越大越划算。
//
// x_stride：X 的相邻 token 间隔。patchify 那种"每行只取输入的一段"的场合
// 必须传，否则第 t 个 token 会跨进邻居的输入。
inline void matmul_qt(const MmTensor& W, int64_t n_in, int64_t n_out, const float* X,
                      int64_t n_tok, float* Y, int64_t x_stride = 0) {
    auto& tr = trace();
    const int64_t t0 = tr.on ? now_ns() : 0;
    const int64_t xs = x_stride > 0 ? x_stride : n_in;
    const size_t  rb = type_row_bytes(W.type, (int64_t)n_in);
    auto body = [&](int64_t o0, int64_t o1) {
        std::vector<float> row((size_t)n_in);
        for (int64_t o = o0; o < o1; o++) {
            dequant_row(W.type, (const char*)W.data + (size_t)o * rb, row.data(), n_in);
            float* yp = Y + o;
            for (int64_t t = 0; t < n_tok; t++, yp += n_out)
                *yp = dot_f32(row.data(), X + t * xs, n_in);
        }
    };
    if (n_out * n_tok < 64 * 1024) {
        body(0, n_out);
    } else {
        const int64_t chunk = (n_out + kThreads - 1) / kThreads;
        std::vector<std::thread> th;
        th.reserve(kThreads);
        for (int t = 0; t < kThreads; t++) {
            const int64_t o0 = t * chunk, o1 = std::min(n_out, o0 + chunk);
            if (o0 < o1) th.emplace_back(body, o0, o1);
        }
        for (auto& t : th) t.join();
    }
    if (tr.on) {
        tr.ns_mm.fetch_add(now_ns() - t0, std::memory_order_relaxed);
        tr.n_calls.fetch_add(1, std::memory_order_relaxed);
        tr.n_elems.fetch_add(n_out * n_tok, std::memory_order_relaxed);
    }
}

// o += a * v。softmax 之后加权求和。
inline void saxpy(float* o, const float* v, float a, int64_t n) {
    int64_t i = 0;
#if defined(__ARM_NEON)
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 4 <= n; i += 4)
        vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), av, vld1q_f32(v + i)));
#endif
    for (; i < n; i++) o[i] += a * v[i];
}

inline void rmsnorm(float* x, int64_t n, const float* w, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
    for (int64_t i = 0; i < n; i++) x[i] = x[i] * r * w[i];
}

inline void layernorm(float* x, int64_t n, const float* w, const float* b, float eps) {
    double s = 0.0, ss = 0.0;
    for (int64_t i = 0; i < n; i++) { s += x[i]; ss += (double)x[i] * x[i]; }
    const float mean = (float)(s / (double)n);
    const float var  = (float)(ss / (double)n) - mean * mean;
    const float r = 1.0f / std::sqrt(var + eps);
    for (int64_t i = 0; i < n; i++) x[i] = (x[i] - mean) * r * w[i] + b[i];
}

// 把整个张量反量化到 f32 缓冲。小张量（norm、卷积核、偏置）用它就够了。
inline bool deq_all(const MmTensor& t, std::vector<float>& dst, std::string* err) {
    if (!t.ok()) { if (err) *err = "张量缺失"; return false; }
    dst.resize((size_t)t.n);
    dequant_row(t.type, t.data, dst.data(), t.n);
    return true;
}

// 按输出行并行跑一个 (rows × n) 的逐行函数。attention 的 head 循环用它。
template <class F>
inline void parallel_rows(int64_t rows, F&& f) {
    if (rows < 2) { f((int64_t)0, rows); return; }
    const int nt = (int)std::min<int64_t>(kThreads, rows);
    std::vector<std::thread> th;
    th.reserve((size_t)nt);
    const int64_t chunk = (rows + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        const int64_t a = (int64_t)t * chunk, b = std::min(rows, a + chunk);
        if (a < b) th.emplace_back([&f, a, b] { f(a, b); });
    }
    for (auto& t : th) t.join();
}

}  // namespace mm
}  // namespace dubhe
