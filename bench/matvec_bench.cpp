// matvec_bench.cpp — 内核吞吐基准：按真实模型形状逐类型测量
#include "../engine/src/internal.h"

#include <dispatch/dispatch.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace dubhe;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Case { const char* name; uint32_t type; int n_out, n_in; };

int main() {
    // Qwen2.5-0.5B 的真实形状分布
    const Case cases[] = {
        {"attn_q   Q5_0", T_Q5_0, 896, 896},
        {"ffn_gate Q5_0", T_Q5_0, 4864, 896},
        {"ffn_down Q6_K", T_Q6_K, 896, 4864},
        {"out_w    Q8_0", T_Q8_0, 151936, 896},
        {"ref      Q4_K", T_Q4_K, 896, 4864},
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.f, 1.f);

    double total_ms = 0;
    for (const auto& c : cases) {
        const size_t rb = type_row_bytes(c.type, c.n_in);
        if (rb == 0) { std::printf("%-16s 跳过（类型不支持）\n", c.name); continue; }

        std::vector<float> x((size_t)c.n_in);
        for (auto& v : x) v = u(rng);
        std::vector<uint8_t> w(rb * (size_t)c.n_out);
        for (auto& b : w) b = (uint8_t)(rng() & 0xFF);
        // 把每个 block 的 scale 置成有限值
        const int bs = type_block_size(c.type);
        const int bb = type_block_bytes(c.type);
        for (int r = 0; r < c.n_out; r++) {
            uint8_t* p = w.data() + (size_t)r * rb;
            for (int b = 0; b < c.n_in / bs; b++) {
                uint16_t d = 0x2400;
                std::memcpy(p + (size_t)b * bb, &d, 2);
                if (bb >= 4) std::memcpy(p + (size_t)b * bb + 2, &d, 2);
            }
        }
        std::vector<float> y((size_t)c.n_out);

        struct Ctx {
            const float* x;
            const void* xq;
            float* y;
            const uint8_t* w;
            size_t rb;
            uint32_t t;
            int n_in;
            bool q8;
        } ctx{x.data(), nullptr, y.data(), w.data(), rb, c.type, c.n_in, false};
        auto work = [](void* p, size_t i) {
            auto* q = (Ctx*)p;
            q->y[i] = q->q8 ? vec_dot_q8(q->t, q->xq, q->w + i * q->rb, q->n_in)
                            : vec_dot(q->t, q->x, q->w + i * q->rb, q->n_in);
        };

        // warmup
        dispatch_apply_f((size_t)c.n_out, DISPATCH_APPLY_AUTO, &ctx, work);

        const int reps = c.n_out > 50000 ? 5 : 20;
        std::vector<uint8_t> aq(act_q8_supported(c.type) ? act_q8_bytes(c.type, c.n_in) : 0);
        if (!aq.empty()) {
            quantize_act(c.type, x.data(), aq.data(), c.n_in);
            ctx.xq = aq.data();
        }
        auto run = [&](bool q8) {
            ctx.q8 = q8;
            const double t0 = now_ms();
            for (int r = 0; r < reps; r++)
                dispatch_apply_f((size_t)c.n_out, DISPATCH_APPLY_AUTO, &ctx, work);
            const double t1 = now_ms();
            return (t1 - t0) / reps;
        };

        const double f32_ms = run(false);
        const double elems = (double)c.n_out * c.n_in;
        double best = f32_ms;
        std::printf("%-16s f32 %7.3f ms (%5.2f Gelem/s)", c.name, f32_ms, elems / f32_ms / 1e6);
        if (act_q8_supported(c.type)) {
            const double q8_ms = run(true);
            best = q8_ms;
            std::printf("   q8 %7.3f ms (%5.2f Gelem/s)  加速 %.2fx", q8_ms,
                        elems / q8_ms / 1e6, f32_ms / q8_ms);
        }
        std::printf("\n");
        total_ms += best;
    }
    std::printf("\n合计（单 token 全部矩阵乘，取较优路径）%.1f ms → %.1f tok/s（仅内核）\n",
                total_ms, 1000.0 / total_ms);
    return 0;
}
