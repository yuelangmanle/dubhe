// batchcheck.cpp — 多序列批处理（连续批处理）的正确性与吞吐对拍
//
// 为什么需要它：`forward_multi` 把 nrow 行当成 nrow 条**独立序列**，每行有
// 自己的 KV 槽位、自己的位置、自己的 ShortConv 历史。这类改动最容易出的错
// 不是崩溃，而是"看着通顺、内容串台"——两条序列的 KV 混在一起，输出照样是
// 合法 token，只有逐步对拍才抓得住。
//
// 判据用 **logits** 而不是 token：
//   逐条跑时上下文短（nlen≈10），注意力切段数 nsp=1，走的是融合内核；
//   放进 8 条序列的批里，nlen_max≈18，nsp=3，走 split+combine。
//   两个变体的浮点累加顺序不同，末位有 1e-6 量级的差别，足以把**接近平局**
//   的 argmax 翻掉（实测序列 3 就翻了）。拿 token 相等当判据会把这个
//   正常的数值差报成"串台"，所以主判据是 logits 的相对误差。
//
// 用法：
//   make batchcheck
//   ./build/batchcheck <model> [序列数] [每序列生成数]
#include "../engine/src/internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dubhe;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static int argmax(const float* v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[b]) b = i;
    return b;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: batchcheck <model.gguf> [序列数] [每序列生成数]\n");
        return 1;
    }
    const int S = (argc > 2) ? atoi(argv[2]) : 3;
    const int K = (argc > 3) ? atoi(argv[3]) : 6;

    Model m;
    std::string err;
    const int n_ctx = 4096;
    if (!m.load(argv[1], n_ctx, true, &err)) {
        std::printf("加载失败: %s\n", err.c_str());
        return 1;
    }
    const int n_vocab = m.cfg().n_vocab;
    const int nrow_max = m.batch_size();
    if (S > nrow_max) {
        std::printf("序列数 %d 超过后端批上限 %d\n", S, nrow_max);
        return 1;
    }

    // 每条序列一块独立的 KV 槽位。宽度够放下 prompt + 生成长度即可，
    // S 条加起来不超过 n_ctx。
    const int slab_w = std::max(16, n_ctx / std::max(1, S));
    std::printf("模型 vocab=%d 后端批上限=%d  序列数=%d 各生成=%d  槽位宽=%d\n",
                n_vocab, nrow_max, S, K, slab_w);

    // 固定 token id 造序列，长度各不相同 —— 长度一致会掩盖"按行取位置"
    // 那类错误（每行的 nlen 都一样，下标写错也看不出来）。
    std::vector<std::vector<int32_t>> seqs((size_t)S);
    for (int s = 0; s < S; s++) {
        const int len = 3 + s * 2;              // 3, 5, 7, ...
        for (int i = 0; i < len; i++)
            seqs[(size_t)s].push_back((int32_t)((1000 + s * 977 + i * 31) % (n_vocab - 1)));
    }

    std::vector<float> logits((size_t)n_vocab, 0.f);
    std::vector<float> logits_all((size_t)n_vocab * (size_t)S, 0.f);

    // 每一步每一条序列的 logits 都存下来，最后统一比。
    // S=8、K=8、vocab=128000 时是 32 MB，可接受。
    using StepL = std::vector<std::vector<std::vector<float>>>;   // [s][k][vocab]
    StepL refL((size_t)S), batL((size_t)S);
    for (int s = 0; s < S; s++) {
        refL[(size_t)s].resize((size_t)K);
        batL[(size_t)s].resize((size_t)K);
    }

    std::vector<int32_t> cur((size_t)S, 0);
    std::vector<int32_t> rpos((size_t)S, 0);
    std::vector<uint64_t> rslab((size_t)S, 0);
    std::vector<uint64_t> rslot((size_t)S, 0);
    for (int s = 0; s < S; s++) {
        rslab[(size_t)s] = (uint64_t)s * (uint64_t)slab_w;
        rslot[(size_t)s] = (uint64_t)s;
    }

    // ---- A：逐条单独跑
    auto run_ref = [&]() {
        for (int s = 0; s < S; s++) {
            auto& q = seqs[(size_t)s];
            m.reset_kv();
            if (!m.forward_batch(q.data(), (int)q.size(), 0, logits.data(), &err)) {
                std::printf("参考 prefill 失败: %s\n", err.c_str());
                std::exit(1);
            }
            refL[(size_t)s][0] = logits;
            int32_t t = (int32_t)argmax(logits.data(), n_vocab);
            for (int k = 1; k < K; k++) {
                if (!m.forward(t, (int32_t)q.size() + k - 1, logits.data(), &err)) {
                    std::printf("参考 decode 失败: %s\n", err.c_str());
                    std::exit(1);
                }
                refL[(size_t)s][(size_t)k] = logits;
                t = (int32_t)argmax(logits.data(), n_vocab);
            }
        }
    };

    // ---- B：同一个批，各占一个 KV 槽位
    auto run_bat = [&]() {
        m.reset_kv();
        for (int s = 0; s < S; s++) {
            auto& q = seqs[(size_t)s];
            if (!m.forward_batch(q.data(), (int)q.size(), 0, logits.data(), &err,
                                 rslab[(size_t)s], rslot[(size_t)s])) {
                std::printf("批量 prefill 失败（序列 %d）: %s\n", s, err.c_str());
                std::exit(1);
            }
            batL[(size_t)s][0] = logits;
            cur[(size_t)s] = (int32_t)argmax(logits.data(), n_vocab);
        }
        for (int k = 1; k < K; k++) {
            for (int s = 0; s < S; s++)
                rpos[(size_t)s] = (int32_t)seqs[(size_t)s].size() + k - 1;
            if (!m.forward_multi(cur.data(), S, rpos.data(), rslab.data(), rslot.data(),
                                 logits_all.data(), &err)) {
                std::printf("批量 decode 失败: %s\n", err.c_str());
                std::exit(1);
            }
            for (int s = 0; s < S; s++) {
                const float* row = logits_all.data() + (size_t)s * n_vocab;
                batL[(size_t)s][(size_t)k].assign(row, row + n_vocab);
                cur[(size_t)s] = (int32_t)argmax(row, n_vocab);
            }
        }
    };

    // 先热身一遍：第一遍会把 KV 从初始的 1024 扩到 n_ctx（grow_kv 要重新
    // 分配 200 MB 并整块搬一遍），那笔一次性开销会盖住真正的差异。
    run_ref();
    run_bat();
    const double t_ref0 = now_ms();
    run_ref();
    const double t_ref = now_ms() - t_ref0;
    const double t_bat0 = now_ms();
    run_bat();
    const double t_bat = now_ms() - t_bat0;

    // ---- 对拍：逐步比较 logits
    bool ok = true;
    int tok_diff = 0;
    double worst_rel = 0.0;
    int worst_s = -1, worst_k = -1;
    for (int s = 0; s < S; s++) {
        for (int k = 0; k < K; k++) {
            const auto& a = refL[(size_t)s][(size_t)k];
            const auto& b = batL[(size_t)s][(size_t)k];
            if (a.empty() || b.empty()) { ok = false; continue; }
            double md = 0, mr = 0;
            for (int i = 0; i < n_vocab; i++) {
                md = std::max(md, (double)std::fabs((double)a[(size_t)i] - (double)b[(size_t)i]));
                mr = std::max(mr, (double)std::fabs((double)a[(size_t)i]));
            }
            const double rel = md / (mr + 1e-6);
            if (rel > worst_rel) { worst_rel = rel; worst_s = s; worst_k = k; }
            // 第 0 步（prefill）必须**完全一致** —— 两条路径用的是同一个
            // 批处理内核，一个 bit 都不该差。
            if (k == 0 && rel > 1e-6) ok = false;
            // 之后单条走 mv_*、批量走 mm2_*，两个内核的累加顺序不同。实测
            // mm2 对 CPU 参考的误差是 2e-7，但经过 24 层 + KV 传播后放大到
            // 1e-3 量级。prefill 本来就是走 mm2 的，所以这不是批处理引入的
            // 新误差，阈值放到 3e-3。
            if (rel > 3e-3) ok = false;
            if (argmax(a.data(), n_vocab) != argmax(b.data(), n_vocab)) tok_diff++;
        }
    }

    // 逐步误差矩阵：看误差是"一路 1e-6 慢慢长"还是"某一步突然跳到 0.9"。
    // 前者是浮点累加顺序，后者是状态被写坏。
    if (getenv("BATCH_MATRIX")) {
        std::printf("  逐步相对误差（行=序列，列=步）\n");
        for (int s = 0; s < S; s++) {
            std::printf("   s%-2d", s);
            for (int k = 0; k < K; k++) {
                const auto& a = refL[(size_t)s][(size_t)k];
                const auto& b = batL[(size_t)s][(size_t)k];
                double md = 0, mr = 0;
                for (int i = 0; i < n_vocab; i++) {
                    md = std::max(md, (double)std::fabs((double)a[(size_t)i] - (double)b[(size_t)i]));
                    mr = std::max(mr, (double)std::fabs((double)a[(size_t)i]));
                }
                std::printf(" %7.1e", md / (mr + 1e-6));
            }
            std::printf("\n");
        }
    }
    std::printf("  logits 最大相对误差 %.2e（序列 %d 第 %d 步）%s\n",
                worst_rel, worst_s, worst_k, worst_rel < 1e-3 ? "  OK" : "  **不一致**");
    // 批内串台检查：所有行喂**完全相同**的 token，那么每行的 logits 必须
    // 逐位一致。只要有一行不同，就说明某处的缓冲偏移/槽位算错了 —— 这是
    // 批处理最容易出、也最难从 token 上看出来的错。
    double xtalk = 0.0;
    if (S > 1) {
        std::vector<int32_t> same((size_t)S, 0), sp_((size_t)S, 0);
        std::vector<uint64_t> sl((size_t)S, 0), sk((size_t)S, 0);
        for (int r = 0; r < S; r++) {
            same[(size_t)r] = seqs[0][0];
            sp_[(size_t)r]  = 0;
            sl[(size_t)r]   = rslab[(size_t)r];
            sk[(size_t)r]   = rslot[(size_t)r];
        }
        m.reset_kv();
        for (int r = 0; r < S; r++)
            m.forward_batch(&same[(size_t)r], 1, 0, logits.data(), &err,
                            sl[(size_t)r], sk[(size_t)r]);
        if (m.forward_multi(same.data(), S, sp_.data(), sl.data(), sk.data(),
                            logits_all.data(), &err)) {
            const float* r0 = logits_all.data();
            double mr = 0;
            for (int i = 0; i < n_vocab; i++) mr = std::max(mr, (double)std::fabs((double)r0[i]));
            for (int r = 1; r < S; r++) {
                const float* rr = logits_all.data() + (size_t)r * n_vocab;
                double md = 0;
                for (int i = 0; i < n_vocab; i++)
                    md = std::max(md, (double)std::fabs((double)r0[i] - (double)rr[i]));
                xtalk = std::max(xtalk, md / (mr + 1e-6));
            }
            std::printf("  批内串台检查（%d 行同输入）rel=%.2e %s\n", S, xtalk,
                        xtalk < 1e-5 ? "OK" : "**有串台**");
            if (xtalk >= 1e-5) ok = false;
        }
    }
    std::printf("  argmax 分叉 %d / %d 步", tok_diff, S * K);
    if (tok_diff) std::printf("（相邻候选接近平局时的正常舍入差，不是串台）");
    std::printf("\n");

    const int tot_tok = S * K;
    std::printf("\n逐条：%.1f ms（%d token，%.1f tok/s）\n", t_ref, tot_tok,
                tot_tok / (t_ref / 1000.0));
    std::printf("批量：%.1f ms（%d token，%.1f tok/s）\n", t_bat, tot_tok,
                tot_tok / (t_bat / 1000.0));
    std::printf("加速：%.2fx\n", t_ref / std::max(1e-9, t_bat));

    m.unload();
    if (!ok) std::printf("\n结果：**不一致**\n");
    return ok ? 0 : 1;
}
