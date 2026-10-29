// tokprobe.cpp — 分词与贪心生成的可复现探针
//
//   ./build/tokprobe model.gguf "The capital of France is" [n_gen]
//
// 用途：判断"同一模型的两个量化档输出不同"到底是分词差异还是前向差异。
// 如果两个模型打印出的 token id 序列完全一致、只有生成结果不同，那就是前向；
// 如果分词 id 就不一样，说明是 tokenizer 元数据（pre/BOS/合并表）的差异。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dubhe/dubhe.h"

static void on_tok(void* ud, int32_t id, const char* piece) {
    auto* out = static_cast<std::string*>(ud);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%d", out->empty() ? "" : " ", id);
    *out += buf;
    std::printf("  gen id=%-7d %s\n", id, piece ? piece : "");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "用法: tokprobe <model.gguf> <text> [n_gen]\n");
        return 2;
    }
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 8;

    dubhe_config cfg{};
    cfg.n_threads = 4;
    cfg.n_ctx = 2048;
    cfg.use_gpu = std::getenv("TOK_GPU") ? 1 : 0;
    cfg.n_batch = 0;
    dubhe_engine* e = dubhe_create(&cfg);
    if (!e || dubhe_load_model(e, argv[1]) != DUBHE_OK) {
        std::fprintf(stderr, "加载失败: %s\n", dubhe_last_error());
        return 1;
    }

    static int32_t ids[8192];
    const int32_t n = dubhe_tokenize(e, argv[2], 0, ids, 8192);
    std::printf("tokens=%d\n", n);
    for (int32_t i = 0; i < n; i++) {
        char buf[128] = {0};
        const int32_t len = dubhe_token_to_piece(e, ids[i], buf, (int32_t)sizeof(buf) - 1);
        if (len < 0) buf[0] = 0;
        std::printf("  [%3d] id=%-7d |%s|\n", i, ids[i], buf);
    }

    dubhe_sampling sp{};
    sp.max_tokens = n_gen;
    sp.temperature = 0.f;
    sp.top_k = 0;
    sp.top_p = 1.f;
    sp.repeat_penalty = 1.f;
    std::string seq;
    dubhe_stats st{};
    if (dubhe_generate(e, ids, n, &sp, on_tok, nullptr, &seq, &st) != DUBHE_OK) {
        std::fprintf(stderr, "生成失败: %s\n", dubhe_last_error());
        dubhe_destroy(e);
        return 1;
    }
    std::printf("greedy_ids=%s\n", seq.c_str());
    std::printf("prefill=%.1f tok/s  decode=%.1f tok/s\n", st.prefill_tps, st.decode_tps);
    dubhe_destroy(e);
    return 0;
}
