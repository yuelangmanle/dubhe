// selftest.cpp — 分步自检：加载 → 分词 → 单步前向 → logits 检查
#include <dubhe/dubhe.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("用法: selftest <model.gguf> [text]\n"); return 1; }
    const char* path = argv[1];
    const char* text = argc > 2 ? argv[2] : "你好";

    dubhe_config cfg{};
    cfg.n_ctx = 512;
    cfg.use_gpu = std::getenv("DUBHE_NO_GPU") ? 0 : 1;   // 默认用 GPU，DUBHE_NO_GPU=1 可关
    dubhe_engine* e = dubhe_create(&cfg);
    std::printf("[1] engine %p\n", (void*)e);
    if (dubhe_load_model(e, path) != DUBHE_OK) {
        std::printf("    加载失败: %s\n", dubhe_last_error());
        return 1;
    }
    dubhe_model_info mi{};
    dubhe_model_info_get(e, &mi);
    std::printf("[2] loaded: %s vocab=%d ctx=%d layer=%d embd=%d head=%d/%d ff=%d\n",
                mi.arch, mi.n_vocab, mi.n_ctx_train, mi.n_layer, mi.n_embd, mi.n_head,
                mi.n_head_kv, mi.n_ff);
    std::printf("    backend: %s\n", mi.backend);

    int32_t n = dubhe_tokenize(e, text, 0, nullptr, 0);
    std::printf("[3] tokenize(%s) -> %d tokens\n", text, n);
    if (n <= 0) return 1;
    std::vector<int32_t> ids((size_t)n);
    dubhe_tokenize(e, text, 0, ids.data(), n);
    for (int32_t i = 0; i < n && i < 32; i++) {
        char buf[256];
        int32_t k = dubhe_token_to_piece(e, ids[i], buf, sizeof(buf));
        std::printf("    %5d -> id=%6d  piece=[%.*s]\n", i, ids[i], k > 0 ? k : 0, buf);
    }

    std::printf("[4] generate 8 tokens greedy…\n");
    dubhe_sampling sp{};
    sp.max_tokens = 8;
    sp.temperature = 0.f;
    sp.top_k = 0;
    sp.top_p = 0.f;
    sp.repeat_penalty = 1.f;
    dubhe_stats st{};
    int rc = dubhe_generate(e, ids.data(), n, &sp,
                            [](void*, int32_t tok, const char* piece) {
                                std::printf("    tok=%d piece=[%s]\n", tok, piece ? piece : "(null)");
                            },
                            nullptr, nullptr, &st);
    std::printf("[5] rc=%d err=%s\n", rc, dubhe_last_error());
    std::printf("    prompt %d tok %.1f ms | gen %d tok %.1f ms (%.2f tok/s)\n",
                st.n_prompt, st.t_prompt_ms, st.n_generated, st.t_gen_ms, st.decode_tps);
    dubhe_destroy(e);
    return 0;
}
