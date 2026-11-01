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

    // ---- [6] 前缀复用对拍
    //
    // 多轮对话每轮都把整段历史重发一遍，引擎会跳过"已经算过的前缀"。
    // 这条路径错位的话，输出照样通顺、只是内容不对 —— 最难发现的那种。
    // 所以直接对拍：复用得到的结果必须和冷启动**逐 token 完全一致**。
    std::printf("[6] 前缀复用对拍…\n");
    auto toks = [&](const char* s) {
        int32_t k = dubhe_tokenize(e, s, 0, nullptr, 0);
        std::vector<int32_t> v((size_t)(k > 0 ? k : 0));
        if (k > 0) dubhe_tokenize(e, s, 0, v.data(), k);
        return v;
    };
    auto gen = [&](const std::vector<int32_t>& p, const char* label) {
        std::vector<int32_t> got;
        dubhe_sampling s{};
        s.max_tokens = 12;
        s.temperature = 0.f;
        s.top_k = 0;
        s.top_p = 0.f;
        s.repeat_penalty = 1.f;
        dubhe_stats stl{};
        dubhe_generate(e, p.data(), (int32_t)p.size(), &s,
                       [](void* ud, int32_t tk, const char*) {
                           ((std::vector<int32_t>*)ud)->push_back(tk);
                       },
                       nullptr, &got, &stl);
        std::printf("    %-14s prompt=%3d reused=%3d gen=%2d\n", label, stl.n_prompt,
                    stl.n_reused, stl.n_generated);
        return std::make_pair(got, stl.n_reused);
    };

    int bad = 0;
    const std::vector<int32_t> p1 = ids;
    const auto r1 = gen(p1, "冷启动 p1");
    const int32_t p1_len = (int32_t)p1.size();
    const int32_t cold_len = p1_len + (int32_t)r1.first.size();   // 这一轮 KV 的真实长度

    std::vector<int32_t> p2 = p1;
    p2.insert(p2.end(), r1.first.begin(), r1.first.end());
    const std::vector<int32_t> extra = toks("\n再补一句。");
    p2.insert(p2.end(), extra.begin(), extra.end());

    const auto r2 = gen(p2, "复用 p2");
    if (r2.second != cold_len) {
        std::printf("    ✗ 期望复用 %d 个 token，实际 %d\n", cold_len, r2.second);
        bad++;
    }

    // 分叉：开头一样，中途改一个 token —— 必须整体重来，不能"复用一半"
    std::vector<int32_t> p4 = p2;
    if ((size_t)p1_len < p4.size()) p4[(size_t)p1_len] = (p4[(size_t)p1_len] + 7) % 1000;
    const auto r4 = gen(p4, "分叉 p4");
    if (r4.second != 0) {
        std::printf("    ✗ 分叉的 prompt 不该复用，实际复用了 %d 个\n", r4.second);
        bad++;
    }

    const auto r3 = gen(p2, "冷启动 p2");
    if (r3.second != 0) {
        std::printf("    ✗ 冷启动那轮不该复用，实际 %d 个\n", r3.second);
        bad++;
    }
    if (r2.first != r3.first) {
        std::printf("    ✗ 复用与冷启动结果不一致（%zu vs %zu 个 token）\n", r2.first.size(),
                    r3.first.size());
        for (size_t i = 0; i < r2.first.size() || i < r3.first.size(); i++) {
            const int32_t a = i < r2.first.size() ? r2.first[i] : -1;
            const int32_t b = i < r3.first.size() ? r3.first[i] : -1;
            if (a != b) { std::printf("      首个差异 @%zu: 复用=%d 冷启动=%d\n", i, a, b); break; }
        }
        bad++;
    } else {
        std::printf("    ✓ 复用与冷启动逐 token 一致（%zu 个）\n", r2.first.size());
    }

    // ---- [7] 分词往返：encode → decode → 再 encode，还是同一串 token 吗
    //
    // 前缀复用靠这条性质：上一轮生成的 token，下一轮是拿同样的文本重新
    // 分词得到的。往返不稳，前缀就会在**生成段落中间**对不上，缓存白做。
    // 这里只报不断言 —— 这是分词器的固有性质，不是可以修的 bug。
    std::printf("[7] 分词往返…\n");
    {
        const char* cases[] = {
            "根据材料描述，天枢支持以下加速方式：",
            "Hello, world! 这是一个 mixed 句子。",
            "\n\n1. **Metal GPU 加速** —— 使用 Apple Metal 并行计算\n",
            "用户要求我使用 read_file 工具读取 .gitignore",
        };
        for (const char* s : cases) {
            const std::vector<int32_t> a = toks(s);
            std::string back;
            for (int32_t id : a) {
                char buf[512];
                const int32_t k = dubhe_token_to_piece(e, id, buf, sizeof(buf));
                if (k > 0) back.append(buf, (size_t)k);
            }
            const std::vector<int32_t> b = toks(back.c_str());
            const bool same = (a == b);
            size_t i = 0;
            while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
            std::printf("    %s  n=%2zu→%2zu  @%zu  [%s]\n", same ? "一致" : "不同", a.size(), b.size(),
                        i, s);
        }
    }

    dubhe_destroy(e);
    return bad ? 1 : 0;
}