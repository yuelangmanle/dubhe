// selftest.cpp — 分步自检：加载、后端、分词、生成和前缀复用
#include <dubhe/dubhe.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* backend = nullptr;
    const char* gpu_name = nullptr;
    int arg = 1;
    if (arg + 1 < argc && std::strcmp(argv[arg], "--backend") == 0) {
        backend = argv[arg + 1];
        if (std::strcmp(backend, "metal") == 0) gpu_name = "Metal";
        if (std::strcmp(backend, "vulkan") == 0) gpu_name = "Vulkan";
        arg += 2;
    }
    if (arg >= argc || argc - arg > 2 || std::strcmp(argv[arg], "--backend") == 0 ||
        (backend && !gpu_name && std::strcmp(backend, "cpu") != 0)) {
        std::fprintf(stderr, "用法: selftest [--backend metal|vulkan|cpu] <model.gguf> [text]\n");
        return 2;
    }
    const char* path = argv[arg];
    const char* text = argc > arg + 1 ? argv[arg + 1] : "你好";

    dubhe_config cfg{};
    cfg.n_ctx = 512;
    cfg.use_gpu = backend ? gpu_name != nullptr : !std::getenv("DUBHE_NO_GPU");
    const std::unique_ptr<dubhe_engine, decltype(&dubhe_destroy)> engine(dubhe_create(&cfg), dubhe_destroy);
    dubhe_engine* e = engine.get();
    if (!e) {
        std::fprintf(stderr, "[1] 创建引擎失败\n");
        return 1;
    }
    std::printf("[1] 引擎已创建\n");
    if (dubhe_load_model(e, path) != DUBHE_OK) {
        std::fprintf(stderr, "    加载失败: %s\n", dubhe_last_error());
        return 1;
    }

    auto backend_ok = [&](const dubhe_model_info& info) {
        if (!backend || (info.is_gpu == cfg.use_gpu &&
            (!gpu_name || std::strncmp(info.backend, gpu_name, std::strlen(gpu_name)) == 0))) return true;
        std::fprintf(stderr, "    需要 %s 后端，实际后端 %s（is_gpu=%d）\n",
                     backend, info.backend, info.is_gpu);
        return false;
    };
    dubhe_model_info mi{};
    if (dubhe_model_info_get(e, &mi) != DUBHE_OK) {
        std::fprintf(stderr, "    读取模型信息失败: %s\n", dubhe_last_error());
        return 1;
    }
    std::printf("[2] loaded: %s vocab=%d ctx=%d layer=%d embd=%d head=%d/%d ff=%d\n",
                mi.arch, mi.n_vocab, mi.n_ctx_train, mi.n_layer, mi.n_embd, mi.n_head,
                mi.n_head_kv, mi.n_ff);
    std::printf("    backend: %s\n", mi.backend);
    if (!backend_ok(mi)) return 1;

    auto toks = [&](const char* s) {
        const int32_t count = dubhe_tokenize(e, s, 0, nullptr, 0);
        if (count <= 0) return std::vector<int32_t>{};
        std::vector<int32_t> ids((size_t)count);
        if (dubhe_tokenize(e, s, 0, ids.data(), count) != count) ids.clear();
        return ids;
    };
    auto piece = [&](int32_t id) {
        const int32_t count = dubhe_token_to_piece(e, id, nullptr, 0);
        if (count <= 0) return std::string{};
        std::string decoded((size_t)count, '\0');
        dubhe_token_to_piece(e, id, decoded.data(), count);
        return decoded;
    };
    const std::vector<int32_t> ids = toks(text);
    std::printf("[3] tokenize(%s) -> %zu tokens\n", text, ids.size());
    if (ids.empty()) {
        std::fprintf(stderr, "    输入分词失败或为空\n");
        return 1;
    }
    for (size_t i = 0; i < ids.size() && i < 32; i++) {
        std::printf("    %5zu -> id=%6d  piece=[%s]\n", i, ids[i], piece(ids[i]).c_str());
    }

    auto generate = [&](const std::vector<int32_t>& prompt, int max_tokens, const char* label,
                        dubhe_token_cb callback, void* data, dubhe_stats& stats) {
        dubhe_sampling sampling{};
        sampling.max_tokens = max_tokens;
        sampling.repeat_penalty = 1.f;
        const int rc = dubhe_generate(e, prompt.data(), (int32_t)prompt.size(), &sampling,
                                      callback, nullptr, data, &stats);
        if (rc != DUBHE_OK) {
            std::fprintf(stderr, "    %s 生成失败 (%d): %s\n", label, rc, dubhe_last_error());
            return false;
        }
        return true;
    };
    std::printf("[4] generate 8 tokens greedy…\n");
    dubhe_stats st{};
    if (!generate(ids, 8, "首次", [](void*, int32_t tok, const char* decoded) {
            std::printf("    tok=%d piece=[%s]\n", tok, decoded ? decoded : "(null)");
        }, nullptr, st)) return 1;
    std::printf("[5] prompt %d tok %.1f ms | gen %d tok %.1f ms (%.2f tok/s)\n",
                st.n_prompt, st.t_prompt_ms, st.n_generated, st.t_gen_ms, st.decode_tps);

    // ---- [6] 前缀复用对拍
    //
    // 多轮对话每轮都把整段历史重发一遍，引擎会跳过"已经算过的前缀"。
    // 这条路径错位的话，输出照样通顺、只是内容不对 —— 最难发现的那种。
    // 所以直接对拍：复用得到的结果必须和冷启动**逐 token 完全一致**。
    std::printf("[6] 前缀复用对拍…\n");
    struct Sample {
        std::vector<int32_t> tokens;
        int32_t reused = 0;
    };
    auto gen = [&](const std::vector<int32_t>& p, const char* label, Sample& sample) {
        dubhe_stats stl{};
        if (!generate(p, 12, label, [](void* ud, int32_t tk, const char*) {
                static_cast<Sample*>(ud)->tokens.push_back(tk);
            }, &sample, stl)) return false;
        std::printf("    %-14s prompt=%3d reused=%3d gen=%2d\n", label, stl.n_prompt,
                    stl.n_reused, stl.n_generated);
        sample.reused = stl.n_reused;
        return true;
    };

    int bad = 0;
    Sample r1, r2, r3, r4;
    if (!gen(ids, "冷启动 p1", r1)) return 1;
    const int32_t cold_len = (int32_t)(ids.size() + r1.tokens.size());

    const std::vector<int32_t> extra = toks("\n再补一句。");
    if (extra.empty() || mi.n_vocab < 2) {
        std::fprintf(stderr, "    无法构造前缀复用用例\n");
        return 1;
    }
    std::vector<int32_t> p2 = ids;
    p2.insert(p2.end(), r1.tokens.begin(), r1.tokens.end());
    p2.insert(p2.end(), extra.begin(), extra.end());

    if (!gen(p2, "复用 p2", r2)) return 1;
    if (r2.reused != cold_len) {
        std::printf("    ✗ 期望复用 %d 个 token，实际 %d\n", cold_len, r2.reused);
        bad++;
    }

    // 分叉：开头一样，中途改一个 token —— 必须整体重来，不能"复用一半"
    std::vector<int32_t> p4 = p2;
    int32_t& fork_token = p4[ids.size()];
    fork_token = fork_token == 0 ? 1 : 0;
    if (!gen(p4, "分叉 p4", r4)) return 1;
    if (r4.reused != 0) {
        std::printf("    ✗ 分叉的 prompt 不该复用，实际复用了 %d 个\n", r4.reused);
        bad++;
    }

    if (!gen(p2, "冷启动 p2", r3)) return 1;
    if (r3.reused != 0) {
        std::printf("    ✗ 冷启动那轮不该复用，实际 %d 个\n", r3.reused);
        bad++;
    }
    if (r2.tokens != r3.tokens) {
        std::printf("    ✗ 复用与冷启动结果不一致（%zu vs %zu 个 token）\n", r2.tokens.size(),
                    r3.tokens.size());
        for (size_t i = 0; i < r2.tokens.size() || i < r3.tokens.size(); i++) {
            const int32_t a = i < r2.tokens.size() ? r2.tokens[i] : -1;
            const int32_t b = i < r3.tokens.size() ? r3.tokens[i] : -1;
            if (a != b) { std::printf("      首个差异 @%zu: 复用=%d 冷启动=%d\n", i, a, b); break; }
        }
        bad++;
    } else {
        std::printf("    ✓ 复用与冷启动逐 token 一致（%zu 个）\n", r2.tokens.size());
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
            for (int32_t id : a) back += piece(id);
            const std::vector<int32_t> b = toks(back.c_str());
            const bool same = (a == b);
            size_t i = 0;
            while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
            std::printf("    %s  n=%2zu→%2zu  @%zu  [%s]\n", same ? "一致" : "不同", a.size(), b.size(),
                        i, s);
        }
    }

    if (dubhe_model_info_get(e, &mi) != DUBHE_OK || !backend_ok(mi)) return 1;
    return bad ? 1 : 0;
}
