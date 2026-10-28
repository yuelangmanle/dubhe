// main.cpp — dubhe CLI
#include <dubhe/dubhe.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string cmd;
    std::string model;
    std::string prompt;
    int   n_predict = 256;
    float temp = 0.7f;
    int   top_k = 40;
    float top_p = 0.95f;
    float rep = 1.1f;
    bool  stats = false;
    bool  raw = false;
    int   ctx = 0;
    int   threads = 0;
    bool  no_gpu = false;
    int   concurrency = 1;
};

void usage() {
    std::printf(
        "dubhe %s — 本地推理引擎\n\n"
        "用法:\n"
        "  dubhe inspect -m <model.gguf>\n"
        "  dubhe run     -m <model.gguf> -p \"提示词\" [-n 256] [--temp 0.7] [--raw] [--stats]\n"
        "  dubhe chat    -m <model.gguf> [--temp 0.7] [-c 4096]\n"
        "  dubhe bench   -m <model.gguf> [-n 128] [--concurrency 1]\n\n"
        "选项:\n"
        "  -m, --model <path>     GGUF 模型路径\n"
        "  -p, --prompt <text>    提示词\n"
        "  -n, --predict <N>      最多生成 token 数 (默认 256)\n"
        "  -c, --ctx <N>          上下文长度 (默认取模型训练值)\n"
        "  -t, --threads <N>      线程数 (默认自动)\n"
        "      --temp <F>         温度，0 = greedy (默认 0.7)\n"
        "      --top-k <N>        (默认 40)\n"
        "      --top-p <F>        (默认 0.95)\n"
        "      --repeat <F>       重复惩罚 (默认 1.1)\n"
        "      --raw              不加 chat 模板，直接续写\n"
        "      --stats            打印性能统计\n"
        "      --no-gpu           禁用 GPU 后端\n"
        "      --concurrency <N>  bench 并发数\n",
        dubhe_version());
}

bool parse(int argc, char** argv, Args* a) {
    if (argc < 2) { usage(); return false; }
    a->cmd = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&](const char** v) -> bool {
            if (i + 1 >= argc) { std::fprintf(stderr, "缺少参数值: %s\n", k.c_str()); return false; }
            *v = argv[++i];
            return true;
        };
        const char* v = nullptr;
        if (k == "-m" || k == "--model") { if (!next(&v)) return false; a->model = v; }
        else if (k == "-p" || k == "--prompt") { if (!next(&v)) return false; a->prompt = v; }
        else if (k == "-n" || k == "--predict") { if (!next(&v)) return false; a->n_predict = atoi(v); }
        else if (k == "-c" || k == "--ctx") { if (!next(&v)) return false; a->ctx = atoi(v); }
        else if (k == "-t" || k == "--threads") { if (!next(&v)) return false; a->threads = atoi(v); }
        else if (k == "--temp") { if (!next(&v)) return false; a->temp = (float)atof(v); }
        else if (k == "--top-k") { if (!next(&v)) return false; a->top_k = atoi(v); }
        else if (k == "--top-p") { if (!next(&v)) return false; a->top_p = (float)atof(v); }
        else if (k == "--repeat") { if (!next(&v)) return false; a->rep = (float)atof(v); }
        else if (k == "--concurrency") { if (!next(&v)) return false; a->concurrency = atoi(v); }
        else if (k == "--stats") a->stats = true;
        else if (k == "--raw") a->raw = true;
        else if (k == "--no-gpu") a->no_gpu = true;
        else if (k == "-h" || k == "--help") { usage(); return false; }
        else if (!k.empty() && k[0] != '-') {
            // 位置参数：第一个当模型路径，第二个当提示词
            if (a->model.empty()) a->model = k;
            else if (a->prompt.empty()) a->prompt = k;
            else { std::fprintf(stderr, "多余参数: %s\n", k.c_str()); return false; }
        }
        else { std::fprintf(stderr, "未知参数: %s\n", k.c_str()); return false; }
    }
    return true;
}

void print_stats(const dubhe_stats& s) {
    std::fprintf(stderr,
                 "\n[统计] prompt %d tok / %.1f ms (%.1f tok/s) | gen %d tok / %.1f ms (%.1f tok/s)\n",
                 s.n_prompt, s.t_prompt_ms, s.prefill_tps, s.n_generated, s.t_gen_ms, s.decode_tps);
}

// ChatML：Qwen 系
std::string build_chatml(const std::vector<std::pair<std::string, std::string>>& msgs) {
    std::string p;
    for (auto& m : msgs) {
        p += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
    }
    p += "<|im_start|>assistant\n";
    return p;
}

void cb_token(void* ud, int32_t, const char* piece) {
    (void)ud;
    if (piece) std::fputs(piece, stdout);
    std::fflush(stdout);
}

int cmd_inspect(const Args& a) {
    if (a.model.empty()) { std::fprintf(stderr, "需要 -m <model.gguf>\n"); return 1; }
    dubhe_config cfg{};
    cfg.n_threads = a.threads;
    cfg.n_ctx = a.ctx;
    cfg.use_gpu = a.no_gpu ? 0 : 1;
    dubhe_engine* e = dubhe_create(&cfg);
    if (!e) { std::fprintf(stderr, "引擎创建失败\n"); return 1; }

    if (dubhe_load_model(e, a.model.c_str()) != DUBHE_OK) {
        std::fprintf(stderr, "加载失败: %s\n", dubhe_last_error());
        dubhe_destroy(e);
        return 1;
    }
    dubhe_model_info mi{};
    dubhe_model_info_get(e, &mi);

    std::printf("模型        %s\n", a.model.c_str());
    std::printf("架构        %s\n", mi.arch);
    std::printf("参数量      %.2f B\n", (double)mi.n_params / 1e9);
    std::printf("文件大小    %.2f GB\n", (double)mi.file_size / (1024.0 * 1024 * 1024));
    std::printf("量化        %s\n", mi.quant_desc);
    std::printf("层数        %d\n", mi.n_layer);
    std::printf("隐藏维度    %d\n", mi.n_embd);
    std::printf("注意力头    %d (KV %d, head_dim %d)\n", mi.n_head, mi.n_head_kv,
                mi.n_head > 0 ? mi.n_embd / mi.n_head : 0);
    std::printf("FFN 维度    %d\n", mi.n_ff);
    std::printf("词表        %d\n", mi.n_vocab);
    std::printf("训练上下文  %d\n", mi.n_ctx_train);
    std::printf("Chat 模板   %s\n", mi.has_chat_template ? "ChatML" : "无");
    std::printf("计算后端    %s\n", mi.backend);
    std::printf("硬件        %s\n", dubhe_hardware_info());

    dubhe_destroy(e);
    return 0;
}

int cmd_run(const Args& a) {
    if (a.model.empty() || a.prompt.empty()) {
        std::fprintf(stderr, "需要 -m 与 -p\n");
        return 1;
    }
    dubhe_config cfg{};
    cfg.n_threads = a.threads;
    cfg.n_ctx = a.ctx;
    cfg.use_gpu = a.no_gpu ? 0 : 1;
    dubhe_engine* e = dubhe_create(&cfg);
    if (!e) return 1;
    if (dubhe_load_model(e, a.model.c_str()) != DUBHE_OK) {
        std::fprintf(stderr, "加载失败: %s\n", dubhe_last_error());
        dubhe_destroy(e);
        return 1;
    }

    std::string text = a.raw ? a.prompt
                             : build_chatml({{"user", a.prompt}});

    int32_t n = dubhe_tokenize(e, text.c_str(), 1, nullptr, 0);
    std::vector<int32_t> toks((size_t)n);
    dubhe_tokenize(e, text.c_str(), 1, toks.data(), n);

    dubhe_sampling sp{};
    sp.max_tokens = a.n_predict;
    sp.temperature = a.temp;
    sp.top_k = a.top_k;
    sp.top_p = a.top_p;
    sp.repeat_penalty = a.rep;
    sp.repeat_last_n = 64;

    dubhe_stats st{};
    int rc = dubhe_generate(e, toks.data(), n, &sp, cb_token, nullptr, nullptr, &st);
    if (rc != DUBHE_OK) std::fprintf(stderr, "\n生成失败: %s\n", dubhe_last_error());
    std::printf("\n");
    if (a.stats) print_stats(st);

    dubhe_destroy(e);
    return rc == DUBHE_OK ? 0 : 1;
}

int cmd_chat(const Args& a) {
    if (a.model.empty()) { std::fprintf(stderr, "需要 -m\n"); return 1; }
    dubhe_config cfg{};
    cfg.n_threads = a.threads;
    cfg.n_ctx = a.ctx > 0 ? a.ctx : 4096;
    cfg.use_gpu = a.no_gpu ? 0 : 1;
    dubhe_engine* e = dubhe_create(&cfg);
    if (!e) return 1;
    std::fprintf(stderr, "加载模型中…\n");
    if (dubhe_load_model(e, a.model.c_str()) != DUBHE_OK) {
        std::fprintf(stderr, "加载失败: %s\n", dubhe_last_error());
        dubhe_destroy(e);
        return 1;
    }
    dubhe_model_info mi{};
    dubhe_model_info_get(e, &mi);
    std::fprintf(stderr, "就绪：%s · %s · %s\n输入 /exit 退出，/clear 清空上下文\n\n",
                 mi.arch, mi.quant_desc, dubhe_hardware_info());

    std::vector<std::pair<std::string, std::string>> hist;
    char line[8192];
    while (true) {
        std::fprintf(stderr, "你 › ");
        if (!std::fgets(line, sizeof(line), stdin)) break;
        std::string input(line);
        while (!input.empty() && (input.back() == '\n' || input.back() == '\r')) input.pop_back();
        if (input.empty()) continue;
        if (input == "/exit" || input == "/quit") break;
        if (input == "/clear") { hist.clear(); std::fprintf(stderr, "已清空\n"); continue; }

        hist.emplace_back("user", input);
        std::string prompt = build_chatml(hist);

        int32_t n = dubhe_tokenize(e, prompt.c_str(), 1, nullptr, 0);
        std::vector<int32_t> toks((size_t)n);
        dubhe_tokenize(e, prompt.c_str(), 1, toks.data(), n);

        dubhe_sampling sp{};
        sp.max_tokens = a.n_predict;
        sp.temperature = a.temp;
        sp.top_k = a.top_k;
        sp.top_p = a.top_p;
        sp.repeat_penalty = a.rep;

        std::string answer;
        dubhe_stats st{};
        std::fprintf(stderr, "AI › ");
        struct Ctx { std::string* out; } ctx{&answer};
        dubhe_generate(e, toks.data(), n, &sp,
                       [](void* ud, int32_t, const char* piece) {
                           auto* c = (Ctx*)ud;
                           if (piece) { std::fputs(piece, stdout); std::fflush(stdout); c->out->append(piece); }
                       },
                       nullptr, &ctx, &st);
        std::printf("\n");
        if (a.stats) print_stats(st);
        hist.emplace_back("assistant", answer);
    }
    dubhe_destroy(e);
    return 0;
}

int cmd_bench(const Args& a) {
    if (a.model.empty()) { std::fprintf(stderr, "需要 -m\n"); return 1; }
    dubhe_config cfg{};
    cfg.n_threads = a.threads;
    cfg.n_ctx = a.ctx;
    cfg.use_gpu = a.no_gpu ? 0 : 1;
    dubhe_engine* e = dubhe_create(&cfg);
    if (!e) return 1;
    if (dubhe_load_model(e, a.model.c_str()) != DUBHE_OK) {
        std::fprintf(stderr, "加载失败: %s\n", dubhe_last_error());
        dubhe_destroy(e);
        return 1;
    }
    dubhe_model_info mi{};
    dubhe_model_info_get(e, &mi);

    const char* prompt = "用三句话解释什么是本地推理引擎。";
    std::string text = build_chatml({{"user", prompt}});
    int32_t n = dubhe_tokenize(e, text.c_str(), 1, nullptr, 0);
    std::vector<int32_t> toks((size_t)n);
    dubhe_tokenize(e, text.c_str(), 1, toks.data(), n);

    dubhe_sampling sp{};
    sp.max_tokens = a.n_predict;
    sp.temperature = 0.f;
    sp.top_k = 0;
    sp.top_p = 0.f;
    sp.repeat_penalty = 1.0f;

    std::printf("设备    %s\n", dubhe_hardware_info());
    std::printf("模型    %s / %s / %.2fB\n", mi.arch, mi.quant_desc, (double)mi.n_params / 1e9);
    std::printf("上下文  %d\n\n", mi.n_ctx_train);

    // warmup
    {
        dubhe_stats st{};
        dubhe_generate(e, toks.data(), n, &sp, nullptr, nullptr, nullptr, &st);
    }
    double sum_p = 0, sum_d = 0;
    const int runs = 3;
    for (int i = 0; i < runs; i++) {
        dubhe_stats st{};
        dubhe_generate(e, toks.data(), n, &sp, nullptr, nullptr, nullptr, &st);
        sum_p += st.prefill_tps;
        sum_d += st.decode_tps;
        std::printf("run %d: prefill %.1f tok/s | decode %.2f tok/s\n", i + 1, st.prefill_tps,
                    st.decode_tps);
    }
    std::printf("\n均值    prefill %.1f tok/s | decode %.2f tok/s\n", sum_p / runs, sum_d / runs);
    dubhe_destroy(e);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, &a)) return 1;
    if (a.cmd == "inspect") return cmd_inspect(a);
    if (a.cmd == "run") return cmd_run(a);
    if (a.cmd == "chat") return cmd_chat(a);
    if (a.cmd == "bench") return cmd_bench(a);
    usage();
    return 1;
}
