// engine.cpp — C ABI 实现：会话、采样、生成循环
#include "internal.h"

#include "../include/dubhe/dubhe.h"
#include "image.h"
#include "mmproj.h"

#include <sys/sysctl.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using namespace dubhe;

struct dubhe_engine {
    Model model;
    dubhe_config cfg{};
    std::atomic<bool> abort{false};
    std::mutex mu;
    // 多模态特征（图像/音频塔的输出）。由 dubhe_set_mm_embeds 拷进来，
    // 因为调用方的缓冲区随时可能释放，而生成是异步长过程。
    std::vector<float> mm_embeds;
    int32_t mm_n = 0;
    int32_t mm_at = -1;
    // 多模态投影器（mmproj GGUF）。独立于语言模型加载，一张图只跑一次。
    std::unique_ptr<MmProj> mmproj;
};

namespace {

thread_local std::string g_err;

void set_err(const std::string& s) { g_err = s; }

std::string hardware_summary() {
    char brand[256] = {0};
    size_t len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) != 0)
        std::snprintf(brand, sizeof(brand), "Apple Silicon");

    uint64_t mem = 0;
    len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);

    int ncpu = 0;
    len = sizeof(ncpu);
    sysctlbyname("hw.ncpu", &ncpu, &len, nullptr, 0);

    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s · %d 核 · %.0f GB 统一内存", brand, ncpu,
                  (double)mem / (1024.0 * 1024.0 * 1024.0));
    return std::string(buf);
}

// 采样：重复惩罚 → temperature → top-k → top-p → 多项式
int32_t sample_token(float* logits, int n_vocab, const dubhe_sampling& sp,
                     const std::vector<int32_t>& history, std::mt19937& rng) {
    if (sp.repeat_penalty > 1.0f && sp.repeat_last_n > 0 && !history.empty()) {
        const int start = std::max(0, (int)history.size() - sp.repeat_last_n);
        for (int i = start; i < (int)history.size(); i++) {
            const int32_t t = history[i];
            if (t < 0 || t >= n_vocab) continue;
            if (logits[t] > 0) logits[t] /= sp.repeat_penalty;
            else logits[t] *= sp.repeat_penalty;
        }
    }

    if (sp.temperature <= 0.f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    const float inv_t = 1.f / sp.temperature;
    std::vector<std::pair<float, int>> cand;
    cand.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) cand.emplace_back(logits[i] * inv_t, i);

    const int k = (sp.top_k > 0 && sp.top_k < n_vocab) ? sp.top_k : n_vocab;
    if (k < n_vocab) {
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        cand.resize(k);
    } else {
        std::sort(cand.begin(), cand.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // softmax
    const float maxl = cand.front().first;
    float sum = 0.f;
    for (auto& c : cand) { c.first = std::exp(c.first - maxl); sum += c.first; }
    for (auto& c : cand) c.first /= sum;

    // top-p
    if (sp.top_p > 0.f && sp.top_p < 1.f) {
        float acc = 0.f;
        size_t cut = cand.size();
        for (size_t i = 0; i < cand.size(); i++) {
            acc += cand[i].first;
            if (acc >= sp.top_p) { cut = i + 1; break; }
        }
        cand.resize(cut);
        sum = 0.f;
        for (auto& c : cand) sum += c.first;
        for (auto& c : cand) c.first /= sum;
    }

    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(rng), acc = 0.f;
    for (auto& c : cand) {
        acc += c.first;
        if (r <= acc) return c.second;
    }
    return cand.back().second;
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// ---------------------------------------------------------------- ABI
extern "C" {

dubhe_engine* dubhe_create(const dubhe_config* cfg) {
    auto* e = new (std::nothrow) dubhe_engine();
    if (!e) return nullptr;
    if (cfg) e->cfg = *cfg;
    if (e->cfg.n_threads <= 0) {
        int ncpu = 0;
        size_t len = sizeof(ncpu);
        sysctlbyname("hw.perflevel0.logicalcpu", &ncpu, &len, nullptr, 0);
        if (ncpu <= 0) {
            len = sizeof(ncpu);
            sysctlbyname("hw.ncpu", &ncpu, &len, nullptr, 0);
        }
        e->cfg.n_threads = ncpu > 0 ? ncpu : 4;
    }
    return e;
}

void dubhe_destroy(dubhe_engine* e) { delete e; }

int dubhe_load_model(dubhe_engine* e, const char* path) {
    if (!e || !path) { set_err("参数为空"); return DUBHE_ERR_INVALID_ARG; }
    std::string err;
    if (!e->model.load(path, e->cfg.n_ctx, e->cfg.use_gpu != 0, &err)) {
        set_err(err);
        return DUBHE_ERR_FORMAT;
    }
    return DUBHE_OK;
}

void dubhe_unload_model(dubhe_engine* e) { if (e) e->model.unload(); }

int dubhe_model_info_get(const dubhe_engine* e, dubhe_model_info* out) {
    if (!e || !out) return DUBHE_ERR_INVALID_ARG;
    const auto& c = e->model.cfg();
    std::memset(out, 0, sizeof(*out));
    std::snprintf(out->arch, sizeof(out->arch), "%s", c.arch.c_str());
    std::snprintf(out->name, sizeof(out->name), "%s", c.arch.c_str());
    out->n_vocab = c.n_vocab;
    out->n_ctx_train = c.n_ctx_train;
    out->n_ctx_limit = e->model.n_ctx_limit();
    out->n_embd = c.n_embd;
    out->n_layer = c.n_layer;
    out->n_head = c.n_head;
    out->n_head_kv = c.n_head_kv;
    out->n_ff = c.n_ff;
    out->n_rot = c.n_rot;
    out->n_params = e->model.n_params();
    out->file_size = e->model.file().file_size();
    std::snprintf(out->quant_desc, sizeof(out->quant_desc), "%s", e->model.quant_desc());
    out->has_chat_template = e->model.tok().is_chatml() ? 1 : 0;
    out->is_gpu = e->model.gpu_active() ? 1 : 0;
    const char* note = e->model.gpu_note();
    std::snprintf(out->backend, sizeof(out->backend), "%s",
                  (note && note[0]) ? note : "CPU · NEON");
    return DUBHE_OK;
}

int32_t dubhe_tokenize(const dubhe_engine* e, const char* text, int add_special,
                       int32_t* out, int32_t cap) {
    if (!e || !text) return -1;
    auto ids = e->model.tok().encode(text, add_special != 0);
    if (out && cap >= (int32_t)ids.size()) {
        for (size_t i = 0; i < ids.size(); i++) out[i] = ids[i];
    }
    return (int32_t)ids.size();
}

int32_t dubhe_token_to_piece(const dubhe_engine* e, int32_t token, char* buf, int32_t cap) {
    if (!e) return -1;
    std::string s = e->model.tok().decode_piece(token);
    if (buf && cap >= (int32_t)s.size()) std::memcpy(buf, s.data(), s.size());
    return (int32_t)s.size();
}

int dubhe_load_mmproj(dubhe_engine* e, const char* path) {
    if (!e || !path) return DUBHE_ERR_INVALID_ARG;
    std::string err;
    auto p = std::make_unique<MmProj>();
    if (!p->load(path, &err)) { set_err("mmproj 加载失败: " + err); return DUBHE_ERR_FORMAT; }
    std::lock_guard<std::mutex> lock(e->mu);
    e->mmproj = std::move(p);
    return DUBHE_OK;
}

int dubhe_encode_image(dubhe_engine* e, const char* image_path, float** embeds,
                       int32_t* n_tokens, int32_t* dim) {
    if (!e || !image_path || !embeds || !n_tokens || !dim) return DUBHE_ERR_INVALID_ARG;
    if (!e->mmproj) { set_err("未加载 mmproj，无法处理图像"); return DUBHE_ERR_STATE; }

    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    std::string err;
    if (!load_image_rgb(image_path, &w, &h, &rgb, &err)) {
        set_err("读图失败: " + err);
        return DUBHE_ERR_IO;
    }
    const auto& c = e->mmproj->config();
    int oh = 0, ow = 0;
    e->mmproj->smart_resize(h, w, &oh, &ow);
    std::vector<uint8_t> resized((size_t)ow * (size_t)oh * 3);
    resize_bicubic(rgb.data(), w, h, resized.data(), ow, oh);

    const int gh = oh / c.patch, gw = ow / c.patch;
    const int64_t np = (int64_t)gh * gw;
    std::vector<float> pixels((size_t)(np * c.patch_dim()));
    if (!e->mmproj->pack_pixels(resized.data(), ow, oh, pixels.data())) {
        set_err("pack_pixels 失败");
        return DUBHE_ERR_FORMAT;
    }
    const int64_t nt = np / (c.merge * c.merge);
    // 引擎的 n_embd 必须和视觉塔输出宽度一致，否则注入时就是乱码。
    const int n_embd = e->model.loaded() ? e->model.cfg().n_embd : c.out_dim;
    if (c.out_dim != n_embd) {
        set_err("视觉塔输出 " + std::to_string(c.out_dim) + " 维，模型需要 " +
                std::to_string(n_embd) + " 维");
        return DUBHE_ERR_FORMAT;
    }
    float* buf = (float*)std::malloc(sizeof(float) * (size_t)nt * (size_t)c.out_dim);
    if (!buf) { set_err("内存不足"); return DUBHE_ERR_OOM; }
    if (!e->mmproj->encode(pixels.data(), gh, gw, buf, &err)) {
        std::free(buf);
        set_err("视觉塔前向失败: " + err);
        return DUBHE_ERR_STATE;
    }
    *embeds = buf;
    *n_tokens = (int32_t)nt;
    *dim = c.out_dim;
    return DUBHE_OK;
}

void dubhe_free_embeds(float* embeds) { std::free(embeds); }

int dubhe_set_mm_embeds(dubhe_engine* e, const float* embeds, int32_t n_tokens,
                        int32_t at_pos, int32_t dim) {
    if (!e || !embeds || n_tokens <= 0) { set_err("多模态特征为空"); return DUBHE_ERR_INVALID_ARG; }
    if (!e->model.loaded()) { set_err("模型未加载"); return DUBHE_ERR_STATE; }
    const int n_embd = e->model.cfg().n_embd;
    if (dim != n_embd) {
        set_err("特征维度 " + std::to_string(dim) + " 与模型 n_embd " +
                std::to_string(n_embd) + " 不符");
        return DUBHE_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(e->mu);
    e->mm_embeds.assign(embeds, embeds + (size_t)n_tokens * (size_t)n_embd);
    e->mm_n = n_tokens;
    e->mm_at = at_pos < 0 ? 0 : at_pos;
    return DUBHE_OK;
}

void dubhe_clear_mm_embeds(dubhe_engine* e) {
    if (!e) return;
    std::lock_guard<std::mutex> lock(e->mu);
    e->mm_embeds.clear();
    e->mm_n = 0;
    e->mm_at = -1;
    e->model.clear_mm_embeds();
}

int dubhe_generate(dubhe_engine* e,
                   const int32_t* prompt, int32_t n_prompt,
                   const dubhe_sampling* sp,
                   dubhe_token_cb on_token,
                   dubhe_stop_cb should_stop,
                   void* ud,
                   dubhe_stats* stats) {
    if (!e || !prompt || n_prompt <= 0) { set_err("空 prompt"); return DUBHE_ERR_INVALID_ARG; }
    if (!e->model.loaded()) { set_err("模型未加载"); return DUBHE_ERR_STATE; }

    dubhe_sampling s{};
    if (sp) s = *sp;
    if (s.max_tokens <= 0) s.max_tokens = 256;
    if (s.repeat_last_n <= 0) s.repeat_last_n = 64;

    std::lock_guard<std::mutex> lock(e->mu);
    e->abort = false;
    e->model.reset_kv();

    // 多模态特征：位置必须在 prompt 之内，否则说明调用方算错了占位 token 数，
    // 静默忽略会得到"模型看得见但没看"的诡异结果，所以直接报错。
    if (e->mm_n > 0 && !e->mm_embeds.empty()) {
        if (e->mm_at + e->mm_n > n_prompt) {
            set_err("多模态特征位置越界：at=" + std::to_string(e->mm_at) + " n=" +
                    std::to_string(e->mm_n) + " 但 prompt 只有 " + std::to_string(n_prompt));
            return DUBHE_ERR_INVALID_ARG;
        }
        e->model.set_mm_embeds(e->mm_embeds.data(), e->mm_n, e->mm_at);
    } else {
        e->model.clear_mm_embeds();
    }

    const int n_vocab = e->model.cfg().n_vocab;
    std::vector<float> logits(n_vocab, 0.f);
    std::vector<int32_t> hist(prompt, prompt + n_prompt);

    std::mt19937 rng(s.seed ? (uint32_t)s.seed : std::random_device{}());

    const double t0 = now_ms();
    std::string err;
    // prefill：按引擎实际批大小切块。GPU 上一次提交吃 8 个 token，
    // 权重只读一遍就摊到 8 个 token 上——这是首字延迟的关键。
    {
        const int nb = std::max(1, e->model.batch_size());
        for (int32_t i = 0; i < n_prompt; ) {
            const int take = std::min<int>(nb, n_prompt - i);
            if (!e->model.forward_batch(prompt + i, take, i, logits.data(), &err)) {
                set_err(err);
                return DUBHE_ERR_STATE;
            }
            i += take;
        }
    }
    const double t1 = now_ms();

    const int eos = e->model.tok().eos_id();
    int32_t n_gen = 0;
    int32_t pos = n_prompt;
    const int n_ctx = e->model.n_ctx_limit();

    while (n_gen < s.max_tokens) {
        if (e->abort.load()) break;
        if (should_stop && should_stop(ud)) break;

        int32_t tok = sample_token(logits.data(), n_vocab, s, hist, rng);
        hist.push_back(tok);

        // EOS 与 <|im_start|> 这类控制 token 不能当成正文吐给用户
        if (tok == eos) break;
        const bool special = e->model.tok().is_special(tok);

        if (on_token && !special) {
            std::string piece = e->model.tok().decode_piece(tok);
            on_token(ud, tok, piece.c_str());
        }
        if (!special) n_gen++;

        if (pos >= n_ctx) break;

        if (!e->model.forward(tok, pos, logits.data(), &err)) {
            set_err(err);
            break;
        }
        pos++;
    }
    const double t2 = now_ms();

    if (stats) {
        stats->n_prompt = n_prompt;
        stats->n_generated = n_gen;
        stats->t_prompt_ms = t1 - t0;
        stats->t_gen_ms = t2 - t1;
        stats->prefill_tps = (t1 - t0) > 0 ? n_prompt / ((t1 - t0) / 1000.0) : 0.0;
        stats->decode_tps = (t2 - t1) > 0 ? std::max(0, n_gen - 1) / ((t2 - t1) / 1000.0) : 0.0;
    }
    return DUBHE_OK;
}

void dubhe_request_abort(dubhe_engine* e) { if (e) e->abort = true; }

const char* dubhe_last_error(void) { return g_err.c_str(); }

const char* dubhe_version(void) {
    static char v[32];
    std::snprintf(v, sizeof(v), "%d.%d.%d", DUBHE_VERSION_MAJOR, DUBHE_VERSION_MINOR,
                  DUBHE_VERSION_PATCH);
    return v;
}

const char* dubhe_hardware_info(void) {
    static std::string s = hardware_summary();
    return s.c_str();
}

}  // extern "C"
