// model.cpp — 模型加载与前向
//
// 设计取舍（M0）：
//   * 权重全程 mmap，不复制、不反量化到内存，避免 4B 模型吃掉 16 GB。
//   * 矩阵乘用"量化行 · f32 向量"的融合点积，逐行多线程。
//   * 这条路径是数值基准（oracle）。Metal 后端必须与它逐 token 一致，
//     所以 load() 里会用真实权重跑一次 GPU/CPU 交叉校验，不达标就不上 GPU。
#include "internal.h"
#include "metal.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dubhe {

namespace {

struct MatvecCtx {
    const void* x;       // f32 或量化激活
    float* y;
    const char* base;
    size_t row_bytes;
    uint32_t type;
    int n_in;
    bool q8;             // true: x 是 Q8 量化激活
};

void matvec_work(void* ctx, size_t i) {
    auto* c = (MatvecCtx*)ctx;
    const void* row = c->base + i * c->row_bytes;
    c->y[i] = c->q8 ? vec_dot_q8(c->type, c->x, row, c->n_in)
                    : vec_dot(c->type, (const float*)c->x, row, c->n_in);
}

struct AttnCtx {
    const float* q;      // [n_head][hd]
    const float* kc;     // [n_ctx][n_kv*hd]
    const float* vc;
    float* out;          // [n_head][hd]
    int n_head, n_head_kv, hd, n_kv, kv_mul;
    float scale;
};

void attn_work(void* ctx, size_t h) {
    auto* c = (AttnCtx*)ctx;
    const int hd = c->hd;
    const float* q = c->q + h * hd;
    const int kvh = (int)h / c->kv_mul;

    std::vector<float> scores((size_t)c->n_kv);
    float maxs = -INFINITY;
    for (int t = 0; t < c->n_kv; t++) {
        const float* k = c->kc + (size_t)t * c->n_head_kv * hd + (size_t)kvh * hd;
        float s = 0.f;
        for (int d = 0; d < hd; d++) s += q[d] * k[d];
        s *= c->scale;
        scores[t] = s;
        if (s > maxs) maxs = s;
    }
    float sum = 0.f;
    for (int t = 0; t < c->n_kv; t++) {
        float e = std::exp(scores[t] - maxs);
        scores[t] = e;
        sum += e;
    }
    const float inv = 1.f / sum;
    float* o = c->out + h * hd;
    for (int d = 0; d < hd; d++) o[d] = 0.f;
    for (int t = 0; t < c->n_kv; t++) {
        const float w = scores[t] * inv;
        const float* v = c->vc + (size_t)t * c->n_head_kv * hd + (size_t)kvh * hd;
        for (int d = 0; d < hd; d++) o[d] += w * v[d];
    }
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

// 低于这个乘加量，一次 GPU 提交 + 同步的开销就大于收益，直接走 CPU。
// 0.5B 模型里最小的 attn_q 是 896×896 ≈ 8e5，远高于阈值，所以全部走 GPU。
constexpr int64_t kGpuMinMac = 1 << 18;

}  // namespace

bool Model::load(const char* path, int n_ctx, bool use_gpu, std::string* err) {
    unload();
    if (!file_.open(path, err)) return false;

    cfg_.arch = file_.get_str("general.architecture", "llama");
    const std::string p = cfg_.arch + ".";

    cfg_.n_vocab     = (int32_t)file_.get_u32(p + "vocab_size", 0);
    cfg_.n_ctx_train = (int32_t)file_.get_u32(p + "context_length", 2048);
    cfg_.n_embd      = (int32_t)file_.get_u32(p + "embedding_length", 0);
    cfg_.n_layer     = (int32_t)file_.get_u32(p + "block_count", 0);
    cfg_.n_head      = (int32_t)file_.get_u32(p + "attention.head_count", 0);
    cfg_.n_head_kv   = (int32_t)file_.get_u32(p + "attention.head_count_kv", cfg_.n_head);
    cfg_.n_ff        = (int32_t)file_.get_u32(p + "feed_forward_length", 0);
    cfg_.rms_eps     = file_.get_f32(p + "attention.layer_norm_rms_epsilon", 1e-5f);
    cfg_.rope_theta  = file_.get_f32(p + "rope.freq_base", 10000.f);

    if (cfg_.n_embd <= 0 || cfg_.n_layer <= 0 || cfg_.n_head <= 0) {
        if (err) *err = "模型配置不完整（embedding_length / block_count / head_count 缺失）";
        return false;
    }
    if (cfg_.n_head_kv <= 0) cfg_.n_head_kv = cfg_.n_head;
    cfg_.head_dim = cfg_.n_embd / cfg_.n_head;
    cfg_.n_rot = cfg_.head_dim;
    cfg_.n_ctx_train = std::max(cfg_.n_ctx_train, 512);
    // KV cache 不预分配满训练上下文（32K 上下文 × 24 层 = 800MB，纯浪费）。
    // 先给一个够用的窗口，不够时按 2 倍增长，上限 = 用户/训练上下文。
    n_ctx_ = (n_ctx > 0) ? std::min(n_ctx, cfg_.n_ctx_train) : std::min(cfg_.n_ctx_train, 4096);
    kv_cap_ = std::min(n_ctx_, 1024);

    if (!tok_.init(file_, err)) return false;

    if (!load_weights(err)) return false;

    // KV cache（可增长）
    const size_t kv_dim = (size_t)cfg_.n_head_kv * cfg_.head_dim;
    k_cache_.assign((size_t)cfg_.n_layer * kv_cap_ * kv_dim, 0.f);
    v_cache_.assign((size_t)cfg_.n_layer * kv_cap_ * kv_dim, 0.f);
    kv_len_ = 0;

    x_.assign(cfg_.n_embd, 0.f);
    xb_.assign(cfg_.n_embd, 0.f);
    xb2_.assign(cfg_.n_ff > cfg_.n_embd ? cfg_.n_ff : cfg_.n_embd, 0.f);
    q_.assign((size_t)cfg_.n_head * cfg_.head_dim, 0.f);
    k_.assign(kv_dim, 0.f);
    v_.assign(kv_dim, 0.f);
    attn_out_.assign(cfg_.n_embd, 0.f);
    gate_.assign(cfg_.n_ff, 0.f);
    up_.assign(cfg_.n_ff, 0.f);
    ffn_out_.assign(cfg_.n_ff, 0.f);
    logits_.assign(cfg_.n_vocab, 0.f);

    loaded_ = true;
    gpu_active_ = false;
    if (use_gpu) enable_gpu(nullptr);
    return true;
}

// 打开 GPU 并做一次交叉校验。
//
// 为什么要校验：Metal 内核是手写的位解包，任何一个 nibble 位移写错都会
// 让模型"能跑但胡言乱语"，这种 bug 从输出上几乎看不出来。所以加载时
// 用真实权重跑一遍 GPU vs CPU 基准，误差超阈值就整体退回 CPU。
bool Model::enable_gpu(std::string* err) {
    gpu_active_ = false;
    if (!metal::available()) {
        gpu_note_ = "无 Metal 设备";
        if (err) *err = gpu_note_;
        return false;
    }
    std::string e;
    if (!metal::init(&e)) {
        gpu_note_ = "Metal 初始化失败: " + e;
        if (err) *err = gpu_note_;
        return false;
    }
    if (!metal::bind_weights(file_.mapping(), file_.file_size(), &e)) {
        gpu_note_ = "权重零拷贝绑定失败: " + e;
        if (err) *err = gpu_note_;
        return false;
    }

    // 校验：抽三张有代表性的权重（注意力投影 / FFN 下投影 / 输出层），
    // GPU 算全量，CPU 只算前 64 行，成本可忽略。
    struct Probe { const TensorInfo* t; int n_out, n_in; const char* tag; };
    const Probe probes[] = {
        {&layers_[0].attn_q,   cfg_.n_head * cfg_.head_dim, cfg_.n_embd, "attn_q"},
        {&layers_[0].ffn_down, cfg_.n_embd, cfg_.n_ff,                    "ffn_down"},
        {&out_w_,              cfg_.n_vocab, cfg_.n_embd,                 "output"},
    };

    double worst = 0.0;
    const char* worst_tag = "";
    for (const Probe& p : probes) {
        if (!metal::supports(p.t->type, p.n_in)) continue;

        std::vector<float> xv((size_t)p.n_in), yg((size_t)p.n_out);
        for (int i = 0; i < p.n_in; i++)
            xv[(size_t)i] = std::sin((float)i * 0.017f) * 0.5f + 0.1f;  // 确定性、有正有负

        const char* tb = (const char*)file_.tensor_data(*p.t);
        const uint64_t woff = (uint64_t)(tb - (const char*)file_.mapping());
        if (!metal::mfvec(p.t->type, xv.data(), yg.data(), p.n_out, p.n_in, woff)) {
            gpu_note_ = std::string("GPU 校验无法执行（") + p.tag + "）: " + metal::last_error();
            if (err) *err = gpu_note_;
            return false;
        }

        const size_t rb = type_row_bytes(p.t->type, p.n_in);
        const int rows = std::min(p.n_out, 64);
        double max_err = 0.0, max_ref = 0.0;
        for (int r = 0; r < rows; r++) {
            const float ref = vec_dot(p.t->type, xv.data(), tb + (size_t)r * rb, p.n_in);
            max_err = std::max(max_err, std::fabs((double)yg[(size_t)r] - (double)ref));
            max_ref = std::max(max_ref, std::fabs((double)ref));
        }
        const double rel = max_err / (max_ref + 1e-6);
        if (rel > worst) { worst = rel; worst_tag = p.tag; }
    }

    if (worst > 1e-3) {
        char b[256];
        std::snprintf(b, sizeof(b), "GPU 校验未通过（%s 相对误差 %.2e），已回退 CPU",
                      worst_tag, worst);
        gpu_note_ = b;
        if (err) *err = gpu_note_;
        return false;
    }

    gpu_active_ = true;
    {
        char b[256];
        std::snprintf(b, sizeof(b), "Metal · %s（校验误差 %.1e）", metal::device_name(), worst);
        gpu_note_ = b;
    }

    // 起全 GPU 前向：整层所有算子编进一个 command buffer，每 token 只同步一次。
    auto off = [&](const TensorInfo& t) -> uint64_t {
        return (uint64_t)((const char*)file_.tensor_data(t) - (const char*)file_.mapping());
    };
    metal::Spec sp;
    sp.n_embd = cfg_.n_embd;
    sp.n_ff = cfg_.n_ff;
    sp.n_vocab = cfg_.n_vocab;
    sp.n_layer = cfg_.n_layer;
    sp.n_head = cfg_.n_head;
    sp.n_head_kv = cfg_.n_head_kv;
    sp.hd = cfg_.head_dim;
    sp.rms_eps = cfg_.rms_eps;
    sp.rope_theta = cfg_.rope_theta;
    sp.kv_cap = kv_cap_;
    sp.off_out_norm = off(out_norm_);
    sp.off_out_w = off(out_w_);
    sp.type_out_w = out_w_.type;
    sp.layers.resize((size_t)cfg_.n_layer);
    for (int l = 0; l < cfg_.n_layer; l++) {
        const Layer& L = layers_[(size_t)l];
        metal::LayerSpec& d = sp.layers[(size_t)l];
        d.off[metal::S_ATTN_NORM] = off(L.attn_norm);  d.type[metal::S_ATTN_NORM] = L.attn_norm.type;
        d.off[metal::S_Q]  = off(L.attn_q);            d.type[metal::S_Q]  = L.attn_q.type;
        d.off[metal::S_K]  = off(L.attn_k);            d.type[metal::S_K]  = L.attn_k.type;
        d.off[metal::S_V]  = off(L.attn_v);            d.type[metal::S_V]  = L.attn_v.type;
        d.off[metal::S_OUT] = off(L.attn_output);      d.type[metal::S_OUT] = L.attn_output.type;
        d.off[metal::S_FFN_NORM] = off(L.ffn_norm);    d.type[metal::S_FFN_NORM] = L.ffn_norm.type;
        d.off[metal::S_GATE] = off(L.ffn_gate);        d.type[metal::S_GATE] = L.ffn_gate.type;
        d.off[metal::S_UP]   = off(L.ffn_up);          d.type[metal::S_UP]   = L.ffn_up.type;
        d.off[metal::S_DOWN] = off(L.ffn_down);        d.type[metal::S_DOWN] = L.ffn_down.type;
        if (L.has_q_bias) d.bias[0] = off(L.attn_q_b);
        if (L.has_k_bias) d.bias[1] = off(L.attn_k_b);
        if (L.has_v_bias) d.bias[2] = off(L.attn_v_b);
    }

    if (!metal::fwd_init(sp, &e)) {
        gpu_active_ = false;
        gpu_note_ = "GPU 前向不可用: " + e;
        if (err) *err = gpu_note_;
        return false;
    }
    return true;
}

bool Model::load_weights(std::string* err) {
    const std::string arch = cfg_.arch;
    tok_embd_ = *file_.find_tensor("token_embd.weight");
    out_norm_ = *file_.find_tensor("output_norm.weight");
    if (const TensorInfo* t = file_.find_tensor("output.weight")) {
        out_w_ = *t;
        cfg_.tied_embd = false;
    } else {
        out_w_ = tok_embd_;
        cfg_.tied_embd = true;
    }

    // vocab 兜底：很多 GGUF（如 Qwen 官方发布）不写 <arch>.vocab_size，
    // 此时以 token_embd 的第二维为准，再用分词器词表交叉校验。
    if (cfg_.n_vocab <= 0) cfg_.n_vocab = (int32_t)tok_embd_.ne[1];
    if (tok_.n_vocab() > 0 && cfg_.n_vocab != tok_.n_vocab()) {
        // 词表里有 padding 到 2 的幂的模型，取较大者避免越界
        cfg_.n_vocab = std::max(cfg_.n_vocab, tok_.n_vocab());
    }
    if (cfg_.n_vocab <= 0) {
        if (err) *err = "无法确定词表大小";
        return false;
    }
    if (cfg_.n_ff <= 0) {
        const TensorInfo* t = file_.find_tensor("blk.0.ffn_up.weight");
        if (t) cfg_.n_ff = (int32_t)t->ne[1];
    }

    layers_.resize(cfg_.n_layer);
    uint64_t params = 0;
    std::map<uint32_t, int> type_count;

    auto count = [&](const TensorInfo& t) {
        int64_t n = 1;
        for (int d = 0; d < t.n_dims; d++) n *= t.ne[d];
        params += (uint64_t)n;
        type_count[t.type]++;
    };
    count(tok_embd_);
    count(out_norm_);
    if (!cfg_.tied_embd) count(out_w_);

    auto need = [&](const std::string& name) -> TensorInfo {
        const TensorInfo* t = file_.find_tensor(name);
        if (!t) {
            if (err) *err = "缺少张量: " + name;
            return TensorInfo{};
        }
        if (!type_supported(t->type)) {
            if (err) *err = std::string("不支持的量化类型 ") + type_name(t->type) + "（张量 " + name + "）";
            return TensorInfo{};
        }
        count(*t);
        return *t;
    };

    for (int i = 0; i < cfg_.n_layer; i++) {
        const std::string b = "blk." + std::to_string(i) + ".";
        auto& L = layers_[i];
        L.attn_norm   = need(b + "attn_norm.weight");
        L.attn_q      = need(b + "attn_q.weight");
        L.attn_k      = need(b + "attn_k.weight");
        L.attn_v      = need(b + "attn_v.weight");
        L.attn_output = need(b + "attn_output.weight");
        L.ffn_norm    = need(b + "ffn_norm.weight");
        L.ffn_gate    = need(b + "ffn_gate.weight");
        L.ffn_up      = need(b + "ffn_up.weight");
        L.ffn_down    = need(b + "ffn_down.weight");
        if (const TensorInfo* t = file_.find_tensor(b + "attn_q.bias")) { L.attn_q_b = *t; L.has_q_bias = true; }
        if (const TensorInfo* t = file_.find_tensor(b + "attn_k.bias")) { L.attn_k_b = *t; L.has_k_bias = true; }
        if (const TensorInfo* t = file_.find_tensor(b + "attn_v.bias")) { L.attn_v_b = *t; L.has_v_bias = true; }
        if (err && !err->empty()) return false;
    }
    n_params_ = params;

    // 量化描述
    std::string desc;
    for (auto& kv : type_count) {
        if (!desc.empty()) desc += "+";
        desc += type_name(kv.first);
    }
    quant_desc_ = desc;
    return true;
}

void Model::unload() {
    loaded_ = false;
    gpu_active_ = false;
    // 必须在 file_.close() 之前释放：MTLBuffer 引用的就是那块 mmap
    metal::unbind_weights();
    layers_.clear();
    k_cache_.clear();
    v_cache_.clear();
    x_.clear();
    xb_.clear();
    xb2_.clear();
    q_.clear();
    k_.clear();
    v_.clear();
    attn_out_.clear();
    gate_.clear();
    up_.clear();
    ffn_out_.clear();
    logits_.clear();
    file_.close();
}

void Model::reset_kv() { kv_len_ = 0; }

bool Model::grow_kv(int need) {
    if (need <= kv_cap_) return true;
    if (need > n_ctx_) return false;
    int cap = kv_cap_ > 0 ? kv_cap_ : 256;
    while (cap < need) cap *= 2;
    if (cap > n_ctx_) cap = n_ctx_;

    if (gpu_active_) {
        std::string e;
        if (!metal::fwd_grow_kv(cap, &e)) return false;
        kv_cap_ = cap;
        return true;
    }

    const size_t kv_dim = (size_t)cfg_.n_head_kv * cfg_.head_dim;
    const size_t old_row = (size_t)kv_cap_ * kv_dim;
    const size_t new_row = (size_t)cap * kv_dim;

    std::vector<float> nk((size_t)cfg_.n_layer * new_row, 0.f);
    std::vector<float> nv((size_t)cfg_.n_layer * new_row, 0.f);
    for (int l = 0; l < cfg_.n_layer; l++) {
        std::memcpy(nk.data() + (size_t)l * new_row, k_cache_.data() + (size_t)l * old_row,
                    sizeof(float) * old_row);
        std::memcpy(nv.data() + (size_t)l * new_row, v_cache_.data() + (size_t)l * old_row,
                    sizeof(float) * old_row);
    }
    k_cache_.swap(nk);
    v_cache_.swap(nv);
    kv_cap_ = cap;
    return true;
}

void Model::rmsnorm(float* out, const float* x, const TensorInfo& w, int n) const {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float scale = 1.f / std::sqrt(ss / n + cfg_.rms_eps);
    const void* wp = file_.tensor_data(w);
    if (w.type == T_F32) {
        const float* wf = (const float*)wp;
        for (int i = 0; i < n; i++) out[i] = x[i] * scale * wf[i];
    } else if (w.type == T_F16) {
        const uint16_t* wf = (const uint16_t*)wp;
        for (int i = 0; i < n; i++) out[i] = x[i] * scale * fp16_to_fp32(wf[i]);
    } else {
        const uint16_t* wf = (const uint16_t*)wp;
        for (int i = 0; i < n; i++) out[i] = x[i] * scale * bf16_to_fp32(wf[i]);
    }
}

void Model::matvec(float* out, const float* x, const TensorInfo& w, int n_out, int n_in) const {
    // DUBHE_F32_REF=1 时强制走纯 f32 点积。CPU 的默认路径会把激活量化成
    // Q8（快 2~4 倍，但有 ~1e-3 的相对误差），所以校验 GPU 数值时必须关掉它，
    // 否则量的是 CPU 自己的近似误差，不是 GPU 的。
    static const bool f32_ref = std::getenv("DUBHE_F32_REF") != nullptr;
    if (gpu_active_ && (int64_t)n_out * n_in >= kGpuMinMac && metal::supports(w.type, n_in)) {
        const uint64_t woff =
            (uint64_t)((const char*)file_.tensor_data(w) - (const char*)file_.mapping());
        if (metal::mfvec(w.type, x, out, n_out, n_in, woff)) return;
        // GPU 一旦出错就永久降级，避免每层都白试一次、把速度拖死。
        gpu_active_ = false;
        std::fprintf(stderr, "[dubhe] GPU 矩阵乘失败，已回退 CPU：%s\n", metal::last_error());
    }
    if (!f32_ref && act_q8_supported(w.type)) {
        // 激活只量化一次，所有行共用
        aq_.resize(act_q8_bytes(w.type, n_in));
        quantize_act(w.type, x, aq_.data(), n_in);
        MatvecCtx ctx{aq_.data(), out, (const char*)file_.tensor_data(w),
                      type_row_bytes(w.type, n_in), w.type, n_in, true};
        dispatch_apply_f((size_t)n_out, DISPATCH_APPLY_AUTO, &ctx, matvec_work);
        return;
    }
    MatvecCtx ctx{x, out, (const char*)file_.tensor_data(w),
                  type_row_bytes(w.type, n_in), w.type, n_in, false};
    dispatch_apply_f((size_t)n_out, DISPATCH_APPLY_AUTO, &ctx, matvec_work);
}

void Model::rope(float* q, float* k, int n_head, int n_head_kv, int head_dim, int pos) const {
    const int half = head_dim / 2;
    for (int h = 0; h < n_head + n_head_kv; h++) {
        float* v = (h < n_head) ? (q + (size_t)h * head_dim)
                                : (k + (size_t)(h - n_head) * head_dim);
        for (int i = 0; i < half; i++) {
            const float freq = 1.f / std::pow(cfg_.rope_theta, (float)(2 * i) / (float)head_dim);
            const float a = pos * freq;
            const float c = std::cos(a), s = std::sin(a);
            const float x0 = v[i], x1 = v[i + half];
            v[i]        = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }
}

int Model::batch_size() const {
    if (!gpu_active_) return 1;
    return metal::fwd_batch_max();
}

// prefill 批处理。GPU 可用时一次提交算完 n 个 token；否则退化成 n 次
// 单 token 前向（结果一致，只是慢——CPU 侧还没有批量内核）。
bool Model::forward_batch(const int32_t* tokens, int n, int32_t pos0, float* logits,
                          std::string* err) {
    if (!loaded_) { if (err) *err = "模型未加载"; return false; }
    if (n <= 0 || !tokens) { if (err) *err = "空批"; return false; }
    if (n == 1) return forward(tokens[0], pos0, logits, err);

    if (!gpu_active_) {
        for (int i = 0; i < n; i++) {
            if (!forward(tokens[i], pos0 + i, logits, err)) return false;
        }
        return true;
    }

    if (pos0 + n > n_ctx_) { if (err) *err = "上下文已满"; return false; }
    if (pos0 + n > kv_cap_ && !grow_kv(pos0 + n)) { if (err) *err = "KV cache 扩容失败"; return false; }

    const int n_embd = cfg_.n_embd;
    xbatch_.resize((size_t)n * (size_t)n_embd);
    for (int i = 0; i < n; i++) {
        const void* row = (const char*)file_.tensor_data(tok_embd_) +
                          (size_t)tokens[i] * type_row_bytes(tok_embd_.type, n_embd);
        dequant_row(tok_embd_.type, row, xbatch_.data() + (size_t)i * n_embd, n_embd);
    }

    if (!metal::fwd_batch(xbatch_.data(), n, pos0, logits, err)) {
        gpu_active_ = false;
        if (err && err->empty()) *err = "GPU 批处理前向失败";
        return false;
    }
    kv_len_ = pos0 + n;
    return true;
}

bool Model::forward(int32_t token, int32_t pos, float* logits, std::string* err) {
    if (!loaded_) { if (err) *err = "模型未加载"; return false; }
    if (pos >= n_ctx_) { if (err) *err = "上下文已满"; return false; }
    if (pos >= kv_cap_ && !grow_kv(pos + 1)) { if (err) *err = "KV cache 扩容失败"; return false; }

    const int n_embd = cfg_.n_embd;
    const int hd = cfg_.head_dim;
    const int n_head = cfg_.n_head;
    const int n_kv = cfg_.n_head_kv;
    const int kv_dim = n_kv * hd;
    const int kv_mul = n_head / n_kv;

    // 1) embedding（CPU 反量化一行，之后整层都留在 GPU 侧）
    {
        const void* row = (const char*)file_.tensor_data(tok_embd_) +
                          (size_t)token * type_row_bytes(tok_embd_.type, n_embd);
        dequant_row(tok_embd_.type, row, x_.data(), n_embd);
    }

    if (gpu_active_) {
        // 整层前向一次提交、一次同步。失败就整体报错——此时 GPU 的 KV 已经
        // 前进了一步，切回 CPU 会得到错误结果，静默降级比报错更危险。
        if (!metal::fwd_token(x_.data(), pos, logits, err)) {
            gpu_active_ = false;
            if (err && err->empty()) *err = "GPU 前向失败";
            return false;
        }
        kv_len_ = pos + 1;
        return true;
    }

    for (int l = 0; l < cfg_.n_layer; l++) {
        const Layer& L = layers_[l];

        // 2) attention
        rmsnorm(xb_.data(), x_.data(), L.attn_norm, n_embd);
        matvec(q_.data(), xb_.data(), L.attn_q, n_head * hd, n_embd);
        matvec(k_.data(), xb_.data(), L.attn_k, kv_dim, n_embd);
        matvec(v_.data(), xb_.data(), L.attn_v, kv_dim, n_embd);

        if (L.has_q_bias) {
            const float* b = (const float*)file_.tensor_data(L.attn_q_b);
            for (int i = 0; i < n_head * hd; i++) q_[i] += b[i];
        }
        if (L.has_k_bias) {
            const float* b = (const float*)file_.tensor_data(L.attn_k_b);
            for (int i = 0; i < kv_dim; i++) k_[i] += b[i];
        }
        if (L.has_v_bias) {
            const float* b = (const float*)file_.tensor_data(L.attn_v_b);
            for (int i = 0; i < kv_dim; i++) v_[i] += b[i];
        }

        rope(q_.data(), k_.data(), n_head, n_kv, hd, pos);

        float* kc = k_cache_.data() + (size_t)l * kv_cap_ * kv_dim;
        float* vc = v_cache_.data() + (size_t)l * kv_cap_ * kv_dim;
        std::memcpy(kc + (size_t)pos * kv_dim, k_.data(), sizeof(float) * kv_dim);
        std::memcpy(vc + (size_t)pos * kv_dim, v_.data(), sizeof(float) * kv_dim);

        AttnCtx actx{q_.data(), kc, vc, attn_out_.data(), n_head, n_kv, hd, pos + 1, kv_mul,
                     1.f / std::sqrt((float)hd)};
        dispatch_apply_f((size_t)n_head, DISPATCH_APPLY_AUTO, &actx, attn_work);

        // 3) attention 输出投影 + 残差
        {
            static thread_local std::vector<float> tmp;
            tmp.resize(n_embd);
            matvec(tmp.data(), attn_out_.data(), L.attn_output, n_embd, n_embd);
            for (int i = 0; i < n_embd; i++) x_[i] += tmp[i];
        }

        // 4) FFN（SwiGLU）
        rmsnorm(xb_.data(), x_.data(), L.ffn_norm, n_embd);
        matvec(gate_.data(), xb_.data(), L.ffn_gate, cfg_.n_ff, n_embd);
        matvec(up_.data(), xb_.data(), L.ffn_up, cfg_.n_ff, n_embd);
        for (int i = 0; i < cfg_.n_ff; i++) ffn_out_[i] = silu(gate_[i]) * up_[i];
        {
            static thread_local std::vector<float> tmp;
            tmp.resize(n_embd);
            matvec(tmp.data(), ffn_out_.data(), L.ffn_down, n_embd, cfg_.n_ff);
            for (int i = 0; i < n_embd; i++) x_[i] += tmp[i];
        }
    }

    // 5) final norm + logits
    rmsnorm(xb_.data(), x_.data(), out_norm_, n_embd);
    matvec(logits, xb_.data(), out_w_, cfg_.n_vocab, n_embd);

    kv_len_ = pos + 1;
    return true;
}

}  // namespace dubhe
