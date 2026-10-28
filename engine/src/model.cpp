// model.cpp — 模型加载与前向（CPU reference 路径）
//
// 设计取舍（M0）：
//   * 权重全程 mmap，不复制、不反量化到内存，避免 4B 模型吃掉 16 GB。
//   * 矩阵乘用"量化行 · f32 向量"的融合点积，逐行多线程。
//   * 这条路径是数值基准（oracle）。Metal 后端必须与它逐 token 一致。
#include "internal.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>

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
    gpu_active_ = false;  // Metal 后端接入后置 true
    (void)use_gpu;
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
    if (act_q8_supported(w.type)) {
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

    // 1) embedding
    {
        const void* row = (const char*)file_.tensor_data(tok_embd_) +
                          (size_t)token * type_row_bytes(tok_embd_.type, n_embd);
        dequant_row(tok_embd_.type, row, x_.data(), n_embd);
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
