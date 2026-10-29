// model.cpp — 模型加载与前向
//
// 设计取舍（M0）：
//   * 权重全程 mmap，不复制、不反量化到内存，避免 4B 模型吃掉 16 GB。
//   * 矩阵乘用"量化行 · f32 向量"的融合点积，逐行多线程。
//   * 这条路径是数值基准（oracle）。Metal 后端必须与它逐 token 一致，
//     所以 load() 里会用真实权重跑一次 GPU/CPU 交叉校验，不达标就不上 GPU。
#include "internal.h"
#include "metal.h"
#include "parallel.h"

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
    int t_start;         // 滑窗下界（含），全注意力为 0
    float scale;
};

void attn_work(void* ctx, size_t h) {
    auto* c = (AttnCtx*)ctx;
    const int hd = c->hd;
    const float* q = c->q + h * hd;
    const int kvh = (int)h / c->kv_mul;
    const int n_vis = c->n_kv - c->t_start;

    std::vector<float> scores((size_t)n_vis);
    float maxs = -INFINITY;
    for (int t = 0; t < n_vis; t++) {
        const float* k = c->kc + (size_t)(t + c->t_start) * c->n_head_kv * hd + (size_t)kvh * hd;
        float s = 0.f;
        for (int d = 0; d < hd; d++) s += q[d] * k[d];
        s *= c->scale;
        scores[t] = s;
        if (s > maxs) maxs = s;
    }
    float sum = 0.f;
    for (int t = 0; t < n_vis; t++) {
        float e = std::exp(scores[t] - maxs);
        scores[t] = e;
        sum += e;
    }
    const float inv = 1.f / sum;
    float* o = c->out + h * hd;
    for (int d = 0; d < hd; d++) o[d] = 0.f;
    for (int t = 0; t < n_vis; t++) {
        const float w = scores[t] * inv;
        const float* v = c->vc + (size_t)(t + c->t_start) * c->n_head_kv * hd + (size_t)kvh * hd;
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
    f32_ref_ = std::getenv("DUBHE_F32_REF") != nullptr;
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

    // ---- LFM3 / LFM2 专属字段
    // 这两族把 conv / 滑窗注意力 / 全局注意力混排，head_count_kv 是逐层的数组，
    // conv 层为 0。滑窗层还单独用一套 RoPE base（1e4 vs 1e6）。
    if (arch_is_lfm3()) {
        cfg_.n_swa          = (int32_t)file_.get_u32(p + "attention.sliding_window", 0);
        cfg_.rope_theta_swa = file_.get_f32(p + "rope.freq_base_swa", 10000.f);
        cfg_.conv_l_cache   = (int32_t)file_.get_u32(p + "shortconv.l_cache", 0);
        cfg_.sandwich_norm  = file_.find_tensor("blk.0.post_attention_norm.weight") != nullptr;
        // head_count_kv 是逐层数组，conv 层为 0。上面按标量读会拿到 0 并退化成
        // n_head，所以这里必须用数组里的最大值覆盖。
        if (const auto* kvh = file_.get_arr_i32(p + "attention.head_count_kv")) {
            int32_t mx = 0;
            for (int32_t v : *kvh) mx = std::max(mx, v);
            if (mx > 0) cfg_.n_head_kv = mx;
        }
        // partial RoPE（zero-tail）：只有前 head_dim/4 个角度真的转。
        // GGUF 不写 partial_rotary_factor，LFM3 固定 0.5，即 (0.5*hd)/2 = hd/4。
        cfg_.rope_angles = cfg_.head_dim / 4;
        if (const GgufValue* v = file_.get(p + "rope.partial_rotary_factor")) {
            if (v->f64 > 0.0) cfg_.rope_angles = (int32_t)(v->f64 * cfg_.head_dim) / 2;
        }
    }
    cfg_.n_ctx_train = std::max(cfg_.n_ctx_train, 512);
    // KV cache 不预分配满训练上下文（32K 上下文 × 24 层 = 800MB，纯浪费）。
    // 先给一个够用的窗口，不够时按 2 倍增长，上限 = 用户/训练上下文。
    n_ctx_ = (n_ctx > 0) ? std::min(n_ctx, cfg_.n_ctx_train) : std::min(cfg_.n_ctx_train, 4096);
    kv_cap_ = std::min(n_ctx_, 1024);

    if (!tok_.init(file_, err)) return false;

    if (!load_weights(err)) return false;

    // DUBHE_LAYERS=N：只跑前 N 层。调试用——LFM3 的三种子层交错排布，
    // 逐层截断配合 GPU/CPU 对拍，可以一步定位是哪类层写错了。
    if (const char* lv = std::getenv("DUBHE_LAYERS")) {
        const int n = std::atoi(lv);
        if (n > 0 && n < cfg_.n_layer) {
            cfg_.n_layer = n;
            cfg_.n_conv = 0;
            conv_slot_.assign((size_t)cfg_.n_layer, -1);
            for (int i = 0; i < cfg_.n_layer; i++) {
                if (layers_[(size_t)i].kind == LK_CONV) conv_slot_[(size_t)i] = cfg_.n_conv++;
            }
        }
    }

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

    // ShortConv 的时序状态：每层存 l_cache-1 个历史 hidden 向量。
    // 13 层 × 6 × 2048 × 4B ≈ 640 KB，可以忽略。
    ffn_down_out_.assign(cfg_.n_embd, 0.f);
    if (cfg_.n_conv > 0 && cfg_.conv_l_cache > 1) {
        conv_bc_.assign((size_t)3 * cfg_.n_embd, 0.f);
        conv_h_.assign(cfg_.n_embd, 0.f);
        conv_state_.assign((size_t)cfg_.n_conv * (cfg_.conv_l_cache - 1) * cfg_.n_embd, 0.f);
    }

    loaded_ = true;
    gpu_active_ = false;
    gpu_fused_ = false;
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
    // LFM3 的第 0 层是 conv 层，没有 attn_q，所以按"第一个注意力层"取。
    const Layer* probe_layer = &layers_[0];
    for (const Layer& L : layers_) {
        if (L.kind != LK_CONV) { probe_layer = &L; break; }
    }
    struct Probe { const TensorInfo* t; int n_out, n_in; const char* tag; };
    const Probe probes[] = {
        {&probe_layer->attn_q, cfg_.n_head * cfg_.head_dim, cfg_.n_embd, "attn_q"},
        {&layers_[0].ffn_down, cfg_.n_embd, cfg_.n_ff,                     "ffn_down"},
        {&out_w_,              cfg_.n_vocab, cfg_.n_embd,                 "output"},
    };

    double worst = 0.0;
    const char* worst_tag = "";
    for (const Probe& p : probes) {
        if (p.t->ne[0] <= 0) continue;
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

    // LFM3 还没有整层融合内核。逐算子的 matvec 照样走 GPU（占前向时间的 99%），
    // 只是不接管整层，所以这里到此为止。
    // （融合内核已覆盖 LFM3，这段保留给将来不支持的新架构。）

    // 起全 GPU 前向：整层所有算子编进一个 command buffer，每 token 只同步一次。
    auto off = [&](const TensorInfo& t) -> uint64_t {
        if (t.ne[0] <= 0) return ~0ull;   // 该层没有这张张量
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
    sp.n_swa = cfg_.n_swa;
    sp.rope_angles = cfg_.rope_angles;
    sp.conv_l_cache = cfg_.conv_l_cache;
    sp.n_conv = cfg_.n_conv;
    sp.layers.resize((size_t)cfg_.n_layer);
    for (int l = 0; l < cfg_.n_layer; l++) {
        const Layer& L = layers_[(size_t)l];
        metal::LayerSpec& d = sp.layers[(size_t)l];
        d.kind = (L.kind == LK_CONV) ? metal::G_CONV
               : (L.kind == LK_ATTN_SWA) ? metal::G_ATTN_SWA : metal::G_ATTN_FULL;
        d.rope_theta_l = (L.kind == LK_ATTN_SWA && cfg_.rope_theta_swa > 0.f)
                             ? cfg_.rope_theta_swa : cfg_.rope_theta;
        d.conv_slot = (L.kind == LK_CONV) ? conv_slot_[(size_t)l] : -1;
        d.rope_angles_l = L.rope_angles_l;

        d.off[metal::S_ATTN_NORM] = off(L.attn_norm);  d.type[metal::S_ATTN_NORM] = L.attn_norm.type;
        d.off[metal::S_FFN_NORM] = off(L.ffn_norm);    d.type[metal::S_FFN_NORM] = L.ffn_norm.type;
        d.off[metal::S_GATE] = off(L.ffn_gate);        d.type[metal::S_GATE] = L.ffn_gate.type;
        d.off[metal::S_UP]   = off(L.ffn_up);          d.type[metal::S_UP]   = L.ffn_up.type;
        d.off[metal::S_DOWN] = off(L.ffn_down);        d.type[metal::S_DOWN] = L.ffn_down.type;

        if (L.kind == LK_CONV) {
            d.ex_off[metal::X_CONV_IN]  = off(L.conv_in);     d.ex_type[metal::X_CONV_IN]  = L.conv_in.type;
            d.ex_off[metal::X_CONV_K]   = off(L.conv_kernel);
            d.ex_off[metal::X_CONV_OUT] = off(L.conv_out);    d.ex_type[metal::X_CONV_OUT] = L.conv_out.type;
        } else {
            d.off[metal::S_Q]  = off(L.attn_q);            d.type[metal::S_Q]  = L.attn_q.type;
            d.off[metal::S_K]  = off(L.attn_k);            d.type[metal::S_K]  = L.attn_k.type;
            d.off[metal::S_V]  = off(L.attn_v);            d.type[metal::S_V]  = L.attn_v.type;
            d.off[metal::S_OUT] = off(L.attn_output);      d.type[metal::S_OUT] = L.attn_output.type;
            if (L.has_q_bias) d.bias[0] = off(L.attn_q_b);
            if (L.has_k_bias) d.bias[1] = off(L.attn_k_b);
            if (L.has_v_bias) d.bias[2] = off(L.attn_v_b);
            if (L.has_qk_norm) {
                d.ex_off[metal::X_QNORM] = off(L.attn_q_norm);
                d.ex_off[metal::X_KNORM] = off(L.attn_k_norm);
            }
            if (L.has_attn_gate) {
                d.ex_off[metal::X_AGATE] = off(L.attn_gate);
                d.ex_type[metal::X_AGATE] = L.attn_gate.type;
            }
        }
        if (L.has_sandwich) {
            d.ex_off[metal::X_POST_ATTN] = off(L.post_attn_norm);
            d.ex_off[metal::X_POST_FFN] = off(L.post_ffn_norm);
        }
    }

    if (!metal::fwd_init(sp, &e)) {
        // 融合内核接不了就退回逐算子 GPU（比 CPU 快不了多少，但至少不用改 KV）
        gpu_fused_ = false;
        gpu_note_ += " · 融合内核未覆盖，退回逐算子";
        return true;
    }
    // LFM3 的短卷积历史缓冲必须清零，否则第一步会吃到脏数据
    if (cfg_.n_conv > 0) metal::fwd_reset_conv();
    gpu_fused_ = true;

    // 融合内核的端到端对拍。
    //
    // mfvec 那轮校验只覆盖单个矩阵乘，覆盖不到层结构（谁在谁前面、残差加在哪、
    // 门乘在哪个位置）。LFM3 的层结构比 llama 复杂得多，这里用同一个 token
    // 跑一遍 GPU 融合前向和 CPU 参考前向，比 logits。差太多就整体退回 CPU——
    // 宁可慢，也不能让用户拿到"能跑但胡言乱语"的输出。
    if (!std::getenv("DUBHE_NO_FUSED_CHECK")) {
        std::vector<float> lg((size_t)cfg_.n_vocab), lc((size_t)cfg_.n_vocab);
        // 用一小段真实 token 跑多步：只对第一步的话，KV cache 与 ShortConv
        // 历史这些"递推状态"完全没被覆盖，而它们正是 LFM3 最容易写错的地方。
        std::vector<int32_t> seq = tok_.encode("The capital of France is", false);
        if (seq.empty()) seq.push_back(tok_.bos_id() >= 0 ? tok_.bos_id() : 1);
        if (seq.size() > 8) seq.resize(8);
        const int nstep = (int)seq.size();

        std::string e2;
        std::vector<float> gpu_steps, cpu_steps;
        bool okg = true, okc = true;
        reset_kv();
        for (int i = 0; i < nstep && okg; i++) {
            okg = forward(seq[(size_t)i], i, lg.data(), &e2);
            if (okg) gpu_steps.insert(gpu_steps.end(), lg.begin(), lg.end());
        }
        // 残差流只在最后一步有意义，先存下来
        std::vector<float> xg((size_t)cfg_.n_embd);
        metal::fwd_copy_x(xg.data(), cfg_.n_embd);
        reset_kv();

        gpu_fused_ = false;
        gpu_active_ = false;
        const bool prev_f32 = f32_ref_;
        f32_ref_ = true;          // 参考必须是无损的，否则量不出 GPU 的问题
        for (int i = 0; i < nstep && okc; i++) {
            okc = forward(seq[(size_t)i], i, lc.data(), &e2);
            if (okc) cpu_steps.insert(cpu_steps.end(), lc.begin(), lc.end());
        }
        f32_ref_ = prev_f32;
        gpu_active_ = true;
        gpu_fused_ = true;
        reset_kv();
        if (okg && okc) {
            for (int i = 0; i < nstep; i++) {
                double m = 0.0, r = 0.0;
                for (int j = 0; j < cfg_.n_vocab; j++) {
                    const size_t o = (size_t)i * (size_t)cfg_.n_vocab + (size_t)j;
                    m = std::max(m, (double)std::fabs(gpu_steps[o] - cpu_steps[o]));
                    r = std::max(r, (double)std::fabs(cpu_steps[o]));
                }
                if (std::getenv("DUBHE_FUSED_CHECK_VERBOSE")) {
                    std::fprintf(stderr, "[dubhe]   pos %d: max|Δ|=%.3e  rel=%.2e\n",
                                 i, m, m / (r + 1e-6));
                }
            }
        }
        if (okg && okc) {
            // 批处理路径：prefill 实际走的就是这条（nrow 最多 8），
            // 它用的是一整套 mm_* 内核，和逐 token 的 mv_* 完全是另一条代码，
            // 所以必须单独对一次。
            double rel_batch = 0.0;
            if (nstep > 1) {
                std::vector<float> lb((size_t)cfg_.n_vocab);
                reset_kv();
                if (forward_batch(seq.data(), nstep, 0, lb.data(), &e2)) {
                    double m = 0.0, r = 0.0;
                    const size_t base = (size_t)(nstep - 1) * (size_t)cfg_.n_vocab;
                    for (int j = 0; j < cfg_.n_vocab; j++) {
                        m = std::max(m, (double)std::fabs(lb[(size_t)j] - cpu_steps[base + (size_t)j]));
                        r = std::max(r, (double)std::fabs(cpu_steps[base + (size_t)j]));
                    }
                    rel_batch = m / (r + 1e-6);
                }
                reset_kv();
                // 前缀扫描：nrow=1,2,3... 逐级放大，找出第一个出错的行
                if (std::getenv("DUBHE_BATCH_SWEEP")) {
                    for (int pfx = 1; pfx <= nstep; pfx++) {
                        std::vector<float> lp((size_t)cfg_.n_vocab);
                        reset_kv();
                        if (!forward_batch(seq.data(), pfx, 0, lp.data(), &e2)) break;
                        double m = 0.0, r = 0.0;
                        const size_t bs = (size_t)(pfx - 1) * (size_t)cfg_.n_vocab;
                        for (int j = 0; j < cfg_.n_vocab; j++) {
                            m = std::max(m, (double)std::fabs(lp[(size_t)j] - cpu_steps[bs + (size_t)j]));
                            r = std::max(r, (double)std::fabs(cpu_steps[bs + (size_t)j]));
                        }
                        std::fprintf(stderr, "[dubhe]   nrow=%d rel=%.2e\n", pfx, m / (r + 1e-6));
                    }
                    reset_kv();
                }
                if (std::getenv("DUBHE_FUSED_CHECK_VERBOSE")) {
                    std::fprintf(stderr, "[dubhe] 批处理对拍(nrow=%d): rel=%.2e\n",
                                 nstep, rel_batch);
                }
            }

            // 再比一次残差流：它排除了 out_norm / out_w 的干扰
            double dx = 0.0, xr = 0.0;
            int first = -1;
            for (int i = 0; i < cfg_.n_embd; i++) {
                const double d = std::fabs((double)xg[(size_t)i] - (double)x_[(size_t)i]);
                if (d > 1e-3 * (std::fabs((double)x_[(size_t)i]) + 1e-3) && first < 0) first = i;
                dx = std::max(dx, d);
                xr = std::max(xr, (double)std::fabs(x_[(size_t)i]));
            }
            if (std::getenv("DUBHE_FUSED_CHECK_VERBOSE")) {
                std::fprintf(stderr,
                             "[dubhe] 残差流: max|Δ|=%.3e  max|x|=%.3e  首个偏离位置=%d\n",
                             dx, xr, first);
            }

            // 判据用"所有位置里最差的那一步"，只要一步崩了就不能上融合路径
            double rel = 0.0, mx = 0.0, mref = 0.0;
            int arg_g = 0, arg_c = 0, bad_step = -1;
            for (int i = 0; i < nstep; i++) {
                double m = 0.0, r = 0.0;
                int ag = 0, ac = 0;
                for (int j = 0; j < cfg_.n_vocab; j++) {
                    const size_t o = (size_t)i * (size_t)cfg_.n_vocab + (size_t)j;
                    m = std::max(m, (double)std::fabs(gpu_steps[o] - cpu_steps[o]));
                    r = std::max(r, (double)std::fabs(cpu_steps[o]));
                    if (gpu_steps[o] > gpu_steps[(size_t)i * (size_t)cfg_.n_vocab + (size_t)ag]) ag = j;
                    if (cpu_steps[o] > cpu_steps[(size_t)i * (size_t)cfg_.n_vocab + (size_t)ac]) ac = j;
                }
                const double rl = m / (r + 1e-6);
                if (rl > rel) { rel = rl; mx = m; mref = r; arg_g = ag; arg_c = ac; bad_step = i; }
            }
            if (rel_batch > rel) { rel = rel_batch; bad_step = -2; }   // -2 标记批处理路径
            if (std::getenv("DUBHE_FUSED_CHECK_VERBOSE")) {
                std::fprintf(stderr,
                             "[dubhe] 融合对拍: 最差 rel=%.2e @pos %d  max|Δ|=%.3e  "
                             "argmax gpu=%d cpu=%d\n", rel, bad_step, mx, arg_g, arg_c);
            }
            if (rel > 5e-3) {
                gpu_fused_ = false;
                char b[192];
                std::snprintf(b, sizeof(b), " · 融合校验未过(rel=%.1e@%d)", rel, bad_step);
                gpu_note_ += b;
            }
        }
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
        L.kind = LK_ATTN_FULL;

        // ---- 层类型：LFM3 由 GGUF 的两张逐层表决定
        if (arch_is_lfm3()) {
            bool is_conv = false;
            if (const auto* kvh = file_.get_arr_i32(cfg_.arch + ".attention.head_count_kv")) {
                if (i < (int)kvh->size()) is_conv = ((*kvh)[(size_t)i] == 0);
            }
            if (is_conv) {
                L.kind = LK_CONV;
            } else if (const auto* swp = file_.get_arr_u8(cfg_.arch + ".attention.sliding_window_pattern")) {
                if (i < (int)swp->size() && (*swp)[(size_t)i]) L.kind = LK_ATTN_SWA;
            }
        }

        // 每层实际旋转的角度数。HF 的 config.json 里 rope_parameters 是
        // **按层类型**分开配的：full_attention 带 partial_rotary_factor=0.5，
        // sliding_attention 没有这个键（缺省 1.0 → 全量旋转）。GGUF 没有
        // 逐层的这个字段，所以按层类型推。
        L.rope_angles_l = (L.kind == LK_ATTN_SWA) ? (cfg_.head_dim / 2)
                                                  : cfg_.rope_angles;

        L.ffn_norm    = need(b + "ffn_norm.weight");
        L.ffn_gate    = need(b + "ffn_gate.weight");
        L.ffn_up      = need(b + "ffn_up.weight");
        L.ffn_down    = need(b + "ffn_down.weight");

        if (L.kind == LK_CONV) {
            // ShortConv：in_proj 一次算出 B/C/x 三份（3*n_embd），conv 是
            // depthwise 的 [l_cache, n_embd] 实数核，out_proj 回 n_embd。
            L.attn_norm   = need(b + "attn_norm.weight");
            L.conv_in     = need(b + "shortconv.in_proj.weight");
            L.conv_kernel = need(b + "shortconv.conv.weight");
            L.conv_out    = need(b + "shortconv.out_proj.weight");
        } else {
            L.attn_norm   = need(b + "attn_norm.weight");
            L.attn_q      = need(b + "attn_q.weight");
            L.attn_k      = need(b + "attn_k.weight");
            L.attn_v      = need(b + "attn_v.weight");
            L.attn_output = need(b + "attn_output.weight");
            if (const TensorInfo* t = file_.find_tensor(b + "attn_q.bias")) { L.attn_q_b = *t; L.has_q_bias = true; }
            if (const TensorInfo* t = file_.find_tensor(b + "attn_k.bias")) { L.attn_k_b = *t; L.has_k_bias = true; }
            if (const TensorInfo* t = file_.find_tensor(b + "attn_v.bias")) { L.attn_v_b = *t; L.has_v_bias = true; }
            // LFM3 的 QK RMSNorm 与逐 head 输出门
            if (const TensorInfo* t = file_.find_tensor(b + "attn_q_norm.weight")) {
                L.attn_q_norm = *t; L.has_qk_norm = true;
                count(*t);
            }
            if (const TensorInfo* t = file_.find_tensor(b + "attn_k_norm.weight")) {
                L.attn_k_norm = *t;
                count(*t);
            }
            if (const TensorInfo* t = file_.find_tensor(b + "attn_gate.weight")) {
                L.attn_gate = *t; L.has_attn_gate = true;
                count(*t);
            }
        }

        // sandwich norm：子层输出先归一化再加残差
        if (const TensorInfo* t = file_.find_tensor(b + "post_attention_norm.weight")) {
            L.post_attn_norm = *t; L.has_sandwich = true;
            count(*t);
        }
        if (const TensorInfo* t = file_.find_tensor(b + "post_ffw_norm.weight")) {
            L.post_ffn_norm = *t;
            count(*t);
        }

        if (err && !err->empty()) return false;
    }
    n_params_ = params;

    // conv 层槽位表 + 层数统计
    cfg_.n_conv = 0;
    conv_slot_.assign((size_t)cfg_.n_layer, -1);
    for (int i = 0; i < cfg_.n_layer; i++) {
        if (layers_[(size_t)i].kind == LK_CONV) {
            conv_slot_[(size_t)i] = cfg_.n_conv++;
        }
    }

    // DUBHE_VERBOSE=1 打印逐层结构表。tri-hybrid 的层类型/角度/base 只要
    // 错一层，症状都是"输出通顺但内容跑偏"，从外面完全看不出，只能看这张表。
    if (std::getenv("DUBHE_VERBOSE")) {
        int n_full = 0, n_swa = 0;
        std::fprintf(stderr, "[dubhe] 层结构（共 %d 层）:\n", cfg_.n_layer);
        for (int i = 0; i < cfg_.n_layer; i++) {
            const Layer& L = layers_[(size_t)i];
            const char* k = L.kind == LK_CONV ? "conv  "
                          : L.kind == LK_ATTN_SWA ? "swa   " : "full  ";
            if (L.kind == LK_CONV) {
                std::fprintf(stderr, "[dubhe]   %2d %s\n", i, k);
            } else {
                const float th = (L.kind == LK_ATTN_SWA && cfg_.rope_theta_swa > 0.f)
                                     ? cfg_.rope_theta_swa : cfg_.rope_theta;
                std::fprintf(stderr, "[dubhe]   %2d %s rope_theta=%.0f angles=%d/%d%s\n",
                             i, k, th, L.rope_angles_l, cfg_.head_dim / 2,
                             L.has_attn_gate ? " gate" : "");
                (L.kind == LK_ATTN_SWA ? n_swa : n_full)++;
            }
        }
        std::fprintf(stderr, "[dubhe] 小计: conv=%d full=%d swa=%d  sandwich_norm=%d\n",
                     cfg_.n_conv, n_full, n_swa, (int)cfg_.sandwich_norm);
    }

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
    gpu_fused_ = false;
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
    conv_bc_.clear();
    conv_h_.clear();
    conv_state_.clear();
    conv_slot_.clear();
    ffn_down_out_.clear();
    cfg_.n_conv = 0;
    file_.close();
}

void Model::reset_kv() {
    kv_len_ = 0;
    clear_conv_state();
    if (gpu_fused_) metal::fwd_reset_conv();
}

void Model::clear_conv_state() { std::fill(conv_state_.begin(), conv_state_.end(), 0.f); }

bool Model::grow_kv(int need) {
    if (need <= kv_cap_) return true;
    if (need > n_ctx_) return false;
    int cap = kv_cap_ > 0 ? kv_cap_ : 256;
    while (cap < need) cap *= 2;
    if (cap > n_ctx_) cap = n_ctx_;

    if (gpu_fused_) {
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
    // f32_ref_ 打开时强制走纯 f32 点积。CPU 的默认路径会把激活量化成
    // Q8（快 2~4 倍，但有 ~1e-3 的相对误差），所以校验 GPU 数值时必须关掉它，
    // 否则量的是 CPU 自己的近似误差，不是 GPU 的。
    if (gpu_active_ && (int64_t)n_out * n_in >= kGpuMinMac && metal::supports(w.type, n_in)) {
        const uint64_t woff =
            (uint64_t)((const char*)file_.tensor_data(w) - (const char*)file_.mapping());
        if (metal::mfvec(w.type, x, out, n_out, n_in, woff)) return;
        // GPU 一旦出错就永久降级，避免每层都白试一次、把速度拖死。
        gpu_active_ = false;
        std::fprintf(stderr, "[dubhe] GPU 矩阵乘失败，已回退 CPU：%s\n", metal::last_error());
    }
    if (!f32_ref_ && act_q8_supported(w.type)) {
        // 激活只量化一次，所有行共用
        aq_.resize(act_q8_bytes(w.type, n_in));
        quantize_act(w.type, x, aq_.data(), n_in);
        MatvecCtx ctx{aq_.data(), out, (const char*)file_.tensor_data(w),
                      type_row_bytes(w.type, n_in), w.type, n_in, true};
        parallel_for((size_t)n_out, matvec_work, &ctx);
        return;
    }
    MatvecCtx ctx{x, out, (const char*)file_.tensor_data(w),
                  type_row_bytes(w.type, n_in), w.type, n_in, false};
    parallel_for((size_t)n_out, matvec_work, &ctx);
}

// 每 head 独立 RMSNorm（LFM3 的 QK norm）。权重只有 head_dim 长，
// 所有 head 共用同一份。原地归一化是安全的：ss 先算完才写回。
void Model::rmsnorm_heads(float* v, const TensorInfo& w, int hd, int n_head) const {
    const void* wp = file_.tensor_data(w);
    const float* wf = (w.type == T_F32) ? (const float*)wp : nullptr;
    std::vector<float> wtmp;
    if (!wf) {
        wtmp.resize((size_t)hd);
        dequant_row(w.type, wp, wtmp.data(), hd);
        wf = wtmp.data();
    }
    const float inv_n = 1.f / (float)hd;
    for (int h = 0; h < n_head; h++) {
        float* x = v + (size_t)h * hd;
        float ss = 0.f;
        for (int i = 0; i < hd; i++) ss += x[i] * x[i];
        const float scale = 1.f / std::sqrt(ss * inv_n + cfg_.rms_eps);
        for (int i = 0; i < hd; i++) x[i] = x[i] * scale * wf[i];
    }
}

// LFM3 ShortConv 子层。
//
//   BCx    = in_proj(x)              → 3*n_embd，按 [B|C|X] 切开
//   hidden = B ⊙ X
//   y      = C ⊙ (kernel ⊛ hidden)   ← depthwise 因果卷积，核宽 l_cache
//   out    = out_proj(y)
//
// 因果卷积靠一个 (l_cache-1) 长的历史缓冲实现：每来一个 token 就把 hidden
// 压进环形缓冲，然后对每个通道做一次长度 l_cache 的点积。参数量只有
// l_cache*n_embd ≈ 14K，但它是这条架构里唯一的"局部时序"算子，
// 相当于给每层加了一个可学习的短 FIR，成本远低于注意力。
void Model::shortconv(const Layer& L, int slot, const float* xin) {
    const int n = cfg_.n_embd;
    const int K = cfg_.conv_l_cache;
    if (K < 1 || conv_bc_.empty()) return;

    matvec(conv_bc_.data(), xin, L.conv_in, 3 * n, n);
    const float* gb = conv_bc_.data();
    const float* gc = conv_bc_.data() + n;
    const float* gx = conv_bc_.data() + 2 * n;

    float* h = conv_h_.data();
    for (int i = 0; i < n; i++) h[i] = gb[i] * gx[i];

    // 历史缓冲 st 存的是 h(t-K+1) .. h(t-1)，共 K-1 个，st[0] 最旧。
    // 卷积必须在移位之前算：t 时刻要看的是 st[0..K-2] 加上当前的 h(t)，
    // 加起来正好 K 个。先移位再卷积会让核整体偏移一位，
    // 并把 h(t) 同时算进 W[K-2] 和 W[K-1] 两次。
    const float* W = (const float*)file_.tensor_data(L.conv_kernel);
    float* y = conv_bc_.data() + 2 * n;   // X 已经用掉了，这块可以覆盖
    const float* st = conv_state_.data() + (size_t)slot * (size_t)(K - 1) * n;
    for (int i = 0; i < n; i++) {
        const float* wi = W + (size_t)i * K;
        float acc = 0.f;
        for (int k = 0; k < K - 1; k++) acc += wi[k] * st[(size_t)k * n + i];
        acc += wi[K - 1] * h[i];
        y[i] = acc * gc[i];
    }

    // 卷积算完再滚动历史：丢掉最旧的，把 h(t) 放到末尾
    if (K > 1) {
        float* stw = conv_state_.data() + (size_t)slot * (size_t)(K - 1) * n;
        if (K > 2) std::memmove(stw, stw + n, sizeof(float) * (size_t)(K - 2) * n);
        std::memcpy(stw + (size_t)(K - 2) * n, h, sizeof(float) * n);
    }

    matvec(attn_out_.data(), y, L.conv_out, n, n);
}

void Model::rope(float* q, float* k, int n_head, int n_head_kv, int head_dim, int pos,
                 float theta, int angles) const {
    const int half = head_dim / 2;
    // partial RoPE：只转前 angles 个角度，后面的 cos=1/sin=0 等价于恒等，
    // 直接跳过更省。LFM3 的 full_attention 只转 32/64，sliding_attention 转满。
    // 频率分母恒为 head_dim（不是 2*angles）——这是 zero-tail 布局的定义：
    // 被旋转的前缀仍然用完整 head_dim 的频率级数。
    if (angles <= 0 || angles > half) angles = half;
    for (int h = 0; h < n_head + n_head_kv; h++) {
        float* v = (h < n_head) ? (q + (size_t)h * head_dim)
                                : (k + (size_t)(h - n_head) * head_dim);
        for (int i = 0; i < angles; i++) {
            const float freq = 1.f / std::pow(theta, (float)(2 * i) / (float)head_dim);
            const float a = pos * freq;
            const float c = std::cos(a), s = std::sin(a);
            const float x0 = v[i], x1 = v[i + half];
            v[i]        = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }
}

int Model::batch_size() const {
    if (!gpu_fused_) return 1;
    return metal::fwd_batch_max();
}

// prefill 批处理。GPU 可用时一次提交算完 n 个 token；否则退化成 n 次
// 单 token 前向（结果一致，只是慢——CPU 侧还没有批量内核）。
bool Model::forward_batch(const int32_t* tokens, int n, int32_t pos0, float* logits,
                          std::string* err) {
    if (!loaded_) { if (err) *err = "模型未加载"; return false; }
    if (n <= 0 || !tokens) { if (err) *err = "空批"; return false; }
    if (n == 1) return forward(tokens[0], pos0, logits, err);

    if (!gpu_fused_) {
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
    // 图像/音频 token 的嵌入由塔算好，直接覆盖掉占位 token 的词嵌入
    if (mm_embeds_) {
        for (int i = 0; i < n; i++) {
            const int p = pos0 + i;
            if (p < mm_at_ || p >= mm_at_ + mm_n_) continue;
            std::memcpy(xbatch_.data() + (size_t)i * n_embd,
                        mm_embeds_ + (size_t)(p - mm_at_) * n_embd,
                        sizeof(float) * (size_t)n_embd);
        }
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
    if (mm_embeds_ && pos >= mm_at_ && pos < mm_at_ + mm_n_) {
        std::memcpy(x_.data(), mm_embeds_ + (size_t)(pos - mm_at_) * n_embd,
                    sizeof(float) * (size_t)n_embd);
    }

    if (gpu_fused_) {
        // 整层前向一次提交、一次同步。失败就整体报错——此时 GPU 的 KV 已经
        // 前进了一步，切回 CPU 会得到错误结果，静默降级比报错更危险。
        if (!metal::fwd_token(x_.data(), pos, logits, err)) {
            gpu_active_ = false;
            gpu_fused_ = false;
            if (err && err->empty()) *err = "GPU 前向失败";
            return false;
        }
        kv_len_ = pos + 1;
        return true;
    }

    const float scale = 1.f / std::sqrt((float)hd);

    for (int l = 0; l < cfg_.n_layer; l++) {
        const Layer& L = layers_[l];

        // ---- 子层 1：ShortConv 或 注意力
        if (L.kind == LK_CONV) {
            rmsnorm(xb_.data(), x_.data(), L.attn_norm, n_embd);
            shortconv(L, conv_slot_[(size_t)l], xb_.data());
        } else {
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

            // LFM3：QK RMSNorm 在 RoPE 之前，作用在每个 head 的 head_dim 上
            if (L.has_qk_norm) {
                rmsnorm_heads(q_.data(), L.attn_q_norm, hd, n_head);
                rmsnorm_heads(k_.data(), L.attn_k_norm, hd, n_kv);
            }

            const float theta = (L.kind == LK_ATTN_SWA && cfg_.rope_theta_swa > 0.f)
                                    ? cfg_.rope_theta_swa : cfg_.rope_theta;
            rope(q_.data(), k_.data(), n_head, n_kv, hd, pos, theta, L.rope_angles_l);

            float* kc = k_cache_.data() + (size_t)l * kv_cap_ * kv_dim;
            float* vc = v_cache_.data() + (size_t)l * kv_cap_ * kv_dim;
            std::memcpy(kc + (size_t)pos * kv_dim, k_.data(), sizeof(float) * kv_dim);
            std::memcpy(vc + (size_t)pos * kv_dim, v_.data(), sizeof(float) * kv_dim);

            // 滑窗层只看最近 n_swa 个位置；全注意力层 t_start 恒为 0
            int t_start = 0;
            if (L.kind == LK_ATTN_SWA && cfg_.n_swa > 0 && pos + 1 > cfg_.n_swa) {
                t_start = pos + 1 - cfg_.n_swa;
            }

            AttnCtx actx{q_.data(), kc, vc, attn_out_.data(), n_head, n_kv, hd, pos + 1, kv_mul,
                         t_start, scale};
            parallel_for((size_t)n_head, attn_work, &actx);

            // LFM3：逐 head 的 sigmoid 输出门，在 o_proj 之前乘上去
            if (L.has_attn_gate) {
                float* g = gate_.data();
                matvec(g, xb_.data(), L.attn_gate, n_head, n_embd);
                for (int h = 0; h < n_head; h++) {
                    const float s = 1.f / (1.f + std::exp(-g[h]));
                    float* o = attn_out_.data() + (size_t)h * hd;
                    for (int d = 0; d < hd; d++) o[d] *= s;
                }
            }

            // 3) attention 输出投影
            matvec(xb2_.data(), attn_out_.data(), L.attn_output, n_embd, n_embd);
            std::memcpy(attn_out_.data(), xb2_.data(), sizeof(float) * n_embd);
        }

        // sandwich norm：子层输出先归一化再加残差（LFM3）
        if (L.has_sandwich) rmsnorm(attn_out_.data(), attn_out_.data(), L.post_attn_norm, n_embd);
        for (int i = 0; i < n_embd; i++) x_[i] += attn_out_[i];

        // ---- 子层 2：FFN（SwiGLU）
        rmsnorm(xb_.data(), x_.data(), L.ffn_norm, n_embd);
        matvec(gate_.data(), xb_.data(), L.ffn_gate, cfg_.n_ff, n_embd);
        matvec(up_.data(), xb_.data(), L.ffn_up, cfg_.n_ff, n_embd);
        for (int i = 0; i < cfg_.n_ff; i++) ffn_out_[i] = silu(gate_[i]) * up_[i];
        {
            float* dst = ffn_down_out_.data();
            matvec(dst, ffn_out_.data(), L.ffn_down, n_embd, cfg_.n_ff);
            if (L.has_sandwich) rmsnorm(dst, dst, L.post_ffn_norm, n_embd);
            for (int i = 0; i < n_embd; i++) x_[i] += dst[i];
        }
    }

    // 5) final norm + logits
    rmsnorm(xb_.data(), x_.data(), out_norm_, n_embd);
    matvec(logits, xb_.data(), out_w_, cfg_.n_vocab, n_embd);

    kv_len_ = pos + 1;
    return true;
}

}  // namespace dubhe
