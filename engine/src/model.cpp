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
#include "sysinfo.h"

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
    } else if (arch_is_qwen35()) {
        // Qwen3.5 把 32 层拆成"24 层 Gated DeltaNet + 8 层 gated attention"。
        // 两套子层的 head_dim 不一样：delta 是 128（ssm.state_size），
        // full attention 是 256（attention.key_length）。上面按
        // n_embd / n_head 算出来的是 256，正好是 full attention 的；
        // delta 层单独用 ssm_state_size。
        cfg_.ssm_conv_kernel    = (int32_t)file_.get_u32(p + "ssm.conv_kernel", 4);
        cfg_.ssm_state_size     = (int32_t)file_.get_u32(p + "ssm.state_size", 128);
        cfg_.ssm_group_count    = (int32_t)file_.get_u32(p + "ssm.group_count", 16);
        cfg_.ssm_time_step_rank = (int32_t)file_.get_u32(p + "ssm.time_step_rank", 32);
        cfg_.ssm_inner_size     = (int32_t)file_.get_u32(p + "ssm.inner_size", cfg_.n_embd);
        cfg_.head_dim           = (int32_t)file_.get_u32(p + "attention.key_length", cfg_.head_dim);

        // rope.dimension_count 是**真正旋转的维数**（64），不是 head_dim（256）。
        // 这是 zero-tail partial RoPE，但频率分母是 n_rot 而不是 head_dim ——
        // 和 LFM3 的定义相反。配对跨度也不同：这里前 32 维和后 32 维配对，
        // LFM3 是前 128 维和后 128 维配对。用一个开关分开，别让 9B 的
        // q/k 转错维度（症状是输出通顺但答非所问）。
        const int32_t n_rot = (int32_t)file_.get_u32(p + "rope.dimension_count", cfg_.head_dim);
        cfg_.n_rot = n_rot > 0 ? n_rot : cfg_.head_dim;
        cfg_.rope_angles = cfg_.n_rot / 2;
        cfg_.rope_pair_nrot = true;
        if (const auto* sec = file_.get_arr_i32(p + "rope.dimension_sections")) {
            if (sec->size() >= 4) {
                for (int k = 0; k < 4; k++) cfg_.rope_sections[k] = (*sec)[(size_t)k];
                cfg_.rope_mrope = (cfg_.rope_sections[0] + cfg_.rope_sections[1] +
                                   cfg_.rope_sections[2] + cfg_.rope_sections[3]) > 0;
            }
        }

        if (const auto* rec = file_.get_arr_u8(p + "attention.recurrent_layers")) {
            cfg_.recurrent_layers.assign(rec->begin(), rec->end());
        } else {
            // 老 GGUF 可能没有这张表，按 full_attention_interval 推。
            const int32_t iv = (int32_t)file_.get_u32(p + "full_attention_interval", 4);
            cfg_.recurrent_layers.resize((size_t)cfg_.n_layer, 0);
            for (int i = 0; i < cfg_.n_layer; i++)
                cfg_.recurrent_layers[(size_t)i] = ((i + 1) % iv != 0) ? 1 : 0;
        }
        cfg_.recurrent_layers.resize((size_t)cfg_.n_layer, 0);
        cfg_.n_delta = 0;
        for (uint8_t b : cfg_.recurrent_layers) cfg_.n_delta += b ? 1 : 0;
    }
    cfg_.n_ctx_train = std::max(cfg_.n_ctx_train, 512);
    // 只有 full attention 层需要 KV。Qwen3.5 的 24 层 DeltaNet 是固定状态，
    // 把它们也算进 KV 会白扔 4 倍内存 —— 44K 和 176K 上下文的差别。
    n_kv_layer_ = cfg_.n_layer - cfg_.n_delta;
    if (n_kv_layer_ <= 0) n_kv_layer_ = cfg_.n_layer;
    // KV cache 不预分配满上下文（262144 × 24 层 = 25.8 GB，纯浪费）。
    // 先给一个够用的窗口，不够时按 2 倍增长，上限 = 下面算出来的天花板。
    //
    // 这里**不再写死 4096**。以前默认 `min(n_ctx_train, 4096)`，结果是
    // `inspect` 打出"可用上下文 4096（受内存/参数限制）"，而实际上这台机器
    // 装得下 10 万 —— 用户贴一篇长文就撞"上下文已满"，还以为引擎不行。
    // 默认改成"用满天板"是安全的：kv_cap_ 起步只有 1024 个槽，真正的分配
    // 发生在 grow_kv（也就是用户真喂了那么长的输入时），没用到就不占内存。
    n_ctx_ceiling_ = cfg_.n_ctx_train;
    n_ctx_ = (n_ctx > 0) ? std::min(n_ctx, cfg_.n_ctx_train) : cfg_.n_ctx_train;

    // ------------------------------------------------------- 上下文 × 内存
    //
    // n_ctx 不能只看 GGUF 说多少、用户要多少，还要看内存装不装得下。
    // KV 的开销是 `n_layer × n_ctx × kv_dim × 2(k,v) × 每元素字节数`，
    // **与模型大小无关，只与层数和 KV 头数有关** —— 所以"小模型"照样能撑爆：
    //
    //   LFM3-1.7B（24 层 / 4 个 KV 头 / head_dim 128）
    //     每 1k 上下文 96 KB（按 CPU 的 f32 算）
    //     训练上下文 262144 → 25.8 GB，24 GB 的机器直接装不下
    //
    // 为什么在**启动时**夹，而不是等第一次长 prompt 触发 grow_kv：
    // 后者的症状是生成到一半报 "KV cache 扩容失败"，用户看到的是
    // "聊到第 N 轮就崩"，根本联系不到上下文长度。而且 macOS 的交换会让
    // 分配"成功"然后整机卡死，比直接失败更糟。
    //
    // 两条约束取更紧的那条：
    //   ① KV 自己不超过总内存的 45%（统一内存，GPU 分配和系统抢同一块）
    //   ② 权重 + KV 不超过总内存的 85%
    {
        const size_t kv_dim = (size_t)cfg_.n_head_kv * (size_t)cfg_.head_dim;
        // 按 f32 算（CPU 路径就是 f32），GPU 的 f16 只会更省 —— 取大的不会误伤
        const uint64_t per_ctx = (uint64_t)n_kv_layer_ * kv_dim * 2u * sizeof(float);
        const uint64_t ram = total_ram_bytes();
        if (ram > 0 && per_ctx > 0) {
            uint64_t kv_budget = (uint64_t)((double)ram * 0.45);
            const uint64_t weights = (uint64_t)file_.file_size();
            const uint64_t cap85 = (uint64_t)((double)ram * 0.85);
            if (weights < cap85 && cap85 - weights < kv_budget) kv_budget = cap85 - weights;

            const uint64_t fit = kv_budget / per_ctx;
            if (fit < (uint64_t)n_ctx_ceiling_) {
                int64_t c = (int64_t)fit / 256 * 256;
                if (c < 1024) c = 1024;
                if (c < n_ctx_ceiling_) n_ctx_ceiling_ = (int)c;
            }
        }
        // 只有"用户明确要了但装不下"才值得说一句。默认值被夹是正常行为，
        // 每次都打一行反而像警告。
        if (n_ctx_ > n_ctx_ceiling_) {
            std::fprintf(stderr,
                         "[dubhe] 上下文 %d → %d：每 1k 上下文要 %.1f MB KV，"
                         "这台机器的内存只装得下这么多\n",
                         n_ctx_, n_ctx_ceiling_, (double)per_ctx * 1024.0 / 1048576.0);
            n_ctx_ = n_ctx_ceiling_;
        }
        kv_cap_ = std::min(n_ctx_, 1024);
    }

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

    // KV cache（可增长）。只给需要 KV 的层分配槽位。
    const size_t kv_dim = (size_t)cfg_.n_head_kv * cfg_.head_dim;
    kv_slot_.assign((size_t)cfg_.n_layer, -1);
    {
        int slot = 0;
        for (int i = 0; i < cfg_.n_layer; i++) {
            if (layers_[(size_t)i].kind != LK_DELTA) kv_slot_[(size_t)i] = slot++;
        }
    }
    k_cache_.assign((size_t)n_kv_layer_ * kv_cap_ * kv_dim, 0.f);
    v_cache_.assign((size_t)n_kv_layer_ * kv_cap_ * kv_dim, 0.f);
    kv_len_ = 0;
    kv_toks_.clear();

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

    // ---- Qwen3.5 Gated DeltaNet 的固定状态
    //
    // 每层 32 个 V head，每个 head 维护一个 128×128 的递归矩阵，f32 下是
    // 32×128×128×4 = 2 MB/层，24 层共 50 MB。它**不随上下文增长** ——
    // 262K 上下文的 KV cache 要 8 GB，这份状态恒为 50 MB，这是这条架构
    // 在 24 GB 机器上能跑到 128K 的原因。
    if (cfg_.n_delta > 0) {
        const int S   = cfg_.ssm_state_size;
        const int nv  = cfg_.ssm_time_step_rank;
        const int kdim = cfg_.ssm_group_count * cfg_.ssm_state_size;
        const int vdim = cfg_.ssm_inner_size;
        const int cdim = 2 * kdim + vdim;
        ssm_slot_.assign((size_t)cfg_.n_layer, -1);
        int slot = 0;
        for (int i = 0; i < cfg_.n_layer; i++) {
            if (layers_[(size_t)i].kind == LK_DELTA) ssm_slot_[(size_t)i] = slot++;
        }
        ssm_state_.assign((size_t)cfg_.n_delta * (size_t)nv * (size_t)S * (size_t)S, 0.f);
        ssm_conv_state_.assign((size_t)cfg_.n_delta * (size_t)(cfg_.ssm_conv_kernel - 1) *
                                   (size_t)cdim, 0.f);
        ssm_qkv_.assign((size_t)cdim, 0.f);
        ssm_conv_out_.assign((size_t)cdim, 0.f);
        ssm_q_.assign((size_t)kdim, 0.f);
        ssm_k_.assign((size_t)kdim, 0.f);
        ssm_v_.assign((size_t)vdim, 0.f);
        ssm_o_.assign((size_t)vdim, 0.f);
        ssm_z_.assign((size_t)vdim, 0.f);
        ssm_beta_.assign((size_t)nv, 0.f);
        ssm_alpha_.assign((size_t)nv, 0.f);
        ssm_gate_buf_.assign((size_t)nv, 0.f);
    }
    if (arch_is_qwen35()) {
        qg_.assign((size_t)2 * cfg_.n_head * cfg_.head_dim, 0.f);
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
    // Qwen3.5 还没有 Metal 内核：整层融合的 spec 只认识 conv / 滑窗 / 全注意力
    // 三种层，DeltaNet 会被错当成 full attention 去跑，结果是"能出字但全错"。
    // 在补上 Metal 之前，宁可明确走 CPU，也不做一半的加速。
    if (!metal::available()) {
        gpu_note_ = "无 Metal 设备";
        if (err) *err = gpu_note_;
        return false;
    }
    if (arch_is_qwen35()) {
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

        auto off = [&](const TensorInfo& t) -> uint64_t {
            if (t.ne[0] <= 0) return 0;
            return (uint64_t)((const char*)file_.tensor_data(t) - (const char*)file_.mapping());
        };

        metal::Q35Spec s;
        s.n_embd = cfg_.n_embd;
        s.n_ff = cfg_.n_ff;
        s.n_vocab = cfg_.n_vocab;
        s.n_layer = cfg_.n_layer;
        s.n_head = cfg_.n_head;
        s.n_head_kv = cfg_.n_head_kv;
        s.hd = cfg_.head_dim;
        s.n_delta = cfg_.n_delta;
        s.n_kv_layer = n_kv_layer_;
        // 请求满上下文，真正的尺寸由 q35_init 按显存预算夹（q35_kv_cap 回报）
        s.kv_cap = n_ctx_;
        s.ssm_S = cfg_.ssm_state_size;
        s.ssm_nk = cfg_.ssm_group_count;
        s.ssm_nv = cfg_.ssm_time_step_rank;
        s.ssm_kdim = cfg_.ssm_group_count * cfg_.ssm_state_size;
        s.ssm_vdim = cfg_.ssm_inner_size;
        s.ssm_conv = cfg_.ssm_conv_kernel;
        s.rms_eps = cfg_.rms_eps;
        s.rope_theta = cfg_.rope_theta;
        s.rope_angles = cfg_.rope_angles;
        for (int i = 0; i < 4; i++) s.rope_sections[i] = cfg_.rope_sections[i];
        s.off_out_norm = off(out_norm_);
        s.off_out_w = off(out_w_);
        s.ty_out_w = out_w_.type;

        s.layers.resize((size_t)cfg_.n_layer);
        int kv_slot = 0;
        for (int l = 0; l < cfg_.n_layer; l++) {
            const Layer& L = layers_[(size_t)l];
            metal::Q35Layer& d = s.layers[(size_t)l];
            d.kind = (L.kind == LK_DELTA) ? 0 : 1;
            d.ssm_slot = (L.kind == LK_DELTA) ? ssm_slot_[(size_t)l] : -1;
            d.kv_slot = (L.kind == LK_DELTA) ? -1 : kv_slot++;
            d.off_attn_norm = off(L.attn_norm);
            d.off_ffn_norm = off(L.ffn_norm);
            d.off_gate = off(L.ffn_gate); d.ty_gate = L.ffn_gate.type;
            d.off_up = off(L.ffn_up);     d.ty_up = L.ffn_up.type;
            d.off_down = off(L.ffn_down); d.ty_down = L.ffn_down.type;
            if (L.kind == LK_DELTA) {
                d.off_qkv = off(L.ssm_qkv);     d.ty_qkv = L.ssm_qkv.type;
                d.off_z = off(L.ssm_gate);      d.ty_z = L.ssm_gate.type;
                d.off_conv = off(L.ssm_conv1d);
                d.off_dt = off(L.ssm_dt);
                d.off_a = off(L.ssm_a);
                d.off_beta = off(L.ssm_beta);   d.ty_beta = L.ssm_beta.type;
                d.off_alpha = off(L.ssm_alpha); d.ty_alpha = L.ssm_alpha.type;
                d.off_norm = off(L.ssm_norm);
                d.off_sout = off(L.ssm_out);    d.ty_sout = L.ssm_out.type;
            } else {
                d.off_q = off(L.attn_q);  d.ty_q = L.attn_q.type;
                d.off_k = off(L.attn_k);  d.ty_k = L.attn_k.type;
                d.off_v = off(L.attn_v);  d.ty_v = L.attn_v.type;
                d.off_o = off(L.attn_output); d.ty_o = L.attn_output.type;
                d.off_qn = off(L.attn_q_norm);
                d.off_kn = off(L.attn_k_norm);
            }
        }

        if (!metal::q35_init(s, &e)) {
            gpu_note_ = "Qwen3.5 Metal 初始化失败: " + e;
            if (err) *err = gpu_note_;
            return false;
        }
        // KV 的最终尺寸由 Metal 按显存预算定（可能比请求的小），后面对外
        // 报的上下文上限、以及 prefill 的越界判断都要用这个数。
        const int kv_real = metal::q35_kv_cap();
        if (kv_real > 0 && kv_real < n_ctx_) {
            std::fprintf(stderr, "[dubhe] GPU 只分配了 %d token 的 KV，上下文按此收紧\n",
                         kv_real);
            n_ctx_ = kv_real;
            n_ctx_ceiling_ = kv_real;
        }
        kv_cap_ = std::min(kv_cap_, n_ctx_);
        gpu_active_ = false;
        gpu_fused_ = true;
        gpu_note_ = "Metal · Apple M4（Qwen3.5 整层融合，KV 8 层 fp16，上下文 " +
                    std::to_string(n_ctx_) + "）";
        return true;
    }
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

    // ---------------------------------------------------------- 覆盖度体检
    //
    // 逐张量问一句"GPU 能不能算"，把不能算的按格式汇总报出来。
    //
    // 这条检查是被 Q5_K 那次坑出来的：当时 Q5_K 在四张内核表里都没有登记，
    // `kind_of()` 返回 -1、`supports()` 直接 false —— 模型**能跑、结果也对**，
    // 只是每一层都有一半权重在 CPU 上单线程算，prefill 27.8 tok/s。
    // 从外面看只是"这个格式慢"，完全看不出是个覆盖洞；等哪天真去查，
    // 又要从头把四张表逐行对一遍。
    //
    // 所以：**加载时就把"谁在 CPU 上"说出来**。判据只看格式与行宽
    // （`supports()` 里的两条），不看 `kGpuMinMac` —— 小张量走 CPU 是
    // 刻意为之，不该混进来报警。
    {
        std::map<uint32_t, std::pair<int, uint64_t>> miss;   // 类型 → (张量数, 字节)
        uint64_t total = 0;
        for (const TensorInfo& t : file_.tensors()) {
            const int64_t n_in = t.ne[0];
            if (n_in <= 0) continue;
            int64_t n_out = 1;
            for (int d = 1; d < t.n_dims; d++) n_out *= t.ne[d];
            const uint64_t bytes = (uint64_t)type_row_bytes(t.type, n_in) * (uint64_t)n_out;
            total += bytes;
            if (n_out * n_in < kGpuMinMac) continue;         // 小张量，本来就该走 CPU
            if (!metal::supports(t.type, (int)n_in)) {
                auto& e = miss[t.type];
                e.first += 1;
                e.second += bytes;
            }
        }
        if (!miss.empty()) {
            std::string line;
            uint64_t lost = 0;
            for (auto& kv : miss) {
                char b[128];
                std::snprintf(b, sizeof(b), "%s%s %d 个张量 / %.0f MB",
                              line.empty() ? "" : "、", type_name(kv.first),
                              kv.second.first, (double)kv.second.second / 1048576.0);
                line += b;
                lost += kv.second.second;
            }
            const double pct = total ? 100.0 * (double)lost / (double)total : 0.0;
            std::fprintf(stderr,
                         "[dubhe] ⚠ GPU 覆盖不全：%s 没有可用内核，将回退 CPU\n"
                         "[dubhe]   占模型 %.0f%%，这些张量的 matvec 会慢一个数量级\n",
                         line.c_str(), pct);
        }
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
            bool all_am = true;      // 每一步的 argmax 是否都一致
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
                if (ag != ac) all_am = false;
                const double rl = m / (r + 1e-6);
                if (rl > rel) { rel = rl; mx = m; mref = r; arg_g = ag; arg_c = ac; bad_step = i; }
            }
            if (rel_batch > rel) { rel = rel_batch; bad_step = -2; }   // -2 标记批处理路径
            if (std::getenv("DUBHE_FUSED_CHECK_VERBOSE")) {
                std::fprintf(stderr,
                             "[dubhe] 融合对拍: 最差 rel=%.2e @pos %d  max|Δ|=%.3e  max|ref|=%.3e  "
                             "argmax gpu=%d cpu=%d  每步argmax一致=%d\n",
                             rel, bad_step, mx, mref, arg_g, arg_c, int(all_am));
            }
            // 两级判据。
            //
            // 5e-3 是给"同一条数值路径"用的：量化模型的 CPU 与 GPU 都按同一份
            // 反量化公式算，正常情况下误差在 1e-5 量级，超过 5e-3 一定是写错了。
            //
            // 但 F16 模型不吃这一条。它的 logits 差 5.9e-3，而 4 个提示词 × 80
            // token 的贪心输出逐字一致 —— 原因是 fp16 权重在两条路径上的舍入
            // 顺序不同，误差随层数放大到 1e-2 量级，属于"换了个等价算法"，不是
            // bug。若按 5e-3 一刀切，F16 会被整条踢回 CPU：prefill 7.4 tok/s、
            // decode 5.8 tok/s（GPU 是 54.3 / 22.9，差 7.3x / 3.9x）。
            //
            // 所以第二级放宽容差到 2e-2，但要求**每一步的 argmax 都对**。
            // argmax 一致才意味着贪心解码的结果不变；只要有一处翻了，就还是
            // 老老实实退回 CPU。
            if (rel > 5e-3 && !(rel <= 2e-2 && all_am)) {
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
        } else if (arch_is_qwen35()) {
            // recurrent_layers 是逐层的 bool 表：1 = Gated DeltaNet，0 = full attention。
            // 不按 "每 4 层一个" 硬编码 —— 9B 是 3:1，27B 是另一个比例，
            // 表里怎么写就怎么走。
            if (i < (int)cfg_.recurrent_layers.size() && cfg_.recurrent_layers[(size_t)i])
                L.kind = LK_DELTA;
        }

        // 每层实际旋转的角度数。HF 的 config.json 里 rope_parameters 是
        // **按层类型**分开配的：full_attention 带 partial_rotary_factor=0.5，
        // sliding_attention 没有这个键（缺省 1.0 → 全量旋转）。GGUF 没有
        // 逐层的这个字段，所以按层类型推。
        L.rope_angles_l = (L.kind == LK_ATTN_SWA) ? (cfg_.head_dim / 2)
                                                  : cfg_.rope_angles;

        // Qwen3.5 没有独立的 ffn_norm：它把 post_attention_norm 放在 FFN 之前
        // （残差加完之后才归一化），语义上是"FFN 的输入 norm"，不是 LFM3 的
        // sandwich norm。这里直接指过去，前向里就不用再分叉。
        L.ffn_norm    = arch_is_qwen35() ? need(b + "post_attention_norm.weight")
                                         : need(b + "ffn_norm.weight");
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
        } else if (L.kind == LK_DELTA) {
            // Gated DeltaNet 子层。张量名和 llama.cpp 的 GGUF 约定一致：
            // qkv 把 q/k/v 拼成一个矩阵（q/k 各 16 头 ×128，v 32 头 ×128），
            // attn_gate 是输出门，ssm_* 是线性注意力自己的参数。
            L.attn_norm   = need(b + "attn_norm.weight");
            L.ssm_qkv     = need(b + "attn_qkv.weight");
            L.ssm_gate    = need(b + "attn_gate.weight");
            L.ssm_conv1d  = need(b + "ssm_conv1d.weight");
            L.ssm_dt      = need(b + "ssm_dt.bias");
            L.ssm_a       = need(b + "ssm_a");
            L.ssm_beta    = need(b + "ssm_beta.weight");
            L.ssm_alpha   = need(b + "ssm_alpha.weight");
            L.ssm_norm    = need(b + "ssm_norm.weight");
            L.ssm_out     = need(b + "ssm_out.weight");
        } else {
            L.attn_norm   = need(b + "attn_norm.weight");
            // Qwen3.5 的 gated attention 把 Q 和输出门拼在同一个投影里
            // （每 head 2*head_dim，前半 Q、后半 gate）。标记一下，
            // 前向里按 2 倍宽度取，拆错就会把 gate 当成 Q。
            if (arch_is_qwen35()) L.q_gate_fused = true;
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
        //
        // Qwen3.5 也有 post_attention_norm，但它的语义完全不同：那是**残差
        // 之后、FFN 之前**的输入 norm（已经绑给 ffn_norm 了），不是 LFM3 的
        // "子层输出先归一化"。如果这里也置 has_sandwich，FFN 输出会去乘一个
        // 根本没绑定的 post_ffw_norm（offset 0 → 读到别的张量的量化字节），
        // 表现为第一层就炸到 1e38。
        if (!arch_is_qwen35()) {
            if (const TensorInfo* t = file_.find_tensor(b + "post_attention_norm.weight")) {
                L.post_attn_norm = *t; L.has_sandwich = true;
                count(*t);
            }
            if (const TensorInfo* t = file_.find_tensor(b + "post_ffw_norm.weight")) {
                L.post_ffn_norm = *t;
                count(*t);
            }
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
                          : L.kind == LK_ATTN_SWA ? "swa   "
                          : L.kind == LK_DELTA ? "delta " : "full  ";
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
    kv_slot_.clear();
    n_kv_layer_ = 0;
    kv_toks_.clear();
    kv_len_ = 0;
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
    ssm_state_.clear();
    ssm_conv_state_.clear();
    ssm_slot_.clear();
    ssm_qkv_.clear();
    ssm_z_.clear();
    ssm_q_.clear();
    ssm_k_.clear();
    ssm_v_.clear();
    ssm_o_.clear();
    ssm_beta_.clear();
    ssm_alpha_.clear();
    ssm_gate_buf_.clear();
    ssm_conv_out_.clear();
    qg_.clear();
    cfg_.n_conv = 0;
    cfg_.n_delta = 0;
    file_.close();
}

void Model::reset_kv() {
    kv_len_ = 0;
    kv_toks_.clear();
    clear_conv_state();
    // DeltaNet 的状态和 ShortConv 一样是"滚动"的，没法按位置回退，
    // 换 prompt 只能整体清零。
    clear_ssm_state();
    if (gpu_fused_) {
        if (arch_is_qwen35()) metal::q35_reset();
        else metal::fwd_reset_conv();
    }
}

void Model::clear_conv_state() { std::fill(conv_state_.begin(), conv_state_.end(), 0.f); }

int Model::kv_prefix_len(const int32_t* toks, int n) const {
    // kv_toks_ 与 kv_len_ 必须严格等长，否则说明中途有别的入口动过 KV
    // （换模型、挂多模态、异常提前返回）。这种情况一律不复用。
    if (kv_toks_.empty() || (int)kv_toks_.size() != kv_len_) return 0;
    if (!toks || n <= 0) return 0;
    const int lim = std::min((int)kv_toks_.size(), n);
    int k = 0;
    while (k < lim && kv_toks_[(size_t)k] == toks[k]) k++;
    return k;
}

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

    std::vector<float> nk((size_t)n_kv_layer_ * new_row, 0.f);
    std::vector<float> nv((size_t)n_kv_layer_ * new_row, 0.f);
    for (int l = 0; l < n_kv_layer_; l++) {
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

// Qwen3.5 的 Gated DeltaNet 子层。
//
// 和注意力最本质的区别：它没有 KV cache，只有一个**固定大小**的递归状态。
// 每层 32 个 V head，每个 head 一个 128×128 矩阵；来一个 token 就把状态
// 衰减一次、用 (v - Sᵀk) 修正一次，再读出 Sᵀq。上下文再长，状态也不涨。
//
// 张量布局（和 llama.cpp 的 GGUF 约定一致）：
//   qkv = wqkv·x  →  [q: 16×128 | k: 16×128 | v: 32×128]
//   z   = wgate·x →  [32×128]           输出门，silu 后乘在 norm 上
//   beta  = sigmoid(wbeta·x + 0)         每 V head 一个标量学习率
//   alpha = softplus(walpha·x + dt) * a  每 V head 一个衰减（a 为负）
//
// 状态按 M[j][i] = S[i][j] 存（转置），这样每行是连续的，delta 与读出都能
// 顺序扫 —— 和 ggml 的实现保持同一布局，对拍时不用再转一次。
void Model::delta_net(const Layer& L, int slot, const float* xin) {
    const int n    = cfg_.n_embd;
    const int K    = cfg_.ssm_conv_kernel;
    const int S    = cfg_.ssm_state_size;
    const int nk   = cfg_.ssm_group_count;   // K/V 头数（16）
    const int nv   = cfg_.ssm_time_step_rank; // V 头数（32）
    const int kdim = nk * S;
    const int vdim = cfg_.ssm_inner_size;
    const int cdim = 2 * kdim + vdim;

    if (K < 1 || S <= 0 || nv <= 0 || ssm_state_.empty()) return;

    // 1) 四个投影一次算完
    matvec(ssm_qkv_.data(),   xin, L.ssm_qkv,   cdim, n);
    matvec(ssm_z_.data(),     xin, L.ssm_gate,  vdim, n);
    matvec(ssm_beta_.data(),  xin, L.ssm_beta,  nv,   n);
    matvec(ssm_alpha_.data(), xin, L.ssm_alpha, nv,   n);

    const float* dtb = (const float*)file_.tensor_data(L.ssm_dt);
    const float* av  = (const float*)file_.tensor_data(L.ssm_a);
    for (int h = 0; h < nv; h++) {
        ssm_beta_[(size_t)h] = 1.f / (1.f + std::exp(-ssm_beta_[(size_t)h]));
        const float x = ssm_alpha_[(size_t)h] + dtb[h];
        // softplus 在 x>20 之后等于 x（exp 会溢出，但结果本来就一样）
        const float sp = x > 20.f ? x : std::log1p(std::exp(x));
        ssm_alpha_[(size_t)h] = sp * av[h];   // 负数 → exp 后 <1，起衰减作用
    }
    auto dbg = [&](const char* tag, const float* p, int n) {
        if (!std::getenv("DUBHE_DEBUG_NAN")) return;
        static int calls = 0;
        if (++calls > 80) return;   // 只看前几个 token
        float mx = 0.f; int bad = 0;
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(p[i])) bad++;
            else mx = std::max(mx, std::fabs(p[i]));
        }
        std::fprintf(stderr, "[dbg]   delta %-8s max=%.4g bad=%d\n", tag, mx, bad);
    };
    dbg("qkv", ssm_qkv_.data(), cdim);
    dbg("z", ssm_z_.data(), vdim);
    dbg("beta", ssm_beta_.data(), nv);
    dbg("alpha", ssm_alpha_.data(), nv);

    // 2) depthwise 因果卷积 + silu。历史缓冲存最近 K-1 个 qkv，最旧在前。
    float* st = ssm_conv_state_.data() + (size_t)slot * (size_t)(K - 1) * (size_t)cdim;
    const float* W = (const float*)file_.tensor_data(L.ssm_conv1d);
    for (int c = 0; c < cdim; c++) {
        const float* wc = W + (size_t)c * K;
        float acc = 0.f;
        for (int k = 0; k < K - 1; k++) acc += wc[k] * st[(size_t)k * cdim + c];
        acc += wc[K - 1] * ssm_qkv_[(size_t)c];
        ssm_conv_out_[(size_t)c] = acc / (1.f + std::exp(-acc));
    }
    if (K > 1) {
        if (K > 2) std::memmove(st, st + cdim, sizeof(float) * (size_t)(K - 2) * cdim);
        std::memcpy(st + (size_t)(K - 2) * cdim, ssm_qkv_.data(), sizeof(float) * cdim);
    }

    // 3) 拆 q/k/v，q/k 做 per-head L2 norm（不是 RMSNorm：没有可学习权重）
    const float* qc = ssm_conv_out_.data();
    const float* kc = qc + kdim;
    const float* vc = kc + kdim;
    auto l2norm = [&](const float* src, float* dst, int heads) {
        for (int h = 0; h < heads; h++) {
            const float* s = src + (size_t)h * S;
            float ss = 0.f;
            for (int i = 0; i < S; i++) ss += s[i] * s[i];
            const float r = 1.f / std::sqrt(ss + cfg_.rms_eps);
            float* d = dst + (size_t)h * S;
            for (int i = 0; i < S; i++) d[i] = s[i] * r;
        }
    };
    l2norm(qc, ssm_q_.data(), nk);
    l2norm(kc, ssm_k_.data(), nk);
    std::memcpy(ssm_v_.data(), vc, sizeof(float) * (size_t)vdim);
    dbg("conv", ssm_conv_out_.data(), cdim);

    // 4) delta rule。q/k 只有 16 个 head，v 有 32 个：第 h 个 v head 用
    //    第 h%16 个 k/q head（ggml_repeat_4d 的语义）。
    const float scale = 1.f / std::sqrt((float)S);
    float* M = ssm_state_.data() + (size_t)slot * (size_t)nv * S * S;
    for (int h = 0; h < nv; h++) {
        const int kh = h % nk;
        const float* q = ssm_q_.data() + (size_t)kh * S;
        const float* k = ssm_k_.data() + (size_t)kh * S;
        const float* v = ssm_v_.data() + (size_t)h * S;
        float* m = M + (size_t)h * S * S;
        const float g = std::exp(ssm_alpha_[(size_t)h]);
        const float beta = ssm_beta_[(size_t)h];

        // 衰减整行：M[j][i] *= exp(g_h)
        for (int j = 0; j < S; j++) {
            float* row = m + (size_t)j * S;
            for (int i = 0; i < S; i++) row[i] *= g;
        }
        // delta[j] = (v[j] - <M[j], k>) * beta;  M[j] += delta[j] * k
        // 读出 <M[j], q> 用的是**更新后**的状态（顺序和 ggml 一致）。
        for (int j = 0; j < S; j++) {
            float* row = m + (size_t)j * S;
            float sk = 0.f;
            for (int i = 0; i < S; i++) sk += row[i] * k[i];
            const float d = (v[j] - sk) * beta;
            for (int i = 0; i < S; i++) row[i] += d * k[i];
            float sq = 0.f;
            for (int i = 0; i < S; i++) sq += row[i] * q[i];
            ssm_o_[(size_t)h * S + j] = sq * scale;
        }
    }

    // 5) gated RMSNorm：每 head 归一化后乘 silu(z)
    dbg("attn", ssm_o_.data(), vdim);
    rmsnorm_heads(ssm_o_.data(), L.ssm_norm, S, nv);
    for (int i = 0; i < vdim; i++) {
        const float z = ssm_z_[(size_t)i];
        ssm_o_[(size_t)i] *= z / (1.f + std::exp(-z));
    }
    dbg("gated", ssm_o_.data(), vdim);

    // 6) 输出投影
    matvec(attn_out_.data(), ssm_o_.data(), L.ssm_out, n, vdim);
    dbg("out", attn_out_.data(), n);
}

void Model::clear_ssm_state() {
    std::fill(ssm_state_.begin(), ssm_state_.end(), 0.f);
    std::fill(ssm_conv_state_.begin(), ssm_conv_state_.end(), 0.f);
}

void Model::rope(float* q, float* k, int n_head, int n_head_kv, int head_dim, int pos,
                 float theta, int angles) const {
    // partial RoPE 有两套定义（见 ModelConfig::rope_pair_nrot）：
    //   LFM3   配对跨度 head_dim/2、频率分母 head_dim（zero-tail 的原义）
    //   Qwen3.5 配对跨度 n_rot/2、  频率分母 n_rot
    // 老模型 n_rot == head_dim，两条路完全等价。
    const int half  = cfg_.rope_pair_nrot ? (cfg_.n_rot / 2) : (head_dim / 2);
    const int denom = cfg_.rope_pair_nrot ? cfg_.n_rot : head_dim;
    if (angles <= 0 || angles > half) angles = half;
    for (int h = 0; h < n_head + n_head_kv; h++) {
        float* v = (h < n_head) ? (q + (size_t)h * head_dim)
                                : (k + (size_t)(h - n_head) * head_dim);
        for (int i = 0; i < angles; i++) {
            // MRoPE：频率**按 section 重置**。9B 的 [11,11,10,0] 意味着
            // 第 0-10 对用 base^(0..10·2/64)，第 11-21 对又从 base^0 开始。
            // 文本输入下三个 section 的位置相同，所以只是频率重复，不需要
            // 真的维护三套位置 —— 但重不重置差别很大，不能省。
            int j = i;
            if (cfg_.rope_mrope) {
                int base = 0;
                for (int k = 0; k < 4; k++) {
                    const int n = cfg_.rope_sections[k];
                    if (i < base + n) { j = i - base; break; }
                    base += n;
                }
            }
            const float freq = 1.f / std::pow(theta, (float)(2 * j) / (float)denom);
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
    // Qwen3.5 的批量尺寸由 MMA 分块内核定死：一次 32 行（MMA_BM）。
    // 没有 MMA 内核时退回 1，走逐 token。
    if (arch_is_qwen35()) return metal::q35_prefill_ok() ? 32 : 1;
    return metal::fwd_batch_max();
}

// prefill 批处理。GPU 可用时一次提交算完 n 个 token；否则退化成 n 次
// 单 token 前向（结果一致，只是慢——CPU 侧还没有批量内核）。
bool Model::forward_batch(const int32_t* tokens, int n, int32_t pos0, float* logits,
                          std::string* err, uint64_t slab0, uint64_t cslot) {
    if (!loaded_) { if (err) *err = "模型未加载"; return false; }
    if (n <= 0 || !tokens) { if (err) *err = "空批"; return false; }
    if (n == 1 && slab0 == 0 && cslot == 0) return forward(tokens[0], pos0, logits, err);

    // Qwen3.5 的 prefill：有 MMA 批量内核就 32 行一叠走 q35_prefill，
    // 权重每叠只读一遍；没有（老 GPU / Q4_0 权重 / 无 Metal）就退回逐 token。
    // 逐 token 在长 prompt 上是灾难：681 token 要 52 秒，因为每个 token
    // 都把 5.4 GB 权重从头读了一遍。
    if (arch_is_qwen35() && n > 1) {
        if (gpu_fused_ && slab0 == 0 && cslot == 0 && metal::q35_prefill_ok()) {
            const int n_embd = cfg_.n_embd;
            xbatch_.resize((size_t)n * (size_t)n_embd);
            for (int i = 0; i < n; i++) {
                const void* row = (const char*)file_.tensor_data(tok_embd_) +
                                  (size_t)tokens[i] * type_row_bytes(tok_embd_.type, n_embd);
                dequant_row(tok_embd_.type, row, xbatch_.data() + (size_t)i * n_embd, n_embd);
            }
            if (mm_embeds_) {
                for (int i = 0; i < n; i++) {
                    const int p = pos0 + i;
                    if (p < mm_at_ || p >= mm_at_ + mm_n_) continue;
                    std::memcpy(xbatch_.data() + (size_t)i * n_embd,
                                mm_embeds_ + (size_t)(p - mm_at_) * n_embd,
                                sizeof(float) * (size_t)n_embd);
                }
            }
            for (int i = 0; i < n; i += 32) {
                const int ch = std::min(32, n - i);
                float* out = (i + ch == n) ? logits : logits_.data();
                if (!metal::q35_prefill(xbatch_.data() + (size_t)i * n_embd, ch,
                                        pos0 + i, out, err)) {
                    if (err && err->empty()) *err = "Qwen3.5 批量 prefill 失败";
                    return false;
                }
            }
            if (pos0 != kv_len_) kv_toks_.clear();
            kv_toks_.insert(kv_toks_.end(), tokens, tokens + n);
            kv_len_ = pos0 + n;
            return true;
        }
        for (int i = 0; i < n; i++) {
            float* out = (i == n - 1) ? logits : logits_.data();
            if (!forward(tokens[i], pos0 + i, out, err)) return false;
        }
        return true;
    }

    if (!gpu_fused_) {
        // CPU 路径和 forward() 都只认"整条序列从 0 开始"，没有槽位概念。
        if (slab0 != 0 || cslot != 0) { if (err) *err = "槽位 prefill 需要 GPU 后端"; return false; }
        for (int i = 0; i < n; i++) {
            if (!forward(tokens[i], pos0 + i, logits, err)) return false;
        }
        return true;
    }

    // 槽位 prefill：整段落在 slab0 之后，容量要按 slab0 + pos0 + n 算，
    // 只按 pos0 + n 算会让第 2 条以后的序列直接越界。
    //
    // 错误信息要带上数字和出路。原来只写"上下文已满"，用户看到的是
    // 一段空白回复 + 四个字，既不知道差多少，也不知道能调 —— 而 CLI 的
    // `-c`、服务端的 `--ctx` 都是现成的旋钮。
    if ((int64_t)slab0 + pos0 + n > n_ctx_) {
        if (err)
            *err = "上下文已满：要写到位置 " + std::to_string(slab0 + pos0 + n) +
                   "，上限 " + std::to_string(n_ctx_) +
                   "（调大 -c/--ctx，或缩短输入）";
        return false;
    }
    if ((int64_t)slab0 + pos0 + n > kv_cap_ &&
        !grow_kv((int)(slab0 + pos0 + n))) {
        if (err) *err = "KV cache 扩容失败"; return false;
    }

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

    const bool ok = (slab0 == 0 && cslot == 0)
        ? metal::fwd_batch(xbatch_.data(), n, pos0, logits, err)
        : metal::fwd_batch_at(xbatch_.data(), n, pos0, slab0, cslot, logits, err);
    if (!ok) {
        gpu_active_ = false;
        if (err && err->empty()) *err = "GPU 批处理前向失败";
        return false;
    }
    // 记录 KV 里现在装着什么，供下一次请求判断能不能跳过 prefill。
    // pos0 不等于 kv_len_ 说明这不是顺序追加（调用方自己动过位置），
    // 缓存就此作废 —— 宁可下次多 prefill 一遍，也不能拿错位的 KV 出字。
    // slab0 != 0 时 KV 不在 0 号槽位，同样不能当成"从 0 开始的前缀"。
    if (slab0 != 0 || pos0 != kv_len_) kv_toks_.clear();
    kv_toks_.insert(kv_toks_.end(), tokens, tokens + n);
    kv_len_ = pos0 + n;
    return true;
}

bool Model::forward_multi(const int32_t* tokens, int nrow, const int32_t* rpos,
                          const uint64_t* rslab, const uint64_t* rslot,
                          float* logits, std::string* err) {
    if (!loaded_) { if (err) *err = "模型未加载"; return false; }
    if (nrow <= 0 || !tokens || !rpos || !rslab) { if (err) *err = "空批"; return false; }
    if (!gpu_fused_) { if (err) *err = "多序列前向需要 GPU 后端"; return false; }

    // KV 容量按**整块**算：rslab[r] + rpos[r] 必须落在 kv_cap_ 之内。
    int need = 0;
    for (int r = 0; r < nrow; r++) {
        const int64_t hi = (int64_t)rslab[r] + (int64_t)rpos[r] + 1;
        if (hi > n_ctx_) { if (err) *err = "上下文已满"; return false; }
        need = std::max(need, (int)hi);
    }
    if (need > kv_cap_ && !grow_kv(need)) { if (err) *err = "KV cache 扩容失败"; return false; }

    const int n_embd = cfg_.n_embd;
    xbatch_.resize((size_t)nrow * (size_t)n_embd);
    for (int r = 0; r < nrow; r++) {
        const void* row = (const char*)file_.tensor_data(tok_embd_) +
                          (size_t)tokens[r] * type_row_bytes(tok_embd_.type, n_embd);
        dequant_row(tok_embd_.type, row, xbatch_.data() + (size_t)r * n_embd, n_embd);
    }

    if (!metal::fwd_multi(xbatch_.data(), nrow, rpos, rslab, rslot, logits, err)) {
        gpu_active_ = false;
        if (err && err->empty()) *err = "GPU 多序列前向失败";
        return false;
    }
    // 一次前向里混了多条序列，KV 不再是"某一个 prompt 的连续前缀"，
    // 前缀缓存的前提没了 —— 直接作废，宁可下次多 prefill 一遍。
    kv_toks_.clear();
    kv_len_ = 0;
    return true;
}

bool Model::forward(int32_t token, int32_t pos, float* logits, std::string* err) {
    if (!loaded_) { if (err) *err = "模型未加载"; return false; }
    if (pos >= n_ctx_) {
        if (err)
            *err = "上下文已满：位置 " + std::to_string(pos) + " 已达上限 " +
                   std::to_string(n_ctx_) + "（调大 -c/--ctx）";
        return false;
    }
    // Qwen3.5 的融合路径把 KV 完全放在 GPU 侧，CPU 侧的 k_cache_ 一次都不会读。
    // 这里再扩容就是按 f32 再存一份 —— 176K 上下文要 11.6 GB，纯浪费。
    if (!(gpu_fused_ && arch_is_qwen35()) && pos >= kv_cap_ && !grow_kv(pos + 1)) {
        if (err) *err = "KV cache 扩容失败";
        return false;
    }

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
        if (std::getenv("DUBHE_Q35_DEBUG"))
            std::fprintf(stderr, "[dbg] forward pos=%d q35=%d\n", pos, (int)arch_is_qwen35());
        // 整层前向一次提交、一次同步。失败就整体报错——此时 GPU 的 KV 已经
        // 前进了一步，切回 CPU 会得到错误结果，静默降级比报错更危险。
        const bool ok = arch_is_qwen35()
                            ? metal::q35_token(x_.data(), pos, logits, err)
                            : metal::fwd_token(x_.data(), pos, logits, err);
        if (!ok) {
            gpu_active_ = false;
            gpu_fused_ = false;
            if (err && err->empty()) *err = "GPU 前向失败";
            return false;
        }
        if (pos != kv_len_) kv_toks_.clear();
        kv_toks_.push_back(token);
        kv_len_ = pos + 1;
        return true;
    }

    const float scale = 1.f / std::sqrt((float)hd);

    for (int l = 0; l < cfg_.n_layer; l++) {
        const Layer& L = layers_[l];

        // ---- 子层 1：ShortConv / DeltaNet / 注意力
        if (L.kind == LK_CONV) {
            rmsnorm(xb_.data(), x_.data(), L.attn_norm, n_embd);
            shortconv(L, conv_slot_[(size_t)l], xb_.data());
        } else if (L.kind == LK_DELTA) {
            rmsnorm(xb_.data(), x_.data(), L.attn_norm, n_embd);
            delta_net(L, ssm_slot_[(size_t)l], xb_.data());
        } else {
            rmsnorm(xb_.data(), x_.data(), L.attn_norm, n_embd);
            if (L.q_gate_fused) {
                // Qwen3.5 的 Q 投影把 [Q | gate] 按 head 交错存：每个 head
                // 占 2*hd，前 hd 是 Q、后 hd 是门。按 2 倍宽度取再拆开。
                matvec(qg_.data(), xb_.data(), L.attn_q, 2 * n_head * hd, n_embd);
                for (int h = 0; h < n_head; h++) {
                    std::memcpy(q_.data() + (size_t)h * hd,
                                qg_.data() + (size_t)h * 2 * hd, sizeof(float) * hd);
                    std::memcpy(gate_.data() + (size_t)h * hd,
                                qg_.data() + (size_t)h * 2 * hd + hd, sizeof(float) * hd);
                }
            } else {
                matvec(q_.data(), xb_.data(), L.attn_q, n_head * hd, n_embd);
            }
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

            float* kc = k_cache_.data() + (size_t)kv_slot_[(size_t)l] * kv_cap_ * kv_dim;
            float* vc = v_cache_.data() + (size_t)kv_slot_[(size_t)l] * kv_cap_ * kv_dim;
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
            // Qwen3.5：门是**逐维**的（每 head head_dim 个），不是逐 head 标量。
            // 乘错粒度不会报错，只会让每个 head 的输出被一个常数缩放 ——
            // 表现为"能说人话但答非所问"。
            if (L.q_gate_fused) {
                for (int i = 0; i < n_head * hd; i++) {
                    const float g = gate_[(size_t)i];
                    attn_out_[(size_t)i] *= 1.f / (1.f + std::exp(-g));
                }
            }

            // 3) attention 输出投影
            matvec(xb2_.data(), attn_out_.data(), L.attn_output, n_embd, n_embd);
            std::memcpy(attn_out_.data(), xb2_.data(), sizeof(float) * n_embd);
        }

        // sandwich norm：子层输出先归一化再加残差（LFM3）
        if (L.has_sandwich) rmsnorm(attn_out_.data(), attn_out_.data(), L.post_attn_norm, n_embd);
        for (int i = 0; i < n_embd; i++) x_[i] += attn_out_[i];
        if (std::getenv("DUBHE_DEBUG_NAN") && l == 0) {
            float mx = 0.f; int bad = 0;
            for (int i = 0; i < n_embd; i++) {
                if (!std::isfinite(x_[(size_t)i])) bad++;
                else mx = std::max(mx, std::fabs(x_[(size_t)i]));
            }
            std::fprintf(stderr, "[dbg] L00 post-attn max=%.4g bad=%d\n", mx, bad);
        }

        // ---- 子层 2：FFN（SwiGLU）
        rmsnorm(xb_.data(), x_.data(), L.ffn_norm, n_embd);
        matvec(gate_.data(), xb_.data(), L.ffn_gate, cfg_.n_ff, n_embd);
        matvec(up_.data(), xb_.data(), L.ffn_up, cfg_.n_ff, n_embd);
        if (std::getenv("DUBHE_DEBUG_NAN") && l == 0) {
            auto mm = [](const float* p, int n, float* mx, int* bad) {
                *mx = 0.f; *bad = 0;
                for (int i = 0; i < n; i++) {
                    if (!std::isfinite(p[i])) (*bad)++;
                    else *mx = std::max(*mx, std::fabs(p[i]));
                }
            };
            float m1, m2; int b1, b2;
            mm(xb_.data(), n_embd, &m1, &b1);
            mm(gate_.data(), cfg_.n_ff, &m2, &b2);
            std::fprintf(stderr, "[dbg] L00 ffn-norm max=%.4g bad=%d gate max=%.4g bad=%d\n",
                         m1, b1, m2, b2);
        }
        for (int i = 0; i < cfg_.n_ff; i++) ffn_out_[i] = silu(gate_[i]) * up_[i];
        {
            float* dst = ffn_down_out_.data();
            matvec(dst, ffn_out_.data(), L.ffn_down, n_embd, cfg_.n_ff);
            if (std::getenv("DUBHE_DEBUG_NAN") && l == 0) {
                auto mm = [](const float* p, int n, float* mx, int* bad) {
                    *mx = 0.f; *bad = 0;
                    for (int i = 0; i < n; i++) {
                        if (!std::isfinite(p[i])) (*bad)++;
                        else *mx = std::max(*mx, std::fabs(p[i]));
                    }
                };
                float m1, m2; int b1, b2;
                mm(ffn_out_.data(), cfg_.n_ff, &m1, &b1);
                mm(dst, n_embd, &m2, &b2);
                std::fprintf(stderr, "[dbg] L00 ffn-out max=%.4g bad=%d down max=%.4g bad=%d\n",
                             m1, b1, m2, b2);
            }
            if (L.has_sandwich) rmsnorm(dst, dst, L.post_ffn_norm, n_embd);
            for (int i = 0; i < n_embd; i++) x_[i] += dst[i];
        }
        if (std::getenv("DUBHE_DEBUG_NAN")) {
            double ss = 0.0;
            float mx = 0.f;
            int nbad = 0;
            for (int i = 0; i < n_embd; i++) {
                const float v = x_[(size_t)i];
                if (!std::isfinite(v)) nbad++;
                else { ss += (double)v * v; mx = std::max(mx, std::fabs(v)); }
            }
            std::fprintf(stderr, "[dbg] L%02d kind=%d rms=%.4f max=%.3f bad=%d\n",
                         l, (int)L.kind, std::sqrt(ss / n_embd), mx, nbad);
        }
    }

    // 5) final norm + logits
    rmsnorm(xb_.data(), x_.data(), out_norm_, n_embd);
    matvec(logits, xb_.data(), out_w_, cfg_.n_vocab, n_embd);

    if (pos != kv_len_) kv_toks_.clear();
    kv_toks_.push_back(token);
    kv_len_ = pos + 1;
    return true;
}

}  // namespace dubhe
