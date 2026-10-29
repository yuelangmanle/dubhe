// audiomm.cpp — LFM3-Audio 语音塔前向（CPU）
//
// 三个部分，都按 HF 的 Lfm3AudioEncoder 逐行对齐：
//   1. Lfm3AudioDownsampler：三层 stride-2 的 3×3 卷积把 (128 mel, T 帧) 压成
//      (16, T/8) 的 256 通道特征图，展平成 4096 维，1×1 卷积到 768。
//   2. Lfm2Model 17 层：conv / conv / attn 重复 5 次 + 2 层 attn。
//   3. Lfm3AudioProjector：LayerNorm → Linear(768→2048) → GELU → Linear(2048→2048)。
//
// 数据布局统一用 **[位置][通道]**（token-major）：降采样的 1×1 卷积正好能直接
// 调 matmul_qt（它要的输入就是每行连续），深度卷积的每个 tap 也正好是连续的一段
// 通道。这样整条链路不需要任何转置 —— 转置在大张量上是最容易写错又最难查的一类
// 代码（视觉塔那次 `reshape(-1, ne[0])` 就吃了这个亏）。
//
// 唯一一次重排是降采样最后一层的 flatten：HF 的 flatten(1,2) 是 [通道][频率]
// 序，而我的布局是 [频率] 在外，所以按 c*16+f 重新排一遍（43 个 token，几十微秒）。
#include "audiomm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "mm_math.h"

namespace dubhe {
using namespace mm;

namespace {

// 单通道 3×3 stride-2 卷积（padding 1）。输入 [IH][IW]，输出 [OH*OW][OC]。
// 权重按 GGUF 的 [oc][ky*3+kx] 排列（in_ch = 1）。
void conv2d_in(const float* in, int IH, int IW, const float* w, const float* b, int OC,
               float* out, int* OH, int* OW, bool act) {
    const int oh = (IH - 1) / 2 + 1, ow = (IW - 1) / 2 + 1;
    *OH = oh; *OW = ow;
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
            float* dst = out + (size_t)(oy * ow + ox) * OC;
            for (int co = 0; co < OC; co++) dst[co] = b ? b[co] : 0.0f;
            for (int ky = 0; ky < 3; ky++) {
                const int iy = oy * 2 + ky - 1;
                if (iy < 0 || iy >= IH) continue;
                for (int kx = 0; kx < 3; kx++) {
                    const int ix = ox * 2 + kx - 1;
                    if (ix < 0 || ix >= IW) continue;
                    const float xv = in[(size_t)iy * IW + ix];
                    const int tap = ky * 3 + kx;
                    for (int co = 0; co < OC; co++) dst[co] += w[(size_t)co * 9 + tap] * xv;
                }
            }
            if (act)
                for (int co = 0; co < OC; co++) dst[co] = silu(dst[co]);
        }
}

// 深度 3×3 stride-2 卷积（groups = C）。输入输出都是 [位置][C]。
void conv2d_dw(const float* in, int IH, int IW, int C, const float* w, const float* b,
               float* out, int* OH, int* OW, bool act) {
    const int oh = (IH - 1) / 2 + 1, ow = (IW - 1) / 2 + 1;
    *OH = oh; *OW = ow;
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
            float* dst = out + (size_t)(oy * ow + ox) * C;
            if (b) std::memcpy(dst, b, sizeof(float) * (size_t)C);
            else   std::memset(dst, 0, sizeof(float) * (size_t)C);
            for (int ky = 0; ky < 3; ky++) {
                const int iy = oy * 2 + ky - 1;
                if (iy < 0 || iy >= IH) continue;
                for (int kx = 0; kx < 3; kx++) {
                    const int ix = ox * 2 + kx - 1;
                    if (ix < 0 || ix >= IW) continue;
                    const float* src = in + (size_t)(iy * IW + ix) * C;
                    const int tap = ky * 3 + kx;
                    for (int ch = 0; ch < C; ch++) dst[ch] += w[(size_t)ch * 9 + tap] * src[ch];
                }
            }
            if (act)
                for (int ch = 0; ch < C; ch++) dst[ch] = silu(dst[ch]);
        }
}

}  // namespace

// ---------------------------------------------------------------- 加载

bool AudioMm::load(const char* path, std::string* err) {
    if (!f_.open(path, err)) return false;
    auto& c = cfg_;

    if (f_.get_str("general.type") != "mmproj") {
        if (err) *err = "不是 mmproj 文件（general.type != mmproj）";
        return false;
    }
    if (f_.get_u32("clip.has_audio_encoder", 0) == 0) {
        if (err) *err = "这个 mmproj 没有语音编码器（clip.has_audio_encoder 缺失）";
        return false;
    }
    c.projector  = f_.get_str("clip.audio.projector_type");
    c.n_mel      = (int)f_.get_u32("clip.audio.num_mel_bins", 128);
    c.hidden     = (int)f_.get_u32("clip.audio.embedding_length");
    c.ffn        = (int)f_.get_u32("clip.audio.feed_forward_length");
    c.n_layer    = (int)f_.get_u32("clip.audio.block_count");
    c.n_head     = (int)f_.get_u32("clip.audio.attention.head_count");
    c.n_head_kv  = (int)f_.get_u32("clip.audio.attention.head_count_kv");
    c.eps        = f_.get_f32("clip.audio.attention.layer_norm_epsilon", 1e-5f);
    c.out_dim    = (int)f_.get_u32("clip.audio.projection_dim", 2048);
    c.window     = (int)f_.get_u32("clip.audio.window_size", 32);
    if (c.hidden <= 0 || c.n_layer <= 0 || c.n_head <= 0) {
        if (err) *err = "clip.audio.* 元数据不完整";
        return false;
    }
    c.head_dim = c.hidden / c.n_head;
    if (c.n_head_kv <= 0) c.n_head_kv = c.n_head;
    c.conv_ch = 256;
    c.conv_k = 3;

    auto bind = [&](const std::string& n) {
        MmTensor t;
        if (const TensorInfo* ti = f_.find_tensor(n)) {
            t.data = f_.tensor_data(*ti);
            t.type = ti->type;
            t.ne0  = ti->ne[0];
            int64_t tot = 1;
            for (int i = 0; i < ti->n_dims; i++) tot *= ti->ne[i];
            t.n = tot;
        }
        return t;
    };

    // 降采样：conv1d.0/2/3/5/6（名字叫 conv1d，其实是把谱图当单通道图像的 Conv2d）
    c0_w = bind("a.conv1d.0.weight"); c0_b = bind("a.conv1d.0.bias");
    c2_w = bind("a.conv1d.2.weight"); c2_b = bind("a.conv1d.2.bias");
    c3_w = bind("a.conv1d.3.weight"); c3_b = bind("a.conv1d.3.bias");
    c5_w = bind("a.conv1d.5.weight"); c5_b = bind("a.conv1d.5.bias");
    c6_w = bind("a.conv1d.6.weight"); c6_b = bind("a.conv1d.6.bias");
    lin_out_w = bind("a.pre_encode.out.weight");
    lin_out_b = bind("a.pre_encode.out.bias");
    if (!c0_w.ok() || !lin_out_w.ok()) {
        if (err) *err = "语音降采样的权重缺失";
        return false;
    }
    post_ln = bind("a.post_ln.weight");

    p_ln_w = bind("mm.a.mlp.0.weight"); p_ln_b = bind("mm.a.mlp.0.bias");
    p1_w   = bind("mm.a.mlp.1.weight"); p1_b   = bind("mm.a.mlp.1.bias");
    p2_w   = bind("mm.a.mlp.3.weight"); p2_b   = bind("mm.a.mlp.3.bias");
    // ne0 是**输入**维（GGUF 的 Linear 存成 [in, out]），输出维要除一下
    if (p1_w.ok() && p1_w.ne0 > 0) c.proj_mid = (int)(p1_w.n / p1_w.ne0);
    if (p2_w.ok() && p2_w.ne0 > 0) c.out_dim  = (int)(p2_w.n / p2_w.ne0);

    layers_.assign((size_t)c.n_layer, AudioMmLayer{});
    c.is_attn.assign((size_t)c.n_layer, 0);
    for (int i = 0; i < c.n_layer; i++) {
        const std::string p = "a.blk." + std::to_string(i) + ".";
        auto& L = layers_[(size_t)i];
        L.ln1 = bind(p + "ln1.weight");
        L.ln2 = bind(p + "ln2.weight");
        L.ffn_gate = bind(p + "ffn_gate.weight");
        L.ffn_up   = bind(p + "ffn_up.weight");
        L.ffn_down = bind(p + "ffn_down.weight");
        L.q = bind(p + "attn_q.weight");
        L.k = bind(p + "attn_k.weight");
        L.v = bind(p + "attn_v.weight");
        L.o = bind(p + "attn_out.weight");
        L.q_norm = bind(p + "attn_q_norm.weight");
        L.k_norm = bind(p + "attn_k_norm.weight");
        L.conv_in  = bind(p + "conv_pw1.weight");
        L.conv_out = bind(p + "conv_pw2.weight");
        L.conv_dw  = bind(p + "conv_dw.weight");
        // 层类型由实际张量决定，不靠外部配置猜
        if (L.q.ok()) c.is_attn[(size_t)i] = 1;
        else if (L.conv_in.ok()) c.is_attn[(size_t)i] = 0;
        else { if (err) *err = "第 " + std::to_string(i) + " 层既无 attn_q 也无 conv_pw1"; return false; }
    }

    loaded_ = true;
    return true;
}

void AudioMm::describe(std::string* out) const {
    const auto& c = cfg_;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "audio mmproj: %s  mel %d  降采样 3×3/s2 ×3（128→16, T→T/8）\n"
                  "  %d 层  hidden %d  ffn %d  head %d/%d  head_dim %d  滑窗 %d\n"
                  "  输出 %d 维，投影中间层 %d，eps %.0e\n",
                  c.projector.c_str(), c.n_mel, c.n_layer, c.hidden, c.ffn, c.n_head,
                  c.n_head_kv, c.head_dim, c.window, c.out_dim, c.proj_mid, (double)c.eps);
    out->append(buf);
    out->append("  层类型: ");
    for (int i = 0; i < c.n_layer; i++) out->append(c.is_attn[(size_t)i] ? "A" : "C");
    out->append("\n");
}

// ---------------------------------------------------------------- 前向

bool AudioMm::encode(const float* mel, int64_t n_frames, float* out, int* n_token,
                     std::string* err) const {
    if (!loaded_) { if (err) *err = "语音塔未加载"; return false; }
    const auto& c = cfg_;
    const int D = c.hidden, C = c.conv_ch, FF = c.ffn;
    const int HD = c.head_dim, H = c.n_head, KV = c.n_head_kv;
    const int qdim = H * HD, kdim = KV * HD;
    const int64_t F0 = c.n_mel;

    // ---- 1. 降采样
    std::vector<float> w0, b0, w2, b2, w3, b3, w5, b5, w6, b6, wlo, blo;
    if (!deq_all(c0_w, w0, err) || !deq_all(c2_w, w2, err) || !deq_all(c3_w, w3, err) ||
        !deq_all(c5_w, w5, err) || !deq_all(c6_w, w6, err) || !deq_all(lin_out_w, wlo, err))
        return false;
    auto opt = [](const MmTensor& t, std::vector<float>& v) {
        if (t.ok()) deq_all(t, v, nullptr);
    };
    opt(c0_b, b0); opt(c2_b, b2); opt(c3_b, b3); opt(c5_b, b5); opt(c6_b, b6); opt(lin_out_b, blo);

    int H1 = 0, W1 = 0, H2 = 0, W2 = 0, H3 = 0, W3 = 0;
    std::vector<float> h0((size_t)((F0 + 1) / 2) * (size_t)((n_frames + 1) / 2) * (size_t)C);
    conv2d_in(mel, (int)F0, (int)n_frames, w0.data(), b0.empty() ? nullptr : b0.data(), C,
              h0.data(), &H1, &W1, true);

    std::vector<float> t1((size_t)H1 * W1 * C), h1((size_t)H2 * W2 * C);
    conv2d_dw(h0.data(), H1, W1, C, w2.data(), b2.empty() ? nullptr : b2.data(),
              t1.data(), &H2, &W2, false);
    h1.resize((size_t)H2 * W2 * C);
    matmul_qt(c3_w, C, C, t1.data(), (int64_t)H2 * W2, h1.data());
    if (!b3.empty())
        for (int64_t p = 0; p < (int64_t)H2 * W2; p++)
            for (int ch = 0; ch < C; ch++) h1[(size_t)p * C + ch] = silu(h1[(size_t)p * C + ch] + b3[(size_t)ch]);
    else
        for (int64_t i = 0; i < (int64_t)H2 * W2 * C; i++) h1[(size_t)i] = silu(h1[(size_t)i]);

    std::vector<float> t2((size_t)H2 * W2 * C), h2((size_t)H3 * W3 * C);
    conv2d_dw(h1.data(), H2, W2, C, w5.data(), b5.empty() ? nullptr : b5.data(),
              t2.data(), &H3, &W3, false);
    h2.resize((size_t)H3 * W3 * C);
    matmul_qt(c6_w, C, C, t2.data(), (int64_t)H3 * W3, h2.data());
    if (!b6.empty())
        for (int64_t p = 0; p < (int64_t)H3 * W3; p++)
            for (int ch = 0; ch < C; ch++) h2[(size_t)p * C + ch] = silu(h2[(size_t)p * C + ch] + b6[(size_t)ch]);
    else
        for (int64_t i = 0; i < (int64_t)H3 * W3 * C; i++) h2[(size_t)i] = silu(h2[(size_t)i]);

    // flatten(1,2)：HF 的行序是 [通道][频率]（c*F + f），我的布局是 [频率] 在外。
    const int64_t T = W3;
    const int64_t din = (int64_t)C * H3;      // 4096
    std::vector<float> flat((size_t)(T * din));
    for (int64_t ox = 0; ox < T; ox++)
        for (int64_t f = 0; f < H3; f++) {
            const float* src = h2.data() + (size_t)(f * T + ox) * C;
            float* dst = flat.data() + (size_t)ox * din;
            for (int ch = 0; ch < C; ch++) dst[(size_t)ch * H3 + f] = src[ch];
        }

    std::vector<float> x((size_t)(T * D));
    matmul_qt(lin_out_w, din, D, flat.data(), T, x.data());
    if (!blo.empty())
        for (int64_t t = 0; t < T; t++)
            for (int d = 0; d < D; d++) x[(size_t)(t * D + d)] += blo[(size_t)d];

    if (on_layer) on_layer(-1, x.data(), T);

    // ---- 2. 17 层 LFM2
    const int half = HD / 2;
    std::vector<float> invf((size_t)half), cosb((size_t)(T * half)), sinb((size_t)(T * half));
    for (int i = 0; i < half; i++)
        invf[(size_t)i] = 1.0f / std::pow(c.rope_theta, (float)(2 * i) / (float)HD);
    for (int64_t t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            const float a = (float)t * invf[(size_t)i];
            cosb[(size_t)(t * half + i)] = std::cos(a);
            sinb[(size_t)(t * half + i)] = std::sin(a);
        }

    std::vector<float> xn((size_t)(T * D)), y((size_t)(T * D));
    std::vector<float> wbuf;
    std::vector<float> q((size_t)(T * qdim)), k((size_t)(T * kdim)), v((size_t)(T * kdim));

    for (int li = 0; li < c.n_layer; li++) {
        const auto& L = layers_[(size_t)li];

        xn.assign(x.begin(), x.end());
        if (!deq_all(L.ln1, wbuf, err)) return false;
        for (int64_t t = 0; t < T; t++) rmsnorm(xn.data() + t * D, D, wbuf.data(), c.eps);

        if (c.is_attn[(size_t)li]) {
            std::vector<float> wqn, wkn;
            if (!deq_all(L.q_norm, wqn, err) || !deq_all(L.k_norm, wkn, err)) return false;
            matmul_qt(L.q, D, qdim, xn.data(), T, q.data());
            matmul_qt(L.k, D, kdim, xn.data(), T, k.data());
            matmul_qt(L.v, D, kdim, xn.data(), T, v.data());

            // 每 head 的 QK RMSNorm（head_dim 宽），再上 RoPE（rotate-half）
            for (int64_t t = 0; t < T; t++) {
                const float* cb = cosb.data() + t * half;
                const float* sb = sinb.data() + t * half;
                for (int hh = 0; hh < H; hh++) {
                    float* qh = q.data() + t * qdim + hh * HD;
                    rmsnorm(qh, HD, wqn.data(), c.eps);
                    for (int d = 0; d < half; d++) {
                        const float a = qh[d], b = qh[d + half];
                        qh[d]        = a * cb[d] - b * sb[d];
                        qh[d + half] = b * cb[d] + a * sb[d];
                    }
                }
                for (int hh = 0; hh < KV; hh++) {
                    float* kh = k.data() + t * kdim + hh * HD;
                    rmsnorm(kh, HD, wkn.data(), c.eps);
                    for (int d = 0; d < half; d++) {
                        const float a = kh[d], b = kh[d + half];
                        kh[d]        = a * cb[d] - b * sb[d];
                        kh[d + half] = b * cb[d] + a * sb[d];
                    }
                }
            }

            // 因果 + 滑窗（window = 32）：query i 只看 [i-window+1, i]
            const float scale = 1.0f / std::sqrt((float)HD);
            const int group = H / KV;
            const int win = c.window > 0 ? c.window : (int)T;
            std::vector<float> o((size_t)(T * qdim));
            auto head_body = [&](int64_t h0, int64_t h1) {
                std::vector<float> sc;
                for (int64_t hh = h0; hh < h1; hh++) {
                    const int kvh = (int)(hh / group);
                    for (int64_t i = 0; i < T; i++) {
                        const int64_t jlo = i - win + 1 > 0 ? i - win + 1 : 0;
                        sc.resize((size_t)(i - jlo + 1));
                        const float* qi = q.data() + i * qdim + hh * HD;
                        float mx = -1e30f;
                        for (int64_t j = jlo; j <= i; j++) {
                            const float s = dot_f32(qi, k.data() + j * kdim + kvh * HD, HD) * scale;
                            sc[(size_t)(j - jlo)] = s;
                            if (s > mx) mx = s;
                        }
                        float sum = 0.0f;
                        for (auto& sv : sc) { sv = std::exp(sv - mx); sum += sv; }
                        const float inv = 1.0f / sum;
                        float* oi = o.data() + i * qdim + hh * HD;
                        for (int d = 0; d < HD; d++) oi[d] = 0.0f;
                        for (int64_t j = jlo; j <= i; j++)
                            saxpy(oi, v.data() + j * kdim + kvh * HD, sc[(size_t)(j - jlo)] * inv, HD);
                    }
                }
            };
            parallel_rows(H, head_body);
            matmul_qt(L.o, qdim, D, o.data(), T, y.data());
        } else {
            // 门控因果深度卷积（宽 3）：in_proj 切三段 (B, C, x)
            //   h = B * x → 因果深度卷积 → y = C * h → out_proj
            // 中间那段 C 是**输出门**，漏掉它不会崩，只是质量悄悄变差。
            std::vector<float> wc;
            if (!deq_all(L.conv_dw, wc, err)) return false;
            const int64_t c3 = 3 * D;
            std::vector<float> g3((size_t)(T * c3));
            matmul_qt(L.conv_in, D, c3, xn.data(), T, g3.data());

            // 因果卷积要在时间轴上是连续的，所以摊成 [通道][时间] 更好走
            std::vector<float> hc((size_t)(D * T)), yc((size_t)(D * T));
            for (int64_t t = 0; t < T; t++)
                for (int ch = 0; ch < D; ch++)
                    hc[(size_t)(ch * T + t)] = g3[(size_t)(t * c3 + ch)] *
                                               g3[(size_t)(t * c3 + 2 * D + ch)];
            const int K = c.conv_k;   // 3
            for (int ch = 0; ch < D; ch++) {
                const float* kern = wc.data() + (size_t)ch * K;
                const float* src = hc.data() + (size_t)ch * T;
                float* dst = yc.data() + (size_t)ch * T;
                for (int64_t t = 0; t < T; t++) {
                    float s = 0.0f;
                    for (int kk = 0; kk < K; kk++) {
                        const int64_t ti = t - (K - 1) + kk;   // 因果：只看 t-K+1..t
                        if (ti < 0) continue;
                        s += kern[kk] * src[ti];
                    }
                    dst[t] = s;
                }
            }
            std::vector<float> back((size_t)(T * D));
            for (int64_t t = 0; t < T; t++)
                for (int ch = 0; ch < D; ch++)
                    back[(size_t)(t * D + ch)] = yc[(size_t)(ch * T + t)] *
                                                 g3[(size_t)(t * c3 + D + ch)];
            matmul_qt(L.conv_out, D, D, back.data(), T, y.data());
        }

        for (int64_t i = 0; i < T * D; i++) x[(size_t)i] += y[(size_t)i];

        xn.assign(x.begin(), x.end());
        if (!deq_all(L.ln2, wbuf, err)) return false;
        for (int64_t t = 0; t < T; t++) rmsnorm(xn.data() + t * D, D, wbuf.data(), c.eps);

        std::vector<float> fg((size_t)(T * FF)), fu((size_t)(T * FF));
        matmul_qt(L.ffn_gate, D, FF, xn.data(), T, fg.data());
        matmul_qt(L.ffn_up, D, FF, xn.data(), T, fu.data());
        for (int64_t i = 0; i < T * FF; i++) fg[(size_t)i] = silu(fg[(size_t)i]) * fu[(size_t)i];
        matmul_qt(L.ffn_down, FF, D, fg.data(), T, y.data());
        for (int64_t i = 0; i < T * D; i++) x[(size_t)i] += y[(size_t)i];

        if (on_layer) on_layer(li, x.data(), T);
    }

    // ---- 3. embedding_norm（GGUF 的 a.post_ln）：名字带 post，确实在最后
    {
        std::vector<float> wpl;
        if (!deq_all(post_ln, wpl, err)) return false;
        for (int64_t t = 0; t < T; t++) rmsnorm(x.data() + t * D, D, wpl.data(), c.eps);
    }
    if (on_layer) on_layer(-2, x.data(), T);

    // ---- 4. 投影器：LayerNorm → Linear → GELU → Linear
    {
        std::vector<float> lnw, lnb, b1, b2v;
        if (!deq_all(p_ln_w, lnw, err) || !deq_all(p_ln_b, lnb, err) ||
            !deq_all(p1_b, b1, err) || !deq_all(p2_b, b2v, err)) return false;
        for (int64_t t = 0; t < T; t++) layernorm(x.data() + t * D, D, lnw.data(), lnb.data(), c.proj_eps);

        std::vector<float> mid((size_t)(T * c.proj_mid));
        matmul_qt(p1_w, D, c.proj_mid, x.data(), T, mid.data());
        for (int64_t t = 0; t < T; t++)
            for (int o = 0; o < c.proj_mid; o++) {
                float& m = mid[(size_t)(t * c.proj_mid + o)];
                m = gelu(m + b1[(size_t)o]);
            }
        matmul_qt(p2_w, c.proj_mid, c.out_dim, mid.data(), T, out);
        for (int64_t t = 0; t < T; t++)
            for (int o = 0; o < c.out_dim; o++) out[t * c.out_dim + o] += b2v[(size_t)o];
    }

    if (n_token) *n_token = (int)T;
    return true;
}

}  // namespace dubhe
