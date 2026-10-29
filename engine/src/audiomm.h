// audiomm.h — LFM3-Audio 语音塔（mmproj 的音频侧）
//
// 和视觉塔（MmProj）是同一个 mmproj 概念、完全不同的结构：
//
//   log-mel(128×T) ──降采样──> T/8 个 768 维 token ──17 层 LFM2──> 投影器 ──> 2048 维
//
// 降采样 = 三层 stride-2 的 3×3 卷积（128 mel → 16，T → T/8），展平成 4096 维
// 再接一个 1×1 卷积到 768。所以一个 token 覆盖 8 帧 = 80 ms 音频。
//
// 17 层是 **LFM2** 骨架（不是 LFM3）：conv/conv/attn 重复 5 次 + 2 层 attn。
// conv 层是门控深度卷积（宽 3，因果）；attn 层是 GQA（8/4 头，head_dim 96）
// + 每 head 的 QK RMSNorm + 滑窗 32 的因果注意力。
//
// 为什么最后才做 embedding_norm：HF 的 Lfm2Model.forward 把 `embedding_norm`
// 放在**所有层之后**（名字骗人），GGUF 里叫 `a.post_ln`。
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "internal.h"
#include "mel.h"
#include "mmproj.h"   // MmTensor

namespace dubhe {

struct AudioMmConfig {
    int   n_mel     = 128;    // clip.audio.num_mel_bins
    int   conv_ch   = 256;    // 降采样通道数
    int   hidden    = 768;    // clip.audio.embedding_length
    int   ffn       = 1792;   // clip.audio.feed_forward_length（不是 config 里的 2560）
    int   n_layer   = 17;
    int   n_head    = 8;
    int   n_head_kv = 4;
    int   head_dim  = 96;
    int   conv_k    = 3;      // conv_L_cache
    int   window    = 32;     // clip.audio.window_size，滑窗注意力
    int   out_dim   = 2048;   // clip.audio.projection_dim
    int   proj_mid  = 2048;
    float eps       = 1e-5f;  // clip.audio.attention.layer_norm_epsilon
    float proj_eps  = 1e-5f;  // 投影器的 nn.LayerNorm 默认 eps
    float rope_theta = 1e6f;
    std::string projector;    // "lfm3a"
    std::vector<uint8_t> is_attn;

    int n_token(int64_t n_frames) const { return (int)((n_frames + 7) / 8); }
};

struct AudioMmLayer {
    MmTensor ln1, ln2;
    MmTensor ffn_gate, ffn_up, ffn_down;
    MmTensor q, k, v, o, q_norm, k_norm;
    MmTensor conv_in, conv_out, conv_dw;   // pw1(768→2304) / pw2(768→768) / dw(768×3)
};

class AudioMm {
public:
    AudioMm() = default;
    AudioMm(const AudioMm&) = delete;
    AudioMm& operator=(const AudioMm&) = delete;

    bool load(const char* path, std::string* err);

    // mel: (n_mel × n_frames) 行主序，来自 log_mel()
    // out: (n_token × out_dim)，可直接注入语言模型
    bool encode(const float* mel, int64_t n_frames, float* out, int* n_token,
                std::string* err) const;

    const AudioMmConfig& config() const { return cfg_; }
    bool loaded() const { return loaded_; }
    void describe(std::string* out) const;

    // 对拍用：每层结束回调 (层号, x, token 数)。-1 = 降采样输出，-2 = 最终投影前。
    std::function<void(int, const float*, int64_t)> on_layer;

private:
    GgufFile f_;
    AudioMmConfig cfg_;
    std::vector<AudioMmLayer> layers_;
    // 降采样
    MmTensor c0_w, c0_b, c2_w, c2_b, c3_w, c3_b, c5_w, c5_b, c6_w, c6_b;
    MmTensor lin_out_w, lin_out_b;
    MmTensor post_ln;
    // 投影器
    MmTensor p_ln_w, p_ln_b, p1_w, p1_b, p2_w, p2_b;
    bool loaded_ = false;
};

}  // namespace dubhe
