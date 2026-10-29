// mmproj.h — 多模态投影器（llama.cpp 的 mmproj / clip 格式）
//
// 为什么单独一个文件：mmproj 是一个**独立的 GGUF**，跟语言模型的权重完全分开。
// 这样同一个视觉塔能配不同的语言模型量化档，也让"只跑文本"的场景不必为
// 视觉权重付内存。llama.cpp 用 `--mmproj` 传，我们跟它对齐。
//
// 当前实现的是 LFM3-VL 的视觉塔（projector_type = "lfm3vl"）。它的结构和
// 语言模型同源 —— 也是 conv / attention 交替的混合体，所以 ShortConv 和
// attention 的调优经验可以直接复用。这不是巧合：Liquid 全线模型都是这个骨架。
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "internal.h"

namespace dubhe {

// 一个权重张量的只读视图。不拥有内存（指向 mmap），所以 MmProj 析构前必须用完。
struct MmTensor {
    const void* data = nullptr;
    uint32_t    type = 0;
    int64_t     ne0 = 0;   // 最内层维度（GGUF 的行宽）
    int64_t     n   = 0;   // 元素总数
    bool ok() const { return data != nullptr; }
};

struct MmProjConfig {
    // 视觉侧
    int   image_size  = 0;   // clip.vision.image_size
    int   patch       = 0;   // clip.vision.patch_size
    int   temporal    = 2;   // temporal_patch_size（静止图像靠复制帧填满）
    int   merge       = 2;   // spatial_merge_size
    int   n_layer     = 0;
    int   hidden      = 0;
    int   ffn         = 0;
    int   n_head      = 0;
    int   n_head_kv   = 0;
    int   head_dim    = 0;
    // 投影器
    int   out_dim     = 0;   // 投影后的宽度 = 文本 hidden
    int   proj_hidden = 0;   // 中间层宽度（mm.0 的输出）
    // 数值
    float norm_eps    = 1e-5f;
    float rope_theta  = 1e6f;
    float mean[3]     = {0.5f, 0.5f, 0.5f};
    float stdv[3]     = {0.5f, 0.5f, 0.5f};
    int   min_pixels  = 65536;        // clip.vision.image_min_pixels
    int   max_pixels  = 16777216;     // clip.vision.image_max_pixels
    std::string projector;               // "lfm3vl"
    std::vector<uint8_t> is_attn;        // 每层：1 = attention，0 = gated conv

    // 一张图切成多少个 patch / 多少个语言 token。
    // 注意网格是**变长**的：参考实现的 smart_resize 保持长宽比、把两边都对齐到
    // patch×merge = 32 的倍数，所以 512×512 只是最坏情况的默认值，不是唯一尺寸。
    int n_patch(int gh, int gw) const { return gh * gw; }
    int n_token(int gh, int gw) const { return merge > 0 ? gh * gw / (merge * merge) : 0; }
    // 每个 patch 向量的长度 = 通道 × 时间片 × patch × patch
    int patch_dim() const { return 3 * temporal * patch * patch; }
};

struct MmProjLayer {
    MmTensor ln1, ln2;                       // RMSNorm
    MmTensor ffn_gate, ffn_up, ffn_down;     // w1 / w3 / w2
    // attention 层
    MmTensor q, k, v, o;
    MmTensor q_norm, k_norm;                 // 每 head 的 RMSNorm（head_dim 宽）
    // conv 层
    MmTensor conv_in, conv_out;              // 1024→3072 / 1024→1024
    MmTensor conv_spatial;                   // 深度卷积核 [3,3,hidden]
};

class MmProj {
public:
    MmProj() = default;
    ~MmProj() = default;
    MmProj(const MmProj&) = delete;
    MmProj& operator=(const MmProj&) = delete;

    bool load(const char* path, std::string* err);

    const MmProjConfig& config() const { return cfg_; }
    const MmProjLayer&  layer(int i) const { return layers_[(size_t)i]; }
    bool loaded() const { return loaded_; }

    // 视觉前向（单帧）。
    //   pixels: (gh·gw × patch_dim)，已归一化、已按 merge-block 顺序排好
    //   out:    (n_token × out_dim)，可直接当嵌入喂给语言模型
    //   gh/gw:  patch 网格的行列数，必须是 merge 的倍数
    bool encode(const float* pixels, int gh, int gw, float* out, std::string* err) const;

    // 预处理：把一张 W×H 的 uint8 RGB（HWC）归一化并按 merge-block 顺序摊成
    // patch 序列。W/H 必须是 patch 的整数倍，且是 merge 的倍数倍（调用方先
    // 用 smart_resize 算好目标尺寸）。
    bool pack_pixels(const uint8_t* rgb_hwc, int w, int h, float* out) const;

    // 视频版：一个 patch 的 temporal_patch_size 个时间片来自**相邻的不同帧**。
    // 静止图像就是 rgb0 == rgb1。
    bool pack_pixels2(const uint8_t* rgb0, const uint8_t* rgb1, int w, int h,
                      float* out) const;

    // 参考实现 smart_resize：保持长宽比，两边对齐到 patch×merge 的倍数，
    // 总像素落在 [min_px, max_px]。和 HF 的 Lfm3VlImageProcessor 一致。
    void smart_resize(int h, int w, int* oh, int* ow) const;

    // 调试用：打印结构摘要
    void describe(std::string* out) const;

    // 调试用：每层结束回调 (层号, x 指针, token 数)。对拍时用来定位分歧点，
    // 只在探针里挂，正常推理保持空。
    std::function<void(int, const float*, int64_t)> on_layer;

private:
    bool encode_frame(const float* pixels, float* out, std::string* err) const;

    GgufFile f_;
    MmProjConfig cfg_;
    std::vector<MmProjLayer> layers_;
    MmTensor patch_w[2];      // 按时间片拆成两块（llama.cpp 的约定）
    bool patch_two_slices_ = false;   // 真的是两块，还是只有一块被复制成两份
    MmTensor emb_norm;        // patchify 之后的 RMSNorm
    MmTensor mm_norm_w, mm_norm_b;   // 投影器的 LayerNorm
    MmTensor mm0_w, mm0_b, mm2_w, mm2_b;
    bool loaded_ = false;
};

}  // namespace dubhe
