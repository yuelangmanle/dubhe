// mel.h — 语音前端的 log-mel 谱（LFM3-Audio 的固定预处理）
//
// 参数来自 preprocessor_config.json，一个都不能改：
//   16 kHz 单声道、25 ms Hann 窗（periodic）每 10 ms 一帧、不做 center 补零、
//   400 点 FFT 取 201 个频点、128 个 Slaney mel 滤波器（Slaney 归一化）、
//   power 谱、log10、下限 -12、再仿射 (x + 4) / 4。
//
// 为什么不自己"近似"：这些常数是训练时定的，错一个（比如 Hann 用对称窗而不是
// 周期窗、mel 用 HTK 刻度而不是 Slaney）不会崩、不会 NaN，只会让语音塔听到
// 一段"稍微不一样的声音"—— 表现是识别率悄悄下降，最难查的那种错。
//
// 为什么不用 FFT 而直接算 DFT：n_fft = 400 = 16×25，不是 2 的幂，写混合基 FFT
// 要做 4×4 和 5×5 两层蝶形，代码量翻几倍；而 201 个频点 × 400 个采样 = 8 万次
// 乘加，8 线程 NEON 下 344 帧只要 2 ms，远不是瓶颈。宁可选笨但绝对不会算错的。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dubhe {

struct MelConfig {
    int   sample_rate = 16000;
    int   n_fft       = 400;
    int   hop         = 160;
    int   win         = 400;
    int   n_mel       = 128;
    float log_floor   = -12.0f;
    float norm_offset = 4.0f;
    float norm_div     = 4.0f;
    int   stride      = 8;      // 每 8 帧谱 = 1 个编码器 token
};

// 采样点数 → 谱帧数（右侧补零到整帧，不丢尾巴上的采样）。
int64_t mel_num_frames(int64_t n_samples, const MelConfig& c);
// 帧数 → 编码器 token 数。
int64_t mel_num_tokens(int64_t n_frames, const MelConfig& c);

// pcm: 16 kHz 单声道 f32（[-1,1]）。
// out: n_mel × n_frames，行主序（mel 在外层），和 HF 的 input_features 一致。
bool log_mel(const float* pcm, int64_t n_samples, const MelConfig& c,
             std::vector<float>* out, int64_t* n_frames, std::string* err);

// 调试/对拍用：把 mel 滤波器组导出来（n_mel × (n_fft/2+1)）。
void mel_filterbank(const MelConfig& c, std::vector<float>* out);

}  // namespace dubhe
