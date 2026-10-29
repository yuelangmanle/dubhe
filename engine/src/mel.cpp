// mel.cpp — log-mel 谱的 C++ 实现
//
// 逐行照抄 transformers 的 Lfm3AudioFeatureExtractor + audio_utils.mel_filter_bank
// 的 Slaney 分支。注释里标出每个常数对应哪一行源码，方便以后有人问"为什么是
// 27/log(6.4)"时能直接找到出处。
#include "mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace dubhe {
namespace {

constexpr int kThreads = 8;

// Slaney mel 刻度。1000 Hz 以下线性（3*f/200 = f/66.67），以上对数。
// logstep 在 hertz_to_mel 里是 27/log(6.4)，在 mel_to_hertz 里是 log(6.4)/27。
inline double hz_to_mel(double f) {
    if (f < 1000.0) return 3.0 * f / 200.0;
    return 15.0 + std::log(f / 1000.0) * (27.0 / std::log(6.4));
}

inline double mel_to_hz(double m) {
    if (m < 15.0) return 200.0 * m / 3.0;
    return 1000.0 * std::exp((std::log(6.4) / 27.0) * (m - 15.0));
}

}  // namespace

int64_t mel_num_frames(int64_t n_samples, const MelConfig& c) {
    // pad_length: 不足一窗就补到一窗，否则右侧补零到 (n - n_fft) 能被 hop 整除
    int64_t padded = n_samples < c.n_fft
                         ? c.n_fft
                         : n_samples + ((-(n_samples - c.n_fft)) % c.hop + c.hop) % c.hop;
    return 1 + (padded - c.n_fft) / c.hop;
}

int64_t mel_num_tokens(int64_t n_frames, const MelConfig& c) {
    return (n_frames + c.stride - 1) / c.stride;
}

void mel_filterbank(const MelConfig& c, std::vector<float>* out) {
    const int nbins = c.n_fft / 2 + 1;                 // 201
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(c.sample_rate / 2.0);
    std::vector<double> ff((size_t)(c.n_mel + 2));
    for (int i = 0; i < c.n_mel + 2; i++) {
        const double m = mel_min + (mel_max - mel_min) * i / (double)(c.n_mel + 1);
        ff[(size_t)i] = mel_to_hz(m);
    }
    // fft_freqs = linspace(0, sr//2, nbins) —— 步长恰好是 sr/n_fft
    out->assign((size_t)(c.n_mel) * (size_t)nbins, 0.0f);
    for (int m = 0; m < c.n_mel; m++) {
        const double lo = ff[(size_t)m], ce = ff[(size_t)(m + 1)], hi = ff[(size_t)(m + 2)];
        // Slaney 面积归一化：enorm = 2 / (f[i+2] - f[i])
        const double enorm = 2.0 / (hi - lo);
        for (int k = 0; k < nbins; k++) {
            const double f = (double)k * c.sample_rate / c.n_fft;
            const double down = (f - lo) / (ce - lo);
            const double up   = (hi - f) / (hi - ce);
            const double w    = std::max(0.0, std::min(down, up));
            (*out)[(size_t)m * nbins + k] = (float)(w * enorm);
        }
    }
}

bool log_mel(const float* pcm, int64_t n_samples, const MelConfig& c,
             std::vector<float>* out, int64_t* n_frames, std::string* err) {
    if (!pcm || n_samples < 0) { if (err) *err = "空音频"; return false; }
    const int nbins = c.n_fft / 2 + 1;
    const int64_t nf = mel_num_frames(n_samples, c);
    if (nf <= 0) { if (err) *err = "音频太短"; return false; }

    // ---- 窗 + DFT 核：把 Hann 窗折进 cos/sin 表，省掉每帧一次乘法
    // periodic Hann（torch.hann_window 默认 periodic=True，分母是 win 不是 win-1）
    std::vector<float> w((size_t)c.win);
    for (int i = 0; i < c.win; i++)
        w[(size_t)i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (float)c.win);

    const size_t tbl = (size_t)nbins * (size_t)c.n_fft;
    std::vector<float> wc(tbl), ws(tbl);
    for (int k = 0; k < nbins; k++)
        for (int n = 0; n < c.n_fft; n++) {
            const double ang = 2.0 * M_PI * k * n / c.n_fft;
            wc[(size_t)k * c.n_fft + n] = (float)(std::cos(ang) * w[(size_t)n]);
            ws[(size_t)k * c.n_fft + n] = (float)(std::sin(ang) * w[(size_t)n]);
        }

    std::vector<float> fb;
    mel_filterbank(c, &fb);

    out->assign((size_t)c.n_mel * (size_t)nf, 0.0f);
    *n_frames = nf;

    // 逐帧：DFT → power → mel → log10 → clamp → 仿射
    auto body = [&](int64_t f0, int64_t f1) {
        std::vector<float> power((size_t)nbins);
        std::vector<float> frame((size_t)c.n_fft);
        for (int64_t t = f0; t < f1; t++) {
            const int64_t base = t * c.hop;
            // 取一窗，右侧越界补零（等于 pad_length 的补零）
            for (int n = 0; n < c.n_fft; n++) {
                const int64_t s = base + n;
                frame[(size_t)n] = s < n_samples ? pcm[s] : 0.0f;
            }
            for (int k = 0; k < nbins; k++) {
                const float* cp = wc.data() + (size_t)k * c.n_fft;
                const float* sp = ws.data() + (size_t)k * c.n_fft;
                float re = 0.0f, im = 0.0f;
#if defined(__ARM_NEON)
                float32x4_t ar = vdupq_n_f32(0.0f), ai = ar;
                float32x4_t br = ar, bi = ar;
                int n = 0;
                for (; n + 8 <= c.n_fft; n += 8) {
                    const float32x4_t x0 = vld1q_f32(frame.data() + n);
                    const float32x4_t x1 = vld1q_f32(frame.data() + n + 4);
                    ar = vfmaq_f32(ar, x0, vld1q_f32(cp + n));
                    ai = vfmaq_f32(ai, x0, vld1q_f32(sp + n));
                    br = vfmaq_f32(br, x1, vld1q_f32(cp + n + 4));
                    bi = vfmaq_f32(bi, x1, vld1q_f32(sp + n + 4));
                }
                re = vaddvq_f32(vaddq_f32(ar, br));
                im = vaddvq_f32(vaddq_f32(ai, bi));
                for (; n < c.n_fft; n++) {
                    re += frame[(size_t)n] * cp[n];
                    im += frame[(size_t)n] * sp[n];
                }
#else
                for (int n = 0; n < c.n_fft; n++) {
                    re += frame[(size_t)n] * cp[n];
                    im += frame[(size_t)n] * sp[n];
                }
#endif
                power[(size_t)k] = re * re + im * im;
            }
            for (int m = 0; m < c.n_mel; m++) {
                const float* fp = fb.data() + (size_t)m * nbins;
                float s = 0.0f;
                for (int k = 0; k < nbins; k++) s += fp[k] * power[(size_t)k];
                // .log10().clamp(min=log_floor)，再 (x + offset) / divisor
                float v = s > 0.0f ? std::log10(s) : -1e30f;
                if (v < c.log_floor) v = c.log_floor;
                (*out)[(size_t)m * nf + t] = (v + c.norm_offset) / c.norm_div;
            }
        }
    };

    if (nf < 16) {
        body(0, nf);
    } else {
        const int64_t chunk = (nf + kThreads - 1) / kThreads;
        std::vector<std::thread> th;
        th.reserve(kThreads);
        for (int t = 0; t < kThreads; t++) {
            const int64_t a = t * chunk, b = std::min(nf, a + chunk);
            if (a < b) th.emplace_back(body, a, b);
        }
        for (auto& t : th) t.join();
    }
    return true;
}

}  // namespace dubhe
