// audprobe.cpp — 语音塔逐层对拍探针
//
// 和 vlprobe 一个用途：把每一层的激活 dump 成裸 f32，和 audio_oracle.py 存的
// .npy 逐个比。语音塔 17 层，错一层外面只能看到"答非所问"，从外面根本猜不到
// 是降采样还是第 11 层。逐层比是唯一能一次定位的办法。
//
// 编号约定（和 audio_diff.py 对齐）：
//   mel.f32       log-mel（128 × T）
//   down.f32      降采样输出（T/8 × 768）
//   blk<N>.f32    第 N 层输出
//   post_ln.f32   embedding_norm 之后
//   mm.f32        投影器输出（T/8 × 2048）
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "audiofile.h"
#include "audiomm.h"
#include "mel.h"

using namespace dubhe;

static bool dump(const std::string& dir, const std::string& name, const float* p, int64_t n) {
    const std::string path = dir + "/" + name + ".f32";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "写不了 %s\n", path.c_str()); return false; }
    std::fwrite(p, sizeof(float), (size_t)n, f);
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "用法: audprobe <mmproj.gguf> <audio> <outdir> [--mel-only]\n");
        return 2;
    }
    const char* mmproj = argv[1];
    const char* audio = argv[2];
    const std::string dir = argv[3];
    const bool mel_only = argc > 4 && std::strcmp(argv[4], "--mel-only") == 0;

    std::string err;
    std::vector<float> pcm;
    if (!load_audio_16k_mono(audio, &pcm, &err)) {
        std::fprintf(stderr, "解码失败: %s\n", err.c_str());
        return 1;
    }
    MelConfig mc;
    std::vector<float> mel;
    int64_t nf = 0;
    if (!log_mel(pcm.data(), (int64_t)pcm.size(), mc, &mel, &nf, &err)) {
        std::fprintf(stderr, "mel 失败: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "采样 %zu  时长 %.2fs  mel %d×%lld  token %lld\n",
                 pcm.size(), (double)pcm.size() / 16000.0, mc.n_mel, (long long)nf,
                 (long long)mel_num_tokens(nf, mc));
    if (!dump(dir, "mel", mel.data(), (int64_t)mc.n_mel * nf)) return 1;
    if (mel_only) return 0;

    AudioMm m;
    if (!m.load(mmproj, &err)) {
        std::fprintf(stderr, "加载失败: %s\n", err.c_str());
        return 1;
    }
    std::string desc;
    m.describe(&desc);
    std::fputs(desc.c_str(), stderr);

    const int D = m.config().out_dim;
    const int64_t nt = mel_num_tokens(nf, mc);
    std::vector<float> out((size_t)(nt * D));

    m.on_layer = [&](int li, const float* x, int64_t n) {
        if (li == -1) dump(dir, "down", x, n * m.config().hidden);
        else if (li == -2) dump(dir, "post_ln", x, n * m.config().hidden);
        else dump(dir, "blk" + std::to_string(li), x, n * m.config().hidden);
    };

    int got = 0;
    if (!m.encode(mel.data(), nf, out.data(), &got, &err)) {
        std::fprintf(stderr, "前向失败: %s\n", err.c_str());
        return 1;
    }
    dump(dir, "mm", out.data(), (int64_t)got * D);
    std::fprintf(stderr, "输出 %d token × %d 维\n", got, D);
    return 0;
}
