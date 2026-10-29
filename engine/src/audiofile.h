// audiofile.h — 音频解码（任意格式 → 16 kHz 单声道 f32）
//
// 和图像一样的分工：容器与编码格式（wav / aiff / mp3 / m4a / caf / flac）不值得
// 自己写，macOS 上 AVAudioFile + AVAudioConverter 一行就全包了，还顺带做了
// 采样率转换（手机录音常见 44.1k/48k，模型只吃 16k）。
// 其他平台走 audiofile_stub.cpp 报错，等 Android 阶段接 miniaudio。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dubhe {

// 解成 16 kHz 单声道 float32（[-1, 1]）。多声道按 AVFoundation 的默认混音。
bool load_audio_16k_mono(const char* path, std::vector<float>* pcm, std::string* err);

}  // namespace dubhe
