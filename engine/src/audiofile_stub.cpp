// audiofile_stub.cpp — 非 Apple 平台的占位实现
//
// 接口一致，让引擎在 Linux / Android 上照常编译（只是不能读音频）。
// Android 阶段接 miniaudio 或 MediaCodec 时，把这里的函数换成真实现即可。
#include "audiofile.h"

namespace dubhe {

bool load_audio_16k_mono(const char* path, std::vector<float>* pcm, std::string* err) {
    (void)path; (void)pcm;
    if (err) *err = "本平台的音频解码尚未接入（macOS 走 AVFoundation，其他平台待补 miniaudio）";
    return false;
}

}  // namespace dubhe
