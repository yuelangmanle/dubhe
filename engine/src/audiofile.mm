// audiofile.mm — AVFoundation 版音频解码
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include <cstring>

#include "audiofile.h"

namespace dubhe {
namespace {

void set_err(std::string* err, NSString* msg) {
    if (!err) return;
    *err = msg ? std::string([msg UTF8String]) : std::string("未知错误");
}

// 读一块。**注意**：AVAudioFile 读到文件末尾不是返回成功 + 0 帧，而是返回 NO
// 并带上 eofErr(-39)。当成错误处理的话，任何正常文件都会在最后一帧报错。
// 返回值：1 = 有数据，0 = 到头了，-1 = 真错误。
int read_chunk(AVAudioFile* file, AVAudioPCMBuffer* buf, std::string* err) {
    buf.frameLength = 0;
    NSError* e = nil;
    if ([file readIntoBuffer:buf error:&e]) return buf.frameLength > 0 ? 1 : 0;
    if (e && e.code == -39) return 0;      // eofErr
    if (file.framePosition >= file.length) return 0;
    set_err(err, e.localizedDescription);
    return -1;
}

}  // namespace

bool load_audio_16k_mono(const char* path, std::vector<float>* pcm, std::string* err) {
    if (!path) { if (err) *err = "路径为空"; return false; }
    pcm->clear();

    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:path];
        NSURL* url = [NSURL fileURLWithPath:p];
        NSError* e = nil;
        AVAudioFile* file = [[AVAudioFile alloc] initForReading:url error:&e];
        if (!file) { set_err(err, e.localizedDescription); return false; }

        AVAudioFormat* in = file.processingFormat;   // 通常是 float32 非交错
        AVAudioFormat* out = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                              sampleRate:16000
                                                                channels:1
                                                             interleaved:NO];
        if (!out) { if (err) *err = "无法创建 16k 单声道格式"; return false; }

        // 已经是 16k 单声道就直接读，省掉一次转换（wav 输入走的就是这条）
        const bool direct = (in.sampleRate == 16000.0 && in.channelCount == 1 &&
                             in.commonFormat == AVAudioPCMFormatFloat32);
        const AVAudioFrameCount cap = 8192;

        if (direct) {
            AVAudioPCMBuffer* buf = [[AVAudioPCMBuffer alloc] initWithPCMFormat:in
                                                                 frameCapacity:cap];
            while (true) {
                const int rc = read_chunk(file, buf, err);
                if (rc == 0) break;
                if (rc < 0) return false;
                const AVAudioFrameCount n = buf.frameLength;
                const float* src = buf.floatChannelData ? buf.floatChannelData[0] : nullptr;
                if (!src) { if (err) *err = "音频不是 float32 布局"; return false; }
                pcm->insert(pcm->end(), src, src + n);
            }
            return true;
        }

        AVAudioConverter* conv = [[AVAudioConverter alloc] initFromFormat:in toFormat:out];
        if (!conv) { if (err) *err = "不支持的音频格式"; return false; }

        AVAudioPCMBuffer* ib = [[AVAudioPCMBuffer alloc] initWithPCMFormat:in frameCapacity:cap];
        // 输出缓冲要够大：16k/8k 是放大 2 倍，留点余量
        const double ratio = 16000.0 / (in.sampleRate > 0 ? in.sampleRate : 16000.0);
        AVAudioPCMBuffer* ob = [[AVAudioPCMBuffer alloc] initWithPCMFormat:out
                                                            frameCapacity:(AVAudioFrameCount)(cap * (ratio > 1 ? ratio : 1) + 256)];
        if (!ib || !ob) { if (err) *err = "缓冲分配失败"; return false; }

        __block BOOL eof = NO;
        while (true) {
            NSError* ce = nil;
            AVAudioConverterOutputStatus st =
                [conv convertToBuffer:ob
                                error:&ce
                   withInputFromBlock:^AVAudioBuffer*(AVAudioPacketCount need, AVAudioConverterInputStatus* status) {
                       if (eof) { *status = AVAudioConverterInputStatus_EndOfStream; return nil; }
                       const int rc = read_chunk(file, ib, nullptr);
                       if (rc <= 0) {
                           eof = YES;
                           *status = AVAudioConverterInputStatus_EndOfStream;
                           return nil;
                       }
                       *status = AVAudioConverterInputStatus_HaveData;
                       return ib;
                   }];
            if (st == AVAudioConverterOutputStatus_HaveData) {
                const AVAudioFrameCount n = ob.frameLength;
                const float* src = ob.floatChannelData ? ob.floatChannelData[0] : nullptr;
                if (n && src) pcm->insert(pcm->end(), src, src + n);
                if (n == 0 && eof) break;
            } else if (st == AVAudioConverterOutputStatus_EndOfStream) {
                break;
            } else if (st == AVAudioConverterOutputStatus_InputRanDry) {
                if (eof) break;
            } else {
                set_err(err, ce.localizedDescription);
                return false;
            }
        }
    }
    if (pcm->empty()) { if (err) *err = "解出来是空音频"; return false; }
    return true;
}

}  // namespace dubhe
