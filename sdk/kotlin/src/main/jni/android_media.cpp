#include "audiofile.h"
#include "image.h"
#include "image_resize.h"

#include <android/bitmap.h>
#include <jni.h>

#include <cstddef>
#include <cstring>

namespace {

JavaVM* vm = nullptr;

struct LocalFrame {
    JNIEnv* env;
    bool ok;

    explicit LocalFrame(JNIEnv* e) : env(e), ok(e && e->PushLocalFrame(16) == JNI_OK) {}
    ~LocalFrame() { if (ok) env->PopLocalFrame(nullptr); }
};

JNIEnv* current_env() {
    JNIEnv* env = nullptr;
    if (vm) vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    return env;
}

bool media_error(JNIEnv* env, std::string* err, const char* fallback) {
    if (err) *err = fallback;
    if (env && env->ExceptionCheck()) {
        jthrowable failure = env->ExceptionOccurred();
        env->ExceptionClear();
        jclass cls = env->GetObjectClass(failure);
        jmethodID method = cls ? env->GetMethodID(cls, "getMessage", "()Ljava/lang/String;") : nullptr;
        auto detail = method ? static_cast<jstring>(env->CallObjectMethod(failure, method)) : nullptr;
        if (detail && err) {
            const char* chars = env->GetStringUTFChars(detail, nullptr);
            if (chars) {
                if (*chars) *err = chars;
                env->ReleaseStringUTFChars(detail, chars);
            }
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    return false;
}

jbyteArray path_bytes(JNIEnv* env, const char* path) {
    const auto size = static_cast<jsize>(std::strlen(path));
    jbyteArray bytes = env->NewByteArray(size);
    if (bytes) env->SetByteArrayRegion(bytes, 0, size, reinterpret_cast<const jbyte*>(path));
    return bytes;
}

bool bitmap_rgb(JNIEnv* env, jobject bitmap, int* w, int* h, std::vector<uint8_t>* rgb) {
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        info.width == 0 || info.height == 0 || info.width > 2048 || info.height > 2048) return false;
    rgb->resize(static_cast<size_t>(info.width) * info.height * 3);
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return false;
    auto* output = rgb->data();
    for (uint32_t y = 0; y < info.height; ++y) {
        auto* row = static_cast<const uint8_t*>(pixels) + static_cast<size_t>(y) * info.stride;
        for (uint32_t x = 0; x < info.width; ++x) {
            *output++ = row[x * 4];
            *output++ = row[x * 4 + 1];
            *output++ = row[x * 4 + 2];
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    *w = static_cast<int>(info.width);
    *h = static_cast<int>(info.height);
    return true;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* machine, void*) {
    vm = machine;
    return JNI_VERSION_1_6;
}

namespace dubhe {

bool load_image_rgb(const char* path, int* w, int* h, std::vector<uint8_t>* rgb,
                    std::string* err) {
    JNIEnv* env = current_env();
    LocalFrame frame(env);
    if (!frame.ok) return media_error(env, err, "无法初始化 Android 图像解码。");
    jclass cls = env->FindClass("dev/dubhe/sdk/AndroidMedia");
    jmethodID method = cls ? env->GetStaticMethodID(cls, "image", "([B)Landroid/graphics/Bitmap;") : nullptr;
    jbyteArray file = method ? path_bytes(env, path) : nullptr;
    jobject bitmap = file && !env->ExceptionCheck() ? env->CallStaticObjectMethod(cls, method, file) : nullptr;
    if (env->ExceptionCheck() || !bitmap || !bitmap_rgb(env, bitmap, w, h, rgb)) {
        return media_error(env, err, "无法解码这张图片。");
    }
    return true;
}

bool load_video_frames_rgb(const char* path, double fps, int max_frames, int max_side,
                           int* w, int* h, std::vector<std::vector<uint8_t>>* frames,
                           std::string* err) {
    JNIEnv* env = current_env();
    LocalFrame frame(env);
    if (!frame.ok) return media_error(env, err, "无法初始化 Android 视频解码。");
    jclass cls = env->FindClass("dev/dubhe/sdk/AndroidMedia");
    jmethodID method = cls
        ? env->GetStaticMethodID(cls, "video", "([BDII)[Landroid/graphics/Bitmap;") : nullptr;
    jbyteArray file = method ? path_bytes(env, path) : nullptr;
    auto pictures = file && !env->ExceptionCheck()
        ? static_cast<jobjectArray>(env->CallStaticObjectMethod(cls, method, file, fps, max_frames, max_side))
        : nullptr;
    if (!pictures || env->ExceptionCheck()) return media_error(env, err, "无法抽取视频帧。");
    const jsize count = env->GetArrayLength(pictures);
    if (count < 2) return media_error(env, err, "视频中可用的帧不足两张。");
    frames->clear();
    frames->reserve(static_cast<size_t>(count) + 1);
    for (jsize i = 0; i < count; ++i) {
        jobject bitmap = env->GetObjectArrayElement(pictures, i);
        int fw = 0, fh = 0;
        std::vector<uint8_t> rgb;
        const bool ok = bitmap && bitmap_rgb(env, bitmap, &fw, &fh, &rgb);
        if (!ok) return media_error(env, err, "无法读取视频帧。");
        env->DeleteLocalRef(bitmap);
        if (frames->empty()) {
            *w = fw;
            *h = fh;
        } else if (fw != *w || fh != *h) {
            std::vector<uint8_t> resized(static_cast<size_t>(*w) * *h * 3);
            resize_bicubic_rgb(rgb.data(), fw, fh, resized.data(), *w, *h);
            rgb.swap(resized);
        }
        frames->push_back(std::move(rgb));
    }
    if (frames->size() % 2) frames->push_back(frames->back());
    return true;
}

bool load_audio_16k_mono(const char* path, std::vector<float>* pcm, std::string* err) {
    JNIEnv* env = current_env();
    LocalFrame frame(env);
    if (!frame.ok) return media_error(env, err, "无法初始化 Android 音频解码。");
    jclass cls = env->FindClass("dev/dubhe/sdk/AndroidMedia");
    jmethodID method = cls ? env->GetStaticMethodID(cls, "audio", "([B)[F") : nullptr;
    jbyteArray file = method ? path_bytes(env, path) : nullptr;
    auto samples = file && !env->ExceptionCheck()
        ? static_cast<jfloatArray>(env->CallStaticObjectMethod(cls, method, file)) : nullptr;
    if (!samples || env->ExceptionCheck()) return media_error(env, err, "无法解码这段音频。");
    const jsize count = env->GetArrayLength(samples);
    pcm->resize(static_cast<size_t>(count));
    env->GetFloatArrayRegion(samples, 0, count, pcm->data());
    return !env->ExceptionCheck() || media_error(env, err, "无法读取音频采样。");
}

void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    resize_bicubic_rgb(src, sw, sh, dst, dw, dh);
}

} // namespace dubhe
