#include <dubhe/dubhe.h>
#include <jni.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct NativeEngine {
    std::unique_ptr<dubhe_engine, decltype(&dubhe_destroy)> engine;
    bool loaded = false;
    int vocabulary = 0;
    int context_limit = 0;
    int embedding_size = 0;
};

NativeEngine& context(jlong handle) {
    return *reinterpret_cast<NativeEngine*>(handle);
}

jstring utf8(JNIEnv* env, const char* text) {
    const auto length = static_cast<jsize>(std::strlen(text));
    jbyteArray bytes = env->NewByteArray(length);
    if (!bytes) return nullptr;
    env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte*>(text));
    if (env->ExceptionCheck()) return nullptr;
    jclass cls = env->FindClass("java/lang/String");
    if (!cls) return nullptr;
    jmethodID ctor = env->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
    if (!ctor) return nullptr;
    jstring charset = env->NewStringUTF("UTF-8");
    if (!charset) return nullptr;
    auto result = static_cast<jstring>(env->NewObject(cls, ctor, bytes, charset));
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(charset);
    env->DeleteLocalRef(cls);
    return result;
}

std::string text(JNIEnv* env, jbyteArray bytes) {
    if (!bytes) throw std::invalid_argument("Missing UTF-8 text.");
    std::string result(static_cast<size_t>(env->GetArrayLength(bytes)), '\0');
    if (!result.empty()) {
        env->GetByteArrayRegion(bytes, 0, static_cast<jsize>(result.size()),
                                reinterpret_cast<jbyte*>(result.data()));
    }
    if (result.find('\0') != std::string::npos) {
        throw std::invalid_argument("Text contains a NUL character.");
    }
    return result;
}

void error(JNIEnv* env, int code, const char* message) {
    if (env->ExceptionCheck()) return;
    jclass cls = env->FindClass("dev/dubhe/sdk/DubheException");
    if (!cls) return;
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
    if (!ctor) return;
    jstring detail = utf8(env, message);
    if (!detail) return;
    auto exception = static_cast<jthrowable>(env->NewObject(cls, ctor, code, detail));
    if (exception) env->Throw(exception);
}

template<typename Function>
auto guarded(JNIEnv* env, Function action) -> decltype(action()) {
    try {
        return action();
    } catch (const std::bad_alloc&) {
        error(env, DUBHE_ERR_OOM, "Not enough memory to run Dubhe.");
    } catch (const std::invalid_argument& e) {
        error(env, DUBHE_ERR_INVALID_ARG, e.what());
    } catch (const std::exception& e) {
        error(env, DUBHE_ERR_STATE, e.what());
    } catch (...) {
        error(env, DUBHE_ERR_STATE, "Unexpected native error.");
    }
    return {};
}

jobject model_info(JNIEnv* env, NativeEngine& native) {
    dubhe_model_info info{};
    const int rc = dubhe_model_info_get(native.engine.get(), &info);
    if (rc != DUBHE_OK) {
        error(env, rc, "Model information is unavailable.");
        return nullptr;
    }
    native.vocabulary = info.n_vocab;
    native.context_limit = info.n_ctx_limit;
    native.embedding_size = info.n_embd;
    jclass cls = env->FindClass("dev/dubhe/sdk/ModelInfo");
    if (!cls) return nullptr;
    jmethodID ctor = env->GetMethodID(
        cls, "<init>", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIZJJ)V");
    if (!ctor) return nullptr;
    jstring arch = utf8(env, info.arch);
    if (!arch) return nullptr;
    jstring backend = utf8(env, info.backend);
    if (!backend) return nullptr;
    jstring quantization = utf8(env, info.quant_desc);
    if (!quantization) return nullptr;
    return env->NewObject(cls, ctor, arch, backend, quantization, info.n_ctx_limit,
                          info.n_ctx_ceiling, info.n_vocab, info.has_chat_template != 0,
                          static_cast<jlong>(info.n_params), static_cast<jlong>(info.file_size));
}

struct Callbacks {
    JNIEnv* env;
    jobject target;
    jmethodID on_bytes;
    jmethodID should_stop;
};

void token(void* opaque, int32_t, const char* piece) {
    auto& callbacks = *static_cast<Callbacks*>(opaque);
    JNIEnv* env = callbacks.env;
    if (!piece || env->ExceptionCheck()) return;
    const auto length = static_cast<jsize>(std::strlen(piece));
    if (length == 0) return;
    jbyteArray bytes = env->NewByteArray(length);
    if (!bytes) return;
    env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte*>(piece));
    if (!env->ExceptionCheck()) env->CallVoidMethod(callbacks.target, callbacks.on_bytes, bytes);
    env->DeleteLocalRef(bytes);
}

int stopped(void* opaque) {
    auto& callbacks = *static_cast<Callbacks*>(opaque);
    if (callbacks.env->ExceptionCheck()) return 1;
    return callbacks.env->CallBooleanMethod(callbacks.target, callbacks.should_stop);
}

struct PendingEmbeds {
    dubhe_engine* engine;
    bool installed = false;
    ~PendingEmbeds() { if (installed) dubhe_clear_mm_embeds(engine); }
};

int encode_attachment(JNIEnv* env, NativeEngine& native, Callbacks& callbacks,
                      jint kind, jbyteArray path, jdouble fps, jint frames, jint max_side,
                      std::vector<jint>& tokens, PendingEmbeds& installed) {
    constexpr const char* markers[] = {"<|image|>", "<|video|>", "<|audio|>"};
    if (kind < 1 || kind > 3) throw std::invalid_argument("Unknown attachment type.");
    auto* engine = native.engine.get();
    const int required = kind == 3 ? 2 : 1;
    if ((dubhe_mm_caps(engine) & required) == 0) {
        error(env, DUBHE_ERR_STATE, "请选择与当前模型配套的 mmproj 文件。");
        return DUBHE_ERR_STATE;
    }
    int32_t marker = -1;
    if (dubhe_tokenize(engine, markers[kind - 1], 0, &marker, 1) != 1 ||
        std::count(tokens.begin(), tokens.end(), marker) != 1) {
        error(env, DUBHE_ERR_INVALID_ARG, "提示词应包含一个匹配的附件占位符。");
        return DUBHE_ERR_INVALID_ARG;
    }
    const auto value = text(env, path);
    if (env->ExceptionCheck()) return DUBHE_ERR_INVALID_ARG;
    float* features = nullptr;
    int32_t count = 0, dim = 0;
    int rc;
    switch (kind) {
        case 1:
            rc = dubhe_encode_image(engine, value.c_str(), &features, &count, &dim);
            break;
        case 2:
            rc = dubhe_encode_video(engine, value.c_str(), fps, frames, max_side, &features, &count, &dim);
            break;
        default:
            rc = dubhe_encode_audio(engine, value.c_str(), &features, &count, &dim);
            break;
    }
    auto buffer = std::unique_ptr<float, decltype(&dubhe_free_embeds)>(features, dubhe_free_embeds);
    if (stopped(&callbacks)) return DUBHE_ERR_CANCELLED;
    if (rc != DUBHE_OK) {
        error(env, rc, dubhe_last_error());
        return rc;
    }
    if (count <= 0 || static_cast<size_t>(count) + tokens.size() > static_cast<size_t>(native.context_limit)) {
        error(env, DUBHE_ERR_INVALID_ARG, "附件和消息超过模型上下文，请缩短输入。");
        return DUBHE_ERR_INVALID_ARG;
    }
    auto slot = std::find(tokens.begin(), tokens.end(), marker);
    auto position = static_cast<int32_t>(slot - tokens.begin());
    tokens.insert(slot, static_cast<size_t>(count) - 1, marker);
    rc = dubhe_set_mm_embeds(engine, buffer.get(), count, position, dim);
    if (rc == DUBHE_OK) installed.installed = true;
    else error(env, rc, dubhe_last_error());
    return rc;
}

} // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_dev_dubhe_sdk_DubheNative_version(JNIEnv* env, jobject) {
    return utf8(env, dubhe_version());
}

JNIEXPORT jstring JNICALL Java_dev_dubhe_sdk_DubheNative_hardwareInfo(JNIEnv* env, jobject) {
    return guarded(env, [&] { return utf8(env, dubhe_hardware_info()); });
}

JNIEXPORT jlong JNICALL Java_dev_dubhe_sdk_DubheNative_create(JNIEnv* env, jobject, jint ctx, jboolean gpu) {
    return guarded(env, [&]() -> jlong {
        if (ctx <= 0) throw std::invalid_argument("Context size must be positive.");
        dubhe_config config{};
        config.n_ctx = ctx;
        config.use_gpu = gpu ? 1 : 0;
        auto engine = std::unique_ptr<dubhe_engine, decltype(&dubhe_destroy)>(
            dubhe_create(&config), dubhe_destroy);
        if (!engine) throw std::bad_alloc();
        return reinterpret_cast<jlong>(new NativeEngine{std::move(engine)});
    });
}

JNIEXPORT void JNICALL Java_dev_dubhe_sdk_DubheNative_destroy(JNIEnv*, jobject, jlong handle) {
    delete &context(handle);
}

JNIEXPORT jobject JNICALL Java_dev_dubhe_sdk_DubheNative_loadModel(
    JNIEnv* env, jobject, jlong handle, jbyteArray file) {
    return guarded(env, [&]() -> jobject {
        auto& native = context(handle);
        const auto path = text(env, file);
        if (env->ExceptionCheck()) return nullptr;
        native.loaded = false;
        dubhe_unload_mmproj(native.engine.get());
        int rc = dubhe_load_model(native.engine.get(), path.c_str());
        if (rc != DUBHE_OK) {
            error(env, rc, dubhe_last_error());
            dubhe_unload_model(native.engine.get());
            return nullptr;
        }
        native.loaded = true;
        return model_info(env, native);
    });
}

JNIEXPORT void JNICALL Java_dev_dubhe_sdk_DubheNative_unloadModel(JNIEnv*, jobject, jlong handle) {
    auto& native = context(handle);
    dubhe_unload_mmproj(native.engine.get());
    dubhe_unload_model(native.engine.get());
    native.loaded = false;
}

JNIEXPORT jint JNICALL Java_dev_dubhe_sdk_DubheNative_loadProjector(
    JNIEnv* env, jobject, jlong handle, jbyteArray file) {
    return guarded(env, [&]() -> jint {
        auto& native = context(handle);
        if (!native.loaded) {
            error(env, DUBHE_ERR_STATE, "Load a model first.");
            return 0;
        }
        const auto path = text(env, file);
        if (env->ExceptionCheck()) return 0;
        auto* engine = native.engine.get();
        dubhe_unload_mmproj(engine);
        const int rc = dubhe_load_mmproj(engine, path.c_str());
        if (rc != DUBHE_OK) {
            error(env, rc, dubhe_last_error());
            return 0;
        }
        const int visual_dim = dubhe_mm_dim(engine);
        if (visual_dim > 0 && visual_dim != native.embedding_size) {
            dubhe_unload_mmproj(engine);
            error(env, DUBHE_ERR_INVALID_ARG, "mmproj 与当前模型的嵌入维度不匹配。");
            return 0;
        }
        return dubhe_mm_caps(engine);
    });
}

JNIEXPORT jintArray JNICALL Java_dev_dubhe_sdk_DubheNative_tokenize(
    JNIEnv* env, jobject, jlong handle, jbyteArray prompt, jboolean add_special) {
    return guarded(env, [&]() -> jintArray {
        auto& native = context(handle);
        if (!native.loaded) {
            error(env, DUBHE_ERR_STATE, "Load a model first.");
            return nullptr;
        }
        const auto value = text(env, prompt);
        if (env->ExceptionCheck()) return nullptr;
        const int count = dubhe_tokenize(native.engine.get(), value.c_str(), add_special, nullptr, 0);
        if (count < 0) {
            error(env, count, "Could not tokenize the prompt.");
            return nullptr;
        }
        jintArray result = env->NewIntArray(count);
        if (!result || count == 0) return result;
        std::vector<jint> ids(static_cast<size_t>(count));
        if (dubhe_tokenize(native.engine.get(), value.c_str(), add_special, ids.data(), count) != count) {
            error(env, DUBHE_ERR_STATE, "Token count changed unexpectedly.");
            return nullptr;
        }
        env->SetIntArrayRegion(result, 0, count, ids.data());
        return result;
    });
}

JNIEXPORT void JNICALL Java_dev_dubhe_sdk_DubheNative_abort(JNIEnv*, jobject, jlong handle) {
    dubhe_request_abort(context(handle).engine.get());
}

JNIEXPORT jobject JNICALL Java_dev_dubhe_sdk_DubheNative_generate(
    JNIEnv* env, jobject, jlong handle, jintArray ids, jint max_tokens, jfloat temperature,
    jint top_k, jfloat top_p, jfloat repeat_penalty, jint repeat_last_n, jint seed, jbyteArray attachment,
    jint kind, jdouble fps, jint frames, jint side, jobject callback) {
    return guarded(env, [&]() -> jobject {
        auto& native = context(handle);
        if (!native.loaded) {
            error(env, DUBHE_ERR_STATE, "Load a model first.");
            return nullptr;
        }
        auto count = env->GetArrayLength(ids);
        if (count <= 0) throw std::invalid_argument("Prompt is empty.");
        std::vector<jint> tokens(static_cast<size_t>(count));
        env->GetIntArrayRegion(ids, 0, count, tokens.data());
        if (env->ExceptionCheck()) return nullptr;
        if (std::any_of(tokens.begin(), tokens.end(), [&](jint x) { return x < 0 || x >= native.vocabulary; })) {
            throw std::invalid_argument("A prompt token is outside the model vocabulary.");
        }
        jclass cb_class = env->GetObjectClass(callback);
        if (!cb_class) return nullptr;
        jmethodID on_bytes = env->GetMethodID(cb_class, "onBytes", "([B)V");
        if (!on_bytes) return nullptr;
        jmethodID should_stop = env->GetMethodID(cb_class, "shouldStop", "()Z");
        if (!should_stop) return nullptr;
        Callbacks callbacks{env, callback, on_bytes, should_stop};
        dubhe_sampling settings{max_tokens, temperature, top_k, top_p, repeat_penalty,
                               repeat_last_n, seed};
        dubhe_stats stats{};
        PendingEmbeds installed{native.engine.get()};
        if (kind != 0 && !stopped(&callbacks)) {
            const int rc = encode_attachment(env, native, callbacks, kind, attachment, fps, frames, side,
                                             tokens, installed);
            if (rc != DUBHE_OK && rc != DUBHE_ERR_CANCELLED) return nullptr;
        }
        if (!stopped(&callbacks)) {
            const int rc = dubhe_generate(native.engine.get(), tokens.data(), static_cast<int32_t>(tokens.size()),
                                          &settings, token,
                                          stopped, &callbacks, &stats);
            if (env->ExceptionCheck()) return nullptr;
            if (rc != DUBHE_OK && rc != DUBHE_ERR_CANCELLED) {
                error(env, rc, dubhe_last_error());
                return nullptr;
            }
        }
        if (env->ExceptionCheck()) return nullptr;
        jclass cls = env->FindClass("dev/dubhe/sdk/GenerationStats");
        if (!cls) return nullptr;
        jmethodID ctor = env->GetMethodID(cls, "<init>", "(IIIDDDD)V");
        if (!ctor) return nullptr;
        return env->NewObject(cls, ctor, stats.n_prompt, stats.n_generated, stats.n_reused,
                              stats.t_prompt_ms, stats.t_gen_ms, stats.prefill_tps, stats.decode_tps);
    });
}

} // extern "C"
