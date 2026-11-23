// dubhe.h — Dubhe 引擎公开 C ABI
//
// 规则：
//   1. 本头文件不得出现 C++ 类型、STL 容器、异常。
//   2. 所有可能失败的函数返回 int（0 = 成功，负数 = 错误码）。
//   3. 错误详情用 dubhe_last_error() 取，线程本地。
//   4. 字符串一律 UTF-8。

#ifndef DUBHE_H
#define DUBHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DUBHE_VERSION_MAJOR 0
#define DUBHE_VERSION_MINOR 37
#define DUBHE_VERSION_PATCH 1

typedef struct dubhe_engine dubhe_engine;

/* ---------- 错误码 ---------- */
enum {
    DUBHE_OK = 0,
    DUBHE_ERR_INVALID_ARG = -1,
    DUBHE_ERR_IO = -2,
    DUBHE_ERR_FORMAT = -3,
    DUBHE_ERR_UNSUPPORTED = -4,
    DUBHE_ERR_OOM = -5,
    DUBHE_ERR_STATE = -6,
    DUBHE_ERR_CANCELLED = -7,
};

/* ---------- 配置 ---------- */
typedef struct {
    int32_t n_threads;   /* 0 = 自动（物理核数） */
    int32_t n_ctx;       /* 0 = 用模型训练上下文 */
    int32_t use_gpu;     /* 1 = 优先 Metal（不可用时自动回落 CPU） */
    int32_t n_batch;     /* prompt 分块大小，0 = 512 */
} dubhe_config;

/* ---------- 模型信息 ---------- */
typedef struct {
    char     arch[32];
    char     name[128];
    int32_t  n_vocab;
    int32_t  n_ctx_train;   /* 模型训练时的上下文长度 */
    int32_t  n_ctx_limit;   /* 当前生效的上下文长度 */
    int32_t  n_ctx_ceiling; /* 这台机器装得下的上限（训练上下文与内存夹完） */
    int32_t  n_embd;
    int32_t  n_layer;
    int32_t  n_head;
    int32_t  n_head_kv;
    int32_t  n_ff;
    int32_t  n_rot;
    uint64_t n_params;
    uint64_t file_size;
    char     quant_desc[64];
    int32_t  has_chat_template;
    int32_t  is_gpu;
    char     backend[96];   /* 计算后端说明，如 "Metal · Apple M4（校验误差 1.2e-07）" */
} dubhe_model_info;

/* ---------- 采样参数 ---------- */
typedef struct {
    int32_t max_tokens;
    float   temperature;     /* 0 = greedy */
    int32_t top_k;           /* <=0 关闭 */
    float   top_p;           /* <=0 或 >=1 关闭 */
    float   repeat_penalty;  /* 1.0 = 关闭 */
    int32_t repeat_last_n;
    int32_t seed;            /* 0 = 随机 */
} dubhe_sampling;

/* ---------- 生成统计 ---------- */
typedef struct {
    int32_t n_prompt;
    int32_t n_generated;
    double  t_prompt_ms;
    double  t_gen_ms;
    double  prefill_tps;
    double  decode_tps;
    /* 本轮跳过 prefill、直接复用上一轮 KV 的 token 数。
     * 多轮对话里这个数接近 n_prompt 就说明前缀缓存生效了。 */
    int32_t n_reused;
} dubhe_stats;

/* 流式回调：每产出一个 token 调一次。piece 可为 NULL（无对应字节片段）。 */
typedef void (*dubhe_token_cb)(void* ud, int32_t token, const char* piece);
/* 返回非 0 表示请求中止生成。 */
typedef int (*dubhe_stop_cb)(void* ud);
/* 约束采样用。piece 为 NULL 表示结束标记，其他情况 len 是原始字节数。 */
typedef int (*dubhe_token_filter)(void* ud, int32_t token, const char* piece, int32_t len);

/* ---------- 生命周期 ---------- */
dubhe_engine* dubhe_create(const dubhe_config* cfg);
void          dubhe_destroy(dubhe_engine* e);

int dubhe_load_model(dubhe_engine* e, const char* path);
void dubhe_unload_model(dubhe_engine* e);
int dubhe_model_info_get(const dubhe_engine* e, dubhe_model_info* out);

/* ---------- 分词 ---------- */
/* 返回 token 数；cap 不足时返回所需数量且不写入。 */
int32_t dubhe_tokenize(const dubhe_engine* e, const char* text, int add_special,
                       int32_t* out, int32_t cap);
/* 返回写入的字节数；buf 为 NULL 时只返回长度。token 可能是半个 UTF-8 字符。 */
int32_t dubhe_token_to_piece(const dubhe_engine* e, int32_t token, char* buf, int32_t cap);

/* ---------- 多模态 ---------- */
/* 加载 mmproj（视觉/音频塔的 GGUF，独立于语言模型）。 */
int  dubhe_load_mmproj(dubhe_engine* e, const char* path);

/* 卸载已加载的视觉塔和语音塔。
 *
 * 为什么必须有这个：塔是跟模型配套的，不是跟引擎配套的。换模型时不换塔，
 * 维度检查也拦不住 —— LFM3-VL-2B 和 LFM3-1.7B 的 hidden 都是 2048，
 * 特征能顺利注入，然后模型回一段完全无关的话（实测切到 F16 后发一张图，
 * 回的是"They provided a block of text that appears to be some garbled
 * mix…"）。HTTP 200、无异常、无日志，比直接报错难查得多。
 * 所以换模型的一方必须显式调用本函数。 */
void dubhe_unload_mmproj(dubhe_engine* e);

/* 已加载的多模态能力位掩码：1 = 图像/视频（视觉塔），2 = 语音（音频塔）。
 * 都没加载返回 0。服务端靠它决定 UI 上该露出哪些附件入口。 */
int  dubhe_mm_caps(dubhe_engine* e);

/* 视觉塔投影后的宽度（= 文本模型的 hidden）。没加载视觉塔返回 0。
 * 换模型时用它做一次对齐检查，比等到注入阶段才发现要早。 */
int  dubhe_mm_dim(dubhe_engine* e);

/* 把一张图编码成可直接注入的特征：返回 *embeds（n_tokens × dim，行主序
 * float32，调用方用 dubhe_free_embeds 释放）。这一步是纯 CPU 的视觉塔前向，
 * 一张 640×640 的图目前要十几秒，调用方应放到后台线程。 */
int  dubhe_encode_image(dubhe_engine* e, const char* image_path, float** embeds,
                        int32_t* n_tokens, int32_t* dim);

/* 把一段视频编码成特征：按 fps 抽帧（最多 max_frames 帧），每 2 帧合成一个
 * temporal patch 过一遍视觉塔，输出按时间顺序拼接。max_side <= 0 表示不缩放，
 * 否则把长边压到 max_side（省时间的主要手段）。fps <= 0 取 2。 */
int  dubhe_encode_video(dubhe_engine* e, const char* video_path, double fps,
                        int32_t max_frames, int32_t max_side,
                        float** embeds, int32_t* n_tokens, int32_t* dim);

/* 把一段音频编码成特征：任意格式（wav/aiff/mp3/m4a/flac）先解成 16 kHz 单声道，
 * 再算 log-mel，过语音塔。一个 token 覆盖 80 ms 音频。需要先加载语音 mmproj
 * （dubhe_load_mmproj 会按 clip.has_audio_encoder 自动区分视觉/语音）。 */
int  dubhe_encode_audio(dubhe_engine* e, const char* audio_path, float** embeds,
                        int32_t* n_tokens, int32_t* dim);
void dubhe_free_embeds(float* embeds);

/* 把图像/音频塔的输出注入下一次 dubhe_generate：序列里 [at_pos, at_pos+n_tokens)
 * 这些位置的 token 嵌入会被 embeds 覆盖（占位 token 本身是什么无所谓）。
 * embeds 是 n_tokens × dim 的行主序 float32，dim 必须等于模型的 n_embd。
 * 数据会被拷贝，调用后即可释放；生成结束前一直有效，直到下一次 set/clear。 */
int  dubhe_set_mm_embeds(dubhe_engine* e, const float* embeds, int32_t n_tokens,
                         int32_t at_pos, int32_t dim);
void dubhe_clear_mm_embeds(dubhe_engine* e);

/* ---------- 生成 ---------- */
int dubhe_generate(dubhe_engine* e,
                   const int32_t* prompt, int32_t n_prompt,
                   const dubhe_sampling* sp,
                   dubhe_token_cb on_token,
                   dubhe_stop_cb  should_stop,
                   void* ud,
                   dubhe_stats* stats);

/* 与 dubhe_generate 相同，但采样时只选取 filter 允许的候选。回调不得重入 e。 */
int dubhe_generate_filtered(dubhe_engine* e, const int32_t* prompt, int32_t n_prompt,
                            const dubhe_sampling* sp, dubhe_token_filter filter,
                            dubhe_token_cb on_token, dubhe_stop_cb should_stop,
                            void* ud, dubhe_stats* stats);

typedef struct {
    const int32_t* prompt;
    int32_t n_prompt;
    dubhe_sampling sampling;
    dubhe_token_cb on_token;
    dubhe_stop_cb should_stop;
    void* user_data;
    dubhe_stats stats;
    int32_t result;
    int32_t length_limited;
} dubhe_batch_item;

/* Maximum number of independent sequences supported by the loaded backend.
 * A result of zero means use dubhe_generate. The combined token budget must
 * fit n_ctx_limit. Each item receives its own cancellation status. */
int32_t dubhe_batch_capacity(const dubhe_engine* e);
int dubhe_generate_batch(dubhe_engine* e, dubhe_batch_item* items, int32_t count);

/* 中止当前生成（可从其他线程调用）。 */
void dubhe_request_abort(dubhe_engine* e);

/* ---------- 工具 ---------- */
const char* dubhe_last_error(void);
const char* dubhe_version(void);
/* 返回硬件能力摘要（芯片名、核心数、内存），用于 UI 展示。 */
const char* dubhe_hardware_info(void);

#ifdef __cplusplus
}
#endif

#endif /* DUBHE_H */
