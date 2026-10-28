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
#define DUBHE_VERSION_MINOR 1
#define DUBHE_VERSION_PATCH 0

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
    int32_t  n_ctx_limit;   /* 当前实际可用的上下文长度 */
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
} dubhe_stats;

/* 流式回调：每产出一个 token 调一次。piece 可为 NULL（无对应字节片段）。 */
typedef void (*dubhe_token_cb)(void* ud, int32_t token, const char* piece);
/* 返回非 0 表示请求中止生成。 */
typedef int (*dubhe_stop_cb)(void* ud);

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

/* ---------- 生成 ---------- */
int dubhe_generate(dubhe_engine* e,
                   const int32_t* prompt, int32_t n_prompt,
                   const dubhe_sampling* sp,
                   dubhe_token_cb on_token,
                   dubhe_stop_cb  should_stop,
                   void* ud,
                   dubhe_stats* stats);

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
