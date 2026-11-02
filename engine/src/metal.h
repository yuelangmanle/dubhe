// metal.h — Metal GPU 后端内部接口（CPU-only 平台会退化成空实现）
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dubhe {
namespace metal {

bool available();
bool init(std::string* err);
void shutdown();
const char* device_name();
const char* last_error();

// 把整块 mmap 权重零拷贝绑成 MTLBuffer。base 必须页对齐。
bool bind_weights(const void* base, size_t size, std::string* err);
// 只释放权重绑定与前向缓冲，**不**拆设备/命令队列。
// 模型卸载必须走这个：device 和 library 是进程级共享的，
// 一个模型 unload 把全局状态拆了，同进程里其它模型立刻失效。
void unbind_weights();

// 该权重类型 + n_in 是否能走 GPU
bool supports(uint32_t type, int n_in);

// y[n_out] = W[n_out][n_in] · x[n_in]。woff 是张量在 mmap 内的字节偏移。
// 单次调用 = 单次提交 + 单次同步，只用于加载期校验。
bool mfvec(uint32_t type, const float* x, float* y, int n_out, int n_in, uint64_t woff);

// ---------------------------------------------------------------- 全 GPU 前向
//
// 整层前向（rmsnorm / qkv / rope / attn / o / ffn / 残差）编进**一个**
// command buffer，CPU 每个 token 只同步一次。这是本后端性能的全部关键：
// 单次同步 200 µs，逐算子同步会直接吃掉 GPU 的算力优势。

// 张量槽位（每层 9 个矩阵）
enum Slot {
    S_ATTN_NORM = 0, S_Q, S_K, S_V, S_OUT, S_FFN_NORM, S_GATE, S_UP, S_DOWN,
    S_COUNT
};

// LFM3 额外需要的权重槽位。llama 系全部留空（off = ~0）。
enum SlotEx {
    X_QNORM = 0, X_KNORM, X_AGATE, X_POST_ATTN, X_POST_FFN,
    X_CONV_IN, X_CONV_K, X_CONV_OUT,
    X_COUNT
};

// 子层种类，与 internal.h 的 LayerKind 对应
enum LayerKindGpu { G_ATTN_FULL = 0, G_ATTN_SWA = 1, G_CONV = 2 };

struct LayerSpec {
    uint64_t off[S_COUNT] = {0};      // 各张量在 mmap 内的字节偏移
    uint32_t type[S_COUNT] = {0};
    uint64_t bias[3] = {~0ull, ~0ull, ~0ull};  // q/k/v bias 字节偏移，~0 = 无

    // ---- LFM3
    int32_t  kind = G_ATTN_FULL;
    uint64_t ex_off[X_COUNT] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
    uint32_t ex_type[X_COUNT] = {0};
    int32_t  conv_slot = -1;          // conv 状态缓冲里的槽位
    float    rope_theta_l = 10000.f;  // 本层实际用的 RoPE base（滑窗层不同）
    int32_t  rope_angles_l = 0;       // 本层实际旋转的角度数（滑窗层是全量）
};

struct Spec {
    int n_embd = 0, n_ff = 0, n_vocab = 0, n_layer = 0;
    int n_head = 0, n_head_kv = 0, hd = 0;
    float rms_eps = 1e-5f;
    float rope_theta = 10000.f;
    int kv_cap = 0;
    uint64_t off_out_norm = 0, off_out_w = 0;
    uint32_t type_out_w = 0;
    // ---- LFM3
    int32_t n_swa = 0;          // 滑窗宽度，0 表示没有滑窗层
    int32_t rope_angles = 0;    // partial RoPE 角度数，0 = 全部
    int32_t conv_l_cache = 0;   // ShortConv 核宽
    int32_t n_conv = 0;         // conv 层个数（决定状态缓冲大小）
    std::vector<LayerSpec> layers;
};

bool fwd_init(const Spec& s, std::string* err);
// KV cache 扩容（会重排已有内容，pos 不变）
bool fwd_grow_kv(int new_cap, std::string* err);
// 一次解码步。x0 是已反量化的 embedding（n_embd 个 float）。
bool fwd_token(const float* x0, int pos, float* logits, std::string* err);
// prefill 批处理：x0 是 [nrow][n_embd] 已反量化的 embedding，pos0 是首行位置。
// 只有最后一行需要 logits，其余行的 out_w 被省掉。
bool fwd_batch(const float* x0, int nrow, int pos0, float* logits, std::string* err);
// 同上，但整段落到 KV 缓冲的 slab0 号槽位（连续批处理时每条序列一块）。
// 位置 pos0..pos0+nrow-1 都相对于 slab0 起算，注意力只看本槽位内的内容。
bool fwd_batch_at(const float* x0, int nrow, int pos0, uint64_t slab0, uint64_t cslot,
                  float* logits, std::string* err);
// 多序列批量前向：nrow 行分属 nrow 个**独立序列**（连续批处理用）。
//   rpos[r]  第 r 行在自己序列里的位置
//   rslab[r] 该序列在本层 KV 缓冲里的起始槽位（单位是位置，不是字节）
//   rslot[r] 该序列的 ShortConv 历史槽位；传 nullptr 时按 r 编号
// logits 必须按 [nrow][n_vocab] 给出。
//
// 和 fwd_batch 的区别只有一处：位置相关的小算子（RoPE / v 写入 / 注意力 /
// 卷积状态）按行各派发一次，权重 matvec 仍然整批一次 —— 摊薄的是权重读取，
// 那才是解码的全部成本。
bool fwd_multi(const float* x0, int nrow, const int* rpos, const uint64_t* rslab,
               const uint64_t* rslot, float* logits, std::string* err);
// 批大小上限（与着色器 BMAX 一致）
int fwd_batch_max();
// 清零 LFM3 的 ShortConv 历史缓冲（新序列开始时调用）
void fwd_reset_conv();
// 调试：把 GPU 侧的残差流（g_bx 第 0 行）拷出来，用于和 CPU 路径逐位置对拍
bool fwd_copy_x(float* dst, int n);

}  // namespace metal
}  // namespace dubhe
