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

struct LayerSpec {
    uint64_t off[S_COUNT] = {0};      // 各张量在 mmap 内的字节偏移
    uint32_t type[S_COUNT] = {0};
    uint64_t bias[3] = {~0ull, ~0ull, ~0ull};  // q/k/v bias 字节偏移，~0 = 无
};

struct Spec {
    int n_embd = 0, n_ff = 0, n_vocab = 0, n_layer = 0;
    int n_head = 0, n_head_kv = 0, hd = 0;
    float rms_eps = 1e-5f;
    float rope_theta = 10000.f;
    int kv_cap = 0;
    uint64_t off_out_norm = 0, off_out_w = 0;
    uint32_t type_out_w = 0;
    std::vector<LayerSpec> layers;
};

bool fwd_init(const Spec& s, std::string* err);
// KV cache 扩容（会重排已有内容，pos 不变）
bool fwd_grow_kv(int new_cap, std::string* err);
// 一次解码步。x0 是已反量化的 embedding（n_embd 个 float）。
bool fwd_token(const float* x0, int pos, float* logits, std::string* err);

}  // namespace metal
}  // namespace dubhe
