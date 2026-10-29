// internal.h — 引擎内部共享定义（不对外）
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dubhe {

// ---------------------------------------------------------------- ggml 类型
enum GgmlType : uint32_t {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3,
    T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8, T_Q8_1 = 9,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_Q8_K = 15,
    T_I8 = 24, T_I16 = 25, T_I32 = 26, T_I64 = 27, T_F64 = 28, T_BF16 = 30,
};

int  type_block_size(uint32_t t);   // 每 block 元素数（1 表示非分块）
int  type_block_bytes(uint32_t t);  // 每 block 字节数
size_t type_row_bytes(uint32_t t, int64_t n);
const char* type_name(uint32_t t);
bool type_supported(uint32_t t);

// ---------------------------------------------------------------- 半精度
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// 就近舍入（ties-to-even），用于把激活量化成 Q8
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));  // inf/nan
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t sh = (uint32_t)(14 - exp);
        uint32_t m = mant >> sh;
        if ((mant >> (sh - 1)) & 1u) m++;
        return (uint16_t)(sign | m);
    }
    uint32_t m = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) { m = 0; exp++; }
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}

// ---------------------------------------------------------------- GGUF
struct TensorInfo {
    std::string name;
    uint32_t    type = 0;
    int64_t     ne[4] = {1, 1, 1, 1};
    int         n_dims = 1;
    uint64_t    offset = 0;   // 相对 data 段
};

struct GgufValue {
    uint32_t type = 0;
    uint64_t u64 = 0;
    int64_t  i64 = 0;
    double   f64 = 0;
    std::string str;
    std::vector<uint8_t>  arr_u8;
    std::vector<int32_t>  arr_i32;
    std::vector<uint32_t> arr_u32;
    std::vector<int64_t>  arr_i64;
    std::vector<float>    arr_f32;
    std::vector<std::string> arr_str;
};

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    bool open(const char* path, std::string* err);
    void close();

    const GgufValue* get(const std::string& key) const;
    std::string  get_str(const std::string& key, const std::string& def = "") const;
    uint32_t     get_u32(const std::string& key, uint32_t def = 0) const;
    float        get_f32(const std::string& key, float def = 0.f) const;
    bool         has(const std::string& key) const;
    // 每层取值不同的架构（LFM3 的 head_count_kv / sliding_window_pattern）要读数组
    const std::vector<int32_t>* get_arr_i32(const std::string& key) const;
    const std::vector<uint8_t>* get_arr_u8(const std::string& key) const;

    const TensorInfo* find_tensor(const std::string& name) const;
    const void* tensor_data(const TensorInfo& t) const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const void* mapping() const { return base_; }   // mmap 起始地址（页对齐，可直接包成 MTLBuffer）
    const std::map<std::string, GgufValue>& kv() const { return kv_; }
    size_t file_size() const { return file_size_; }
    size_t data_size() const { return data_size_; }

private:
    bool parse_header(std::string* err);

    void*  map_ = nullptr;
    size_t file_size_ = 0;
    const uint8_t* base_ = nullptr;
    const uint8_t* data_ = nullptr;
    size_t data_size_ = 0;
    std::map<std::string, GgufValue> kv_;
    std::vector<TensorInfo> tensors_;
};

// ---------------------------------------------------------------- 量化
// 布局与 llama.cpp 的 ggml 完全一致（GGUF 的事实标准）。
#pragma pack(push, 1)
struct block_q4_0 { uint16_t d; uint8_t qs[16]; };
struct block_q4_1 { uint16_t d; uint16_t m; uint8_t qs[16]; };
struct block_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q5_1 { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q8_0 { uint16_t d; int8_t  qs[32]; };
struct block_q4_K { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q5_K { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "q4_0");
static_assert(sizeof(block_q4_1) == 20, "q4_1");
static_assert(sizeof(block_q5_0) == 22, "q5_0");
static_assert(sizeof(block_q5_1) == 24, "q5_1");
static_assert(sizeof(block_q8_0) == 34, "q8_0");
static_assert(sizeof(block_q4_K) == 144, "q4_K");
static_assert(sizeof(block_q5_K) == 176, "q5_K");
static_assert(sizeof(block_q6_K) == 210, "q6_K");
static_assert(sizeof(block_q8_K) == 292, "q8_K");

// K-quant 的 6-bit (scale, min) 解包
void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m);

// 把一行（n 个元素）反量化到 dst（n 个 float）。
void dequant_row(uint32_t type, const void* src, float* dst, int64_t n);
// 点积：x(f32) · row(quant)。n 必须能被 block size 整除。这条路径是数值基准。
float vec_dot(uint32_t type, const float* x, const void* row, int64_t n);

// ---- 量化激活路径（整数点积，比 f32 路径快数倍）
// 是否有整数路径
bool   act_q8_supported(uint32_t wtype);
// 该权重类型需要多大的激活缓冲（字节）
size_t act_q8_bytes(uint32_t wtype, int64_t n);
// 把 x 量化到 dst
void   quantize_act(uint32_t wtype, const float* x, void* dst, int64_t n);
// 量化激活 · 量化权重
float  vec_dot_q8(uint32_t wtype, const void* xq, const void* row, int64_t n);

// ---------------------------------------------------------------- 分词
class Tokenizer {
public:
    bool init(const GgufFile& f, std::string* err);

    const std::vector<std::string>& vocab() const { return id_to_token_; }
    int32_t n_vocab() const { return (int32_t)id_to_token_.size(); }

    std::vector<int32_t> encode(const std::string& text, bool add_special) const;
    // 返回该 token 对应的原始字节（可能不是完整 UTF-8）
    std::string decode_piece(int32_t id) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t token_id(const std::string& piece) const;  // -1 表示不存在
    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }
    int32_t im_start_id() const { return im_start_; }
    int32_t im_end_id() const { return im_end_; }
    // 该 token 是否是特殊 token（<|im_start|> 之类），不应输出给用户
    bool    is_special(int32_t id) const {
        return id >= 0 && id < (int32_t)special_flag_.size() && special_flag_[(size_t)id] != 0;
    }
    bool    is_chatml() const { return im_start_ >= 0 && im_end_ >= 0; }
    const std::string& chat_template() const { return chat_template_; }

private:
    void   build_byte_map();
    void   pretokenize(const std::string& text, std::vector<std::string>* out) const;
    std::vector<int32_t> bpe(const std::string& piece) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::map<std::pair<std::string, std::string>, int> merges_;
    std::vector<int32_t> specials_;
    std::vector<uint8_t> special_flag_;
    std::string byte_to_char_[256];
    std::unordered_map<std::string, uint8_t> char_to_byte_;
    std::string chat_template_;
    int32_t bos_ = -1, eos_ = -1, im_start_ = -1, im_end_ = -1;
    bool    add_bos_ = false;
    bool    is_spm_ = false;
};

// ---------------------------------------------------------------- 模型
// 层的算子种类。LFM3 把三种子层交错排布（conv / 滑窗注意力 / 全局注意力），
// llama 系全部是 LK_ATTN_FULL。
enum LayerKind : uint8_t {
    LK_ATTN_FULL = 0,   // 全局因果注意力
    LK_ATTN_SWA  = 1,   // 滑窗注意力（只看最近 n_swa 个位置）
    LK_CONV      = 2,   // LFM3 的 ShortConv：门控深度可分离因果卷积
};

struct Layer {
    LayerKind kind = LK_ATTN_FULL;

    // ---- 注意力子层
    TensorInfo attn_norm, attn_q, attn_k, attn_v, attn_output;
    TensorInfo attn_q_b, attn_k_b, attn_v_b;   // 可选 bias
    TensorInfo attn_q_norm, attn_k_norm;       // LFM3: 每 head 的 QK RMSNorm
    TensorInfo attn_gate;                      // LFM3: 逐 head 的 sigmoid 输出门
    bool has_q_bias = false, has_k_bias = false, has_v_bias = false;
    bool has_qk_norm = false, has_attn_gate = false;

    // ---- ShortConv 子层（仅 LK_CONV）
    TensorInfo conv_in, conv_kernel, conv_out;

    // ---- FFN 子层
    TensorInfo ffn_norm, ffn_gate, ffn_up, ffn_down;

    // ---- LFM3 的 sandwich norm：子层输出再过一次 norm 才加残差
    TensorInfo post_attn_norm, post_ffn_norm;
    bool has_sandwich = false;
};

struct ModelConfig {
    std::string arch = "llama";
    int32_t n_vocab = 0, n_ctx_train = 0, n_embd = 0, n_layer = 0;
    int32_t n_head = 0, n_head_kv = 0, n_ff = 0, n_rot = 0;
    float   rms_eps = 1e-5f;
    float   rope_theta = 10000.f;
    float   rope_scale = 1.f;
    bool    tied_embd = false;
    int32_t head_dim = 0;
    int32_t n_swa = 0;              // 滑窗宽度，0 表示没有滑窗层

    // ---- LFM3 专属
    int32_t rope_angles = 0;        // partial RoPE：真正旋转的角度数，0 = 全部
    float   rope_theta_swa = 0.f;   // 滑窗层用另一套 base（1e4 vs 1e6）
    int32_t conv_l_cache = 0;       // ShortConv 的卷积核宽度
    bool    sandwich_norm = false;
    int     n_conv = 0;             // conv 层个数
};

class Model {
public:
    bool load(const char* path, int n_ctx, bool use_gpu, std::string* err);
    void unload();

    const ModelConfig& cfg() const { return cfg_; }
    const Tokenizer&   tok() const { return tok_; }
    const GgufFile&    file() const { return file_; }
    bool               loaded() const { return loaded_; }

    // 一次前向：给定最后一个 token 与已有 KV，输出 logits（n_vocab）
    bool forward(int32_t token, int32_t pos, float* logits, std::string* err);
    // 批量前向（prefill）：tokens[0..n) 连续喂入，pos0 是首 token 的位置。
    // 只有最后一个 token 的 logits 有效。n==1 时等价于 forward()。
    bool forward_batch(const int32_t* tokens, int n, int32_t pos0, float* logits, std::string* err);
    // 引擎实际使用的批大小（GPU 不可用时为 1）
    int  batch_size() const;
    void reset_kv();

    // 供 UI / 统计使用
    uint64_t n_params() const { return n_params_; }
    int      n_ctx_limit() const { return n_ctx_; }
    const char* quant_desc() const { return quant_desc_.c_str(); }
    bool gpu_active() const { return gpu_active_; }
    const char* gpu_note() const { return gpu_note_.c_str(); }
    bool arch_is_lfm3() const { return cfg_.arch == "lfm3" || cfg_.arch == "lfm2"; }

private:
    bool load_weights(std::string* err);
    bool enable_gpu(std::string* err);
    bool grow_kv(int need);
    void rmsnorm(float* out, const float* x, const TensorInfo& w, int n) const;
    // 每 head 独立归一化（LFM3 的 QK norm），v 是 [n_head][hd] 的连续布局
    void rmsnorm_heads(float* v, const TensorInfo& w, int hd, int n_head) const;
    void matvec(float* out, const float* x, const TensorInfo& w, int n_out, int n_in) const;
    void rope(float* q, float* k, int n_head, int n_head_kv, int head_dim, int pos,
              float theta) const;
    // LFM3 ShortConv 子层：结果写进 attn_out_
    void shortconv(const Layer& L, int slot, const float* xin);
    void clear_conv_state();

    GgufFile file_;
    Tokenizer tok_;
    ModelConfig cfg_;
    bool loaded_ = false;
    mutable bool gpu_active_ = false;   // GPU 出错时 matvec 会就地降级，故 mutable
    // CPU 参考路径是否强制用纯 f32 点积（关掉激活 Q8 量化）。
    // GPU 校验时必须打开：Q8 激活本身就有 ~1e-2 的相对误差，
    // 不关掉的话对拍测的是 CPU 自己的近似，量不出 GPU 的问题。
    mutable bool f32_ref_ = false;
    // 整层前向是否由 Metal 融合内核接管。LFM3 还没有对应的融合内核，
    // 但逐算子的 matvec 仍然走 GPU，所以这两个标志必须分开。
    bool gpu_fused_ = false;
    std::string gpu_note_;              // GPU 状态说明（供 UI / 日志展示）
    int  n_ctx_ = 0;    // 可用上下文上限
    int  kv_cap_ = 0;   // KV cache 已分配槽位

    TensorInfo tok_embd_, out_norm_, out_w_;
    std::vector<Layer> layers_;

    // KV cache: [n_layer][n_ctx][n_head_kv * head_dim]
    std::vector<float> k_cache_, v_cache_;
    int kv_len_ = 0;

    std::vector<float> x_, xb_, xb2_, q_, k_, v_, attn_out_, gate_, up_, ffn_out_;
    std::vector<float> logits_;
    std::vector<float> xbatch_;        // 批量 embedding 暂存 [batch][n_embd]
    mutable std::vector<uint8_t> aq_;  // 量化激活缓冲（matvec 复用）

    // ---- LFM3 ShortConv 的时序状态
    std::vector<float> conv_bc_;       // in_proj 输出 B/C/x 暂存，3*n_embd
    std::vector<float> conv_h_;        // hidden = B*x，n_embd
    std::vector<float> conv_state_;    // [n_conv][l_cache-1][n_embd]，环形历史
    std::vector<int>   conv_slot_;     // 层号 → conv_state_ 槽位，-1 表示非 conv
    std::vector<float> ffn_down_out_;  // ffn down 输出，n_embd

    uint64_t n_params_ = 0;
    std::string quant_desc_;
};

}  // namespace dubhe
