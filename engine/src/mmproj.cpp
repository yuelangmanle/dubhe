// mmproj.cpp — LFM3-VL 视觉塔的前向实现（CPU）
//
// 这个文件做三件事：解析 mmproj 的元数据与张量表、把一张图预处理成 patch 序列、
// 跑完视觉塔 + 投影器得到可以直接塞进语言模型的嵌入。
//
// 结构（来自 clip.vision.* 与 modeling_lfm3_vl.py）：
//   patchify(1536→1024) → RMSNorm → axial 2D RoPE
//   → 14 层，conv / attention 交替（7 + 7）
//       conv 层：RMSNorm → in_proj(1024→3072) 切三段 (B, C, x)
//                → 门控 gated = B*x → 2D 深度卷积(3×3) → 输出门 C*conv → out_proj
//       attn 层：RMSNorm → QKV(16/8 头 GQA) → 每 head QK RMSNorm → 双向 attention
//   每层都是 sandwich：x += op(norm1(x)); x += ffn(norm2(x))
//   → 投影器：LayerNorm → 每 merge² 个 token 拼成一个(4096) → Linear(4096→4096)
//             → GELU → Linear(4096→2048)
//
// 为什么先在 CPU 上写：视觉塔只跑一次（一张图），不是 hot path；而正确性的坑
// 极多（轴向 RoPE 的 cos/sin 拼接方式、merge-block 的 token 顺序、门控卷积的
// 栅格重排）。CPU 版本容易逐层对拍，跑通之后再搬 Metal 才不容易白写。
#include "mmproj.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>

namespace dubhe {
namespace {

constexpr int kThreads = 8;

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
// ACT2FN["gelu"] 是 erf 版，不是 tanh 近似。两者差 1e-3 量级，
// 在投影器最后一层会被放大成可见的输出差异，所以必须用 erf。
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.70710678f)); }

// Y[tok][o] = sum_i W[o][i] * X[tok][i]
//
// x_stride 是 X 相邻两行之间的间隔，0 表示等于 n_in。patchify 必须显式传
// patch_dim（1536）：那里一个 patch 只取前 768 维当一个时间片，如果按 n_in
// 跨步，第 t 个 patch 会读到 t×768 的位置，实际跨进了相邻 patch —— 纯白区域
// 看不出来（内容相同），只有带细节的图才炸，极难查。
//
// 按输出行并行、token 在行内循环：一行权重只从内存读一次就复用于所有 token，
// 而 X 是热数据。早期版本写成"每个 token 调一次 matvec_t"，结果每个 matmul
// 都要起 8 个线程（14 层 × 6 个 matmul × N 个 token），线程创建的开销比计算
// 本身还大 —— 64 个 token 的图要跑 6.5 秒。
void matmul_t(const float* W, int64_t n_in, int64_t n_out, const float* X, int64_t n_tok,
              float* Y, int64_t x_stride = 0) {
    const int64_t xs = x_stride > 0 ? x_stride : n_in;
    auto body = [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; o++) {
            const float* row = W + o * n_in;
            float* yp = Y + o;
            for (int64_t t = 0; t < n_tok; t++, yp += n_out) {
                const float* x = X + t * xs;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                int64_t i = 0;
                for (; i + 4 <= n_in; i += 4) {
                    s0 += row[i] * x[i];
                    s1 += row[i + 1] * x[i + 1];
                    s2 += row[i + 2] * x[i + 2];
                    s3 += row[i + 3] * x[i + 3];
                }
                float s = s0 + s1 + s2 + s3;
                for (; i < n_in; i++) s += row[i] * x[i];
                *yp = s;
            }
        }
    };
    // 工作量太小就别付线程启动的钱
    if (n_out * n_tok < 64 * 1024) { body(0, n_out); return; }
    const int64_t chunk = (n_out + kThreads - 1) / kThreads;
    std::vector<std::thread> th;
    th.reserve(kThreads);
    for (int t = 0; t < kThreads; t++) {
        const int64_t o0 = t * chunk, o1 = std::min(n_out, o0 + chunk);
        if (o0 < o1) th.emplace_back(body, o0, o1);
    }
    for (auto& t : th) t.join();
}

void rmsnorm(float* x, int64_t n, const float* w, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
    for (int64_t i = 0; i < n; i++) x[i] = x[i] * r * w[i];
}

void layernorm(float* x, int64_t n, const float* w, const float* b, float eps) {
    double s = 0.0, ss = 0.0;
    for (int64_t i = 0; i < n; i++) { s += x[i]; ss += (double)x[i] * x[i]; }
    const float mean = (float)(s / (double)n);
    const float var  = (float)(ss / (double)n) - mean * mean;
    const float r = 1.0f / std::sqrt(var + eps);
    for (int64_t i = 0; i < n; i++) x[i] = (x[i] - mean) * r * w[i] + b[i];
}

// 把整个张量反量化到 f32 缓冲。视觉塔只跑一次，不值得为它写分块反量化。
bool deq_all(const MmTensor& t, std::vector<float>& dst, std::string* err) {
    if (!t.ok()) { if (err) *err = "张量缺失"; return false; }
    dst.resize((size_t)t.n);
    dequant_row(t.type, t.data, dst.data(), t.n);
    return true;
}

}  // namespace

// ---------------------------------------------------------------- 加载

bool MmProj::load(const char* path, std::string* err) {
    if (!f_.open(path, err)) return false;
    auto& c = cfg_;

    if (f_.get_str("general.type") != "mmproj") {
        if (err) *err = "不是 mmproj 文件（general.type != mmproj）";
        return false;
    }
    c.projector   = f_.get_str("clip.vision.projector_type");
    c.image_size  = (int)f_.get_u32("clip.vision.image_size");
    c.patch       = (int)f_.get_u32("clip.vision.patch_size");
    c.merge       = (int)f_.get_u32("clip.vision.spatial_merge_size", 1);
    c.n_layer     = (int)f_.get_u32("clip.vision.block_count");
    c.hidden      = (int)f_.get_u32("clip.vision.embedding_length");
    c.ffn         = (int)f_.get_u32("clip.vision.feed_forward_length");
    c.n_head      = (int)f_.get_u32("clip.vision.attention.head_count");
    c.n_head_kv   = (int)f_.get_u32("clip.vision.attention.head_count_kv");
    c.norm_eps    = f_.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
    c.out_dim     = (int)f_.get_u32("clip.vision.projection_dim");
    c.min_pixels  = (int)f_.get_u32("clip.vision.image_min_pixels", 65536);
    c.max_pixels  = (int)f_.get_u32("clip.vision.image_max_pixels", 16777216);
    if (c.image_size <= 0 || c.patch <= 0 || c.hidden <= 0 || c.n_layer <= 0) {
        if (err) *err = "clip.vision.* 元数据不完整";
        return false;
    }
    if (c.merge <= 0) c.merge = 1;
    if (c.n_head <= 0) { if (err) *err = "head_count 缺失"; return false; }
    c.head_dim = c.hidden / c.n_head;

    // 归一化参数（默认 0.5/0.5 → x/127.5-1）
    if (const GgufValue* v = f_.get("clip.vision.image_mean"); v && v->arr_f32.size() >= 3)
        for (int i = 0; i < 3; i++) c.mean[i] = v->arr_f32[(size_t)i];
    if (const GgufValue* v = f_.get("clip.vision.image_std"); v && v->arr_f32.size() >= 3)
        for (int i = 0; i < 3; i++) c.stdv[i] = v->arr_f32[(size_t)i];

    auto bind = [&](const std::string& n) {
        MmTensor t;
        if (const TensorInfo* ti = f_.find_tensor(n)) {
            t.data = f_.tensor_data(*ti);
            t.type = ti->type;
            t.ne0  = ti->ne[0];
            int64_t tot = 1;
            for (int i = 0; i < ti->n_dims; i++) tot *= ti->ne[i];
            t.n = tot;
        }
        return t;
    };

    // patchifier：llama.cpp 按时间片拆成 weight / weight.1 两块
    patch_w[0] = bind("v.patch_embd.weight");
    patch_w[1] = bind("v.patch_embd.weight.1");
    patch_two_slices_ = patch_w[1].ok();
    if (!patch_two_slices_) patch_w[1] = patch_w[0];   // 只有一块时按重复处理
    emb_norm = bind("v.embedding_norm.weight");

    mm_norm_w = bind("mm.input_norm.weight");
    mm_norm_b = bind("mm.input_norm.bias");
    mm0_w = bind("mm.0.weight"); mm0_b = bind("mm.0.bias");
    mm2_w = bind("mm.2.weight"); mm2_b = bind("mm.2.bias");
    if (mm0_w.ok()) c.proj_hidden = (int)mm0_w.ne0;
    if (c.out_dim <= 0 && mm2_w.ok()) c.out_dim = (int)mm2_w.ne0;

    // rope：轴向 2D，base 从 GGUF 拿不到（没存），LFM3-VL 视觉塔固定 1e6。
    c.rope_theta = 1e6f;

    layers_.assign((size_t)c.n_layer, MmProjLayer{});
    c.is_attn.assign((size_t)c.n_layer, 0);
    for (int i = 0; i < c.n_layer; i++) {
        const std::string p = "v.blk." + std::to_string(i) + ".";
        auto& L = layers_[(size_t)i];
        L.ln1 = bind(p + "ln1.weight");
        L.ln2 = bind(p + "ln2.weight");
        L.ffn_gate = bind(p + "ffn_gate.weight");
        L.ffn_up   = bind(p + "ffn_up.weight");
        L.ffn_down = bind(p + "ffn_down.weight");
        L.q = bind(p + "attn_q.weight");
        L.k = bind(p + "attn_k.weight");
        L.v = bind(p + "attn_v.weight");
        L.o = bind(p + "attn_out.weight");
        L.q_norm = bind(p + "attn_q_norm.weight");
        L.k_norm = bind(p + "attn_k_norm.weight");
        L.conv_in      = bind(p + "conv2d_in_proj.weight");
        L.conv_out     = bind(p + "conv2d_out_proj.weight");
        L.conv_spatial = bind(p + "conv2d_spatial.weight");
        // 层类型由实际存在的张量决定，不靠外部配置猜
        if (L.q.ok()) c.is_attn[(size_t)i] = 1;
        else if (L.conv_in.ok()) c.is_attn[(size_t)i] = 0;
        else { if (err) *err = "第 " + std::to_string(i) + " 层既无 attn_q 也无 conv2d_in_proj"; return false; }
    }

    loaded_ = true;
    return true;
}

void MmProj::describe(std::string* out) const {
    const auto& c = cfg_;
    char buf[640];
    std::snprintf(buf, sizeof buf,
                  "mmproj: %s  patch %d  merge %d  默认 %d px\n"
                  "  %d 层  hidden %d  ffn %d  head %d/%d  head_dim %d\n"
                  "  输出 %d 维（文本 hidden），投影中间层 %d，归一化 %.3f/%.3f\n"
                  "  像素范围 [%d, %d]\n",
                  c.projector.c_str(), c.patch, c.merge, c.image_size,
                  c.n_layer, c.hidden, c.ffn, c.n_head, c.n_head_kv, c.head_dim,
                  c.out_dim, c.proj_hidden, c.mean[0], c.stdv[0],
                  c.min_pixels, c.max_pixels);
    out->append(buf);
    out->append("  层类型: ");
    for (int i = 0; i < c.n_layer; i++) out->append(c.is_attn[(size_t)i] ? "A" : "C");
    out->append("\n");
}

// ---------------------------------------------------------------- 预处理

// 与 HF 的 Lfm3VlImageProcessor.smart_resize 等价：两边对齐到 patch×merge，
// 总像素落在 [min_pixels, max_pixels]，尽量保持长宽比。
void MmProj::smart_resize(int h, int w, int* oh, int* ow) const {
    const int factor = cfg_.patch * cfg_.merge;
    const double mn = (double)cfg_.min_pixels, mx = (double)cfg_.max_pixels;
    int hb = (int)(std::llround((double)h / factor) * factor);
    int wb = (int)(std::llround((double)w / factor) * factor);
    hb = std::max(hb, factor);
    wb = std::max(wb, factor);
    if ((double)hb * wb > mx) {
        const double beta = std::sqrt((double)h * w / mx);
        hb = std::max(factor, (int)(std::floor(h / beta / factor) * factor));
        wb = std::max(factor, (int)(std::floor(w / beta / factor) * factor));
    } else if ((double)hb * wb < mn) {
        const double beta = std::sqrt(mn / ((double)h * w));
        hb = (int)(std::ceil(h * beta / factor) * factor);
        wb = (int)(std::ceil(w * beta / factor) * factor);
    }
    *oh = hb;
    *ow = wb;
}

// 把 HWC 的 uint8 RGB 摊成 patch 序列。
//
// 两个容易错的点：
//  1) token 顺序是 **merge-block 序**，不是栅格序：每 2×2 个 patch 合成一个语言
//     token，4 个 sub-patch 依次排在一起。参考实现的 reshape(-1, merge²·C) 就是
//     这个含义。
//  2) patch 内元素顺序是 [c][ph][pw]（通道最外、pw 最内）。这是 GGUF 里
//     v.patch_embd.weight 的 ne=(16,16,3) 决定的：ne0 最快 → 下标 = pw + 16·ph + 256·c。
bool MmProj::pack_pixels(const uint8_t* rgb, int w, int h, float* out) const {
    const auto& c = cfg_;
    if (c.patch <= 0 || w % c.patch || h % c.patch) return false;
    const int P = c.patch, M = c.merge;
    const int gh = h / P, gw = w / P;
    if (gh % M || gw % M) return false;
    const int gh_m = gh / M, gw_m = gw / M;
    const int slice = 3 * P * P;           // 一个时间片的长度
    const float scale[3] = {1.0f / (c.stdv[0] * 255.0f), 1.0f / (c.stdv[1] * 255.0f),
                            1.0f / (c.stdv[2] * 255.0f)};
    const float bias[3] = {c.mean[0] / c.stdv[0], c.mean[1] / c.stdv[1],
                           c.mean[2] / c.stdv[2]};

    int64_t p = 0;   // 输出 patch 序号（merge-block 序）
    for (int by = 0; by < gh_m; by++)
        for (int bx = 0; bx < gw_m; bx++)
            for (int dy = 0; dy < M; dy++)
                for (int dx = 0; dx < M; dx++) {
                    const int py = (by * M + dy) * P;   // 该 patch 的左上角像素
                    const int px = (bx * M + dx) * P;
                    float* dst = out + p * c.patch_dim();
                    // 静止图像：两个时间片填同一份内容（与参考实现的复制行为一致）
                    for (int t = 0; t < c.temporal; t++) {
                        float* s = dst + (int64_t)t * slice;
                        for (int ch = 0; ch < 3; ch++) {
                            for (int yy = 0; yy < P; yy++) {
                                const uint8_t* rowp = rgb + ((size_t)(py + yy) * w + px) * 3 + ch;
                                float* o = s + (int64_t)ch * P * P + (int64_t)yy * P;
                                for (int xx = 0; xx < P; xx++)
                                    o[xx] = (float)rowp[(size_t)xx * 3] * scale[ch] - bias[ch];
                            }
                        }
                    }
                    p++;
                }
    return true;
}

// ---------------------------------------------------------------- 前向

bool MmProj::encode(const float* pixels, int gh, int gw, float* out, std::string* err) const {
    if (!loaded_) { if (err) *err = "mmproj 未加载"; return false; }
    const auto& c = cfg_;
    const int M = c.merge;
    if (gh % M || gw % M) { if (err) *err = "patch 网格必须是 merge 的倍数"; return false; }

    const int64_t N = (int64_t)gh * gw;    // patch 数
    const int64_t D = c.hidden;
    const int     H = c.n_head, KV = c.n_head_kv, HD = c.head_dim;
    const int     P = c.patch;
    const int64_t gw_m = gw / M;   // 每行多少个 merge block
    const int64_t n_tok = N / (M * M);

    std::vector<float> x((size_t)(N * D));
    std::vector<float> wbuf, y((size_t)(N * std::max<int64_t>(D, c.ffn)));
    std::vector<float> xn;

    // token 序号 → (blk_y, blk_x, sub)。merge-block 序，sub = dy·M + dx。
    auto coords = [&](int64_t n, int* by, int* bx, int* sub) {
        const int64_t blk = n / (M * M);
        *by  = (int)(blk / gw_m);
        *bx  = (int)(blk % gw_m);
        *sub = (int)(n % (M * M));
    };

    // ---- patchify：y = W0·x0 + W1·x1（每块 3·P·P 维）
    // 所有 patch 一起做一次 matmul，不要逐 patch 调 matvec —— 后者每次都要起
    // 8 个线程，256 个 patch 就是 4096 次线程创建，实测占了 70% 的墙钟时间。
    {
        const int64_t slice = (int64_t)3 * P * P;
        std::vector<float> w0, w1;
        if (!deq_all(patch_w[0], w0, err)) return false;
        const int64_t pw = w0.size() / D;   // 每行宽度 = 一个时间片的元素数
        const int64_t pstride = c.patch_dim();   // 一个 patch 的总长度（两个时间片）
        matmul_t(w0.data(), pw, D, pixels, N, x.data(), pstride);
        if (patch_two_slices_ && c.temporal > 1 && pw == slice) {
            if (!deq_all(patch_w[1], w1, err)) return false;
            std::vector<float> tmp((size_t)(N * D));
            matmul_t(w1.data(), pw, D, pixels + slice, N, tmp.data(), pstride);
            for (int64_t i = 0; i < N * D; i++) x[(size_t)i] += tmp[(size_t)i];
        }
    }

    // ---- embedding_norm
    {
        std::vector<float> ew;
        if (!deq_all(emb_norm, ew, err)) return false;
        for (int64_t k = 0; k < N; k++) rmsnorm(x.data() + k * D, D, ew.data(), c.norm_eps);
    }
    // 对齐 HF 的 hidden_states[0]（嵌入输出，此时还没上 RoPE）
    if (on_layer) on_layer(-1, x.data(), N);

    // ---- 轴向 2D RoPE：每 token HD 维 = [h(f) w(f) h(f) w(f)]，f = HD/4 个频率
    //
    // 坐标是 **patch 行/列号**，不是 merge-block 号。transformers 的
    // get_vision_position_ids 里 hpos_ids.reshape(h//M, M, w//M, M).transpose(1,2)
    // 展平后元素值仍是 by*M+dy（patch 下标）。用 block 号会让整张图被当成缩小
    // 一倍的网格 —— 不报错、不 NaN，只是模型对空间位置的感知整体错位。
    const int nf = HD / 4;
    std::vector<float> invf((size_t)nf);
    for (int i = 0; i < nf; i++)
        invf[(size_t)i] = 1.0f / std::pow(c.rope_theta, (float)(2 * i) / (float)(HD / 2));
    std::vector<float> cosb((size_t)(N * HD)), sinb((size_t)(N * HD));
    for (int64_t n = 0; n < N; n++) {
        int by, bx, sub;
        coords(n, &by, &bx, &sub);
        float* cb = cosb.data() + n * HD;
        float* sb = sinb.data() + n * HD;
        for (int i = 0; i < nf; i++) {
            const float ah = (float)(by * M + sub / M) * invf[(size_t)i];
            const float aw = (float)(bx * M + sub % M) * invf[(size_t)i];
            const float ch = std::cos(ah), sh = std::sin(ah);
            const float cw = std::cos(aw), sw = std::sin(aw);
            cb[i] = ch;          cb[nf + i] = cw;
            cb[2 * nf + i] = ch; cb[3 * nf + i] = cw;
            sb[i] = sh;          sb[nf + i] = sw;
            sb[2 * nf + i] = sh; sb[3 * nf + i] = sw;
        }
    }

    std::vector<float> q((size_t)(N * D)), k((size_t)(N * KV * HD)), v((size_t)(N * KV * HD));
    const int64_t qdim = (int64_t)H * HD, kdim = (int64_t)KV * HD;
    const int half = HD / 2;

    for (int li = 0; li < c.n_layer; li++) {
        const auto& L = layers_[(size_t)li];

        // ---------------- sandwich 第一段：x += op(norm1(x))
        xn.assign(x.begin(), x.end());
        if (!deq_all(L.ln1, wbuf, err)) return false;
        for (int64_t n = 0; n < N; n++) rmsnorm(xn.data() + n * D, D, wbuf.data(), c.norm_eps);

        if (c.is_attn[(size_t)li]) {
            std::vector<float> wq, wk, wv, wo, wqn, wkn;
            if (!deq_all(L.q, wq, err) || !deq_all(L.k, wk, err) || !deq_all(L.v, wv, err) ||
                !deq_all(L.o, wo, err) || !deq_all(L.q_norm, wqn, err) ||
                !deq_all(L.k_norm, wkn, err)) return false;

            matmul_t(wq.data(), D, qdim, xn.data(), N, q.data());
            matmul_t(wk.data(), D, kdim, xn.data(), N, k.data());
            matmul_t(wv.data(), D, kdim, xn.data(), N, v.data());

            // 每 head 的 QK RMSNorm，再套 RoPE（rotate-half；cos/sin 后一半是前一半的副本）
            for (int64_t n = 0; n < N; n++) {
                const float* cb = cosb.data() + n * HD;
                const float* sb = sinb.data() + n * HD;
                for (int hh = 0; hh < H; hh++) {
                    float* qh = q.data() + n * qdim + hh * HD;
                    rmsnorm(qh, HD, wqn.data(), c.norm_eps);
                    for (int d = 0; d < half; d++) {
                        const float a = qh[d], b = qh[d + half];
                        qh[d]        = a * cb[d] - b * sb[d];
                        qh[d + half] = b * cb[d] + a * sb[d];
                    }
                }
                for (int hh = 0; hh < KV; hh++) {
                    float* kh = k.data() + n * kdim + hh * HD;
                    rmsnorm(kh, HD, wkn.data(), c.norm_eps);
                    for (int d = 0; d < half; d++) {
                        const float a = kh[d], b = kh[d + half];
                        kh[d]        = a * cb[d] - b * sb[d];
                        kh[d + half] = b * cb[d] + a * sb[d];
                    }
                }
            }

            // 双向 GQA。N 能到几千，这是视觉塔的主要开销，所以做两件事：
            //  1) 把 K/V 转成 [kvh][j][HD] —— 原来按 j 跨 kdim 跳读，每次跨 512
            //     个 float，cache 基本全废；
            //  2) 按 head 并行，每个线程用自己的 softmax 暂存。
            const float scale = 1.0f / std::sqrt((float)HD);
            const int group = H / KV;
            std::vector<float> kt((size_t)(KV * N * HD)), vt((size_t)(KV * N * HD));
            for (int kvh = 0; kvh < KV; kvh++)
                for (int64_t j = 0; j < N; j++) {
                    std::memcpy(kt.data() + ((size_t)kvh * N + j) * HD,
                                k.data() + j * kdim + kvh * HD, sizeof(float) * HD);
                    std::memcpy(vt.data() + ((size_t)kvh * N + j) * HD,
                                v.data() + j * kdim + kvh * HD, sizeof(float) * HD);
                }
            std::vector<float> o((size_t)(N * qdim));
            auto head_body = [&](int h0, int h1) {
                std::vector<float> sc((size_t)N);
                for (int hh = h0; hh < h1; hh++) {
                    const int kvh = hh / group;
                    const float* kb = kt.data() + (size_t)kvh * N * HD;
                    const float* vb = vt.data() + (size_t)kvh * N * HD;
                    for (int64_t i = 0; i < N; i++) {
                        const float* qi = q.data() + i * qdim + hh * HD;
                        float mx = -1e30f;
                        for (int64_t j = 0; j < N; j++) {
                            const float* kj = kb + j * HD;
                            float s = 0.0f;
                            for (int d = 0; d < HD; d++) s += qi[d] * kj[d];
                            s *= scale;
                            sc[(size_t)j] = s;
                            if (s > mx) mx = s;
                        }
                        float sum = 0.0f;
                        for (int64_t j = 0; j < N; j++) {
                            const float e = std::exp(sc[(size_t)j] - mx);
                            sc[(size_t)j] = e;
                            sum += e;
                        }
                        const float inv = 1.0f / sum;
                        float* oi = o.data() + i * qdim + hh * HD;
                        for (int d = 0; d < HD; d++) oi[d] = 0.0f;
                        for (int64_t j = 0; j < N; j++) {
                            const float a = sc[(size_t)j] * inv;
                            const float* vj = vb + j * HD;
                            for (int d = 0; d < HD; d++) oi[d] += a * vj[d];
                        }
                    }
                }
            };
            if (H >= kThreads) {
                std::vector<std::thread> th;
                const int hc = (H + kThreads - 1) / kThreads;
                for (int t = 0; t < kThreads; t++) {
                    const int h0 = t * hc, h1 = std::min(H, h0 + hc);
                    if (h0 < h1) th.emplace_back(head_body, h0, h1);
                }
                for (auto& t : th) t.join();
            } else {
                head_body(0, H);
            }
            matmul_t(wo.data(), qdim, D, o.data(), N, y.data());
        } else {
            // 门控 2D 深度卷积
            std::vector<float> wi, wc, wout;
            if (!deq_all(L.conv_in, wi, err) || !deq_all(L.conv_out, wout, err) ||
                !deq_all(L.conv_spatial, wc, err)) return false;

            const int64_t c3 = 3 * D;
            std::vector<float> g3((size_t)(N * c3));
            matmul_t(wi.data(), D, c3, xn.data(), N, g3.data());

            // 栅格化：[token][merge²][C] → [C][gh][gw]
            // 注意 in_proj 的输出被切成三段 (B, C, x)：
            //   门控用第 0 段 × 第 2 段，卷完再乘第 1 段 —— 中间那段是**输出门**。
            //   这里曾经漏掉那一段：数值不会崩、也不会报错，只是整体质量悄悄变差。
            // grid / conv 都是 D×N 个 float（1600 patch 时各 6.5 MB），会被完整
            // 覆盖，所以用不初始化的裸数组 —— std::vector 的零初始化在这里是纯浪费。
            auto grid = std::make_unique<float[]>((size_t)(D * N));
            for (int64_t n = 0; n < N; n++) {
                int by, bx, sub;
                coords(n, &by, &bx, &sub);
                const int yy = by * M + sub / M;
                const int xx = bx * M + sub % M;
                for (int ch = 0; ch < D; ch++) {
                    const float B = g3[(size_t)(n * c3 + ch)];
                    const float X = g3[(size_t)(n * c3 + 2 * D + ch)];
                    grid[(size_t)(ch * N + yy * gw + xx)] = B * X;
                }
            }

            // 3×3 depthwise，padding 1
            const int K = 3, pad = 1;
            auto conv = std::make_unique<float[]>((size_t)(D * N));
            for (int ch = 0; ch < D; ch++) {
                const float* kern = wc.data() + (size_t)ch * K * K;
                const float* src = grid.get() + (size_t)ch * N;
                float* dst = conv.get() + (size_t)ch * N;
                for (int yy = 0; yy < gh; yy++)
                    for (int xx = 0; xx < gw; xx++) {
                        float s = 0.0f;
                        for (int ky = 0; ky < K; ky++) {
                            const int sy = yy + ky - pad;
                            if (sy < 0 || sy >= gh) continue;
                            for (int kx = 0; kx < K; kx++) {
                                const int sx = xx + kx - pad;
                                if (sx < 0 || sx >= gw) continue;
                                s += kern[ky * K + kx] * src[sy * gw + sx];
                            }
                        }
                        dst[yy * gw + xx] = s;
                    }
            }

            // 摊回 merge-block 序，乘**输出门**（in_proj 的第 2 段），再过 out_proj。
            // HF 原式：return self.out_proj(C * conv_output)，C 就是中间那段。
            std::vector<float> back((size_t)(N * D));
            for (int64_t n = 0; n < N; n++) {
                int by, bx, sub;
                coords(n, &by, &bx, &sub);
                const int yy = by * M + sub / M;
                const int xx = bx * M + sub % M;
                for (int ch = 0; ch < D; ch++)
                    back[(size_t)(n * D + ch)] = conv[(size_t)(ch * N + yy * gw + xx)] *
                                                 g3[(size_t)(n * c3 + D + ch)];
            }
            matmul_t(wout.data(), D, D, back.data(), N, y.data());
        }

        for (int64_t i = 0; i < N * D; i++) x[(size_t)i] += y[(size_t)i];

        // ---------------- sandwich 第二段：x += ffn(norm2(x))
        xn.assign(x.begin(), x.end());
        if (!deq_all(L.ln2, wbuf, err)) return false;
        for (int64_t n = 0; n < N; n++) rmsnorm(xn.data() + n * D, D, wbuf.data(), c.norm_eps);

        std::vector<float> wg, wu, wd;
        if (!deq_all(L.ffn_gate, wg, err) || !deq_all(L.ffn_up, wu, err) ||
            !deq_all(L.ffn_down, wd, err)) return false;
        std::vector<float> fg((size_t)(N * c.ffn)), fu((size_t)(N * c.ffn));
        matmul_t(wg.data(), D, c.ffn, xn.data(), N, fg.data());
        matmul_t(wu.data(), D, c.ffn, xn.data(), N, fu.data());
        for (int64_t i = 0; i < N * c.ffn; i++) fg[(size_t)i] = silu(fg[(size_t)i]) * fu[(size_t)i];
        matmul_t(wd.data(), c.ffn, D, fg.data(), N, y.data());
        for (int64_t i = 0; i < N * D; i++) x[(size_t)i] += y[(size_t)i];

        if (on_layer) on_layer(li, x.data(), N);
    }

    // ---- 投影器：LayerNorm → 每 merge² 个 token 拼起来 → Linear → GELU → Linear
    {
        std::vector<float> nw, nb, w0, b0, w2, b2;
        if (!deq_all(mm_norm_w, nw, err) || !deq_all(mm_norm_b, nb, err) ||
            !deq_all(mm0_w, w0, err) || !deq_all(mm0_b, b0, err) ||
            !deq_all(mm2_w, w2, err) || !deq_all(mm2_b, b2, err)) return false;

        for (int64_t n = 0; n < N; n++)
            layernorm(x.data() + n * D, D, nw.data(), nb.data(), 1e-6f);

        const int64_t din = D * M * M;              // 4096
        const int64_t mid = c.proj_hidden;          // 4096
        std::vector<float> hh((size_t)(n_tok * mid));
        matmul_t(w0.data(), din, mid, x.data(), n_tok, hh.data());
        for (int64_t n = 0; n < n_tok; n++)
            for (int64_t o = 0; o < mid; o++)
                hh[(size_t)(n * mid + o)] = gelu(hh[(size_t)(n * mid + o)] + b0[(size_t)o]);
        matmul_t(w2.data(), mid, c.out_dim, hh.data(), n_tok, out);
        for (int64_t n = 0; n < n_tok; n++)
            for (int64_t o = 0; o < c.out_dim; o++) out[n * c.out_dim + o] += b2[(size_t)o];
    }
    return true;
}

}  // namespace dubhe
