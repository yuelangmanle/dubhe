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
#include "mm_math.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace dubhe {
using namespace mm;   // 共用的量化 matmul / NEON 点积 / norm，见 mm_math.h

// ---------------------------------------------------------------- 加载

MmProj::~MmProj() { vit_gpu_release(); }

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

    // 把 mmap 登记给视觉塔的 GPU matmul。很轻（不建设备、不编着色器），
    // 真正的初始化推迟到第一次 matmul —— 纯文本场景不该为视觉塔付这笔钱。
    vit_bind(f_.mapping(), f_.file_size());

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
    return pack_pixels2(rgb, rgb, w, h, out);
}

bool MmProj::pack_pixels2(const uint8_t* rgb0, const uint8_t* rgb1, int w, int h,
                          float* out) const {
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
                    for (int t = 0; t < c.temporal; t++) {
                        // 静止图像两个时间片是同一帧；视频里 t=0/1 取相邻两帧
                        const uint8_t* src_rgb = (t == 0 || !rgb1) ? rgb0 : rgb1;
                        if (c.temporal > 1 && t > 1) src_rgb = rgb1 ? rgb1 : rgb0;
                        float* s = dst + (int64_t)t * slice;
                        for (int ch = 0; ch < 3; ch++) {
                            for (int yy = 0; yy < P; yy++) {
                                const uint8_t* rowp =
                                    src_rgb + ((size_t)(py + yy) * w + px) * 3 + ch;
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

    // ---- DUBHE_VIT_TRACE=1：分阶段计时
    //
    // 塔的耗时分布已经猜错过两次（先猜"反量化流量"，再猜"cache 未命中"），
    // 所以这次先量再说。RAII 计数器的好处是中途 return 也不会漏。
    struct Phase {
        bool on = std::getenv("DUBHE_VIT_TRACE") != nullptr;
        int64_t t0 = on ? now_ns() : 0;
        int64_t* acc;
        explicit Phase(int64_t* a) : acc(a) {}
        ~Phase() { if (on) *acc += now_ns() - t0; }
    };
    int64_t T_patch = 0, T_rope = 0, T_attn = 0, T_conv = 0, T_ffn = 0, T_proj = 0;
    struct PhaseDump {
        bool on = std::getenv("DUBHE_VIT_TRACE") != nullptr;
        const int64_t *p, *r, *a, *cv, *f, *pj;
        PhaseDump(const int64_t* pp, const int64_t* rr, const int64_t* aa,
                  const int64_t* cc, const int64_t* ff, const int64_t* jj)
            : p(pp), r(rr), a(aa), cv(cc), f(ff), pj(jj) {}
        ~PhaseDump() {
            if (!on) return;
            std::fprintf(stderr,
                         "[vit] patchify %.0f / rope %.0f / attn %.0f / conv %.0f / "
                         "ffn %.0f / proj %.0f ms\n",
                         *p / 1e6, *r / 1e6, *a / 1e6, *cv / 1e6, *f / 1e6, *pj / 1e6);
        }
    } dump(&T_patch, &T_rope, &T_attn, &T_conv, &T_ffn, &T_proj);

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

    // ---- 每层的临时缓冲：**在循环外声明一次**
    //
    // 原来这些都写在 14 层循环体内（`std::vector<float> kt(KV*N*HD)` 之类），
    // 一层合计约 50 MB：分配 + 零初始化 + 首次触页。14 层就是 ~700 MB 的
    // 无谓搬运 —— 把 matmul 搬到 GPU 之后，这反而成了最大的单项开销
    // （GPU 那版 1875 ms 里有 767 ms 在 matmul 之外）。
    //
    // 这些都是"整个缓冲区会被完全覆盖"的用途，所以只 resize（首次会零初始化，
    // 之后复用）就够了，不需要每次 assign 清零。
    std::vector<float> wqn, wkn;                                   // 每 head 的 QK norm
    std::vector<float> kt, vt, o;                                  // 注意力转置 / 输出
    std::vector<float> wc, g3, grid, conv, back;                   // 深度卷积那一路
    std::vector<float> fg, fu;                                     // FFN 门控

    kt.resize((size_t)(KV * N * HD));
    vt.resize((size_t)(KV * N * HD));
    o.resize((size_t)(N * H * HD));
    g3.resize((size_t)(N * 3 * D));
    grid.resize((size_t)(N * D));
    conv.resize((size_t)(N * D));
    back.resize((size_t)(N * D));
    fg.resize((size_t)(N * c.ffn));
    fu.resize((size_t)(N * c.ffn));

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
        Phase _t(&T_patch);
        const int64_t slice = (int64_t)3 * P * P;
        const int64_t pw = patch_w[0].n / D;   // 每行宽度 = 一个时间片的元素数
        const int64_t pstride = c.patch_dim();   // 一个 patch 的总长度（两个时间片）
        matmul_qt(patch_w[0], pw, D, pixels, N, x.data(), pstride);
        if (patch_two_slices_ && c.temporal > 1 && pw == slice) {
            std::vector<float> tmp((size_t)(N * D));
            matmul_qt(patch_w[1], pw, D, pixels + slice, N, tmp.data(), pstride);
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
    std::vector<float> cosb((size_t)(N * HD)), sinb((size_t)(N * HD));
    {
        Phase _t(&T_rope);
        std::vector<float> invf((size_t)nf);
        for (int i = 0; i < nf; i++)
            invf[(size_t)i] = 1.0f / std::pow(c.rope_theta, (float)(2 * i) / (float)(HD / 2));
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
            Phase _t(&T_attn);
            wqn.clear(); wkn.clear();
            if (!deq_all(L.q_norm, wqn, err) || !deq_all(L.k_norm, wkn, err)) return false;

            matmul_qt(L.q, D, qdim, xn.data(), N, q.data());
            matmul_qt(L.k, D, kdim, xn.data(), N, k.data());
            matmul_qt(L.v, D, kdim, xn.data(), N, v.data());

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

            // 双向 GQA。N 能到几千，这是视觉塔的主要开销，所以做三件事：
            //  1) 把 K/V 转成 [kvh][j][HD] —— 原来按 j 跨 kdim 跳读，每次跨 512
            //     个 float，cache 基本全废；
            //  2) 按 head 并行，每个线程用自己的 softmax 暂存；
            //  3) **按 query 分块**（见下面的 i0 循环）。
            //
            // 第 3 条是这一轮加的，也是最大的一条。原来的循环是 i 在外、j 在内：
            // 每算一行分数都要把整个 kb 流一遍（N=3840 时 983 KB，已经超过
            // M 系列的单核 L2 有效容量），于是 N 行就是 N 遍全量重读 ——
            // 16 个头 × 7 层 × 3840 行 × 983 KB ≈ **424 GB** 的流量，
            // 而真正算的只有 105 GFLOP。实测这 4 秒里绝大部分是在等内存。
            //
            // 改成 i 分块之后，kb / vb 每块只读一次，流量除以 IB：
            // 3840/32 = 120 遍 × 2 MB ≈ 236 MB/头/层，整体 26 GB，16 倍之差。
            //
            // **数值完全不变**：每个 (i,j) 还是同一个 dot_f32、同一个累加顺序；
            // 缩放 inv 也在同一步乘进去。所以改完可以拿 vlprobe 直接对拍，
            // 期望逐位一致（不是"误差在阈值内"）。
            const float scale = 1.0f / std::sqrt((float)HD);
            const int group = H / KV;
            // 分数暂存上限 512 KB：块越大复用越好，但太大会把 L2 挤掉。
            const int64_t IB = std::max<int64_t>(
                1, std::min<int64_t>(32, (int64_t)(131072 / std::max<int64_t>(1, N))));
            for (int kvh = 0; kvh < KV; kvh++)
                for (int64_t j = 0; j < N; j++) {
                    std::memcpy(kt.data() + ((size_t)kvh * N + j) * HD,
                                k.data() + j * kdim + kvh * HD, sizeof(float) * HD);
                    std::memcpy(vt.data() + ((size_t)kvh * N + j) * HD,
                                v.data() + j * kdim + kvh * HD, sizeof(float) * HD);
                }
            auto head_body = [&](int h0, int h1) {
                std::vector<float> sc((size_t)IB * (size_t)N);
                for (int hh = h0; hh < h1; hh++) {
                    const int kvh = hh / group;
                    const float* kb = kt.data() + (size_t)kvh * N * HD;
                    const float* vb = vt.data() + (size_t)kvh * N * HD;
                    for (int64_t i0 = 0; i0 < N; i0 += IB) {
                        const int64_t ib = std::min<int64_t>(IB, N - i0);
                        // 1) 分数：kb 的每一行读一次，和 ib 个 q 各点一次
                        for (int64_t j = 0; j < N; j++) {
                            const float* kj = kb + j * HD;
                            for (int64_t r = 0; r < ib; r++) {
                                const float* qi = q.data() + (i0 + r) * qdim + hh * HD;
                                sc[(size_t)r * (size_t)N + (size_t)j] = dot_f32(qi, kj, HD) * scale;
                            }
                        }
                        // 2) softmax：逐行，和原来完全同序
                        for (int64_t r = 0; r < ib; r++) {
                            float* row = sc.data() + (size_t)r * (size_t)N;
                            float mx = -1e30f;
                            for (int64_t j = 0; j < N; j++)
                                if (row[j] > mx) mx = row[j];
                            float sum = 0.0f;
                            for (int64_t j = 0; j < N; j++) {
                                const float e = std::exp(row[j] - mx);
                                row[j] = e;
                                sum += e;
                            }
                            const float inv = 1.0f / sum;
                            for (int64_t j = 0; j < N; j++) row[j] *= inv;
                        }
                        // 3) PV：vb 的每一行读一次，累加到 ib 个输出行
                        for (int64_t r = 0; r < ib; r++) {
                            float* oi = o.data() + (i0 + r) * qdim + hh * HD;
                            for (int d = 0; d < HD; d++) oi[d] = 0.0f;
                        }
                        for (int64_t j = 0; j < N; j++) {
                            const float* vj = vb + j * HD;
                            for (int64_t r = 0; r < ib; r++) {
                                const float w = sc[(size_t)r * (size_t)N + (size_t)j];
                                if (w != 0.0f)
                                    saxpy(o.data() + (i0 + r) * qdim + hh * HD, vj, w, HD);
                            }
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
            matmul_qt(L.o, qdim, D, o.data(), N, y.data());
        } else {
            // 门控 2D 深度卷积
            Phase _t(&T_conv);
            wc.clear();
            if (!deq_all(L.conv_spatial, wc, err)) return false;   // 每层只有 9×1024，随便反量化

            const int64_t c3 = 3 * D;
            matmul_qt(L.conv_in, D, c3, xn.data(), N, g3.data());

            // 栅格化：[token][merge²][C] → [C][gh][gw]
            // 注意 in_proj 的输出被切成三段 (B, C, x)：
            //   门控用第 0 段 × 第 2 段，卷完再乘第 1 段 —— 中间那段是**输出门**。
            //   这里曾经漏掉那一段：数值不会崩、也不会报错，只是整体质量悄悄变差。
            // grid / conv 都是 D×N 个 float（1600 patch 时各 6.5 MB），会被完整
            // 覆盖，所以用不初始化的裸数组 —— std::vector 的零初始化在这里是纯浪费。
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
            for (int ch = 0; ch < D; ch++) {
                const float* kern = wc.data() + (size_t)ch * K * K;
                const float* src = grid.data() + (size_t)ch * N;
                float* dst = conv.data() + (size_t)ch * N;
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
            for (int64_t n = 0; n < N; n++) {
                int by, bx, sub;
                coords(n, &by, &bx, &sub);
                const int yy = by * M + sub / M;
                const int xx = bx * M + sub % M;
                for (int ch = 0; ch < D; ch++)
                    back[(size_t)(n * D + ch)] = conv[(size_t)(ch * N + yy * gw + xx)] *
                                                 g3[(size_t)(n * c3 + D + ch)];
            }
            matmul_qt(L.conv_out, D, D, back.data(), N, y.data());
        }

        for (int64_t i = 0; i < N * D; i++) x[(size_t)i] += y[(size_t)i];

        // ---------------- sandwich 第二段：x += ffn(norm2(x))
        Phase _tffn(&T_ffn);
        xn.assign(x.begin(), x.end());
        if (!deq_all(L.ln2, wbuf, err)) return false;
        for (int64_t n = 0; n < N; n++) rmsnorm(xn.data() + n * D, D, wbuf.data(), c.norm_eps);

        matmul_qt(L.ffn_gate, D, c.ffn, xn.data(), N, fg.data());
        matmul_qt(L.ffn_up, D, c.ffn, xn.data(), N, fu.data());
        for (int64_t i = 0; i < N * c.ffn; i++) fg[(size_t)i] = silu(fg[(size_t)i]) * fu[(size_t)i];
        matmul_qt(L.ffn_down, c.ffn, D, fg.data(), N, y.data());
        for (int64_t i = 0; i < N * D; i++) x[(size_t)i] += y[(size_t)i];

        if (on_layer) on_layer(li, x.data(), N);
    }

    // ---- 投影器：LayerNorm → 每 merge² 个 token 拼起来 → Linear → GELU → Linear
    {
        Phase _t(&T_proj);
        std::vector<float> nw, nb, b0, b2;
        if (!deq_all(mm_norm_w, nw, err) || !deq_all(mm_norm_b, nb, err) ||
            !deq_all(mm0_b, b0, err) || !deq_all(mm2_b, b2, err)) return false;

        for (int64_t n = 0; n < N; n++)
            layernorm(x.data() + n * D, D, nw.data(), nb.data(), 1e-6f);

        const int64_t din = D * M * M;              // 4096
        const int64_t mid = c.proj_hidden;          // 4096
        std::vector<float> hh((size_t)(n_tok * mid));
        matmul_qt(mm0_w, din, mid, x.data(), n_tok, hh.data());
        for (int64_t n = 0; n < n_tok; n++)
            for (int64_t o = 0; o < mid; o++)
                hh[(size_t)(n * mid + o)] = gelu(hh[(size_t)(n * mid + o)] + b0[(size_t)o]);
        matmul_qt(mm2_w, mid, c.out_dim, hh.data(), n_tok, out);
        for (int64_t n = 0; n < n_tok; n++)
            for (int64_t o = 0; o < c.out_dim; o++) out[n * c.out_dim + o] += b2[(size_t)o];
    }
    return true;
}

}  // namespace dubhe
