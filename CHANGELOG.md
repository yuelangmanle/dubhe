# CHANGELOG

从立项（2026-10-29）起记录所有变更。版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

分类：`新增` `变更` `废弃` `移除` `修复` `安全` `文档` `基准`

---

## [0.8.0] · 2026-10-29 · 三模态接产品面

### 新增

- **HTTP 多模态**：`/v1/chat/completions` 支持 OpenAI 的 content 数组 ——
  `image_url`、`video_url`（社区写法，官方还没定视频协议）、`input_audio`。
  data URL 和本机绝对路径都收；不抓 http(s)，避免把网关变成 SSRF 入口。
- **`--mmproj` 可给多次**：Omni 模型是视觉塔 + 语音塔两个独立 mmproj，
  引擎按 `clip.has_audio_encoder` 自动分流，网关只要允许重复参数。
- **`dubhe_mm_caps(e)`**：返回能力位掩码（1 视觉 / 2 语音），`/api/state`
  的 `mm` / `mm_vision` / `mm_audio` 由它驱动，UI 据此决定露出哪些附件入口。
- **Web UI 附件**：输入框左侧 `+` 按钮，`accept` 按服务端能力动态拼；
  选中的媒体先显示预览 chip，发送时展开成 content 数组。
  历史消息只存 192px 缩略图（约 10 KB），不存原始 data URL。
- **纯音频自动转写模式**：最后一条 user 消息只有音频、没有文字时走续写而不是
  chat 模板。LFM3-Audio 进对话状态会把音频当"待推理的问题"，输出整段分析过程，
  转写被埋在引号里；走续写第一句就是正文。也可用 `dubhe_raw: true` 显式强制。

### 修复

- **纯文本请求正文被吃光**（回归）：`flatten_messages` 里用
  `content->s("")` 取字符串型 content，那是"取名为空串的子键"，恒返回默认值，
  导致 prompt 只剩 9 个模板 token，模型以为用户发了空消息。改用 `content->str`。
- **raw 模式无限复读**：不补 BOS 时模型不知道自己站在序列开头，转写内容对但
  一句接一句重复到 `max_tokens` 也不吐 EOS。raw 和 chat 模板两条路都补 BOS。
- **模态不匹配的报错**：区分"词表里没有这个占位 token"（语言模型不支持该模态）
  和"prompt 里没写占位符"（调用方问题）。原来只说后者，把配置问题误导成调用方问题。

### 基准

| 输入 | 端到端 | prompt token | decode |
| --- | ---: | ---: | ---: |
| 图像 448×448 | 6.8 s | 165 | 49.6 tok/s |
| 视频 8 帧 448 | 13.1 s | 804 | 52.0 tok/s |
| 语音 3.45 s | 4.3 s | 44 | 59.1 tok/s |
| 纯文本 | 0.5 s | 17 | 61.5 tok/s |

decode 与纯文本一致 —— 嵌入注入没有额外开销。

### 已知限制

- 一次请求只接受一个媒体。`dubhe_set_mm_embeds` 只保存一段区间，两段特征要
  两套前缀，等 Omni 那一轮改成多段。
- 塔能加载不等于模态能用：占位 token 在语言模型词表里。VL 模型挂音频塔会得到
  "这个语言模型的词表里没有 `<|audio|>`"。
- raw 转写偶尔把同一句重复一遍。流式下无法回退，暂不处理。
- `/api/load` 切换模型不会重挂 mmproj。

---

## [0.7.0] · 2026-10-29 · 视频与语音

### 新增

- **视频输入**（`--video`）：`AVAssetImageGenerator` 按 fps 均匀抽帧
  （`appliesPreferredTrackTransform` 处理竖拍旋转，奇数帧补帧），
  `pack_pixels2(rgb0, rgb1, …)` 把相邻两帧叠进一个 temporal patch。
  `--video-fps` / `--video-frames` / `--video-side` 控制抽帧与缩放。
- **语音输入**（`--audio`）：任意格式（wav/aiff/mp3/m4a/flac）经
  AVFoundation 解成 16 kHz 单声道 → log-mel → 17 层语音塔 → 2048 维嵌入。
  占位符 `<|audio|>`，按 `ceil(帧数/8)` 展开。
- **log-mel 前端**（`engine/src/mel.*`）：周期 Hann、`center=False`、400 点
  DFT（表驱动，窗已折进 cos/sin 表）、power 谱、128 个 Slaney mel 滤波器
  （Slaney 刻度 + 面积归一化）、`log10` 下限 −12、仿射 `(x+4)/4`。
- **语音塔**（`engine/src/audiomm.*`）：三层 stride-2 的 3×3 卷积降采样
  （128 mel → 16，T → T/8）→ 17 层 LFM2（conv/conv/attn ×5 + attn ×2，
  门控因果深度卷积宽 3、GQA 8/4 头、head_dim 96、滑窗 32 因果注意力）
  → embedding_norm → 投影器（LayerNorm → 768→2048 → GELU → 2048→2048）。
- **音频解码**（`engine/src/audiofile.*`）：AVAudioFile + AVAudioConverter。
- **共用核函数**（`engine/src/mm_math.h`）：量化 matmul、NEON 点积、
  RMSNorm/LayerNorm、SiLU/GELU、逐行并行，视觉塔与语音塔共用。
- **工具**：`audio_oracle.py`（HF 的 `Lfm3AudioEncoder` 真跑一遍当参考）、
  `audio_diff.py`（逐层对拍）、`audprobe`（逐层 dump）。

### 修复

- **跨模态漏 BOS**：LFM 系的 chat 模板第一行是 `{{- bos_token -}}`
  （`<|startoftext|>`），但 GGUF 里 `tokenizer.ggml.add_bos_token = false`。
  少了它模型不进入对话状态，而是把提示词当成待续写正文 —— 语音上表现为
  "转写对了、然后复读指令、然后胡说"，视觉上只是"答得糙"，所以一直没被发现。
  `run`（非 `--raw`）与 `chat` 现在都会补。
- **`AVAudioFile` 的 EOF 不是错误**：读到末尾返回 NO + `eofErr(-39)`，
  当成错误处理会让任何正常音频在最后一帧报错。

### 基准

- 语音：mel 前端 23 ms / 3.45 s 音频（实时率 ~150×），整塔 ~0.2 s（~17×）。
- 视频：8 帧 512×512（784 token）编码 28 s；prefill 88.8 tok/s。
- 精度：语音塔与 HF 逐层对拍 22 个检查点，相对误差 ≤6.2e-6，cos = 1.000000。

---

## [0.6.0] · 2026-10-29 · 多模态第一步：LFM3-VL 视觉塔

### 新增

- **mmproj 加载器**（`engine/src/mmproj.*`）：读 `clip.vision.*` 元数据，按实际
  存在的张量判定每层是 conv 还是 attention（`attn_q` → A，`conv2d_in_proj` → C），
  不靠外部配置猜。
- **图像预处理**（`engine/src/image.*`）：macOS 走 ImageIO（PNG/JPEG/HEIC/WebP
  + EXIF 方向），自写 Catmull-Rom 双三次缩放（与 PIL BICUBIC 对齐），
  `smart_resize` 与 HF 等价，`pack_pixels` 按 merge-block 序摊平。
- **视觉塔前向**：patchify → RMSNorm → 轴向 2D RoPE → 14 层 conv/attn 交替
  → 投影器（LayerNorm → 4 patch 拼 4096 → GELU → 2048）。
- **嵌入注入 ABI**：`dubhe_load_mmproj` / `dubhe_encode_image` /
  `dubhe_set_mm_embeds`。注入按**绝对位置**覆盖 token 嵌入。
- **CLI**：`dubhe run -m vl.gguf --mmproj mmproj.gguf --image pic.png -p '<|image|>…'`
- **工具**：`vl_oracle.py`（跑 HF 自己的 forward 当权威参考）、`vl_ref.py`
  （numpy 逐行转写）、`vl_diff.py`（逐层对拍）、`gguf_read.py`（共享读取器）、
  `vlprobe --dump`（逐层 dump）、`DUBHE_MM_TRACE=1`（分阶段计时）。

### 修复（四个静默 bug，都不报错、只让结果悄悄错）

- 门控卷积漏了**输出门**：`out_proj(C * conv)`，中间那段是门。
- RoPE 坐标用了 merge-block 号而不是 **patch 号**，等于把图当成缩小一倍的网格。
- `Gguf.get()` 对 4D 张量用 `reshape(-1, ne[0])`，只对 2D 成立；`patch_embd`
  是 `[16,16,3,1024]`，会得到 `(1536,16)` —— 形状刚好能乘，结果全错。
- patchify 的 matmul 没传步长：每 patch 1536 维只取前 768，却按 768 跨步，
  第 t 个 patch 跨进了相邻 patch。纯白区域看不出来，带细节的图才炸。

### 基准

| 项 | 结果 |
|---|---|
| 与 HF 官方 forward 对拍 | 16 个检查点全部一致，最大相对误差 **1.7e-5**，余弦 1.000000 |
| 640×640（400 token）视觉塔 | 24.9 s → **13.5 s**（1.84x） |
| 384×384（144 token）视觉塔 | ~8 s → **3.5 s** |
| matmul 吞吐 | 标量 22.4 → NEON **48 GFLOP/s**（515 GFLOP/图） |
| 端到端（含模型加载） | 红圆 → *a red circle*；蓝方块+绿长条 → *a blue square and a green rectangle* |

### 说明

- 视觉塔的 matmul 在 CPU 上已贴着 NEON 吞吐上限，再快只能上 Metal。
- 剩下的 2.3 s 主要是 softmax 的 2.87 亿次 `expf`。
- `make engine` 只构建 `build/dubhe`，探针要 `make tools` —— 这一轮因此跑了
  几次旧二进制，白追一个假 bug。

---

## [0.5.0] · 2026-10-29 · prefill 二维分块 GEMM + 权重预取

### 基准（LFM3-1.7B Q4_K_M，601 token prompt，6 轮交错配对取比值中位数）

| 版本 | prefill | decode（对照） | prefill 相对 |
|---|---:|---:|---:|
| HEAD `47b0e27` | 72.9 tok/s | 29.2 tok/s | 1.00x |
| + 二维分块 | 94.6 tok/s | 29.9 tok/s | 1.30x |
| + 权重预取 | **102.2 tok/s** | 32.4 tok/s | **1.42x** |

裸内核单层 GEMM（LFM3-1.7B 真实形状）：422 → 672–688 GFLOP/s。

> 绝对 tok/s 与 0.4.0 的读数**不可比**：那批是另一段窗口、机器负载不同。
> 本机 load 在几分钟内能从 2 漂到 15，只有同一批交错运行的比值可用。
> 0.4.0 里"prefill 66 vs MLX 495（7.5x）"是跨会话读数，已作废；
> 同会话配对后是 **6.11x**（102 / 716）。

### 新增
- **`mm2_*` 分块 GEMM 内核族**（f32 / f16 / q8_0 / q4_k / q6_k）：一个
  threadgroup = 8 个 simdgroup × 4 行 = 32 行输出，K 方向按 256 元素分片，
  激活读放大 ÷32。lane 固定负责 8 个连续元素，scale 解包退化成每块一次。
- **权重软件预取**：解量化拆成 `LD_*`（纯加载）/ `CV_*`（纯换算），在算当前
  K 片之前先发下一片的加载，把 600 拍的 DRAM 往返藏进 256 拍的计算里。
- `tools/ab.py`：交错 A/B/C 基准，**逐轮配对算比值再取中位数**，并用
  decode 做内部对照（decode 走 `mv_*`，改动只碰 prefill 时它的比值必须
  ≈1.00x，偏离多少就是残余漂移）。
- `tools/sweep_mm.py`：分块参数扫描。MSL 是字符串，`-D` 传不进去，每个
  变体必须改头文件重编译 —— 脚本负责改、编、跑、`finally` 还原。
- `tools/mma_probe.mm`：float4 FMA / half4 FMA / 8×8×8 MMA 三者对比。
- `tools/mm2_check.mm`：`mm2_*` 与旧 `mm_*`、与 CPU 参考实现三方对拍，
  外加 `--bench` 裸内核吞吐。
- Makefile 新增 `shader-check`（离线 `xcrun metal` 编译 MSL 字符串）与
  `mm2` 目标。

### 变更
- 激活**不经过 threadgroup memory**，直接读 device 内存。违反直觉但实测
  更快（560 → 653 GFLOP/s）：每个 lane 只读自己那 8 个元素，tgmem 唯一
  作用是"把搬运摊给 256 线程"，而每片一次 barrier 要等 400 拍、计算只有
  280 拍。去掉后静态 tgmem 归零、占用率上升。
- `engine/src/metal_kernels.h` 末尾新增 `namespace dubhe_msl` 常量真源：
  MSL 里的 `#define` 在字符串里，C++ 侧看不见，派发参数与工具都从这里取。

### 修复
- **Makefile 的 `engine` 假目标**：仓库里有同名目录，`make engine` 被当成
  "已存在、无先决条件"的文件目标，直接报 `Nothing to be done` 而**不重编
  任何东西**。一次参数 A/B 因此比了两个完全相同的二进制。已加
  `engine: $(BIN)` 别名。
- **跨会话指标作废**：进度书里 prefill `0.13`（66 / 495）、decode `0.54`
  （47.2 / 87.1）都是跨会话相除，已替换为同会话配对值（0.16 / 0.37），
  并在指标表里保留一行"已作废"说明，避免再被引用。

### 已实测否掉的方案（写下来避免重走）
| 方案 | 实测 |
|---|---|
| `MM_SG=6/8` | 236 / 218 GFLOP/s（基线 672）。越过 64 寄存器线，溢出 |
| `BMAX=16/32` | 405 / 351，每 token 反而慢 1.7x。acc 被 `MM_SG×BMAX` 锁死 |
| threadgroup memory 暂存 | 560（无 tgmem 653）。barrier 等 L2 往返不划算 |
| 上述 + 双缓冲 | 518。tgmem 翻倍把每 core 的 threadgroup 数从 4 压到 2 |
| `simdgroup_matrix` 8×8×8 MMA | 2.45 TFLOP/s **低于** float4 FMA 的 2.84（苹果是标量模拟） |
| fp16 FMA | 3.11 vs 2.84 TFLOP/s，只有 1.19x |

---

## [0.4.0] · 2026-10-29 · LFM3 tri-hybrid + 量化 matvec 重写

### 基准（LFM3-1.7B，M4 24GB，同一会话背靠背，load≈4）

| 后端 | 权重 | 体积 | decode | 有效带宽 |
|---|---|---:|---:|---:|
| 天枢 Metal | Q4_K_M | 1.10 GB | **47.2 tok/s** | 52 GB/s |
| 天枢 Metal | Q8_0 | 1.85 GB | 33.5 tok/s | 62 GB/s |
| 天枢 Metal | F16 | 3.47 GB | 20.8 tok/s | 72 GB/s |
| 天枢 CPU | Q4_K_M | 1.10 GB | 10.7 tok/s | 12 GB/s |
| 苹果 MLX | 4bit | 0.97 GB | 87.1 tok/s | 85 GB/s |
| 苹果 MLX | 8bit | 1.90 GB | 47.9 tok/s | 91 GB/s |

prefill（601 token）：天枢 66 tok/s，MLX 495 tok/s。完整矩阵见
[`docs/references/LFM3-精度与后端实测.md`](docs/references/LFM3-精度与后端实测.md)。

### 新增
- **LFM3 tri-hybrid 架构支持**（CPU + Metal 双路径）：13 个 ShortConv 层、
  6 个全局注意力层、5 个滑窗注意力层，外加每 head 的 QK RMSNorm、
  sigmoid 输出门、zero-tail partial RoPE、sandwich norm。层类型直接读
  GGUF 的 `head_count_kv` / `sliding_window_pattern` 两张逐层表。
- **Metal 融合内核 4 个**：`k_headnorm` / `k_gate_mul` / `k_conv`+`k_conv_shift`
  / `k_mul`，全部编进同一个 command buffer，每 token 仍只同步一次。
- `tools/tokprobe.cpp`：分词 + 贪心生成探针，用于区分"输出不同"是分词
  差异还是前向差异。
- `tools/mlx_lfm3.py`：补上 mlx-lm 缺失的 lfm3 实现，于是 Liquid 官方的
  `LFM3-*-MLX-4bit/8bit` 能在苹果 MLX 上跑 —— 既是独立参考实现，
  也是"苹果的版本"的速度基准。
- `tools/bench_matrix.py`：精度 × 后端全矩阵，输出 markdown 表格。
- `DUBHE_VERBOSE=1`：打印逐层结构表（层类型 / rope base / 角度 / 门）。
- `DUBHE_F32_REF=1`：强制 CPU 参考路径走纯 f32，把对拍噪声底从 1e-1 降到 1e-5。

### 变更
- **量化 matvec 重写**：Q4_K / Q6_K 从"一个 lane 在每个 32 元素段各取一个
  元素"改成"一个 lane 固定负责一个 scale 组的 8 个元素"。scale 查询从
  每块 8 次降到 1 次，指令数从 10 条/元素降到约 3 条；8 个量化字节用一条
  `uint2` / 4 条 `ushort` 向量读。decode **13.8 → 47.2 tok/s（3.4x）**。
- mv 内核改为"一个 simdgroup 同时算 `MV_R` 行"（`MV_R=2`）。
- 层结构里新增 `rope_angles_l`。

### 修复
- **滑窗层的 RoPE 角度**：HF 的 `rope_parameters` 是按层类型分开配的，
  `sliding_attention` 没有 `partial_rotary_factor`（缺省 1.0 → 64/64 全量），
  早先按全局一个值套所有层，5 个滑窗层的 q/k 有一半维度该转没转。
- `k_mul` 把字节偏移当元素下标（残差流 rel=0.23）、行跨度写死成 `n`
  （nrow=1 精确、nrow=2 崩）、`k_conv` 写 y 不带行偏移（多行全写第 0 行）。
- ShortConv 历史缓冲改为**先卷积、后移位**（反过来会让核整体偏移一位）。

### 基准
- ANE 实测：batch=1 无优势（脉动阵列空转），batch≥32 快 2–3 倍，
  `conv_in` 在 batch=128 上比 GPU 快 **9.7 倍**。结论：ANE 属于 prefill。

---

## [0.3.0] · 2026-10-29 · 底层三处结构性优化

### 基准（Qwen2.5-0.5B-Instruct Q4_K_M，M4 24GB，背靠背交替测量）

| 场景 | 上一版 | 本版 | 倍数 |
|---|---|---|---|
| 短上下文 prefill | 87–92 tok/s | **382 tok/s** | **4.3x** |
| 短上下文 decode | 78–79 tok/s | **79–84 tok/s** | 持平 |
| 631 tok prefill | 47 tok/s | **276–318 tok/s** | **6.0x** |
| 631 tok decode | 33–35 tok/s | **47–52 tok/s** | **1.5x** |

### 新增
- **批处理 prefill**（`mm_*` 内核 + `fwd_batch`）：权重读一遍喂 8 个 token，
  算术强度 ×8。prefill 内 `out_w` 只对最后一行计算（它是全模型最大的单个
  矩阵乘，占权重流量 36%）。
- **激活转置布局** `[n_in][BMAX]`：一个权重值的 8 次 fma 从 8 条标量加载
  变成 2 条 `float4`。转置时对 `j >= nrow` 填 0，内核里彻底去掉分支。
- **split-K 注意力**（`k_attn_split` + `k_attn_comb`）：长上下文把 key 区间
  切成 nsp 段并行算局部 (m, l, acc) 再归并。nsp 随上下文长度增长，上限 8。
- `make optest`：7 个 Metal 融合算子逐个与 C 参考实现对拍
  （`tools/gpu_optest.mm`），覆盖 rows=1/3、nsp=1/4、多种 head_dim。
- `metal::unbind_weights()`：卸载模型只释放权重绑定，不再拆全局 device /
  library（否则同进程里其它模型一起失效）。
- `dubhe_model_info.backend` 字段；`make check` 覆盖 GPU/CPU 双路径。

### 变更
- **matvec 并行粒度**：一线程一行 → 一个 simdgroup 一行（32 lane 分摊行内
  元素 + `simd_sum` 归约）。ffn_down 的线程数从 896 → 28672。
- Q4_K 的 min 项折进逐元素 fma，省掉一遍 x 求和。
- 注意力在 `nsp == 1`（短上下文）时走融合版内核——split 版在这个区间是
  纯亏的，实测会拖低 decode 25%。

### 修复
- **公共 lambda 漏绑 buffer**：`add`/`silu` 收进只 `setPipeline` 的公共
  lambda 后，`k_add` 读到上一次 `gemm` 的绑定，`d` 指向权重 mmap。症状是
  "输出等于输入 embedding"。
- **原地 rmsnorm 的 aliasing**：最后一层 norm 改为写入独立的 `g_bfinal`。

### 文档
- 进度书 v2.3：§9.6 记录三处优化、两个 bug、以及"绝对数字不可信，只信
  交替对比"的测量方法。

---

## [0.2.0] · 2026-10-29 · Metal GPU 后端

### 基准（Qwen2.5-0.5B-Instruct Q4_K_M，M4 24GB，本机）
| 指标 | CPU (NEON) | Metal GPU | 倍数 |
|---|---|---|---|
| decode | 20.6 tok/s | **60.9 tok/s** | 2.96x |
| prefill | 21.3 tok/s | **70.0 tok/s** | 3.29x |
| 与纯 f32 基准的 logits 差 | 3e-2（Q8 激活量化） | **1e-4** | — |

### 新增
- **Metal 计算后端**（`engine/src/metal.mm` + `metal_kernels.h`）：
  - **运行时编译 MSL**（`newLibraryWithSource:`）。绕开缺失的离线 Metal
    Toolchain（`xcodebuild -downloadComponent MetalToolchain`），任何装了
    Xcode CLT 的机器都能跑。
  - **权重零拷贝**：整块 GGUF mmap 直接包成 `MTLStorageModeShared` buffer，
    每张量用字节偏移定位。内存占用与 CPU 路径相同，不翻倍。
  - 量化内核：`mv_f32/f16/q4_0/q5_0/q8_0/q4_k/q6_k`（一线程一行，按 GGUF
    原始量化格式直读，不解包）。
  - 融合算子：`k_rmsnorm` / `k_rope` / `k_attn`（GQA + 在线 softmax）/
    `k_silu_mul` / `k_add` / `k_bias` / `k_copy`。
  - **整层前向编进一个 command buffer**，每 token 只同步一次。
  - **加载期交叉校验**：用真实权重跑 GPU vs 纯 f32 基准，相对误差 > 1e-3
    就整体退回 CPU（避免"能跑但胡言乱语"的静默错误）。
- `dubhe_model_info.backend`：暴露计算后端与校验误差。
- `DUBHE_F32_REF=1`：强制 CPU 走纯 f32 点积（校验 GPU 数值时必须关掉 Q8 激活
  量化，否则量的是 CPU 自己的近似误差）。
- `make gpu`：一条命令对比 CPU / GPU 吞吐。`make check` 现在两条路径都跑。

### 关键发现
- 本机 **单次 command buffer commit+wait 延迟 201 µs**。0.5B 模型单 token 有
  169 次 matvec，逐算子同步 = 34 ms，把 GPU 算力全埋掉。实测逐 matvec 同步
  的版本只比 CPU 快 8%。改成整层一个 command buffer 后才有 3x。
- 本机 GPU 实测读带宽 47 GB/s、CPU 17 GB/s（真机 M4 应分别在 100 GB/s 量级），
  环境受限，绝对数值不可当真机结论，加速比可信。

### 修复
- `k_attn` 的在线 softmax 在最大值更新时只缩放了分母 `l`，**忘了同步缩放已累加
  的 `acc`**——flash-attention 最经典的坑。表现为部分 head 输出错、部分正确
  （数据相关），输出乱码但程序不崩。
- MSL 里 `half` 是内建 16 位浮点类型名，不能当变量名。

---

## [0.1.0] · 2026-10-29 · M0 交付

### 新增
- **引擎**：GGUF v2/v3 解析（mmap 零拷贝）、11 种量化类型内核、
  GPT-2 字节级 BPE + Qwen2 pre-tokenize、Llama/Qwen2 前向
  （RMSNorm → QKV → RoPE → GQA → SwiGLU）、C ABI（`dubhe.h` 0.1.0）。
- **NEON 整数点积**：量化激活（Q8_0）+ `vdotq_s32`，覆盖 Q4_0/Q4_1/Q5_0/
  Q5_1/Q8_0/Q4_K/Q6_K。实测 1.4–3.8x，端到端 7.7 → 20.0 tok/s。
- **CLI**：`inspect` / `run` / `chat` / `bench`。
- **服务**：`dubhed`，OpenAI 兼容 `/v1/chat/completions`（含 SSE 流式）、
  `/v1/models`、`/api/state`。
- **Web UI**：iOS 风格单页应用（会话管理、Markdown + 代码块、工具卡片、
  设置面板、API 网关面板）。
- **工具调用**：`read_file` / `list_dir` / `write_file` / `shell`，
  模型输出 `<tool_call>` → 执行 → 结果回灌。
- **API 网关**：`--share` 监听 0.0.0.0 并自动生成 `sk-dubhe-…`；
  局域网需 `Authorization: Bearer`，本机回环免鉴权。
- **自检**：`tests/test_quant.cpp`（位解包 + 整数路径一致性）、
  `tools/selftest.cpp`、`tools/ggufdump.cpp`、`tools/ui_shot.mjs`。
- `bench/matvec_bench.cpp`：按真实模型形状测各类型吞吐。

### 修复（都是会让结果静默错掉的真实 bug）
- 字节映射表对 161–255 的字节用了原始字节而非其 UTF-8 编码，导致中文切不开。
- BPE 初始符号按 UTF-8 字符而非字节切分，merges 永不命中。
- Makefile 缺头文件依赖跟踪，改结构体后部分目标文件不重编 → ABI 错位段错误。
- KV cache 按训练上下文预分配（805 MB），改为按需翻倍。
- EOS / `<|im_end|>` 被当正文输出；HTTP 头名大小写敏感导致鉴权失效。

### 变更
- **首期范围扩大**：UI 从"推迟到 M5"提前到 M0（产品形态要求）。
- `dubhed` 从 M2 提前到 M0。

---

## [Unreleased] · v2.0 计划基线

### 变更（重要：v1 计划作废）

- **定名**：项目由代号 Aether 定名为 **天枢 / Dubhe**（北斗第一星 + 中枢；Dubhe 为该星天文名）。
- **开发模式**：由"8 人团队 × 63 周"改为 **AI 主开发 + 人类验证**，里程碑按 Round 计（27–43 轮，5–8 个月）。
- **首期范围收敛**：只交付引擎 + CLI + 本地 HTTP，**UI 推迟到 M5**。
- **许可证**：由 open-core（引擎 Apache-2.0）改为 **全闭源**，保留 4 条开源复审条件。
- **进程模型**：ADR-0005 引擎独立进程 **推迟到 M5**，M0–M4 单进程。

### 文档
- 重写《总项目开发书》v2.0：新增 §7 开发模式（协作协议）、§9 需你提供的环境信息、按 Round 排的里程碑。
- 重写《进度书》v2.0：M0 任务重切（9 项）、阻塞项 B1–B4、轮次记录模板。
- 新增 ADR-0007：AI 主开发 + 人类验证。
- 重写 ADR-0006：全闭源策略与"闭源 ≠ 不能用开源"的关键澄清。
- 新增 `LICENSE`（Proprietary）与 `THIRD_PARTY_NOTICES.md`（含排除清单）。

### 新增
- 仓库骨架：`engine/` `sdk/` `cli/` `server/` `apps/` `tools/` `bench/` `docs/`。
- `tools/docs/build_docx.sh` + 表格列宽预处理器（Markdown → docx，备用）。

---

## 版本规划

| 版本 | 里程碑 | 内容 |
| --- | --- | --- |
| 0.1.0 | M0 | GGUF 加载 + CPU reference + `dubhe run` 出字 |
| 0.2.0 | M1 | Metal 加速，B=1 追上 llama.cpp |
| 0.3.0 | M2 | 并发调度 + `dubhed` |
| 0.4.0 | M3 | MLX 直读 + 长上下文 + 结构化输出 |
| 0.5.0 | M4 | Android（CPU + Vulkan） |
| 1.0.0 | M5 | 双端 App + 打包发布 |

---

## v0.0.1-docs · 2026-10-29

### 文档
- 首版《总项目开发书》《进度书》《推理引擎开发书索引》。
- ADR-0001..0006 首批落档。
- 归档引擎开发书 v1.0 docx 至 `docs/references/`（正文仍用旧代号，属历史原貌）。
