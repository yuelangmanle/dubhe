# CHANGELOG

从立项（2026-10-29）起记录所有变更。版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

分类：`新增` `变更` `废弃` `移除` `修复` `安全` `文档` `基准`

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
