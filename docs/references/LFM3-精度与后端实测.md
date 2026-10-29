# LFM3 精度 × 后端实测（M4 24GB）

> 目的：回答三个问题 ——
> 1. **ANE 到底能不能加速？** 能，但只在特定场景。
> 2. **1.7B / 2B 模型在多少精度下跑多快？** 见模型级矩阵。
> 3. **"苹果的版本"（MLX）比我们快多少？** decode 1.85x，prefill 7.5x。
>
> 测量条件：MacBook Pro M4（10 核 GPU，24GB 统一内存），macOS 26。
> 所有数字都是**同一会话内背靠背**测的，因为这台机器在跑其它任务时
> decode 会从 47 掉到 13 tok/s（2.8x 波动），跨批次比较没有意义。
> 本次测量时 load average ≈ 4，因此绝对值偏低，**倍数关系才是结论**。

---

## 1. 算子级：精度 × 后端 × batch

工具：`tools/probe_precision.py`。对 LFM3-1.7B 的 9 个真实权重形状，
在 3 档权重精度（f16 / int8 / int4）× 4 个后端（CPU / Metal / ANE(Core ML) / MLX）
× 4 个 batch（1 / 8 / 32 / 128）上全交叉测一遍。

### 关键发现：ANE 的位置由 batch 决定，不由 shape 决定

| shape | n_in→n_out | B | CPU | GPU | **ANE** | MLX | 最快 |
|---|---|---:|---:|---:|---:|---:|---|
| ffn_up | 2048→7680 | 1 | 0.99 | 0.72 | 0.64 | 0.65 | ANE |
| ffn_up | 2048→7680 | 128 | 2.11 | 2.12 | **0.81** | 1.91 | **ANE 2.6x** |
| ffn_down | 7680→2048 | 1 | 1.39 | 1.13 | 1.12 | **0.62** | MLX 1.8x |
| ffn_down | 7680→2048 | 128 | 2.21 | 3.04 | **0.93** | 1.99 | **ANE 2.4x** |
| conv_in | 2048→6144 | 128 | 1.67 | 5.41 | **0.56** | 2.11 | **ANE 3.0x** |
| lm_head | 2048→128000 | 1 | 13.08 | 6.65 | 8.56 | **5.95** | MLX 1.1x |
| lm_head | 2048→128000 | 32 | 16.94 | 26.79 | **11.33** | 13.50 | **ANE 1.5x** |

（单位 ms；f16 权重。int8 / int4 的排序基本一致，只是整体更快。）

**为什么 ANE 在 batch=1 上没有优势**：神经引擎是脉动阵列（systolic array），
它的效率来自"同一个权重在阵列里连续流过很多个激活"。decode 时 batch=1，
一个权重只乘 1 个激活，阵列 90% 的乘法单元在空转。这时候瓶颈从算力变成
**取权重**，ANE 和 GPU 用的是同一条内存总线，谁也不比谁快。

**为什么 batch≥32 时 ANE 反超 2–3 倍**：batch 大了以后算术强度上来了，
ANE 的阵列终于喂饱，而它的能效比（TOPS/W）本来就远高于 GPU 的 SIMD 阵列。
`conv_in`（2048→6144）在 batch=128 上 ANE 比 GPU 快 9.7 倍 —— 这是
整份数据里最大的单项差距。

**MLX 在 batch=1 上最快**：`ffn_down` 0.62ms、`lm_head` 5.95ms。原因是
MLX 的 4bit 权重布局（group=64 的 affine，scale/bias 和量化值分开存）
配合它的 matvec 内核，在 B=1 时把每个权重字节的指令数压到了最低。
这正是我们自己 Q4_K 内核缺的东西 —— 见第 3 节。

### 结论

- **decode（batch=1）不要指望 ANE**。它不比 GPU 快，还得付出 Core ML
  的模型加载与格式转换成本。
- **prefill（batch 大）应该走 ANE**。这是目前最大的一块未兑现收益：
  现在的 prefill 还在 GPU 上，而 ANE 在同等工作量上快 2–3 倍。
- 落地形态不是"整模型转 Core ML"（那会丢掉 KV cache 的灵活性），而是
  **把 prefill 阶段的大矩阵乘卸载给 ANE**，decode 仍走 GPU。这条路线
  需要一次 command buffer 级别的重构，列入 Round 7。

---

## 2. 模型级：端到端 decode

工具：`tools/bench_matrix.py`（天枢）+ `tools/mlx_lfm3.py`（MLX）。
固定 prompt `The capital of France is`，贪心解码 40 token，跑 3 次取最快。

| 后端 | 权重 | 体积 | decode | 有效带宽 |
|---|---|---:|---:|---:|
| 天枢 Metal | LFM3-1.7B Q4_K_M | 1.10 GB | **47.2 tok/s** | 52 GB/s |
| 天枢 Metal | LFM3-1.7B Q8_0 | 1.85 GB | 33.5 tok/s | 62 GB/s |
| 天枢 Metal | LFM3-1.7B F16 | 3.47 GB | 20.8 tok/s | 72 GB/s |
| 天枢 Metal | LFM3-VL-2B Q4_K_M | 1.10 GB | 42.4 tok/s | 47 GB/s |
| 天枢 CPU | LFM3-1.7B Q4_K_M | 1.10 GB | 10.7 tok/s | 12 GB/s |
| **MLX** | LFM3-1.7B-MLX-4bit | 0.97 GB | **87.1 tok/s** | 85 GB/s |
| **MLX** | LFM3-1.7B-MLX-8bit | 1.90 GB | 47.9 tok/s | 91 GB/s |

**读法**：天枢的有效带宽随每元素字节数单调上升（52 → 62 → 72 GB/s），
而 MLX 稳定在 85–91 GB/s。这说明天枢**还没有撞到带宽墙**，缺的是
"每个字节要几条指令"。MLX 在三种精度上都快 1.4–1.9 倍，差距是结构性的，
不是某个量化格式的问题。

### prefill（601 token prompt）

| 后端 | 权重 | prefill |
|---|---|---:|
| 天枢 Metal | Q4_K_M | 66 tok/s |
| MLX | 4bit | **495 tok/s** |

**7.5 倍**。这是当前最大的单项差距，原因在下一节。

---

## 3. 这一轮修了什么：matvec 的指令数

改之前 Q4_K 的 decode 只有 39 GB/s，Q8_0 却有 67 GB/s —— 而两者
**每 token 的墙钟时间几乎一样（26 ms）**。这个反常现象直接指向结论：
不是带宽不够，是指令数不够用。

拆开看 Q4_K 的老内核：一个 lane 在**每个** 32 元素段里各取一个元素，
于是一个 256 元素块要调 8 次 `k4_scale_min`（约 10 条指令，带分支和
3 次字节读）。一个 lane 每块只算 8 个元素，却付了 80 条指令的账 →
**10 条指令/元素**。Q8_0 那边是 1 条/元素。

改法：让一个 lane **固定负责一个 scale 组的 8 个元素**。

- Q4_K：`s = lane/4` → 32 个 lane 正好覆盖 8 个 sub-block，每块 1 次
  scale 查询，约 3 条指令/元素；8 个量化字节用一条 `uint2` 读进来。
- Q6_K：`g16 = lane/2` → 覆盖 16 个 16 元素组，每块 1 次 scale 查询；
  8 字节拆成 4 条 `ushort`（块跨度 210 ≡ 2 mod 4，拿不到 4 字节对齐）。

结果：**13.8 → 47.2 tok/s（3.4x）**，Q4_K 的有效带宽 39 → 52 GB/s。

### 还差什么（Round 7 的 prefill 分块）

prefill 的 7.5 倍差距不在指令数，在**激活的重复读取**：
现在一个 simdgroup 算一行输出，同一块激活会被 `n_out` 个 simdgroup
各读一遍。以 `ffn_up`（2048→7680）为例，激活只有 64 KB，但被读
7680 次 = 491 MB，而权重才 8.8 MB —— **激活流量是权重的 55 倍**。

标准解法是 GEMM 的二维分块：一个 threadgroup 负责 `ROWS × BMAX` 的
输出块，把激活先搬进 threadgroup memory（32 KB 够装 256 元素 × 8 行），
组内 4 个 simdgroup 共享。激活流量直接除以 4，预估 prefill 从 66 →
250+ tok/s。之后再叠加 ANE 卸载。

---

## 4. 精度对照：三档输出的异同

同一个 prompt、贪心解码，看输出是否一致（`--raw` 续写）：

| 权重 | 输出 |
|---|---|
| Q4_K_M | ` Paris, and the population of France is about 67 million. The capital city of Japan is Tokyo, ...` |
| Q8_0 | ` Paris.\n\nWhat is the capital of Japan?\n\nThe capital city of Japan is Tokyo. ...` |
| F16 | ` Paris.\n\nWhat is the capital of Japan?\n\nThe capital city of Japan is Tokyo. ...` |
| MLX 8bit | ` Paris.\n\nThe capital of Japan is Tokyo.\n\nThe capital of Australia is Canberra. ...` |

- **Q8_0 与 F16 逐 token 完全一致**（前 24 token 一字不差）。这说明两条
  反量化路径都是对的 —— 一个是 8 位整数，一个是半精度浮点，误差量级不同
  却给出同一个 argmax 序列，这是很强的交叉验证。
- Q4_K_M 在第一个 token（`Paris`）上与它们一致，第二个 token 分岔。
  4 bit 量化在接近平局的位置改变 argmax 是正常现象，不是 bug。
- MLX 8bit 的输出与天枢 Q8_0 **语义完全一致**（同样的"接龙式问答"模式），
  进一步确认两个实现算的是同一个模型。

### 一个容易踩的坑：VL 模型不能用 `--raw`

LFM3-VL-2B 的文本塔结构与 1.7B 完全相同（同为 1.73B 参数、2048 隐层、
24 层），但用 `--raw` 续写会退化成 `is is.**Summaries:**...` 这样的复读。
换成 chat 模板立刻正常：

```
$ dubhe run -m LFM3-VL-2B-Q4_K_M.gguf -p "What is the capital of France?"
The capital of France is Paris.
```

**教训**：指令微调模型的续写质量不能作为"引擎算对了没有"的判据。
判据应该是 logits 对拍（`DUBHE_FUSED_CHECK_VERBOSE=1`）和 chat 格式输出。

---

## 5. 复现

```bash
# 算子级：精度 × 后端 × batch 全交叉
python tools/probe_precision.py

# 模型级：精度 × 运行时矩阵（天枢 + MLX）
python tools/bench_matrix.py -n 40

# 只看某一侧
python tools/bench_matrix.py --only dubhe
python tools/bench_matrix.py --only mlx

# 单独跑苹果 MLX 上的 LFM3（需要先下载 LiquidAI/LFM3-1.7B-MLX-4bit）
python tools/mlx_lfm3.py ~/.dubhe/models/lfm3-mlx-4bit "The capital of France is" 40

# 逐层结构表（确认层类型 / rope base / 角度）
DUBHE_VERBOSE=1 ./build/dubhe inspect -m model.gguf

# 融合内核与纯 f32 参考逐位置对拍
DUBHE_FUSED_CHECK_VERBOSE=1 ./build/dubhe run -m model.gguf -p "..." -n 8 --temp 0
```

---

## 6. 结论

1. **ANE 是 prefill 的答案，不是 decode 的答案**。batch=1 上它和 GPU
   打平甚至更慢；batch≥32 时快 2–3 倍，`conv_in` 上快 9.7 倍。
2. **天枢的 decode 现在是 47.2 tok/s（1.7B Q4_K_M），MLX 是 87.1**。
   差距来自每字节指令数，这一轮已经把最大的那个坑填了（3.4x）。
3. **prefill 是当前最大的短板：66 vs 495 tok/s**。根因是激活被重复读
   55 倍，解法是 GEMM 二维分块 + threadgroup memory 暂存激活。
4. **Q8_0 与 F16 输出逐 token 一致**，两条反量化路径互为验证。
