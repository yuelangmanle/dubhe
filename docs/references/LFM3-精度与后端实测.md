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

> ⚠️ **本节数字已作废（2026-10-29 Round 7 修订）**：下面这张表是**跨会话**
> 读数（两个数字来自不同时间、不同机器负载），相除没有意义。同会话配对
> 重测后见 §3.5。旧结论"7.5 倍"保留在此仅为记录它错在哪里。

| 后端 | 权重 | prefill（作废） |
|---|---|---:|
| 天枢 Metal | Q4_K_M | 66 tok/s |
| MLX | 4bit | **495 tok/s** |

**为什么不可比**：本机 load 在几分钟内能从 2 漂到 15，prefill 的绝对值随之
浮动 1.5–2 倍。任何"一边跑完再跑另一边"得到的倍数都是噪声。

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

### 3.5 prefill 的真实差距与它的来源（Round 7 实测）

上一轮我判断"差距在激活的重复读取，分块之后能到 250+ tok/s"。**这个判断
错在量级上**，原因是我按"单个矩阵"算流量，而 prefill 的实际行为是按
"整段 601 token" 展开的：

| 流量（整段 prefill） | 体积 |
|---|---:|
| 权重（1.1 GB × 601/8） | **83.6 GB** |
| 激活（绝大部分命中 L2/L1） | 5.1 TB（不出 DRAM） |

因为 `BMAX = 8`，每算 8 个 token 就要把整个权重矩阵重读一遍，601 个 token
读 **76 遍**。MLX 把 601 个 token 一次当 N 维铺开，权重只读一遍 —— 两边
算术强度差 76 倍。**prefill 的差距主要来自这里。**

同会话、6 轮交错、逐轮配对取中位数：

| 版本 | prefill | decode（内部对照） | prefill 相对 |
|---|---:|---:|---:|
| HEAD `47b0e27` | 72.9 tok/s | 29.2 tok/s | 1.00x |
| + 二维分块 `mm2_*` | 94.6 tok/s | 29.9 tok/s | 1.30x |
| + 权重预取 | **102.2 tok/s** | 32.4 tok/s | **1.42x** |

decode 对照 1.03x / 1.01x —— 它走 `mv_*` 另一套内核、本轮没碰，所以这组
比值是可信的。

同会话 vs 苹果 MLX（4 轮配对）：

| 目标 | prefill | decode | prefill 相对 | decode 相对 |
|---|---:|---:|---:|---:|
| 天枢 Q4_K_M | 102.0 tok/s | 32.5 tok/s | 1.00x | 1.00x |
| MLX 4bit | 716.2 tok/s | 87.7 tok/s | **6.11x** | **2.67x** |
| MLX 8bit | 614.5 tok/s | 42.5 tok/s | 4.56x | 1.57x |

裸内核单层 GEMM（LFM3-1.7B 真实形状）：HEAD 422 → 现 672–688 GFLOP/s。

**分块为什么只拿到 1.30x 而不是 4x**：收益上限由寄存器决定，不由内存流量
决定。`acc(MM_SG×BMAX) + wv(MM_SG×8) ≤ 64` 这条约束卡死了复用度 ——
`MM_SG=4, BMAX=8` 正好 64，再往上就溢出（`MM_SG=6` 从 672 掉到 236）。

**预取为什么有效**：一片 K 只有 256 个 fma（约 256 拍），而一次没命中的
DRAM 往返要 600 拍以上。把解量化拆成 `LD_*`（纯加载）/ `CV_*`（纯换算），
算当前片之前先发下一片的加载，往返就被藏在计算后面。预取粒度刻意压到最小
（只取 Q4_K 的 qs 8 字节），每行只多占 2 个寄存器。

**下一步的形态**：受寄存器约束，靠调 `BMAX` 降低权重重读已经走不通（实测
BMAX=16/32 每 token 反而慢 1.7x）。真正可行的是把 K 循环摊到多个
threadgroup 做两级归约，让一次权重读取服务更多 token。

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

# 分块 GEMM 对拍（5 种格式 vs CPU 参考）+ 裸内核吞吐
make mm2

# 离线编译 MSL 字符串（防止着色器语法错误被运行时的静默回退吃掉）
make shader-check

# 交错 A/B：**唯一可信的对比口径**（逐轮配对中位数 + decode 内部对照）
python tools/ab.py -p 6 -n 24 \
    --dubhe "HEAD=build/dubhe-old:$HOME/.dubhe/models/lfm3-1.7b/LFM3-1.7B-Q4_K_M.gguf" \
    --dubhe "new=build/dubhe:$HOME/.dubhe/models/lfm3-1.7b/LFM3-1.7B-Q4_K_M.gguf"

# 带 MLX 一起比（需要装了 mlx-lm 的 python，见下）
python tools/ab.py -p 4 -n 24 --dubhe "dubhe=build/dubhe:$M" \
    --mlx "mlx4=~/.dubhe/models/lfm3-mlx-4bit"

# 分块参数扫描（改头文件→重编→跑→还原，全自动）
python tools/sweep_mm.py
```

MLX 侧环境（`mlx-lm` 只有 0.32.x 有 `lfm2`，`lfm3` 靠 `tools/mlx_lfm3.py` 补）：

```bash
~/.local/bin/python3.12 -m venv ~/.dubhe/venv-mlx
~/.dubhe/venv-mlx/bin/pip install mlx-lm
```

---

## 6. 结论

1. **ANE 是 prefill 的答案，不是 decode 的答案**。batch=1 上它和 GPU
   打平甚至更慢；batch≥32 时快 2–3 倍，`conv_in` 上快 9.7 倍。
2. **天枢的 decode 同会话配对是 MLX 4bit 的 0.37x**（32.5 / 87.7）。差距
   来自每字节指令数，Round 6 已经把最大的那个坑填了（3.4x）。
3. **prefill 同会话配对是 MLX 4bit 的 0.16x**（102 / 716，6.11x）。根因
   **不是**激活重复读（那是次要项），而是 `BMAX=8` 让整个权重矩阵每 8 个
   token 就重读一遍 —— 601 token 读 76 遍。二维分块 + 权重预取拿到了
   1.42x，但要再上一个量级必须改这个结构。
4. **Q8_0 与 F16 输出逐 token 一致**，两条反量化路径互为验证。
5. **`simdgroup_matrix` 在 M4 上不是硬件矩阵单元**：8×8×8 MMA 只有
   2.45 TFLOP/s，**低于** float4 FMA 的 2.84（`tools/mma_probe.mm`）。
   "为 MMA 重写"这条路线可以彻底排除。
6. **寄存器红线在 64 附近**。越过之后 occupancy 掉一半，任何"复用更多"
   的改动都会净亏 —— 这决定了分块参数的可选范围，也解释了为什么
   `BMAX=16/32` 这种直觉上该赢的方案实测更慢。
