# tools/ · 工具链

离线工具，不进推理 hot path。许可：闭源（Proprietary，见根目录 LICENSE）。

| 子目录 | 内容 | 里程碑 |
| --- | --- | --- |
| `convert/` | HF ↔ GGUF ↔ MLX ↔ DubhePack 转换、量化 | M1 |
| `autotune/` | Kernel 形状扫描、profile 采集与入库 | M1 |
| `studio/` | 数据清洗、LoRA/QLoRA 训练、评测、adapter 导出、parity 检查 | M4 |
| `docs/` | Markdown → docx 生成（`build_docx.sh`） | M0 |
| `release/` | 制品打包、SBOM 生成、NOTICE 汇总 | M0 |

## 约束

- Python 3.11+，依赖锁在 `tools/requirements.txt`，用 `uv` 或 `pip-tools` 管理。
- 所有工具必须能离线运行（企业无网场景）。
- 转换工具必须输出 `manifest.json`（含来源、许可、哈希、量化参数）。

---

## 现有工具（平铺在本目录）

| 文件 | 用途 |
| --- | --- |
| `ab.py` | **交错 A/B/C 基准**。逐轮配对算比值再取中位数，并用 decode 做内部对照。本机唯一可信的对比口径 |
| `bench_matrix.py` | 精度 × 后端全矩阵，输出可直接贴进文档的 markdown 表 |
| `sweep_mm.py` | 分块参数扫描（改 `#define` → 重编 → 跑 → `finally` 还原） |
| `mm2_check.mm` | `mm2_*` 与旧 `mm_*`、与 CPU 参考三方对拍；`--bench` 裸内核吞吐 |
| `mma_probe.mm` | float4 FMA / half4 FMA / 8×8×8 MMA 三者峰值对比 |
| `gpu_optest.mm` | 融合算子逐个对拍（整层前向有 400+ dispatch，错了只看得出"乱码"） |
| `mlx_lfm3.py` | 补上 mlx-lm 缺失的 lfm3 实现，让官方 MLX 权重能跑 |
| `probe_precision.py` | 算子级精度 × 后端 × batch 全交叉 |
| `tokprobe.cpp` | 分词 + 贪心生成探针，区分"分词差异"与"前向差异" |
| `ggufdump.cpp` | GGUF 元数据与张量表 |
| `selftest.cpp` | 引擎自检 |
| `ane_probe.py` | ANE（Core ML）算子级测速 |
| `ui_shot.mjs` | Web UI 截图 |

### 多模态（视觉塔）对拍

| 文件 | 用途 |
| --- | --- |
| `vl_oracle.py` | **权威参考**：直接跑 HF 的 `modeling_lfm3_vl.py`，权重取自同一份 mmproj。首次运行会把 HF 那 5 个 .py 抓到 `~/.dubhe/oracle/` |
| `vl_ref.py` | numpy 版参考，逐行对照源码转写，不依赖 torch |
| `vl_diff.py` | 逐层比 max\|Δ\| / 相对误差 / 余弦 |
| `gguf_read.py` | 共享 GGUF 读取器。**4D 张量用 `reshape(ne[-1], -1)`**，不是 `reshape(-1, ne[0])` |
| `vlprobe.cpp` | 视觉塔探针：`--dump` 逐层写 f32，编号与 HF `hidden_states` 对齐 |

```bash
./build/vlprobe mmproj.gguf pic.png --dump /tmp/vl
python tools/vl_oracle.py mmproj.gguf /tmp/vl.pixels.f32 /tmp/vl.meta.txt --dump /tmp/vl
python tools/vl_diff.py /tmp/vl /tmp/vl        # 全部 ✓ 才算过
DUBHE_MM_TRACE=1 ./build/vlprobe ...           # 打印 matmul / 反量化分阶段耗时
```

**为什么要有 oracle**：两个独立实现如果是同一个人照着同一份理解写的，就共享
同一批误解。第一版 `vl_ref.py` 和 C++ 犯了同样两个错，对拍全绿 —— 假绿灯。
只有把 HF 自己的 forward 拉进来才能证伪。

### 语音（语音塔）对拍

| 文件 | 用途 |
| --- | --- |
| `audio_oracle.py` | **权威参考**：把 mmproj 的权重装进 HF 的 `Lfm3AudioEncoder` + `Lfm3AudioProjector` 真跑一遍，逐层存 .npy |
| `audio_diff.py` | 逐层比 max\|Δ\| / 相对误差 / 余弦（`down` 那步 HF 是 (通道, 时间)，工具里会自动转置） |
| `audprobe.cpp` | 语音塔探针：mel → 降采样 → 17 层 → embedding_norm → 投影器，逐层写 f32 |

```bash
# 1. 造测试音频（macOS 自带 TTS，免下载）
say -o /tmp/a.aiff "The quick brown fox jumps over the lazy dog."
ffmpeg -y -i /tmp/a.aiff -ar 16000 -ac 1 -c:a pcm_s16le /tmp/a.wav

# 2. 两路各跑一遍
python tools/audio_oracle.py --wav /tmp/a.wav --out /tmp/ao
./build/audprobe ~/.dubhe/models/lfm3-audio-2b/mmproj-LFM3-Audio-2B-Q8_0.gguf /tmp/a.wav /tmp/ac

# 3. 比
python tools/audio_diff.py --cpp /tmp/ac --oracle /tmp/ao   # 全部一致才算过
./build/audprobe ... /tmp/a.wav /tmp/ac --mel-only          # 只查 mel 前端
```

**映射上的两个坑**（写在这里省得下次再查）：

1. 语音塔内部是 **LFM2**，不是 LFM3 —— `AutoModel.from_config` 走的是
   transformers 内置的 `Lfm2Model`，所以 HF 的参数名是 `q_layernorm` /
   `out_proj`，跟 `modeling_lfm3.py` 里写的 `q_norm` / `o_proj` 不一样。
2. GGUF 的 `a.post_ln` 对应 HF 的 `embedding_norm`，名字带 post，但它在
   **所有层之后**才作用。`clip.audio.feed_forward_length = 1792` 也和
   `config.json` 的 2560 不一致（后者是 2/3 规则前的值），以 GGUF 为准。

**端到端还有一道独立闸门**：用 `say` 合成一句人声，看引擎能不能逐字转写。
逐层对拍只能证明"C++ 和 HF 算的一样"，如果权重映射本身错了，两边会一起错。

### 测量纪律（踩过的坑，别再踩）

1. **只信同条件交错 A/B。** 一边跑完再跑另一边得到的倍数一律作废 —— 本机
   load 几分钟内能从 2 漂到 15。
2. **看 decode 对照。** 改动只碰 prefill 时，decode 的配对比值必须 ≈1.00x；
   偏离多少就说明还有多少漂移没被约掉。
3. **微基准会骗人。** 裸内核基准让权重常驻 L2，真实模型每秒上 GB 的权重把
   激活挤出去。裸基准说快 17% 的方案，端到端可能持平甚至更慢。
4. **别信 `make <目录名>`。** 仓库里有 `engine/`、`server/`、`tools/` 目录，
   如果 Makefile 里没有同名目标，make 会当成"已存在、无先决条件"的文件目标
   直接返回 `Nothing to be done`，**不编译任何东西**。`engine` 已加别名。
