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
