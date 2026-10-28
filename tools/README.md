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
