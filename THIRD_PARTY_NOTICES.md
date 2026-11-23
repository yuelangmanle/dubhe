# THIRD_PARTY_NOTICES

本项目为闭源软件，但**包含或参考了**以下第三方开源组件。各自遵循原许可证，其条款优先于本项目 LICENSE 适用于对应部分。

> 维护规则：每引入一个上游文件（哪怕是改写），都在下表登记 `来源仓库 + commit + 文件路径 + 许可`。
> CI 会校验此表与依赖树一致；缺失即阻断。

## 1. 参考实现与设计来源（尚未引入代码）

| 项目 | 许可 | 用途 | 引入方式 |
| --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | GGUF 格式定义、量化方案、CPU 参考实现 | 参考 / 部分移植（待登记具体文件） |
| [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | Apple 原生算子语义、数值 oracle | 可选 fallback 后端（待定） |
| [Locals](https://github.com/Generated-Computers/Locals) | Apache-2.0 | small-M Kernel 思路、agent cache 设计 | 仅参考思想（不引入代码） |
| [vLLM Metal](https://github.com/vllm-project/vllm-metal) | Apache-2.0 | paged state、scheduler 设计 | 仅参考思想 |
| [MNN](https://github.com/alibaba/MNN) | Apache-2.0 | ARM 低比特 Kernel、Android 后端 | 参考 / 可能移植 |
| [ExecuTorch](https://github.com/pytorch/executorch) | BSD-3-Clause | NPU delegate 设计 | 仅参考 |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Apache-2.0 | 自动调优（TensorIR/MetaSchedule）思路 | 仅参考 |
| [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) | Apache-2.0 | Android 竞争基线、MTP | 仅参考 |

## 2. 已引入的第三方代码

（M0 起逐条登记，格式如下）

| 文件 | 来源仓库 | commit | 许可 | 修改说明 |
| --- | --- | --- | --- | --- |
| `web/vendor/katex-0.19.1.min.js` · `web/vendor/katex.LICENSE` | [KaTeX](https://github.com/KaTeX/KaTeX) · `dist/katex.min.js` | `8000fbd4e29cca2c0cb1e88b745af3ed2f436b66` | MIT | npm 0.19.1 原始发行文件，用于本地公式排版 |

系统 SQLite 用于桌面知识库全文检索和任务存档，采用 [公有领域授权](https://www.sqlite.org/copyright.html)。Android 使用 [Room](https://developer.android.com/jetpack/androidx/releases/room)、[WorkManager](https://developer.android.com/jetpack/androidx/releases/work) 和 [Multiplatform Markdown Renderer](https://github.com/mikepenz/multiplatform-markdown-renderer)，均采用 Apache-2.0 许可证。Mac 的可选文档导入调用用户本机安装的 [MarkItDown](https://github.com/microsoft/markitdown)，采用 MIT 许可证。

## 3. 明确排除

以下许可的代码**不得**进入本仓库或任何交付二进制：

- GPL-2.0 / GPL-3.0
- AGPL-3.0
- LGPL（静态链接）
- 任何 "Non-Commercial" / "Research Only" 条款的模型或代码

## 4. 模型权重

模型权重不属于本项目，其许可由模型提供方决定。
每个模型的许可记录在 `~/.dubhe/models/<id>/manifest.json` 的 `license` 字段，运行时不改写、不再分发。
