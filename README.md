# 天枢 · Dubhe

> Apple Silicon MacBook 与 Android 手机上的本地大模型推理引擎。GGUF 与 MLX 原生直读，单路与多路都不掉速。

状态：M0（骨架）· 闭源 · 私有仓库

---

## 一句话

让个人设备把大模型真正跑起来、跑得久、跑得省，并且不要求用户"先转换一下"。

## 核心主张

| 别人的问题 | 我们的做法 |
| --- | --- |
| 单路快就多路慢，多路快就单路慢 | 双工况调度（ADRS）+ 分形状 Kernel，两条曲线同时压住 |
| 手机跑 10 分钟就降频，benchmark 只测前 60 秒 | 热感知控制器，持续 tokens/J 是正式指标 |
| Mac 上 GGUF 与 MLX 是两套世界 | 三图 IR（Compute / Quant / State），两种格式原生直读 |
| 只能二选一：速度或正确 | 先过 parity 门禁，再谈性能；过不了就不许提交 |

## 交付物（首期只做前四个）

| 名称 | 形态 | 阶段 |
| --- | --- | --- |
| `dubhe` | CLI：`run` / `chat` / `bench` / `inspect` | M0 |
| `libdubhe` | C++ 库 + 稳定 C ABI | M0 |
| `dubhed` | 本地 HTTP 服务，OpenAI-compatible | M2 |
| `dubhe-bench` | 基准客户端，同时打 llama.cpp / MLX / 自己 | M0 |
| macOS App / Android App | SwiftUI / Compose | M5 |

## 技术栈

C++20（内核）· Objective-C++ + Metal（Apple）· Vulkan Compute + ARM NEON（Android）· CMake + Ninja（构建）· Python 3.11（只做工具，不进 hot path）

## 目录

```
docs/      开发书 / 进度书 / ADR / RFC
engine/    推理内核（include/dubhe 是唯一对外接口）
cli/       dubhe 命令行
server/    dubhed
bench/     DubheBench
tools/     转换 / autotune / 文档生成
```

## 文档（先读这两份）

| 文档 | 内容 |
| --- | --- |
| [总开发书](docs/00-总项目开发书.md) | 做什么、不做什么、怎么验收、里程碑、协作模式 |
| [进度书](docs/02-进度书.md) | 现在到哪了、卡在哪、下一轮做什么 |
| [引擎开发书索引](docs/01-推理引擎开发书-索引.md) | 内核技术规格（正文在 `docs/references/`） |
| [ADR](docs/adr/README.md) | 每个不可逆选择为什么这么定 |

## 开发模式

**AI 写全部代码，人类只做验证与拍板。** 时间单位是 Round：

```
我交付可运行增量 → 你跑一条命令（≤5 分钟）→ 回传输出 → 我修 → 下一轮
```

所以每个交付都自带"一条命令 + 预期输出"，验证不需要读代码。

## 许可

闭源（Proprietary）。复用的第三方组件（MIT / Apache-2.0 / BSD）声明见 `THIRD_PARTY_NOTICES.md`。
**GPL/AGPL 一律不进。** 详见 [ADR-0006](docs/adr/ADR-0006-许可证策略.md)。
