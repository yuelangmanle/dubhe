# Aether

> 面向个人设备的本地 AI 执行系统：Apple Silicon MacBook 与 Android 旗舰手机上的大模型推理内核 + 应用 + 工具链。

[![status](https://img.shields.io/badge/status-M0%20立项-blue)]()
[![license](https://img.shields.io/badge/license-Apache--2.0%20(engine)-green)]()

---

## 这是什么

Aether 分三层：

| 层 | 组件 | 干什么 | 许可 |
| --- | --- | --- | --- |
| 内核 | **Aether Engine** (`engine/`) | 格式无关的推理运行时：GGUF / MLX 原生直读、双工况调度、序列状态管理、自适应 Kernel 规划 | Apache-2.0 |
| 接口 | **AetherKit** (`sdk/`) · **`aetherd`** | C ABI / Swift / Kotlin 绑定；OpenAI-compatible 本地服务 | Apache-2.0 |
| 产品 | **Aether Desktop** (`apps/macos/`) · **Aether Mobile** (`apps/android/`) · **Aether Studio** (`tools/`) | 用户界面、模型库、会话、本地微调闭环 | 商业许可 |

**核心主张**：单路延迟、多路吞吐、长跑能耗、格式兼容，四件事同时进第一梯队——而不是在其中一件上刷分。

## 快速开始

> 当前处于 M0（立项与基线），尚无可用构建。以下为 M1 起的预期路径。

```bash
# 构建引擎（macOS，需要 Xcode 16+ / CMake 3.28+）
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 跑基准
./bench/run.sh --engine aether --model <path> --workload chat-1
```

## 文档

| 文档 | 说明 |
| --- | --- |
| [总项目开发书](docs/00-总项目开发书.md) | **从这里开始**：产品定义、架构、里程碑、治理、验收 |
| [推理引擎开发书索引](docs/01-推理引擎开发书-索引.md) | 引擎技术规格（正文见 `docs/references/`） |
| [进度书](docs/02-进度书.md) | 每周进展、里程碑看板、任务分解 |
| [ADR](docs/adr/README.md) | 架构决策记录 |
| [RFC](docs/rfcs/README.md) | 方案提案 |
| [文档规范](docs/README.md) | 写作与版本规则 |

## 仓库结构

```
engine/     推理内核（C++20 + Metal + Vulkan + ARM）
sdk/        AetherKit：C ABI 与语言绑定
apps/       桌面端（SwiftUI）与移动端（Compose）
tools/      模型转换、adapter 训练、打包、文档生成
bench/      AetherBench：客户端、数据集、结果 schema
docs/       开发书、进度书、ADR、RFC
```

## 路线图

| 里程碑 | 时间 | 目标 |
| --- | --- | --- |
| M0 | 2026-10 → 2026-11 | 基线、IR、基准框架 |
| M1 | 2026-11 → 2027-01 | 引擎单路：GGUF + MLX 直读、Metal small-M |
| M2 | 2027-01 → 2027-03 | Mac 产品 Alpha：并发调度、状态管理 |
| M3 | 2027-04 → 2027-06 | Android Beta：ARM + Vulkan + 热控 |
| M4 | 2027-06 → 2027-09 | NPU、本地微调闭环、推测解码 |
| M5 | 2027-09 → 2027-12 | 硬化与公开基准 |
| M6 | 2027-12 → 2028-01 | 1.0 发布 |

## 贡献

M0–M2 期间以内部开发为主。外部贡献通道在 M3 开放，届时见 `CONTRIBUTING.md`。

## 许可

- `engine/`、`sdk/`、`tools/`：Apache-2.0（见 [ADR-0006](docs/adr/ADR-0006-许可证策略.md)）
- `apps/`：商业许可
- 第三方组件与商标声明见 `THIRD_PARTY_NOTICES.md`（M0 建立）
