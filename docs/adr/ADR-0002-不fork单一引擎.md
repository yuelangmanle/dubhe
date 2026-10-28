# ADR-0002: 不 fork 单一现有引擎，新建格式无关运行时

- 状态：Accepted
- 日期：2026-10-29
- 决策者：Tech Lead
- 关联：DUBHE-DEV-001 §1、§2

## 背景

目标是"在 Mac 与 Android 上同时超越现有推理引擎"。直觉方案是选一个最强的开源项目 fork 下来补齐短板。但候选项目的优势高度错位：

| 项目 | 强项 | 缺什么 |
| --- | --- | --- |
| llama.cpp | GGUF、量化覆盖、硬件可移植性 | Apple small-M 优化、paged state、agent 并发调度 |
| MLX / mlx-lm | Apple 原生、训练与量化 | 不是 Android runtime、不是 GGUF runtime |
| Locals | Apple small-M Kernel、agent cache | 只做 MLX、不做 GGUF/Android |
| vLLM Metal | paged serving、scheduler、hybrid state | 只面向 Apple |
| MNN / ExecuTorch / LiteRT | Android 硬件覆盖 | 没有 Mac small-M 的完整优化路线 |

fork 任何一个，都等于把其余四个的优势变成要自己重写的欠账，而且是在一个**为别的目标设计**的架构上重写。

## 决策

**不 fork。** 新建独立运行时核心，通过**格式无关 IR** 与**可插拔后端**获得能力；对现有项目采取"选择性吸收"——复用解析器、量化定义、Kernel 思想与正确性参考，而不是整体继承其架构。

## 备选方案

| 方案 | 优点 | 缺点 | 为什么没选 |
| --- | --- | --- | --- |
| fork llama.cpp 并魔改 Metal | 上手最快，GGUF 与后端现成 | 架构以 ggml graph 为中心，state/scheduler 抽象弱，改造成本最终超过重写；上游 rebase 地狱 | 长期成本最高 |
| fork MLX + 自写 Android | Apple 侧最强 | MLX 深度绑定 Apple，Android 等于从零；且无法直读 GGUF | 只解决一半问题 |
| 基于 vLLM Metal 扩展 | serving 架构最完整 | 强绑定 MLX 与 Apple；Android 无路径 | 与"双端"目标冲突 |
| **新建运行时 + 选择性吸收** | 架构自主，双端可共用上层语义 | 前期慢，需要自己写 loader/IR/Kernel | **本方案** |

## 后果

正面：

- 上层（调度、state、模型语义）Mac 与 Android 共用，只有 Kernel 分叉——这是唯一能在 8 人规模下同时打两端的结构。
- 可以自由替换任一后端而不动上层。
- 不存在上游 rebase 债务。

负面 / 代价：

- P0/P1 比"fork 一个能跑的"慢 2–3 个月。
- 必须自己维护 GGUF/MLX 解析器（可参考实现，但不能直接搬架构）。
- 需要自建正确性 oracle（reference CPU 路径）来保证不走偏。

需要跟进：

- [ ] P0 建立 reference CPU 执行路径作为数值基准（M0 任务 #12）
- [ ] 明确各开源项目的"可复用边界"清单（代码 / 思想 / 仅参考）
- [ ] 每个吸收点记录 provenance，进入 SBOM

## 复审条件

- 若 M1 结束时自研 loader + Kernel 路线落后最佳 25% 以上，且降级预案（回退 MLX backend）也无法交付 Alpha；
- 若上游出现同时覆盖 GGUF + MLX + Android 且架构开放的实现。
