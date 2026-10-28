# ADR-0003: IR 拆分为 Compute / Quant / State 三图

- 状态：Accepted
- 日期：2026-10-29
- 决策者：Tech Lead
- 关联：DUBHE-DEV-001 §4

## 背景

2026 年的模型已经不是"纯 Transformer + 一段连续 KV"。同一批模型里同时存在：full attention、sliding attention、Gated DeltaNet、Mamba/SSM、MLA，以及它们的混合体。

如果 IR 只有一张计算图，把 KV cache 当成"计算图的一个附带张量"，那么每新增一种 state 类型（GDN、SSM、draft state、speculative rollback cursor），调度器与 cache manager 都要被改一次。这在 3 个模型上能忍，在 30 个模型上必然崩。

同时，量化语义也被混在计算图里：`QuantLinear` 既表达"做矩阵乘"又隐含"权重是 4-bit affine group=64"。结果 Kernel 与文件格式耦合，GGUF 的 K-quant 与 MLX 的 affine 要写两套。

## 决策

**IR 拆成三张正交图**：

| 图 | 表达什么 | 不表达什么 |
| --- | --- | --- |
| **ComputeGraph** | 算子与数据流（matmul、attention、norm、rope…） | 权重怎么存、state 怎么活 |
| **QuantGraph** | 每个张量的 storage / dequant / accumulate 语义（scheme、group size、scale dtype、累加精度） | 算子拓扑 |
| **StateGraph** | KV / GDN / SSM / sliding / draft / prefix checkpoint 的生命周期与版本 | 具体数值计算 |

三者通过**张量 ID** 关联，而不是通过指针或结构体嵌套。

## 备选方案

| 方案 | 优点 | 缺点 | 为什么没选 |
| --- | --- | --- | --- |
| 单图（llama.cpp / ggml 风格） | 实现简单、调试直观 | state 与量化语义被压进算子，扩展即改核心 | 长期维护成本不可控 |
| 双图（Compute + State） | 解决 state 问题 | 量化仍与算子耦合，格式中立做不到 | 解决一半 |
| **三图** | 三者可独立演进；Kernel 只看 Compute+Quant 切片，调度只看 State | 实现复杂度高，需要 IR 校验器 | **本方案** |
| 直接用 ONNX / StableHLO | 生态标准 | 缺少 state 生命周期与量化 block 语义，需大量扩展 | 扩展量 > 自研 |

## 后果

正面：

- 新增一种 state 类型 = 在 StateGraph 加一类节点 + 在 backend 实现对应 kernel，调度器不用改。
- 同一份 ComputeGraph 可以挂不同的 QuantGraph（GGUF Q4_K_M / MLX affine 4-bit），Kernel 只认 block descriptor。
- 可以做"同一 tensor 多格式 bit-exact 测试"，量化正确性可被独立验证。

负面 / 代价：

- 需要写 IR 校验器（三图一致性、张量 ID 对齐、生命周期完整性），否则 bug 极难定位。
- 编译/加载路径变长，模型加载时间增加，需要缓存编译结果。
- 团队必须理解三图分离的约束，新人上手成本更高。

需要跟进：

- [ ] 三图 schema 定稿 + JSON Schema 校验（M0 任务 #11）
- [ ] IR 校验器：三图张量 ID 双向对齐检查（M1）
- [ ] 加载期编译结果缓存（避免每次启动重编译）

## 复审条件

- 若 M2 结束时 StateGraph 仍只覆盖 full attention 一种 state，说明抽象过度，考虑合并回双图；
- 若 IR 校验器维护成本超过收益（每月 > 20h 修校验器 bug）。
