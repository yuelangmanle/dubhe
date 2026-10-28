# engine/ · Aether Engine

推理内核。许可：Apache-2.0。技术规格见 [AETHER-DEV-001](../docs/01-推理引擎开发书-索引.md)。

## 边界

**负责**：模型加载（GGUF / MLX / safetensors）、IR 构建、调度、执行、采样、约束解码、推测解码、序列状态管理、Kernel 规划与自动调优。

**不负责**：UI、模型下载、鉴权、计费、会话持久化的产品逻辑（这些在 `sdk/` 与 `apps/`）。

## 目录

```
include/aether/     公开 C ABI（唯一对外接口）
src/
  runtime/          scheduler · state · execution · sampler · constrained · speculative
  model/            ir · gguf · mlx · safetensors · tokenizer
  quant/            schemes · calibrate
  backends/         metal · vulkan · arm · qnn · mediatek
  autotune/         database · profiler
tests/              单测 · 数值 parity · fuzz
```

## 三条不可违反的约束

1. **公开头文件里不出现 C++ 类型、STL 容器、异常。** 所有错误通过 `ae_status_t`。
2. **decode hot path 不穿过 Python，也不做逐 token 的跨进程回调。**
3. **任何 Kernel 变更必须附 AetherBench 前后对比**，否则 review 打回。

## 状态

M0 阶段：只有目录与 schema 设计，无可构建产物。

## 上游参考

llama.cpp（GGUF/量化/CPU-Metal 参考）、MLX（Apple 原生与数值 oracle）、Locals（small-M Kernel 与 agent cache 思路）、vLLM Metal（paged state 与 scheduler）、MNN（ARM 低比特）。复用边界与 provenance 记录见 `THIRD_PARTY_NOTICES.md`。
