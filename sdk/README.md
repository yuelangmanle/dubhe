# sdk/ · AetherKit

引擎对外的稳定接口层。许可：Apache-2.0。

## 组成

| 目录 | 内容 |
| --- | --- |
| `include/aether.h` | 稳定 C ABI（与 `engine/include/aether/` 同源，此处为发布快照） |
| `swift/` | Swift Package：`AetherKit`，供 macOS 宿主与第三方 App 使用 |
| `kotlin/` | Android AAR：JNI 封装 + Kotlin 协程接口 |
| `tests/abi/` | ABI 兼容性测试（`abi-compliance-checker`） |

## 版本策略

- 跟随引擎 MINOR 版本发布；**ABI 破坏性变更必须升 MAJOR 并写 ADR**。
- 每个发布版本在 `sdk/abi-baseline/` 存一份 ABI dump，CI 与之比对。

## 设计原则

1. **薄**：不做业务逻辑，只做生命周期、错误映射、内存管理、线程调度封装。
2. **不隐藏代价**：所有可能阻塞的调用都是 `async` 或显式标注；不做"看起来同步"的假封装。
3. **可诊断**：暴露性能计数器与错误上下文，让上层能回答"为什么慢了"。

## 状态

M0：接口草案讨论中，未发布。
