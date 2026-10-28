# RFC · 方案提案

> ADR 记录**已经做出的决策**，RFC 记录**还没定的方案**。
> 一个方案进入实现前，如果它涉及跨模块接口、格式定义、性能契约，必须先有 RFC。

## 流程

```
Draft → Review（≥3 人，含 1 名相关模块 owner）→ Accepted → 实现 → Implemented
                                             ↘ Rejected（留档，不删除）
```

## 命名

```
RFC-0001-aetherpack-format.md
RFC-0002-ipc-protocol.md
RFC-0003-constrained-decoding-bytecode.md
```

## 模板

```markdown
# RFC-000N: 标题

- 状态：Draft / Review / Accepted / Rejected / Implemented
- 作者：
- 日期：
- 关联：ADR-000X / issue #N

## 问题
（现在哪里不够用，具体到场景）

## 提案
（方案本身，含接口定义、数据结构、伪代码）

## 备选方案与取舍
| 方案 | 优点 | 缺点 |

## 影响面
（涉及哪些模块、是否破坏兼容、迁移成本）

## 验证方式
（怎么证明它有效：基准、测试、fuzz）

## 未决问题
```

## 已登记

| RFC | 标题 | 状态 |
| --- | --- | --- |
| — | 暂无（M0 期间产生第一批） | — |

## M0 期间预期产出

| 预期 RFC | 内容 | 触发时点 |
| --- | --- | --- |
| AetherPack v1 格式 | manifest / tensor map / prepack metadata 的字段定义 | M0 后期 |
| IPC 协议 v1 | 消息类型、背压、错误码、版本协商 | M1 前期 |
| AetherBench 结果 schema | 单次 run 的字段与聚合口径 | M0（任务 #7） |
| Kernel autotune DB 格式 | key 组成、失效策略、迁移 | M1 |
