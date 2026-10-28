# apps/macos/ · Aether Desktop

macOS 宿主（SwiftUI）。许可：商业。

## 关键约束

| 项 | 决定 | 依据 |
| --- | --- | --- |
| 最低系统 | macOS 14 Sonoma | Metal 特性与 SwiftUI 能力 |
| 硬件 | Apple Silicon only（M1 起），Intel 不支持 | 统一内存是架构前提 |
| 分发 | DMG 直分发（签名 + 公证）为主；MAS 版本后置评估 | 沙盒限制模型目录与大文件下载 |
| 引擎承载 | XPC Service `com.aether.engine` | ADR-0005 |
| 与引擎通信 | XPC + 共享内存 ring buffer | ADR-0005 |

## 模块

```
App/            SwiftUI 主界面（会话、模型库、设置、Agent 面板）
EngineBridge/   XPC 客户端、ring buffer 读取、错误恢复
ModelManager/   仓库索引、分片下载、断点续传、哈希校验
SessionStore/   会话 CRUD、分支、导入导出、加密
AgentRunner/    多 agent 并发调度与进度展示
```

## 状态

M0：技术选型验证（SwiftUI + XPC 最小 demo）。M2 交付 Alpha。
