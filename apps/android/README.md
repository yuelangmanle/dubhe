# apps/android/ · Aether Mobile

Android 宿主（Kotlin + Compose）。许可：商业。

## 关键约束

| 项 | 决定 | 依据 |
| --- | --- | --- |
| 最低系统 | Android 12（API 31） | 前台服务与 Vulkan 驱动成熟度 |
| ABI | `arm64-v8a` only | 32 位设备无法承载 4B 模型 |
| 内存门槛 | 建议 8 GB 起，6 GB 可用但只跑 ≤2B | 实测经验 |
| 引擎承载 | 前台 Service，独立进程 `:engine` | ADR-0005 |
| 与引擎通信 | AIDL + ashmem / `HardwareBuffer` | ADR-0005 |
| 前台服务类型 | `dataSync`（下载）/ `specialUse`（推理），Android 14 需显式声明 | Play 政策 |
| 模型存储 | `filesDir/models`（不放 external，避开 scoped storage） | — |

## 支持机型分档（宣传口径必须与之一致）

| 档位 | 代表机型 | 能力 |
| --- | --- | --- |
| A 档 旗舰 | 骁龙 8 Elite / 天玑 9 系，12 GB+ | GPU + NPU 全开，4B 可用 |
| B 档 次旗舰 | 骁龙 8 Gen 2/3，12 GB | GPU 可用，4B 勉强，2B 舒适 |
| C 档 兼容 | 骁龙 7 系 / 天玑 8 系，8 GB | 仅 CPU，≤2B |
| 其他 | 低于上述 | 尽力而为，不承诺 |

## 状态

M0：最小 demo（前台 Service + JNI 加载 GGUF）。M3 交付 Beta。
