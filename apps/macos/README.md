# apps/macos/ · Dubhe Desktop

macOS 宿主。许可：闭源（Proprietary，见根目录 LICENSE）。

## 关键约束

| 项 | 决定 | 依据 |
| --- | --- | --- |
| 最低系统 | macOS 14 Sonoma | Metal 特性与 WKWebView 能力 |
| 硬件 | Apple Silicon only（M1 起），Intel 不支持 | 统一内存是架构前提 |
| 分发 | DMG 直分发（签名 + 公证）为主；MAS 版本后置评估 | 沙盒限制模型目录与大文件下载 |
| 引擎承载 | 第一版：子进程 `dubhed`；M5 起：XPC Service `com.dubhe.engine` | 见下 |
| 与引擎通信 | 第一版：本地 HTTP（`127.0.0.1`）；M5 起：XPC + 共享内存 ring buffer | ADR-0005 |

## 状态

**外壳已跑通，正在按"像个 App"的标准补齐。** `./apps/macos/build.sh --run` 出一个
双击可开的 `build/Dubhe.app`，界面是引擎自带的 Web UI，打包时一起塞进
`Contents/Resources/web/`。

之所以不等 M5，是因为外壳缺的东西和引擎缺的东西不是同一批：

- 引擎侧已经齐了：三条推理链路（文本 / 视觉 / 语音）、`/v1/chat/completions`、
  `/api/models`、`/api/load`、`/api/state`、`/healthz`，以及一份能用的 Web UI。
- 用户侧还缺"双击就能开"：没有 App 包，就得先开终端敲 `dubhed -m ...`。

所以第一版只做**薄外壳**，只解决三件 Web 做不到的事：

1. **双击就能开** —— 不用先开终端
2. **进程生命周期** —— 关窗口就把引擎杀掉，不留孤儿进程占内存
3. **端口不撞车** —— 7777 被占就往后找（最多到 7799）

不在外壳里重写会话、模型库、设置。重写只会得到两份不同步的实现；等 M5 做
原生 UI 时再一次性替换（`Sources/main.swift` 里"起引擎、等就绪、杀进程"这三段
代码可以原样搬进 SwiftUI + XPC 版本）。

## 已经补上的（按提交顺序）

| 能力 | 做法 | 为什么要有 |
| --- | --- | --- |
| 应用图标 | `assets/branding/app-icon-1024.png` → `iconutil` → `AppIcon.icns` | Dock / 切换器里不能是白纸 |
| 引擎日志落盘 | stderr → `~/Library/Logs/Dubhe/engine.log`，首次启动截断、重启追加 | 白屏时唯一能看的东西 |
| 强杀残留清理 | `proc_listpids` + `proc_pidpath` 比对路径后 `SIGKILL` | SIGKILL 接不到信号，引擎会留下来占内存 |
| 单实例 | 发现同 bundle id 的另一个实例就激活它再退出 | 否则双击两次 = 两个引擎 |
| 记住上次的模型 | 每 10 s 读一次 `/api/state`，把 `model` 存进 UserDefaults | 每次打开都从"未加载"开始太烦 |
| 坏模型自动退回 | 启动失败且用的是记住的模型 → 忘掉它、无模型重启一次 | 别让用户卡在起不来的循环里 |
| 错误页可重试 | 转圈页出按钮，JS → 原生只认 `retry` / `openLog` 两个字符串 | 一个转不完的圈不是错误提示 |
| 重启引擎 | 菜单 ⌘⇧R，走和冷启动同一条路 | 换模型/卡住时不用退出重开 |
| 窗口标题同步 | KVO `WKWebView.title` | 实测变成「Dubhe · 天枢」 |

## 构建

```bash
./apps/macos/build.sh            # 出 build/Dubhe.app
./apps/macos/build.sh --run      # 出完直接打开
./apps/macos/build.sh --dmg      # 再打一个 build/Dubhe.dmg
```

产物结构：

```
build/Dubhe.app/Contents/MacOS/Dubhe          外壳（swiftc 编，~200 KB）
build/Dubhe.app/Contents/Resources/dubhed     引擎
build/Dubhe.app/Contents/Resources/web/       Web UI
build/Dubhe.app/Contents/Resources/AppIcon.icns
```

不依赖 xcodeproj：`main.swift` 是单文件 AppKit 程序，`swiftc -O` 直接编。
好处是 CI 上不需要 `xcodebuild`，坏处是没有 SwiftUI 预览和资源编译——
等 M5 换原生 UI 时再建工程。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `DUBHE_MODEL` | 启动时直接加载这个模型（优先于"上次用的模型"） |
| `DUBHE_BIN` | 指定引擎路径（默认优先用包内那份） |
| `DUBHE_SHELL_DEBUG=1` | 关键节点追加到 `/tmp/dubhe-shell.log` |
| `DUBHE_SHELL_SNAPSHOT=/tmp/x.png` | 界面加载完存一张快照（不需要屏幕录制权限） |
| `DUBHE_SHELL_AUTOCLOSE=3` | 界面加载完 3 秒后自关窗口（验收"关窗即退"） |
| `DUBHE_SHELL_AUTORESTART=3` | 界面加载完 3 秒后重启引擎（验收 ⌘⇧R 那条路） |

## 两个踩过的坑

**端口探测不能用 connect。** connect 只能发现"有人在听"，发现不了"被绑了但
没在听"；而且非阻塞 connect 在 macOS 上对没开的端口也返回 `EINPROGRESS`，
把它当成功会直接连到空气上（实测：转圈页闪一下就说"引擎界面打不开"）。
正确的问法是"我能不能像引擎那样 bind 住"——所以要带 `SO_REUSEADDR`，
和 `server/http.h:185` 保持一致；不带的话，刚关掉还留在 TIME_WAIT 的端口
会被误判成占用（实测被挤到 7778）。

**子进程不会跟着 App 一起死。** AppKit 只有走正常退出流程才会调
`applicationWillTerminate`；被 `kill`、崩溃、或从终端 Ctrl-C 时不会。不接
`SIGTERM`/`SIGINT`/`SIGHUP` 的话，`dubhed` 会变成孤儿进程继续占着端口和
几个 G 的统一内存。`main.swift` 里用 `DispatchSourceSignal` 接住了这三个信号，
走和正常退出一样的 `stopEngine()`（SIGTERM → 等 3 秒 → SIGKILL）。

**重启会把上一次失败的日志擦掉。** 第一版每次 `startEngine` 都截断日志，结果
"坏模型 → 自动退回 → 重启"这条路上，真正有用的那几行（`加载失败: 不是 GGUF
文件（magic 不匹配）`）正好被下一次启动抹掉。现在只在**进程内第一次**启动时
截断，之后追加一行 `──── 重启引擎 ────` 再接着写。顺带一个坑中坑：
`FileManager.createFile(atPath:contents:)` 对已存在的文件是**覆盖**，不能拿它
当"确保文件存在"用。

## 待做

- 拖拽文件到窗口：现在靠 Web UI 自己的 `<input type=file>`，拖进输入框能用，
  但"拖到 Dock 图标上"这条路径还没接（`application(_:open:)` 收到路径后没有
  地方转交给界面）。
- 生成完成的通知：窗口不在前台时没有任何提示。WKWebView 的 Web Notification
  需要自己实现 delegate，工作量比看起来大，先记着。
- 公证：`build.sh` 现在只做 ad-hoc 签名。对外分发要换 Developer ID 并 `notarytool`。
- 引擎崩溃后的自动重试：现在只给"重试"按钮，不会自己拉起来。做的话要限次数，
  否则一个必崩的模型会变成重启风暴。
