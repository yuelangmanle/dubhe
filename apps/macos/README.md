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
2. **进程生命周期** —— 退出 Dubhe 时关闭引擎，不留孤儿进程占内存
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
| 拖 gguf 到 Dock 图标 | `CFBundleDocumentTypes` 声明 + `application(_:open:)` | 换个模型不用先进界面 |
| **⌘O 打开模型…** | 文件菜单 + `NSOpenPanel`（默认落到 `~/.dubhe/models`）| Mac 用户找「打开」的第一反应就是 ⌘O |
| **Dock 右键菜单** | `applicationDockMenu`：打开模型 / 重启引擎 / 看日志 | 不用切回窗口也能救 |
| **生成完成通知** | 页面 `shellPost` 事件 → `UNUserNotificationCenter` + Dock 跳 + 角标 | 长回答几十秒，切走之后得知道完了没 |
| **崩溃自愈** | 界面起来之后引擎挂了 → 1/2/3 秒退避重启，第 4 次停手并给重试按钮 | 一次性崩溃不该让用户面对一个死页面 |
| **非 gguf 文件有话说** | 拖别的文件到 Dock 图标时弹一句"图片/文档请拖进窗口" | 原来是什么都不发生 |

## 通知这条路有个前提

`UNUserNotificationCenter` 要求 App 有**能注册到 LaunchServices 的 bundle**。
从 Finder / Dock 双击 `.app` 时没问题；直接跑
`build/Dubhe.app/Contents/MacOS/Dubhe` 这种裸二进制，系统会回
`Notifications are not allowed for this application`（实测）。

所以通知**不是唯一手段**：真发通知之前先做两件一定生效的事 ——
Dock 图标跳一下（`requestUserAttention`）+ 挂角标（`dockTile.badgeLabel`），
切回 App 时清掉。通知只是锦上添花，被系统拒了也不影响"知道它跑完了"。

ad-hoc 签名（`build.sh` 的默认）在部分系统上也会被拒。要稳定发通知得走
Developer ID + 公证。

## 构建

离线翻译使用本机 Python 3 与 llama.cpp。打包时可将官方 macOS llama.cpp 解压目录传给 `DUBHE_LLAMA_RUNTIME_DIR`，将 `llama-server`、依赖库和许可证纳入 DMG。模型仍由应用内商店按需下载。

在翻译页用「读文件」导入不超过 16 MB 的 PDF 或 DOCX。PDF 导出将原页与译文并排，长译文可续页。只有图像的扫描件需要先做 OCR。Word 导出保留段落、表格、样式、图片与公式，分页会随译文长度变化。绘图、文本框和批注保留原文。

翻译完成后可导出 Markdown、TXT、PDF 或 Word。Markdown 和 TXT 以 UTF-8 保存译文。粘贴的文本和纯文本文件导出为 PDF 或 Word 时，按标题、段落和列表排版。翻译时可切去其他页面，回来后可以查看进度。从失败或手动停止的位置继续时，会复用已经完成的段落。

在其他应用中按 **⌃⇧T** 翻译选中文字，或按 **⌘⌥⇧T** 打开截图框选。关闭主窗口后，Dubhe 留在菜单栏，快捷键继续工作。右上角的取景框图标提供翻译入口，也可以从这里退出 Dubhe 并释放模型。跨应用划词需要辅助功能权限。在浮窗里切换「原文」「译文」可核对识别结果并复制。

如果系统设置里的屏幕录制权限未生效，Dubhe 会打开 macOS 的一次性选择器。选择要截图的屏幕，再拖动框选文字。也可以按 macOS 的 **⌃⌘⇧4** 框选到剪贴板，然后选择「翻译剪贴板图片」。本地用临时签名构建的应用，每次更新都可能改变系统识别的身份。有稳定的 Apple 签名证书时，可以通过 `DUBHE_SIGN_IDENTITY` 在后续构建中沿用。

```sh
DUBHE_LLAMA_RUNTIME_DIR=/path/to/llama-macos-arm64 make dmg
```

```bash
./apps/macos/build.sh            # 出 build/Dubhe.app
./apps/macos/build.sh --run      # 出完直接打开
./apps/macos/build.sh --dmg      # 再打一个 build/Dubhe.dmg
```

产物结构：

```
build/Dubhe.app/Contents/MacOS/Dubhe          外壳（swiftc 编，~200 KB）
build/Dubhe.app/Contents/Helpers/dubhed     引擎
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
| `DUBHE_SHELL_AUTOCLOSE=3` | 界面加载完 3 秒后退出应用（验收引擎清理） |
| `DUBHE_SHELL_AUTORESTART=3` | 界面加载完 3 秒后重启引擎（验收 ⌘⇧R 那条路） |
| `DUBHE_SHELL_NOTIFY_TEST=3` | 3 秒后按"生成完成"走一遍通知路径（连角标和 Dock 跳） |
| `DUBHE_SHELL_EVAL="js"` | 加载完 2 秒后在页面里跑一段 JS，结果写进调试日志（仅 `DUBHE_SHELL_DEBUG=1` 时） |

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

- 公证：`build.sh` 现在只做 ad-hoc 签名。对外分发要换 Developer ID 并 `notarytool`。
  这条同时卡着"通知一定发得出来"（见上）。
- 拖文件到窗口**空白处**：现在网页侧接管了整窗拖放（`window` 上的
  dragover/drop），图片/文档/音频都能用；`<input type=file>` 也挂了
  `uiDelegate`。唯一没打通的是"把 .gguf 拖进窗口"——WebKit 自己吃掉了
  文件拖放，外壳拿不到路径。现在给的是明确提示（去「模型」页或拖 Dock 图标），
  要真做就得关掉 WebKit 的拖放、由外壳转发，代价是网页侧全部拖放要重写。
- 生成完成的通知在**没有 bundle 注册**的场景会被系统拒（见上）。
