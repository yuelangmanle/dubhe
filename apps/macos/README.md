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

**外壳已跑通（M1 期间提前起步）。** 现在 `./apps/macos/build.sh --run` 就能出一个
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
```

不依赖 xcodeproj：`main.swift` 是单文件 AppKit 程序，`swiftc -O` 直接编。
好处是 CI 上不需要 `xcodebuild`，坏处是没有 SwiftUI 预览和资源编译——
等 M5 换原生 UI 时再建工程。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `DUBHE_MODEL` | 启动时直接加载这个模型；不给就在界面里选 |
| `DUBHE_BIN` | 指定引擎路径（默认优先用包内那份） |
| `DUBHE_SHELL_DEBUG=1` | 关键节点追加到 `/tmp/dubhe-shell.log` |
| `DUBHE_SHELL_SNAPSHOT=/tmp/x.png` | 界面加载完存一张快照（不需要屏幕录制权限） |
| `DUBHE_SHELL_AUTOCLOSE=3` | 界面加载完 3 秒后自关窗口（验收"关窗即退"用） |

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

## 待做

- 引擎日志：现在整份丢给 `/dev/null`。界面上"系统日志"面板看的是引擎内存里的
  环形缓冲，够用；但外壳自身出问题（比如白屏）时没有地方看。下一步给一个
  "打开日志"菜单项，把 stderr 落到 `~/Library/Logs/Dubhe/`。
- 崩溃残留清理：App 被强杀（SIGKILL）时接不到信号，`dubhed` 会留下来。
  下次启动应该扫一遍自家 `Resources/dubhed` 的残留进程并清掉。
- 拖拽文件到窗口：现在是 Web UI 自己处理的 `<input type=file>`，拖拽也能用，
  但拿不到"拖到 Dock 图标上"这条路径。
- 公证：`build.sh` 现在只做 ad-hoc 签名。对外分发要换 Developer ID 并 `notarytool`。
