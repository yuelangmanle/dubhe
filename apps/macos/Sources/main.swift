// Dubhe Desktop — macOS 外壳
//
// 这一版刻意做得很薄：起一个 `dubhed` 子进程，等它就绪，然后用 WKWebView
// 打开它自己那份 Web UI。理由写在 apps/macos/README.md 里：
//
//   引擎已经有完整的本地 HTTP 接口（/v1/chat/completions、/api/models、
//   /api/load、/api/state），UI 也已经是一份能用的实现（web/，1700 行）。
//   在"外壳"这一层重新写一遍会话、模型库、设置，只会得到两份不同步的实现。
//   所以第一版外壳只解决**三件 Web 做不到的事**：
//
//     1. 双击就能开 —— 用户不需要先开终端敲 `dubhed -m ...`
//     2. 进程生命周期 —— 关窗口就把引擎杀掉，不留孤儿进程占内存
//     3. 端口不撞车 —— 7777 被占就往后找，而不是起来就报错
//
// 等 M5 真正做原生 UI 时，这一层换成 SwiftUI + XPC（ADR-0005），
// 但"起引擎、等就绪、杀进程"这三件事的代码可以原样搬过去。
import Cocoa
import Carbon.HIToolbox
import UserNotifications
import UniformTypeIdentifiers
import WebKit
import Vision

// ---------------------------------------------------------------- 端口探测
//
// 判断"这个端口能不能用"要和引擎真正 bind 时的条件一致，否则会得到两种
// 错误结论：
//
//   * 用 connect 试连 —— 只能发现"有人在听"，发现不了"被绑了但没听"；
//   * bind 但不带 SO_REUSEADDR —— 会把刚关掉、还留在 TIME_WAIT 的端口
//     误判成占用（上一轮就是这么被挤到 7778 去的）。
//
// 引擎（server/http.h:185）bind 时是带 SO_REUSEADDR 的，所以这里也带。
// 带了之后 bind 失败只意味着"另有一个 socket 正绑着这个端口"，正是我们要问的。
private func portIsFree(_ port: UInt16) -> Bool {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0 { return false }
    defer { close(fd) }
    var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let rc = withUnsafePointer(to: &addr) { p -> Int32 in
        p.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    return rc == 0
}

// ---------------------------------------------------------------- 找 dubhed
//
// 新包把引擎放在 Contents/Helpers。也兼容旧包、DUBHE_BIN 和开发构建。
private func resolveEngine() -> String? {
    let fm = FileManager.default
    var candidates = [Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/dubhed").path]
    if let res = Bundle.main.resourcePath {
        candidates.append(res + "/dubhed")
    }
    if let env = ProcessInfo.processInfo.environment["DUBHE_BIN"] {
        candidates.append(env)
    }
    candidates.append(fm.currentDirectoryPath + "/build/dubhed")
    return candidates.first { fm.isExecutableFile(atPath: $0) }
}

private func resolveWebRoot() -> String? {
    let fm = FileManager.default
    var candidates: [String] = []
    if let res = Bundle.main.resourcePath {
        candidates.append(res + "/web")
    }
    candidates.append(fm.currentDirectoryPath + "/web")
    return candidates.first { fm.fileExists(atPath: $0 + "/index.html") }
}

// ---------------------------------------------------------------- 日志
//
// 引擎的 stderr 以前整份丢给 /dev/null。这样 Console 里干净，但外壳出问题
// （白屏、引擎起不来）时就没有任何地方可查。现在落到
// ~/Library/Logs/Dubhe/engine.log：每次启动重写，超过 8 MB 时留一份 .1。
private func engineLogURL() -> URL {
    let base = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
        ?? URL(fileURLWithPath: NSHomeDirectory() + "/Library")
    let custom = ProcessInfo.processInfo.environment["DUBHE_DATA_DIR"] ?? ""
    let dir = custom.isEmpty ? base.appendingPathComponent("Logs/Dubhe", isDirectory: true)
        : URL(fileURLWithPath: custom, isDirectory: true).appendingPathComponent("logs", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent("engine.log")
}

// ---------------------------------------------------------------- 进程
//
// libproc 没有 Swift overlay，这里直接绑 C 符号。只为了两件事：
//   proc_pidpath  → 判断某个 pid 是不是"我们自己那份 dubhed"
//   proc_listpids → 遍历所有 pid
// 用 kill(pid, 0) + 路径比对，而不是记 pid 文件：pid 会被系统复用，
// 路径不会。
@_silgen_name("proc_listpids")
private func c_proc_listpids(_ type: UInt32, _ typeinfo: UInt32,
                             _ buffer: UnsafeMutableRawPointer?, _ size: Int32) -> Int32
@_silgen_name("proc_pidpath")
private func c_proc_pidpath(_ pid: Int32, _ buffer: UnsafeMutableRawPointer?, _ size: UInt32) -> Int32

private let PROC_ALL_PIDS: UInt32 = 1

private func executablePath(of pid: pid_t) -> String? {
    var buf = [CChar](repeating: 0, count: 4096)
    let n = c_proc_pidpath(pid, &buf, UInt32(buf.count))
    guard n > 0 else { return nil }
    return String(cString: buf)
}

// 上一次被 SIGKILL 强杀时，dubhed 会留下来（信号接不到）。它占着端口和
// 几个 G 的统一内存，还会让下一次启动以为自己"端口被占"。这里在启动时
// 扫一遍：凡是可执行文件路径等于我们自己那份 dubhed 的进程，全部清掉。
private func killStaleEngines(bundleEnginePath: String) -> Int {
    let stride = MemoryLayout<pid_t>.size
    var pids = [pid_t](repeating: 0, count: 8192)
    let bytes = c_proc_listpids(PROC_ALL_PIDS, 0, &pids, Int32(pids.count * stride))
    guard bytes > 0 else { return 0 }
    var killed = 0
    for i in 0..<(Int(bytes) / stride) {
        let pid = pids[i]
        if pid <= 0 || pid == getpid() { continue }
        guard executablePath(of: pid) == bundleEnginePath else { continue }
        kill(pid, SIGKILL)
        killed += 1
    }
    return killed
}

// ---------------------------------------------------------------- 调试日志
//
// 外壳平时不该往任何地方写日志，但"双击之后一直转圈"这种问题在 Console 里
// 找不到线索（引擎的 stderr 已经丢进 /dev/null 了）。DUBHE_SHELL_DEBUG=1
// 时把关键节点追加到 /tmp/dubhe-shell.log，排查完就关掉。
private let debugOn = ProcessInfo.processInfo.environment["DUBHE_SHELL_DEBUG"] == "1"
private func dbg(_ s: String) {
    guard debugOn else { return }
    let line = "\(Date().timeIntervalSince1970) \(s)\n"
    let path = "/tmp/dubhe-shell.log"
    if let fh = FileHandle(forWritingAtPath: path) {
        fh.seekToEndOfFile()
        fh.write(line.data(using: .utf8) ?? Data())
        fh.closeFile()
    } else {
        try? line.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

// ---------------------------------------------------------------- 主界面
//
// 引擎的退出原因必须落到 engine.log 里，不能只在 DUBHE_SHELL_DEBUG 打开时
// 才记。用户报上来的现象是"聊到一半报错，然后自己重启了"，而日志里只有
// 一行「──── 重启引擎 ────」—— 看不出是崩溃、被 OOM 杀掉，还是我们自己
// 重启的。这一段就是补上那个缺口：谁、什么码、什么时候。
private func logEngineEvent(_ s: String) {
    let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/Dubhe", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("engine.log")
    guard let data = "\n[dubhe-shell] \(s)\n".data(using: .utf8) else { return }
    if let fh = try? FileHandle(forWritingTo: url) {
        fh.seekToEndOfFile()
        fh.write(data)
        fh.closeFile()
    } else {
        try? data.write(to: url)
    }
}

// 单实例锁。
//
// 用 flock 而不是 pid 文件、也不查 LaunchServices：锁由内核在进程退出时
// 自动释放，SIGKILL 也不会留下"死锁文件"，更不怕 pid 被复用。
// fd 故意不关 —— 一关锁就没了。
private var instanceLockFD: Int32 = -1
private func acquireInstanceLock() -> Bool {
    let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Dubhe", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let path = dir.appendingPathComponent("instance.lock").path
    let fd = open(path, O_CREAT | O_RDWR, 0o644)
    if fd < 0 { return true }   // 锁文件都建不了就别把人挡在门外
    if flock(fd, LOCK_EX | LOCK_NB) != 0 {
        close(fd)
        return false
    }
    instanceLockFD = fd
    return true
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate,
                         WKNavigationDelegate, WKScriptMessageHandler, WKUIDelegate,
                         UNUserNotificationCenterDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var engine: Process?
    private var port: UInt16 = 0
    private var pollTimer: DispatchSourceTimer?
    private var pollsLeft = 0
    private var uiLoaded = false
    private var shuttingDown = false
    private var signalSources: [DispatchSourceSignal] = []
    private var logHandle: FileHandle?
    private var titleObserver: NSKeyValueObservation?
    private var stateTimer: DispatchSourceTimer?
    private var usedRememberedModel = false
    private var logInitialized = false
    private var pendingModelURL: URL?
    // 引擎在界面已经起来之后挂掉，自动拉起的次数与上次时间。
    // 连续崩要在有限次内停手，否则一个必崩的模型会变成重启风暴。
    private var crashCount = 0
    private var lastCrashAt = Date.distantPast
    // 界面**至少**成功加载过一次。不能用 uiLoaded 判断"这是运行中崩溃还是
    // 启动失败"：showSplash 会把 uiLoaded 置回 false，于是崩溃重启循环里
    // 第 2 次以后会被误判成启动失败，计数永远涨不到上限。
    private var everLoadedUI = false
    private var notifyAllowed = false
    private var checkingForUpdate = false
    private var translationHotKeyHandler: EventHandlerRef?
    private var translationHotKeys: [EventHotKeyRef] = []
    private var unavailableTranslationHotKeys: [String] = []
    private var capture: Process?
    private var recognizingScreenshot = false
    private var overlay: NSPanel?
    private var overlayText: NSTextView?
    private var overlayTabs: NSSegmentedControl?
    private var overlayCopy: NSButton?
    private var overlayScreenSettings: NSButton?
    private var overlayClipboard: NSButton?
    private var overlaySource = ""
    private var overlayResult = ""
    private var overlayHasTranslation = false
    private var overlayTranslating = false

    // 上次在界面里加载过的模型。存路径而不是名字：同名模型可能在两个目录里。
    private static let lastModelKey = "lastModelPath"

    private func rememberedModel() -> String? {
        guard let p = UserDefaults.standard.string(forKey: Self.lastModelKey), !p.isEmpty else {
            return nil
        }
        // 文件被删了就别再拿它去启动，否则每次开都失败一次。
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        dbg("launch pid \(getpid()) bundle \(Bundle.main.bundleIdentifier ?? "?")")
        // 第二次双击时不要再起一份引擎：macOS 一般会把已经开着的那个激活，
        // 但从终端直接跑二进制、或者两份不同路径的拷贝，都会真的起两个
        // 引擎（各占一个端口、各吃一份内存）。这里主动退让。
        // 只认真正活着的那个实例。
        //
        // 原来用 NSRunningApplication.runningApplications(withBundleIdentifier:)
        // 判断"是不是已经开了一份"。这条路有两个坑，而且都会变成"双击没反应"：
        //   1. LaunchServices 会留下"进程已经没了"的残留记录；
        //   2. 就算加 kill(pid, 0) 复核也不够 —— 那个 pid 可能早被别的进程复用，
        //      探活成功，于是一份正常运行的新实例把自己 exit 掉。
        // 换成 flock：锁由内核在进程退出时自动释放，pid 复用、LS 残留都影响不到。
        if ProcessInfo.processInfo.environment["DUBHE_SHELL_ALLOW_MULTI"] != "1",
           !acquireInstanceLock() {
            dbg("another instance holds the lock, activating it and bowing out")
            NSRunningApplication
                .runningApplications(withBundleIdentifier: Bundle.main.bundleIdentifier ?? "com.dubhe.desktop")
                .first { $0.processIdentifier != getpid() }?
                .activate()
            exit(0)
        }

        installSignalHandlers()
        setupNotifications()
        buildMenu()
        buildWindow()
        registerTranslationHotKeys()
        // 由"打开 .gguf"触发启动时，第一个要加载的就是它，别再先用
        // 记住的那个起一遍再重启。
        if let u = pendingModelURL {
            UserDefaults.standard.set(u.path, forKey: Self.lastModelKey)
            pendingModelURL = nil
        }
        startEngine()
    }

    // 被 kill / Ctrl-C / 关终端时，AppKit 不会走 applicationWillTerminate，
    // 引擎就会变成孤儿进程继续占着端口和几个 G 的内存。这里把这三个信号
    // 接过来，走和正常退出一样的收尾。
    private func installSignalHandlers() {
        for sig in [SIGTERM, SIGINT, SIGHUP] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler { [weak self] in
                self?.shuttingDown = true
                self?.stopEngine()
                exit(0)
            }
            src.resume()
            signalSources.append(src)
        }
    }

    // 关窗口 = 退出（这是一个单窗口工具，不是文档型 App）
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    // ------------------------------------------------------------ 生成完成的通知
    //
    // 长回答动辄几十秒，用户切去干别的，回来时既不知道生成完没完、也不知道
    // 结果如何。这里做两件事（都只在"人不在这个窗口前面"时发生）：
    //   1. 系统通知（带回答开头一段，扫一眼就知道值不值得切回来）
    //   2. Dock 图标跳一下 + 挂角标
    //
    // UNUserNotificationCenter 在**没有 bundle** 的裸二进制里会抛异常
    // （bundleProxyForCurrentProcess 为 nil），所以先确认 bundleIdentifier 在。
    private func setupNotifications() {
        guard Bundle.main.bundleIdentifier != nil else { return }
        let center = UNUserNotificationCenter.current()
        center.delegate = self
        center.requestAuthorization(options: [.alert, .sound]) { [weak self] granted, err in
            DispatchQueue.main.async { self?.notifyAllowed = granted }
            dbg("notify auth granted=\(granted) err=\(err?.localizedDescription ?? "-")")
        }
    }

    // 前台时不弹横幅：用户正看着屏幕，通知只会挡住内容。
    // 这里的判断和 notifyTurnEnd 里的不重复——通知可能在派发途中 App 被激活。
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                willPresent notification: UNNotification,
                                withCompletionHandler completionHandler:
                                    @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler(NSApp.isActive ? [] : [.banner, .sound])
    }

    private func notifyTurnEnd(_ body: String) {
        let front = NSApp.isActive && window != nil && window.isKeyWindow
                    && !window.isMiniaturized
        if front { return }
        dbg("notify turn end (front=\(front), allowed=\(notifyAllowed))")
        NSApp.requestUserAttention(.informationalRequest)
        NSApp.dockTile.badgeLabel = "1"
        guard notifyAllowed else { return }
        let c = UNMutableNotificationContent()
        c.title = "Dubhe · 天枢"
        c.body = body
        c.sound = .default
        UNUserNotificationCenter.current().add(
            UNNotificationRequest(identifier: UUID().uuidString, content: c, trigger: nil))
    }

    func applicationDidBecomeActive(_ note: Notification) {
        // 用户回来了，角标就该消失——它表达的是"有你没看的东西"
        NSApp.dockTile.badgeLabel = nil
    }

    // 把 .gguf 拖到 Dock 图标上（或在 Finder 里"打开方式 → Dubhe"）。
    // 拖进来的文件直接当"这次要加载的模型"，走重启引擎那条路。
    func application(_ sender: NSApplication, open urls: [URL]) {
        guard let gguf = urls.first(where: { $0.pathExtension.lowercased() == "gguf" }) else {
            // 不是模型文件。以前这里直接 return，用户拖一张图到 Dock 图标上
            // 得到的是"什么都没发生"。说清楚能做什么，比沉默好。
            if webView != nil, let first = urls.first {
                let a = NSAlert()
                a.messageText = "只能这样打开 .gguf 模型"
                a.informativeText = "「\(first.lastPathComponent)」不是模型文件。"
                    + "图片、文档、音频请直接拖进对话窗口，或者在输入框左边点附件。"
                a.alertStyle = .informational
                a.runModal()
            }
            return
        }
        dbg("open file \(gguf.path)")
        // 如果是"用这个文件启动"的场景，窗口还没建好，先记下来，
        // 等 applicationDidFinishLaunching 里再决定用哪个模型起引擎。
        if webView == nil {
            pendingModelURL = gguf
            return
        }
        UserDefaults.standard.set(gguf.path, forKey: Self.lastModelKey)
        usedRememberedModel = false
        restartEngine()
    }

    func applicationWillTerminate(_ note: Notification) {
        shuttingDown = true
        translationHotKeys.forEach { UnregisterEventHotKey($0) }
        if let translationHotKeyHandler { RemoveEventHandler(translationHotKeyHandler) }
        if capture?.isRunning == true { capture?.terminate() }
        stopEngine()
    }

    // ------------------------------------------------------------ 菜单
    //
    // 没有主菜单的话，⌘C / ⌘V 在 WKWebView 的输入框里是不响应的 ——
    // 剪贴板快捷键靠菜单项的 keyEquivalent 走响应链。这几项必须自己建。
    private func buildMenu() {
        let main = NSMenu()

        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "关于 Dubhe", action: #selector(showAbout), keyEquivalent: "")
        appMenu.addItem(withTitle: "检查更新…", action: #selector(checkForUpdates), keyEquivalent: "").target = self
        appMenu.addItem(withTitle: "更新日志", action: #selector(showChangelog), keyEquivalent: "").target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "隐藏 Dubhe",
                        action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "隐藏其他",
                        action: #selector(NSApplication.hideOtherApplications(_:)),
                        keyEquivalent: "h").keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "打开模型目录", action: #selector(openModelDir), keyEquivalent: "")
        appMenu.addItem(withTitle: "打开引擎日志", action: #selector(openEngineLog), keyEquivalent: "")
        appMenu.addItem(withTitle: "在 Finder 中显示日志", action: #selector(revealEngineLog),
                        keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出 Dubhe",
                        action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)

        // 「文件」菜单：Mac 用户找「打开」的第一反应就在这里。⌘O 是系统级的
        // 肌肉记忆，只把它藏在应用菜单里等于没有。
        let fileItem = NSMenuItem()
        let file = NSMenu(title: "文件")
        file.addItem(withTitle: "打开模型…", action: #selector(openModel), keyEquivalent: "o")
        file.addItem(.separator())
        file.addItem(withTitle: "关闭窗口", action: #selector(NSWindow.performClose(_:)),
                     keyEquivalent: "w")
        fileItem.submenu = file
        main.addItem(fileItem)

        let editItem = NSMenuItem()
        let edit = NSMenu(title: "编辑")
        // 用字符串 selector 走响应链交给 WKWebView，自己不实现
        edit.addItem(withTitle: "撤销", action: NSSelectorFromString("undo:"), keyEquivalent: "z")
        edit.addItem(withTitle: "重做", action: NSSelectorFromString("redo:"), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "剪切", action: NSSelectorFromString("cut:"), keyEquivalent: "x")
        edit.addItem(withTitle: "拷贝", action: NSSelectorFromString("copy:"), keyEquivalent: "c")
        edit.addItem(withTitle: "粘贴", action: NSSelectorFromString("paste:"), keyEquivalent: "v")
        edit.addItem(withTitle: "全选", action: NSSelectorFromString("selectAll:"), keyEquivalent: "a")
        editItem.submenu = edit
        main.addItem(editItem)

        let translationItem = NSMenuItem()
        let translation = NSMenu(title: "翻译")
        let selection = translation.addItem(withTitle: "翻译选中文字", action: #selector(translateSelection),
                                            keyEquivalent: "t")
        selection.keyEquivalentModifierMask = [.control, .shift]
        selection.target = self
        let screenshot = translation.addItem(withTitle: "截图翻译…", action: #selector(captureAndOCR),
                                             keyEquivalent: "t")
        screenshot.keyEquivalentModifierMask = [.command, .shift, .option]
        screenshot.target = self
        let clipboard = translation.addItem(withTitle: "翻译剪贴板图片", action: #selector(translateClipboardImage),
                                            keyEquivalent: "")
        clipboard.target = self
        translationItem.submenu = translation
        main.addItem(translationItem)

        let viewItem = NSMenuItem()
        let view = NSMenu(title: "显示")
        view.addItem(withTitle: "重新载入", action: #selector(reload), keyEquivalent: "r")
        view.addItem(withTitle: "重启引擎", action: #selector(restartEngine),
                     keyEquivalent: "r").keyEquivalentModifierMask = [.command, .shift]
        view.addItem(.separator())
        view.addItem(withTitle: "放大", action: #selector(zoomIn), keyEquivalent: "+")
        view.addItem(withTitle: "缩小", action: #selector(zoomOut), keyEquivalent: "-")
        view.addItem(withTitle: "实际大小", action: #selector(zoomReset), keyEquivalent: "0")
        viewItem.submenu = view
        main.addItem(viewItem)

        let winItem = NSMenuItem()
        let win = NSMenu(title: "窗口")
        win.addItem(withTitle: "最小化", action: #selector(NSWindow.performMiniaturize(_:)),
                    keyEquivalent: "m")
        winItem.submenu = win
        main.addItem(winItem)

        NSApp.mainMenu = main
    }

    @objc private func reload() { webView.reload() }
    @objc private func zoomIn() { webView.pageZoom = min(3.0, webView.pageZoom + 0.1) }
    @objc private func zoomOut() { webView.pageZoom = max(0.5, webView.pageZoom - 0.1) }
    @objc private func zoomReset() { webView.pageZoom = 1.0 }

    @objc private func openEngineLog() {
        let url = engineLogURL()
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        NSWorkspace.shared.open(url)
    }

    @objc private func revealEngineLog() {
        NSWorkspace.shared.activateFileViewerSelecting([engineLogURL()])
    }

    @objc private func openModelDir() {
        let dir = NSHomeDirectory() + "/.dubhe/models"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(URL(fileURLWithPath: dir))
    }

    // ⌘O：选一个 .gguf 直接换模型。走的是和"拖到 Dock 图标"完全同一条路
    // （记住路径 → 重启引擎），所以两种入口不可能出现行为差异。
    @objc private func openModel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "选择一个 GGUF 模型"
        panel.prompt = "加载"
        // UTType(filenameExtension:) 对自定义扩展名会给出动态类型，
        // 只要文件真的是 .gguf 就能选中；万一系统解析不出来也不拦。
        if let t = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [t]
        }
        let start = NSHomeDirectory() + "/.dubhe/models"
        if FileManager.default.fileExists(atPath: start) {
            panel.directoryURL = URL(fileURLWithPath: start)
        }
        let handler: (NSApplication.ModalResponse) -> Void = { [weak self] resp in
            guard let self, resp == .OK, let url = panel.url else { return }
            dbg("open model \(url.path)")
            UserDefaults.standard.set(url.path, forKey: Self.lastModelKey)
            self.usedRememberedModel = false
            self.restartEngine()
        }
        if let w = window { panel.beginSheetModal(for: w, completionHandler: handler) }
        else { panel.begin(completionHandler: handler) }
    }

    // Dock 图标右键：不打开窗口也能换模型/重启/看日志。
    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        let m = NSMenu()
        for (title, sel) in [("打开模型…", #selector(openModel)),
                             ("重启引擎", #selector(restartEngine)),
                             ("打开引擎日志", #selector(openEngineLog))] {
            let it = NSMenuItem(title: title, action: sel, keyEquivalent: "")
            it.target = self
            m.addItem(it)
        }
        return m
    }

    // 关于面板里放真正有用的东西：版本、引擎端口、当前模型。用户来这儿
    // 通常就是为了抄这几个数去提 issue。
    @objc private func showAbout() {
        let ver = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        var lines = ["本地大模型推理引擎", "版本 \(ver)"]
        if port != 0 { lines.append("引擎端口 \(port)") }
        if let m = UserDefaults.standard.string(forKey: Self.lastModelKey), !m.isEmpty {
            lines.append("模型 \((m as NSString).lastPathComponent)")
        }
        NSApp.orderFrontStandardAboutPanel(options: [.credits: lines.joined(separator: "\n")])
    }

    @objc private func checkForUpdates() {
        guard !checkingForUpdate else { return }
        checkingForUpdate = true
        Task { @MainActor in
            defer { checkingForUpdate = false }
            let current = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
            let alert = NSAlert()
            var showRelease = false
            do {
                let release = try await ReleaseUpdates.latest()
                if let version = try release.newerVersion(than: current) {
                    alert.messageText = "Dubhe \(version) 可更新"
                    alert.informativeText = "当前版本 \(current)。可在 GitHub 下载新版 macOS 安装包。"
                    showRelease = true
                } else {
                    alert.messageText = "已是最新版本"
                    alert.informativeText = "当前版本 \(current)。"
                }
            } catch {
                alert.messageText = "未能完成更新检查"
                alert.informativeText = error.localizedDescription
                showRelease = true
            }
            if showRelease { alert.addButton(withTitle: "打开 GitHub") }
            alert.addButton(withTitle: "关闭")
            alert.beginSheetModal(for: window) { result in
                if showRelease && result == .alertFirstButtonReturn { NSWorkspace.shared.open(ReleaseUpdates.page) }
            }
        }
    }

    @objc private func showChangelog() {
        let alert = NSAlert()
        alert.messageText = "更新日志"
        if let url = Bundle.main.url(forResource: "CHANGELOG", withExtension: "md"),
           let content = try? String(contentsOf: url, encoding: .utf8) {
            let scroll = NSTextView.scrollableTextView()
            scroll.frame = NSRect(x: 0, y: 0, width: 570, height: 365)
            if let text = scroll.documentView as? NSTextView {
                text.isEditable = false
                text.font = .systemFont(ofSize: 13)
                text.textContainerInset = NSSize(width: 8, height: 8)
                text.string = content
            }
            alert.accessoryView = scroll
        } else {
            alert.informativeText = "无法读取随应用提供的更新日志。"
        }
        alert.addButton(withTitle: "关闭")
        alert.beginSheetModal(for: window)
    }

    private func buildWindow() {
        let rect = NSRect(x: 0, y: 0, width: 1100, height: 760)
        window = NSWindow(contentRect: rect,
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Dubhe"
        window.minSize = NSSize(width: 720, height: 520)
        window.delegate = self
        window.center()
        window.setFrameAutosaveName("DubheMainWindow")

        let cfg = WKWebViewConfiguration()
        // 关掉"用 ⌘ 点链接开新窗口"这类网页行为，桌面 App 里不该有
        cfg.preferences.javaScriptCanOpenWindowsAutomatically = false
        // 转圈页上的"重试"按钮没法直接调 Swift，走一条 JS → 原生的小通道。
        // 只认两个字符串，不做通用 bridge。
        let ucc = WKUserContentController()
        ucc.add(self, name: "shell")
        cfg.userContentController = ucc
        webView = WKWebView(frame: rect, configuration: cfg)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        // 文件选择必须挂 uiDelegate。
        //
        // WKWebView **默认不会**弹文件选择框：页面里的 <input type="file">
        // 点了以后什么都不会发生，而且不报错、不打日志。这就是"界面上传不了
        // 文件/图片"的根因 —— 网页在浏览器里正常，塞进 WKWebView 就哑了。
        // 要让它工作，必须实现 runOpenPanelWith 并用 NSOpenPanel 自己弹。
        webView.uiDelegate = self
        // 窗口标题跟着页面的 <title> 走，这样切到"模型"页时标题栏也跟着变，
        // Dock 的悬停提示也一样。WKWebView.title 是 KVO 可观察的，不用额外请求。
        titleObserver = webView.observe(\.title, options: [.new]) { [weak self] wv, _ in
            guard let self, self.uiLoaded else { return }
            let t = (wv.title?.isEmpty == false) ? wv.title! : "Dubhe"
            dbg("title -> \(t)")
            self.window.title = t
        }
        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        showSplash("正在启动本地引擎…")
    }

    private func registerTranslationHotKeys() {
        let shortcuts = [
            (id: UInt32(1), modifiers: UInt32(controlKey | shiftKey), label: "⌃⇧T 划词翻译"),
            (id: UInt32(2), modifiers: UInt32(cmdKey | optionKey | shiftKey), label: "⌘⌥⇧T 截图翻译"),
        ]
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let signature: OSType = 0x44425452 // DBTR
        let target = GetApplicationEventTarget()
        let status = InstallEventHandler(target, { _, event, context in
            guard let event, let context else { return OSStatus(eventNotHandledErr) }
            var key = EventHotKeyID()
            guard GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                    nil, MemoryLayout<EventHotKeyID>.size, nil, &key) == noErr,
                  key.signature == 0x44425452 else { return OSStatus(eventNotHandledErr) }
            let app = Unmanaged<AppDelegate>.fromOpaque(context).takeUnretainedValue()
            switch key.id {
            case 1: app.translateSelection()
            case 2: app.captureAndOCR()
            default: return OSStatus(eventNotHandledErr)
            }
            return noErr
        }, 1, &eventType, Unmanaged.passUnretained(self).toOpaque(), &translationHotKeyHandler)
        guard status == noErr else {
            unavailableTranslationHotKeys = shortcuts.map(\.label)
            dbg("global translation shortcut handler failed: \(status)")
            return
        }
        for shortcut in shortcuts {
            var ref: EventHotKeyRef?
            let result = RegisterEventHotKey(UInt32(kVK_ANSI_T), shortcut.modifiers,
                                             EventHotKeyID(signature: signature, id: shortcut.id),
                                             target, UInt32(kEventHotKeyExclusive), &ref)
            if result == noErr, let ref { translationHotKeys.append(ref) }
            else { unavailableTranslationHotKeys.append(shortcut.label) }
            dbg("registered translation shortcut \(shortcut.id): \(result)")
        }
    }

    // retry = true 时多给两个按钮：重试、看日志。出错的页面不该只有一个
    // 转不完的圈 —— 用户能做的最合理动作就是"再来一次"，其次是"看看为什么"。
    private func showSplash(_ text: String, retry: Bool = false) {
        if overlayTranslating {
            showOverlay("本地引擎连接中断。引擎恢复后可以重试。", title: "翻译失败", source: overlaySource)
        }
        let buttons = retry ? """
        <div class="row">
          <button onclick="post('retry')">重试</button>
          <button class="ghost" onclick="post('openLog')">查看日志</button>
        </div>
        """ : ""
        let spinner = retry ? "" : #"<div class="dot"></div>"#
        let html = """
        <!doctype html><meta charset="utf-8">
        <style>
          :root { color-scheme: dark light; }
          body { margin:0; height:100vh; display:flex; align-items:center; justify-content:center;
                 font: 15px/1.6 -apple-system, "PingFang SC", sans-serif;
                 background:#0d0f12; color:#e8eaed; }
          .box { text-align:center; max-width:520px; padding:0 24px; }
          .dot { width:34px; height:34px; margin:0 auto 18px; border-radius:50%;
                 border:3px solid rgba(232,234,237,.18); border-top-color:#7aa2ff;
                 animation:spin .9s linear infinite; }
          @keyframes spin { to { transform:rotate(360deg); } }
          .sub { opacity:.55; font-size:13px; margin-top:6px; }
          .row { margin-top:20px; display:flex; gap:10px; justify-content:center; }
          button { font:inherit; font-size:14px; padding:7px 18px; border-radius:8px;
                   border:0; background:#7aa2ff; color:#0b1020; cursor:pointer; }
          button:hover { filter:brightness(1.08); }
          button.ghost { background:transparent; color:#e8eaed;
                         box-shadow:inset 0 0 0 1px rgba(232,234,237,.28); }
        </style>
        <div class="box">\(spinner)<div>\(text)</div>
        <div class="sub">首次启动要加载权重，可能要几秒</div>
        \(buttons)</div>
        <script>
          function post(m){ try { window.webkit.messageHandlers.shell.postMessage(m); } catch (e) {} }
        </script>
        """
        uiLoaded = false
        webView.loadHTMLString(html, baseURL: nil)
    }

    // ------------------------------------------------------------ 起引擎
    private func startEngine() {
        guard let bin = resolveEngine() else {
            showSplash("找不到 dubhed —— 用 DUBHE_BIN 指一下路径，或者先 make all", retry: true)
            return
        }

        // 清掉上一次被强杀留下的引擎（SIGKILL 接不到，只能下次启动补）。
        // 只清"可执行文件路径就是包里这份"的进程，不会误伤用户自己开的
        // dubhed（那通常是开发时从 build/dubhed 起的）。
        let stale = killStaleEngines(bundleEnginePath: bin)
        if stale > 0 {
            dbg("killed \(stale) stale engine(s)")
            usleep(200_000)                 // 等端口真的释放
        }

        var p: UInt16 = 7777
        while p < 7800 && !portIsFree(p) { p += 1 }
        port = p
        dbg("engine \(bin) port \(p)")

        var args: [String] = ["--port", "\(p)"]
        if let web = resolveWebRoot() {
            args += ["--web", web]
        }
        // 模型可以不指定：UI 会用 /api/models 列目录、/api/load 换模型。
        // 但每次打开都从"未加载"开始太烦 —— 记住上次在界面里选的那个，
        // 下次直接加载。DUBHE_MODEL 优先（调试/多模型切换时用）。
        var modelArg: String?
        if let m = ProcessInfo.processInfo.environment["DUBHE_MODEL"], !m.isEmpty {
            modelArg = m
        } else if let m = rememberedModel() {
            modelArg = m
            usedRememberedModel = true
            dbg("remembered model \(m)")
        }
        if let m = modelArg {
            args += ["-m", m]
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: bin)
        proc.arguments = args
        // 引擎的日志走 stderr。不能给管道：没人读的话写满 64 KB 引擎就阻塞；
        // 也不能丢 /dev/null：白屏的时候没有地方可查。落文件。
        //
        // 只在本次进程的第一次启动时截断，之后（重启引擎、坏模型退回）都
        // 追加 —— 否则"上一次为什么失败"正好会被下一次启动擦掉，而那恰恰
        // 是最需要看的一段。
        let logURL = engineLogURL()
        if !logInitialized {
            logInitialized = true
            if let old = try? FileManager.default.attributesOfItem(atPath: logURL.path),
               let size = old[.size] as? Int, size > 8 << 20 {
                let prev = logURL.deletingPathExtension().appendingPathExtension("log.1")
                try? FileManager.default.removeItem(at: prev)
                try? FileManager.default.moveItem(at: logURL, to: prev)
            }
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        } else {
            // 注意别用 createFile 兜底：它对已存在的文件是"覆盖"，
            // 会把上一次失败的日志直接抹掉。
            if !FileManager.default.fileExists(atPath: logURL.path) {
                FileManager.default.createFile(atPath: logURL.path, contents: nil)
            }
            if let f = FileHandle(forWritingAtPath: logURL.path) {
                f.seekToEndOfFile()
                f.write("\n──── 重启引擎 ────\n".data(using: .utf8) ?? Data())
                f.closeFile()
            }
        }
        let fh = FileHandle(forWritingAtPath: logURL.path)
        fh?.seekToEndOfFile()
        logHandle = fh
        proc.standardOutput = fh ?? FileHandle(forWritingAtPath: "/dev/null")
        proc.standardError = fh ?? FileHandle(forWritingAtPath: "/dev/null")
        proc.terminationHandler = { [weak self] pr in
            DispatchQueue.main.async {
                guard let self, !self.shuttingDown, self.engine === pr else { return }
                let code = pr.terminationStatus
                logEngineEvent("引擎退出：退出码 \(code)（\(pr.terminationReason == .uncaughtSignal ? "被信号终止" : "正常退出")）")
                self.stopPolling()
                self.stopStateWatch()
                self.engine = nil

                // ① 界面上跑着跑着引擎没了：OOM、崩溃、被外部 kill。
                //    自己拉起来，但要有上限 —— 一个必然 OOM 的模型会变成
                //    "重启 → 崩 → 重启"的风暴，那比停下来更糟。
                //    上限用**时间窗**判定：60 秒内连着崩才算"连续"，
                //    跑了五分钟才崩一次的是新问题，计数重新开始。
                if self.everLoadedUI {
                    let now = Date()
                    if now.timeIntervalSince(self.lastCrashAt) > 60 { self.crashCount = 0 }
                    self.lastCrashAt = now
                    self.crashCount += 1
                    if self.crashCount <= 3 {
                        dbg("engine died (exit \(code)) after UI load, auto-restart #\(self.crashCount)")
                        self.showSplash("引擎意外退出（退出码 \(code)），"
                                        + "第 \(self.crashCount) 次自动重启…")
                        let delay = Double(self.crashCount)      // 1s / 2s / 3s
                        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                            guard let self, !self.shuttingDown, self.engine == nil else { return }
                            self.startEngine()
                        }
                    } else {
                        dbg("engine keeps dying, giving up after \(self.crashCount)")
                        self.showSplash("引擎连续 \(self.crashCount) 次意外退出（退出码 \(code)）。"
                                        + "多半是模型太大、文件损坏，或者内存不够。",
                                        retry: true)
                    }
                    return
                }

                // ② 启动阶段就挂了，而且用的是"上次记住的模型"——多半是那个
                //    模型文件坏了/不兼容。忘掉它，退回到无模型状态再来一次，
                //    别让用户卡在一个永远起不来的循环里。只退一次。
                if self.usedRememberedModel {
                    self.usedRememberedModel = false
                    UserDefaults.standard.removeObject(forKey: Self.lastModelKey)
                    dbg("remembered model failed (exit \(code)), forgetting it")
                    self.showSplash("上次用的模型没能加载（退出码 \(code)），已重置。正在重新启动…")
                    self.startEngine()
                    return
                }

                self.showSplash("引擎启动失败（退出码 \(code)）。", retry: true)
            }
        }
        do {
            try proc.run()
            engine = proc
        } catch {
            showSplash("起不来 dubhed：\(error.localizedDescription)", retry: true)
            return
        }
        waitForReady()
    }

    private func stopEngine() {
        stopPolling()
        stopStateWatch()
        guard let e = engine, e.isRunning else { return }
        e.terminate()                       // SIGTERM，引擎会走正常收尾
        let deadline = Date().addingTimeInterval(3)
        while e.isRunning && Date() < deadline {
            usleep(50_000)
        }
        if e.isRunning { kill(e.processIdentifier, SIGKILL) }
        engine = nil
    }

    // 换模型、卡住、或者干脆想重来时用。走的是和冷启动完全一样的一条路，
    // 所以"重启能修好的问题，冷启动也一定能修好"这句话是成立的。
    @objc private func restartEngine() {
        dbg("restart engine")
        // 人主动点重启（或点了错误页的重试）：崩溃计数重新开始。
        // 自动重启那条路不经过这里，所以不会被它绕开上限。
        crashCount = 0
        stopEngine()
        uiLoaded = false
        showSplash("正在重启本地引擎…")
        startEngine()
    }

    // ------------------------------------------------------------ 转圈页上的按钮
    //
    // 两条消息来源：
    //   * 转圈页的字符串命令（retry / openLog）—— 页面是我们自己写死的一小段 HTML
    //   * 正页发来的字典事件（turnEnd）—— web/app.js 里的 shellPost
    // 只认白名单里的 type，不做通用 bridge：任何一条都必须是"外壳能独立验证
    // 语义"的动作，不能变成网页随意调 Swift 的入口。
    func userContentController(_ ucc: WKUserContentController,
                               didReceive message: WKScriptMessage) {
        guard message.name == "shell", message.frameInfo.isMainFrame else { return }
        if let cmd = message.body as? String {
            dbg("shell message \(cmd)")
            switch cmd {
            case "retry":   restartEngine()
            case "openLog": openEngineLog()
            default:        break
            }
            return
        }
        let origin = message.frameInfo.securityOrigin
        guard uiLoaded, origin.protocol == "http", origin.host == "127.0.0.1",
              origin.port == Int(port),
              let obj = message.body as? [String: Any],
              let type = obj["type"] as? String else { return }
        dbg("shell event \(type)")
        switch type {
        case "checkForUpdates":
            checkForUpdates()
        case "showChangelog":
            showChangelog()
        case "turnEnd":
            notifyTurnEnd((obj["body"] as? String) ?? "回答已生成")
        case "ocr":
            captureAndOCR()
        case "translateSelection":
            translateSelection()
        case "openAx":
            // 直接跳到「隐私与安全性 → 辅助功能」，省得用户自己翻三层菜单。
            if let u = URL(string:
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
                NSWorkspace.shared.open(u)
            }
        case "axStatus":
            let ok = AXIsProcessTrusted()
            webView.evaluateJavaScript(
                "window.dubheAxStatus && window.dubheAxStatus(\(ok ? "true" : "false"))") { _, _ in }
        case "openScreenCapture":
            openScreenCaptureSettings()
        case "translateClipboardImage":
            translateClipboardImage()
        case "translationHotKeyStatus":
            webView.callAsyncJavaScript("window.dubheTranslationHotKeyStatus?.(unavailable)",
                                       arguments: ["unavailable": unavailableTranslationHotKeys],
                                       in: nil, in: .page, completionHandler: nil)
        case "overlay":
            if capture == nil, !recognizingScreenshot, let text = obj["text"] as? String {
                showOverlay(text, title: (obj["title"] as? String) ?? "译文", source: obj["source"] as? String ?? "")
            }
        case "historyExport":
            if let text = obj["text"] as? String { saveHistory(text) }
        case "modelUnloaded":
            usedRememberedModel = false
            UserDefaults.standard.removeObject(forKey: Self.lastModelKey)
        case "translationDocument":
            handleTranslationDocument(obj)
        default:
            break
        }
    }

    // ------------------------------------------------------------ 截图取词
    //
    // 为什么必须在外壳里：网页拿不到屏幕。截屏和文字识别（Vision）都是
    // 原生框架，只有 App 进程能调。
    //
    // 框选后识别文字，交给页面翻译，结果再回到原生浮窗。
    @objc private func captureAndOCR() {
        dbg("screenshot translation requested")
        withOCRLanguage { [weak self] language in self?.beginScreenshot(language: language) }
    }

    private func withOCRLanguage(_ action: @escaping (String) -> Void) {
        guard uiLoaded else {
            showOverlay("等本地引擎启动后再试。", title: "截图翻译")
            return
        }
        webView.evaluateJavaScript("window.dubheOCRLanguage?.()") { value, _ in
            action(value as? String ?? "auto")
        }
    }

    private func beginScreenshot(language: String) {
        guard capture == nil, !recognizingScreenshot else { return }
        // 调试口：给一张现成的图，跳过交互选区直接识别。
        // 没有这个，验证 OCR 只能靠人手框选，没法自动化。
        if let fixed = ProcessInfo.processInfo.environment["DUBHE_OCR_FILE"], !fixed.isEmpty {
            overlay?.close()
            ocrFile(fixed, language: language)
            return
        }
        overlay?.close()
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("dubhe-ocr-\(UUID().uuidString).png")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        p.arguments = ["-i", "-s", "-x", tmp.path]
        let errors = Pipe()
        p.standardError = errors
        p.terminationHandler = { [weak self] process in
            let message = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            DispatchQueue.main.async {
                defer { try? FileManager.default.removeItem(at: tmp) }
                guard let self else { return }
                self.capture = nil
                if process.terminationStatus == 0, FileManager.default.fileExists(atPath: tmp.path) {
                    self.ocrFile(tmp.path, language: language)
                } else if !message.isEmpty || process.terminationStatus > 1 {
                    dbg("interactive screenshot failed (\(process.terminationStatus)): \(message)")
                    if !CGPreflightScreenCaptureAccess() {
                        self.showScreenCapturePermission()
                    } else {
                        self.showOverlay("系统未能生成这张截图，可以重新框选。", title: "截图失败")
                    }
                }
            }
        }
        do {
            try p.run()
            capture = p
        } catch {
            showOverlay("无法启动截图工具。", title: "截图失败")
        }
    }

    @objc private func openScreenCaptureSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
            NSWorkspace.shared.open(url)
        }
    }

    private func showScreenCapturePermission() {
        let location = Bundle.main.bundleURL.path
        showOverlay(
            """
            macOS 没有允许当前这份 Dubhe 截图。请在系统设置中选择下面这个应用，然后退出并重新打开。

            \(location)

            也可以按 ⌃⌘⇧4 用 macOS 框选并复制截图，再点「翻译剪贴板图片」。
            """, title: "截图权限")
    }

    @objc private func translateClipboardImage() {
        withOCRLanguage { [weak self] language in
            guard let self, self.capture == nil, !self.recognizingScreenshot else { return }
            guard let image = NSImage(pasteboard: .general),
                  let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                self.showOverlay("剪贴板里没有图片。按 ⌃⌘⇧4 框选截图后再试。", title: "翻译剪贴板图片")
                return
            }
            self.overlay?.close()
            self.recognizeScreenshot(cg, language: language)
        }
    }

    private func ocrFile(_ path: String, language: String) {
        guard let img = NSImage(contentsOfFile: path),
              let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            showOverlay("无法读取刚截取的图片。", title: "截图失败")
            return
        }
        recognizeScreenshot(cg, language: language)
    }

    private func recognizeScreenshot(_ cg: CGImage, language: String) {
        recognizingScreenshot = true
        showOverlay("正在从截图中识别文字…", title: "识别中")
        DispatchQueue.global(qos: .userInitiated).async {
            let byLanguage = [
                "zh": ["zh-Hans", "zh-Hant"], "en": ["en-US"], "ja": ["ja-JP"],
                "ko": ["ko-KR"], "fr": ["fr-FR"], "de": ["de-DE"],
            ]
            let automatic = byLanguage[language] == nil
            let languages = byLanguage[language] ?? ["en-US", "zh-Hans", "zh-Hant", "ja-JP", "ko-KR", "fr-FR", "de-DE"]
            // Vision's quick recognizer covers printed English, French and German.
            // Use its multilingual accurate recognizer for CJK and mixed screenshots.
            let level: VNRequestTextRecognitionLevel = ["en", "fr", "de"].contains(language) ? .fast : .accurate
            func recognize(_ level: VNRequestTextRecognitionLevel) throws -> String {
                let req = VNRecognizeTextRequest()
                req.recognitionLevel = level
                if let supported = try? req.supportedRecognitionLanguages() {
                    req.recognitionLanguages = languages.filter(supported.contains)
                }
                req.automaticallyDetectsLanguage = automatic
                req.usesLanguageCorrection = true
                try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
                return (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
            }
            do {
                let text: String
                do {
                    text = try recognize(level)
                } catch {
                    guard automatic else { throw error }
                    dbg("accurate OCR failed, trying the quick recognizer: \(error)")
                    text = try recognize(.fast)
                    if text.isEmpty { throw error }
                }
                dbg("ocr recognized \(text.count) characters")
                DispatchQueue.main.async {
                    self.recognizingScreenshot = false
                    guard self.overlay?.title == "识别中" else { return }
                    self.requestTranslation(text, callback: "dubheOcr")
                }
            } catch {
                dbg("ocr perform failed: \(error)")
                DispatchQueue.main.async {
                    self.recognizingScreenshot = false
                    guard self.overlay?.title == "识别中" else { return }
                    self.showOverlay("文字识别失败，请重新截取清晰的文字区域。", title: "识别失败")
                }
            }
        }
    }

    // ------------------------------------------------------------ 划词翻译
    //
    // 读"系统里当前选中的文字"，靠的是辅助功能（Accessibility）接口：
    // AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElement) 拿到
    // 当前焦点控件，再读它的 kAXSelectedTextAttribute。
    //
    // 它需要用户在系统设置里授予「辅助功能」权限 —— 不给就是读不到，
    // 而且 macOS 不会报错，只会返回 -25211。所以第一次调用前先
    // AXIsProcessTrusted()，未授权就弹系统引导，别让用户以为程序坏了。
    @objc private func translateSelection() {
        dbg("selection translation requested")
        guard capture == nil, !recognizingScreenshot else { return }
        if NSApp.isActive {
            dbg("local translation shortcut")
            guard uiLoaded else {
                showOverlay("等本地引擎启动后再试。", title: "翻译")
                return
            }
            webView.evaluateJavaScript("window.dubheTranslateLocalSelection?.()") { [weak self] value, _ in
                if value as? Bool != true {
                    self?.showOverlay("翻译还没有就绪，请稍后再试。", title: "翻译")
                }
            }
            return
        }
        let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
        guard AXIsProcessTrustedWithOptions(opts as CFDictionary) else {
            dbg("accessibility 未授权")
            showOverlay("请在「辅助功能」里允许当前这份 Dubhe：\n\n\(Bundle.main.bundleURL.path)",
                        title: "划词权限")
            return
        }
        let sys = AXUIElementCreateSystemWide()
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(sys, kAXFocusedUIElementAttribute as CFString,
                                            &focused) == .success,
              let element = focused, CFGetTypeID(element) == AXUIElementGetTypeID() else {
            requestTranslation("", callback: "dubheText")
            return
        }
        let text = selectedText(in: element as! AXUIElement) ?? ""
        dbg("ax 选中 \(text.count) 字")
        requestTranslation(text, callback: "dubheText")
    }

    private func selectedText(in element: AXUIElement) -> String? {
        var sel: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXSelectedTextAttribute as CFString, &sel) == .success,
           let text = sel as? String, !text.isEmpty {
            return text
        }
        var savedRange: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &savedRange) == .success,
              let savedRange, CFGetTypeID(savedRange) == AXValueGetTypeID() else { return nil }
        let value = savedRange as! AXValue
        var range = CFRange()
        guard AXValueGetValue(value, .cfRange, &range), range.length > 0,
              AXUIElementCopyParameterizedAttributeValue(element, kAXStringForRangeParameterizedAttribute as CFString,
                                                         value, &sel) == .success else { return nil }
        return sel as? String
    }

    private func showOverlay(_ text: String, title: String, source: String = "") {
        dbg("overlay \(title): original \(source.count) chars, output \(text.count) chars")
        let p: NSPanel
        if let existing = overlay {
            p = existing
        } else {
            let mouse = NSEvent.mouseLocation
            let screen = NSScreen.screens.first { $0.frame.contains(mouse) } ?? NSScreen.main
            let visible = screen?.visibleFrame
            let size = NSSize(width: min(460, (visible?.width ?? 476) - 16),
                              height: min(280, (visible?.height ?? 296) - 16))
            p = NSPanel(contentRect: NSRect(origin: .zero, size: size),
                        styleMask: [.titled, .closable, .resizable, .utilityWindow, .nonactivatingPanel],
                        backing: .buffered, defer: false)
            p.delegate = self
            p.isReleasedWhenClosed = false
            p.isFloatingPanel = true
            p.hidesOnDeactivate = false
            p.becomesKeyOnlyIfNeeded = true
            p.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
            p.minSize = NSSize(width: 220, height: 140)

            let scroll = NSTextView.scrollableTextView()
            let tv = scroll.documentView as? NSTextView
            tv?.isEditable = false
            tv?.isSelectable = true
            tv?.font = .systemFont(ofSize: 13)
            tv?.textContainerInset = NSSize(width: 12, height: 12)
            let tabs = NSSegmentedControl(labels: ["译文", "原文"], trackingMode: .selectOne,
                                          target: self, action: #selector(refreshOverlayText))
            tabs.selectedSegment = 0
            let copy = NSButton(title: "复制", target: self, action: #selector(copyOverlayText))
            let screenSettings = NSButton(title: "打开系统设置", target: self, action: #selector(openScreenCaptureSettings))
            let clipboard = NSButton(title: "翻译剪贴板图片", target: self, action: #selector(translateClipboardImage))
            let controls = NSStackView(views: [tabs, copy, screenSettings, clipboard])
            controls.spacing = 8
            let content = NSView()
            for subview in [controls, scroll] as [NSView] {
                subview.translatesAutoresizingMaskIntoConstraints = false
                content.addSubview(subview)
            }
            p.contentView = content
            NSLayoutConstraint.activate([
                controls.topAnchor.constraint(equalTo: content.topAnchor, constant: 8),
                controls.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 12),
                scroll.topAnchor.constraint(equalTo: controls.bottomAnchor, constant: 6),
                scroll.leadingAnchor.constraint(equalTo: content.leadingAnchor),
                scroll.trailingAnchor.constraint(equalTo: content.trailingAnchor),
                scroll.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            ])
            overlay = p
            overlayText = tv
            overlayTabs = tabs
            overlayCopy = copy
            overlayScreenSettings = screenSettings
            overlayClipboard = clipboard

            // 浮窗跟着鼠标所在的屏幕，边缘留出关闭按钮。
            let frameSize = p.frame.size
            let bounds = visible?.insetBy(dx: 8, dy: 8)
            let x = mouse.x + 16, y = mouse.y - frameSize.height - 16
            if let bounds {
                p.setFrameOrigin(NSPoint(x: max(bounds.minX, min(x, bounds.maxX - frameSize.width)),
                                         y: max(bounds.minY, min(y, bounds.maxY - frameSize.height))))
            } else {
                p.setFrameOrigin(NSPoint(x: x, y: y))
            }
        }
        p.title = title
        let screenPermission = title == "截图权限"
        overlayTabs?.isHidden = screenPermission
        overlayCopy?.isHidden = screenPermission
        overlayScreenSettings?.isHidden = !screenPermission
        overlayClipboard?.isHidden = !screenPermission
        overlayTranslating = title == "翻译中"
        overlayHasTranslation = title == "译文"
        if overlaySource != source { overlayTabs?.selectedSegment = 0 }
        overlaySource = source
        overlayResult = text
        refreshOverlayText()
        p.orderFrontRegardless()
    }

    @objc private func refreshOverlayText() {
        overlayTabs?.setEnabled(!overlaySource.isEmpty, forSegment: 1)
        let original = overlayTabs?.selectedSegment == 1
        overlayText?.string = original ? overlaySource : overlayResult
        overlayText?.scrollRangeToVisible(NSRange(location: 0, length: 0))
        overlayCopy?.isEnabled = original || overlayHasTranslation
        overlayCopy?.title = "复制"
    }

    @objc private func copyOverlayText() {
        guard let text = overlayText?.string, !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        if NSPasteboard.general.setString(text, forType: .string) { overlayCopy?.title = "已复制" }
    }

    func windowWillClose(_ notification: Notification) {
        guard let closed = notification.object as? NSWindow else { return }
        if closed === window { overlay?.close() }
        guard closed === overlay else { return }
        if overlayTranslating {
            webView.evaluateJavaScript("window.dubheCancelTranslation?.()") { _, _ in }
        }
        overlayTranslating = false
        overlay = nil
        overlayText = nil
        overlayTabs = nil
        overlayCopy = nil
        overlayScreenSettings = nil
        overlayClipboard = nil
        overlaySource = ""
        overlayResult = ""
    }

    private func requestTranslation(_ text: String, callback: String) {
        guard uiLoaded else {
            showOverlay("等本地引擎启动后再试。", title: "翻译")
            return
        }
        guard let data = try? JSONEncoder().encode([callback, text]),
              let args = String(data: data, encoding: .utf8) else { return }
        webView.evaluateJavaScript("""
            (([callback, text]) => {
                const translate = window[callback];
                if (typeof translate !== 'function') return false;
                translate(text);
                return true;
            })(\(args))
            """) { [weak self] value, _ in
            if value as? Bool == true { return }
            self?.showOverlay("翻译还没有就绪，请稍后再试。", title: "翻译")
        }
    }

    private func saveHistory(_ text: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "Dubhe-history.json"
        panel.beginSheetModal(for: window) { result in
            guard result == .OK, let url = panel.url else { return }
            do {
                try Data(text.utf8).write(to: url, options: .atomic)
            } catch {
                let alert = NSAlert(error: error)
                alert.beginSheetModal(for: self.window)
            }
        }
    }

    private func handleTranslationDocument(_ message: [String: Any]) {
        guard let id = message["id"] as? Int else { return }
        let reply: ([String: Any]) -> Void = { [weak self] result in
            guard let self else { return }
            self.webView.callAsyncJavaScript(
                "window.dubheDocumentComplete(id, result)",
                arguments: ["id": id, "result": result], in: nil, in: .page, completionHandler: nil
            )
        }
        let action = message["action"] as? String
        if action == "export" {
            guard let format = message["format"] as? String, ["md", "txt", "pdf", "docx"].contains(format),
                  let text = message["text"] as? String, !text.isEmpty, text.utf8.count <= 24 * 1024 * 1024 else {
                reply(["error": "译文为空、过大或导出格式无效。"])
                return
            }
            let name = (message["name"] as? String) ?? "译文." + format
            let save: (Data) -> Void = { [weak self] data in
                self?.saveTranslation(data, format: format, name: name, reply: reply)
            }
            if format == "md" || format == "txt" {
                save(Data(text.utf8))
            } else {
                let blocks = message["blocks"] as? [[String: String]] ?? [["kind": "p", "text": text]]
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let data = try TranslationExport.render(format: format, blocks: blocks)
                        DispatchQueue.main.async { save(data) }
                    } catch {
                        DispatchQueue.main.async { reply(["error": error.localizedDescription]) }
                    }
                }
            }
            return
        }
        let saving = action == "save"
        let encodedLimit = (saving ? 64 : 24) * 1024 * 1024
        guard let format = message["format"] as? String, ["docx", "pdf"].contains(format),
              let encoded = message["document"] as? String, encoded.utf8.count <= encodedLimit else {
            reply(["error": saving ? "导出的文件过大或格式无效。" : "请选择不超过 16 MB 的 PDF 或 Word 文件。"])
            return
        }
        if saving {
            guard let data = Data(base64Encoded: encoded) else {
                reply(["error": "译文文件格式无效。"])
                return
            }
            saveTranslation(data, format: format, name: (message["name"] as? String) ?? "译文." + format, reply: reply)
            return
        }
        let translations = message["translations"] as? [String]
        let bundledWordTool = Bundle.main.url(forResource: "translation_docx", withExtension: "py",
                                                subdirectory: "backends")
        DispatchQueue.global(qos: .userInitiated).async {
            let result: [String: Any]
            do {
                if format == "pdf" {
                    guard let data = Data(base64Encoded: encoded) else {
                        throw CocoaError(.fileReadCorruptFile)
                    }
                    if let translations {
                        let rendered = try TranslationPDF.rebuild(data, translations: translations)
                        result = ["document": rendered.base64EncodedString()]
                    } else {
                        result = ["paragraphs": try TranslationPDF.paragraphs(data)]
                    }
                } else {
                    result = try Self.processWordDocument(encoded, translations: translations, tool: bundledWordTool)
                }
            } catch {
                DispatchQueue.main.async { reply(["error": error.localizedDescription]) }
                return
            }
            DispatchQueue.main.async { reply(result) }
        }
    }

    private func saveTranslation(_ data: Data, format: String, name: String, reply: @escaping ([String: Any]) -> Void) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: format, conformingTo: .data) ?? .data]
        panel.nameFieldStringValue = name
        panel.beginSheetModal(for: window) { result in
            guard result == .OK, let url = panel.url else { reply(["saved": false]); return }
            do {
                try data.write(to: url, options: .atomic)
                reply(["saved": true])
            } catch {
                reply(["error": "保存译文失败：" + error.localizedDescription])
            }
        }
    }

    private static func processWordDocument(_ document: String, translations: [String]?,
                                            tool: URL?) throws -> [String: Any] {
        guard let tool else { return ["error": "安装包缺少 Word 文档组件。"] }
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("dubhe-docx-" + UUID().uuidString)
        try fm.createDirectory(at: tmp, withIntermediateDirectories: false,
                               attributes: [.posixPermissions: 0o700])
        defer { try? fm.removeItem(at: tmp) }
        var data: [String: Any] = ["document": document]
        if let translations { data["translations"] = translations }
        let source = tmp.appendingPathComponent("request.json")
        try JSONSerialization.data(withJSONObject: data).write(to: source, options: .atomic)
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let output = Pipe()
        let task = Process()
        let preferred = ProcessInfo.processInfo.environment["DUBHE_BACKEND_PYTHON"] ?? ""
        let python = preferred.isEmpty ? "python3" : preferred
        let absolute = python.hasPrefix("/")
        task.executableURL = URL(fileURLWithPath: absolute ? python : "/usr/bin/env")
        task.arguments = (absolute ? [] : [python]) + ["-I", tool.path]
        task.standardInput = input
        task.standardOutput = output
        task.standardError = FileHandle.nullDevice
        try task.run()
        let response = output.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        guard task.terminationStatus == 0,
              let result = try JSONSerialization.jsonObject(with: response) as? [String: Any] else {
            return ["error": "无法处理 Word 文件。请检查本机 Python 3 和文件格式。"]
        }
        return result
    }

    // ------------------------------------------------------------ 等就绪
    //
    // 不 sleep 固定秒数：加载 1.7B 权重在冷启动时要 1 秒，在机械盘或首次
    // page-in 时要十几秒，写死任何数字都会在某一端出错。
    //
    // 探测用裸 TCP connect，不走 URLSession —— 少两层变量：没有 ATS、
    // 没有系统代理、没有 DNS。"端口能连上" 才是我们要问的问题本身。
    // 计时器用 DispatchSourceTimer 而不是 Timer：后者的触发依赖 run loop
    // 模式，启动阶段 schedule 进去有被挂起的风险。
    //
    // 引擎是「先加载模型、再监听端口」，所以大模型下这段时间会很长。
    // 30 秒后不判死刑，只是把转圈页的文案换掉，一直等到 5 分钟。
    private func waitForReady() {
        pollsLeft = 6000                        // 50 ms × 6000 = 5 分钟
        var ticks = 0
        let t = DispatchSource.makeTimerSource(queue: .main)
        t.schedule(deadline: .now() + 0.05, repeating: 0.05, leeway: .milliseconds(10))
        t.setEventHandler { [weak self] in
            guard let self else { return }
            self.pollsLeft -= 1
            ticks += 1
            if self.pollsLeft <= 0 {
                self.stopPolling()
                self.showSplash("引擎 5 分钟没有就绪，请检查 dubhed 能否单独跑起来", retry: true)
                return
            }
            if ticks == 600 {                   // 30 秒
                self.showSplash("还在加载模型…")
            }
            if self.portAccepts(self.port) {
                self.stopPolling()
                self.loadUI()
            }
        }
        pollTimer = t
        t.resume()
    }

    private func stopPolling() {
        pollTimer?.cancel()
        pollTimer = nil
    }

    // 非阻塞 connect 只是"开始连"，EINPROGRESS 不代表成功 —— 端口没开的时候
    // macOS 一样先返回 EINPROGRESS，真正的结果要等 poll 可写之后看 SO_ERROR。
    // 这里给 50 ms：loopback 上要么立刻连上，要么立刻被拒，等不到就是没有。
    private func portAccepts(_ port: UInt16) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        if fd < 0 { return false }
        defer { close(fd) }
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK)
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let rc = withUnsafePointer(to: &addr) { p -> Int32 in
            p.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if rc == 0 { return true }
        if errno != EINPROGRESS { return false }

        var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
        if poll(&pfd, 1, 50) <= 0 { return false }
        var err: Int32 = 0
        var len = socklen_t(MemoryLayout<Int32>.size)
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len)
        return err == 0
    }

    private func loadUI() {
        dbg("port accepting, loading UI on \(port)")
        guard let url = URL(string: "http://127.0.0.1:\(port)/") else { return }
        uiLoaded = true
        everLoadedUI = true
        webView.load(URLRequest(url: url))
        window.makeFirstResponder(webView)
        startStateWatch()
        scheduleAutoCloseIfAsked()
    }

    // ------------------------------------------------------------ 记住模型
    //
    // 界面里换模型走的是 /api/load，外壳不知道。与其把 web/app.js 和外壳
    // 耦合起来，不如每 10 秒问一次 /api/state —— 本地请求，200 字节，
    // 而且顺带能发现"引擎还活着但模型掉了"这类情况。
    private func startStateWatch() {
        stateTimer?.cancel()
        let t = DispatchSource.makeTimerSource(queue: .main)
        t.schedule(deadline: .now() + 2, repeating: 10)
        t.setEventHandler { [weak self] in self?.captureModel() }
        stateTimer = t
        t.resume()
    }

    private func stopStateWatch() {
        stateTimer?.cancel()
        stateTimer = nil
    }

    private func captureModel() {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/state") else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 3
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            guard let self, let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  obj["loaded"] as? Bool == true,
                  let m = obj["model"] as? String, !m.isEmpty else { return }
            // 走到这儿说明"上次记住的模型"这次也加载成功了，不用再退。
            self.usedRememberedModel = false
            if UserDefaults.standard.string(forKey: Self.lastModelKey) != m {
                UserDefaults.standard.set(m, forKey: Self.lastModelKey)
                dbg("remember model \(m)")
            }
        }.resume()
    }

    // 验收用：DUBHE_SHELL_AUTOCLOSE=3 表示界面加载完 3 秒后自己关窗口。
    // 走的是真实路径（关窗口 → applicationShouldTerminateAfterLastWindowClosed
    // → applicationWillTerminate → stopEngine），不是直接 exit()。
    private func scheduleAutoCloseIfAsked() {
        let env = ProcessInfo.processInfo.environment
        if let s = env["DUBHE_SHELL_AUTOCLOSE"], let secs = Double(s), secs > 0 {
            dbg("autoclose in \(secs)s")
            DispatchQueue.main.asyncAfter(deadline: .now() + secs) { [weak self] in
                self?.window.performClose(nil)
            }
        }
        // 同理：DUBHE_SHELL_AUTORESTART=2 验证"重启引擎"这条路（⌘⇧R 同一条）。
        if let s = env["DUBHE_SHELL_AUTORESTART"], let secs = Double(s), secs > 0 {
            dbg("autorestart in \(secs)s")
            DispatchQueue.main.asyncAfter(deadline: .now() + secs) { [weak self] in
                self?.restartEngine()
            }
        }
        // DUBHE_SHELL_NOTIFY_TEST=3：3 秒后按"生成完成"走一遍通知路径。
        // 通知本身要靠人看，但这条至少能验证断链（授权、delegate、角标）不崩。
        if let s = env["DUBHE_SHELL_NOTIFY_TEST"], let secs = Double(s), secs > 0 {
            dbg("notify test in \(secs)s")
            DispatchQueue.main.asyncAfter(deadline: .now() + secs) { [weak self] in
                self?.notifyTurnEnd("测试通知：这一轮回答已经生成完了。")
            }
        }
        // DUBHE_SHELL_EVAL="js…"：界面加载完后在页面里跑一段 JS，只给调试用。
        // 验证页面侧到外壳的那条通道（window.webkit.messageHandlers）时，
        // 没有这条就只能靠手点。
        if debugOn, let js = env["DUBHE_SHELL_EVAL"], !js.isEmpty {
            dbg("eval \(js)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                self?.webView.evaluateJavaScript(js) { val, err in
                    if let err { dbg("eval error \(err.localizedDescription)") }
                    else { dbg("eval -> \(val.map { String(describing: $0) } ?? "nil")") }
                }
            }
        }
    }

    // ------------------------------------------------------------ 文件选择
    //
    // 页面里点「文件」/「图像」会走到这里。WKOpenPanelParameters 只暴露
    // allowsMultipleSelection / allowsDirectories，读不到 accept 属性 ——
    // 所以这里不做类型过滤，选错了类型由网页侧再校验（readAttachment
    // 会按 mime 判断，给不支持的格式会明确报出来）。
    func webView(_ webView: WKWebView,
                 runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.treatsFilePackagesAsDirectories = true
        panel.message = "选择要发送给模型的文件"
        panel.prompt = "选择"
        // 用 sheet 而不是 begin：不挡住窗口，取消也不会留下悬空的面板
        if let w = window {
            panel.beginSheetModal(for: w) { resp in
                completionHandler(resp == .OK ? panel.urls : nil)
            }
        } else {
            panel.begin { resp in
                completionHandler(resp == .OK ? panel.urls : nil)
            }
        }
    }

    // ------------------------------------------------------------ 导航
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        dbg("didFinish \(webView.url?.absoluteString ?? "?")")
        if let t = webView.title, !t.isEmpty { window.title = t }
        window.makeFirstResponder(webView)
        maybeSnapshot()
    }

    // 给自动化验收用：DUBHE_SHELL_SNAPSHOT=/tmp/x.png 时，界面加载完就存一张。
    // 走 WKWebView 自己的快照，不需要"屏幕录制"权限 —— 在无头/无人值守的
    // 环境里 screencapture 是拿不到画面的，但这一条仍然有效。
    private func maybeSnapshot() {
        guard uiLoaded,
              let out = ProcessInfo.processInfo.environment["DUBHE_SHELL_SNAPSHOT"],
              !out.isEmpty else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            guard let self else { return }
            let cfg = WKSnapshotConfiguration()
            cfg.rect = self.webView.bounds
            self.webView.takeSnapshot(with: cfg) { img, err in
                guard let img,
                      let tiff = img.tiffRepresentation,
                      let rep = NSBitmapImageRep(data: tiff),
                      let png = rep.representation(using: .png, properties: [:]) else {
                    dbg("snapshot failed: \(err?.localizedDescription ?? "no image")")
                    return
                }
                try? png.write(to: URL(fileURLWithPath: out))
                dbg("snapshot \(out) \(Int(img.size.width))x\(Int(img.size.height))")
            }
        }
    }

    // 白屏最容易发生在这一层（ATS 拒绝、端口对了但引擎没起完）。把真正的
    // 原因写进转圈页，比让用户对着一个永远转的圈猜要好。
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
                 withError error: Error) {
        dbg("didFailProvisional \(error.localizedDescription)")
        if !uiLoaded { return }        // 转圈页自己失败不算
        showSplash("打不开引擎界面：\(error.localizedDescription)", retry: true)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        dbg("didFail \(error.localizedDescription)")
    }

    // 只允许留在本地引擎里：外部链接交给系统浏览器，别把 WebView 变成浏览器
    func webView(_ webView: WKWebView,
                 decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { return decisionHandler(.allow) }
        let host = url.host ?? ""
        if host == "127.0.0.1" || host == "localhost" || url.scheme == "about" || host.isEmpty {
            return decisionHandler(.allow)
        }
        NSWorkspace.shared.open(url)
        decisionHandler(.cancel)
    }

    // target="_blank" 的链接走这里：联网来源的标题、正文里的引用编号 [1]。
    //
    // 不实现这个代理方法，WebKit 拿到"新窗口"请求就直接丢掉 —— 点上去一点
    // 反应都没有，控制台也不报错。而这两类链接的价值恰恰在于"能点开核一下"。
    // 处理方式和上面一致：外部链接交给系统浏览器，WebView 自己不留。
    func webView(_ webView: WKWebView,
                 createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url { NSWorkspace.shared.open(url) }
        return nil
    }
}

// ---------------------------------------------------------------- 入口
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
