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
import WebKit

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
// 优先用 App 包里的（build.sh 会把二进制和 web/ 一起塞进 Contents/Resources），
// 其次看 DUBHE_BIN，最后退回开发时的 build/dubhed。
private func resolveEngine() -> String? {
    let fm = FileManager.default
    var candidates: [String] = []
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
    let dir = base.appendingPathComponent("Logs/Dubhe", isDirectory: true)
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
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate,
                         WKNavigationDelegate, WKScriptMessageHandler {
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
        // 第二次双击时不要再起一份引擎：macOS 一般会把已经开着的那个激活，
        // 但从终端直接跑二进制、或者两份不同路径的拷贝，都会真的起两个
        // 引擎（各占一个端口、各吃一份内存）。这里主动退让。
        let others = NSRunningApplication
            .runningApplications(withBundleIdentifier: Bundle.main.bundleIdentifier ?? "com.dubhe.desktop")
            .filter { $0.processIdentifier != getpid() }
        if let other = others.first {
            other.activate()
            dbg("another instance running (pid \(other.processIdentifier)), bowing out")
            exit(0)
        }

        installSignalHandlers()
        buildMenu()
        buildWindow()
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

    func applicationWillTerminate(_ note: Notification) {
        shuttingDown = true
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
        win.addItem(withTitle: "关闭窗口", action: #selector(NSWindow.performClose(_:)),
                    keyEquivalent: "w")
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

    // retry = true 时多给两个按钮：重试、看日志。出错的页面不该只有一个
    // 转不完的圈 —— 用户能做的最合理动作就是"再来一次"，其次是"看看为什么"。
    private func showSplash(_ text: String, retry: Bool = false) {
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
                self.stopPolling()
                self.engine = nil

                // 启动阶段就挂了，而且用的是"上次记住的模型"——多半是那个
                // 模型文件坏了/不兼容。忘掉它，退回到无模型状态再来一次，
                // 别让用户卡在一个永远起不来的循环里。只退一次。
                if !self.uiLoaded && self.usedRememberedModel {
                    self.usedRememberedModel = false
                    UserDefaults.standard.removeObject(forKey: Self.lastModelKey)
                    dbg("remembered model failed (exit \(code)), forgetting it")
                    self.showSplash("上次用的模型没能加载（退出码 \(code)），已重置。正在重新启动…")
                    self.startEngine()
                    return
                }

                self.showSplash(self.uiLoaded
                    ? "引擎已退出（退出码 \(code)）。"
                    : "引擎启动失败（退出码 \(code)）。", retry: true)
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
        stopEngine()
        uiLoaded = false
        showSplash("正在重启本地引擎…")
        startEngine()
    }

    // ------------------------------------------------------------ 转圈页上的按钮
    //
    // 只认两个字符串，不做通用 bridge：转圈页是我们自己写死的一小段 HTML，
    // 页面本身不加载任何外部内容。
    func userContentController(_ ucc: WKUserContentController,
                               didReceive message: WKScriptMessage) {
        guard message.name == "shell", let cmd = message.body as? String else { return }
        dbg("shell message \(cmd)")
        switch cmd {
        case "retry":   restartEngine()
        case "openLog": openEngineLog()
        default:        break
        }
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
}

// ---------------------------------------------------------------- 入口
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
