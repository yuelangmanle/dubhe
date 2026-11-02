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
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var engine: Process?
    private var port: UInt16 = 0
    private var pollTimer: DispatchSourceTimer?
    private var pollsLeft = 0
    private var uiLoaded = false
    private var shuttingDown = false
    private var signalSources: [DispatchSourceSignal] = []

    func applicationDidFinishLaunching(_ note: Notification) {
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
        appMenu.addItem(withTitle: "关于 Dubhe",
                        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
                        keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "隐藏 Dubhe",
                        action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "隐藏其他",
                        action: #selector(NSApplication.hideOtherApplications(_:)),
                        keyEquivalent: "h").keyEquivalentModifierMask = [.command, .option]
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
        webView = WKWebView(frame: rect, configuration: cfg)
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        showSplash("正在启动本地引擎…")
    }

    private func showSplash(_ text: String) {
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
        </style>
        <div class="box"><div class="dot"></div><div>\(text)</div>
        <div class="sub">首次启动要加载权重，可能要几秒</div></div>
        """
        uiLoaded = false
        webView.loadHTMLString(html, baseURL: nil)
    }

    // ------------------------------------------------------------ 起引擎
    private func startEngine() {
        guard let bin = resolveEngine() else {
            showSplash("找不到 dubhed —— 用 DUBHE_BIN 指一下路径，或者先 make all")
            return
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
        // 指定了就用它，省一次点击。
        if let m = ProcessInfo.processInfo.environment["DUBHE_MODEL"], !m.isEmpty {
            args += ["-m", m]
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: bin)
        proc.arguments = args
        // 引擎的日志走 stderr。不接管道的话会跟着 App 一起进 Console，
        // 接上就没人读、写满 64 KB 后引擎会阻塞 —— 所以直接丢给 /dev/null，
        // 真要调试就 DUBHE_BIN 指向一个包装脚本。
        proc.standardOutput = FileHandle(forWritingAtPath: "/dev/null")
        proc.standardError = FileHandle(forWritingAtPath: "/dev/null")
        proc.terminationHandler = { [weak self] pr in
            DispatchQueue.main.async {
                guard let self, !self.shuttingDown, self.engine === pr else { return }
                let code = pr.terminationStatus
                self.stopPolling()
                self.engine = nil
                self.showSplash(self.uiLoaded
                    ? "引擎已退出（退出码 \(code)）。关掉窗口重开试试。"
                    : "引擎启动失败（退出码 \(code)）。")
            }
        }
        do {
            try proc.run()
            engine = proc
        } catch {
            showSplash("起不来 dubhed：\(error.localizedDescription)")
            return
        }
        waitForReady()
    }

    private func stopEngine() {
        stopPolling()
        guard let e = engine, e.isRunning else { return }
        e.terminate()                       // SIGTERM，引擎会走正常收尾
        let deadline = Date().addingTimeInterval(3)
        while e.isRunning && Date() < deadline {
            usleep(50_000)
        }
        if e.isRunning { kill(e.processIdentifier, SIGKILL) }
        engine = nil
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
                self.showSplash("引擎 5 分钟没有就绪，请检查 dubhed 能否单独跑起来")
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
        scheduleAutoCloseIfAsked()
    }

    // 验收用：DUBHE_SHELL_AUTOCLOSE=3 表示界面加载完 3 秒后自己关窗口。
    // 走的是真实路径（关窗口 → applicationShouldTerminateAfterLastWindowClosed
    // → applicationWillTerminate → stopEngine），不是直接 exit()。
    private func scheduleAutoCloseIfAsked() {
        guard let s = ProcessInfo.processInfo.environment["DUBHE_SHELL_AUTOCLOSE"],
              let secs = Double(s), secs > 0 else { return }
        dbg("autoclose in \(secs)s")
        DispatchQueue.main.asyncAfter(deadline: .now() + secs) { [weak self] in
            self?.window.performClose(nil)
        }
    }

    // ------------------------------------------------------------ 导航
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        dbg("didFinish \(webView.url?.absoluteString ?? "?")")
        window.title = "Dubhe"
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
        showSplash("打不开引擎界面：\(error.localizedDescription)")
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
