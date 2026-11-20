import AppKit
import ScreenCaptureKit

/// Uses the system's per-capture consent when Dubhe has no persistent screen permission.
@MainActor
final class OneShotScreenCapture: NSObject, SCContentSharingPickerObserver {
    private let picker = SCContentSharingPicker.shared
    private var completion: ((Result<CGImage?, Error>) -> Void)?
    private var panel: NSPanel?
    private var picking = false

    var isActive: Bool { completion != nil }

    func begin(completion: @escaping (Result<CGImage?, Error>) -> Void) {
        guard !isActive else { return }
        self.completion = completion
        picking = true
        var configuration = SCContentSharingPickerConfiguration()
        configuration.allowedPickerModes = [.singleDisplay]
        configuration.allowsChangingSelectedContent = false
        picker.defaultConfiguration = configuration
        picker.add(self)
        picker.isActive = true
        picker.present(using: .display)
    }

    func cancel() { finish(.success(nil)) }

    nonisolated func contentSharingPicker(_ picker: SCContentSharingPicker, didCancelFor stream: SCStream?) {
        Task { @MainActor in
            if picking { finish(.success(nil)) }
        }
    }

    nonisolated func contentSharingPickerStartDidFailWithError(_ error: Error) {
        Task { @MainActor in
            if picking { finish(.failure(error)) }
        }
    }

    nonisolated func contentSharingPicker(_ picker: SCContentSharingPicker, didUpdateWith filter: SCContentFilter,
                                          for stream: SCStream?) {
        Task { @MainActor in
            guard picking else { return }
            picking = false
            picker.remove(self)
            do {
                let configuration = SCStreamConfiguration()
                let scale = max(1, CGFloat(filter.pointPixelScale))
                configuration.width = max(1, Int((filter.contentRect.width * scale).rounded()))
                configuration.height = max(1, Int((filter.contentRect.height * scale).rounded()))
                configuration.showsCursor = false
                let screenshot = try await SCScreenshotManager.captureImage(contentFilter: filter,
                                                                            configuration: configuration)
                picker.isActive = false
                guard isActive else { return }
                let screen = Self.screen(for: filter)
                try selectRegion(on: screen, image: screenshot)
            } catch {
                finish(.failure(error))
            }
        }
    }

    private static func screen(for filter: SCContentFilter) -> NSScreen? {
        if #available(macOS 15.2, *), let display = filter.includedDisplays.first,
           let screen = NSScreen.screens.first(where: { displayID(for: $0) == display.displayID }) {
            return screen
        }
        return NSScreen.screens.first {
            guard let id = displayID(for: $0) else { return false }
            return CGDisplayBounds(id) == filter.contentRect
        } ?? NSScreen.screens.first { $0.frame.contains(NSEvent.mouseLocation) } ?? NSScreen.main
    }

    private static func displayID(for screen: NSScreen) -> CGDirectDisplayID? {
        (screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
    }

    private func selectRegion(on screen: NSScreen?, image: CGImage) throws {
        guard let screen else { throw CaptureError.noScreen }
        let window = RegionPanel(contentRect: screen.frame, styleMask: [.borderless, .nonactivatingPanel],
                                 backing: .buffered, defer: false)
        window.setFrame(screen.frame, display: false)
        window.level = .screenSaver
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        window.hidesOnDeactivate = false
        window.hasShadow = false
        let region = CaptureRegionView(frame: NSRect(origin: .zero, size: screen.frame.size), image: image) {
            [weak self] rect in
            guard let self else { return }
            if let rect, let crop = image.cropping(to: rect) { finish(.success(crop)) }
            else if rect == nil { finish(.success(nil)) }
            else { finish(.failure(CaptureError.invalidSelection)) }
        }
        window.contentView = region
        panel = window
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(region)
    }

    private func finish(_ result: Result<CGImage?, Error>) {
        guard let completion else { return }
        self.completion = nil
        if picking {
            picking = false
            picker.remove(self)
        }
        picker.isActive = false
        panel?.orderOut(nil)
        panel = nil
        completion(result)
    }

    private enum CaptureError: LocalizedError {
        case noScreen, invalidSelection

        var errorDescription: String? {
            switch self {
            case .noScreen: return "找不到要框选的屏幕。"
            case .invalidSelection: return "无法读取所选区域。"
            }
        }
    }
}

private final class RegionPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

private final class CaptureRegionView: NSView {
    private let image: NSImage
    private let pixelSize: NSSize
    private let onFinish: (CGRect?) -> Void
    private var origin: NSPoint?
    private var selection = NSRect.zero

    init(frame: NSRect, image: CGImage, onFinish: @escaping (CGRect?) -> Void) {
        self.image = NSImage(cgImage: image, size: frame.size)
        self.pixelSize = NSSize(width: image.width, height: image.height)
        self.onFinish = onFinish
        super.init(frame: frame)
    }

    required init?(coder: NSCoder) { nil }
    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override var mouseDownCanMoveWindow: Bool { false }

    override func resetCursorRects() { addCursorRect(bounds, cursor: .crosshair) }

    override func draw(_ dirtyRect: NSRect) {
        image.draw(in: bounds, from: .zero, operation: .copy, fraction: 1, respectFlipped: true, hints: nil)
        let shade = NSBezierPath(rect: bounds)
        shade.windingRule = .evenOdd
        shade.appendRect(selection)
        NSColor.black.withAlphaComponent(0.42).setFill()
        shade.fill()
        if origin != nil {
            NSColor.white.setStroke()
            NSBezierPath(rect: selection).stroke()
        } else {
            let hint = NSAttributedString(string: "拖动框选要翻译的文字 · Esc 取消", attributes: [
                .foregroundColor: NSColor.white, .font: NSFont.systemFont(ofSize: 16, weight: .semibold),
            ])
            let size = hint.size()
            let position = NSPoint(x: (bounds.width - size.width) / 2, y: 40)
            let backdrop = NSRect(origin: position, size: size).insetBy(dx: -14, dy: -8)
            NSColor.black.withAlphaComponent(0.72).setFill()
            NSBezierPath(roundedRect: backdrop, xRadius: 10, yRadius: 10).fill()
            hint.draw(at: position)
        }
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        origin = convert(event.locationInWindow, from: nil)
        selection = .zero
        needsDisplay = true
    }

    override func mouseDragged(with event: NSEvent) {
        guard let origin else { return }
        let end = convert(event.locationInWindow, from: nil)
        selection = NSRect(x: min(origin.x, end.x), y: min(origin.y, end.y),
                           width: abs(end.x - origin.x), height: abs(end.y - origin.y)).intersection(bounds)
        needsDisplay = true
    }

    override func mouseUp(with event: NSEvent) {
        mouseDragged(with: event)
        origin = nil
        guard selection.width >= 3, selection.height >= 3 else {
            selection = .zero
            needsDisplay = true
            return
        }
        let scale = CGAffineTransform(scaleX: pixelSize.width / bounds.width, y: pixelSize.height / bounds.height)
        onFinish(selection.applying(scale).integral)
    }

    override func cancelOperation(_ sender: Any?) { onFinish(nil) }
    override func keyDown(with event: NSEvent) {
        if event.charactersIgnoringModifiers == "\u{1b}" { cancelOperation(nil) }
        else { super.keyDown(with: event) }
    }
    override func rightMouseDown(with event: NSEvent) { onFinish(nil) }
}
