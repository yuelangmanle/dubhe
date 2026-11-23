import Darwin
import Foundation

enum MarkItDown {
    static let extensions = ["docx", "xlsx", "xls", "pptx", "pdf", "epub", "html", "htm"]
    static let inputLimit = 20 * 1024 * 1024
    static let outputLimit = 256 * 1024

    static func executable() -> URL? {
        let configured = ProcessInfo.processInfo.environment["DUBHE_MARKITDOWN"]
        let candidates = [configured, NSHomeDirectory() + "/.local/bin/markitdown",
                          "/opt/homebrew/bin/markitdown", "/usr/local/bin/markitdown"]
        return candidates.compactMap { $0 }.first {
            $0.hasPrefix("/") && FileManager.default.isExecutableFile(atPath: $0)
        }.map { URL(fileURLWithPath: $0) }
    }

    static func convert(_ document: URL, with executable: URL) throws -> String {
        let ext = document.pathExtension.lowercased()
        guard document.isFileURL, extensions.contains(ext) else { throw Failure.unsupported }
        let source = try FileHandle(forReadingFrom: document)
        defer { try? source.close() }
        var metadata = stat()
        guard fstat(source.fileDescriptor, &metadata) == 0,
              metadata.st_mode & S_IFMT == S_IFREG else { throw Failure.unsupported }
        guard metadata.st_size <= inputLimit else { throw Failure.inputTooLarge }

        let process = Process()
        let output = Pipe()
        let exited = DispatchSemaphore(value: 0)
        process.executableURL = executable
        // MarkItDown receives only the selected file's bytes, with a format hint.
        process.arguments = ["--extension", ext]
        process.environment = ["PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"]
        process.standardInput = source
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { _ in exited.signal() }
        try process.run()
        defer {
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
        }

        let deadline = DispatchTime.now() + .seconds(30)
        let reader = output.fileHandleForReading
        let fd = reader.fileDescriptor
        defer { try? reader.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 8192)
        var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        while true {
            let now = DispatchTime.now().uptimeNanoseconds
            guard now < deadline.uptimeNanoseconds else { throw Failure.timedOut }
            let remaining = (deadline.uptimeNanoseconds - now + 999_999) / 1_000_000
            let ready = poll(&descriptor, 1, Int32(remaining))
            if ready == 0 { throw Failure.timedOut }
            if ready < 0 {
                if errno == EINTR { continue }
                throw Failure.conversion
            }
            let count = buffer.withUnsafeMutableBytes { read(fd, $0.baseAddress, $0.count) }
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw Failure.conversion
            }
            guard data.count + count <= outputLimit else { throw Failure.outputTooLarge }
            data.append(contentsOf: buffer.prefix(count))
        }
        guard exited.wait(timeout: deadline) == .success else { throw Failure.timedOut }
        guard process.terminationStatus == 0, let text = String(data: data, encoding: .utf8) else {
            throw Failure.conversion
        }
        let markdown = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !markdown.isEmpty else { throw Failure.empty }
        return markdown
    }

    private enum Failure: LocalizedError {
        case unsupported, inputTooLarge, timedOut, conversion, outputTooLarge, empty

        var errorDescription: String? {
            switch self {
            case .unsupported: return "请选择受支持的本地文档。"
            case .inputTooLarge: return "文档超过 20 MB。请分成几个文件后再导入。"
            case .timedOut: return "转换超过 30 秒。请分成几个文件后再试。"
            case .conversion: return "转换失败。请确认 MarkItDown 已安装相应格式的支持，且文档可以打开。"
            case .outputTooLarge: return "转换后的正文超过 256 KB。请分成几个文件后再导入。"
            case .empty: return "没有可导入的文字。扫描版 PDF 可以在扫描页识别。"
            }
        }
    }
}
