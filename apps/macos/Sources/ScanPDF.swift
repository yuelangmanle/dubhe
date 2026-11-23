import AppKit
import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers

/// Keep the document on disk and render one page at a time on the scanner queue.
final class ScanPDF {
    enum Resolution: String {
        case fast, standard, fine

        var side: CGFloat {
            switch self {
            case .fast: return 1280
            case .standard: return ScanImage.maxSide
            case .fine: return 3400
            }
        }
    }

    enum TextSource: String {
        case auto, embedded, ocr, system
    }

    private let document: PDFDocument
    private let temporaryDirectory: URL?
    private var locatedPages: [Int: [ScanTextLine]] = [:]
    let workingURL: URL
    let pageCount: Int

    static func supports(_ url: URL) -> Bool {
        ["pdf", "docx"].contains(url.pathExtension.lowercased())
    }

    init(url: URL) throws {
        if url.pathExtension.lowercased() == "docx" {
            let converted = try Self.convertWord(url)
            workingURL = converted.pdf
            temporaryDirectory = converted.directory
        } else {
            workingURL = url
            temporaryDirectory = nil
        }
        do {
            document = try Self.open(workingURL)
            pageCount = document.pageCount
        } catch {
            if let temporaryDirectory { try? FileManager.default.removeItem(at: temporaryDirectory) }
            throw error
        }
    }

    deinit {
        if let temporaryDirectory { try? FileManager.default.removeItem(at: temporaryDirectory) }
    }

    private static func open(_ url: URL) throws -> PDFDocument {
        guard let document = PDFDocument(url: url) else { throw ScanError.invalidPDF }
        guard !document.isLocked else { throw ScanError.lockedPDF }
        guard document.pageCount > 0 else { throw ScanError.invalidPDF }
        return document
    }

    private static func convertWord(_ url: URL) throws -> (pdf: URL, directory: URL) {
        let manager = FileManager.default
        let installed = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "org.libreoffice.script")
            ?? URL(fileURLWithPath: "/Applications/LibreOffice.app", isDirectory: true)
        let binary = installed.appendingPathComponent("Contents/MacOS/soffice")
        guard manager.isExecutableFile(atPath: binary.path) else { throw ScanError.wordRuntime }
        let directory = manager.temporaryDirectory.appendingPathComponent("Dubhe-Scan-" + UUID().uuidString)
        try manager.createDirectory(at: directory, withIntermediateDirectories: false,
                                    attributes: [.posixPermissions: 0o700])
        let pdf = directory.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".pdf")
        do {
            let process = Process()
            process.executableURL = binary
            let profile = directory.appendingPathComponent("profile", isDirectory: true)
            process.arguments = ["-env:UserInstallation=" + profile.absoluteString, "--headless",
                                 "--convert-to", "pdf:writer_pdf_Export", "--outdir", directory.path, url.path]
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            let finished = DispatchSemaphore(value: 0)
            process.terminationHandler = { _ in finished.signal() }
            try process.run()
            if finished.wait(timeout: .now() + 60) == .timedOut {
                process.terminate()
                throw ScanError.wordConversion
            }
            guard process.terminationStatus == 0, manager.fileExists(atPath: pdf.path) else {
                throw ScanError.wordConversion
            }
            return (pdf, directory)
        } catch {
            try? manager.removeItem(at: directory)
            throw error
        }
    }

    func selectableText(number: Int) -> String {
        guard (1...pageCount).contains(number) else { return "" }
        let text = document.page(at: number - 1)?.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return text.utf8.count <= 256 * 1024 ? text : ""
    }

    func originalPages(_ pages: [Int]? = nil) throws -> Data {
        guard let pages else { return try Data(contentsOf: workingURL, options: .mappedIfSafe) }
        let selected = PDFDocument()
        for number in pages {
            guard (1...pageCount).contains(number),
                  let page = document.page(at: number - 1)?.copy() as? PDFPage else {
                throw ScanError.invalidPage(number)
            }
            selected.insert(page, at: selected.pageCount)
        }
        guard let data = selected.dataRepresentation(), selected.pageCount > 0 else { throw ScanError.invalidPDF }
        return data
    }

    func searchablePDF(pages: [Int], notes: [String], reviewedPages: Set<Int> = []) throws -> Data {
        guard !pages.isEmpty, pages.count == notes.count, Set(pages).count == pages.count else {
            throw ScanError.invalidPDF
        }
        let text = Dictionary(uniqueKeysWithValues: zip(pages, notes))
        if let updated = try ScanSearchablePDF.render(source: document, textByPage: text, locate: { number in
            if reviewedPages.contains(number) { return [] }
            if let lines = locatedPages[number] { return lines }
            return (try? locateText(number: number, resolution: .standard)) ?? []
        }) { return updated }
        return try originalPages()
    }

    static func imagePDF(_ image: NSImage) throws -> Data {
        let pdf = PDFDocument()
        guard let page = PDFPage(image: image) else { throw ScanError.image }
        pdf.insert(page, at: 0)
        guard let data = pdf.dataRepresentation() else { throw ScanError.image }
        return data
    }

    func preparePage(number: Int, resolution: Resolution, textSource: TextSource) throws -> (image: String?, text: String) {
        guard (1...pageCount).contains(number) else { throw ScanError.invalidPage(number) }
        if textSource == .system {
            let lines = try locateText(number: number, resolution: resolution)
            locatedPages[number] = lines
            return (nil, lines.map(\.text).joined(separator: "\n"))
        }
        let text = textSource == .ocr ? "" : selectableText(number: number)
        if textSource == .embedded || textSource == .auto && Self.hasReadableText(text) {
            return (nil, text)
        }
        return (try pageDataURL(number: number, resolution: resolution), text)
    }

    static func hasReadableText(_ text: String) -> Bool {
        text.unicodeScalars.lazy.filter { CharacterSet.alphanumerics.contains($0) }.prefix(20).count == 20
    }

    func pageDataURL(number: Int, resolution: Resolution = .standard) throws -> String {
        try ScanImage.encodeJPEG(rasterImage(number: number, resolution: resolution))
    }

    private func locateText(number: Int, resolution: Resolution) throws -> [ScanTextLine] {
        try autoreleasepool { try ScanTextLine.recognize(in: rasterImage(number: number, resolution: resolution)) }
    }

    private func rasterImage(number: Int, resolution: Resolution) throws -> CGImage {
        guard (1...pageCount).contains(number), let page = document.page(at: number - 1)?.pageRef else {
            throw ScanError.invalidPage(number)
        }
        let crop = page.getBoxRect(.cropBox)
        let rotated = page.rotationAngle % 180 != 0
        let width = rotated ? crop.height : crop.width
        let height = rotated ? crop.width : crop.height
        guard crop.minX.isFinite, crop.minY.isFinite, crop.maxX.isFinite, crop.maxY.isFinite,
              width > 0, height > 0 else { throw ScanError.invalidPage(number) }
        let span = max(width, height)
        let maxSide = resolution.side
        let w = max(1, Int((width / span * maxSide).rounded()))
        let h = max(1, Int((height / span * maxSide).rounded()))
        let scale = min(CGFloat(w) / width, CGFloat(h) / height)
        guard scale.isFinite else { throw ScanError.invalidPage(number) }
        let context = try ScanImage.whiteCanvas(width: w, height: h)
        let pageBounds = CGRect(x: 0, y: 0, width: width, height: height)
        let transform = page.getDrawingTransform(.cropBox, rect: pageBounds, rotate: 0, preserveAspectRatio: true)
        context.translateBy(x: (CGFloat(w) - width * scale) / 2, y: (CGFloat(h) - height * scale) / 2)
        context.scaleBy(x: scale, y: scale)
        context.clip(to: pageBounds)
        context.concatenate(transform)
        context.drawPDFPage(page)
        guard let image = context.makeImage() else { throw ScanError.invalidPage(number) }
        return image
    }
}

enum ScanImage {
    static let maxSide: CGFloat = 2400

    static func dataURL(_ image: CGImage) throws -> String {
        let scale = min(1, maxSide / CGFloat(max(image.width, image.height)))
        let width = max(1, Int((CGFloat(image.width) * scale).rounded()))
        let height = max(1, Int((CGFloat(image.height) * scale).rounded()))
        let context = try whiteCanvas(width: width, height: height)
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resized = context.makeImage() else { throw ScanError.image }
        return try encodeJPEG(resized)
    }

    fileprivate static func whiteCanvas(width: Int, height: Int) throws -> CGContext {
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { throw ScanError.image }
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context
    }

    fileprivate static func encodeJPEG(_ image: CGImage) throws -> String {
        for quality in [0.9, 0.72] {
            let bytes = NSMutableData()
            guard let writer = CGImageDestinationCreateWithData(bytes as CFMutableData,
                                                                 UTType.jpeg.identifier as CFString, 1, nil) else {
                throw ScanError.image
            }
            CGImageDestinationAddImage(writer, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
            guard CGImageDestinationFinalize(writer) else { throw ScanError.image }
            if bytes.length <= 4 * 1024 * 1024 {
                return "data:image/jpeg;base64," + bytes.base64EncodedString()
            }
        }
        throw ScanError.image
    }
}

private enum ScanError: LocalizedError {
    case invalidPDF, lockedPDF, invalidPage(Int), image, wordRuntime, wordConversion

    var errorDescription: String? {
        switch self {
        case .invalidPDF: return "无法读取 PDF，文件可能损坏或没有页面。"
        case .lockedPDF: return "此 PDF 需要密码。请先解锁，再重新打开。"
        case .invalidPage(let number): return "无法读取 PDF 第 \(number) 页。"
        case .image: return "无法准备扫描图片。请重试，或缩小选取范围。"
        case .wordRuntime: return "打开 Word 原版面需要 LibreOffice。也可在 Word 中另存为 PDF 后打开。"
        case .wordConversion: return "无法生成 Word 页面。请检查文档是否可打开，或另存为 PDF 后重试。"
        }
    }
}
