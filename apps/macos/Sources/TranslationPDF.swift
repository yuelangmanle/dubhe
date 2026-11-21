import AppKit
import CoreGraphics
import CoreText
import Foundation
import PDFKit

enum TranslationPDF {
    static func paragraphs(_ data: Data) throws -> [String] {
        let pdf = try open(data)
        let text = try (0..<pdf.pageCount).map { index in
            guard let page = pdf.page(at: index) else { throw Failure.invalidPage(index + 1) }
            return (page.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard text.contains(where: { !$0.isEmpty }) else { throw Failure.noText }
        return text
    }

    static func rebuild(_ data: Data, translations: [String]) throws -> Data {
        let source = try open(data)
        guard translations.count == source.pageCount else { throw Failure.translationCount }
        return try rebuild(source, pages: Array(1...source.pageCount), translations: translations)
    }

    static func rebuildSelectedPages(url: URL, pages: [Int], translations: [String]) throws -> Data {
        guard let source = PDFDocument(url: url), source.pageCount > 0 else { throw Failure.invalidPDF }
        guard !source.isLocked else { throw Failure.locked }
        return try rebuild(source, pages: pages, translations: translations)
    }

    private static func rebuild(_ source: PDFDocument, pages: [Int], translations: [String]) throws -> Data {
        guard !pages.isEmpty, pages.count == translations.count else { throw Failure.translationCount }
        let buffer = NSMutableData()
        guard let consumer = CGDataConsumer(data: buffer as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: nil, nil) else {
            throw Failure.output
        }
        for (number, translation) in zip(pages, translations) {
            guard (1...source.pageCount).contains(number), let page = source.page(at: number - 1) else {
                throw Failure.invalidPage(number)
            }
            try addPage(page, number: number, translation: translation, to: context)
        }
        // Closing the context writes the PDF trailer before the buffer is copied.
        context.closePDF()
        guard buffer.length > 0 else { throw Failure.output }
        return buffer as Data
    }

    private static func open(_ data: Data) throws -> PDFDocument {
        guard !data.isEmpty, let pdf = PDFDocument(data: data) else { throw Failure.invalidPDF }
        guard !pdf.isLocked else { throw Failure.locked }
        guard pdf.pageCount > 0 else { throw Failure.invalidPDF }
        return pdf
    }

    private static func addPage(_ page: PDFPage, number: Int, translation: String, to context: CGContext) throws {
        let crop = page.bounds(for: .cropBox)
        let rotated = page.rotation % 180 != 0
        guard let pageRef = page.pageRef else { throw Failure.invalidPage(number) }
        var unit: CGFloat = 1
        if let dictionary = pageRef.dictionary {
            CGPDFDictionaryGetNumber(dictionary, "UserUnit", &unit)
        }
        let width = (rotated ? crop.height : crop.width) * unit
        let height = (rotated ? crop.width : crop.height) * unit
        let pageWidth = width + max(width, 260)
        let pageHeight = max(height, 160)
        guard crop.minX.isFinite, crop.minY.isFinite, crop.maxX.isFinite, crop.maxY.isFinite,
              width > 0, height > 0, pageWidth.isFinite, pageHeight.isFinite else {
            throw Failure.invalidPage(number)
        }

        let original = CGRect(x: 0, y: pageHeight - height, width: width, height: height)
        var bounds = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)
        let box = Data(bytes: &bounds, count: MemoryLayout<CGRect>.size)
        let properties = [kCGPDFContextMediaBox as String: box] as CFDictionary
        let margin: CGFloat = 28
        let content = styled(translation.trimmingCharacters(in: .whitespacesAndNewlines), size: 12)
        let typesetter = CTFramesetterCreateWithAttributedString(content)
        var offset = 0

        repeat {
            let isContinuation = offset > 0
            context.beginPDFPage(properties)
            defer { context.endPDFPage() }
            if !isContinuation {
                context.saveGState()
                context.clip(to: original)
                context.translateBy(x: original.minX, y: original.minY)
                context.scaleBy(x: unit, y: unit)
                page.draw(with: .cropBox, to: context)
                context.restoreGState()
            }
            if content.length == 0 { break }

            let left = (isContinuation ? 0 : width) + margin
            let label = isContinuation ? "第 \(number) 页译文（续）" : "译文"
            context.textMatrix = .identity
            context.textPosition = CGPoint(x: left, y: pageHeight - margin - 13)
            CTLineDraw(CTLineCreateWithAttributedString(styled(label, size: 13)), context)
            let textBounds = CGRect(x: left, y: margin,
                                    width: min(pageWidth - left - margin, 560),
                                    height: pageHeight - 2 * margin - 28)
            let path = CGPath(rect: textBounds, transform: nil)
            let frame = CTFramesetterCreateFrame(typesetter, CFRange(location: offset, length: 0), path, nil)
            let visible = CTFrameGetVisibleStringRange(frame)
            guard visible.location == offset, visible.length > 0, visible.length <= content.length - offset else {
                throw Failure.layout(number)
            }
            CTFrameDraw(frame, context)
            offset += visible.length
        } while offset < content.length
    }

    private static func styled(_ text: String, size: CGFloat) -> NSAttributedString {
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineSpacing = 3
        return NSAttributedString(string: text, attributes: [
            .font: CTFontCreateWithName("Helvetica Neue" as CFString, size, nil),
            .paragraphStyle: paragraph,
        ])
    }

    private enum Failure: LocalizedError {
        case invalidPDF, locked, noText, translationCount, invalidPage(Int), layout(Int), output

        var errorDescription: String? {
            switch self {
            case .invalidPDF: return "无法读取 PDF，文件可能损坏或没有页面。"
            case .locked: return "此 PDF 需要密码，请先解锁再翻译。"
            case .noText: return "PDF 中没有可提取的文字。如果它是扫描件，请先进行文字识别。"
            case .translationCount: return "译文页数与原 PDF 页数不一致。"
            case .invalidPage(let number): return "无法读取 PDF 第 \(number) 页的内容或页面尺寸。"
            case .layout(let number): return "无法排版 PDF 第 \(number) 页的译文。"
            case .output: return "无法生成双语 PDF。"
            }
        }
    }
}
