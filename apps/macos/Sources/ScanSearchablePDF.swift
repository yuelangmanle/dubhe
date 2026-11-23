import AppKit
import CoreGraphics
import CoreText
import Foundation
import PDFKit

enum ScanSearchablePDF {
    static func render(source: PDFDocument, textByPage: [Int: String]) throws -> Data? {
        var overlays: [(index: Int, page: CGPDFPage, text: String)] = []
        for (number, text) in textByPage.sorted(by: { $0.key < $1.key }) {
            guard number >= 1, number <= source.pageCount,
                  let page = source.page(at: number - 1), let drawing = page.pageRef else {
                throw Failure.invalidPage(number)
            }
            // Keep native text and the original PDF page objects where the document already has a text layer.
            if !ScanPDF.hasReadableText(page.string ?? ""), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                overlays.append((number - 1, drawing, text))
            }
        }
        guard !overlays.isEmpty else { return nil }

        let buffer = NSMutableData()
        guard let consumer = CGDataConsumer(data: buffer as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: nil, nil) else { throw Failure.output }
        for (index, page, text) in overlays {
            try addPage(page, text: text, number: index + 1, context: context)
        }
        context.closePDF()
        guard let recognized = PDFDocument(data: buffer as Data), recognized.pageCount == overlays.count else {
            throw Failure.output
        }

        let replacement = Dictionary(uniqueKeysWithValues: overlays.enumerated().map { ($0.element.index, $0.offset) })
        let result = PDFDocument()
        result.documentAttributes = source.documentAttributes
        for index in 0..<source.pageCount {
            let page = replacement[index].flatMap { recognized.page(at: $0) } ?? source.page(at: index)
            guard let copy = page?.copy() as? PDFPage else { throw Failure.invalidPage(index + 1) }
            result.insert(copy, at: result.pageCount)
        }
        guard let data = result.dataRepresentation() else { throw Failure.output }
        return data
    }

    private static func addPage(_ page: CGPDFPage, text: String, number: Int, context: CGContext) throws {
        let crop = page.getBoxRect(.cropBox)
        var unit: CGFloat = 1
        if let dictionary = page.dictionary { CGPDFDictionaryGetNumber(dictionary, "UserUnit", &unit) }
        let rotated = page.rotationAngle % 180 != 0
        let width = (rotated ? crop.height : crop.width) * unit
        let height = (rotated ? crop.width : crop.height) * unit
        guard width.isFinite, height.isFinite, width > 0, height > 0 else { throw Failure.invalidPage(number) }
        var bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let box = Data(bytes: &bounds, count: MemoryLayout<CGRect>.size)
        context.beginPDFPage([kCGPDFContextMediaBox as String: box] as CFDictionary)
        defer { context.endPDFPage() }

        context.saveGState()
        context.clip(to: bounds)
        context.concatenate(page.getDrawingTransform(.cropBox, rect: bounds, rotate: 0, preserveAspectRatio: true))
        context.drawPDFPage(page)
        context.restoreGState()

        let margin = min(18, min(width, height) * 0.04)
        let path = CGPath(rect: bounds.insetBy(dx: margin, dy: margin), transform: nil)
        var pointSize = min(11, min(width, height) / 8)
        let content = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Fit the reading-order text to the existing page. Its glyph positions are not a geometric OCR map.
        for _ in 0..<20 {
            // Arial Unicode keeps the Chinese glyphs mapped to CJK code points in the PDF's ToUnicode table.
            let font = CTFontCreateWithName("ArialUnicodeMS" as CFString, pointSize, nil)
            let attributed = NSAttributedString(string: content, attributes: [.font: font])
            let setter = CTFramesetterCreateWithAttributedString(attributed)
            let frame = CTFramesetterCreateFrame(setter, CFRange(location: 0, length: 0), path, nil)
            if CTFrameGetVisibleStringRange(frame).length == attributed.length {
                context.saveGState()
                context.textMatrix = .identity
                context.setTextDrawingMode(.invisible)
                CTFrameDraw(frame, context)
                context.restoreGState()
                return
            }
            pointSize *= 0.78
        }
        throw Failure.invalidPage(number)
    }

    private enum Failure: LocalizedError {
        case invalidPage(Int), output

        var errorDescription: String? {
            switch self {
            case .invalidPage(let number): return "无法为第 \(number) 页生成可搜索 PDF。"
            case .output: return "无法生成可搜索 PDF。"
            }
        }
    }
}
