import AppKit
import CoreGraphics
import CoreText
import Foundation
import PDFKit
import Vision

struct ScanTextLine {
    let text: String
    let bounds: CGRect

    static func recognize(in image: CGImage) throws -> [ScanTextLine] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.automaticallyDetectsLanguage = true
        request.usesLanguageCorrection = false
        if let supported = try? request.supportedRecognitionLanguages() {
            request.recognitionLanguages = ["en-US", "zh-Hans", "zh-Hant", "ja-JP", "ko-KR"].filter(supported.contains)
        }
        try VNImageRequestHandler(cgImage: image).perform([request])
        return (request.results ?? []).compactMap { observation in
            guard let text = observation.topCandidates(1).first?.string, !text.isEmpty else { return nil }
            return ScanTextLine(text: text, bounds: observation.boundingBox)
        }
    }
}

enum ScanSearchablePDF {
    static func render(source: PDFDocument, textByPage: [Int: String],
                       locate: (Int) -> [ScanTextLine]) throws -> Data? {
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
            try autoreleasepool {
                try addPage(page, text: text, lines: locate(index + 1), number: index + 1, context: context)
            }
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

    private static func addPage(_ page: CGPDFPage, text: String, lines: [ScanTextLine], number: Int,
                                context: CGContext) throws {
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

        context.saveGState()
        defer { context.restoreGState() }
        context.textMatrix = .identity
        context.setTextDrawingMode(.invisible)
        if !lines.isEmpty {
            for line in lines { draw(line, width: width, height: height, context: context) }
            return
        }
        // Keep the recognized notes searchable when the system cannot locate any lines.
        let margin = min(18, min(width, height) * 0.04)
        let path = CGPath(rect: bounds.insetBy(dx: margin, dy: margin), transform: nil)
        var pointSize = min(11, min(width, height) / 8)
        let content = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for _ in 0..<20 {
            let attributed = attributedText(content, size: pointSize)
            let setter = CTFramesetterCreateWithAttributedString(attributed)
            let frame = CTFramesetterCreateFrame(setter, CFRange(location: 0, length: 0), path, nil)
            if CTFrameGetVisibleStringRange(frame).length == attributed.length {
                CTFrameDraw(frame, context)
                return
            }
            pointSize *= 0.78
        }
        throw Failure.invalidPage(number)
    }

    private static func draw(_ located: ScanTextLine, width: CGFloat, height: CGFloat, context: CGContext) {
        let rect = CGRect(x: located.bounds.minX * width, y: located.bounds.minY * height,
                          width: located.bounds.width * width, height: located.bounds.height * height)
        let line = CTLineCreateWithAttributedString(attributedText(located.text, size: max(1, rect.height)))
        var ascent: CGFloat = 0
        var descent: CGFloat = 0
        let advance = CTLineGetTypographicBounds(line, &ascent, &descent, nil)
        guard advance > 0, ascent + descent > 0, rect.width > 0, rect.height > 0 else { return }
        let yScale = rect.height / (ascent + descent)
        context.saveGState()
        context.translateBy(x: rect.minX, y: rect.minY + descent * yScale)
        context.scaleBy(x: rect.width / advance, y: yScale)
        context.textPosition = .zero
        CTLineDraw(line, context)
        context.restoreGState()
    }

    private static func attributedText(_ text: String, size: CGFloat) -> NSAttributedString {
        // This font keeps Chinese characters mapped to their original Unicode values in PDF search.
        let font = CTFontCreateWithName("ArialUnicodeMS" as CFString, size, nil)
        return NSAttributedString(string: text, attributes: [.font: font])
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
