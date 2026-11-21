// Run on macOS with make integration-test.
import AppKit
import CoreGraphics
import CoreText
import Foundation
import PDFKit

@main
enum TestTranslationPDF {
    static func main() throws {
        let original = samplePDF(text: "Original English text")
        let source = PDFDocument(data: original)!
        require(source.pageCount == 2, "the fixture has a text page and a blank page")
        require(try TranslationPDF.paragraphs(original) == ["Original English text", ""],
                "extraction keeps the source page indexes")

        let translated = "中文译文。日本語の翻訳。한국어 번역. English translation."
        let output = try TranslationPDF.rebuild(original, translations: [translated, ""])
        let result = PDFDocument(data: output)!
        require(result.pageCount == 2, "one output page per source page when the text fits")
        require(result.page(at: 0)?.string?.contains("Original English text") == true,
                "original text remains searchable")
        require(result.page(at: 0)?.string?.contains(translated) == true,
                "Chinese, Japanese, Korean and English translations remain searchable")
        let left = result.findString("Original English text", withOptions: []).first!
        let right = result.findString("中文译文", withOptions: []).first!
        require(left.bounds(for: result.page(at: 0)!).origin.x < 50, "original text stays on the left")
        require(right.bounds(for: result.page(at: 0)!).origin.x > 400, "translation starts to the right of the source")
        let resultPage = result.page(at: 0)!
        require(resultPage.bounds(for: .mediaBox).height == 300, "preserves the source height")
        require(isRed(pixelOnPage(resultPage, at: CGPoint(x: 28, y: 28))), "preserves source graphics at the same position and scale")

        let rotatedPage = source.page(at: 0)!
        let square = PDFAnnotation(bounds: CGRect(x: 60, y: 18, width: 24, height: 24),
                                   forType: .square, withProperties: nil)
        square.interiorColor = .red
        rotatedPage.addAnnotation(square)
        rotatedPage.rotation = 90
        rotatedPage.setBounds(CGRect(x: 8, y: 8, width: 380, height: 280), for: .cropBox)
        let rotated = try TranslationPDF.rebuild(source.dataRepresentation()!, translations: [translated, ""])
        let rotatedDocument = PDFDocument(data: rotated)!
        let rotatedOutputPage = rotatedDocument.page(at: 0)!
        require(rotatedOutputPage.bounds(for: .mediaBox).height == 380, "output uses the displayed, rotated crop dimensions")
        require(isRed(pixelOnPage(rotatedOutputPage, at: CGPoint(x: 20, y: 360))), "rotates cropped source graphics at their exact scale")
        require(isRed(pixelOnPage(rotatedOutputPage, at: CGPoint(x: 20, y: 316))), "preserves the appearance of PDF annotations")

        let words = (1...75).map { "Section \($0) 这一句必须完整保留。" }
        let longOutput = try TranslationPDF.rebuild(original, translations: [words.joined(separator: "\n"), ""])
        let longResult = PDFDocument(data: longOutput)!
        require(longResult.pageCount > 2, "long translations generate continuation pages")
        for word in words {
            require(longResult.string?.contains(word) == true, "continuations preserve \(word)")
        }
        require(longResult.findString("Original English text", withOptions: []).count == 1,
                "continuation pages do not repeat the source")

        let smallOriginal = samplePDF(text: "Original", width: 100, height: 100)
        let smallResult = PDFDocument(data: try TranslationPDF.rebuild(smallOriginal, translations: [translated, ""]))!
        let smallPage = smallResult.page(at: 0)!
        require(smallPage.bounds(for: .mediaBox).size == CGSize(width: 360, height: 160),
                "a small source gets enough room for the translation without being scaled")
        require(isRed(pixelOnPage(smallPage, at: CGPoint(x: 28, y: 88))), "small source is drawn at its original size")

        let scan = samplePDF(text: nil)
        let scanURL = FileManager.default.temporaryDirectory.appendingPathComponent("Dubhe-\(UUID()).pdf")
        try scan.write(to: scanURL)
        defer { try? FileManager.default.removeItem(at: scanURL) }
        let pairedScan = PDFDocument(data: try TranslationPDF.rebuildSelectedPages(
            url: scanURL, pages: [1], translations: [translated]))!
        require(pairedScan.pageCount == 1, "only requested pages are exported")
        let pairedPage = pairedScan.page(at: 0)!
        require(isRed(pixelOnPage(pairedPage, at: CGPoint(x: 28, y: 28))), "scan graphics remain on the left")
        require(pairedPage.string?.contains(translated) == true, "the scan translation is searchable")
        expectError("文字") { _ = try TranslationPDF.paragraphs(scan) }
        expectError("PDF") { _ = try TranslationPDF.paragraphs(Data()) }
        expectError("页数") { _ = try TranslationPDF.rebuild(original, translations: [translated]) }
        let locked = samplePDF(text: "Secret text", password: "reader")
        expectError("密码") { _ = try TranslationPDF.paragraphs(locked) }

        let blocks: [[String: String]] = [
            ["kind": "h1", "text": "流域洪水响应"],
            ["kind": "p", "text": "NSE 从 0.72 降到 0.61。English 中文。"],
            ["kind": "li", "text": "• 使用相同的观测站。"],
            ["kind": "table", "text": "流域 │ 峰值\n上游 │ 6"],
        ] + (1...100).map { ["kind": "p", "text": "段落 \($0) 需要留在导出结果里。"] }
        let standalone = try TranslationExport.render(format: "pdf", blocks: blocks)
        let standalonePDF = PDFDocument(data: standalone)!
        require(standalonePDF.pageCount > 1, "standalone translations paginate")
        require(standalonePDF.string?.contains("段落 100") == true, "the standalone PDF retains its last paragraph")
        let docx = try TranslationExport.render(format: "docx", blocks: blocks)
        let word = try NSAttributedString(data: docx,
            options: [.documentType: NSAttributedString.DocumentType.officeOpenXML], documentAttributes: nil)
        require(word.string.contains("NSE 从 0.72 降到 0.61。"), "Word round trip preserves Unicode and values")
        require(word.string.contains("段落 100"), "Word round trip preserves the last paragraph")

        if CommandLine.arguments.count > 1 {
            let destination = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
            try original.write(to: destination.appendingPathComponent("source.pdf"))
            try output.write(to: destination.appendingPathComponent("bilingual.pdf"))
            try rotated.write(to: destination.appendingPathComponent("bilingual-rotated.pdf"))
            try longOutput.write(to: destination.appendingPathComponent("bilingual-long.pdf"))
            try standalone.write(to: destination.appendingPathComponent("translation.pdf"))
            try docx.write(to: destination.appendingPathComponent("translation.docx"))
        }
        print("PDF extraction, bilingual layout, searchable Unicode and pagination passed")
    }

    private static func require(_ condition: Bool, _ message: String) {
        guard condition else { fatalError(message) }
    }

    private static func expectError(_ message: String, _ operation: () throws -> Void) {
        do {
            try operation()
            fatalError("expected a PDF error containing \(message)")
        } catch {
            require(error.localizedDescription.contains(message), "unexpected error: \(error.localizedDescription)")
        }
    }

    private static func samplePDF(text: String?, width: CGFloat = 400, height: CGFloat = 300, password: String? = nil) -> Data {
        let buffer = NSMutableData()
        let consumer = CGDataConsumer(data: buffer as CFMutableData)!
        var bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let options = password.map { [kCGPDFContextUserPassword as String: $0,
                                      kCGPDFContextOwnerPassword as String: "owner"] as CFDictionary }
        let context = CGContext(consumer: consumer, mediaBox: &bounds, options)!
        context.beginPDFPage(nil)
        context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 18, y: 18, width: 24, height: 24))
        if let text {
            let attributes = [NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 14, nil)]
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
            context.textPosition = CGPoint(x: 24, y: height - 50)
            CTLineDraw(line, context)
        }
        context.endPDFPage()
        context.beginPDFPage(nil)
        context.endPDFPage()
        context.closePDF()
        return buffer as Data
    }

    private static func pixelOnPage(_ page: PDFPage, at point: CGPoint) -> [UInt8] {
        var rgba = [UInt8](repeating: 0, count: 4)
        rgba.withUnsafeMutableBytes { data in
            let context = CGContext(data: data.baseAddress, width: 1, height: 1, bitsPerComponent: 8,
                                    bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.translateBy(x: -point.x, y: -point.y)
            context.drawPDFPage(page.pageRef!)
        }
        return rgba
    }

    private static func isRed(_ pixel: [UInt8]) -> Bool {
        pixel[0] > 220 && pixel[1] < 30 && pixel[2] < 30 && pixel[3] > 220
    }
}
