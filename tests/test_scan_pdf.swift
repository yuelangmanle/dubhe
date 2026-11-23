import CoreGraphics
import Foundation
import ImageIO
import PDFKit

@main
enum ScanPDFTests {
    static func main() throws {
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        let pdf = PDFDocument(data: makePDF())!
        let first = pdf.page(at: 0)!
        first.setBounds(CGRect(x: 20, y: 30, width: 200, height: 140), for: .cropBox)
        first.rotation = 90
        let url = temp.appendingPathComponent("report.pdf")
        try pdf.dataRepresentation()!.write(to: url)

        let scan = try ScanPDF(url: url)
        precondition(scan.pageCount == 2, "PDF pages use the original order")
        let original = try scan.originalPages()
        let input = try Data(contentsOf: url)
        precondition(original == input, "whole-original PDF export preserves the original bytes")
        let rotated = try decode(scan.pageDataURL(number: 1))
        precondition(rotated.width == 1680 && rotated.height == 2400, "rotation determines page dimensions")
        let red = pixel(rotated, x: 250, y: 290)
        precondition(red[0] > 200 && red[1] < 60 && red[2] < 60,
                     "the cropped, rotated mark failed to appear in place: \(red)")
        let blue = pixel(rotated, x: 1320, y: 1800)
        precondition(blue[2] > 200 && blue[0] < 60 && blue[1] < 60,
                     "the other mark failed to appear in place: \(blue)")
        let white = pixel(rotated, x: 600, y: 600)
        precondition(white.prefix(3).allSatisfy { $0 > 245 }, "the blank page background is white")
        let second = try decode(scan.pageDataURL(number: 2))
        precondition(second.width == 2400 && second.height == 1920, "the second page renders independently")
        let fine = try decode(scan.pageDataURL(number: 1, resolution: .fine))
        precondition(fine.height == 3400, "high-detail mode renders the PDF at a larger size")
        let fast = try scan.preparePage(number: 1, resolution: .fast, textSource: .auto)
        precondition(fast.text.isEmpty && decode(fast.image!).height == 1280, "quick OCR uses a smaller image directly")
        let textOnly = try scan.preparePage(number: 1, resolution: .standard, textSource: .embedded)
        precondition(textOnly.image == nil, "text-only requests never rasterize the page")
        let exported = PDFDocument(data: try scan.originalPages([1]))!
        precondition(exported.pageCount == 1 && exported.page(at: 0)?.rotation == 90)
        precondition(exported.page(at: 0)?.bounds(for: .cropBox) == first.bounds(for: .cropBox),
                     "export preserves the crop box")
        let note = "Daily river discharge 水位 10.2"
        let searchableData = try scan.searchablePDF(pages: [1], notes: [note])
        let searchable = PDFDocument(data: searchableData)!
        precondition(searchable.pageCount == 2, "export preserves the unprocessed second page")
        let extracted = searchable.page(at: 0)?.string ?? ""
        precondition(extracted.split(whereSeparator: \.isWhitespace).joined(separator: " ").contains(note),
                     "the scanned page has searchable Unicode OCR text: \(extracted.debugDescription)")
        precondition(searchable.findString("水位").count == 1, "Chinese keywords are findable in the PDF")
        precondition(searchable.page(at: 1)?.string?.contains(note) != true,
                     "the OCR text stays on its page")
        let searchableFile = temp.appendingPathComponent("searchable.pdf")
        try searchableData.write(to: searchableFile)
        let searchableScan = try ScanPDF(url: searchableFile)
        let layered = try decode(searchableScan.pageDataURL(number: 1))
        precondition(layered.width == rotated.width && layered.height == rotated.height,
                     "export preserves the visible size and orientation of a rotated crop")
        precondition(pixel(layered, x: 250, y: 290) == red && pixel(layered, x: 1320, y: 1800) == blue)
        let unchanged = try searchableScan.searchablePDF(pages: [1], notes: ["Text already present"])
        precondition(unchanged == searchableData, "never add a duplicate text layer to a searchable page")
        for number in [0, 3] {
            do {
                _ = try scan.pageDataURL(number: number)
                fatalError("invalid page numbers must fail")
            } catch {
                precondition(error.localizedDescription.contains("第 \(number) 页"))
            }
        }

        let locked = temp.appendingPathComponent("locked.pdf")
        try makePDF(password: "open-me").write(to: locked)
        do {
            _ = try ScanPDF(url: locked)
            fatalError("password-protected PDFs must report their state")
        } catch {
            precondition(error.localizedDescription.contains("密码"))
        }

        let tall = CGContext(data: nil, width: 800, height: 3200, bitsPerComponent: 8,
                             bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let resized = try decode(ScanImage.dataURL(tall))
        precondition(resized.width == 600 && resized.height == 2400, "oversize screenshots scale proportionally")
        precondition(pixel(resized, x: 300, y: 1200).prefix(3).allSatisfy { $0 > 245 },
                     "transparent screenshots render over white")

        let request = try ScanOCR.request(port: 7790, image: tall)
        precondition(request.url?.absoluteString == "http://127.0.0.1:7790/api/scan")
        precondition(request.value(forHTTPHeaderField: "Origin") == "http://127.0.0.1:7790")
        let payload = try JSONDecoder().decode([String: String].self, from: request.httpBody!)
        precondition(payload["mode"] == "text")
        precondition(payload["image"]?.hasPrefix("data:image/jpeg;base64,") == true)
        let ok = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)
        let recognized = try ScanOCR.recognizedText(Data(#"{"text":"  OCR 原文  "}"#.utf8), response: ok, error: nil)
        precondition(recognized == "OCR 原文")
        let missing = HTTPURLResponse(url: request.url!, statusCode: 404, httpVersion: nil, headerFields: nil)
        do {
            _ = try ScanOCR.recognizedText(Data(#"{"error":"请先下载小米 OCR"}"#.utf8), response: missing, error: nil)
            fatalError("the native screenshot path must surface the scanner's missing-model error")
        } catch {
            precondition(error.localizedDescription == "请先下载小米 OCR")
        }
        print("Scanner PDF rendering, screenshot payload, and missing-model feedback passed")
    }

    private static func makePDF(password: String? = nil) -> Data {
        let buffer = NSMutableData()
        let consumer = CGDataConsumer(data: buffer as CFMutableData)!
        var box = CGRect(x: 0, y: 0, width: 250, height: 200)
        let options = password.map { [kCGPDFContextUserPassword as String: $0,
                                      kCGPDFContextOwnerPassword as String: "owner"] as CFDictionary }
        let context = CGContext(consumer: consumer, mediaBox: &box, options)!
        context.beginPDFPage(nil)
        context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 35, y: 40, width: 20, height: 20))
        context.setFillColor(red: 0, green: 0, blue: 1, alpha: 1)
        context.fill(CGRect(x: 160, y: 130, width: 20, height: 20))
        context.endPDFPage()
        context.beginPDFPage(nil)
        context.endPDFPage()
        context.closePDF()
        return buffer as Data
    }

    private static func decode(_ dataURL: String) -> CGImage {
        let prefix = "data:image/jpeg;base64,"
        precondition(dataURL.hasPrefix(prefix))
        let jpg = Data(base64Encoded: String(dataURL.dropFirst(prefix.count)))!
        precondition(jpg.count <= 4 * 1024 * 1024, "the scan payload stays below the server limit")
        return CGImageSourceCreateImageAtIndex(CGImageSourceCreateWithData(jpg as CFData, nil)!, 0, nil)!
    }

    private static func pixel(_ image: CGImage, x: Int, y: Int) -> [UInt8] {
        let cropped = image.cropping(to: CGRect(x: x, y: y, width: 1, height: 1))!
        var rgba = [UInt8](repeating: 0, count: 4)
        rgba.withUnsafeMutableBytes { data in
            let context = CGContext(data: data.baseAddress, width: 1, height: 1, bitsPerComponent: 8,
                                    bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(cropped, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        }
        return rgba
    }
}
