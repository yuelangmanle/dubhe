import AppKit
import CoreText
import ImageIO
import PDFKit
import UniformTypeIdentifiers
import Vision

struct FixtureError: Error {
    let message: String
}

func canvas(width: Int, height: Int) throws -> CGContext {
    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
        throw FixtureError(message: "无法创建图像。")
    }
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context
}

func raster(_ page: PDFPage, scale: CGFloat) throws -> CGImage {
    let bounds = page.bounds(for: .mediaBox)
    let context = try canvas(width: Int(ceil(bounds.width * scale)), height: Int(ceil(bounds.height * scale)))
    context.scaleBy(x: scale, y: scale)
    context.translateBy(x: -bounds.minX, y: -bounds.minY)
    page.draw(with: .mediaBox, to: context)
    guard let image = context.makeImage() else { throw FixtureError(message: "无法绘制 PDF 页面。") }
    return image
}

func save(_ image: CGImage?, to url: URL) throws {
    guard let image,
          let target = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw FixtureError(message: "无法创建 PNG。")
    }
    CGImageDestinationAddImage(target, image, nil)
    guard CGImageDestinationFinalize(target) else { throw FixtureError(message: "无法写入 PNG。") }
}

func document(_ path: String) throws -> PDFDocument {
    let url = URL(fileURLWithPath: path)
    guard let pdf = PDFDocument(url: url), !pdf.isLocked, pdf.pageCount > 0 else {
        throw FixtureError(message: "无法读取 PDF。")
    }
    return pdf
}

func scannedCopy(_ input: String, _ output: String) throws {
    let pdf = try document(input)
    guard let context = CGContext(URL(fileURLWithPath: output) as CFURL, mediaBox: nil, nil) else {
        throw FixtureError(message: "无法创建扫描件 PDF。")
    }
    defer { context.closePDF() }
    for index in 0..<pdf.pageCount {
        try autoreleasepool {
            guard let page = pdf.page(at: index) else { throw FixtureError(message: "PDF 页面不可用。") }
            var size = page.bounds(for: .mediaBox)
            let info = [kCGPDFContextMediaBox: Data(bytes: &size, count: MemoryLayout<CGRect>.size)] as CFDictionary
            let image = try raster(page, scale: 1.15)
            context.beginPDFPage(info)
            context.draw(image, in: size)
            context.endPDFPage()
        }
    }
}

func bilingual(_ path: String) throws {
    let canvas = try canvas(width: 1000, height: 390)
    canvas.setFillColor(CGColor(red: 0.94, green: 0.97, blue: 0.94, alpha: 1))
    canvas.fill(CGRect(x: 64, y: 78, width: 870, height: 260))
    let rows = ["河流观测 · River monitoring", "站点编号 A-17 · Station ID", "2026-05-03"]
    for (index, row) in rows.enumerated() {
        let value = NSAttributedString(string: row, attributes: [
            .font: NSFont.systemFont(ofSize: 42, weight: index == 0 ? .semibold : .regular),
            .foregroundColor: NSColor.black,
        ])
        canvas.textPosition = CGPoint(x: 110, y: 265 - index * 65)
        CTLineDraw(CTLineCreateWithAttributedString(value), canvas)
    }
    try save(canvas.makeImage(), to: URL(fileURLWithPath: path))
}

func recognize(_ path: String) throws {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    try VNImageRequestHandler(url: URL(fileURLWithPath: path)).perform([request])
    print((request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n"))
}

do {
    let args = Array(CommandLine.arguments.dropFirst())
    switch (args.first, args.count) {
    case ("raster", 3):
        let pdf = try document(args[1])
        guard let page = pdf.page(at: 0) else { throw FixtureError(message: "PDF 没有可用页面。") }
        try save(raster(page, scale: 2), to: URL(fileURLWithPath: args[2]))
    case ("scan-copy", 3):
        try scannedCopy(args[1], args[2])
    case ("bilingual", 2):
        try bilingual(args[1])
    case ("vision", 2):
        try recognize(args[1])
    default:
        throw FixtureError(message: "用法 raster PDF PNG | scan-copy PDF PDF | bilingual PNG | vision PNG")
    }
} catch {
    let message = (error as? FixtureError)?.message ?? error.localizedDescription
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}
