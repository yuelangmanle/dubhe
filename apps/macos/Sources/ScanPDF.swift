import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers

/// Keep the document on disk and render one page at a time on the scanner queue.
final class ScanPDF {
    private let document: PDFDocument
    let pageCount: Int

    init(url: URL) throws {
        guard let document = PDFDocument(url: url) else { throw ScanError.invalidPDF }
        guard !document.isLocked else { throw ScanError.lockedPDF }
        guard document.pageCount > 0 else { throw ScanError.invalidPDF }
        self.document = document
        pageCount = document.pageCount
    }

    func pageDataURL(number: Int) throws -> String {
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
        let w = max(1, Int((width / span * ScanImage.maxSide).rounded()))
        let h = max(1, Int((height / span * ScanImage.maxSide).rounded()))
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
        return try ScanImage.encodeJPEG(image)
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
    case invalidPDF, lockedPDF, invalidPage(Int), image

    var errorDescription: String? {
        switch self {
        case .invalidPDF: return "无法读取 PDF，文件可能损坏或没有页面。"
        case .lockedPDF: return "此 PDF 需要密码。请先解锁，再重新打开。"
        case .invalidPage(let number): return "无法读取 PDF 第 \(number) 页。"
        case .image: return "无法准备扫描图片。请重试，或缩小选取范围。"
        }
    }
}
