import AppKit
import CoreText

enum TranslationExport {
    static func render(format: String, blocks: [[String: String]]) throws -> Data {
        let content = NSMutableAttributedString()
        for block in blocks {
            guard let kind = block["kind"], let text = block["text"], !text.isEmpty else { continue }
            let style = NSMutableParagraphStyle()
            style.lineSpacing = 4
            style.paragraphSpacing = 9
            var font = NSFont.systemFont(ofSize: 12)
            switch kind {
            case "h1", "h2", "h3", "h4", "h5", "h6":
                let size: CGFloat = kind == "h1" ? 23 : kind == "h2" ? 17 : 14
                font = .boldSystemFont(ofSize: size)
                style.paragraphSpacingBefore = content.length == 0 ? 0 : 10
            case "li":
                style.firstLineHeadIndent = 8
                style.headIndent = 21
                style.paragraphSpacing = 4
            case "blockquote":
                style.firstLineHeadIndent = 16
                style.headIndent = 16
            case "code", "table":
                font = .monospacedSystemFont(ofSize: 10, weight: .regular)
            default:
                break
            }
            content.append(NSAttributedString(string: text + "\n", attributes: [
                .font: font, .paragraphStyle: style, .foregroundColor: NSColor.black,
            ]))
        }
        guard content.length > 0 else { throw Failure.empty }
        switch format {
        case "docx":
            return try content.data(from: NSRange(location: 0, length: content.length),
                                    documentAttributes: [.documentType: NSAttributedString.DocumentType.officeOpenXML])
        case "pdf":
            return try pdf(content)
        default:
            throw Failure.format
        }
    }

    private static func pdf(_ content: NSAttributedString) throws -> Data {
        let buffer = NSMutableData()
        var page = CGRect(x: 0, y: 0, width: 595.28, height: 841.89)
        guard let consumer = CGDataConsumer(data: buffer as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: &page, nil) else {
            throw Failure.layout
        }
        let path = CGPath(rect: page.insetBy(dx: 54, dy: 54), transform: nil)
        let framesetter = CTFramesetterCreateWithAttributedString(content)
        var offset = 0
        var pageNumber = 1
        while offset < content.length {
            let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: offset, length: 0), path, nil)
            let visible = CTFrameGetVisibleStringRange(frame)
            guard visible.location == offset, visible.length > 0 else { throw Failure.layout }
            context.beginPDFPage(nil)
            context.textMatrix = .identity
            CTFrameDraw(frame, context)
            let footer = NSAttributedString(string: String(pageNumber), attributes: [
                .font: NSFont.systemFont(ofSize: 9), .foregroundColor: NSColor.darkGray,
            ])
            let line = CTLineCreateWithAttributedString(footer)
            context.textPosition = CGPoint(x: page.midX - CTLineGetTypographicBounds(line, nil, nil, nil) / 2, y: 30)
            CTLineDraw(line, context)
            context.endPDFPage()
            offset += visible.length
            pageNumber += 1
        }
        context.closePDF()
        return buffer as Data
    }

    private enum Failure: LocalizedError {
        case empty, format, layout

        var errorDescription: String? {
            switch self {
            case .empty: return "没有可导出的译文。"
            case .format: return "不支持此导出格式。"
            case .layout: return "无法排版译文 PDF。"
            }
        }
    }
}
