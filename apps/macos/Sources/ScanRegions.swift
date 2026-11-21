import CoreGraphics
import Foundation
import Vision

enum ScanRegions {
    static func detect(_ image: Data) async throws -> [[String: Any]] {
        guard #available(macOS 26, *) else { throw Failure.unavailable }
        var request = RecognizeDocumentsRequest()
        request.textRecognitionOptions.automaticallyDetectLanguage = true
        let documents = try await request.perform(on: image)
        let frame = CGRect(x: 0, y: 0, width: 1, height: 1)
        let regions = documents.flatMap { observation -> [(CGRect, String)] in
            let document = observation.document
            let tables = document.tables.map { $0.boundingRegion.normalizedPath.boundingBox }
            let paragraphs = document.paragraphs.compactMap { paragraph -> CGRect? in
                let bounds = paragraph.boundingRegion.normalizedPath.boundingBox
                guard !tables.contains(where: { $0.contains(CGPoint(x: bounds.midX, y: bounds.midY)) }) else {
                    return nil
                }
                return bounds
            }
            // Keep Vision's paragraph order, inserting each table above the next paragraph on that column.
            var ordered = paragraphs.map { ($0, "text") }
            for table in tables {
                let index = ordered.firstIndex { bounds, _ in
                    bounds.midY < table.midY && bounds.maxX > table.minX && bounds.minX < table.maxX
                } ?? ordered.endIndex
                ordered.insert((table, "table"), at: index)
            }
            return ordered
        }
        return regions.prefix(80).compactMap { bounds, mode in
            // Vision measures from the lower left. The page preview measures from the upper left.
            let rect = bounds.insetBy(dx: -0.007, dy: -0.007).intersection(frame)
            guard !rect.isNull, rect.width >= 0.025, rect.height >= 0.006 else { return nil }
            return ["x": rect.minX, "y": 1 - rect.maxY, "width": rect.width, "height": rect.height, "mode": mode]
        }
    }

    private enum Failure: LocalizedError {
        case unavailable

        var errorDescription: String? { "自动找段落和表格需要 macOS 26。也可以手动拖出选区。" }
    }
}
