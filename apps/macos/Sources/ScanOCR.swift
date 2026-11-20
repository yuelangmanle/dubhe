import CoreGraphics
import Foundation

enum ScanOCR {
    static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        return URLSession(configuration: configuration, delegate: NoRedirects(), delegateQueue: nil)
    }()

    static func request(port: UInt16, image: CGImage) throws -> URLRequest {
        let origin = "http://127.0.0.1:\(port)"
        guard port != 0, let url = URL(string: origin + "/api/scan") else {
            throw Failure("本地扫描服务还没就绪。")
        }
        let source = try ScanImage.dataURL(image)
        var request = URLRequest(url: url, timeoutInterval: 120)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(origin, forHTTPHeaderField: "Origin")
        request.httpBody = try JSONEncoder().encode(["mode": "text", "image": source])
        return request
    }

    static func recognizedText(_ data: Data?, response: URLResponse?, error: Error?) throws -> String {
        if let error {
            if (error as? URLError)?.code == .timedOut { throw Failure("等待小米 OCR 超时。") }
            throw Failure("无法连接 Dubhe 的本地扫描服务。请确认本地引擎已经启动。")
        }
        guard let http = response as? HTTPURLResponse else { throw Failure("本地扫描服务没有返回结果。") }
        let result = data.flatMap { try? JSONDecoder().decode(Response.self, from: $0) }
        guard http.statusCode == 200 else {
            throw Failure(result?.error ?? "小米 OCR 暂不可用。可到「扫描」页检查模型安装状态。")
        }
        guard let text = result?.text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            throw Failure("这张截图中没有识别到可翻译的文字。")
        }
        return text
    }

    private struct Response: Decodable {
        var text: String?
        var error: String?
    }

    private struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// A redirect must not move the screenshot outside the local engine.
    private final class NoRedirects: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            completionHandler(nil)
        }
    }
}
