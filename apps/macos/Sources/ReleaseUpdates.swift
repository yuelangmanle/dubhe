import CoreServices
import CryptoKit
import Foundation

enum ReleaseUpdates {
    private static let repository = "yuelangmanle/dubhe"
    private static let releases = "https://github.com/\(repository)/releases"
    static let page = URL(string: "\(releases)/latest")!
    private static let endpoint = "repos/\(repository)/releases/latest"

    struct Release: Decodable {
        struct Asset: Decodable {
            let name: String
            let state: String?
            let browser_download_url: String?
            let size: Int64?
            let digest: String?
        }

        let tag_name: String
        let body: String?
        let assets: [Asset]

        func update(than current: String) throws -> Update? {
            let version = String(tag_name.dropFirst())
            guard tag_name.hasPrefix("v"), let offered = ReleaseUpdates.version(version),
                  let installed = ReleaseUpdates.version(current) else { throw Failure.invalidResponse }
            guard installed.lexicographicallyPrecedes(offered) else { return nil }

            let name = "Dubhe-\(version)-macOS-arm64.dmg"
            let url = URL(string: "\(releases)/download/\(tag_name)/\(name)")!
            let package = assets.first { $0.name == name && $0.state == "uploaded" }.flatMap { asset -> Package? in
                guard asset.browser_download_url == url.absoluteString,
                      let size = asset.size, size > 0,
                      let digest = asset.digest?.lowercased(), digest.hasPrefix("sha256:") else { return nil }
                let hash = digest.dropFirst("sha256:".count)
                guard hash.utf8.count == 64, hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
                else { return nil }
                return Package(name: name, size: size, url: url, sha256: String(hash))
            }
            return Update(version: version, notes: body?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "",
                          page: URL(string: "\(releases)/tag/\(tag_name)")!, package: package)
        }
    }

    struct Update {
        let version: String
        let notes: String
        let page: URL
        let package: Package?
    }

    enum DownloadProgress: Sendable {
        case downloading(Int64)
        case verifying
    }

    struct Package {
        let name: String
        let size: Int64
        fileprivate let url: URL
        fileprivate let sha256: String

        func download(progress: @escaping @MainActor @Sendable (DownloadProgress) -> Void) async throws -> URL {
            try Task.checkCancellation()
            let files = FileManager.default
            let directory = files.temporaryDirectory.appendingPathComponent("Dubhe-Update-\(UUID())", isDirectory: true)
            try files.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
            var verified = false
            defer { if !verified { try? files.removeItem(at: directory) } }
            var file = directory.appendingPathComponent(name)

            let (events, continuation) = AsyncThrowingStream<Int64, Error>.makeStream(bufferingPolicy: .bufferingNewest(1))
            let delegate = DownloadDelegate(destination: file, events: continuation)
            let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
            defer { session.finishTasksAndInvalidate() }
            let task = session.downloadTask(with: URLRequest(url: url, timeoutInterval: 60))
            continuation.onTermination = { @Sendable termination in
                if case .cancelled = termination { task.cancel() }
            }
            task.resume()
            for try await bytes in events { await progress(.downloading(bytes)) }
            try Task.checkCancellation()

            await progress(.verifying)
            try verify(file)
            // Keep macOS's normal checks for files downloaded from the internet.
            var metadata = URLResourceValues()
            metadata.quarantineProperties = [kLSQuarantineAgentNameKey as String: "Dubhe",
                                             kLSQuarantineTypeKey as String: kLSQuarantineTypeWebDownload,
                                             kLSQuarantineDataURLKey as String: url]
            try file.setResourceValues(metadata)
            try Task.checkCancellation()
            verified = true
            return file
        }

        func verify(_ file: URL) throws {
            let input = try FileHandle(forReadingFrom: file)
            defer { try? input.close() }
            guard try input.seekToEnd() == UInt64(size) else { throw Failure.wrongSize }
            try input.seek(toOffset: 0)
            var hasher = SHA256()
            while let chunk = try input.read(upToCount: 1024 * 1024), !chunk.isEmpty {
                try Task.checkCancellation()
                hasher.update(data: chunk)
            }
            let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard actual == sha256 else { throw Failure.wrongDigest }
        }
    }

    private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
        let destination: URL
        let events: AsyncThrowingStream<Int64, Error>.Continuation

        init(destination: URL, events: AsyncThrowingStream<Int64, Error>.Continuation) {
            self.destination = destination
            self.events = events
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                        totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
            events.yield(totalBytesWritten)
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
            do {
                guard let http = downloadTask.response as? HTTPURLResponse,
                      http.statusCode == 200, http.url?.scheme == "https" else { throw Failure.downloadFailed }
                try FileManager.default.moveItem(at: location, to: destination)
            } catch {
                events.finish(throwing: error)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            events.finish(throwing: error)
        }
    }

    static func latest() async throws -> Release {
        var request = URLRequest(url: URL(string: "https://api.github.com/\(endpoint)")!, timeoutInterval: 15)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("Dubhe Desktop", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw Failure.invalidResponse }
        if http.statusCode == 404 { throw Failure.noRelease }
        if http.value(forHTTPHeaderField: "X-RateLimit-Remaining") == "0" && http.statusCode != 200 {
            throw Failure.rateLimited
        }
        guard http.statusCode == 200 else { throw Failure.invalidResponse }
        guard let release = try? JSONDecoder().decode(Release.self, from: data) else { throw Failure.invalidResponse }
        return release
    }

    private static func version(_ value: String) -> [Int]? {
        let fields = value.split(separator: ".", omittingEmptySubsequences: false)
        guard fields.count == 3, fields.allSatisfy({ !$0.isEmpty && $0.utf8.allSatisfy { (48...57).contains($0) } })
        else { return nil }
        let numbers = fields.compactMap { Int($0) }
        return numbers.count == 3 ? numbers : nil
    }

    private enum Failure: LocalizedError {
        case invalidResponse, noRelease, rateLimited, downloadFailed, wrongSize, wrongDigest

        var errorDescription: String? {
            switch self {
            case .invalidResponse: return "暂时无法读取 GitHub 发布信息。"
            case .noRelease: return "GitHub 上暂时没有可查询的正式版本。"
            case .rateLimited: return "GitHub 的查询次数用完了，稍后再试。"
            case .downloadFailed: return "暂时无法从 GitHub 下载安装包。"
            case .wrongSize: return "安装包大小与 GitHub 发布信息不符。请重新下载。"
            case .wrongDigest: return "安装包未通过校验。请重新下载。"
            }
        }
    }
}
