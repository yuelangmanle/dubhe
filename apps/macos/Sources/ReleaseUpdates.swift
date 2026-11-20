import Foundation

enum ReleaseUpdates {
    private static let repository = "yuelangmanle/dubhe"
    static let page = URL(string: "https://github.com/\(repository)/releases/latest")!
    private static let endpoint = "repos/\(repository)/releases/latest"

    struct Release: Decodable {
        struct Asset: Decodable { let name: String }
        let tag_name: String
        let assets: [Asset]

        func newerVersion(than current: String) throws -> String? {
            guard tag_name.hasPrefix("v"),
                  let offered = ReleaseUpdates.version(String(tag_name.dropFirst())),
                  let installed = ReleaseUpdates.version(current) else { throw Failure.invalidResponse }
            guard installed.lexicographicallyPrecedes(offered) else { return nil }
            let version = String(tag_name.dropFirst())
            guard assets.contains(where: { $0.name == "Dubhe-\(version)-macOS-arm64.dmg" }) else {
                throw Failure.missingPackage
            }
            return version
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
        return try JSONDecoder().decode(Release.self, from: data)
    }

    private static func version(_ value: String) -> [Int]? {
        let fields = value.split(separator: ".", omittingEmptySubsequences: false)
        let numbers = fields.compactMap { Int($0) }
        return fields.count == 3 && numbers.count == 3 && numbers.allSatisfy { $0 >= 0 } ? numbers : nil
    }

    private enum Failure: LocalizedError {
        case invalidResponse, noRelease, rateLimited, missingPackage

        var errorDescription: String? {
            switch self {
            case .invalidResponse: return "暂时无法读取 GitHub 发布信息。"
            case .noRelease: return "GitHub 上暂时没有可查询的正式版本。"
            case .rateLimited: return "GitHub 的查询次数用完了，稍后再试。"
            case .missingPackage: return "这个版本还未上传 macOS 安装包。"
            }
        }
    }
}
