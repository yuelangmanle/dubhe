import Foundation

@main
enum ReleaseUpdatesTests {
    private static let macAsset = "Dubhe-0.32.1-macOS-arm64.dmg"
    private static let downloadURL = "https://github.com/yuelangmanle/dubhe/releases/download/v0.32.1/\(macAsset)"
    private static let checksum = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    static func main() async throws {
        let release = try fixture()
        let current = try release.update(than: "0.32.1")
        let ahead = try release.update(than: "0.33.0")
        precondition(current == nil && ahead == nil, "Only offer strictly newer releases")

        let update = try release.update(than: "0.9.20")!
        precondition(update.version == "0.32.1" && update.notes == "- 修复扫描问题", "Keep version and release notes")
        precondition(update.page.absoluteString == "https://github.com/yuelangmanle/dubhe/releases/tag/v0.32.1")
        let package = update.package!
        precondition(package.name == macAsset && package.size == 3, "Select the exact macOS asset")

        for unusable in [try fixture(name: "Dubhe-0.32.1-Android-arm64.apk"),
                         try fixture(state: "starter"),
                         try fixture(url: "https://example.com/\(macAsset)"),
                         try fixture(size: 0),
                         try fixture(digest: nil),
                         try fixture(digest: "sha256:not-a-digest")] {
            let newer = try unusable.update(than: "0.32.0")!
            precondition(newer.package == nil && newer.notes == update.notes,
                         "Keep release details and disable in-app downloads when the asset cannot be verified")
        }

        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("abc".utf8).write(to: file)
        try package.verify(file)
        for (contents, message) in [("ab", "大小"), ("abd", "校验")] {
            try Data(contents.utf8).write(to: file)
            do {
                try package.verify(file)
                fatalError("A truncated or modified installer must be rejected")
            } catch {
                precondition(error.localizedDescription.contains(message))
            }
        }
        print("macOS update version, package selection, and integrity checks passed")

        if CommandLine.arguments.contains("--live") { try await checkLiveDownload() }
    }

    private static func fixture(name: String = macAsset, state: String = "uploaded",
                                url: String = downloadURL, size: Int64 = 3,
                                digest: String? = checksum) throws -> ReleaseUpdates.Release {
        var asset: [String: Any] = ["name": name, "state": state, "browser_download_url": url, "size": size]
        asset["digest"] = digest
        let data = try JSONSerialization.data(withJSONObject: [
            "tag_name": "v0.32.1", "body": "\n- 修复扫描问题\n\n", "assets": [asset],
        ])
        return try JSONDecoder().decode(ReleaseUpdates.Release.self, from: data)
    }

    @MainActor
    private static func checkLiveDownload() async throws {
        let release = try await ReleaseUpdates.latest()
        let update = try release.update(than: "0.0.0")!
        let package = update.package!
        var receivedBytes: Int64 = 0
        var verifying = false
        let file = try await package.download { progress in
            switch progress {
            case let .downloading(bytes): receivedBytes = max(receivedBytes, bytes)
            case .verifying: verifying = true
            }
        }
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        precondition(receivedBytes > 0 && verifying, "The real download reports progress and verification")
        precondition(file.lastPathComponent == package.name)
        let metadata = try file.resourceValues(forKeys: [.quarantinePropertiesKey])
        precondition(metadata.quarantineProperties != nil, "macOS can identify the installer as a download")
        try package.verify(file)
        print("Downloaded and verified GitHub release \(update.version)")

        let cancelled = Task {
            try await package.download { progress in
                if case .downloading = progress { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        do {
            let unexpectedFile = try await cancelled.value
            try? FileManager.default.removeItem(at: unexpectedFile.deletingLastPathComponent())
            fatalError("A cancelled download must not produce an installer")
        } catch {
            guard error is CancellationError || (error as? URLError)?.code == .cancelled else { throw error }
        }
        print("Cancelled a live macOS update download")
    }
}
