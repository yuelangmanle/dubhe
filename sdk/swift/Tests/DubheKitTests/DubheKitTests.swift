import Foundation
import XCTest
@testable import DubheKit

final class DubheKitTests: XCTestCase {
    func testErrorsAndClose() async throws {
        let engine = try DubheEngine(useGPU: false)
        XCTAssertFalse(DubheEngine.version.isEmpty)
        do {
            _ = try await engine.loadModel(at: URL(fileURLWithPath: "/missing-dubhe-test-model.gguf"))
            XCTFail("A missing model must fail")
        } catch let error as DubheError {
            XCTAssertFalse(error.message.isEmpty)
        }

        engine.close()
        engine.close()
        do {
            _ = try await engine.tokenize("hello")
            XCTFail("A closed engine must reject work")
        } catch let error as DubheError {
            XCTAssertEqual(error.code, -6)
        }
    }

    func testUTF8PiecesAreStreamedWithoutMojibake() {
        var stream = UTF8Stream()
        XCTAssertEqual(stream.append([0xE4]), "")
        XCTAssertEqual(stream.append([0xB8, 0xAD, 0x61, 0xF0, 0x9F]), "中a")
        XCTAssertEqual(stream.append([0x98, 0x8A]), "😊")
        XCTAssertEqual(stream.finish(), "")
    }

    func testModelGenerationCancellationAndReuse() async throws {
        guard let path = ProcessInfo.processInfo.environment["DUBHE_TEST_MODEL"] else {
            throw XCTSkip("Set DUBHE_TEST_MODEL to a small ChatML GGUF.")
        }
        let engine = try DubheEngine(contextSize: 256, useGPU: false)
        defer { engine.close() }
        let model = try await engine.loadModel(at: URL(fileURLWithPath: path))
        XCTAssertGreaterThan(model.vocabularySize, 0)
        let prompt = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n"
        let tokens = try await engine.tokenize(prompt)
        XCTAssertFalse(tokens.isEmpty)

        let cancelled = try await engine.generate(prompt, sampling: DubheSampling(maxTokens: 64)) { _ in
            engine.cancelGeneration()
        }
        XCTAssertTrue(cancelled.cancelled)

        let next = try await engine.generate(prompt, sampling: DubheSampling(maxTokens: 4))
        XCTAssertFalse(next.cancelled)
        XCTAssertFalse(next.text.isEmpty)
        XCTAssertGreaterThan(next.stats.generatedTokens, 0)
        try await engine.unloadModel()
    }
}
