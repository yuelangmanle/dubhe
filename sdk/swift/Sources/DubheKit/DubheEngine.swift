import CDubhe
import Foundation

public final class DubheEngine: @unchecked Sendable {
    private let queue = DispatchQueue(label: "dev.dubhe.sdk.engine", qos: .userInitiated)
    private let lock = NSLock()
    private var handle: OpaquePointer?
    private var closed = false
    private var active: StopSignal?
    private var modelLoaded = false // Accessed only on the worker queue.

    public static let version = String(cString: dubhe_version())
    public static let hardwareInfo = String(cString: dubhe_hardware_info())

    public init(contextSize: Int32 = 4096, useGPU: Bool = true) throws {
        guard contextSize > 0 else {
            throw DubheError(code: -1, message: "Context size must be positive.")
        }
        var config = dubhe_config(n_threads: 0, n_ctx: contextSize, use_gpu: useGPU ? 1 : 0, n_batch: 0)
        guard let handle = dubhe_create(&config) else {
            throw DubheError(code: -5, message: "Could not create a Dubhe engine.")
        }
        self.handle = handle
    }

    deinit {
        if let handle { dubhe_destroy(handle) }
    }

    /// Signals the currently running generation. This method is safe to call from a token callback.
    public func cancelGeneration() {
        lock.withLock {
            guard let active, let handle else { return }
            active.stop()
            dubhe_request_abort(handle)
        }
    }

    /// Rejects new work immediately and releases the model after in-progress native calls return.
    public func close() {
        let shouldClose = lock.withLock {
            guard !closed else { return false }
            closed = true
            if let active, let handle {
                active.stop()
                dubhe_request_abort(handle)
            }
            return true
        }
        guard shouldClose else { return }
        queue.async {
            let handle = self.lock.withLock {
                let handle = self.handle
                self.handle = nil
                return handle
            }
            if let handle { dubhe_destroy(handle) }
        }
    }

    @discardableResult
    public func loadModel(at url: URL) async throws -> DubheModelInfo {
        guard url.isFileURL else {
            throw DubheError(code: -1, message: "Model URL must be a local file.")
        }
        return try await perform { handle in
            self.modelLoaded = false
            dubhe_unload_mmproj(handle)
            let status = url.path.withCString { dubhe_load_model(handle, $0) }
            guard status == 0 else {
                let error = DubheError.native(status, fallback: "Could not load model.")
                dubhe_unload_model(handle)
                throw error
            }
            self.modelLoaded = true
            return try readModelInfo(handle)
        }
    }

    public func unloadModel() async throws {
        try await perform { handle in
            dubhe_unload_mmproj(handle)
            dubhe_unload_model(handle)
            self.modelLoaded = false
        }
    }

    public func modelInfo() async throws -> DubheModelInfo {
        try await perform { handle in
            try self.checkModelLoaded()
            return try readModelInfo(handle)
        }
    }

    public func tokenize(_ text: String, addSpecial: Bool = true) async throws -> [Int32] {
        try await perform { handle in
            try self.checkModelLoaded()
            return try tokenizeText(handle, text, addSpecial: addSpecial)
        }
    }

    /// Accepts a rendered prompt. The optional callback runs serially on the engine queue.
    public func generate(
        _ prompt: String, sampling: DubheSampling = DubheSampling(),
        onText: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> DubheGeneration {
        let stop = StopSignal()
        let result = try await withTaskCancellationHandler {
            try await perform { handle in
                try self.checkModelLoaded()
                try self.beginGeneration(stop)
                defer { self.lock.withLock { self.active = nil } }
                let tokens = try tokenizeText(handle, prompt, addSpecial: true)
                return try generateTokens(handle, tokens, sampling: sampling, stop: stop, onText: onText)
            }
        } onCancel: {
            stop.stop()
        }
        try Task.checkCancellation()
        return result
    }

    private func perform<T: Sendable>(
        _ operation: @escaping @Sendable (OpaquePointer) throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result {
                    let handle: OpaquePointer = try self.lock.withLock {
                        guard !self.closed, let handle = self.handle else {
                            throw DubheError(code: -6, message: "The engine is closed.")
                        }
                        return handle
                    }
                    return try operation(handle)
                })
            }
        }
    }

    private func checkModelLoaded() throws {
        guard modelLoaded else { throw DubheError(code: -6, message: "Load a model first.") }
    }

    private func beginGeneration(_ stop: StopSignal) throws {
        try lock.withLock {
            guard !closed else { throw DubheError(code: -6, message: "The engine is closed.") }
            active = stop
        }
    }
}

private func readModelInfo(_ handle: OpaquePointer) throws -> DubheModelInfo {
    var info = dubhe_model_info()
    let status = dubhe_model_info_get(handle, &info)
    guard status == 0 else { throw DubheError.native(status, fallback: "Model information is unavailable.") }
    return DubheModelInfo(info)
}

private func tokenizeText(_ handle: OpaquePointer, _ text: String, addSpecial: Bool) throws -> [Int32] {
    guard !text.utf8.contains(0) else {
        throw DubheError(code: -1, message: "Prompt contains a NUL character.")
    }
    return try text.withCString { value in
        let count = dubhe_tokenize(handle, value, addSpecial ? 1 : 0, nil, 0)
        guard count >= 0 else { throw DubheError(code: count, message: "Could not tokenize the prompt.") }
        var tokens = [Int32](repeating: 0, count: Int(count))
        guard dubhe_tokenize(handle, value, addSpecial ? 1 : 0, &tokens, count) == count else {
            throw DubheError(code: -6, message: "Token count changed unexpectedly.")
        }
        return tokens
    }
}

private func generateTokens(
    _ handle: OpaquePointer,
    _ tokens: [Int32],
    sampling: DubheSampling,
    stop: StopSignal,
    onText: @escaping @Sendable (String) -> Void
) throws -> DubheGeneration {
    guard !tokens.isEmpty else { throw DubheError(code: -1, message: "Prompt is empty.") }
    var stats = dubhe_stats()
    var settings = sampling.native
    let response = GenerationOutput(stop: stop, onText: onText)
    if !stop.isStopped {
        let status = dubhe_generate(handle, tokens, Int32(tokens.count), &settings, { opaque, _, piece in
            guard let opaque, let piece else { return }
            let response = Unmanaged<GenerationOutput>.fromOpaque(opaque).takeUnretainedValue()
            response.append(piece)
        }, { opaque in
            guard let opaque else { return 1 }
            let response = Unmanaged<GenerationOutput>.fromOpaque(opaque).takeUnretainedValue()
            return response.stop.isStopped ? 1 : 0
        }, Unmanaged.passUnretained(response).toOpaque(), &stats)
        guard status == 0 || status == -7 else {
            throw DubheError.native(status, fallback: "Could not generate a response.")
        }
        if status == -7 { stop.stop() }
    }
    response.finish()
    return DubheGeneration(text: response.text, stats: DubheGenerationStats(stats), cancelled: stop.isStopped)
}

private final class StopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false

    var isStopped: Bool { lock.withLock { stopped } }
    func stop() { lock.withLock { stopped = true } }
}

private final class GenerationOutput {
    let stop: StopSignal
    var text = ""
    private var stream = UTF8Stream()
    private let onText: (String) -> Void

    init(stop: StopSignal, onText: @escaping (String) -> Void) {
        self.stop = stop
        self.onText = onText
    }

    func append(_ piece: UnsafePointer<CChar>) {
        guard !stop.isStopped else { return }
        let bytes = UnsafeRawBufferPointer(start: piece, count: strlen(piece))
        publish(stream.append(bytes))
    }

    func finish() {
        if !stop.isStopped { publish(stream.finish()) }
    }

    private func publish(_ value: String) {
        guard !value.isEmpty else { return }
        text += value
        onText(value)
    }
}
