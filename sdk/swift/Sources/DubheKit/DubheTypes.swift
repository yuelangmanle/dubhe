import CDubhe
import Foundation

public struct DubheError: Error, Sendable, LocalizedError {
    public let code: Int32
    public let message: String
    public var errorDescription: String? { message }

    static func native(_ code: Int32, fallback: String) -> DubheError {
        let detail = dubhe_last_error().map { String(cString: $0) } ?? ""
        return DubheError(code: code, message: detail.isEmpty ? fallback : detail)
    }
}

public struct DubheModelInfo: Sendable {
    public let architecture: String
    public let quantization: String
    public let backend: String
    public let vocabularySize: Int
    public let contextSize: Int
    public let contextCeiling: Int
    public let parameters: UInt64
    public let fileSize: UInt64
    public let usesGPU: Bool
    public let supportsChatML: Bool

    init(_ info: dubhe_model_info) {
        architecture = readCString(info.arch)
        quantization = readCString(info.quant_desc)
        backend = readCString(info.backend)
        vocabularySize = Int(info.n_vocab)
        contextSize = Int(info.n_ctx_limit)
        contextCeiling = Int(info.n_ctx_ceiling)
        parameters = info.n_params
        fileSize = info.file_size
        usesGPU = info.is_gpu != 0
        supportsChatML = info.has_chat_template != 0
    }
}

public struct DubheSampling: Sendable {
    public var maxTokens: Int32
    public var temperature: Float
    public var topK: Int32
    public var topP: Float
    public var repeatPenalty: Float
    public var repeatLastCount: Int32
    public var seed: Int32

    public init(
        maxTokens: Int32 = 256, temperature: Float = 0.7, topK: Int32 = 40,
        topP: Float = 0.95, repeatPenalty: Float = 1.05, repeatLastCount: Int32 = 64,
        seed: Int32 = 0
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repeatPenalty = repeatPenalty
        self.repeatLastCount = repeatLastCount
        self.seed = seed
    }

    var native: dubhe_sampling {
        dubhe_sampling(max_tokens: maxTokens, temperature: temperature, top_k: topK, top_p: topP,
                       repeat_penalty: repeatPenalty, repeat_last_n: repeatLastCount, seed: seed)
    }
}

public struct DubheGenerationStats: Sendable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let reusedTokens: Int
    public let prefillMilliseconds: Double
    public let decodeMilliseconds: Double
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double

    init(_ stats: dubhe_stats) {
        promptTokens = Int(stats.n_prompt)
        generatedTokens = Int(stats.n_generated)
        reusedTokens = Int(stats.n_reused)
        prefillMilliseconds = stats.t_prompt_ms
        decodeMilliseconds = stats.t_gen_ms
        prefillTokensPerSecond = stats.prefill_tps
        decodeTokensPerSecond = stats.decode_tps
    }
}

public struct DubheGeneration: Sendable {
    public let text: String
    public let stats: DubheGenerationStats
    public let cancelled: Bool
}

private func readCString<T>(_ value: T) -> String {
    withUnsafeBytes(of: value) { bytes in
        String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
}
