package dev.dubhe.sdk

import java.io.File

data class ModelInfo(
    val architecture: String,
    val backend: String,
    val quantization: String,
    val contextSize: Int,
    val contextCeiling: Int,
    val vocabularySize: Int,
    val supportsChatML: Boolean,
    val parameters: Long,
    val fileSize: Long,
)

data class Sampling(
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val repeatPenalty: Float = 1.05f,
    val repeatLastCount: Int = 64,
    val seed: Int = 0,
)

data class GenerationStats(
    val promptTokens: Int,
    val generatedTokens: Int,
    val reusedTokens: Int,
    val prefillMilliseconds: Double,
    val decodeMilliseconds: Double,
    val prefillTokensPerSecond: Double,
    val decodeTokensPerSecond: Double,
)

data class Generation(val text: String, val stats: GenerationStats, val cancelled: Boolean)

data class ProjectorCapabilities(val vision: Boolean, val audio: Boolean)

enum class AttachmentKind(val token: String, internal val nativeId: Int) {
    Image("<|image|>", 1),
    Video("<|video|>", 2),
    Audio("<|audio|>", 3),
}

data class Attachment(
    val file: File,
    val kind: AttachmentKind,
    val videoFramesPerSecond: Double = 2.0,
    val maxVideoFrames: Int = 4,
    val maxVideoSide: Int = 384,
)

class DubheException(val code: Int, message: String) : RuntimeException(message)
