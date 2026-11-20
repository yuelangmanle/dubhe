package dev.dubhe.sdk

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CodingErrorAction
import java.util.concurrent.atomic.AtomicBoolean

internal object DubheNative {
    init {
        System.loadLibrary("dubhe_jni")
    }

    external fun version(): String
    external fun hardwareInfo(): String
    external fun create(contextSize: Int, useGPU: Boolean): Long
    external fun destroy(handle: Long)
    external fun loadModel(handle: Long, path: ByteArray): ModelInfo
    external fun unloadModel(handle: Long)
    external fun loadProjector(handle: Long, path: ByteArray): Int
    external fun tokenize(handle: Long, text: ByteArray, addSpecial: Boolean): IntArray
    external fun abort(handle: Long)
    external fun generate(
        handle: Long,
        tokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repeatPenalty: Float,
        repeatLastCount: Int,
        seed: Int,
        attachmentPath: ByteArray?,
        attachmentKind: Int,
        videoFramesPerSecond: Double,
        maxVideoFrames: Int,
        maxVideoSide: Int,
        callbacks: NativeGenerationCallbacks,
    ): GenerationStats
}

internal class NativeGenerationCallbacks(
    private val stop: AtomicBoolean,
    private val onText: (String) -> Unit,
) {
    private val decoder = Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)
    private val text = StringBuilder()
    private var remaining = byteArrayOf()

    fun shouldStop(): Boolean = stop.get()

    fun onBytes(bytes: ByteArray) {
        if (stop.get()) return
        val input = ByteBuffer.wrap(remaining + bytes)
        val output = CharBuffer.allocate(input.remaining())
        decoder.decode(input, output, false)
        remaining = ByteArray(input.remaining()).also { input.get(it) }
        output.flip()
        publish(output.toString())
    }

    fun finish(): String {
        if (!stop.get()) publish(remaining.toString(Charsets.UTF_8))
        return text.toString()
    }

    private fun publish(piece: String) {
        if (piece.isEmpty()) return
        text.append(piece)
        onText(piece)
    }
}
