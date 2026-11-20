package dev.dubhe.sdk

import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.suspendCancellableCoroutine

/** An engine owns one model and runs all model operations on a dedicated worker. */
class DubheEngine(contextSize: Int = 2048, useGPU: Boolean = false) : Closeable {
    private val lock = Any()
    private val handle = DubheNative.create(contextSize, useGPU)
    private val worker = Executors.newSingleThreadExecutor { task -> Thread(task, "Dubhe inference") }
    private val waiting = ArrayDeque<AtomicBoolean>()
    private var active: AtomicBoolean? = null
    private var closed = false

    val version: String get() = engineVersion
    val hardwareInfo: String get() = DubheNative.hardwareInfo()

    suspend fun loadModel(file: File): ModelInfo = submit { DubheNative.loadModel(handle, file.absolutePath.utf8()) }
    suspend fun unloadModel() = submit { DubheNative.unloadModel(handle) }

    /** Loads a projector GGUF paired with the current language model. */
    suspend fun loadProjector(file: File): ProjectorCapabilities = submit {
        val flags = DubheNative.loadProjector(handle, file.absolutePath.utf8())
        ProjectorCapabilities(vision = flags and 1 != 0, audio = flags and 2 != 0)
    }

    suspend fun tokenize(text: String, addSpecial: Boolean = true): IntArray =
        submit { DubheNative.tokenize(handle, text.utf8(), addSpecial) }

    /** The callback runs on the engine worker. Its text is always complete UTF-8. */
    suspend fun generate(
        prompt: String,
        sampling: Sampling = Sampling(),
        addSpecial: Boolean = true,
        attachment: Attachment? = null,
        onText: (String) -> Unit = {},
    ): Generation = generateWith(sampling, attachment, onText) {
        DubheNative.tokenize(handle, prompt.utf8(), addSpecial)
    }

    suspend fun generate(
        tokens: IntArray,
        sampling: Sampling = Sampling(),
        attachment: Attachment? = null,
        onText: (String) -> Unit = {},
    ): Generation = generateWith(sampling, attachment, onText) { tokens.copyOf() }

    /** Requests a stop for the running generation, or the next one waiting for the worker. */
    fun cancelGeneration() {
        synchronized(lock) {
            val stop = active ?: waiting.firstOrNull() ?: return
            requestStop(stop)
        }
    }

    /** Rejects new work and releases native resources when the current call returns. */
    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            active?.let(::requestStop)
            worker.execute { DubheNative.destroy(handle) }
            worker.shutdown()
        }
    }

    private suspend fun generateWith(
        sampling: Sampling,
        attachment: Attachment?,
        onText: (String) -> Unit,
        tokenize: () -> IntArray,
    ): Generation {
        val stop = AtomicBoolean()
        return submit(stop) {
            val callbacks = NativeGenerationCallbacks(stop, onText)
            if (stop.get()) return@submit Generation("", emptyStats, cancelled = true)
            val stats = DubheNative.generate(
                handle, tokenize(), sampling.maxTokens, sampling.temperature, sampling.topK,
                sampling.topP, sampling.repeatPenalty, sampling.repeatLastCount, sampling.seed,
                attachment?.file?.absolutePath?.utf8(), attachment?.kind?.nativeId ?: 0,
                attachment?.videoFramesPerSecond ?: 0.0, attachment?.maxVideoFrames ?: 0,
                attachment?.maxVideoSide ?: 0, callbacks,
            )
            Generation(callbacks.finish(), stats, stop.get())
        }
    }

    private suspend fun <T> submit(stop: AtomicBoolean? = null, operation: () -> T): T =
        suspendCancellableCoroutine { continuation ->
            stop?.let { continuation.invokeOnCancellation { requestStop(stop) } }
            synchronized(lock) {
                if (closed) {
                    continuation.resumeWith(Result.failure(DubheException(-6, "The engine is closed.")))
                    return@suspendCancellableCoroutine
                }
                stop?.let(waiting::addLast)
                worker.execute {
                    val result = runCatching {
                        synchronized(lock) {
                            stop?.let(waiting::remove)
                            if (closed) throw DubheException(-6, "The engine is closed.")
                            active = stop
                        }
                        try {
                            if (!continuation.isActive) return@execute
                            operation()
                        } finally {
                            synchronized(lock) { active = null }
                        }
                    }
                    continuation.resumeWith(result)
                }
            }
        }

    private fun requestStop(stop: AtomicBoolean) = synchronized(lock) {
        stop.set(true)
        if (active === stop) DubheNative.abort(handle)
    }

    private companion object {
        val engineVersion by lazy(DubheNative::version)
    }
}

private fun String.utf8(): ByteArray {
    require('\u0000' !in this) { "Text contains a NUL character." }
    return toByteArray(Charsets.UTF_8)
}

private val emptyStats = GenerationStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
