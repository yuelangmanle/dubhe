package dev.dubhe.mobile

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.PowerManager
import android.os.SystemClock
import dev.dubhe.sdk.Attachment
import dev.dubhe.sdk.AttachmentKind
import dev.dubhe.sdk.DubheEngine
import dev.dubhe.sdk.ModelInfo
import dev.dubhe.sdk.ProjectorCapabilities
import dev.dubhe.sdk.Sampling
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/** The non-exported service owns the model in the :engine process. */
class EngineService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val instance = lazy { DubheEngine(useGPU = BuildConfig.DUBHE_VULKAN) }
    private val engine by instance
    private val endpoint by lazy { Messenger(Handler(Looper.getMainLooper(), ::receive)) }
    private var job: Job? = null
    private var operationId = 0
    private var stopRequested = false
    private var thermalStop = false
    private var overheated = false
    private var generationReply: Messenger? = null
    private var model: ModelInfo? = null
    private var modelFile: File? = null
    private var projector: ProjectorCapabilities? = null
    private var projectorFile: File? = null
    private val power by lazy { requireNotNull(getSystemService(PowerManager::class.java)) }
    private val thermalListener = PowerManager.OnThermalStatusChangedListener { status ->
        overheated = status >= PowerManager.THERMAL_STATUS_SEVERE
        val reply = generationReply
        if (overheated && reply != null && !thermalStop) {
            thermalStop = true
            stop()
            reply.post(EngineProtocol.StoppingForHeat, operationId, Bundle())
        }
    }

    override fun onCreate() {
        super.onCreate()
        overheated = power.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE
        power.addThermalStatusListener(mainExecutor, thermalListener)
    }

    override fun onBind(intent: Intent) = endpoint.binder

    override fun onDestroy() {
        power.removeThermalStatusListener(thermalListener)
        scope.cancel()
        if (instance.isInitialized()) engine.close()
        super.onDestroy()
    }

    private fun receive(message: Message): Boolean {
        val reply = message.replyTo ?: return true
        val id = message.arg1
        if (message.what == EngineProtocol.Stop) {
            if (operationId == id) stop()
            return true
        }
        if (job?.isActive == true) {
            reply.failed(id, "推理引擎正忙，请稍候。")
            return true
        }
        operationId = id
        stopRequested = false
        thermalStop = false
        val data = Bundle(message.data)
        val action = message.what
        job = scope.launch {
            try {
                when (action) {
                    EngineProtocol.LoadModel -> loadModel(File(requireNotNull(data.getString("path"))), id, reply)
                    EngineProtocol.LoadProjector -> loadProjector(File(requireNotNull(data.getString("path"))), id, reply)
                    EngineProtocol.Generate -> {
                        check(!overheated) { "设备温度过高，降温后再试。" }
                        generationReply = reply
                        generate(data, id, reply)
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Exception) {
                reply.failed(id, failure.localizedMessage ?: "推理引擎出错。")
            } catch (_: LinkageError) {
                reply.failed(id, "无法启动推理引擎。请检查手机是否支持 arm64。")
            } finally {
                generationReply = null
            }
        }
        return true
    }

    private fun stop() {
        stopRequested = true
        if (instance.isInitialized()) engine.cancelGeneration()
    }

    private suspend fun loadModel(file: File, id: Int, reply: Messenger) {
        val previous = modelFile
        val previousProjector = projectorFile
        model = null
        modelFile = null
        projector = null
        projectorFile = null
        val loaded = try {
            engine.loadModel(file)
        } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            if (previous != null && previous != file) {
                try {
                    model = engine.loadModel(previous)
                    modelFile = previous
                    restoreProjector(previousProjector)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (_: Exception) {
                    // Report the original failure and leave the engine unloaded.
                }
            }
            throw failure
        }
        model = loaded
        modelFile = file
        reply.post(EngineProtocol.ModelReady, id, Bundle().apply {
            putString("backend", loaded.backend)
            putString("quantization", loaded.quantization)
            putInt("context", loaded.contextSize)
        })
    }

    private suspend fun loadProjector(file: File, id: Int, reply: Messenger) {
        val previous = projectorFile
        projectorFile = null
        projector = null
        val capabilities = try {
            engine.loadProjector(file)
        } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            if (previous != file) restoreProjector(previous)
            throw failure
        }
        projectorFile = file
        projector = capabilities
        reply.post(EngineProtocol.ProjectorReady, id, Bundle().apply {
            putBoolean("vision", capabilities.vision)
            putBoolean("audio", capabilities.audio)
        })
    }

    private suspend fun restoreProjector(file: File?) {
        if (file == null) return
        try {
            projector = engine.loadProjector(file)
            projectorFile = file
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: Exception) {
            // The language model remains usable.
        }
    }

    private suspend fun generate(data: Bundle, id: Int, reply: Messenger) {
        val info = model ?: error("请先导入模型。")
        val allMessages = requireNotNull(data.getStringArrayList("messages"))
        val attachment = data.getString("attachmentPath")?.let { path ->
            Attachment(File(path), AttachmentKind.valueOf(requireNotNull(data.getString("attachmentKind"))))
        }
        val maxTokens = minOf(256, maxOf(1, info.contextSize / 4))
        var messages = allMessages.takeLast(13)
        val tokens: IntArray
        while (true) {
            val candidate = engine.tokenize(renderPrompt(info, messages, attachment?.kind))
            if (stopRequested) {
                reply.post(EngineProtocol.Complete, id, Bundle().apply {
                    putBoolean("cancelled", true)
                    putBoolean("thermal", thermalStop)
                })
                return
            }
            if (candidate.size + maxTokens <= info.contextSize) {
                tokens = candidate
                break
            }
            if (messages.size <= 1) error("这条消息太长了，请缩短后再发。")
            messages = messages.drop(2)
        }
        var lastUpdate = 0L
        val chunk = StringBuilder()
        val response = engine.generate(tokens, Sampling(maxTokens = maxTokens), attachment) { text ->
            chunk.append(text)
            val now = SystemClock.elapsedRealtime()
            if (chunk.length >= 512 || now - lastUpdate >= 65L) {
                reply.post(EngineProtocol.Text, id, Bundle().apply { putString("text", chunk.toString()) })
                chunk.setLength(0)
                lastUpdate = now
            }
        }
        reply.post(EngineProtocol.Complete, id, Bundle().apply {
            putString("text", response.text)
            putBoolean("cancelled", response.cancelled)
            putBoolean("thermal", response.cancelled && thermalStop)
            putDouble("speed", response.stats.decodeTokensPerSecond)
        })
    }

    private fun Messenger.failed(id: Int, detail: String) {
        post(EngineProtocol.Failed, id, Bundle().apply {
            putString("detail", detail)
            putBoolean("modelReady", model != null)
            putBoolean("projectorReady", projector != null)
        })
    }
}

private fun renderPrompt(model: ModelInfo, messages: List<String>, kind: AttachmentKind?): String = buildString {
    if (model.supportsChatML && (model.architecture == "lfm2" || model.architecture == "lfm3")) {
        append("<|startoftext|>")
    }
    messages.forEachIndexed { index, content ->
        val user = index % 2 == 0
        if (model.supportsChatML) {
            append(if (user) "<|im_start|>user\n" else "<|im_start|>assistant\n")
            if (index == messages.lastIndex && kind != null) append(kind.token).append('\n')
            append(content).append("<|im_end|>\n")
        } else {
            append(if (user) "User: " else "Assistant: ")
            if (index == messages.lastIndex && kind != null) append(kind.token).append('\n')
            append(content).append('\n')
        }
    }
    append(if (model.supportsChatML) "<|im_start|>assistant\n" else "Assistant: ")
}
