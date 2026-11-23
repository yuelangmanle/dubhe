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
import kotlinx.coroutines.withContext

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
    private var chatResident = false
    private var projector: ProjectorCapabilities? = null
    private var projectorFile: File? = null
    private var projectorModality: CatalogModality? = null
    private val translations by lazy { TranslationTaskStore(this) }
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
                    EngineProtocol.LoadModel -> loadModel(data, id, reply)
                    EngineProtocol.LoadProjector -> loadProjector(File(requireNotNull(data.getString("path"))), id, reply)
                    EngineProtocol.UnloadModel -> {
                        unloadChatModel()
                        reply.post(EngineProtocol.ModelUnloaded, id, Bundle())
                    }
                    EngineProtocol.Generate -> {
                        check(!overheated) { "设备温度过高，降温后再试。" }
                        generationReply = reply
                        generate(data, id, reply)
                    }
                    EngineProtocol.Translate -> {
                        check(!overheated) { "设备温度过高，降温后再试。" }
                        generationReply = reply
                        translate(data, id, reply)
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

    private suspend fun loadModel(data: Bundle, id: Int, reply: Messenger) {
        val file = File(requireNotNull(data.getString("path")))
        val component = data.getString("projectorPath")?.let(::File)
        val modality = data.getString("projectorModality")?.let(CatalogModality::valueOf)
        val previousFile = modelFile
        val previousProjector = projectorFile
        val previousModality = projectorModality
        val loaded = try {
            activateChatModel(file, component, modality)
        } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            try {
                if (previousFile == null) unloadChatModel()
                else activateChatModel(previousFile, previousProjector, previousModality)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                unloadChatModel()
            }
            throw failure
        }
        reply.post(EngineProtocol.ModelReady, id, Bundle().apply {
            putString("backend", loaded.backend)
            putString("quantization", loaded.quantization)
            putInt("context", loaded.contextSize)
            projectorFile?.let { putString("projectorPath", it.absolutePath) }
            projector?.let {
                putBoolean("vision", it.vision)
                putBoolean("audio", it.audio)
            }
        })
    }

    private suspend fun loadProjector(file: File, id: Int, reply: Messenger) {
        ensureChatModel()
        check(projectorModality == null) { "这个模型已经配套加载了多模态组件。" }
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

    private suspend fun ensureChatModel(): ModelInfo {
        if (chatResident) return checkNotNull(model)
        val file = modelFile ?: error("请先在模型库加载一个对话模型。")
        return try {
            activateChatModel(file, projectorFile, projectorModality)
        } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            unloadChatModel()
            throw failure
        }
    }

    private suspend fun activateChatModel(file: File, component: File?, modality: CatalogModality?): ModelInfo {
        val loaded = engine.loadModel(file)
        val capabilities = if (modality == null) null else {
            val required = if (modality == CatalogModality.AUDIO) AttachmentKind.Audio else AttachmentKind.Image
            val result = engine.loadProjector(requireNotNull(component))
            check((if (modality == CatalogModality.AUDIO) result.audio else result.vision) &&
                engine.tokenize(required.token, addSpecial = false).size == 1) {
                "多模态组件与模型不匹配，请重新下载模型。"
            }
            result
        }
        model = loaded
        modelFile = file
        projector = capabilities
        projectorFile = component.takeIf { capabilities != null }
        projectorModality = modality
        chatResident = true
        if (modality == null) restoreProjector(component)
        return loaded
    }

    private suspend fun unloadChatModel() {
        model = null
        modelFile = null
        projector = null
        projectorFile = null
        projectorModality = null
        chatResident = false
        engine.unloadModel()
    }

    private suspend fun generate(data: Bundle, id: Int, reply: Messenger) {
        val info = ensureChatModel()
        val allMessages = requireNotNull(data.getStringArrayList("messages"))
        val attachment = data.getString("attachmentPath")?.let { path ->
            Attachment(File(path), AttachmentKind.valueOf(requireNotNull(data.getString("attachmentKind"))))
        }
        val maxTokens = minOf(256, maxOf(1, info.contextSize / 4))
        val transcribing = attachment?.kind == AttachmentKind.Audio && allMessages.lastOrNull().isNullOrBlank()
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
        val sampling = if (transcribing) Sampling(maxTokens = maxTokens, temperature = 0f) else Sampling(maxTokens = maxTokens)
        val response = engine.generate(tokens, sampling, attachment) { text ->
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

    private suspend fun translate(data: Bundle, id: Int, reply: Messenger) {
        val taskId = requireNotNull(data.getString("taskId"))
        var task = withContext(Dispatchers.IO) { translations.read(taskId) }
            ?: error("找不到这条翻译任务。")
        require(!task.finished) { "这条翻译任务已完成。" }
        val from = task.input.from
        val to = task.input.to
        val glossary = readTranslationGlossary(task.input.glossary)
        val dedicated = data.getString("modelPath")?.let(::File)
        require((dedicated != null) == task.dedicated)
        if (dedicated != null) {
            engine.unloadModel()
            chatResident = false
        }
        try {
            val info = if (dedicated == null) ensureChatModel() else engine.loadModel(dedicated)
            val blocks = translationBlocks(task.input.source)
            for (index in task.nextBlock until blocks.size) {
                if (stopRequested) break
                val block = blocks[index]
                val translated = if (block.isBlank()) block else {
                    val instruction = translationPrompt(block, from, to, dedicated != null, glossary)
                    val prompt = if (dedicated != null) instruction else renderPrompt(info, listOf(instruction), null)
                        .let { if (info.architecture == "qwen35") "$it<think>\n\n</think>\n\n" else it }
                    val tokens = engine.tokenize(prompt, addSpecial = dedicated == null)
                    if (stopRequested) break
                    val budget = minOf(1024, info.contextSize - tokens.size)
                    check(budget >= 32) { "这段原文超出了模型可处理的长度。" }
                    val result = engine.generate(tokens, Sampling(maxTokens = budget, temperature = 0f, repeatPenalty = 1f))
                    if (result.cancelled || stopRequested) break
                    check(result.stats.generatedTokens < budget) { "译文达到长度上限，请缩短这段原文后重试。" }
                    result.text.replace(Regex("<think>[\\s\\S]*?</think>"), "").trim().also {
                        check(it.isNotEmpty()) { "模型没有输出译文。" }
                    }
                }
                task = task.checkpoint(index, translated, block.isNotBlank(), index == blocks.lastIndex)
                val checkpoint = task
                withContext(Dispatchers.IO) { translations.write(checkpoint) }
                if (!reply.post(EngineProtocol.Text, id, Bundle().apply {
                    putString("text", translated)
                    putInt("nextBlock", checkpoint.nextBlock)
                    putInt("completed", checkpoint.completed)
                    putBoolean("finished", checkpoint.finished)
                    putLong("updatedAt", checkpoint.updatedAt)
                })) break
            }
        } finally {
            if (dedicated != null) engine.unloadModel()
        }
        reply.post(EngineProtocol.Complete, id, Bundle().apply {
            putBoolean("cancelled", stopRequested)
            putBoolean("thermal", thermalStop)
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
    if (model.architecture == "lfm3" && model.supportsChatML &&
        kind == AttachmentKind.Audio && messages.lastOrNull().isNullOrBlank()) {
        append("<|im_start|>system\nPerform ASR.<|im_end|>\n")
        append("<|im_start|>user\n${kind.token}<|im_end|>\n<|im_start|>assistant\n")
        return@buildString
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
