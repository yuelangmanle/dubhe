package dev.dubhe.mobile

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.dubhe.sdk.AttachmentKind
import dev.dubhe.sdk.ProjectorCapabilities
import java.io.File
import java.util.ArrayList
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

internal data class ChatTurn(val prompt: String, val attachmentName: String? = null, val answer: String = "")

internal enum class Activity { Connecting, Idle, Importing, Loading, Generating, Translating }

internal data class ChatState(
    val activity: Activity = Activity.Connecting,
    val modelName: String? = null,
    val activeModel: LocalModel? = null,
    val modelDetail: String? = null,
    val projectorName: String? = null,
    val projector: ProjectorCapabilities? = null,
    val attachmentName: String? = null,
    val turns: List<ChatTurn> = emptyList(),
    val progress: Float? = null,
    val status: String? = null,
    val error: String? = null,
    val stopping: Boolean = false,
    val connected: Boolean = false,
    val models: List<LocalModel> = emptyList(),
    val partialDownloads: Set<String> = emptySet(),
    val downloading: String? = null,
    val modelStorage: ModelStorage? = null,
)

internal class ChatViewModel(application: Application) : AndroidViewModel(application) {
    var state by mutableStateOf(ChatState())
        private set
    var translation by mutableStateOf(TranslationState())
        private set
    var conversations by mutableStateOf(Conversations())
        private set
    var translationTasks by mutableStateOf(emptyList<TranslationTask>())
        private set
    var translationTask by mutableStateOf<TranslationTask?>(null)
        private set
    var scanText by mutableStateOf("")
        private set
    var hasSavedScan by mutableStateOf(false)
        private set

    private val store = ModelStore(application)
    private val historyStore = ConversationStore(application)
    private val translationStore = TranslationTaskStore(application)
    private val translationPrefs = application.getSharedPreferences("dubhe-translation", Context.MODE_PRIVATE)
    private val historyIO = Dispatchers.IO.limitedParallelism(1)
    private val responses = Messenger(Handler(Looper.getMainLooper(), ::receive))
    private var engine: Messenger? = null
    private var requestId = 0
    private var pending: ModelRequest? = null
    private var pendingRemoval: LocalModel? = null
    private var draftAttachment: LocalAttachment? = null
    private var sentAttachment: LocalAttachment? = null
    private var fileJob: Job? = null
    private var persistJob: Job? = null
    private var bound = false

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            val endpoint = Messenger(binder)
            engine = endpoint
            viewModelScope.launch {
                persistJob?.join()
                if (engine !== endpoint) return@launch
                state = state.copy(connected = true, activity = Activity.Idle)
                store.current(ModelSlot.Language)?.let { load(ModelRequest(ModelSlot.Language, it)) }
            }
        }

        override fun onServiceDisconnected(name: ComponentName) {
            disconnected()
        }

        override fun onBindingDied(name: ComponentName) {
            disconnected()
            if (bound) application.unbindService(this)
            bind()
        }

        override fun onNullBinding(name: ComponentName) {
            disconnected()
        }
    }

    init {
        viewModelScope.launch {
            store.discardUnreferencedFiles()
            refreshLibrary()
            try {
                conversations = withContext(historyIO) { historyStore.read() }
                state = state.copy(turns = conversations.items.firstOrNull { it.id == conversations.selected }?.turns.orEmpty())
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                showError("无法读取本地对话历史。")
            }
            withContext(Dispatchers.IO) { translationStore.list() }.let { translationTasks = it }
            scanText = withContext(Dispatchers.IO) { translationStore.readScan() }
            hasSavedScan = scanText.isNotEmpty()
            val terms = withContext(Dispatchers.IO) { translationPrefs.getString("glossary", "").orEmpty() }
            translation = translation.copy(glossary = terms)
            bind()
        }
    }

    fun importModel(uri: Uri, slot: ModelSlot) {
        if (slot == ModelSlot.Projector && state.activeModel?.bundledCatalog != null) return
        importFile("正在复制 GGUF 文件") { progress ->
            val model = store.importModel(uri, progress)
            if (slot == ModelSlot.Language && model.catalog?.translation == true) {
                val saved = withContext(Dispatchers.IO) { store.remember(model) }
                if (!saved) {
                    deleteLater(model.file)
                    error("无法更新模型库。")
                }
                refreshLibrary()
                state = state.copy(activity = Activity.Idle, status = "翻译模型已导入")
            } else {
                load(ModelRequest(slot, model, saveSelection = true, temporaryFile = true))
            }
        }
    }

    fun downloadModel(item: CatalogModel, loadAfter: Boolean = false) {
        if (!state.connected || state.activity != Activity.Idle) return
        state = state.copy(downloading = item.file)
        importFile("正在下载 ${item.name}") { update ->
            val model = store.download(item) { update(it) }
            if (loadAfter && !item.translation) load(ModelRequest(ModelSlot.Language, model, saveSelection = true))
            else state = state.copy(activity = Activity.Idle, progress = null,
                status = if (item.projector == null) "模型已下载" else "模型和多模态组件已下载")
        }
    }

    fun prepareLightChat() {
        if (state.activity != Activity.Idle) return
        val installed = state.models.filter { it.catalog?.translation != true }.minByOrNull { it.file.length() }
        val suggested = recommendedChatModel()
        if (installed != null) selectModel(installed)
        else if (suggested != null) downloadModel(suggested, loadAfter = true)
    }

    fun selectModel(model: LocalModel) {
        if (state.activity != Activity.Idle || state.activeModel == model || model.catalog?.translation == true) return
        load(ModelRequest(ModelSlot.Language, model, saveSelection = true))
    }

    fun deleteModel(model: LocalModel) {
        if (state.activity != Activity.Idle) return
        if (state.activeModel?.file == model.file) {
            pendingRemoval = model
            state = state.copy(activity = Activity.Loading, status = "正在卸载模型", error = null)
            send(EngineProtocol.UnloadModel, Bundle())
        } else {
            deleteModelFile(model)
        }
    }

    private fun deleteModelFile(model: LocalModel) {
        state = state.copy(activity = Activity.Loading, status = "正在删除模型", error = null)
        persistJob = viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { store.remove(model) }
            } catch (failure: Exception) {
                showError(failure.localizedMessage ?: "无法删除模型。")
            } finally {
                refreshLibrary()
                if (state.connected) state = state.copy(activity = Activity.Idle, status = null)
            }
        }
    }

    private fun refreshLibrary() {
        state = state.copy(models = store.library(), partialDownloads = store.partials(), modelStorage = store.storage())
    }

    fun clearPartialDownloads() {
        importFile("正在清理未完成的下载") {
            store.clearPartialDownloads()
            refreshLibrary()
            state = state.copy(activity = Activity.Idle, status = "已清理未完成的下载")
        }
    }

    fun importAttachment(uri: Uri) {
        val projector = state.projector ?: return
        importFile("正在复制附件") { progress ->
            val attachment = store.importAttachment(uri, projector, progress)
            removeAttachment()
            draftAttachment = attachment
            state = state.copy(activity = Activity.Idle, attachmentName = attachment.name, status = null)
        }
    }

    fun removeAttachment() {
        deleteLater(draftAttachment?.attachment?.file)
        draftAttachment = null
        state = state.copy(attachmentName = null)
    }

    private fun importFile(label: String, operation: suspend ((Float?) -> Unit) -> Unit) {
        if (!state.connected || state.activity != Activity.Idle) return
        state = state.copy(activity = Activity.Importing, progress = null, error = null, status = label)
        fileJob = viewModelScope.launch {
            try {
                operation { progress ->
                    viewModelScope.launch {
                        if (state.activity == Activity.Importing) state = state.copy(progress = progress)
                    }
                }
            } catch (cancelled: CancellationException) {
                if (state.activity == Activity.Importing && state.connected) {
                    state = state.copy(activity = Activity.Idle, progress = null, stopping = false,
                        status = if (state.downloading == null) "已取消导入" else "已暂停下载")
                }
                throw cancelled
            } catch (failure: Exception) {
                state = state.copy(
                    activity = Activity.Idle, status = null, stopping = false,
                    error = failure.localizedMessage ?: "导入失败。",
                )
            } finally {
                if (state.downloading != null) {
                    refreshLibrary()
                    state = state.copy(downloading = null)
                }
            }
        }
    }

    fun ask(value: String) {
        val attachment = draftAttachment
        val input = value.trim()
        if (state.activity != Activity.Idle || state.modelName == null || (input.isEmpty() && attachment == null)) return
        val prompt = input.ifEmpty { defaultMediaPrompt(attachment?.attachment?.kind) }
        val previous = state.turns.asReversed().asSequence().filter { it.answer.isNotBlank() }
            .take(6).toList().asReversed()
        val history = ArrayList<String>()
        var chars = prompt.length
        for (turn in previous.asReversed()) {
            val priorPrompt = turn.prompt.ifEmpty { if (turn.attachmentName != null) "请转写音频。" else "" }
            chars += priorPrompt.length + turn.answer.length
            if (chars > 32_000) break
            history.add(0, turn.answer)
            history.add(0, priorPrompt)
        }
        history.add(prompt)
        sentAttachment = attachment
        draftAttachment = null
        state = state.copy(
            activity = Activity.Generating, turns = state.turns + ChatTurn(prompt, attachment?.name),
            stopping = false, status = if (attachment == null) "正在回答" else "正在读取附件",
            error = null, attachmentName = null,
        )
        saveCurrentConversation()
        send(EngineProtocol.Generate, Bundle().apply {
            putStringArrayList("messages", history)
            attachment?.attachment?.let {
                putString("attachmentPath", it.file.absolutePath)
                putString("attachmentKind", it.kind.name)
            }
        })
    }

    fun stop() {
        if (state.activity == Activity.Importing && !state.stopping) {
            state = state.copy(stopping = true, status = "正在取消")
            fileJob?.cancel()
        } else if ((state.activity == Activity.Generating || state.activity == Activity.Translating) && !state.stopping) {
            state = state.copy(stopping = true, status = "正在停止")
            if (engine?.post(EngineProtocol.Stop, requestId, Bundle(), responses) != true) disconnected()
        }
    }

    fun newChat() {
        if (state.activity != Activity.Idle) return
        removeAttachment()
        conversations = conversations.copy(selected = null)
        state = state.copy(turns = emptyList(), status = null, error = null)
        writeHistory()
    }

    fun selectConversation(chat: Conversation) {
        if (state.activity != Activity.Idle) return
        removeAttachment()
        conversations = conversations.copy(selected = chat.id)
        state = state.copy(turns = chat.turns, status = null, error = null)
        writeHistory()
    }

    fun deleteConversation(chat: Conversation) {
        if (state.activity != Activity.Idle) return
        if (conversations.selected == chat.id) {
            removeAttachment()
            conversations = conversations.copy(selected = null)
            state = state.copy(turns = emptyList())
        }
        conversations = conversations.copy(items = conversations.items.filterNot { it.id == chat.id })
        writeHistory()
    }

    private fun saveCurrentConversation() {
        if (state.turns.isEmpty()) return
        val id = conversations.selected ?: UUID.randomUUID().toString()
        conversations = Conversations(id, listOf(Conversation(id, state.turns)) + conversations.items.filterNot { it.id == id })
        writeHistory()
    }

    private fun writeHistory() {
        val snapshot = conversations
        viewModelScope.launch {
            try {
                withContext(historyIO) { historyStore.write(snapshot) }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                showError("无法保存对话历史。")
            }
        }
    }

    fun updateTranslation(update: (TranslationState) -> TranslationState) {
        if (state.activity == Activity.Translating) return
        val next = update(translation)
        if (next.copy(result = "") != translation.copy(result = "")) {
            translationTask = null
            translation = next.copy(result = "")
        }
    }

    fun setTranslationSource(text: String, label: String? = null) {
        if (text.length > translationLimit) {
            showError("一次最多翻译 $translationLimit 个字符，请分段输入。")
            return
        }
        updateTranslation { it.copy(source = text, result = "", inputLabel = label) }
    }

    fun importTranslation(uri: Uri, image: Boolean) {
        importFile(if (image) "正在识别图片文字" else "正在读取文件") {
            val text = readTranslationInput(getApplication(), uri, image)
            setTranslationSource(text, if (image) "来自图片" else "来自文件")
            state = state.copy(activity = Activity.Idle, status = null)
        }
    }

    fun scanImage(uri: Uri) {
        importFile("正在本机识别图片") {
            val text = readTranslationInput(getApplication(), uri, image = true)
            withContext(Dispatchers.IO) { translationStore.writeScan(text) }
            scanText = text
            hasSavedScan = true
            state = state.copy(activity = Activity.Idle, status = null)
        }
    }

    fun editScanText(text: String) {
        if (state.activity == Activity.Idle) scanText = text.take(translationLimit)
    }

    fun saveScan() {
        if (state.activity != Activity.Idle) return
        val text = scanText
        importFile("正在保存识别结果") {
            withContext(Dispatchers.IO) { translationStore.writeScan(text) }
            hasSavedScan = text.isNotEmpty()
            state = state.copy(activity = Activity.Idle, status = "识别结果已保存")
        }
    }

    fun clearScan() {
        importFile("正在清除扫描结果") {
            withContext(Dispatchers.IO) { translationStore.writeScan("") }
            scanText = ""
            hasSavedScan = false
            state = state.copy(activity = Activity.Idle, status = null)
        }
    }

    fun selectTranslationTask(task: TranslationTask) {
        if (state.activity != Activity.Idle) return
        translationTask = task
        translation = task.input
        state = state.copy(error = null, status = null)
    }

    fun newTranslation() {
        if (state.activity != Activity.Idle) return
        translationTask = null
        translation = translation.copy(
            source = "", result = "", inputLabel = null,
            glossary = translationPrefs.getString("glossary", "").orEmpty(),
        )
        state = state.copy(error = null, status = null)
    }

    fun changeTranslationModel() {
        if (state.activity != Activity.Idle) return
        translationTask = null
        translation = translation.copy(result = "")
        state = state.copy(error = null, status = null)
    }

    fun saveTranslationGlossary(value: String): String? {
        if (state.activity != Activity.Idle) return "请等当前任务结束后再修改术语。"
        val normalized = value.trim()
        return try {
            readTranslationGlossary(normalized)
            translationPrefs.edit().putString("glossary", normalized).apply()
            updateTranslation { it.copy(glossary = normalized) }
            null
        } catch (error: IllegalArgumentException) {
            error.message
        }
    }

    fun deleteTranslationTask(task: TranslationTask) {
        importFile("正在删除任务") {
            withContext(Dispatchers.IO) { translationStore.remove(task.id) }
            translationTasks = translationTasks.filterNot { it.id == task.id }
            if (translationTask?.id == task.id) translationTask = null
            state = state.copy(activity = Activity.Idle, status = null)
        }
    }

    private fun rememberTranslationTask(task: TranslationTask) {
        translationTask = task
        translationTasks = listOf(task) + translationTasks.filterNot { it.id == task.id }
        translation = task.input
    }

    fun translate() {
        val input = translation
        if (state.activity != Activity.Idle || input.source.isBlank()) return
        val resumed = translationTask?.takeUnless { it.finished }
        val dedicated = if (resumed != null) {
            state.models.firstOrNull { resumed.dedicated && it.translationKey == resumed.modelKey }
        } else {
            state.models.firstOrNull { it.catalog?.translation == true }.takeIf { input.preferDedicated || state.activeModel == null }
        }
        if (resumed != null && (if (resumed.dedicated) dedicated == null else state.activeModel?.translationKey != resumed.modelKey)) {
            showError("继续需要原来的模型 ${resumed.modelName}。请到模型页下载或加载它。")
            return
        }
        if (dedicated == null && state.activeModel == null) {
            showError("请先在模型页下载混元翻译，或加载一个对话模型。")
            return
        }
        if (input.from == input.to) {
            showError("请选择不同的原文和目标语言。")
            return
        }
        val model = dedicated ?: requireNotNull(state.activeModel)
        val source = normalizeTranslationSource(input.source)
        val task = resumed ?: TranslationTask(
            UUID.randomUUID().toString(), input.copy(source = source, result = ""),
            model.translationKey, model.name, dedicated != null,
            total = translationBlocks(source).count { it.isNotBlank() },
        )
        state = state.copy(activity = Activity.Translating, status = "正在准备翻译", progress = null, error = null, stopping = false)
        val endpoint = engine
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { if (resumed == null) translationStore.write(task) }
                rememberTranslationTask(task)
                if (engine !== endpoint || state.activity != Activity.Translating) return@launch
                if (state.stopping) {
                    state = state.copy(activity = Activity.Idle, stopping = false, status = "已暂停翻译")
                    return@launch
                }
                send(EngineProtocol.Translate, Bundle().apply {
                    putString("taskId", task.id)
                    dedicated?.let { putString("modelPath", it.file.absolutePath) }
                })
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                if (state.activity == Activity.Translating) state = state.copy(
                    activity = Activity.Idle, stopping = false, status = null, error = "无法保存翻译任务。",
                )
            }
        }
    }

    fun exportTranslation(uri: Uri, format: TranslationFormat, result: String) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val output = getApplication<Application>().contentResolver.openOutputStream(uri, "wt")
                        ?: error("无法打开保存位置。")
                    output.use { format.write(result, it) }
                }
            } catch (failure: Exception) {
                showError(failure.localizedMessage ?: "无法导出译文。")
            }
        }
    }

    fun showError(message: String) {
        state = state.copy(error = message)
    }

    fun dismissError() {
        state = state.copy(error = null)
    }

    private fun bind() {
        val app = getApplication<Application>()
        bound = app.bindService(Intent(app, EngineService::class.java), connection, Context.BIND_AUTO_CREATE)
        if (!bound) state = state.copy(error = "无法连接推理引擎。")
    }

    private fun load(request: ModelRequest) {
        val bundle = request.model.bundledCatalog
        val bundledProjector = store.catalogProjector(request.model)
        if (bundle != null && bundledProjector == null) {
            refreshLibrary()
            state = state.copy(activity = Activity.Idle, progress = null, status = null,
                error = "模型文件不完整，请到模型商店继续下载。")
            return
        }
        val legacyProjector = if (request.slot == ModelSlot.Language &&
            !request.saveSelection && bundle == null) store.current(ModelSlot.Projector) else null
        pending = request
        state = state.copy(activity = Activity.Loading, progress = null, status = null)
        val action = if (request.slot == ModelSlot.Language) EngineProtocol.LoadModel else EngineProtocol.LoadProjector
        send(action, Bundle().apply {
            putString("path", request.model.file.absolutePath)
            (bundledProjector ?: legacyProjector)?.let { putString("projectorPath", it.file.absolutePath) }
            putString("projectorModality", bundle?.modality?.name)
        })
    }

    private fun send(what: Int, data: Bundle) {
        if (engine?.post(what, ++requestId, data, responses) != true) disconnected()
    }

    private fun receive(message: Message): Boolean {
        if (message.arg1 != requestId) return true
        val data = message.data
        when (message.what) {
            EngineProtocol.ModelReady -> {
                val request = pending ?: return true
                pending = null
                val detail = listOfNotNull(
                    data.getString("backend"), data.getString("quantization"),
                    "${data.getInt("context")} token 上下文",
                ).joinToString("  ·  ")
                val projectorPath = data.getString("projectorPath")
                state = state.copy(
                    activity = Activity.Idle,
                    modelName = request.model.name, modelDetail = detail, activeModel = request.model,
                    projector = projectorPath?.let { ProjectorCapabilities(data.getBoolean("vision"), data.getBoolean("audio")) },
                    projectorName = projectorPath?.let { path ->
                        store.current(ModelSlot.Projector)?.takeIf { it.file.absolutePath == path }?.name ?: File(path).name
                    },
                    error = null,
                )
                removeAttachment()
                if (request.saveSelection) persist(request)
            }
            EngineProtocol.ProjectorReady -> {
                val request = pending ?: return true
                pending = null
                removeAttachment()
                state = state.copy(
                    activity = Activity.Idle, projectorName = request.model.name,
                    projector = ProjectorCapabilities(data.getBoolean("vision"), data.getBoolean("audio")),
                )
                if (request.saveSelection) persist(request)
            }
            EngineProtocol.ModelUnloaded -> {
                val model = pendingRemoval ?: return true
                pendingRemoval = null
                removeAttachment()
                state = state.copy(
                    modelName = null, modelDetail = null, activeModel = null,
                    projector = null, projectorName = null,
                )
                deleteModelFile(model)
            }
            EngineProtocol.Text -> {
                val text = data.getString("text").orEmpty()
                if (state.activity == Activity.Translating) {
                    val task = translationTask ?: return true
                    val completed = data.getInt("completed")
                    rememberTranslationTask(task.copy(
                        input = translation.copy(result = translation.result + text),
                        completed = completed, nextBlock = data.getInt("nextBlock"),
                        finished = data.getBoolean("finished"), updatedAt = data.getLong("updatedAt"),
                    ))
                    if (!state.stopping) {
                        state = state.copy(status = "已翻译 $completed / ${task.total} 段", progress = completed.toFloat() / task.total)
                    }
                } else {
                    val latest = state.turns.lastOrNull() ?: return true
                    replaceAnswer(latest.answer + text)
                    if (!state.stopping) state = state.copy(status = "正在回答")
                }
            }
            EngineProtocol.StoppingForHeat -> {
                state = state.copy(stopping = true, status = "设备温度过高，正在停止")
            }
            EngineProtocol.Complete -> {
                val translating = state.activity == Activity.Translating
                if (!translating) {
                    clearSentAttachment()
                    replaceAnswer(data.getString("text").orEmpty())
                    saveCurrentConversation()
                }
                val speed = data.getDouble("speed")
                val status = when {
                    data.getBoolean("thermal") -> "设备温度过高，已停止。降温后再试。"
                    data.getBoolean("cancelled") -> "已停止"
                    translating -> "翻译完成"
                    speed > 0 -> "%.1f token/s".format(speed)
                    else -> null
                }
                state = state.copy(activity = Activity.Idle, stopping = false, status = status)
            }
            EngineProtocol.Failed -> {
                if (state.activity == Activity.Generating) saveCurrentConversation()
                discardPending()
                pendingRemoval = null
                clearSentAttachment()
                val ready = data.getBoolean("modelReady")
                val projectorReady = data.getBoolean("projectorReady")
                if (!projectorReady) removeAttachment()
                state = state.copy(
                    activity = Activity.Idle, stopping = false, status = null,
                    error = data.getString("detail"),
                    modelName = state.modelName.takeIf { ready },
                    modelDetail = state.modelDetail.takeIf { ready },
                    activeModel = state.activeModel.takeIf { ready },
                    projector = state.projector.takeIf { projectorReady },
                    projectorName = state.projectorName.takeIf { projectorReady },
                )
            }
        }
        return true
    }

    private fun persist(request: ModelRequest) {
        state = state.copy(activity = Activity.Loading)
        val id = requestId
        persistJob = viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) { store.retain(request.slot, request.model) }
            refreshLibrary()
            if (state.connected && requestId == id) state = state.copy(
                activity = Activity.Idle,
                error = if (saved) null else "模型已加载，但没有保存导入记录。",
            )
        }
    }

    private fun replaceAnswer(text: String) {
        val latest = state.turns.lastOrNull() ?: return
        state = state.copy(turns = state.turns.dropLast(1) + latest.copy(answer = text))
    }

    private fun disconnected() {
        if (state.activity == Activity.Generating) saveCurrentConversation()
        engine = null
        requestId++
        fileJob?.cancel()
        discardPending()
        pendingRemoval = null
        clearSentAttachment()
        removeAttachment()
        state = state.copy(
            activity = Activity.Connecting, modelName = null, modelDetail = null, activeModel = null, connected = false,
            projector = null, projectorName = null,
            error = "推理引擎已断开，正在尝试重连。", stopping = false, status = null,
        )
    }

    private fun discardPending() {
        deleteLater(pending?.takeIf { it.temporaryFile }?.model?.file)
        pending = null
    }

    private fun clearSentAttachment() {
        deleteLater(sentAttachment?.attachment?.file)
        sentAttachment = null
    }

    private fun deleteLater(file: File?) {
        if (file != null) viewModelScope.launch(Dispatchers.IO) { file.delete() }
    }

    override fun onCleared() {
        if (bound) getApplication<Application>().unbindService(connection)
        pending?.takeIf { it.temporaryFile }?.model?.file?.delete()
        draftAttachment?.attachment?.file?.delete()
        sentAttachment?.attachment?.file?.delete()
        super.onCleared()
    }

    private data class ModelRequest(
        val slot: ModelSlot,
        val model: LocalModel,
        val saveSelection: Boolean = false,
        val temporaryFile: Boolean = false,
    )
}

private fun defaultMediaPrompt(kind: AttachmentKind?): String = when (kind) {
    AttachmentKind.Image -> "请描述这张图片。"
    AttachmentKind.Video -> "请描述这段视频。"
    AttachmentKind.Audio, null -> ""
}
