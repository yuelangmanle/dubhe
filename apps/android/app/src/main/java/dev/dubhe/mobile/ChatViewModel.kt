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
import dev.dubhe.sdk.ProjectorCapabilities
import java.io.File
import java.util.ArrayList
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

internal data class ChatTurn(val prompt: String, val attachmentName: String? = null, val answer: String = "")

internal enum class Activity { Connecting, Idle, Importing, Loading, Generating }

internal data class ChatState(
    val activity: Activity = Activity.Connecting,
    val modelName: String? = null,
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
)

internal class ChatViewModel(application: Application) : AndroidViewModel(application) {
    var state by mutableStateOf(ChatState())
        private set

    private val store = ModelStore(application)
    private val responses = Messenger(Handler(Looper.getMainLooper(), ::receive))
    private var engine: Messenger? = null
    private var requestId = 0
    private var pending: ModelRequest? = null
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
            bind()
        }
    }

    fun importModel(uri: Uri, slot: ModelSlot) {
        importFile("正在复制 GGUF 文件") { progress ->
            load(ModelRequest(slot, store.importModel(uri, progress), imported = true))
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
                    state = state.copy(activity = Activity.Idle, progress = null, status = "已取消导入", stopping = false)
                }
                throw cancelled
            } catch (failure: Exception) {
                state = state.copy(
                    activity = Activity.Idle, status = null, stopping = false,
                    error = failure.localizedMessage ?: "导入失败。",
                )
            }
        }
    }

    fun ask(value: String) {
        val prompt = value.trim()
        if (state.activity != Activity.Idle || state.modelName == null || prompt.isEmpty()) return
        val previous = state.turns.asReversed().asSequence().filter { it.answer.isNotBlank() }
            .take(6).toList().asReversed()
        val history = ArrayList<String>()
        var chars = prompt.length
        for (turn in previous.asReversed()) {
            chars += turn.prompt.length + turn.answer.length
            if (chars > 32_000) break
            history.add(0, turn.answer)
            history.add(0, turn.prompt)
        }
        history.add(prompt)
        val attachment = draftAttachment
        sentAttachment = attachment
        draftAttachment = null
        state = state.copy(
            activity = Activity.Generating, turns = state.turns + ChatTurn(prompt, attachment?.name),
            stopping = false, status = if (attachment == null) "正在回答" else "正在读取附件",
            error = null, attachmentName = null,
        )
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
        } else if (state.activity == Activity.Generating && !state.stopping) {
            state = state.copy(stopping = true, status = "正在停止")
            if (engine?.post(EngineProtocol.Stop, requestId, Bundle(), responses) != true) disconnected()
        }
    }

    fun newChat() {
        if (state.activity != Activity.Idle) return
        removeAttachment()
        state = state.copy(turns = emptyList(), status = null, error = null)
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
        pending = request
        state = state.copy(activity = Activity.Loading, progress = null, status = null)
        val action = if (request.slot == ModelSlot.Language) EngineProtocol.LoadModel else EngineProtocol.LoadProjector
        send(action, Bundle().apply { putString("path", request.model.file.absolutePath) })
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
                state = state.copy(
                    activity = Activity.Idle,
                    modelName = request.model.name, modelDetail = detail,
                    turns = if (request.imported) emptyList() else state.turns,
                    projector = null, projectorName = null, error = null,
                )
                removeAttachment()
                if (request.imported) {
                    persist(request)
                } else {
                    store.current(ModelSlot.Projector)?.let { load(ModelRequest(ModelSlot.Projector, it)) }
                }
            }
            EngineProtocol.ProjectorReady -> {
                val request = pending ?: return true
                pending = null
                removeAttachment()
                state = state.copy(
                    activity = Activity.Idle, projectorName = request.model.name,
                    projector = ProjectorCapabilities(data.getBoolean("vision"), data.getBoolean("audio")),
                )
                if (request.imported) persist(request)
            }
            EngineProtocol.Text -> {
                val latest = state.turns.lastOrNull() ?: return true
                replaceAnswer(latest.answer + data.getString("text").orEmpty())
                if (!state.stopping) state = state.copy(status = "正在回答")
            }
            EngineProtocol.StoppingForHeat -> {
                state = state.copy(stopping = true, status = "设备温度过高，正在停止")
            }
            EngineProtocol.Complete -> {
                clearSentAttachment()
                replaceAnswer(data.getString("text").orEmpty())
                val speed = data.getDouble("speed")
                val status = when {
                    data.getBoolean("thermal") -> "设备温度过高，已停止。降温后再试。"
                    data.getBoolean("cancelled") -> "已停止"
                    speed > 0 -> "%.1f token/s".format(speed)
                    else -> null
                }
                state = state.copy(activity = Activity.Idle, stopping = false, status = status)
            }
            EngineProtocol.Failed -> {
                discardPending()
                clearSentAttachment()
                val ready = data.getBoolean("modelReady")
                val projectorReady = data.getBoolean("projectorReady")
                if (!projectorReady) removeAttachment()
                state = state.copy(
                    activity = Activity.Idle, stopping = false, status = null,
                    error = data.getString("detail"),
                    modelName = state.modelName.takeIf { ready },
                    modelDetail = state.modelDetail.takeIf { ready },
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
        engine = null
        requestId++
        fileJob?.cancel()
        discardPending()
        clearSentAttachment()
        removeAttachment()
        state = state.copy(
            activity = Activity.Connecting, modelName = null, modelDetail = null, connected = false,
            projector = null, projectorName = null,
            error = "推理引擎已断开，正在尝试重连。", stopping = false, status = null,
        )
    }

    private fun discardPending() {
        deleteLater(pending?.takeIf { it.imported }?.model?.file)
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
        pending?.takeIf { it.imported }?.model?.file?.delete()
        draftAttachment?.attachment?.file?.delete()
        sentAttachment?.attachment?.file?.delete()
        super.onCleared()
    }

    private data class ModelRequest(
        val slot: ModelSlot,
        val model: LocalModel,
        val imported: Boolean = false,
    )
}
