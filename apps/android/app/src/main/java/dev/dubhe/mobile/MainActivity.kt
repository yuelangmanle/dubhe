package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    private val chat by viewModels<ChatViewModel>()
    private var sharedInput by mutableStateOf<SharedInput?>(null)

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        sharedInput = readShare(intent)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (savedInstanceState == null) sharedInput = readShare(intent)
        enableEdgeToEdge()
        setContent {
            val openModel = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
                if (uri != null) chat.importModel(uri, ModelSlot.Language)
            }
            val openProjector = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
                if (uri != null) chat.importModel(uri, ModelSlot.Projector)
            }
            val openAttachment = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
                if (uri != null) chat.importAttachment(uri)
            }
            fun select(launch: () -> Unit) {
                try {
                    launch()
                } catch (_: ActivityNotFoundException) {
                    chat.showError("这台设备没有可用的文件选择器。")
                }
            }
            DubheTheme {
                DubheScreen(
                    chat, sharedInput, onShareHandled = { sharedInput = null },
                    openModel = { select { openModel.launch(arrayOf("*/*")) } },
                    openProjector = { select { openProjector.launch(arrayOf("*/*")) } },
                    openAttachment = { types -> select { openAttachment.launch(types) } },
                )
            }
        }
    }
}

private data class SharedInput(val text: String? = null, val image: Uri? = null)

private fun readShare(intent: Intent): SharedInput? {
    if (intent.action == Intent.ACTION_PROCESS_TEXT) {
        return intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()?.let { SharedInput(text = it) }
    }
    if (intent.action != Intent.ACTION_SEND) return null
    if (intent.type?.startsWith("image/") == true) {
        val uri = if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        }
        return uri?.takeIf { it.scheme == "content" }?.let { SharedInput(image = it) }
    }
    return intent.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString()?.let { SharedInput(text = it) }
}

private enum class Workspace(val label: String, val glyph: Glyph) {
    Chat("对话", Glyph.Chat), Translate("翻译", Glyph.Translate),
    Scan("扫描", Glyph.Image), Models("模型", Glyph.Model), Settings("设置", Glyph.Settings),
    Tasks("任务与存储", Glyph.History),
}

@Composable
private fun DubheScreen(
    chat: ChatViewModel,
    shared: SharedInput?,
    onShareHandled: () -> Unit,
    openModel: () -> Unit,
    openProjector: () -> Unit,
    openAttachment: (Array<String>) -> Unit,
) {
    val state = chat.state
    var workspace by rememberSaveable { mutableStateOf(Workspace.Chat) }
    var previousWorkspace by rememberSaveable { mutableStateOf(Workspace.Settings) }
    var updateDialog by rememberSaveable { mutableStateOf<Boolean?>(null) }
    var showHistory by rememberSaveable { mutableStateOf(false) }
    var showModelPicker by rememberSaveable { mutableStateOf(false) }
    var openModelShop by rememberSaveable { mutableStateOf(false) }
    var input by rememberSaveable { mutableStateOf("") }
    val focus = LocalFocusManager.current
    val keyboard = WindowInsets.ime.getBottom(LocalDensity.current) > 0
    val idle = state.activity == Activity.Idle
    val chatModels = state.models.filter { it.catalog?.translation != true }
    val workspaceState = rememberSaveableStateHolder()

    fun navigate(item: Workspace) {
        focus.clearFocus()
        workspace = item
    }

    fun openTasks() {
        previousWorkspace = workspace
        navigate(Workspace.Tasks)
    }

    fun openMediaModels() {
        openModelShop = true
        navigate(Workspace.Models)
    }

    LaunchedEffect(shared, state.activity) {
        if (shared == null) return@LaunchedEffect
        workspace = Workspace.Translate
        if (idle) {
            shared.text?.let { chat.setTranslationSource(it, "来自其他应用") }
            shared.image?.let { chat.importTranslation(it, image = true) }
            onShareHandled()
        }
    }
    BackHandler(workspace != Workspace.Chat) { navigate(if (workspace == Workspace.Tasks) previousWorkspace else Workspace.Chat) }

    BoxWithConstraints(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        val expanded = maxWidth >= 720.dp
        WorkspaceBackdrop(Modifier.matchParentSize())
        Scaffold(
            containerColor = Color.Transparent,
            contentColor = MaterialTheme.colorScheme.onBackground,
            bottomBar = {
                if (!expanded && !keyboard) {
                    WorkspaceNavigation(workspace, false, ::navigate, Modifier.navigationBarsPadding())
                }
            },
        ) { insets ->
            Row(Modifier.fillMaxSize().padding(insets).consumeWindowInsets(insets).imePadding()) {
                if (expanded) {
                    WorkspaceNavigation(workspace, true, ::navigate, Modifier.fillMaxHeight())
                }
                Box(Modifier.weight(1f).fillMaxHeight(), contentAlignment = Alignment.TopCenter) {
                    Column(
                        Modifier.widthIn(max = 920.dp).fillMaxSize().padding(horizontal = 18.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        WorkspaceHeader(
                            workspace, idle, history = { showHistory = true },
                            newChat = { chat.newChat(); input = ""; focus.clearFocus() },
                        )
                        state.error?.let { ErrorBanner(it, chat::dismissError) }
                        when (workspace) {
                            Workspace.Chat -> {
                                ChatModelBar(state) {
                                    if (chatModels.isEmpty()) navigate(Workspace.Models)
                                    else showModelPicker = true
                                }
                                val canAsk = state.modelName != null && idle
                                val ask = {
                                    if (canAsk && (input.isNotBlank() || state.attachmentName != null)) {
                                        chat.ask(input)
                                        input = ""
                                        focus.clearFocus()
                                    }
                                }
                                val body = Modifier.weight(1f).fillMaxWidth()
                                if (state.turns.isEmpty()) {
                                    EmptyConversation(
                                        state, body,
                                        startChat = chat::prepareLightChat, stop = chat::stop,
                                        showTranslation = { navigate(Workspace.Translate) },
                                        showScan = { navigate(Workspace.Scan) },
                                        showMediaModels = ::openMediaModels,
                                    )
                                } else {
                                    ChatTranscript(state, body)
                                }
                                if (state.modelName != null) {
                                    Composer(
                                        state, input, { input = it.take(12_000) }, canAsk, ask, chat::stop,
                                        openAttachment, chat::removeAttachment, ::openMediaModels,
                                    )
                                }
                            }
                            Workspace.Translate -> workspaceState.SaveableStateProvider(Workspace.Translate.name) {
                                TranslationScreen(chat, { navigate(Workspace.Models) }, ::openTasks)
                            }
                            Workspace.Scan -> workspaceState.SaveableStateProvider(Workspace.Scan.name) {
                                ScanScreen(chat) { text ->
                                    chat.setTranslationSource(text, "来自扫描")
                                    navigate(Workspace.Translate)
                                }
                            }
                            Workspace.Models -> workspaceState.SaveableStateProvider(Workspace.Models.name) {
                                ModelsScreen(
                                    chat, openModel, openProjector, openTranslation = { navigate(Workspace.Translate) },
                                    openChat = { navigate(Workspace.Chat) },
                                    openShop = openModelShop, onShopOpened = { openModelShop = false },
                                )
                            }
                            Workspace.Settings -> SettingsScreen(state, { updateDialog = false }, { updateDialog = true }, ::openTasks)
                            Workspace.Tasks -> TasksScreen(
                                chat, { navigate(Workspace.Models) },
                                openTranslation = { task -> chat.selectTranslationTask(task); navigate(Workspace.Translate) },
                                openScan = { navigate(Workspace.Scan) },
                            )
                        }
                    }
                }
            }
        }
    }
    updateDialog?.let { showChangelog -> UpdateDialog(startOnChangelog = showChangelog) { updateDialog = null } }
    if (showHistory) HistoryDialog(
        chat, dismiss = { showHistory = false },
        choose = { selected -> chat.selectConversation(selected); input = ""; showHistory = false },
    )
    if (showModelPicker) ChatModelDialog(
        models = chatModels, active = state.activeModel, enabled = idle,
        dismiss = { showModelPicker = false },
        choose = { chat.selectModel(it); showModelPicker = false },
        manage = { showModelPicker = false; navigate(Workspace.Models) },
    )
}

@Composable
private fun WorkspaceHeader(workspace: Workspace, idle: Boolean, history: () -> Unit, newChat: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(top = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        DubheMark(38.dp)
        if (workspace == Workspace.Chat) {
            Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("天枢", style = MaterialTheme.typography.headlineSmall)
                Text("Dubhe", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Row {
                IconAction(Glyph.History, "对话历史", onClick = history)
                IconAction(Glyph.Add, "新对话", idle, newChat)
            }
        } else {
            Text(workspace.label, Modifier.weight(1f).padding(vertical = 9.dp), style = MaterialTheme.typography.headlineSmall)
        }
    }
}

@Composable
private fun WorkspaceNavigation(
    selected: Workspace, rail: Boolean, select: (Workspace) -> Unit, modifier: Modifier,
) {
    val items = Workspace.entries.filterNot { it == Workspace.Tasks }
    val active = if (selected == Workspace.Tasks) Workspace.Settings else selected
    WorkspaceCard(
        modifier.then(
            if (rail) Modifier.width(112.dp).padding(start = 14.dp, top = 12.dp, bottom = 14.dp)
            else Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
        ),
    ) {
        if (rail) {
            Column(Modifier.selectableGroup().padding(8.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                items.forEach { item ->
                    NavigationChoice(item, active == item, true, { select(item) }, Modifier.fillMaxWidth())
                }
            }
        } else {
            Row(
                Modifier.selectableGroup().padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                items.forEach { item ->
                    NavigationChoice(item, active == item, false, { select(item) }, Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun NavigationChoice(item: Workspace, selected: Boolean, rail: Boolean, select: () -> Unit, modifier: Modifier) {
    val colors = MaterialTheme.colorScheme
    val tint = if (selected) colors.onPrimaryContainer else colors.onSurfaceVariant
    val choice = modifier.clip(MaterialTheme.shapes.medium)
        .background(if (selected) colors.primaryContainer else Color.Transparent)
        .selectable(selected, onClick = select, role = Role.Tab)
        .heightIn(min = 54.dp)
        .padding(vertical = 4.dp)
    val label: @Composable () -> Unit = {
        Text(item.label, style = MaterialTheme.typography.labelMedium, color = tint,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium)
    }
    if (rail) Row(
        choice, horizontalArrangement = Arrangement.spacedBy(7.dp, Alignment.CenterHorizontally),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        WorkspaceIcon(item.glyph, Modifier.size(20.dp), tint)
        label()
    } else Column(
        choice,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp, Alignment.CenterVertically),
    ) {
        WorkspaceIcon(item.glyph, color = tint)
        label()
    }
}

@Composable
private fun ChatModelBar(state: ChatState, onClick: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    val loading = state.activity == Activity.Loading || state.activity == Activity.Connecting
    val media = buildList {
        if (state.projector?.vision == true) add("图片 / 视频")
        if (state.projector?.audio == true) add("音频")
    }
    val status = when {
        state.activity == Activity.Loading -> "正在加载模型"
        !state.connected -> "正在连接引擎"
        state.modelName != null -> (media + (state.modelDetail ?: "本机运行 · 已加载")).joinToString("  ·  ")
        else -> "下载或导入本地模型"
    }
    Surface(
        onClick = onClick, enabled = !loading, modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large, color = colors.surface,
        border = BorderStroke(1.dp, colors.outlineVariant),
    ) {
        Row(
            Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier.size(36.dp).clip(MaterialTheme.shapes.small).background(colors.secondaryContainer),
                contentAlignment = Alignment.Center,
            ) { WorkspaceIcon(Glyph.Model, color = colors.onSecondaryContainer) }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Text(state.modelName ?: "选择对话模型", style = MaterialTheme.typography.titleMedium,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                    if (state.modelName != null && !loading) {
                        Box(Modifier.size(6.dp).clip(CircleShape).background(colors.primary))
                    }
                    Text(status, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
                        maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
            if (loading) CircularProgressIndicator(Modifier.size(17.dp), strokeWidth = 2.dp)
            else WorkspaceIcon(Glyph.Chevron)
        }
    }
}

@Composable
private fun EmptyConversation(
    state: ChatState, modifier: Modifier, startChat: () -> Unit, stop: () -> Unit,
    showTranslation: () -> Unit, showScan: () -> Unit, showMediaModels: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val context = LocalContext.current
    val suggested = recommendedChatModel()
    val hasModel = state.models.any { it.catalog?.translation != true }
    Column(
        modifier.verticalScroll(rememberScrollState()).padding(horizontal = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Column(
            Modifier.widthIn(max = 370.dp).padding(vertical = 22.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(11.dp),
        ) {
            DubheMark(62.dp)
            Text(
                if (state.modelName == null) "在手机上做什么？" else "有什么想聊的？",
                style = MaterialTheme.typography.headlineMedium, textAlign = TextAlign.Center,
            )
            if (state.modelName == null) {
                if (!hasModel && suggested != null) {
                    val size = Formatter.formatShortFileSize(context, suggested.bytes)
                    Text(
                        "对话推荐 ${suggested.name} · $size",
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center, color = colors.onSurfaceVariant,
                    )
                }
                if (state.downloading != null) {
                    TransferProgress(state.progress)
                    TextButton(stop, enabled = !state.stopping) { Text("暂停下载") }
                } else {
                    Button(startChat, enabled = state.activity == Activity.Idle) {
                        Text(if (hasModel) "开始本机对话" else "下载模型并开始对话")
                    }
                }
                TextButton(onClick = showTranslation) {
                    WorkspaceIcon(Glyph.Translate, Modifier.size(18.dp), colors.primary)
                    Spacer(Modifier.width(7.dp))
                    Text("翻译文字或截图")
                }
                TextButton(onClick = showScan) {
                    WorkspaceIcon(Glyph.Image, Modifier.size(18.dp), colors.primary)
                    Spacer(Modifier.width(7.dp))
                    Text("从图片提取文字")
                }
                TextButton(onClick = showMediaModels) {
                    WorkspaceIcon(Glyph.Attachment, Modifier.size(18.dp), colors.primary)
                    Spacer(Modifier.width(7.dp))
                    Text("用图片、视频或音频对话")
                }
            }
        }
    }
}

@Composable
private fun ErrorBanner(message: String, dismiss: () -> Unit) {
    Surface(color = MaterialTheme.colorScheme.errorContainer, shape = MaterialTheme.shapes.medium) {
        Row(Modifier.fillMaxWidth().padding(start = 12.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                message, Modifier.weight(1f).padding(vertical = 8.dp),
                color = MaterialTheme.colorScheme.onErrorContainer, style = MaterialTheme.typography.bodyMedium,
            )
            IconAction(Glyph.Close, "关闭提示", onClick = dismiss)
        }
    }
}

@Composable
private fun ChatTranscript(state: ChatState, modifier: Modifier) {
    val scroll = rememberLazyListState()
    val colors = MaterialTheme.colorScheme
    LaunchedEffect(state.turns.size, state.turns.lastOrNull()?.answer) {
        val visible = scroll.layoutInfo.visibleItemsInfo
        if (visible.isEmpty() || (visible.last().index >= state.turns.lastIndex - 1 && !scroll.isScrollInProgress)) {
            scroll.scrollToItem(state.turns.size)
        }
    }
    LazyColumn(
        state = scroll, modifier = modifier,
        contentPadding = PaddingValues(vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        itemsIndexed(state.turns) { index, turn ->
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(13.dp)) {
                Surface(
                    modifier = Modifier.align(Alignment.End).widthIn(max = 520.dp),
                    shape = RoundedCornerShape(20.dp, 20.dp, 7.dp, 20.dp),
                    color = colors.primary, contentColor = colors.onPrimary,
                ) {
                    SelectionContainer {
                        Column(Modifier.padding(horizontal = 15.dp, vertical = 11.dp)) {
                            turn.attachmentName?.let {
                                Text(it, style = MaterialTheme.typography.labelMedium)
                            }
                            if (turn.prompt.isNotBlank()) Text(turn.prompt, style = MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
                val generating = state.activity == Activity.Generating && index == state.turns.lastIndex
                if (turn.answer.isNotBlank() || generating) {
                    Row(horizontalArrangement = Arrangement.spacedBy(9.dp), verticalAlignment = Alignment.Top) {
                        DubheMark(29.dp)
                        Column(Modifier.weight(1f).padding(top = 3.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text(
                                if (generating && turn.answer.isBlank()) "正在生成…" else "天枢",
                                style = MaterialTheme.typography.labelLarge, color = colors.onSurfaceVariant,
                            )
                            if (turn.answer.isNotBlank()) {
                                SelectionContainer { Text(turn.answer, style = MaterialTheme.typography.bodyLarge) }
                            } else {
                                CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                            }
                        }
                    }
                }
            }
        }
        item { Spacer(Modifier.height(2.dp)) }
    }
}

@Composable
private fun Composer(
    state: ChatState, input: String, onInput: (String) -> Unit,
    canAsk: Boolean, ask: () -> Unit, stop: () -> Unit, openAttachment: (Array<String>) -> Unit,
    removeAttachment: () -> Unit, showMediaModels: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val busy = state.activity == Activity.Generating || state.activity == Activity.Importing
    val enabled = if (busy) !state.stopping else canAsk && (input.isNotBlank() || state.attachmentName != null)
    val attachmentName = state.attachmentName
    WorkspaceCard(Modifier.fillMaxWidth().padding(bottom = 7.dp)) {
        Column(Modifier.padding(start = 12.dp, end = 8.dp, top = 12.dp, bottom = 5.dp)) {
            if (attachmentName != null) {
                Row(Modifier.fillMaxWidth().padding(start = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    WorkspaceIcon(Glyph.Attachment, Modifier.size(18.dp))
                    Spacer(Modifier.width(7.dp))
                    Text(attachmentName, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall,
                        color = colors.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    IconAction(Glyph.Close, "移除附件", canAsk, removeAttachment)
                }
            }
            BasicTextField(
                value = input, onValueChange = onInput, enabled = canAsk, maxLines = 5,
                textStyle = MaterialTheme.typography.bodyLarge.copy(color = colors.onSurface),
                cursorBrush = SolidColor(colors.primary),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp).semantics { contentDescription = "输入消息" },
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { ask() }),
                decorationBox = { field ->
                    Box {
                        if (input.isEmpty()) Text(
                            if (attachmentName == null) "输入消息" else "输入问题（可选）",
                            style = MaterialTheme.typography.bodyLarge, color = colors.onSurfaceVariant,
                        )
                        field()
                    }
                },
            )
            if (state.activity == Activity.Importing) TransferProgress(state.progress)
            Row(
                Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                if (state.projector?.vision == true) {
                    TextButton({ openAttachment(arrayOf("image/*", "video/*")) }, enabled = canAsk) { Text("图片 / 视频") }
                } else if (state.projector?.audio == true) {
                    TextButton(showMediaModels, enabled = canAsk) { Text("图片 / 视频") }
                }
                if (state.projector?.audio == true) {
                    TextButton({ openAttachment(arrayOf("audio/*")) }, enabled = canAsk) { Text("音频") }
                } else if (state.projector?.vision == true) {
                    TextButton(showMediaModels, enabled = canAsk) { Text("音频") }
                }
                if (state.projector == null) {
                    TextButton(showMediaModels, enabled = canAsk) { Text("图片 / 视频 / 音频") }
                }
                Text(
                    state.status.orEmpty(),
                    Modifier.weight(1f),
                    style = MaterialTheme.typography.bodySmall, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    color = colors.onSurfaceVariant,
                )
                val action = if (busy) "停止" else "发送"
                Surface(
                    onClick = if (busy) stop else ask, enabled = enabled,
                    modifier = Modifier.size(48.dp).semantics { contentDescription = action },
                    shape = CircleShape,
                    color = if (enabled) colors.primary else colors.surfaceVariant,
                    contentColor = if (enabled) colors.onPrimary else colors.onSurfaceVariant,
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        WorkspaceIcon(if (busy) Glyph.Stop else Glyph.Send,
                            color = if (enabled) colors.onPrimary else colors.onSurfaceVariant)
                    }
                }
            }
        }
    }
}
