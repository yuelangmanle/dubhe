package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Build
import androidx.activity.compose.BackHandler
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
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
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkPalette else lightPalette) {
                DubheScreen(
                    chat, sharedInput, onShareHandled = { sharedInput = null },
                    openModel = { select { openModel.launch(arrayOf("*/*")) } },
                    openProjector = { select { openProjector.launch(arrayOf("*/*")) } },
                    openAttachment = {
                        val projector = chat.state.projector
                        val types = buildList {
                            if (projector?.vision == true) addAll(listOf("image/*", "video/*"))
                            if (projector?.audio == true) add("audio/*")
                        }
                        if (types.isNotEmpty()) select { openAttachment.launch(types.toTypedArray()) }
                    },
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

private val lightPalette = lightColorScheme(
    primary = Color(0xFF086353), onPrimary = Color.White,
    primaryContainer = Color(0xFFD0F0E5), onPrimaryContainer = Color(0xFF07372D),
    secondaryContainer = Color(0xFFD6ECE4), onSecondaryContainer = Color(0xFF113B30),
    background = Color(0xFFF4F7F6), onBackground = Color(0xFF152822),
    surface = Color.White, onSurface = Color(0xFF152822),
    surfaceVariant = Color(0xFFE2EBE7), onSurfaceVariant = Color(0xFF465953),
)

private val darkPalette = darkColorScheme(
    primary = Color(0xFF92E0CD), onPrimary = Color(0xFF0C362C),
    primaryContainer = Color(0xFF1F4940), onPrimaryContainer = Color(0xFFE1FFF6),
    secondaryContainer = Color(0xFF294B42), onSecondaryContainer = Color(0xFFD6EFE6),
    background = Color(0xFF0F1917), onBackground = Color(0xFFE1F0EB),
    surface = Color(0xFF172522), onSurface = Color(0xFFE1F0EB),
    surfaceVariant = Color(0xFF293B36), onSurfaceVariant = Color(0xFFC0D5CC),
)

private enum class Workspace(val label: String) {
    Chat("对话"), Translate("翻译"), Models("模型"), Settings("设置"),
}

@Composable
private fun DubheScreen(
    chat: ChatViewModel,
    shared: SharedInput?,
    onShareHandled: () -> Unit,
    openModel: () -> Unit,
    openProjector: () -> Unit,
    openAttachment: () -> Unit,
) {
    val state = chat.state
    var workspace by rememberSaveable { mutableStateOf(Workspace.Chat) }
    var showUpdates by rememberSaveable { mutableStateOf(false) }
    var showHistory by rememberSaveable { mutableStateOf(false) }
    var input by rememberSaveable { mutableStateOf("") }
    val focus = LocalFocusManager.current
    val keyboard = WindowInsets.ime.getBottom(LocalDensity.current) > 0
    LaunchedEffect(shared, state.activity) {
        if (shared == null) return@LaunchedEffect
        workspace = Workspace.Translate
        if (state.activity == Activity.Idle) {
            shared.text?.let { chat.setTranslationSource(it, "来自其他应用") }
            shared.image?.let { chat.importTranslation(it, image = true) }
            onShareHandled()
        }
    }
    BackHandler(workspace != Workspace.Chat) { workspace = Workspace.Chat }

    Scaffold(
        bottomBar = {
            if (!keyboard) NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
                Workspace.entries.forEachIndexed { index, item ->
                    NavigationBarItem(
                        selected = workspace == item,
                        onClick = { focus.clearFocus(); workspace = item },
                        icon = { WorkspaceIcon(index) },
                        label = { Text(item.label) },
                    )
                }
            }
        },
    ) { insets ->
        Box(
            Modifier.fillMaxSize().padding(insets).consumeWindowInsets(insets).imePadding(),
            contentAlignment = Alignment.TopCenter,
        ) {
            Column(Modifier.widthIn(max = 640.dp).fillMaxSize().padding(horizontal = 20.dp)) {
                Row(
                    Modifier.fillMaxWidth().padding(top = 10.dp, bottom = 14.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (workspace == Workspace.Chat) {
                        Image(
                            painterResource(R.mipmap.ic_launcher), contentDescription = null,
                            modifier = Modifier.size(40.dp).clip(RoundedCornerShape(12.dp)),
                        )
                    }
                    Column(Modifier.weight(1f)) {
                        Text(
                            if (workspace == Workspace.Chat) "天枢" else workspace.label,
                            style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold,
                        )
                        if (workspace == Workspace.Chat) state.modelName?.let { name ->
                            Text(name, style = MaterialTheme.typography.labelMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                    if (workspace == Workspace.Chat) {
                        TextButton(onClick = { showHistory = true }) { Text("历史") }
                        TextButton(
                            onClick = { chat.newChat(); input = "" }, enabled = state.activity == Activity.Idle,
                        ) { Text("新对话") }
                    }
                }
                state.error?.let { ErrorBanner(it, chat::dismissError) }
                when (workspace) {
                    Workspace.Chat -> {
                        val canAsk = state.modelName != null && state.activity == Activity.Idle
                        val ask = {
                            if (canAsk && input.isNotBlank()) {
                                chat.ask(input)
                                input = ""
                                focus.clearFocus()
                            }
                        }
                        if (state.turns.isEmpty()) {
                            EmptyConversation(
                                state, Modifier.weight(1f).fillMaxWidth(),
                                showModels = { workspace = Workspace.Models },
                                showTranslation = { workspace = Workspace.Translate },
                            )
                        } else {
                            ChatTranscript(state, Modifier.weight(1f).fillMaxWidth())
                        }
                        if (state.modelName != null) {
                            Composer(
                                state, input, { input = it.take(12_000) }, canAsk, ask, chat::stop,
                                openAttachment, chat::removeAttachment,
                            )
                        }
                    }
                    Workspace.Translate -> TranslationScreen(chat, openModels = { workspace = Workspace.Models })
                    Workspace.Models -> ModelsScreen(chat, openModel, openProjector) { workspace = Workspace.Translate }
                    Workspace.Settings -> SettingsScreen(state, { showUpdates = true })
                }
            }
        }
    }
    if (showUpdates) UpdateDialog { showUpdates = false }
    if (showHistory) HistoryDialog(
        chat, dismiss = { showHistory = false },
        choose = { selected -> chat.selectConversation(selected); input = ""; showHistory = false },
    )
}

@Composable
private fun EmptyConversation(
    state: ChatState,
    modifier: Modifier,
    showModels: () -> Unit,
    showTranslation: () -> Unit,
) {
    Column(
        modifier.verticalScroll(rememberScrollState()).padding(top = 24.dp, bottom = 12.dp),
        verticalArrangement = Arrangement.spacedBy(15.dp),
    ) {
        Text(
            if (state.modelName == null) "在手机上离线使用" else "聊点什么？",
            style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold,
        )
        if (state.modelName == null) {
            Text(
                "先下载或导入一个模型。下载完成后，聊天和翻译都可以在本机运行。",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Button(onClick = showModels) { Text("浏览模型") }
        }
        Surface(
            onClick = showTranslation, shape = RoundedCornerShape(18.dp),
            color = MaterialTheme.colorScheme.surface,
        ) {
            Row(
                Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                WorkspaceIcon(1)
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Text("离线翻译", fontWeight = FontWeight.SemiBold)
                    Text("文本、选中文字和截图", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Text("›", style = MaterialTheme.typography.titleLarge)
            }
        }
        if (state.activity == Activity.Connecting || state.activity == Activity.Loading) {
            Text(if (state.activity == Activity.Connecting) "正在连接推理引擎" else "正在加载模型",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.labelMedium)
            LinearProgressIndicator(Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun ErrorBanner(message: String, dismiss: () -> Unit) {
    Surface(color = MaterialTheme.colorScheme.errorContainer, shape = RoundedCornerShape(14.dp)) {
        Row(
            Modifier.fillMaxWidth().padding(start = 12.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                message, Modifier.weight(1f).padding(vertical = 8.dp),
                color = MaterialTheme.colorScheme.onErrorContainer, style = MaterialTheme.typography.bodyMedium,
            )
            TextButton(onClick = dismiss) { Text("关闭") }
        }
    }
}

@Composable
private fun ChatTranscript(state: ChatState, modifier: Modifier) {
    val scroll = rememberLazyListState()
    LaunchedEffect(state.turns.size, state.turns.lastOrNull()?.answer) {
        val visible = scroll.layoutInfo.visibleItemsInfo
        if (visible.isEmpty() || (visible.last().index >= state.turns.lastIndex - 1 && !scroll.isScrollInProgress)) {
            scroll.scrollToItem(state.turns.size)
        }
    }
    LazyColumn(
        state = scroll, modifier = modifier, contentPadding = PaddingValues(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        items(state.turns) { turn ->
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(9.dp)) {
                Surface(
                    modifier = Modifier.align(Alignment.End).widthIn(max = 520.dp),
                    shape = RoundedCornerShape(16.dp),
                    color = MaterialTheme.colorScheme.primaryContainer,
                ) {
                    SelectionContainer {
                        Column(Modifier.padding(13.dp)) {
                            turn.attachmentName?.let {
                                Text(it, style = MaterialTheme.typography.labelMedium)
                            }
                            Text(turn.prompt, color = MaterialTheme.colorScheme.onPrimaryContainer)
                        }
                    }
                }
                if (turn.answer.isNotBlank()) {
                    Surface(shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
                        SelectionContainer { Text(turn.answer, Modifier.padding(13.dp)) }
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
    canAsk: Boolean, ask: () -> Unit, stop: () -> Unit,
    openAttachment: () -> Unit, removeAttachment: () -> Unit,
) {
    Column(Modifier.padding(top = 8.dp, bottom = 12.dp)) {
        if (state.projector != null) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = openAttachment, enabled = canAsk) { Text("选择附件") }
                state.attachmentName?.let { name ->
                    Text(
                        name, Modifier.weight(1f), style = MaterialTheme.typography.labelMedium,
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                    )
                    TextButton(onClick = removeAttachment, enabled = canAsk) { Text("移除") }
                }
            }
        }
        state.status?.let {
            Text(it, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(bottom = 6.dp))
        }
        if (state.activity == Activity.Importing) {
            val progress = state.progress
            if (progress == null) LinearProgressIndicator(Modifier.fillMaxWidth())
            else LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                input, onValueChange = onInput, modifier = Modifier.weight(1f),
                label = { Text("输入消息") }, maxLines = 5, enabled = canAsk,
                shape = RoundedCornerShape(16.dp), keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { ask() }),
            )
            if (state.activity == Activity.Generating || state.activity == Activity.Importing) {
                Button(onClick = stop, enabled = !state.stopping) { Text("停止") }
            } else {
                Button(onClick = ask, enabled = canAsk && input.isNotBlank()) { Text("发送") }
            }
        }
    }
}
