package dev.dubhe.mobile

import android.content.ActivityNotFoundException
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
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    private val chat by viewModels<ChatViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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
                ChatScreen(
                    chat,
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

private val lightPalette = lightColorScheme(
    primary = Color(0xFF086353), onPrimary = Color.White,
    primaryContainer = Color(0xFFD0F0E5), onPrimaryContainer = Color(0xFF07372D),
    background = Color(0xFFF4F7F6), onBackground = Color(0xFF152822),
    surface = Color.White, onSurface = Color(0xFF152822),
    surfaceVariant = Color(0xFFE2EBE7), onSurfaceVariant = Color(0xFF465953),
)

private val darkPalette = darkColorScheme(
    primary = Color(0xFF92E0CD), onPrimary = Color(0xFF0C362C),
    primaryContainer = Color(0xFF1F4940), onPrimaryContainer = Color(0xFFE1FFF6),
    background = Color(0xFF0F1917), onBackground = Color(0xFFE1F0EB),
    surface = Color(0xFF172522), onSurface = Color(0xFFE1F0EB),
    surfaceVariant = Color(0xFF293B36), onSurfaceVariant = Color(0xFFC0D5CC),
)

@Composable
private fun ChatScreen(
    chat: ChatViewModel,
    openModel: () -> Unit,
    openProjector: () -> Unit,
    openAttachment: () -> Unit,
) {
    val state = chat.state
    var input by rememberSaveable { mutableStateOf("") }
    var showUpdates by rememberSaveable { mutableStateOf(false) }
    val focus = LocalFocusManager.current
    val canAsk = state.modelName != null && state.activity == Activity.Idle
    val ask = {
        if (canAsk && input.isNotBlank()) {
            chat.ask(input)
            input = ""
            focus.clearFocus()
        }
    }
    Scaffold { insets ->
        Box(
            Modifier.fillMaxSize().padding(insets).consumeWindowInsets(insets).imePadding(),
            contentAlignment = Alignment.TopCenter,
        ) {
            Column(Modifier.widthIn(max = 760.dp).fillMaxSize().padding(horizontal = 16.dp)) {
                ChatHeader(state, chat::newChat, openModel, openProjector, { showUpdates = true })
                state.error?.let { ErrorBanner(it, chat::dismissError) }
                if (state.activity == Activity.Loading && state.modelName != null) {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                if (state.turns.isEmpty()) {
                    Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                        EmptyConversation(state, openModel, chat::stop)
                    }
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
        }
    }
    if (showUpdates) UpdateDialog { showUpdates = false }
}

@Composable
private fun ChatHeader(
    state: ChatState, newChat: () -> Unit, openModel: () -> Unit, openProjector: () -> Unit,
    openUpdates: () -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Image(
            painterResource(R.mipmap.ic_launcher), contentDescription = null,
            Modifier.size(39.dp).clip(RoundedCornerShape(11.dp)),
        )
        Column(Modifier.weight(1f)) {
            Text("Dubhe", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            state.modelName?.let { name ->
                Text(name, style = MaterialTheme.typography.bodySmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
        if (state.modelName != null) {
            TextButton(onClick = openModel, enabled = state.activity == Activity.Idle) { Text("换模型") }
            TextButton(onClick = newChat, enabled = state.activity == Activity.Idle && state.turns.isNotEmpty()) {
                Text("新对话")
            }
        }
        TextButton(onClick = openUpdates) { Text("更新") }
    }
    state.modelDetail?.let { detail ->
        Text(
            detail, Modifier.padding(bottom = 6.dp), style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
    if (state.modelName != null) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = openProjector, enabled = state.activity == Activity.Idle) {
                Text(if (state.projectorName == null) "导入多模态组件" else "更换组件")
            }
            state.projectorName?.let { name ->
                Text(name, style = MaterialTheme.typography.labelMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

@Composable
private fun EmptyConversation(state: ChatState, openModel: () -> Unit, stop: () -> Unit) {
    Column(
        Modifier.widthIn(max = 320.dp).fillMaxWidth().padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(13.dp),
        horizontalAlignment = Alignment.Start,
    ) {
        Text(
            if (state.modelName == null) "在手机上运行模型" else "开始一段对话",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
        )
        when (state.activity) {
            Activity.Connecting -> Text("正在连接推理引擎。")
            Activity.Importing -> if (state.modelName == null) {
                Text(state.status.orEmpty())
                val progress = state.progress
                if (progress == null) LinearProgressIndicator(Modifier.fillMaxWidth())
                else LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                TextButton(onClick = stop, enabled = !state.stopping) { Text("取消导入") }
            }
            Activity.Loading -> if (state.modelName == null) {
                Text("正在加载模型…")
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            else -> if (state.modelName == null) {
                Text("选一个本地 GGUF 文件，导入后即可离线聊天。")
                Button(onClick = openModel, enabled = state.connected) { Text("选择 GGUF 文件") }
            }
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
