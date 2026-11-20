package dev.dubhe.mobile

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.util.Locale

@Composable
internal fun WorkspaceIcon(index: Int) {
    val color = MaterialTheme.colorScheme.onSurface
    Canvas(Modifier.size(23.dp)) {
        scale(size.width / 24f, size.height / 24f, pivot = Offset.Zero) {
            val line = Stroke(width = 1.65f, cap = StrokeCap.Round)
            when (index) {
                0 -> {
                    drawPath(Path().apply {
                        moveTo(3f, 10f)
                        quadraticTo(3f, 4f, 9f, 4f)
                        lineTo(15f, 4f)
                        quadraticTo(21f, 4f, 21f, 10f)
                        lineTo(21f, 12f)
                        quadraticTo(21f, 17.5f, 15f, 17.5f)
                        lineTo(10f, 17.5f)
                        lineTo(5f, 21f)
                        lineTo(5f, 16f)
                        quadraticTo(3f, 14f, 3f, 12f)
                        close()
                    }, color, style = line)
                    for (x in listOf(8f, 12f, 16f)) drawCircle(color, 0.85f, Offset(x, 10.8f))
                }
                1 -> {
                    drawCircle(color, radius = 8.5f, center = Offset(12f, 12f), style = line)
                    drawOval(color, Offset(8.5f, 3.5f), Size(7f, 17f), style = line)
                    drawLine(color, Offset(4f, 12f), Offset(20f, 12f), strokeWidth = 1.55f)
                }
                2 -> {
                    drawPath(Path().apply {
                        moveTo(12f, 2.8f)
                        lineTo(20.8f, 7.8f)
                        lineTo(20.8f, 16.5f)
                        lineTo(12f, 21.3f)
                        lineTo(3.2f, 16.5f)
                        lineTo(3.2f, 7.8f)
                        close()
                        moveTo(3.2f, 7.8f)
                        lineTo(12f, 12.8f)
                        lineTo(20.8f, 7.8f)
                        moveTo(12f, 12.8f)
                        lineTo(12f, 21.3f)
                    }, color, style = line)
                }
                else -> {
                    for ((y, x) in listOf(6f to 8f, 12f to 15f, 18f to 11f)) {
                        drawLine(color, Offset(3.5f, y), Offset(20.5f, y), strokeWidth = 1.6f, cap = StrokeCap.Round)
                        drawRoundRect(color, Offset(x - 1.8f, y - 2.8f), Size(3.6f, 5.6f), CornerRadius(1.4f))
                    }
                }
            }
        }
    }
}

@Composable
internal fun ModelsScreen(chat: ChatViewModel, importModel: () -> Unit, importProjector: () -> Unit, openTranslation: () -> Unit) {
    val state = chat.state
    val idle = state.activity == Activity.Idle
    var deleting by remember { mutableStateOf<LocalModel?>(null) }
    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            SectionTitle("我的模型", Modifier.weight(1f))
            OutlinedButton(importModel, enabled = idle) { Text("导入 GGUF") }
        }
        if (state.models.isEmpty()) {
            Text("从商店下载，或导入已有的 GGUF 文件。", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        for (model in state.models) {
            val translationModel = model.catalog?.translation == true
            val active = state.activeModel?.file == model.file
            ModelCard(
                model.catalog?.name ?: model.name,
                if (active) "正在使用" else if (translationModel) "专用翻译" else "本地 GGUF",
            ) {
                TextButton(
                    onClick = { if (translationModel) openTranslation() else chat.selectModel(model) },
                    enabled = !active && (translationModel || idle),
                ) { Text(if (translationModel) "去翻译" else if (active) "已加载" else "加载") }
                TextButton(onClick = { deleting = model }, enabled = idle) { Text("删除") }
            }
        }
        if (state.activeModel != null) {
            ModelCard("多模态组件", state.projectorName ?: "配合对话模型读取图片、视频或音频") {
                TextButton(importProjector, enabled = idle) { Text(if (state.projectorName == null) "导入" else "更换") }
            }
        }
        if (state.activity == Activity.Importing && state.downloading == null) {
            ModelCard(state.status.orEmpty(), null) { TextButton(chat::stop, enabled = !state.stopping) { Text("取消") } }
            TransferProgress(state.progress)
        } else if (state.activity == Activity.Loading) {
            TransferProgress(null)
        }
        HorizontalDivider(Modifier.padding(top = 6.dp))
        SectionTitle("模型商店")
        for (item in mobileCatalog) {
            val installed = state.models.any { it.catalog?.file == item.file }
            val downloading = state.downloading == item.file
            val size = if (item.bytes < 1_000_000_000) {
                String.format(Locale.CHINA, "%.0f MB", item.bytes / 1_000_000.0)
            } else {
                String.format(Locale.CHINA, "%.2f GB", item.bytes / 1_000_000_000.0)
            }
            ModelCard(item.name, item.description + " · " + size) {
                when {
                    installed -> Text("已下载", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                    downloading -> TextButton(chat::stop, enabled = !state.stopping) { Text("取消") }
                    else -> TextButton({ chat.downloadModel(item) }, enabled = idle) {
                        Text(if (item.file in state.partialDownloads) "继续下载" else "下载")
                    }
                }
            }
            if (downloading) {
                TransferProgress(state.progress)
                val progress = state.progress
                Text(
                    if (state.stopping) "正在取消" else if (progress == 1f) "正在校验模型" else
                        progress?.let { "已下载 " + (it * 100).toInt() + "%" } ?: "正在连接下载源",
                    style = MaterialTheme.typography.labelMedium,
                )
            }
        }
    }
    deleting?.let { model ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("删除模型？") },
            text = { Text("将移除这台手机里的 " + (model.catalog?.name ?: model.name) + "。") },
            confirmButton = {
                Button({ chat.deleteModel(model); deleting = null }) { Text("删除") }
            },
            dismissButton = { TextButton({ deleting = null }) { Text("取消") } },
        )
    }
}

@Composable
internal fun ModelCard(title: String, description: String?, actions: @Composable () -> Unit) {
    Surface(shape = RoundedCornerShape(17.dp), color = MaterialTheme.colorScheme.surface) {
        Row(
            Modifier.fillMaxWidth().padding(start = 15.dp, end = 9.dp, top = 10.dp, bottom = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(title, fontWeight = FontWeight.SemiBold, maxLines = 2, overflow = TextOverflow.Ellipsis)
                description?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            actions()
        }
    }
}

@Composable
internal fun TransferProgress(progress: Float?) {
    if (progress == null) LinearProgressIndicator(Modifier.fillMaxWidth())
    else LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
}

@Composable
internal fun SectionTitle(text: String, modifier: Modifier = Modifier) {
    Text(text, modifier, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
}

@Composable
internal fun SettingsScreen(state: ChatState, checkUpdates: () -> Unit) {
    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        SectionTitle("版本")
        ModelCard("Dubhe " + BuildConfig.VERSION_NAME, null) {
            TextButton(checkUpdates) { Text("更新与日志") }
        }
        SectionTitle("运行状态")
        ModelCard(state.modelName ?: "未加载对话模型", state.modelDetail) {}
    }
}

@Composable
internal fun HistoryDialog(chat: ChatViewModel, dismiss: () -> Unit, choose: (Conversation) -> Unit) {
    val history = chat.conversations
    val idle = chat.state.activity == Activity.Idle
    var deleting by remember { mutableStateOf<Conversation?>(null) }
    AlertDialog(
        onDismissRequest = dismiss,
        title = { Text("对话历史") },
        text = {
            Column(
                Modifier.fillMaxWidth().heightIn(max = 410.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                if (history.items.isEmpty()) Text("发送第一条消息后，对话会自动保存。")
                for (conversation in history.items) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton({ choose(conversation) }, Modifier.weight(1f), enabled = idle) {
                            Text(conversation.title, maxLines = 2, overflow = TextOverflow.Ellipsis)
                        }
                        TextButton({ deleting = conversation }, enabled = idle) { Text("删除") }
                    }
                    HorizontalDivider()
                }
            }
        },
        confirmButton = { TextButton(dismiss) { Text("关闭") } },
    )
    deleting?.let { conversation ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("删除这段对话？") },
            text = { Text(conversation.title) },
            confirmButton = { Button({ chat.deleteConversation(conversation); deleting = null }) { Text("删除") } },
            dismissButton = { TextButton({ deleting = null }) { Text("取消") } },
        )
    }
}
