package dev.dubhe.mobile

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import android.text.format.Formatter
import java.util.Locale

private enum class StoreCategory(val label: String) {
    All("全部"), Chat("对话"), Media("多模态"), Translate("翻译");

    fun includes(item: CatalogModel) = when (this) {
        All -> true
        Chat -> !item.translation && item.modality == null
        Media -> item.modality != null
        Translate -> item.translation
    }
}

@Composable
internal fun ModelsScreen(
    chat: ChatViewModel, importModel: () -> Unit, importProjector: () -> Unit,
    openTranslation: () -> Unit, openChat: () -> Unit,
    openShop: Boolean = false, onShopOpened: () -> Unit = {},
) {
    val state = chat.state
    val idle = state.activity == Activity.Idle
    val links = LocalUriHandler.current
    val context = LocalContext.current
    val freeMemory = availableMemory()
    var selectedShop by rememberSaveable { mutableStateOf<Boolean?>(null) }
    var category by rememberSaveable { mutableStateOf(StoreCategory.All) }
    val shop = openShop || (selectedShop ?: state.models.isEmpty())
    LaunchedEffect(openShop) {
        if (openShop) {
            selectedShop = true
            category = StoreCategory.Media
            onShopOpened()
        }
    }
    var deleting by remember { mutableStateOf<LocalModel?>(null) }

    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(11.dp),
    ) {
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            SegmentedButton(
                selected = !shop, onClick = { selectedShop = false },
                shape = SegmentedButtonDefaults.itemShape(0, 2),
            ) { Text("我的模型") }
            SegmentedButton(
                selected = shop, onClick = { selectedShop = true },
                shape = SegmentedButtonDefaults.itemShape(1, 2),
            ) { Text("模型商店") }
        }
        if (state.activity == Activity.Importing && state.downloading == null) {
            ModelCard(state.status ?: "正在导入", null, Glyph.File) {
                TextButton(chat::stop, enabled = !state.stopping) { Text("取消") }
            }
            TransferProgress(state.progress)
        } else if (state.activity == Activity.Loading) {
            TransferProgress(null)
        }

        if (shop) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "下载后在本机使用", Modifier.weight(1f),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                TextButton(onClick = {
                    try {
                        links.openUri("https://github.com/yuelangmanle/dubhe/blob/main/docs/model-downloads.md")
                    } catch (_: IllegalArgumentException) {
                        chat.showError("无法打开模型下载来源。")
                    }
                }) { Text("下载来源") }
            }
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (option in StoreCategory.entries) {
                    FilterChip(selected = category == option, onClick = { category = option }, label = { Text(option.label) })
                }
            }
            for (item in mobileCatalog.filter { category.includes(it) || state.downloading == it.file }) {
                val installed = state.models.firstOrNull { it.file.name == item.file }
                val downloading = state.downloading == item.file
                val size = if (item.totalBytes < 1_000_000_000) {
                    String.format(Locale.CHINA, "%.0f MB", item.totalBytes / 1_000_000.0)
                } else {
                    String.format(Locale.CHINA, "%.2f GB", item.totalBytes / 1_000_000_000.0)
                }
                val fit = if (freeMemory > 0 && item.totalBytes >= freeMemory) " · 当前内存较紧" else ""
                ModelCard(item.name, item.description + " · " + size + fit, item.glyph()) {
                    when {
                        installed != null -> TextButton({
                            if (item.translation) openTranslation()
                            else {
                                chat.selectModel(installed)
                                openChat()
                            }
                        }, enabled = idle) { Text(if (item.translation) "去翻译" else "去对话") }
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
            if (freeMemory > 0) Text(
                "此刻可用内存 ${Formatter.formatShortFileSize(context, freeMemory)}。运行模型还需要额外内存。",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall,
            )
        } else {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "保存在此设备", Modifier.weight(1f),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                TextButton(importModel, enabled = idle) { Text("导入 GGUF") }
            }
            if (state.models.isEmpty()) {
                ModelCard("还没有本地模型", "从模型商店下载，或导入已有的 GGUF 文件。", Glyph.Model) {
                    TextButton({ selectedShop = true }) { Text("浏览商店") }
                }
            }
            for (model in state.models) {
                val translationModel = model.catalog?.translation == true
                val active = state.activeModel?.file == model.file
                val bundle = model.bundledCatalog
                val size = Formatter.formatShortFileSize(context, bundle?.totalBytes ?: model.file.length())
                val description = when {
                    active -> "正在使用"
                    bundle?.modality == CatalogModality.IMAGE -> "含视觉组件"
                    bundle?.modality == CatalogModality.AUDIO -> "含音频组件"
                    translationModel -> "专用翻译"
                    else -> "本地 GGUF"
                }
                ModelCard(
                    model.catalog?.name ?: model.name,
                    description + " · " + size,
                    model.catalog.glyph(),
                ) {
                    TextButton(
                        onClick = { if (translationModel) openTranslation() else chat.selectModel(model) },
                        enabled = !active && (translationModel || idle),
                    ) { Text(if (translationModel) "去翻译" else if (active) "已加载" else "加载") }
                    TextButton(onClick = { deleting = model }, enabled = idle) { Text("删除") }
                }
            }
            val activeModel = state.activeModel
            if (activeModel != null) {
                val bundleProjector = activeModel.bundledCatalog?.projector
                ModelCard("多模态组件", state.projectorName ?: "为对话模型接入图片、视频或音频", Glyph.Attachment) {
                    if (bundleProjector == null) {
                        TextButton(importProjector, enabled = idle) { Text(if (state.projectorName == null) "导入" else "更换") }
                    }
                }
            }
        }
    }
    deleting?.let { model ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("删除模型？") },
            text = {
                val bundle = model.bundledCatalog
                Text("将移除这台手机里的 " + (model.catalog?.name ?: model.name) +
                    (if (bundle != null) " 及其配套组件。" else "。"))
            },
            confirmButton = { Button({ chat.deleteModel(model); deleting = null }) { Text("删除") } },
            dismissButton = { TextButton({ deleting = null }) { Text("取消") } },
        )
    }
}

private fun CatalogModel?.glyph(): Glyph = when {
    this?.translation == true -> Glyph.Translate
    this?.modality == CatalogModality.IMAGE -> Glyph.Image
    this?.modality == CatalogModality.AUDIO -> Glyph.Attachment
    else -> Glyph.Model
}

@Composable
internal fun ModelCard(title: String, description: String?, glyph: Glyph, actions: @Composable () -> Unit = {}) {
    val colors = MaterialTheme.colorScheme
    WorkspaceCard(Modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth().padding(start = 12.dp, end = 5.dp, top = 10.dp, bottom = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier.size(34.dp).clip(MaterialTheme.shapes.small).background(colors.secondaryContainer),
                contentAlignment = Alignment.Center,
            ) { WorkspaceIcon(glyph, Modifier.size(19.dp), colors.onSecondaryContainer) }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
                description?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) { actions() }
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
internal fun SettingsScreen(state: ChatState, checkUpdates: () -> Unit, showChangelog: () -> Unit, openTasks: () -> Unit) {
    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 20.dp),
        verticalArrangement = Arrangement.spacedBy(13.dp),
    ) {
        WorkspaceCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                    DubheMark(38.dp)
                    Column(Modifier.weight(1f)) {
                        Text("天枢 Dubhe", style = MaterialTheme.typography.titleMedium)
                        Text("版本 ${BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
                    TextButton(showChangelog) { Text("更新日志") }
                    Button(checkUpdates) { Text("检查更新") }
                }
            }
        }
        ModelCard("任务与存储", "最近翻译、扫描结果和模型占用", Glyph.History) {
            TextButton(openTasks) { Text("打开") }
        }
        SectionTitle("运行状态", Modifier.padding(start = 3.dp, top = 6.dp))
        ModelCard(
            state.modelName ?: "未加载对话模型",
            state.modelDetail ?: if (state.connected) "本地引擎已连接" else "正在连接本地引擎",
            Glyph.Model,
        )
    }
}

@Composable
internal fun ChatModelDialog(
    models: List<LocalModel>, active: LocalModel?, enabled: Boolean,
    dismiss: () -> Unit, choose: (LocalModel) -> Unit, manage: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = dismiss,
        title = { Text("选择对话模型") },
        text = {
            Column(
                Modifier.fillMaxWidth().heightIn(max = 360.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (models.isEmpty()) Text("当前没有可加载的对话模型。")
                for (model in models) {
                    val selected = active?.file == model.file
                    ModelCard(model.catalog?.name ?: model.name, if (selected) "本机运行" else null, Glyph.Model) {
                        TextButton({ choose(model) }, enabled = enabled && !selected) {
                            Text(if (selected) "已加载" else "加载")
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(manage) { Text("管理模型") } },
        dismissButton = { TextButton(dismiss) { Text("关闭") } },
    )
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
