package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

@Composable
internal fun TranslationScreen(chat: ChatViewModel, openModels: () -> Unit, openTasks: () -> Unit) {
    val state = chat.state
    val translation = chat.translation
    val active = state.activity == Activity.Translating
    val idle = state.activity == Activity.Idle
    val dedicated = state.models.firstOrNull { it.catalog?.translation == true }
    val canTranslate = state.modelName != null || dedicated != null
    val task = chat.translationTask?.takeUnless { it.finished }
    val selectedTaskModel = task?.let { current ->
        if (current.dedicated) state.models.firstOrNull { it.translationKey == current.modelKey }
        else state.activeModel?.takeIf { it.translationKey == current.modelKey }
    }
    val context = LocalContext.current
    var showGlossary by rememberSaveable { mutableStateOf(false) }
    var exportName by rememberSaveable { mutableStateOf<String?>(null) }
    var exportText by rememberSaveable { mutableStateOf("") }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { response ->
        val name = exportName
        val destination = response.data?.data
        if (response.resultCode == android.app.Activity.RESULT_OK && name != null && destination != null) {
            chat.exportTranslation(destination, TranslationFormat.valueOf(name), exportText)
        }
        exportName = null
        exportText = ""
    }
    val image = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) chat.importTranslation(uri, image = true)
    }
    val document = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) chat.importTranslation(uri, image = false)
    }
    fun launchPicker(action: () -> Unit) {
        try {
            action()
        } catch (_: ActivityNotFoundException) {
            chat.showError("这台设备没有可用的文件选择器。")
        }
    }
    val source: @Composable (Modifier, Int) -> Unit = { modifier, minLines ->
        TranslationSource(
            translation = translation, editable = !active && state.activity != Activity.Importing, idle = idle,
            modifier = modifier, minLines = minLines, edit = { chat.setTranslationSource(it) },
            openImage = { launchPicker { image.launch(arrayOf("image/*")) } },
            openFile = {
                launchPicker { document.launch(arrayOf("text/*", "application/markdown", "application/octet-stream")) }
            },
        )
    }
    val output: @Composable (Modifier, Dp) -> Unit = { modifier, minHeight ->
        TranslationResult(
            translation.result, active, modifier, minHeight,
            copy = {
                context.getSystemService(ClipboardManager::class.java)
                    ?.setPrimaryClip(ClipData.newPlainText("译文", translation.result))
            },
            share = {
                launchPicker {
                    context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        putExtra(Intent.EXTRA_TEXT, translation.result)
                    }, "分享译文"))
                }
            },
            save = { format ->
                launchPicker {
                    exportName = format.name
                    exportText = translation.result
                    export.launch(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = format.mime
                        putExtra(Intent.EXTRA_TITLE, "Dubhe 译文." + format.extension)
                    })
                }
            },
        )
    }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val twoPanels = maxWidth >= 620.dp
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            WorkspaceCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(horizontal = 8.dp, vertical = 4.dp)) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        TextButton(openTasks) { Text("最近任务") }
                        TextButton({ showGlossary = true }, enabled = idle) { Text("常用术语") }
                        Spacer(Modifier.weight(1f))
                        TextButton(chat::newTranslation, enabled = idle) { Text("新翻译") }
                    }
                    if (task != null && selectedTaskModel == null && !active) {
                        Text("继续翻译需要原来的模型 ${task.modelName}。", Modifier.padding(horizontal = 8.dp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                        Row {
                            TextButton(openModels) { Text("管理模型") }
                            TextButton(chat::changeTranslationModel, enabled = idle) { Text("换模型重新翻译") }
                        }
                    } else if (canTranslate || active) {
                        TranslationControls(
                            state.modelName, dedicated?.name, translation.preferDedicated, active,
                            canRun = idle && translation.source.isNotBlank(), stopping = state.stopping,
                            action = when {
                                task != null -> "继续翻译"
                                chat.translationTask?.finished == true -> "重新翻译"
                                else -> "翻译"
                            },
                            lockedModel = task?.modelName,
                            choose = { dedicatedModel -> chat.updateTranslation { it.copy(preferDedicated = dedicatedModel) } },
                            run = chat::translate, stop = chat::stop,
                        )
                    } else {
                        val recommended = mobileCatalog.firstOrNull { it.translation }
                        if (recommended != null) {
                            val size = Formatter.formatShortFileSize(context, recommended.bytes)
                            Text(
                                "${recommended.name} · $size", Modifier.padding(horizontal = 8.dp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall,
                            )
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                TextButton({ chat.downloadModel(recommended) }, enabled = idle) {
                                    Text(if (recommended.file in state.partialDownloads) "继续下载" else "下载翻译模型")
                                }
                                TextButton(openModels) { Text("选用对话模型") }
                                if (state.downloading == recommended.file) {
                                    TextButton(chat::stop, enabled = !state.stopping) { Text("暂停") }
                                }
                            }
                        } else {
                            TextButton(openModels) { Text("选择翻译模型") }
                        }
                    }
                    if (active || state.activity == Activity.Importing) {
                        TransferProgress(state.progress)
                        state.status?.let { Text(it, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(5.dp)) }
                    } else if (task != null && task.completed > 0) {
                        Text("已保存 ${task.completed} / ${task.total} 段", Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelMedium)
                    }
                }
            }
            Column(
                Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(bottom = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                WorkspaceCard(Modifier.fillMaxWidth()) {
                    Row(
                        Modifier.fillMaxWidth().padding(horizontal = 6.dp, vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        LanguagePicker(translation.from, true, !active, Modifier.weight(1f)) { language ->
                            chat.updateTranslation { it.copy(from = language, result = "") }
                        }
                        IconAction(Glyph.Swap, "交换语言", !active && translation.from != Language.Auto) {
                            chat.updateTranslation {
                                it.copy(from = it.to, to = it.from, source = it.result.ifBlank { it.source }, result = "")
                            }
                        }
                        LanguagePicker(translation.to, false, !active, Modifier.weight(1f)) { language ->
                            chat.updateTranslation { it.copy(to = language, result = "") }
                        }
                    }
                }
                if (twoPanels) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        source(Modifier.weight(1f), 3)
                        output(Modifier.weight(1f), 184.dp)
                    }
                } else {
                    source(Modifier.fillMaxWidth(), 5)
                    output(Modifier.fillMaxWidth(), 128.dp)
                }
            }
        }
    }
    if (showGlossary) {
        var terms by rememberSaveable { mutableStateOf(translation.glossary) }
        var error by rememberSaveable { mutableStateOf<String?>(null) }
        AlertDialog(
            onDismissRequest = { showGlossary = false },
            title = { Text("常用术语") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("每行填一个「原文 = 译法」。保存后可用于新的翻译。")
                    OutlinedTextField(
                        terms, { terms = it.take(1200); error = null }, Modifier.fillMaxWidth(),
                        minLines = 4, maxLines = 8, label = { Text("术语表") },
                        placeholder = { Text("reservoir = 水库") },
                        isError = error != null,
                        supportingText = error?.let { message -> { Text(message) } },
                    )
                    if (task != null) Text("修改会为这份原文开始一条新任务。原任务仍可从「最近任务」续做。")
                }
            },
            confirmButton = {
                Button({
                    error = chat.saveTranslationGlossary(terms)
                    if (error == null) showGlossary = false
                }) { Text("保存") }
            },
            dismissButton = { TextButton({ showGlossary = false }) { Text("取消") } },
        )
    }
}

@Composable
private fun TranslationSource(
    translation: TranslationState, editable: Boolean, idle: Boolean, modifier: Modifier, minLines: Int,
    edit: (String) -> Unit, openImage: () -> Unit, openFile: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    WorkspaceCard(modifier) {
        Column {
            Column(Modifier.fillMaxWidth().padding(15.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("原文")
                BasicTextField(
                    value = translation.source, onValueChange = edit, enabled = editable,
                    minLines = minLines, maxLines = 12,
                    modifier = Modifier.fillMaxWidth().semantics { contentDescription = "原文" },
                    textStyle = MaterialTheme.typography.bodyLarge.copy(color = colors.onSurface),
                    cursorBrush = SolidColor(colors.primary),
                    decorationBox = { field ->
                        Box {
                            if (translation.source.isEmpty()) {
                                Text(
                                    "输入或粘贴要翻译的文字",
                                    style = MaterialTheme.typography.bodyLarge, color = colors.onSurfaceVariant,
                                )
                            }
                            field()
                        }
                    },
                )
                translation.inputLabel?.let { label ->
                    Text(label, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                }
            }
            HorizontalDivider(color = colors.outlineVariant)
            Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                TextButton(openImage, enabled = idle) {
                    WorkspaceIcon(Glyph.Image, Modifier.size(17.dp))
                    Spacer(Modifier.width(5.dp))
                    Text("图片 / 截图")
                }
                TextButton(openFile, enabled = idle) {
                    WorkspaceIcon(Glyph.File, Modifier.size(17.dp))
                    Spacer(Modifier.width(5.dp))
                    Text("打开文件")
                }
            }
        }
    }
}

@Composable
private fun TranslationResult(
    result: String, active: Boolean, modifier: Modifier, minHeight: Dp,
    copy: () -> Unit, share: () -> Unit, save: (TranslationFormat) -> Unit,
) {
    var exportMenu by remember { mutableStateOf(false) }
    val canUse = result.isNotBlank()
    WorkspaceCard(modifier) {
        Column(Modifier.fillMaxWidth().heightIn(min = minHeight).padding(start = 15.dp, end = 6.dp, bottom = 15.dp)) {
            Row(Modifier.fillMaxWidth().heightIn(min = 47.dp), verticalAlignment = Alignment.CenterVertically) {
                SectionTitle("译文", Modifier.weight(1f))
                if (canUse) {
                    IconAction(Glyph.Copy, "复制译文", onClick = copy)
                    TextButton(share, enabled = !active) { Text("分享") }
                    Box {
                        IconAction(Glyph.Download, "导出译文", enabled = !active) { exportMenu = true }
                        DropdownMenu(exportMenu, { exportMenu = false }) {
                            for (format in TranslationFormat.entries) {
                                DropdownMenuItem(text = { Text(format.label) }, onClick = {
                                    exportMenu = false
                                    save(format)
                                })
                            }
                        }
                    }
                }
            }
            if (canUse) {
                SelectionContainer {
                    Text(result, Modifier.padding(end = 9.dp), style = MaterialTheme.typography.bodyLarge)
                }
            } else {
                Text(
                    if (active) "正在翻译…" else "译文会显示在这里",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun TranslationControls(
    modelName: String?, dedicatedName: String?, preferDedicated: Boolean, active: Boolean,
    canRun: Boolean, stopping: Boolean, action: String, lockedModel: String?,
    choose: (Boolean) -> Unit, run: () -> Unit, stop: () -> Unit,
) {
    var modelMenu by remember { mutableStateOf(false) }
    val selected = lockedModel ?: if (preferDedicated) (dedicatedName ?: modelName).orEmpty()
        else (modelName ?: dedicatedName).orEmpty()
    val canSwitch = lockedModel == null && dedicatedName != null && modelName != null
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f)) {
            if (canSwitch) {
                TextButton({ modelMenu = true }, enabled = !active) {
                    Text(selected, Modifier.weight(1f, fill = false), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    WorkspaceIcon(Glyph.Chevron, Modifier.size(16.dp))
                }
                DropdownMenu(modelMenu, { modelMenu = false }) {
                    DropdownMenuItem(text = { Text(dedicatedName.orEmpty()) }, onClick = {
                        choose(true)
                        modelMenu = false
                    })
                    DropdownMenuItem(text = { Text(modelName.orEmpty()) }, onClick = {
                        choose(false)
                        modelMenu = false
                    })
                }
            } else {
                Text(
                    selected, Modifier.padding(start = 4.dp), style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis,
                )
            }
        }
        Spacer(Modifier.width(8.dp))
        if (active) Button(stop, enabled = !stopping) { Text("停止翻译") }
        else Button(run, enabled = canRun) { Text(action) }
    }
}

@Composable
private fun LanguagePicker(
    language: Language, allowAuto: Boolean, enabled: Boolean, modifier: Modifier, select: (Language) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier, contentAlignment = Alignment.Center) {
        TextButton({ expanded = true }, enabled = enabled) {
            Text(language.label)
            Spacer(Modifier.width(3.dp))
            WorkspaceIcon(Glyph.Chevron, Modifier.size(16.dp))
        }
        DropdownMenu(expanded, { expanded = false }) {
            Language.entries.filter { allowAuto || it != Language.Auto }.forEach { choice ->
                DropdownMenuItem(text = { Text(choice.label) }, onClick = {
                    expanded = false
                    select(choice)
                })
            }
        }
    }
}
