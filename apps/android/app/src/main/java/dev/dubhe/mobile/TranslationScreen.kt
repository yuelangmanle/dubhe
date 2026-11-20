package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
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
internal fun TranslationScreen(chat: ChatViewModel, openModels: () -> Unit) {
    val state = chat.state
    val translation = chat.translation
    val active = state.activity == Activity.Translating
    val idle = state.activity == Activity.Idle
    val dedicated = state.models.any { it.catalog?.translation == true }
    val canTranslate = state.modelName != null || dedicated
    val context = LocalContext.current
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
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val twoPanels = maxWidth >= 620.dp
        Column(
            Modifier.verticalScroll(rememberScrollState()).padding(bottom = 16.dp),
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
            }
            if (canTranslate) {
                TranslationControls(
                    state.modelName, dedicated, translation.preferDedicated, active,
                    canRun = idle && translation.source.isNotBlank(), stopping = state.stopping,
                    choose = { dedicatedModel -> chat.updateTranslation { it.copy(preferDedicated = dedicatedModel) } },
                    run = chat::translate, stop = chat::stop,
                )
            } else {
                ModelCard("选择翻译模型", "下载混元翻译，或加载一个对话模型。", Glyph.Model) {
                    TextButton(openModels) { Text("选模型") }
                }
            }
            if (active || state.activity == Activity.Importing && state.downloading == null) {
                TransferProgress(state.progress)
                state.status?.let { Text(it, style = MaterialTheme.typography.labelMedium) }
            }
            if (!twoPanels) output(Modifier.fillMaxWidth(), 128.dp)
        }
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
    result: String, active: Boolean, modifier: Modifier, minHeight: Dp, copy: () -> Unit, save: (TranslationFormat) -> Unit,
) {
    var exportMenu by remember { mutableStateOf(false) }
    val canUse = result.isNotBlank()
    WorkspaceCard(modifier) {
        Column(Modifier.fillMaxWidth().heightIn(min = minHeight).padding(start = 15.dp, end = 6.dp, bottom = 15.dp)) {
            Row(Modifier.fillMaxWidth().heightIn(min = 47.dp), verticalAlignment = Alignment.CenterVertically) {
                SectionTitle("译文", Modifier.weight(1f))
                if (canUse) {
                    IconAction(Glyph.Copy, "复制译文", onClick = copy)
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
    modelName: String?, dedicated: Boolean, preferDedicated: Boolean, active: Boolean,
    canRun: Boolean, stopping: Boolean, choose: (Boolean) -> Unit, run: () -> Unit, stop: () -> Unit,
) {
    var modelMenu by remember { mutableStateOf(false) }
    val selected = if (dedicated && (preferDedicated || modelName == null)) "混元翻译 Hy-MT2" else modelName.orEmpty()
    val canSwitch = dedicated && modelName != null
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f)) {
            if (canSwitch) {
                TextButton({ modelMenu = true }, enabled = !active) {
                    Text(selected, Modifier.weight(1f, fill = false), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    WorkspaceIcon(Glyph.Chevron, Modifier.size(16.dp))
                }
                DropdownMenu(modelMenu, { modelMenu = false }) {
                    DropdownMenuItem(text = { Text("混元翻译 Hy-MT2") }, onClick = {
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
        else Button(run, enabled = canRun) { Text("翻译") }
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
