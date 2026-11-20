package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
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
    var exportMenu by remember { mutableStateOf(false) }
    var modelMenu by remember { mutableStateOf(false) }
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
    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surface) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 3.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                LanguagePicker(translation.from, true, !active, Modifier.weight(1f)) { language ->
                    chat.updateTranslation { it.copy(from = language, result = "") }
                }
                TextButton(
                    onClick = {
                        chat.updateTranslation {
                            it.copy(from = it.to, to = it.from, source = it.result.ifBlank { it.source }, result = "")
                        }
                    },
                    enabled = !active && translation.from != Language.Auto,
                ) { Text("⇄") }
                LanguagePicker(translation.to, false, !active, Modifier.weight(1f)) { language ->
                    chat.updateTranslation { it.copy(to = language, result = "") }
                }
            }
        }
        OutlinedTextField(
            value = translation.source,
            onValueChange = { text -> chat.setTranslationSource(text) },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("原文") },
            placeholder = { Text("输入或粘贴要翻译的文字") },
            supportingText = translation.inputLabel?.let { label -> { Text(label) } },
            minLines = 4,
            maxLines = 10,
            enabled = !active && state.activity != Activity.Importing,
            shape = RoundedCornerShape(16.dp),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton({ launchPicker { image.launch(arrayOf("image/*")) } }, enabled = idle) { Text("图片 / 截图") }
            OutlinedButton({
                launchPicker { document.launch(arrayOf("text/*", "application/markdown", "application/octet-stream")) }
            }, enabled = idle) { Text("打开文件") }
        }
        if (!canTranslate) {
            ModelCard("需要一个翻译模型", "下载混元翻译，或使用已加载的对话模型。") {
                TextButton(openModels) { Text("选模型") }
            }
        } else {
            val picked = if (dedicated && (translation.preferDedicated || state.modelName == null)) {
                "混元翻译 Hy-MT2"
            } else {
                state.modelName.orEmpty()
            }
            Box {
                OutlinedButton({ modelMenu = true }, enabled = !active && dedicated && state.modelName != null) {
                    Text(picked, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                DropdownMenu(modelMenu, { modelMenu = false }) {
                    DropdownMenuItem(text = { Text("混元翻译 Hy-MT2") }, onClick = {
                        chat.updateTranslation { it.copy(preferDedicated = true) }
                        modelMenu = false
                    })
                    DropdownMenuItem(text = { Text(state.modelName.orEmpty()) }, onClick = {
                        chat.updateTranslation { it.copy(preferDedicated = false) }
                        modelMenu = false
                    })
                }
            }
        }
        if (active || state.activity == Activity.Importing && state.downloading == null) {
            TransferProgress(state.progress)
            state.status?.let { Text(it, style = MaterialTheme.typography.labelMedium) }
        }
        if (active) Button(chat::stop, Modifier.fillMaxWidth(), enabled = !state.stopping) { Text("停止翻译") }
        else Button(chat::translate, Modifier.fillMaxWidth(),
            enabled = idle && canTranslate && translation.source.isNotBlank(),
        ) { Text("翻译") }

        if (translation.result.isNotBlank()) {
            Surface(shape = RoundedCornerShape(17.dp), color = MaterialTheme.colorScheme.surface) {
                Column(Modifier.fillMaxWidth().padding(15.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SectionTitle("译文")
                    SelectionContainer { Text(translation.result, style = MaterialTheme.typography.bodyLarge) }
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        TextButton({
                            context.getSystemService(ClipboardManager::class.java)
                                ?.setPrimaryClip(ClipData.newPlainText("译文", translation.result))
                        }) { Text("复制") }
                        Box(contentAlignment = Alignment.CenterStart) {
                            TextButton({ exportMenu = true }, enabled = !active) { Text("导出") }
                            DropdownMenu(exportMenu, { exportMenu = false }) {
                                for (format in TranslationFormat.entries) {
                                    DropdownMenuItem(text = { Text(format.label) }, onClick = {
                                        exportMenu = false
                                        exportName = format.name
                                        exportText = translation.result
                                        launchPicker {
                                            export.launch(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                                                addCategory(Intent.CATEGORY_OPENABLE)
                                                type = format.mime
                                                putExtra(Intent.EXTRA_TITLE, "Dubhe 译文." + format.extension)
                                            })
                                        }
                                    })
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun LanguagePicker(
    language: Language,
    allowAuto: Boolean,
    enabled: Boolean,
    modifier: Modifier,
    select: (Language) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier, contentAlignment = Alignment.Center) {
        TextButton({ expanded = true }, enabled = enabled) { Text(language.label) }
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
