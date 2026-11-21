package dev.dubhe.mobile

import android.app.ActivityManager
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

internal fun recommendedChatModel() = mobileCatalog.firstOrNull { it.recommended && !it.translation }
    ?: mobileCatalog.filterNot { it.translation }.minByOrNull { it.bytes }

@Composable
internal fun ScanScreen(chat: ChatViewModel, translate: (String) -> Unit) {
    val context = LocalContext.current
    val state = chat.state
    val idle = state.activity == Activity.Idle
    val openImage = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) chat.scanImage(uri)
    }
    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        WorkspaceCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                SectionTitle("图片转文字")
                Text(
                    "可从相册选照片或截图。中文和拉丁字母可在手机本机识别。",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Button(
                    onClick = {
                        try {
                            openImage.launch(arrayOf("image/*"))
                        } catch (_: ActivityNotFoundException) {
                            chat.showError("这台设备没有可用的文件选择器。")
                        }
                    },
                    enabled = idle,
                ) { Text(if (chat.scanText.isEmpty()) "选择图片" else "换一张图片") }
            }
        }
        if (state.activity == Activity.Importing && state.downloading == null) {
            TransferProgress(state.progress)
            state.status?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
        if (chat.hasSavedScan || chat.scanText.isNotEmpty()) {
            OutlinedTextField(
                chat.scanText, chat::editScanText, Modifier.fillMaxWidth(),
                enabled = idle, minLines = 5, maxLines = 15,
                label = { Text("识别结果 · 可编辑") },
                shape = MaterialTheme.shapes.large,
            )
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row {
                    IconAction(Glyph.Copy, "复制识别文字", enabled = chat.scanText.isNotBlank()) {
                        context.getSystemService(ClipboardManager::class.java)
                            ?.setPrimaryClip(ClipData.newPlainText("扫描", chat.scanText))
                    }
                    TextButton(onClick = chat::saveScan, enabled = idle) { Text("保存校对") }
                }
                Button(onClick = { translate(chat.scanText) }, enabled = idle && chat.scanText.isNotBlank()) { Text("送去翻译") }
            }
            state.status?.takeIf { it == "识别结果已保存" }?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

@Composable
internal fun TasksScreen(
    chat: ChatViewModel, openModelLibrary: () -> Unit, openTranslation: (TranslationTask) -> Unit, openScan: () -> Unit,
) {
    val context = LocalContext.current
    val state = chat.state
    val idle = state.activity == Activity.Idle
    var clearDownloads by remember { mutableStateOf(false) }
    var clearScan by remember { mutableStateOf(false) }
    var deleteTask by remember { mutableStateOf<TranslationTask?>(null) }
    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        SectionTitle("本机存储")
        state.modelStorage?.let { storage ->
            val models = Formatter.formatShortFileSize(context, storage.installed)
            val partials = Formatter.formatShortFileSize(context, storage.incomplete)
            val available = Formatter.formatShortFileSize(context, storage.available)
            ModelCard("已安装模型 $models", "设备可用 $available", Glyph.Model) {
                TextButton(openModelLibrary) { Text("管理模型") }
            }
            ModelCard("未完成下载 $partials", "清理后，下次下载将从头开始。", Glyph.Download) {
                TextButton({ clearDownloads = true }, enabled = idle && storage.incomplete > 0) { Text("清理") }
            }
        }
        SectionTitle("最近任务", Modifier.padding(top = 6.dp))
        if (chat.hasSavedScan || chat.scanText.isNotEmpty()) {
            ModelCard("最近一次扫描", "查看或校对本机识别文字", Glyph.Image) {
                TextButton(openScan, enabled = idle) { Text("打开") }
                IconAction(Glyph.Close, "清除扫描结果", enabled = idle) { clearScan = true }
            }
        }
        if (chat.translationTasks.isEmpty() && !chat.hasSavedScan && chat.scanText.isEmpty()) {
            Text("扫描图片或开始翻译后，任务会出现在这里。", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        for (task in chat.translationTasks) {
            val progress = if (task.finished) "已完成" else "已翻译 ${task.completed} / ${task.total} 段"
            ModelCard(task.title, "$progress · ${task.input.from.label} → ${task.input.to.label}", Glyph.Translate) {
                TextButton({ openTranslation(task) }, enabled = idle) { Text(if (task.finished) "查看" else "继续") }
                IconAction(Glyph.Close, "删除这条翻译任务", enabled = idle) { deleteTask = task }
            }
        }
    }
    if (clearDownloads) AlertDialog(
        onDismissRequest = { clearDownloads = false },
        title = { Text("清理未完成的下载？") },
        text = { Text("已安装模型可以继续使用，清理的下载需要从头开始。") },
        confirmButton = { Button(onClick = { chat.clearPartialDownloads(); clearDownloads = false }) { Text("清理") } },
        dismissButton = { TextButton(onClick = { clearDownloads = false }) { Text("取消") } },
    )
    if (clearScan) AlertDialog(
        onDismissRequest = { clearScan = false },
        title = { Text("清除扫描结果？") },
        text = { Text("将清除此应用保存的识别文字。相册里的原图片可以继续使用。") },
        confirmButton = { Button(onClick = { chat.clearScan(); clearScan = false }) { Text("清除") } },
        dismissButton = { TextButton(onClick = { clearScan = false }) { Text("取消") } },
    )
    deleteTask?.let { task ->
        AlertDialog(
            onDismissRequest = { deleteTask = null },
            title = { Text("删除这条翻译任务？") },
            text = { Text("将删除这个任务里的原文和译文。已经导出的文件可以继续使用。") },
            confirmButton = { Button(onClick = { chat.deleteTranslationTask(task); deleteTask = null }) { Text("删除") } },
            dismissButton = { TextButton(onClick = { deleteTask = null }) { Text("取消") } },
        )
    }
}

@Composable
internal fun availableMemory(): Long {
    val context = LocalContext.current
    return remember(context) {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java)?.getMemoryInfo(memory)
        memory.availMem
    }
}
