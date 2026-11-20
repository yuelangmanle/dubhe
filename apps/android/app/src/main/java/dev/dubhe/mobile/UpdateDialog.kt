package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
internal fun UpdateDialog(close: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var checking by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    var changelog by remember { mutableStateOf<String?>(null) }

    fun openGitHub() {
        try {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(ReleaseUpdates.page)))
        } catch (_: ActivityNotFoundException) {
            status = "这台设备没有可用的浏览器。"
        }
    }

    AlertDialog(
        onDismissRequest = close,
        title = { Text(if (changelog == null) "版本与更新" else "更新日志") },
        text = {
            Column(
                Modifier.heightIn(max = 440.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                val notes = changelog
                if (notes != null) {
                    Text(notes)
                } else {
                    Text("当前版本 ${BuildConfig.VERSION_NAME}")
                    if (status.isNotBlank()) Text(status)
                    Button(
                        enabled = !checking,
                        onClick = {
                            checking = true
                            status = "正在检查…"
                            scope.launch {
                                try {
                                    val latest = withContext(Dispatchers.IO) {
                                        ReleaseUpdates.newerVersion(BuildConfig.VERSION_NAME)
                                    }
                                    status = if (latest == null) "已是最新版本。"
                                        else "Dubhe $latest 可更新。请在 GitHub 下载 Android 安装包。"
                                } catch (error: CancellationException) {
                                    throw error
                                } catch (error: IllegalStateException) {
                                    status = error.message ?: "未能完成更新检查。"
                                } catch (_: Exception) {
                                    status = "未能完成更新检查，请检查网络后重试。"
                                } finally {
                                    checking = false
                                }
                            }
                        },
                    ) { Text("检查更新") }
                    TextButton(onClick = {
                        changelog = runCatching {
                            context.assets.open("CHANGELOG.md").bufferedReader(Charsets.UTF_8).use { it.readText() }
                        }.getOrDefault("无法读取随应用提供的更新日志。")
                    }) { Text("更新日志") }
                    TextButton(onClick = { openGitHub() }) { Text("打开 GitHub") }
                }
            }
        },
        confirmButton = { TextButton(onClick = close) { Text("关闭") } },
        dismissButton = if (changelog == null) null else {
            { TextButton(onClick = { changelog = null }) { Text("返回") } }
        },
    )
}
