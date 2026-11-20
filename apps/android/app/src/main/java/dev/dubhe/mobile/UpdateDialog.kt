package dev.dubhe.mobile

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
internal fun UpdateDialog(startOnChangelog: Boolean = false, close: () -> Unit) {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val updates: UpdateViewModel = viewModel()
    val state by updates.state.collectAsStateWithLifecycle()
    var checkedByOpening by rememberSaveable { mutableStateOf(false) }
    var showingChangelog by rememberSaveable { mutableStateOf(startOnChangelog) }
    var browserError by remember { mutableStateOf(false) }

    LaunchedEffect(updates, startOnChangelog) {
        if (!startOnChangelog && !checkedByOpening) {
            checkedByOpening = true
            updates.check()
        }
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (context.packageManager.canRequestPackageInstalls()) {
            updates.setInstallStep(InstallStep.Requested)
        } else {
            updates.setInstallStep(InstallStep.Ready, "允许 Dubhe 安装应用后，就可以继续更新。")
        }
    }
    val installer = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        updates.installerReturned(it.resultCode == Activity.RESULT_OK)
    }
    LaunchedEffect(lifecycle, context, updates) {
        lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
            updates.state.collect { current ->
                val downloaded = current as? UpdateState.Downloaded ?: return@collect
                if (downloaded.step != InstallStep.Requested) return@collect
                if (!downloaded.file.isFile) {
                    updates.missingFile()
                    return@collect
                }
                try {
                    if (context.packageManager.canRequestPackageInstalls()) {
                        updates.setInstallStep(InstallStep.InInstaller)
                        installer.launch(UpdateInstaller.intent(context, downloaded.file))
                    } else {
                        updates.setInstallStep(InstallStep.AwaitingPermission)
                        val app = Uri.fromParts("package", context.packageName, null)
                        permission.launch(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, app))
                    }
                } catch (_: ActivityNotFoundException) {
                    updates.setInstallStep(InstallStep.Ready, "这台设备无法打开系统安装界面。")
                } catch (_: SecurityException) {
                    updates.setInstallStep(InstallStep.Ready, "系统限制了 Dubhe 安装更新。")
                }
            }
        }
    }

    fun dismiss() {
        updates.dismiss()
        close()
    }

    AlertDialog(
        onDismissRequest = ::dismiss,
        title = { Text(if (showingChangelog) "更新日志" else "版本与更新") },
        text = {
            Column(
                Modifier.heightIn(max = 440.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                if (showingChangelog) {
                    val changelog = remember(context) {
                        runCatching {
                            context.assets.open("CHANGELOG.md").bufferedReader(Charsets.UTF_8).use { it.readText() }
                        }.getOrDefault("无法读取随应用提供的更新日志。")
                    }
                    Text(changelog)
                } else {
                    Text("当前版本 ${BuildConfig.VERSION_NAME}")
                    UpdateStatus(state)
                    if (state != UpdateState.Checking && state !is UpdateState.Downloading) {
                        TextButton(onClick = { showingChangelog = true }) { Text("更新日志") }
                        TextButton(onClick = {
                            browserError = try {
                                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(ReleaseUpdates.page)))
                                false
                            } catch (_: ActivityNotFoundException) {
                                true
                            }
                        }) { Text("打开 GitHub") }
                        if (browserError) Text("这台设备没有可用的浏览器。", color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        },
        confirmButton = {
            if (!showingChangelog) UpdateAction(state, updates)
        },
        dismissButton = {
            if (showingChangelog && !startOnChangelog) {
                TextButton(onClick = { showingChangelog = false }) { Text("返回") }
            } else {
                TextButton(onClick = ::dismiss) { Text("关闭") }
            }
        },
    )
}

@Composable
private fun UpdateStatus(state: UpdateState) {
    when (state) {
        UpdateState.Idle -> Unit
        UpdateState.Checking -> {
            Text("正在检查更新…")
            LinearProgressIndicator(Modifier.fillMaxWidth())
        }
        UpdateState.UpToDate -> Text("已是最新版本。")
        is UpdateState.CheckFailed -> Text(state.message, color = MaterialTheme.colorScheme.error)
        is UpdateState.Available -> {
            ReleaseDetails(state.release)
            state.message?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            ReleaseNotes(state.release.notes)
        }
        is UpdateState.Downloading -> {
            ReleaseDetails(state.release)
            Text(if (state.percent == 100) "正在校验安装包…" else "正在下载 ${state.percent}%")
            LinearProgressIndicator(progress = { state.percent / 100f }, modifier = Modifier.fillMaxWidth())
            ReleaseNotes(state.release.notes)
        }
        is UpdateState.Downloaded -> {
            ReleaseDetails(state.release)
            Text(
                when (state.step) {
                    InstallStep.Ready, InstallStep.Requested -> "安装包已下载并通过校验。"
                    InstallStep.AwaitingPermission -> "请在系统设置中允许 Dubhe 安装应用。"
                    InstallStep.InInstaller -> "请在系统界面确认安装。"
                },
            )
            state.message?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            ReleaseNotes(state.release.notes)
        }
        UpdateState.Installed -> Text("更新已安装。")
    }
}

@Composable
private fun ReleaseDetails(release: AndroidRelease) {
    val size = Formatter.formatFileSize(LocalContext.current, release.size)
    Text("可更新到 Dubhe ${release.version}", style = MaterialTheme.typography.titleSmall)
    Text("安装包 $size")
}

@Composable
private fun ReleaseNotes(notes: String) {
    if (notes.isNotBlank()) {
        Text("更新内容", style = MaterialTheme.typography.labelLarge)
        Text(notes)
    }
}

@Composable
private fun UpdateAction(state: UpdateState, updates: UpdateViewModel) {
    when (state) {
        UpdateState.Idle -> Button(onClick = updates::check) { Text("检查更新") }
        UpdateState.UpToDate, is UpdateState.CheckFailed -> Button(onClick = updates::check) { Text("重新检查") }
        is UpdateState.Available -> Button(onClick = updates::download) { Text("下载并安装") }
        is UpdateState.Downloaded -> if (state.step == InstallStep.Ready) {
            Button(onClick = { updates.setInstallStep(InstallStep.Requested) }) { Text("安装更新") }
        }
        else -> Unit
    }
}
