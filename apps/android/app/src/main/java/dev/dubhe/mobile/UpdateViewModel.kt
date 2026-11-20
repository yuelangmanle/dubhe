package dev.dubhe.mobile

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

internal sealed interface UpdateState {
    data object Idle : UpdateState
    data object Checking : UpdateState
    data object UpToDate : UpdateState
    data class CheckFailed(val message: String) : UpdateState
    data class Available(val release: AndroidRelease, val message: String? = null) : UpdateState
    data class Downloading(val release: AndroidRelease, val percent: Int = 0) : UpdateState
    data class Downloaded(
        val release: AndroidRelease,
        val file: File,
        val step: InstallStep,
        val message: String? = null,
    ) : UpdateState
    data object Installed : UpdateState
}

internal enum class InstallStep { Ready, Requested, AwaitingPermission, InInstaller }

internal class UpdateViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow<UpdateState>(UpdateState.Idle)
    val state = _state.asStateFlow()
    private var installAfterDownload = false

    fun check() {
        when (_state.value) {
            UpdateState.Idle, UpdateState.UpToDate, is UpdateState.CheckFailed, is UpdateState.Available -> Unit
            else -> return
        }
        _state.value = UpdateState.Checking
        viewModelScope.launch {
            try {
                val release = withContext(Dispatchers.IO) {
                    UpdateInstaller.pruneOldVersions(getApplication())
                    ReleaseUpdates.check(BuildConfig.VERSION_NAME)
                }
                _state.value = release?.let { UpdateState.Available(it) } ?: UpdateState.UpToDate
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                val message = (error as? UpdateException)?.message ?: "检查更新失败，请检查网络后重试。"
                _state.value = UpdateState.CheckFailed(message)
            }
        }
    }

    fun download() {
        val available = _state.value as? UpdateState.Available ?: return
        val release = available.release
        installAfterDownload = true
        _state.value = UpdateState.Downloading(release)
        viewModelScope.launch {
            try {
                val file = withContext(Dispatchers.IO) {
                    UpdateInstaller.prepare(getApplication(), release) { percent ->
                        _state.value = UpdateState.Downloading(release, percent)
                    }
                }
                val step = if (installAfterDownload) InstallStep.Requested else InstallStep.Ready
                _state.value = UpdateState.Downloaded(release, file, step)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                val message = (error as? UpdateException)?.message ?: "下载失败，请检查网络和可用空间后重试。"
                _state.value = UpdateState.Available(release, message)
            }
        }
    }

    fun dismiss() {
        installAfterDownload = false
        val downloaded = _state.value as? UpdateState.Downloaded ?: return
        if (downloaded.step == InstallStep.Requested) setInstallStep(InstallStep.Ready)
    }

    fun setInstallStep(step: InstallStep, message: String? = null) {
        val downloaded = _state.value as? UpdateState.Downloaded ?: return
        _state.value = downloaded.copy(step = step, message = message)
    }

    fun missingFile() {
        val downloaded = _state.value as? UpdateState.Downloaded ?: return
        _state.value = UpdateState.Available(downloaded.release, "安装包已被清理，请重新下载。")
    }

    fun installerReturned(installed: Boolean) {
        if (installed) {
            _state.value = UpdateState.Installed
        } else {
            setInstallStep(InstallStep.Ready, "安装未完成，可以重试。")
        }
    }
}
