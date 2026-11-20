package dev.dubhe.mobile

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import androidx.core.content.FileProvider
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal class UpdateFileProvider : FileProvider()

internal object UpdateInstaller {
    private const val apkType = "application/vnd.android.package-archive"

    fun pruneOldVersions(context: Context) {
        for (directory in listOf("staging", "verified")) {
            File(context.cacheDir, "updates/$directory").listFiles()?.forEach { cache ->
                val version = cache.name.toIntOrNull()
                if (version != null && version < BuildConfig.VERSION_CODE) cache.deleteRecursively()
            }
        }
    }

    suspend fun prepare(context: Context, release: AndroidRelease, progress: (Int) -> Unit): File {
        val version = BuildConfig.VERSION_CODE
        val stagingDirectory = File(context.cacheDir, "updates/staging/$version")
        val verified = File(context.cacheDir, "updates/verified/$version")
        for (directory in listOf(stagingDirectory, verified)) {
            if (!directory.mkdirs() && !directory.isDirectory) throw UpdateException("无法保存更新安装包。")
        }
        val staging = File.createTempFile("dubhe-", ".apk", stagingDirectory)
        try {
            staging.outputStream().buffered().use { ReleaseUpdates.download(release, it, progress) }
            validatePackage(context, release, staging)
            val destination = File(verified, "${release.version}-${release.sha256}.apk")
            Files.move(staging.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE)
            return destination
        } finally {
            staging.delete()
        }
    }

    @Suppress("DEPRECATION") // PackageInfoFlags requires API 33. The minimum SDK is 31.
    private fun validatePackage(context: Context, release: AndroidRelease, apk: File) {
        val manager = context.packageManager
        val flags = PackageManager.GET_SIGNING_CERTIFICATES
        val installed = manager.getPackageInfo(context.packageName, flags)
        val update = manager.getPackageArchiveInfo(apk.absolutePath, flags)
        if (update == null || update.packageName != context.packageName ||
            update.versionName != release.version || update.longVersionCode <= installed.longVersionCode
        ) {
            throw UpdateException("安装包与这次 Dubhe 更新不符。")
        }
        val installedSigners = installed.signingInfo?.apkContentsSigners?.toSet()
        if (installedSigners.isNullOrEmpty() || update.signingInfo?.apkContentsSigners?.toSet() != installedSigners) {
            throw UpdateException("安装包签名与当前 Dubhe 不同，无法覆盖安装。")
        }
    }

    @Suppress("DEPRECATION") // Uses the system installer to obtain explicit confirmation.
    fun intent(context: Context, apk: File): Intent {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.updates", apk)
        val install = Intent(Intent.ACTION_INSTALL_PACKAGE).apply {
            setDataAndType(uri, apkType)
            clipData = ClipData.newRawUri("Dubhe", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            putExtra(Intent.EXTRA_RETURN_RESULT, true)
        }
        val systemInstaller = context.packageManager.queryIntentActivities(
            install, PackageManager.MATCH_SYSTEM_ONLY or PackageManager.MATCH_DEFAULT_ONLY,
        ).firstOrNull()?.activityInfo ?: throw ActivityNotFoundException()
        return install.setClassName(systemInstaller.packageName, systemInstaller.name)
    }
}
