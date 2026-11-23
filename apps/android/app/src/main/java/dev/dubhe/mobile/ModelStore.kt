package dev.dubhe.mobile

import android.content.Context
import android.net.Uri
import android.os.SystemClock
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import dev.dubhe.sdk.Attachment
import dev.dubhe.sdk.AttachmentKind
import dev.dubhe.sdk.ProjectorCapabilities
import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

internal enum class ModelSlot { Language, Projector }

internal data class LocalModel(val file: File, val name: String) {
    val catalog: CatalogModel? get() = mobileCatalog.firstOrNull {
        (it.file == file.name || it.file == name) && it.bytes == file.length()
    }
    val bundledCatalog: CatalogModel? get() = catalog?.takeIf { it.file == file.name && it.projector != null }
}
internal data class LocalAttachment(val attachment: Attachment, val name: String)
internal data class ModelStorage(val installed: Long, val incomplete: Long, val available: Long)

internal class ModelStore(context: Context) {
    private companion object {
        val transfers = Mutex()
    }

    private val resolver = context.contentResolver
    private val directory = File(context.filesDir, "models")
    private val mediaDirectory = File(context.cacheDir, "dubhe-media")
    private val prefs = context.getSharedPreferences("dubhe-models", Context.MODE_PRIVATE)

    fun current(slot: ModelSlot): LocalModel? {
        val filename = prefs.getString("${slot.name}.file", null) ?: return null
        val file = File(directory, filename)
        if (!ready(file)) return null
        return LocalModel(file, prefs.getString("${slot.name}.name", null) ?: file.name)
    }

    fun library(): List<LocalModel> {
        val catalogModels = mobileCatalog.mapNotNull { item ->
            val file = File(directory, item.file)
            LocalModel(file, item.name).takeIf { ready(file) }
        }
        val importedModels = prefs.all.mapNotNull { (key, name) ->
            if (!key.startsWith("library.") || name !is String) return@mapNotNull null
            val file = File(directory, key.removePrefix("library."))
            if (ready(file)) LocalModel(file, name) else null
        }
        return (catalogModels + importedModels + listOfNotNull(current(ModelSlot.Language)))
            .distinctBy { it.file }.sortedBy { it.name.lowercase() }
    }

    fun remember(model: LocalModel): Boolean = prefs.edit().putString("library.${model.file.name}", model.name).commit()

    fun retain(slot: ModelSlot, model: LocalModel): Boolean {
        val previous = current(slot)
        val oldProjector = current(ModelSlot.Projector)
        val edit = prefs.edit()
            .putString("${slot.name}.file", model.file.name)
            .putString("${slot.name}.name", model.name)
        if (slot == ModelSlot.Language) {
            edit.putString("library.${model.file.name}", model.name)
            previous?.let { edit.putString("library.${it.file.name}", it.name) }
        }
        if (slot == ModelSlot.Language && previous?.file != model.file) {
            edit.remove("${ModelSlot.Projector.name}.file")
            edit.remove("${ModelSlot.Projector.name}.name")
        }
        if (!edit.commit()) return false
        if (slot == ModelSlot.Projector && previous?.file != model.file) previous?.file?.delete()
        if (slot == ModelSlot.Language && previous?.file != model.file) oldProjector?.file?.delete()
        return true
    }

    suspend fun remove(model: LocalModel) = transfers.withLock {
        val item = mobileCatalog.firstOrNull { it.file == model.file.name }
        deleteIfExists(model.file, "无法删除模型文件。")
        item?.projector?.let { deleteIfExists(File(directory, it.file), "无法删除多模态组件。") }
        val edit = prefs.edit().remove("library.${model.file.name}")
        if (prefs.getString("Language.file", null) == model.file.name) {
            val projector = current(ModelSlot.Projector)
            edit.remove("Language.file").remove("Language.name").remove("Projector.file").remove("Projector.name")
            projector?.file?.delete()
        }
        check(edit.commit()) { "无法更新模型库。" }
    }

    fun catalogProjector(model: LocalModel): LocalModel? = model.bundledCatalog?.projector?.let { component ->
        val file = File(directory, component.file)
        LocalModel(file, component.file).takeIf { matches(file, component.bytes) }
    }

    fun partials(): Set<String> = mobileCatalog.mapNotNullTo(mutableSetOf()) { item ->
        item.file.takeIf { incompleteFiles(item).any { it.length() > 0 } }
    }

    fun storage(): ModelStorage {
        val files = library().flatMap { listOfNotNull(it.file, catalogProjector(it)?.file) }.toSet() +
            listOfNotNull(current(ModelSlot.Projector)?.file)
        val pending = mobileCatalog.flatMap(::incompleteFiles).toSet().sumOf(File::length)
        val volume = if (directory.exists()) directory else requireNotNull(directory.parentFile)
        return ModelStorage(files.sumOf { it.length() }, pending, volume.usableSpace)
    }

    suspend fun clearPartialDownloads() = transfers.withLock {
        withContext(Dispatchers.IO) {
            mobileCatalog.flatMap(::incompleteFiles).toSet().forEach {
                deleteIfExists(it, "无法清理未完成的下载。")
            }
        }
    }

    private fun matches(file: File, size: Long): Boolean = file.isFile && file.length() == size

    private fun ready(file: File): Boolean {
        if (!file.isFile) return false
        val item = mobileCatalog.firstOrNull { it.file == file.name } ?: return true
        val projector = item.projector
        return matches(file, item.bytes) && (projector == null || matches(File(directory, projector.file), projector.bytes))
    }

    private fun incompleteFiles(item: CatalogModel): List<File> {
        val weights = File(directory, item.file)
        val names = listOfNotNull(item.file, item.projector?.file)
        val staging = names.map { File(directory, "$it.part") }
        return if (ready(weights)) staging else staging + names.map { File(directory, it) }
    }

    private fun deleteIfExists(file: File, error: String) {
        if (file.exists() && !file.delete()) throw IOException(error)
    }

    suspend fun discardUnreferencedFiles() = withContext(Dispatchers.IO) {
        val kept = library().map { it.file }.toSet() + listOfNotNull(current(ModelSlot.Projector)?.file)
        directory.listFiles()?.filter { it.name.startsWith("import-") && it !in kept }?.forEach { it.delete() }
        mediaDirectory.listFiles()?.filter { it.name.startsWith("import-") }?.forEach { it.delete() }
    }

    suspend fun importModel(uri: Uri, onProgress: (Float?) -> Unit): LocalModel {
        val copied = copy(uri, onProgress)
        return LocalModel(copied.file, copied.name)
    }

    suspend fun download(item: CatalogModel, onProgress: (Float) -> Unit): LocalModel = transfers.withLock {
        withContext(Dispatchers.IO) {
            if (!directory.isDirectory && !directory.mkdirs()) throw IOException("无法创建模型目录。")
            val weights = CatalogFile(item.file, item.bytes, item.repository, item.revision, item.sha256)
            val artifacts = listOfNotNull(weights, item.projector)
            val requiredBytes = artifacts.sumOf { artifact ->
                val partialBytes = File(directory, "${artifact.file}.part").length()
                if (matches(File(directory, artifact.file), artifact.bytes)) 0L
                else artifact.bytes - (partialBytes.takeIf { it <= artifact.bytes } ?: 0L)
            }
            if (requiredBytes > directory.usableSpace) throw IOException("手机的存储空间不足。")
            var completedBytes = 0L
            for (artifact in artifacts) {
                ensureActive()
                downloadFile(artifact) { bytes ->
                    onProgress(((completedBytes + bytes).toDouble() / item.totalBytes).toFloat())
                }
                completedBytes += artifact.bytes
            }
            LocalModel(File(directory, item.file), item.name)
        }
    }

    private fun CoroutineScope.downloadFile(item: CatalogFile, onProgress: (Long) -> Unit) {
        val part = File(directory, "${item.file}.part")
        val file = File(directory, item.file)
        if (matches(file, item.bytes)) {
            onProgress(item.bytes)
            return
        }
        var offset = part.length()
        if (offset > item.bytes) {
            deleteIfExists(part, "无法清理不完整的模型文件。")
            offset = 0
        }
        onProgress(offset)
        val buffer = ByteArray(256 * 1024)
        if (offset < item.bytes) {
            if (item.bytes - offset > directory.usableSpace) throw IOException("手机的存储空间不足。")
            val connection = URI(item.url).toURL().openConnection() as HttpURLConnection
            try {
                connection.connectTimeout = 15_000
                connection.readTimeout = 10_000
                connection.setRequestProperty("Accept-Encoding", "identity")
                if (offset > 0) connection.setRequestProperty("Range", "bytes=$offset-")
                when (val status = connection.responseCode) {
                    HttpURLConnection.HTTP_PARTIAL -> {
                        val range = "bytes $offset-${item.bytes - 1}/${item.bytes}"
                        if (offset == 0L || connection.getHeaderField("Content-Range") != range) {
                            throw IOException("下载源返回了错误的续传范围。")
                        }
                    }
                    HttpURLConnection.HTTP_OK -> offset = 0
                    else -> throw IOException("下载失败（HTTP $status）。")
                }
                if (item.bytes - offset > directory.usableSpace) throw IOException("手机的存储空间不足。")
                connection.inputStream.use { source ->
                    FileOutputStream(part, offset != 0L).buffered().use { output ->
                        var lastUpdate = 0L
                        while (true) {
                            ensureActive()
                            val read = source.read(buffer)
                            if (read < 0) break
                            offset += read
                            if (offset > item.bytes) throw IOException("下载文件大小与模型清单不符。")
                            output.write(buffer, 0, read)
                            val now = SystemClock.elapsedRealtime()
                            if (now - lastUpdate >= 125L) {
                                onProgress(offset)
                                lastUpdate = now
                            }
                        }
                    }
                }
            } finally {
                connection.disconnect()
            }
        }
        if (part.length() != item.bytes) throw IOException("下载还未完成，可以继续下载。")
        onProgress(item.bytes)
        val digest = MessageDigest.getInstance("SHA-256")
        part.inputStream().use { input ->
            while (true) {
                ensureActive()
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        if (digest.digest().joinToString("") { "%02x".format(it) } != item.sha256) {
            part.delete()
            throw IOException("模型校验失败，请重新下载。")
        }
        ensureActive()
        if (!part.renameTo(file)) throw IOException("无法保存模型。")
    }

    suspend fun importAttachment(
        uri: Uri,
        capabilities: ProjectorCapabilities,
        onProgress: (Float?) -> Unit,
    ): LocalAttachment {
        val copied = copy(uri, onProgress, capabilities)
        return LocalAttachment(Attachment(copied.file, requireNotNull(copied.kind)), copied.name)
    }

    private data class CopiedFile(val file: File, val name: String, val kind: AttachmentKind?)

    private suspend fun copy(
        uri: Uri,
        onProgress: (Float?) -> Unit,
        capabilities: ProjectorCapabilities? = null,
    ): CopiedFile {
        val candidate = AtomicReference<File>()
        try {
            return withContext(Dispatchers.IO) {
                var displayName = if (capabilities == null) "导入模型.gguf" else "附件"
                var expectedSize: Long? = null
                resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)
                    ?.use { cursor ->
                        if (cursor.moveToFirst()) {
                            displayName = cursor.getString(0) ?: displayName
                            if (!cursor.isNull(1)) expectedSize = cursor.getLong(1).takeIf { it > 0 }
                        }
                    }
                if (displayName.isBlank()) displayName = if (capabilities == null) "导入模型.gguf" else "附件"
                val kind = capabilities?.let { attachmentKind(uri, displayName, it) }
                val folder = if (kind == null) directory else mediaDirectory
                if (!folder.isDirectory && !folder.mkdirs()) throw IOException("无法创建文件目录。")
                val size = expectedSize
                if (size != null && size > folder.usableSpace) throw IOException("手机的存储空间不足。")
                val staged = File.createTempFile("import-", if (kind == null) ".gguf" else ".bin", folder)
                candidate.set(staged)
                val input = resolver.openInputStream(uri) ?: throw IOException("无法读取所选文件。")
                DataInputStream(input).use { source ->
                    staged.outputStream().buffered(256 * 1024).use { output ->
                        var count = 0L
                        if (kind == null) {
                            val magic = ByteArray(4)
                            source.readFully(magic)
                            if (!magic.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46))) {
                                throw IOException("请选择 GGUF 文件。")
                            }
                            output.write(magic)
                            count = magic.size.toLong()
                        }
                        val buffer = ByteArray(256 * 1024)
                        var updatedAt = 0L
                        while (true) {
                            ensureActive()
                            val bytes = source.read(buffer)
                            if (bytes < 0) break
                            output.write(buffer, 0, bytes)
                            count += bytes
                            val now = SystemClock.elapsedRealtime()
                            if (now - updatedAt >= 100L) {
                                onProgress(size?.let { (count.toDouble() / it).coerceIn(0.0, 1.0).toFloat() })
                                updatedAt = now
                            }
                        }
                        if (count == 0L || (size != null && count != size)) throw IOException("文件未完整复制。")
                    }
                }
                CopiedFile(staged, displayName.take(160), kind)
            }
        } catch (failure: Throwable) {
            withContext(NonCancellable + Dispatchers.IO) { candidate.get()?.delete() }
            throw failure
        }
    }

    private fun attachmentKind(uri: Uri, name: String, capabilities: ProjectorCapabilities): AttachmentKind {
        val mime = resolver.getType(uri)?.takeUnless { it == "application/octet-stream" }
            ?: MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase())
        return when (mime?.substringBefore('/')) {
            "image" -> AttachmentKind.Image.takeIf { capabilities.vision }
            "video" -> AttachmentKind.Video.takeIf { capabilities.vision }
            "audio" -> AttachmentKind.Audio.takeIf { capabilities.audio }
            else -> null
        } ?: throw IOException("当前多模态组件无法读取这种附件。")
    }
}
