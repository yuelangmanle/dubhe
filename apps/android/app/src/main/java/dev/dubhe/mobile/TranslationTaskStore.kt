package dev.dubhe.mobile

import android.content.Context
import android.util.AtomicFile
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.util.UUID
import org.json.JSONException
import org.json.JSONObject

internal data class TranslationTask(
    val id: String,
    val input: TranslationState,
    val modelKey: String,
    val modelName: String,
    val dedicated: Boolean,
    val nextBlock: Int = 0,
    val completed: Int = 0,
    val total: Int,
    val finished: Boolean = false,
    val updatedAt: Long = System.currentTimeMillis(),
) {
    val title: String
        get() = input.inputLabel?.takeUnless { it in setOf("来自图片", "来自文件", "来自其他应用", "来自扫描") }
            ?: input.source.lineSequence().firstOrNull { it.isNotBlank() }?.trim()?.take(50)
            ?: "翻译任务"

    fun checkpoint(index: Int, text: String, translated: Boolean, finished: Boolean) = copy(
        input = input.copy(result = input.result + text),
        nextBlock = index + 1,
        completed = completed + if (translated) 1 else 0,
        finished = finished,
        updatedAt = System.currentTimeMillis(),
    )
}

internal val LocalModel.translationKey: String get() = catalog?.file ?: file.name

/** The engine writes checkpoints before emitting completed segments to the UI. */
internal class TranslationTaskStore(context: Context) {
    private val directory = File(context.filesDir, "translation-tasks")
    private val scan = AtomicFile(File(context.filesDir, "last-scan.txt"))

    fun readScan(): String = try {
        scan.readFully().decodeToString()
    } catch (_: IOException) {
        ""
    }

    fun writeScan(text: String) = writeAtomic(scan, text)

    fun list(): List<TranslationTask> = directory.listFiles().orEmpty()
        .filter { it.extension == "json" && isTaskId(it.nameWithoutExtension) }
        .mapNotNull {
            try {
                read(it.nameWithoutExtension)
            } catch (_: IOException) {
                null
            } catch (_: JSONException) {
                null
            } catch (_: IllegalArgumentException) {
                null
            }
        }.sortedByDescending { it.updatedAt }

    fun read(id: String): TranslationTask? {
        val document = try {
            JSONObject(file(id).readFully().decodeToString())
        } catch (_: FileNotFoundException) {
            return null
        }
        val input = TranslationState(
            source = document.getString("source"),
            from = Language.valueOf(document.getString("from")),
            to = Language.valueOf(document.getString("to")),
            result = document.getString("result"),
            preferDedicated = document.getBoolean("preferDedicated"),
            inputLabel = document.optString("label").takeIf { it.isNotEmpty() },
            glossary = document.optString("glossary"),
        )
        val checkpoint = document.getInt("nextBlock")
        val completed = document.getInt("completed")
        val total = document.getInt("total")
        val finished = document.getBoolean("finished")
        val blocks = translationBlocks(input.source)
        require(input.source.isNotBlank() && input.source.length <= translationLimit)
        require(checkpoint in 0..blocks.size && total == blocks.count { it.isNotBlank() })
        require(completed == blocks.take(checkpoint).count { it.isNotBlank() } && finished == (checkpoint == blocks.size))
        return TranslationTask(
            id, input, document.getString("modelKey"), document.getString("modelName"),
            document.getBoolean("dedicated"), checkpoint, completed, total, finished, document.getLong("updatedAt"),
        )
    }

    fun write(task: TranslationTask) {
        if (!directory.isDirectory && !directory.mkdirs()) throw IOException("无法建立任务目录。")
        val target = file(task.id)
        val record = JSONObject()
            .put("source", task.input.source).put("from", task.input.from.name).put("to", task.input.to.name)
            .put("result", task.input.result).put("preferDedicated", task.input.preferDedicated).put("label", task.input.inputLabel)
            .put("glossary", task.input.glossary)
            .put("modelKey", task.modelKey).put("modelName", task.modelName).put("dedicated", task.dedicated)
            .put("nextBlock", task.nextBlock).put("completed", task.completed).put("total", task.total)
            .put("finished", task.finished).put("updatedAt", task.updatedAt)
        writeAtomic(target, record.toString())
    }

    private fun writeAtomic(target: AtomicFile, content: String) {
        val output = target.startWrite()
        try {
            output.write(content.encodeToByteArray())
            target.finishWrite(output)
        } catch (error: Exception) {
            target.failWrite(output)
            throw error
        }
    }

    fun remove(id: String) {
        val target = file(id)
        target.delete()
        if (target.baseFile.exists()) throw IOException("无法删除翻译任务。")
    }

    private fun file(id: String): AtomicFile {
        require(isTaskId(id))
        return AtomicFile(File(directory, "$id.json"))
    }

    private fun isTaskId(id: String): Boolean = runCatching { UUID.fromString(id).toString() == id }.getOrDefault(false)
}
