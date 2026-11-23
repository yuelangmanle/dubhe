package dev.dubhe.mobile

import android.content.Context
import android.content.ContextWrapper
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.ZipInputStream
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WorkspaceSmokeTest {
    private val base = InstrumentationRegistry.getInstrumentation().targetContext
    private val folder = File.createTempFile("workspace-", ".test", base.cacheDir).apply {
        check(delete() && mkdir())
    }
    private val context = object : ContextWrapper(base) {
        override fun getFilesDir() = folder
        override fun getCacheDir() = folder
        override fun getSharedPreferences(name: String, mode: Int) =
            base.getSharedPreferences(folder.name + name, mode)
    }

    @After
    fun cleanup() {
        base.deleteSharedPreferences(folder.name + "dubhe-models")
        folder.deleteRecursively()
    }

    @Test
    fun switchingModelsKeepsTheLegacyImportUntilExplicitDeletion() = runBlocking {
        val models = File(folder, "models").apply { mkdir() }
        val legacy = File(models, "import-legacy.gguf").apply { writeText("GGUF") }
        val new = File(models, "import-new.gguf").apply { writeText("GGUF") }
        context.getSharedPreferences("dubhe-models", Context.MODE_PRIVATE).edit()
            .putString("Language.file", legacy.name).putString("Language.name", "旧模型").commit()
        val store = ModelStore(context)
        assertEquals(listOf("旧模型"), store.library().map { it.name })
        assertTrue(store.retain(ModelSlot.Language, LocalModel(new, "新模型")))
        store.discardUnreferencedFiles()
        assertEquals(setOf(legacy, new), store.library().map { it.file }.toSet())
        assertEquals(new, store.current(ModelSlot.Language)?.file)
        store.remove(LocalModel(new, "新模型"))
        assertEquals(null, store.current(ModelSlot.Language))
        assertFalse(new.exists())
        assertTrue(legacy.exists())
    }

    @Test
    fun savesConversationAndRestoresSelectedId() {
        val first = Conversation("a", listOf(ChatTurn("你好", answer = "您好")))
        val second = Conversation("b", listOf(ChatTurn("写稿", "参考图.png")))
        val state = Conversations(second.id, listOf(second, first))
        ConversationStore(context).write(state)
        assertEquals(state, ConversationStore(context).read())
        ConversationStore(context).write(Conversations(first.id, listOf(first)))
        assertEquals(listOf(first), ConversationStore(context).read().items)
    }

    @Test
    fun usesAttachmentNameAsTheTitleForAudioOnlyMessages() {
        assertEquals("录音.m4a", Conversation("audio", listOf(ChatTurn("", "录音.m4a"))).title)
    }

    @Test
    fun restoresTranslationAtNextUnfinishedBlock() {
        val store = TranslationTaskStore(context)
        val input = TranslationState(
            source = normalizeTranslationSource("Reservoir level.\r\n\r\nMonitor the gate."),
            from = Language.English, to = Language.Chinese, glossary = "Reservoir = 水库",
        )
        val task = TranslationTask(UUID.randomUUID().toString(), input, "approved-model.gguf", "专用翻译", true, total = 2)
        store.write(task.checkpoint(0, "水库水位。", translated = true, finished = false)
            .checkpoint(1, "\n\n", translated = false, finished = false))

        val reopened = TranslationTaskStore(context)
        val pending = reopened.list().single()
        assertEquals(task.id, pending.id)
        assertEquals("approved-model.gguf", pending.modelKey)
        assertEquals("水库水位。\n\n", pending.input.result)
        assertEquals(1, pending.completed)
        assertEquals(listOf("Monitor the gate."), translationBlocks(pending.input.source).drop(pending.nextBlock))
        assertEquals(input.glossary, pending.input.glossary)

        reopened.write(pending.checkpoint(2, "监控闸门。", translated = true, finished = true))
        val finished = TranslationTaskStore(context).read(task.id)
        assertTrue(finished?.finished == true)
        assertEquals("水库水位。\n\n监控闸门。", finished?.input?.result)
        reopened.remove(task.id)
        assertTrue(TranslationTaskStore(context).list().isEmpty())
    }

    @Test
    fun keepsLastScanAndAppliesOnlyRelevantGlossaryTerms() {
        TranslationTaskStore(context).writeScan("闸门水位")
        assertEquals("闸门水位", TranslationTaskStore(context).readScan())
        val glossary = readTranslationGlossary("reservoir = 水库\nspillway = 泄洪道")
        val prompt = translationPrompt("Reservoir level", Language.English, Language.Chinese, true, glossary)
        assertTrue(prompt.contains("reservoir = 水库"))
        assertFalse(prompt.contains("泄洪道"))
        assertTrue(runCatching { readTranslationGlossary("无分隔符") }.isFailure)
        assertTrue(runCatching { readTranslationGlossary("Gate = 闸门\ngate = 门") }.isFailure)
    }

    @Test
    fun reportsAndClearsOnlyCatalogPartials() = runBlocking {
        val catalogItem = mobileCatalog.first()
        val models = File(folder, "models").apply { mkdir() }
        val partial = File(models, "${catalogItem.file}.part").apply { writeText("GGUF-") }
        val installed = File(models, "existing.gguf").apply { writeText("GGUF") }
        val unrelated = File(models, "user-file.part").apply { writeText("keep") }
        val store = ModelStore(context)
        assertTrue(store.remember(LocalModel(installed, "本地模型")))
        assertEquals(5L, store.storage().incomplete)
        assertEquals(4L, store.storage().installed)
        store.clearPartialDownloads()
        assertFalse(partial.exists())
        assertTrue(installed.exists())
        assertTrue(unrelated.exists())
        assertEquals(0L, store.storage().incomplete)
    }

    @Test
    fun catalogPairIsReadyOnlyAfterBothFilesArePresent() = runBlocking {
        val item = mobileCatalog.first { it.projector != null }
        val component = requireNotNull(item.projector)
        val models = File(folder, "models").apply { mkdir() }
        val weights = File(models, item.file)
        val projector = File(models, component.file)
        RandomAccessFile(weights, "rw").use { it.setLength(item.bytes) }
        val model = LocalModel(weights, item.name)
        val store = ModelStore(context)
        assertTrue(store.remember(model))

        assertTrue(store.library().isEmpty())
        assertEquals(setOf(item.file), store.partials())
        assertEquals(item.bytes, store.storage().incomplete)

        RandomAccessFile(projector, "rw").use { it.setLength(component.bytes) }
        assertEquals(listOf(model), store.library())
        assertEquals(item.totalBytes, store.storage().installed)
        assertEquals(projector, store.catalogProjector(model)?.file)
        assertTrue(store.retain(ModelSlot.Language, model))

        val other = File(models, "import-other.gguf").apply { writeText("GGUF") }
        assertTrue(store.retain(ModelSlot.Language, LocalModel(other, "其他模型")))
        assertTrue(projector.isFile)
        store.remove(model)
        assertFalse(weights.exists())
        assertFalse(projector.exists())
        assertTrue(other.exists())
    }

    @Test
    fun clearingIncompletePairRemovesTheCompletedHalfAndTheRemainingPartial() = runBlocking {
        val item = mobileCatalog.first { it.projector != null }
        val component = requireNotNull(item.projector)
        val models = File(folder, "models").apply { mkdir() }
        val weights = File(models, item.file)
        RandomAccessFile(weights, "rw").use { it.setLength(item.bytes) }
        val remaining = File(models, "${component.file}.part").apply { writeText("GGUF") }

        val store = ModelStore(context)
        assertEquals(setOf(item.file), store.partials())
        assertEquals(item.bytes + 4L, store.storage().incomplete)
        store.clearPartialDownloads()
        assertFalse(weights.exists())
        assertFalse(remaining.exists())
        assertEquals(0L, store.storage().incomplete)
    }

    @Test
    fun resumesACompleteWeightsFileAfterItsProjectorFailsVerification() = runBlocking {
        val weights = "GGUFweights"
        val projection = "GGUFright"
        val component = CatalogFile("test-projector.gguf", projection.length.toLong(), "fixture", "revision", sha256(projection))
        val item = CatalogModel("测试模型", "", "test-weights.gguf", weights.length.toLong(), "fixture", "revision",
            sha256(weights), projector = component, modality = CatalogModality.IMAGE)
        val models = File(folder, "models").apply { mkdir() }
        File(models, "${item.file}.part").writeText(weights)
        val stagedProjector = File(models, "${component.file}.part").apply { writeText("GGUFwrong") }

        val store = ModelStore(context)
        val failure = runCatching { store.download(item) {} }.exceptionOrNull()
        assertEquals("模型校验失败，请重新下载。", failure?.message)
        val downloadedWeights = File(models, item.file)
        assertEquals(weights, downloadedWeights.readText())
        assertFalse(stagedProjector.exists())
        assertFalse(File(models, component.file).exists())

        stagedProjector.writeText(projection)
        val progress = mutableListOf<Float>()
        assertEquals(downloadedWeights, store.download(item) { progress.add(it) }.file)
        assertEquals(projection, File(models, component.file).readText())
        assertEquals(item.bytes.toFloat() / item.totalBytes, progress.first())
        assertEquals(1f, progress.last())
    }

    private fun sha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.encodeToByteArray()).joinToString("") { "%02x".format(it) }

    @Test
    fun keepsTranslationParagraphsAndWritesReadableDocuments() {
        val pages = List(120) { "水库 <闸门> & 泄洪。" }.joinToString("\n\n")
        assertEquals(pages, translationBlocks(pages).joinToString(""))
        val input = "甲".repeat(599) + "🌍" + "乙".repeat(12)
        assertEquals(input, translationBlocks(input).filter { it.isNotBlank() }.joinToString(""))
        assertTrue(translationBlocks(input).filter { it.isNotBlank() }.none { it.last().isHighSurrogate() })
        val pdf = File(folder, "译文.pdf")
        pdf.outputStream().use { TranslationFormat.Pdf.write(pages, it) }
        PdfRenderer(ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY)).use {
            assertTrue(it.pageCount > 1)
        }
        val bytes = ByteArrayOutputStream().apply { TranslationFormat.Word.write(pages, this) }.toByteArray()
        var word = ""
        ZipInputStream(ByteArrayInputStream(bytes)).use { document ->
            while (true) {
                val item = document.nextEntry ?: break
                if (item.name == "word/document.xml") word = document.readBytes().decodeToString()
            }
        }
        assertTrue(word.contains("水库 &lt;闸门&gt; &amp; 泄洪。"))
        assertTrue(word.contains("space=\"preserve\""))
    }

    @Test
    fun readsScreenshotTextFromAContentUri() = runBlocking {
        val resolver = base.contentResolver
        val uri = requireNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "dubhe-ocr-check.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        }))
        try {
            val bitmap = Bitmap.createBitmap(900, 180, Bitmap.Config.ARGB_8888)
            Canvas(bitmap).apply {
                drawColor(Color.WHITE)
                drawText("Dubhe screenshot 12345", 30f, 120f, Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 68f })
            }
            requireNotNull(resolver.openOutputStream(uri)).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            assertTrue(readTranslationInput(base, uri, image = true).contains("12345"))
        } finally {
            resolver.delete(uri, null, null)
        }
    }
}
