package dev.dubhe.mobile

import android.content.Context
import android.graphics.ImageDecoder
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.util.Xml
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import java.io.OutputStream
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal suspend fun readTranslationInput(context: Context, uri: Uri, image: Boolean): String =
    withContext(Dispatchers.IO) {
        require(uri.scheme == "content") { "请选择设备中的文件。" }
        val text = if (image) {
            val bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val size = info.size
                val scale = maxOf(size.width, size.height).toFloat() / 2400
                if (scale > 1) decoder.setTargetSize(maxOf(1, (size.width / scale).toInt()), maxOf(1, (size.height / scale).toInt()))
            }
            TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build()).use { recognizer ->
                Tasks.await(recognizer.process(InputImage.fromBitmap(bitmap, 0)), 25, TimeUnit.SECONDS).text
            }
        } else {
            val stream = context.contentResolver.openInputStream(uri) ?: error("无法打开文件。")
            val bytes = ByteArray(translationLimit * 4 + 4)
            val count = stream.use { input ->
                var size = 0
                while (size < bytes.size) {
                    val read = input.read(bytes, size, bytes.size - size)
                    if (read < 0) break
                    size += read
                }
                size
            }
            require(count < bytes.size) { "一次最多读取 $translationLimit 个字符，请分段导入。" }
            try {
                bytes.decodeToString(endIndex = count, throwOnInvalidSequence = true).removePrefix("\uFEFF")
            } catch (_: java.nio.charset.CharacterCodingException) {
                error("请选择 UTF-8 编码的 TXT 或 Markdown 文件。")
            }
        }
        require(text.length <= translationLimit) { "一次最多读取 $translationLimit 个字符，请分段导入。" }
        require(text.isNotBlank() && '\u0000' !in text) {
            if (image) "图片里没有识别到文字。" else "请选择包含文字的 UTF-8 TXT 或 Markdown 文件。"
        }
        text
    }

internal enum class TranslationFormat(val label: String, val extension: String, val mime: String) {
    Plain("纯文本 TXT", "txt", "text/plain"),
    Markdown("Markdown", "md", "text/markdown"),
    Pdf("PDF", "pdf", "application/pdf"),
    Word("Word", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document");

    fun write(text: String, output: OutputStream) {
        when (this) {
            Plain, Markdown -> output.write(text.encodeToByteArray())
            Pdf -> writePdf(text, output)
            Word -> writeWord(text, output)
        }
    }
}

private fun writePdf(text: String, output: OutputStream) {
    val width = 595
    val height = 842
    val margin = 52
    val paint = TextPaint(TextPaint.ANTI_ALIAS_FLAG).apply { textSize = 12f }
    val layout = StaticLayout.Builder.obtain(text, 0, text.length, paint, width - 2 * margin)
        .setAlignment(Layout.Alignment.ALIGN_NORMAL).setLineSpacing(5f, 1f).build()
    val pdf = PdfDocument()
    try {
        var line = 0
        var pageNumber = 1
        while (line < layout.lineCount) {
            val start = layout.getLineTop(line)
            var next = line + 1
            while (next < layout.lineCount && layout.getLineBottom(next) - start <= height - 2 * margin) next++
            val end = layout.getLineBottom(next - 1)
            val page = pdf.startPage(PdfDocument.PageInfo.Builder(width, height, pageNumber).create())
            val canvas = page.canvas
            canvas.save()
            canvas.translate(margin.toFloat(), (margin - start).toFloat())
            canvas.clipRect(0, start, width - 2 * margin, end)
            layout.draw(canvas)
            canvas.restore()
            canvas.drawText(pageNumber.toString(), (width / 2).toFloat(), (height - 27).toFloat(), paint)
            pdf.finishPage(page)
            line = next
            pageNumber++
        }
        pdf.writeTo(output)
    } finally {
        pdf.close()
    }
}

private fun writeWord(text: String, output: OutputStream) {
    val word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ZipOutputStream(output).use { zip ->
        fun entry(path: String, xml: String) {
            zip.putNextEntry(ZipEntry(path))
            zip.write(xml.encodeToByteArray())
            zip.closeEntry()
        }
        entry("[Content_Types].xml", """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
            <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
            <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
            </Types>""".trimIndent())
        entry("_rels/.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
            </Relationships>""".trimIndent())
        zip.putNextEntry(ZipEntry("word/document.xml"))
        val xml = Xml.newSerializer()
        xml.setOutput(zip, "UTF-8")
        xml.startDocument("UTF-8", true)
        xml.setPrefix("w", word)
        xml.startTag(word, "document").startTag(word, "body")
        for (line in text.lineSequence()) {
            xml.startTag(word, "p").startTag(word, "r").startTag(word, "t")
                .attribute("http://www.w3.org/XML/1998/namespace", "space", "preserve")
                .text(line.filter { it >= ' ' || it == '\t' }).endTag(word, "t").endTag(word, "r").endTag(word, "p")
        }
        xml.endTag(word, "body").endTag(word, "document")
        xml.endDocument()
        zip.closeEntry()
    }
}
