package dev.dubhe.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import androidx.core.content.FileProvider
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

internal data class ScanProgress(val source: Int, val sources: Int, val page: Int, val pages: Int)

class ScanFileProvider : FileProvider()

internal fun createScanPhoto(context: Context): Uri {
    val directory = File(context.cacheDir, "scan-captures")
    check(directory.isDirectory || directory.mkdirs()) { "无法建立拍照目录。" }
    val file = File.createTempFile("scan-", ".jpg", directory)
    return FileProvider.getUriForFile(context, "${context.packageName}.scan", file)
}

internal suspend fun scanDocuments(
    context: Context, sources: List<Uri>, onProgress: suspend (ScanProgress) -> Unit, onText: suspend (String) -> Unit,
) = withContext(Dispatchers.IO) {
    val resolver = context.contentResolver
    TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build()).use { recognizer ->
        sources.forEachIndexed { index, uri ->
            require(uri.scheme == "content") { "请选择设备中的文件。" }
            coroutineContext.ensureActive()
            val type = resolver.getType(uri)
            val isPdf = type == "application/pdf" || (type?.startsWith("image/") != true &&
                resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                    cursor.moveToFirst() && cursor.getString(0)?.endsWith(".pdf", ignoreCase = true) == true
                } == true)
            if (isPdf) {
                onProgress(ScanProgress(index + 1, sources.size, 0, 0))
                readPdf(context, uri) { pdf ->
                    repeat(pdf.pageCount) { pageIndex ->
                        coroutineContext.ensureActive()
                        onProgress(ScanProgress(index + 1, sources.size, pageIndex + 1, pdf.pageCount))
                        val bitmap = pdf.openPage(pageIndex).use { page ->
                            val ratio = 2400f / maxOf(page.width, page.height)
                            Bitmap.createBitmap(
                                maxOf(1, (page.width * ratio).toInt()), maxOf(1, (page.height * ratio).toInt()),
                                Bitmap.Config.ARGB_8888,
                            ).also {
                                it.eraseColor(Color.WHITE)
                                val scale = Matrix().apply { setScale(ratio, ratio) }
                                page.render(it, null, scale, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                            }
                        }
                        onText(recognizer.read(bitmap))
                    }
                }
            } else {
                onProgress(ScanProgress(index + 1, sources.size, 1, 1))
                onText(recognizer.read(decodeScanImage(context, uri)))
            }
        }
    }
}

private suspend fun readPdf(context: Context, uri: Uri, scan: suspend (PdfRenderer) -> Unit) {
    // A document provider can return a pipe, whereas PdfRenderer requires a seekable file.
    val file = File.createTempFile("scan-", ".pdf", context.cacheDir)
    try {
        val input = context.contentResolver.openInputStream(uri) ?: error("无法打开 PDF。")
        input.use { source ->
            file.outputStream().use { output ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    coroutineContext.ensureActive()
                    val count = source.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                }
            }
        }
        ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY).use { descriptor ->
            PdfRenderer(descriptor).use { scan(it) }
        }
    } finally {
        file.delete()
    }
}

private fun decodeScanImage(context: Context, uri: Uri): Bitmap =
    ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val size = info.size
        val scale = maxOf(size.width, size.height).toFloat() / 2400
        if (scale > 1) decoder.setTargetSize(maxOf(1, (size.width / scale).toInt()), maxOf(1, (size.height / scale).toInt()))
    }

private fun TextRecognizer.read(bitmap: Bitmap): String = try {
    Tasks.await(process(InputImage.fromBitmap(bitmap, 0)), 25, TimeUnit.SECONDS).text
} finally {
    bitmap.recycle()
}

internal suspend fun readImageText(context: Context, uri: Uri): String = withContext(Dispatchers.IO) {
    TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build()).use { recognizer ->
        recognizer.read(decodeScanImage(context, uri))
    }
}
