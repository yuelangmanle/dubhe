package dev.dubhe.sdk

import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import android.media.MediaMetadataRetriever
import java.io.File
import kotlin.math.ceil
import kotlin.math.roundToInt

internal object AndroidMedia {
    @JvmStatic
    fun image(path: ByteArray): Bitmap =
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(File(path.toString(Charsets.UTF_8)))) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
            val (width, height) = scaled(info.size.width, info.size.height, 2048)
            decoder.setTargetSize(width, height)
        }.asArgb8888()

    @JvmStatic
    fun video(path: ByteArray, fps: Double, maxFrames: Int, maxSide: Int): Array<Bitmap> {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(path.toString(Charsets.UTF_8))
            val duration = retriever.metadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            require(duration > 0) { "无法读取视频时长。" }
            val rotation = retriever.metadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
            val width = retriever.metadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH).toInt()
            val height = retriever.metadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT).toInt()
            require(width > 0 && height > 0) { "找不到可解码的视频轨道。" }
            val side = if (maxSide > 0) maxSide.coerceAtMost(2048) else 2048
            val (targetWidth, targetHeight) = if (rotation == 90L || rotation == 270L) {
                scaled(height, width, side)
            } else {
                scaled(width, height, side)
            }
            val frameRate = fps.takeIf { it.isFinite() && it > 0 } ?: 2.0
            val count = ceil(duration / 1000.0 * frameRate).toInt().coerceIn(2, maxFrames.coerceIn(2, 16))
            val lastTime = (duration - 1) * 1000
            val frames = (0 until count).mapNotNull { index ->
                val timeUs = (index * 1_000_000.0 / frameRate).toLong().coerceIn(0, lastTime)
                retriever.getScaledFrameAtTime(
                    timeUs, MediaMetadataRetriever.OPTION_CLOSEST, targetWidth, targetHeight,
                )?.asArgb8888()
            }
            require(frames.isNotEmpty()) { "视频中没有可解码的帧。" }
            return if (frames.size == 1) arrayOf(frames[0], frames[0]) else frames.toTypedArray()
        } finally {
            retriever.release()
        }
    }

    @JvmStatic
    fun audio(path: ByteArray): FloatArray = decodeMonoAudio(path.toString(Charsets.UTF_8))
}

private fun scaled(width: Int, height: Int, limit: Int): Pair<Int, Int> {
    val scale = minOf(1.0, limit.toDouble() / maxOf(width, height))
    return maxOf(1, (width * scale).roundToInt()) to maxOf(1, (height * scale).roundToInt())
}

private fun MediaMetadataRetriever.metadata(key: Int): Long = extractMetadata(key)?.toLongOrNull() ?: 0

private fun Bitmap.asArgb8888(): Bitmap {
    if (config == Bitmap.Config.ARGB_8888) return this
    return requireNotNull(copy(Bitmap.Config.ARGB_8888, false)).also { recycle() }
}
