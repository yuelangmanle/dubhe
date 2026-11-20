package dev.dubhe.sdk

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

private const val OUTPUT_RATE = 16_000
private const val MAX_SECONDS = 60
private const val BUFFER_TIMEOUT_US = 10_000L

internal fun decodeMonoAudio(path: String): FloatArray {
    val extractor = MediaExtractor()
    try {
        extractor.setDataSource(path)
        val track = (0 until extractor.trackCount).firstOrNull { index ->
            extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("找不到可解码的音轨。")
        val format = extractor.getTrackFormat(track)
        if (format.containsKey(MediaFormat.KEY_DURATION)) {
            check(format.getLong(MediaFormat.KEY_DURATION) <= MAX_SECONDS * 1_000_000L) {
                "音频最长支持 ${MAX_SECONDS} 秒。"
            }
        }
        extractor.selectTrack(track)
        val mime = requireNotNull(format.getString(MediaFormat.KEY_MIME))
        val output = MonoSamples()
        if (mime == "audio/raw") {
            val size = format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 256 * 1024)
            val buffer = ByteBuffer.allocateDirect(size.coerceIn(256 * 1024, 2 * 1024 * 1024))
            while (true) {
                buffer.clear()
                val count = extractor.readSampleData(buffer, 0)
                if (count < 0) break
                buffer.position(0)
                buffer.limit(count)
                output.append(buffer, format)
                extractor.advance()
            }
        } else {
            decodeCompressed(extractor, format, mime, output)
        }
        return output.resample()
    } finally {
        extractor.release()
    }
}

private fun decodeCompressed(
    extractor: MediaExtractor,
    inputFormat: MediaFormat,
    mime: String,
    output: MonoSamples,
) {
    val codec = MediaCodec.createDecoderByType(mime)
    try {
        codec.configure(inputFormat, null, null, 0)
        codec.start()
        val info = MediaCodec.BufferInfo()
        var inputEnded = false
        var outputEnded = false
        var idle = 0
        while (!outputEnded) {
            var advanced = false
            if (!inputEnded) {
                val index = codec.dequeueInputBuffer(BUFFER_TIMEOUT_US)
                if (index >= 0) {
                    val input = requireNotNull(codec.getInputBuffer(index))
                    input.clear()
                    val count = extractor.readSampleData(input, 0)
                    if (count < 0) {
                        codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputEnded = true
                    } else {
                        codec.queueInputBuffer(index, 0, count, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                    advanced = true
                }
            }
            val index = codec.dequeueOutputBuffer(info, BUFFER_TIMEOUT_US)
            if (index >= 0) {
                try {
                    if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                        val buffer = requireNotNull(codec.getOutputBuffer(index))
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        output.append(buffer, codec.getOutputFormat(index))
                    }
                    outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                } finally {
                    codec.releaseOutputBuffer(index, false)
                }
                advanced = true
            }
            idle = if (advanced) 0 else idle + 1
            check(idle < 500) { "音频解码超时。" }
        }
    } finally {
        codec.release()
    }
}

private class MonoSamples {
    private var rate = 0
    private var count = 0
    private var samples = FloatArray(8192)

    fun append(buffer: ByteBuffer, format: MediaFormat) {
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        require(channels in 1..32 && sampleRate in 8000..192000) { "音频声道或采样率不受支持。" }
        if (rate == 0) rate = sampleRate
        check(rate == sampleRate) { "音频中途改变了采样率。" }

        val encoding = format.getInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
        val bytes = when (encoding) {
            AudioFormat.ENCODING_PCM_8BIT -> 1
            AudioFormat.ENCODING_PCM_16BIT -> 2
            AudioFormat.ENCODING_PCM_24BIT_PACKED -> 3
            AudioFormat.ENCODING_PCM_32BIT, AudioFormat.ENCODING_PCM_FLOAT -> 4
            else -> error("不支持这种音频采样格式。")
        }
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        val frameSize = bytes * channels
        check(buffer.remaining() % frameSize == 0) { "音频采样数据不完整。" }
        val frames = buffer.remaining() / frameSize
        val limit = rate * MAX_SECONDS
        check(frames <= limit - count) { "音频最长支持 ${MAX_SECONDS} 秒。" }
        val needed = count + frames
        if (samples.size < needed) samples = samples.copyOf(maxOf(needed, minOf(samples.size * 2, limit)))
        repeat(frames) {
            var sum = 0f
            repeat(channels) { sum += readSample(buffer, encoding) }
            samples[count++] = (sum / channels).coerceIn(-1f, 1f)
        }
    }

    fun resample(): FloatArray {
        check(count > 0) { "音轨里没有可用的声音。" }
        if (rate == OUTPUT_RATE) return samples.copyOf(count)
        // A low-pass filter avoids aliasing normal 44.1/48 kHz recordings when downsampling.
        val phases = 64
        val radius = 12
        val cutoff = 0.95 * minOf(1.0, OUTPUT_RATE.toDouble() / rate)
        val weights = Array(phases) { phase ->
            FloatArray(2 * radius + 1) { i ->
                val x = i - radius - phase.toDouble() / phases
                val angle = PI * x * cutoff
                val sinc = if (x == 0.0) cutoff else sin(angle) / (PI * x)
                val window = if (abs(x) <= radius) 0.5 + 0.5 * cos(PI * x / radius) else 0.0
                (sinc * window).toFloat()
            }.also { filter ->
                val sum = filter.sum()
                for (i in filter.indices) filter[i] /= sum
            }
        }
        val outputCount = (count.toLong() * OUTPUT_RATE / rate).toInt().coerceAtLeast(1)
        return FloatArray(outputCount) { index ->
            val position = index.toLong() * rate
            val center = (position / OUTPUT_RATE).toInt()
            val phase = ((position % OUTPUT_RATE) * phases / OUTPUT_RATE).toInt()
            val filter = weights[phase]
            var result = 0f
            for (tap in filter.indices) {
                val source = (center + tap - radius).coerceIn(0, count - 1)
                result += filter[tap] * samples[source]
            }
            result.coerceIn(-1f, 1f)
        }
    }
}

private fun readSample(buffer: ByteBuffer, encoding: Int): Float = when (encoding) {
    AudioFormat.ENCODING_PCM_8BIT -> ((buffer.get().toInt() and 255) - 128) / 128f
    AudioFormat.ENCODING_PCM_16BIT -> buffer.short / 32768f
    AudioFormat.ENCODING_PCM_24BIT_PACKED -> {
        val low = buffer.get().toInt() and 255
        val mid = buffer.get().toInt() and 255
        val high = buffer.get().toInt()
        (low or (mid shl 8) or (high shl 16)) / 8388608f
    }
    AudioFormat.ENCODING_PCM_32BIT -> buffer.int / 2147483648f
    AudioFormat.ENCODING_PCM_FLOAT -> buffer.float.takeIf { it.isFinite() } ?: 0f
    else -> error("不支持这种音频采样格式。")
}
