package dev.dubhe.mobile

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.dubhe.sdk.Attachment
import dev.dubhe.sdk.AttachmentKind
import dev.dubhe.sdk.DubheEngine
import dev.dubhe.sdk.Sampling
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MediaSmokeTest {
    @Test
    fun switchingLanguageModelsResetsTokenizer() = runBlocking {
        val files = InstrumentationRegistry.getInstrumentation().targetContext.filesDir
        val qwen = File(files, "smoke.gguf")
        assumeTrue("Place a small ChatML GGUF in files/smoke.gguf", qwen.isFile)
        val vision = fixture("vision.gguf")
        val input = "<|startoftext|><|im_start|>user\n<|image|>\nWhat color is the image?<|im_end|>\n" +
            "<|im_start|>assistant\n"
        val expected = DubheEngine(contextSize = 512).use { engine ->
            engine.loadModel(vision)
            engine.tokenize(input)
        }
        DubheEngine(contextSize = 512).use { engine ->
            engine.loadModel(qwen)
            val info = engine.loadModel(vision)
            val tokens = engine.tokenize(input)
            assertTrue(tokens.all { it in 0 until info.vocabularySize })
            assertArrayEquals(expected, tokens)
        }
    }

    @Test
    fun visionAcceptsImageAndVideo() = runBlocking {
        val model = fixture("vision.gguf")
        val projector = fixture("vision-mmproj.gguf")
        val image = fixture("red.png")
        val video = fixture("red.mp4")
        DubheEngine(contextSize = 512).use { engine ->
            val info = engine.loadModel(model)
            assertTrue(info.supportsChatML)
            assertTrue(engine.loadProjector(projector).vision)
            checkAttachment(engine, Attachment(image, AttachmentKind.Image), "Describe the image.")
            checkAttachment(engine, Attachment(video, AttachmentKind.Video), "Describe the video.")
        }
    }

    @Test
    fun audioAcceptsWavAndAac() = runBlocking {
        val model = fixture("audio.gguf")
        val projector = fixture("audio-mmproj.gguf")
        val wave = fixture("hello.wav")
        val aac = fixture("hello.m4a")
        DubheEngine(contextSize = 512).use { engine ->
            engine.loadModel(model)
            assertTrue(engine.loadProjector(projector).audio)
            checkAttachment(engine, Attachment(wave, AttachmentKind.Audio), "Transcribe the audio.")
            checkAttachment(engine, Attachment(aac, AttachmentKind.Audio), "Transcribe the audio.")
        }
    }

    private suspend fun checkAttachment(engine: DubheEngine, attachment: Attachment, question: String) {
        val prompt = "<|startoftext|><|im_start|>user\n${attachment.kind.token}\n$question<|im_end|>\n" +
            "<|im_start|>assistant\n"
        val plainCount = engine.tokenize(prompt).size
        val response = engine.generate(prompt, Sampling(maxTokens = 12, temperature = 0f), attachment = attachment)
        assertFalse(response.cancelled)
        assertTrue("Attachment was not expanded into the prompt", response.stats.promptTokens > plainCount)
        assertTrue("The engine returned no text for the attachment", response.text.isNotBlank())
    }

    private fun fixture(name: String): File {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val file = File(context.filesDir, "smoke-media/$name")
        assumeTrue("Place $name under files/smoke-media", file.isFile)
        return file
    }
}
