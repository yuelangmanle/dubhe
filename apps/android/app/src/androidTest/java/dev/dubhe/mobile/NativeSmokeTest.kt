package dev.dubhe.mobile

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.dubhe.sdk.DubheEngine
import dev.dubhe.sdk.DubheException
import dev.dubhe.sdk.Sampling
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeSmokeTest {
    @Test
    fun nativeLibraryLoadsAndRejectsMissingModels() = runBlocking {
        DubheEngine(contextSize = 256).use { engine ->
            assertTrue(engine.version.isNotBlank())
            try {
                engine.loadModel(File("/missing-dubhe-test-model.gguf"))
                fail("Loading a missing model must fail")
            } catch (failure: DubheException) {
                assertTrue(failure.message.orEmpty().isNotBlank())
            }
        }
    }

    @Test
    fun generationCanCancelAndRunAgain() = runBlocking {
        val registry = InstrumentationRegistry.getInstrumentation()
        val model = File(registry.targetContext.filesDir, "smoke.gguf")
        assumeTrue("Place a small ChatML GGUF in files/smoke.gguf", model.isFile)
        DubheEngine(contextSize = 256).use { engine ->
            val info = engine.loadModel(model)
            assertTrue(info.supportsChatML)
            val prompt = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n"
            val cancelled = engine.generate(prompt, Sampling(maxTokens = 64)) {
                engine.cancelGeneration()
            }
            assertTrue(cancelled.cancelled)
            val reply = engine.generate(prompt, Sampling(maxTokens = 4))
            assertFalse(reply.cancelled)
            assertTrue(reply.text.isNotBlank())
            assertTrue(reply.stats.generatedTokens in 1..4)
            engine.unloadModel()
        }
    }
}
