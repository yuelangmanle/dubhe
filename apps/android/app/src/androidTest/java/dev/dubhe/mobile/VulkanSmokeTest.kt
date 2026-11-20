package dev.dubhe.mobile

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Messenger
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VulkanSmokeTest {
    @Test
    fun vulkanVariantLoadsAndGeneratesThroughService() {
        assumeTrue(BuildConfig.DUBHE_VULKAN)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = File(context.filesDir, "smoke.gguf")
        assumeTrue("Place a supported Q4_K GGUF in files/smoke.gguf", model.isFile)

        val service = ArrayBlockingQueue<Messenger>(1)
        val replies = LinkedBlockingQueue<Pair<Int, Bundle>>()
        val receiver = Messenger(Handler(Looper.getMainLooper()) { message ->
            replies.offer(message.what to Bundle(message.data))
        })
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                service.offer(Messenger(binder))
            }

            override fun onServiceDisconnected(name: ComponentName) = Unit
        }

        assertTrue(context.bindService(Intent(context, EngineService::class.java), connection, Context.BIND_AUTO_CREATE))
        try {
            val endpoint = service.poll(15, TimeUnit.SECONDS) ?: error("Engine service did not connect")
            assertTrue(endpoint.post(EngineProtocol.LoadModel, 1, Bundle().apply {
                putString("path", model.path)
            }, receiver))
            val (status, loaded) = replies.poll(30, TimeUnit.SECONDS) ?: error("Model load timed out")
            assertEquals(loaded.getString("detail"), EngineProtocol.ModelReady, status)
            val backend = loaded.getString("backend")
            assertTrue("Expected active Vulkan, got $backend", backend?.startsWith("Vulkan") == true)

            assertTrue(endpoint.post(EngineProtocol.Generate, 2, Bundle().apply {
                putStringArrayList("messages", arrayListOf("Say hello in one short sentence."))
            }, receiver))
            val (event, reply) = replies.poll(40, TimeUnit.SECONDS) ?: error("Generation timed out")
            assertEquals(reply.getString("detail"), EngineProtocol.Text, event)
            assertTrue(!reply.getString("text").isNullOrEmpty())
            endpoint.post(EngineProtocol.Stop, 2, Bundle(), receiver)
        } finally {
            context.unbindService(connection)
        }
    }
}
