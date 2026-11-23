package dev.dubhe.mobile

import android.os.Bundle
import android.os.Message
import android.os.Messenger
import android.os.RemoteException

internal object EngineProtocol {
    const val LoadModel = 1
    const val Generate = 2
    const val Stop = 3
    const val LoadProjector = 4
    const val UnloadModel = 5
    const val Translate = 6
    const val ModelReady = 11
    const val Text = 12
    const val Complete = 13
    const val Failed = 14
    const val ProjectorReady = 15
    const val StoppingForHeat = 16
    const val ModelUnloaded = 17
    const val GenerationPlan = 18
}

internal fun Messenger.post(what: Int, id: Int, data: Bundle, reply: Messenger? = null): Boolean =
    try {
        send(Message.obtain(null, what, id, 0).apply {
            this.data = data
            replyTo = reply
        })
        true
    } catch (_: RemoteException) {
        false
    }
