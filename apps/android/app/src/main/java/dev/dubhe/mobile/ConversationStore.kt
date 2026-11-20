package dev.dubhe.mobile

import android.content.Context
import android.util.AtomicFile
import java.io.File
import java.io.FileNotFoundException
import org.json.JSONArray
import org.json.JSONObject

internal data class Conversation(val id: String, val turns: List<ChatTurn>) {
    val title: String get() = turns.first().prompt.lineSequence().first().take(48)
}

internal data class Conversations(val selected: String? = null, val items: List<Conversation> = emptyList())

internal class ConversationStore(context: Context) {
    private val file = AtomicFile(File(context.filesDir, "conversations.json"))

    fun read(): Conversations {
        val root = try {
            JSONObject(file.readFully().decodeToString())
        } catch (_: FileNotFoundException) {
            return Conversations()
        }
        val items = root.getJSONArray("items")
        val chats = List(items.length()) { index ->
            val item = items.getJSONObject(index)
            val messages = item.getJSONArray("turns")
            val turns = List(messages.length()) { turnIndex ->
                val turn = messages.getJSONObject(turnIndex)
                ChatTurn(turn.getString("prompt"), turn.optString("attachment").ifEmpty { null }, turn.getString("answer"))
            }
            Conversation(item.getString("id"), turns)
        }.filter { it.turns.isNotEmpty() }
        return Conversations(root.optString("selected").ifEmpty { null }, chats)
    }

    fun write(history: Conversations) {
        val items = JSONArray()
        for (chat in history.items) {
            val turns = JSONArray()
            for (turn in chat.turns) turns.put(JSONObject()
                .put("prompt", turn.prompt).put("attachment", turn.attachmentName).put("answer", turn.answer))
            items.put(JSONObject().put("id", chat.id).put("turns", turns))
        }
        val content = JSONObject().put("selected", history.selected).put("items", items).toString()
        val output = file.startWrite()
        try {
            output.write(content.encodeToByteArray())
            file.finishWrite(output)
        } catch (failure: Exception) {
            file.failWrite(output)
            throw failure
        }
    }
}
