package dev.dubhe.mobile

import android.content.Context
import android.util.AtomicFile
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Upsert
import java.io.File
import java.io.FileNotFoundException
import java.util.concurrent.Callable
import org.json.JSONObject

internal data class Conversation(val id: String, val turns: List<ChatTurn>) {
    val title: String get() = turns.first().let { turn ->
        turn.prompt.lineSequence().first().ifBlank { turn.attachmentName.orEmpty() }.take(48)
    }
}

internal data class Conversations(val selected: String? = null, val items: List<Conversation> = emptyList())

@Entity(tableName = "conversation")
internal data class ConversationRow(@PrimaryKey val id: String, val position: Int)

@Entity(
    tableName = "chat_turn", primaryKeys = ["conversationId", "position"],
    foreignKeys = [ForeignKey(
        entity = ConversationRow::class, parentColumns = ["id"], childColumns = ["conversationId"],
        onDelete = ForeignKey.CASCADE,
    )],
)
internal data class ChatTurnRow(
    val conversationId: String,
    val position: Int,
    val prompt: String,
    val attachment: String?,
    val answer: String,
    val canContinue: Boolean,
)

@Entity(tableName = "conversation_selection")
internal data class ConversationSelection(@PrimaryKey val id: Int = 0, val selected: String?)

@Dao
internal interface ConversationDao {
    @Query("SELECT * FROM conversation ORDER BY position") fun conversations(): List<ConversationRow>
    @Query("SELECT * FROM chat_turn ORDER BY position") fun turns(): List<ChatTurnRow>
    @Query("SELECT * FROM conversation_selection WHERE id = 0") fun selection(): ConversationSelection?
    @Upsert fun saveConversation(conversation: ConversationRow)
    @Upsert fun saveTurn(turn: ChatTurnRow)
    @Upsert fun saveSelection(selection: ConversationSelection)
    @Query("DELETE FROM conversation WHERE id = :id") fun deleteConversation(id: String)
    @Query("DELETE FROM chat_turn WHERE conversationId = :id AND position >= :position")
    fun trimTurns(id: String, position: Int)
}

@Database(entities = [ConversationRow::class, ChatTurnRow::class, ConversationSelection::class], version = 1)
internal abstract class ConversationDatabase : RoomDatabase() {
    abstract fun history(): ConversationDao
}

internal class ConversationStore(context: Context) {
    private val legacy = AtomicFile(File(context.filesDir, "conversations.json"))
    private val database = Room.databaseBuilder(
        context.applicationContext, ConversationDatabase::class.java,
        File(context.filesDir, "conversations.db").absolutePath,
    ).build()

    fun read(): Conversations = database.runInTransaction(Callable {
        val dao = database.history()
        val selection = migrateIfNeeded(dao)
        val turns = dao.turns().groupBy { it.conversationId }
        val items = dao.conversations().mapNotNull { row ->
            turns[row.id]?.let { messages ->
                Conversation(row.id, messages.map { ChatTurn(it.prompt, it.attachment, it.answer, it.canContinue) })
            }
        }
        Conversations(selection.selected, items)
    })

    fun write(history: Conversations) = database.runInTransaction {
        val dao = database.history()
        val selection = migrateIfNeeded(dao)
        val rows = dao.conversations().associateBy { it.id }
        val turns = dao.turns().groupBy { it.conversationId }
        val ids = history.items.mapTo(mutableSetOf()) { it.id }
        (rows.keys - ids).forEach(dao::deleteConversation)
        history.items.forEachIndexed { position, chat ->
            val row = ConversationRow(chat.id, position)
            if (rows[chat.id] != row) dao.saveConversation(row)
            val saved = turns[chat.id].orEmpty()
            if (saved.size > chat.turns.size) dao.trimTurns(chat.id, chat.turns.size)
            chat.turns.forEachIndexed { index, turn ->
                val message = ChatTurnRow(chat.id, index, turn.prompt, turn.attachmentName, turn.answer, turn.canContinue)
                if (saved.getOrNull(index) != message) dao.saveTurn(message)
            }
        }
        if (selection.selected != history.selected) dao.saveSelection(ConversationSelection(selected = history.selected))
    }

    private fun migrateIfNeeded(dao: ConversationDao): ConversationSelection {
        dao.selection()?.let { return it }
        val previous = readLegacy()
        previous.items.forEachIndexed { position, chat ->
            dao.saveConversation(ConversationRow(chat.id, position))
            chat.turns.forEachIndexed { index, turn ->
                dao.saveTurn(ChatTurnRow(chat.id, index, turn.prompt, turn.attachmentName, turn.answer, false))
            }
        }
        return ConversationSelection(selected = previous.selected).also(dao::saveSelection)
    }

    private fun readLegacy(): Conversations {
        val root = try {
            JSONObject(legacy.readFully().decodeToString())
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
}
