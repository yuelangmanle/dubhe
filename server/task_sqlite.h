#pragma once

#include <sqlite3.h>

namespace storage {

// Each completed page or translation fragment is a row. Metadata and changed
// rows commit together so readers never see a checkpoint halfway through a save.
class Tasks {
public:
    explicit Tasks(const std::string& data_dir)
        : dir_(data_dir + "/tasks"), legacy_(data_dir) {}

    using Result = FileTasks::Result;
    static constexpr auto OK = FileTasks::OK;
    static constexpr auto INVALID = FileTasks::INVALID;
    static constexpr auto MISSING = FileTasks::MISSING;
    static constexpr auto TOO_LARGE = FileTasks::TOO_LARGE;
    static constexpr auto CONFLICT = FileTasks::CONFLICT;
    static constexpr auto IO = FileTasks::IO;
    static constexpr auto max_payload_bytes = FileTasks::max_payload_bytes;
    static constexpr bool incremental = true;

    Result list(json::Value* result) const {
        json::Value legacy;
        if (legacy_.list(&legacy) != OK) return IO;
        *result = json::Value::make_arr();
        DB db(dir_);
        if (!db) return IO;
        Statement rows(db, "SELECT id,kind,title,updated,"
                           " (SELECT coalesce(sum(length(CAST(v AS BLOB))),0) FROM fragment WHERE task=id)"
                           " FROM task");
        if (!rows) return IO;
        std::unordered_set<std::string> ids;
        int rc;
        while ((rc = rows.step()) == SQLITE_ROW) {
            auto item = json::Value::make_obj();
            ids.insert(rows.text(0));
            item.set("id", json::Value::make_str(rows.text(0)));
            item.set("kind", json::Value::make_str(rows.text(1)));
            item.set("title", json::Value::make_str(rows.text(2)));
            item.set("updated", json::Value::make_num(rows.number(3)));
            item.set("bytes", json::Value::make_num(rows.number(4)));
            result->arr.push_back(std::move(item));
        }
        if (rc != SQLITE_DONE) return IO;
        for (auto& item : legacy.arr) {
            if (!ids.count(item.s("id"))) result->arr.push_back(std::move(item));
        }
        std::sort(result->arr.begin(), result->arr.end(), [](const auto& a, const auto& b) {
            const auto at = a.n("updated"), bt = b.n("updated");
            return at != bt ? at > bt : a.s("id") < b.s("id");
        });
        return OK;
    }

    Result read(const std::string& id, json::Value* item) const {
        if (!valid_id(id)) return INVALID;
        DB db(dir_);
        if (!db || !db.exec("BEGIN")) return IO;
        Statement task(db, "SELECT kind,title,updated FROM task WHERE id=?");
        if (!task || !task.bind(1, id)) return IO;
        const auto status = task.step();
        if (status == SQLITE_DONE) return legacy_.read(id, item);
        if (status != SQLITE_ROW) return IO;
        auto payload = json::Value::make_obj();
        Statement parts(db, "SELECT k,v FROM fragment WHERE task=? ORDER BY k");
        if (!parts || !parts.bind(1, id)) return IO;
        auto pages = json::Value::make_arr();
        auto translated = json::Value::make_arr();
        std::string segment;
        std::string source_text;
        int rc;
        while ((rc = parts.step()) == SQLITE_ROW) {
            const auto key = parts.text(0);
            json::Value value;
            if (!json::parse(parts.text(1), &value)) return IO;
            if (key == "meta") payload = std::move(value);
            else if (key == "source") payload.set("source", std::move(value));
            else if (key.starts_with("p/")) pages.arr.push_back(std::move(value));
            else if (key.starts_with("t/")) translated.arr.push_back(std::move(value));
            else if (key.starts_with("s/")) segment += value.str;
            else if (key.starts_with("i/")) source_text += value.str;
        }
        if (rc != SQLITE_DONE || payload.type != json::Value::OBJ) return IO;
        const auto kind = task.text(0);
        if (kind == "scan") payload.set("results", std::move(pages));
        else {
            payload.set("text", json::Value::make_str(std::move(source_text)));
            payload.set("translated", std::move(translated));
            if (const auto* progress = payload.get("segment"); progress && progress->type == json::Value::OBJ) {
                auto restored = *progress;
                restored.set("result", json::Value::make_str(std::move(segment)));
                payload.set("segment", std::move(restored));
            }
        }
        *item = json::Value::make_obj();
        item->set("id", json::Value::make_str(id));
        item->set("kind", json::Value::make_str(kind));
        item->set("title", json::Value::make_str(task.text(1)));
        item->set("updated", json::Value::make_num(task.number(2)));
        item->set("incremental", json::Value::make_bool(true));
        item->set("payload", std::move(payload));
        return OK;
    }

    Result apply(const json::Value& change, double* revision = nullptr) const {
        const auto* patch = change.get("checkpoint");
        if (!patch) {
            if (const auto* remove = change.get("delete")) {
                if (change.obj.size() != 1 || remove->type != json::Value::STR || !valid_id(remove->str))
                    return INVALID;
                DB db(dir_);
                if (!db || !db.exec("BEGIN IMMEDIATE") || legacy_.apply(change) != OK) return IO;
                Statement del(db, "DELETE FROM task WHERE id=?");
                return del && del.bind(1, remove->str) && del.step() == SQLITE_DONE &&
                       db.exec("COMMIT") ? OK : IO;
            }
            return legacy_.apply(change);
        }
        if (change.obj.size() != 1 || patch->type != json::Value::OBJ) return INVALID;
        const auto id = patch->s("id"), kind = patch->s("kind"), title = patch->s("title");
        const auto* parts = patch->get("parts");
        const auto* removed = patch->get("removed");
        const auto* base = patch->get("base");
        if (!valid_id(id) || (kind != "scan" && kind != "translation") || title.empty() ||
            title.size() > 512 || !parts || parts->type != json::Value::OBJ ||
            !removed || removed->type != json::Value::ARR ||
            (base && (base->type != json::Value::NUM || base->num < 0 || std::floor(base->num) != base->num)))
            return INVALID;
        size_t bytes = 0;
        for (const auto& [key, value] : parts->obj) {
            if (!valid_part(key, kind, &value)) return INVALID;
            bytes += json::dump(value).size();
            if (bytes > max_payload_bytes) return TOO_LARGE;
        }
        for (const auto& key : removed->arr)
            if (key.type != json::Value::STR || key.str == "meta" || !valid_part(key.str, kind, nullptr)) return INVALID;

        DB db(dir_);
        if (!db || !db.exec("BEGIN IMMEDIATE")) return IO;
        Statement found(db, "SELECT kind,title,updated FROM task WHERE id=?");
        if (!found || !found.bind(1, id)) return IO;
        const auto existing = found.step();
        double last = 0;
        if (existing == SQLITE_ROW) {
            if (kind != found.text(0)) return INVALID;
            last = found.number(2);
            if (!base || base->num != last) {
                // A save whose acknowledgement was lost can be retried safely.
                if (title != found.text(1) || !matches(db, id, *parts, *removed)) return CONFLICT;
                if (revision) *revision = last;
                return OK;
            }
        } else if (existing == SQLITE_DONE) {
            if (base) return CONFLICT;
            if (!parts->get("meta")) return INVALID;
        } else return IO;
        found.reset();
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        now = std::max(now, static_cast<decltype(now)>(last) + 1);
        Statement head(db, "INSERT INTO task VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,updated=excluded.updated");
        if (!head || !head.bind(1, id) || !head.bind(2, kind) || !head.bind(3, title) ||
            sqlite3_bind_int64(head.value, 4, now) != SQLITE_OK || head.step() != SQLITE_DONE) return IO;
        Statement del(db, "DELETE FROM fragment WHERE task=? AND k=?");
        Statement put(db, "INSERT INTO fragment VALUES(?,?,?) ON CONFLICT(task,k) DO UPDATE SET v=excluded.v");
        if (!del || !put) return IO;
        for (const auto& key : removed->arr) {
            if (!del.bind(1, id) || !del.bind(2, key.str) || del.step() != SQLITE_DONE) return IO;
            del.reset();
        }
        for (const auto& [key, value] : parts->obj) {
            if (!put.bind(1, id) || !put.bind(2, key) || !put.bind(3, json::dump(value)) ||
                put.step() != SQLITE_DONE) return IO;
            put.reset();
        }
        if (!db.exec("COMMIT")) return IO;
        if (revision) *revision = static_cast<double>(now);
        return OK;
    }

private:
    struct DB {
        sqlite3* value = nullptr;
        bool ok = false;
        explicit DB(const std::string& dir) {
            if (!ensure_dir(dir)) return;
            const auto path = dir + "/.checkpoints.sqlite3";
            const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (fd < 0) return;
            close(fd);
            if (sqlite3_open_v2(path.c_str(), &value, SQLITE_OPEN_READWRITE, nullptr) != SQLITE_OK) return;
            sqlite3_busy_timeout(value, 3000);
            ok = exec("PRAGMA foreign_keys=ON;"
                      "CREATE TABLE IF NOT EXISTS task(id TEXT PRIMARY KEY,kind TEXT NOT NULL,title TEXT NOT NULL,updated INTEGER NOT NULL);"
                      "CREATE TABLE IF NOT EXISTS fragment(task TEXT REFERENCES task(id) ON DELETE CASCADE,k TEXT,v TEXT NOT NULL,PRIMARY KEY(task,k));");
        }
        ~DB() { if (value) sqlite3_close(value); }
        explicit operator bool() const { return ok; }
        bool exec(const char* sql) { return sqlite3_exec(value, sql, nullptr, nullptr, nullptr) == SQLITE_OK; }
    };

    struct Statement {
        sqlite3_stmt* value = nullptr;
        Statement(DB& db, const char* sql) { sqlite3_prepare_v2(db.value, sql, -1, &value, nullptr); }
        ~Statement() { sqlite3_finalize(value); }
        explicit operator bool() const { return value != nullptr; }
        bool bind(int n, const std::string& s) {
            return sqlite3_bind_text(value, n, s.data(), (int)s.size(), SQLITE_TRANSIENT) == SQLITE_OK;
        }
        int step() { return sqlite3_step(value); }
        void reset() { sqlite3_reset(value); sqlite3_clear_bindings(value); }
        double number(int col) { return sqlite3_column_double(value, col); }
        std::string text(int col) {
            const auto* str = sqlite3_column_text(value, col);
            return str ? std::string((const char*)str, sqlite3_column_bytes(value, col)) : "";
        }
    };

    static bool matches(DB& db, const std::string& id, const json::Value& changed, const json::Value& removed) {
        Statement row(db, "SELECT v FROM fragment WHERE task=? AND k=?");
        if (!row) return false;
        for (const auto& [key, value] : changed.obj) {
            if (!row.bind(1, id) || !row.bind(2, key) || row.step() != SQLITE_ROW ||
                row.text(0) != json::dump(value)) return false;
            row.reset();
        }
        for (const auto& key : removed.arr) {
            if (!row.bind(1, id) || !row.bind(2, key.str) || row.step() != SQLITE_DONE) return false;
            row.reset();
        }
        return true;
    }

    static bool valid_id(std::string_view id) {
        return !id.empty() && id.size() <= 80 && id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") == id.npos;
    }

    static bool valid_part(std::string_view key, std::string_view kind, const json::Value* v) {
        if (key == "meta") return !v || v->type == json::Value::OBJ;
        if (key == "source") return kind == "translation" && (!v || v->type == json::Value::OBJ || v->type == json::Value::NUL);
        if (key.size() != 10 || key[1] != '/' ||
            key.substr(2).find_first_not_of("0123456789") != key.npos) return false;
        if (kind == "scan") return key[0] == 'p' && (!v || (v->type == json::Value::ARR && v->arr.size() == 2));
        return (key[0] == 't' || key[0] == 's' || key[0] == 'i') && (!v || v->type == json::Value::STR);
    }

    std::string dir_;
    FileTasks legacy_;
};

} // namespace storage
