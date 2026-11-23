#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef DUBHE_SQLITE
#include <sqlite3.h>
#endif

namespace knowledge {

struct Document {
    std::string path, stamp, error;
    int64_t mtime = 0, bytes = 0, chunks = 0;
    bool missing = false, truncated = false;
};

struct Chunk {
    size_t offset;
    std::string text, terms;
};

struct Match {
    std::string path, text, stamp;
    double rank;
};

#ifdef DUBHE_SQLITE
class Index {
    class Statement {
    public:
        Statement(sqlite3* db, const char* sql) : db_(db) {
            check(sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr));
        }
        ~Statement() { sqlite3_finalize(stmt_); }
        Statement(const Statement&) = delete;
        Statement& operator=(const Statement&) = delete;

        void bind(int i, std::string_view value) {
            check(sqlite3_bind_text(stmt_, i, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT));
        }
        void bind(int i, int64_t value) { check(sqlite3_bind_int64(stmt_, i, value)); }

        bool row() {
            const int result = sqlite3_step(stmt_);
            if (result == SQLITE_ROW) return true;
            if (result == SQLITE_DONE) return false;
            check(result);
            return false;
        }
        void run() { (void)row(); }
        void reset() {
            check(sqlite3_reset(stmt_));
            check(sqlite3_clear_bindings(stmt_));
        }
        int64_t integer(int i) const { return sqlite3_column_int64(stmt_, i); }
        double number(int i) const { return sqlite3_column_double(stmt_, i); }
        std::string text(int i) const {
            const auto* ptr = sqlite3_column_text(stmt_, i);
            return ptr ? std::string(reinterpret_cast<const char*>(ptr), sqlite3_column_bytes(stmt_, i)) : "";
        }

    private:
        void check(int code) {
            if (code != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(db_));
        }
        sqlite3* db_;
        sqlite3_stmt* stmt_ = nullptr;
    };

public:
    explicit Index(const std::string& path) {
        sqlite3* db = nullptr;
        const int rc = sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
        db_.reset(db);
        if (rc != SQLITE_OK) throw std::runtime_error(db ? sqlite3_errmsg(db) : "无法打开 SQLite");
        sqlite3_busy_timeout(db_.get(), 3000);
        exec("PRAGMA foreign_keys=ON");
        exec(R"SQL(
            CREATE TABLE IF NOT EXISTS kb_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_docs (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                stamp TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                bytes INTEGER NOT NULL,
                chunks INTEGER NOT NULL,
                missing INTEGER NOT NULL,
                truncated INTEGER NOT NULL,
                error TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id INTEGER PRIMARY KEY,
                doc INTEGER NOT NULL REFERENCES kb_docs(id) ON DELETE CASCADE,
                offset INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS kb_chunks_doc ON kb_chunks(doc);
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_terms USING fts5(terms, title);
            CREATE TRIGGER IF NOT EXISTS kb_remove_terms AFTER DELETE ON kb_chunks BEGIN
                DELETE FROM kb_terms WHERE rowid=old.id;
            END;
        )SQL");
    }

    class Write {
    public:
        explicit Write(Index& index) : index_(index) { index_.exec("BEGIN IMMEDIATE"); }
        ~Write() {
            if (!done_) sqlite3_exec(index_.db_.get(), "ROLLBACK", nullptr, nullptr, nullptr);
        }
        Write(const Write&) = delete;
        Write& operator=(const Write&) = delete;
        void commit() { index_.exec("COMMIT"); done_ = true; }
    private:
        Index& index_;
        bool done_ = false;
    };

    std::string setting(std::string_view key) {
        Statement sql(db_.get(), "SELECT value FROM kb_meta WHERE key=?");
        sql.bind(1, key);
        return sql.row() ? sql.text(0) : "";
    }

    void set(std::string_view key, std::string_view value) {
        Statement sql(db_.get(), "INSERT INTO kb_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value");
        sql.bind(1, key);
        sql.bind(2, value);
        sql.run();
    }

    std::vector<Document> documents() {
        Statement sql(db_.get(), R"SQL(
            SELECT path, stamp, mtime, bytes, chunks, missing, truncated, error
            FROM kb_docs ORDER BY path
        )SQL");
        std::vector<Document> docs;
        while (sql.row()) {
            docs.push_back({sql.text(0), sql.text(1), sql.text(7), sql.integer(2), sql.integer(3),
                            sql.integer(4), sql.integer(5) != 0, sql.integer(6) != 0});
        }
        return docs;
    }

    struct Previous { int64_t id = 0; std::string stamp, error; };

    Previous previous(std::string_view path) {
        Statement sql(db_.get(), "SELECT id,stamp,error FROM kb_docs WHERE path=?");
        sql.bind(1, path);
        return sql.row() ? Previous{sql.integer(0), sql.text(1), sql.text(2)} : Previous{};
    }

    void update(const Document& doc, int64_t previous_id, const std::vector<Chunk>& chunks, std::string_view title) {
        Statement upsert(db_.get(), R"SQL(
            INSERT INTO kb_docs(path,stamp,mtime,bytes,chunks,missing,truncated,error)
            VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET stamp=excluded.stamp,
                mtime=excluded.mtime,bytes=excluded.bytes,chunks=excluded.chunks,missing=excluded.missing,
                truncated=excluded.truncated,error=excluded.error
        )SQL");
        upsert.bind(1, doc.path);
        upsert.bind(2, doc.stamp);
        upsert.bind(3, doc.mtime);
        upsert.bind(4, doc.bytes);
        upsert.bind(5, chunks.size());
        upsert.bind(6, doc.missing);
        upsert.bind(7, doc.truncated);
        upsert.bind(8, doc.error);
        upsert.run();

        const int64_t id = previous_id ? previous_id : sqlite3_last_insert_rowid(db_.get());
        if (previous_id) {
            Statement remove(db_.get(), "DELETE FROM kb_chunks WHERE doc=?");
            remove.bind(1, id);
            remove.run();
        }
        Statement add_chunk(db_.get(), "INSERT INTO kb_chunks(doc,offset,text) VALUES (?,?,?)");
        Statement add_terms(db_.get(), "INSERT INTO kb_terms(rowid,terms,title) VALUES (?,?,?)");
        for (const auto& chunk : chunks) {
            add_chunk.bind(1, id);
            add_chunk.bind(2, chunk.offset);
            add_chunk.bind(3, chunk.text);
            add_chunk.run();
            add_chunk.reset();

            add_terms.bind(1, sqlite3_last_insert_rowid(db_.get()));
            add_terms.bind(2, chunk.terms);
            add_terms.bind(3, title);
            add_terms.run();
            add_terms.reset();
        }
    }

    int remove(std::string_view path) {
        Statement sql(db_.get(), "DELETE FROM kb_docs WHERE path=?");
        sql.bind(1, path);
        sql.run();
        return sqlite3_changes(db_.get());
    }

    void clear() { exec("DELETE FROM kb_docs"); }

    std::vector<Match> search(std::string_view query, int limit) {
        Statement sql(db_.get(), R"SQL(
            SELECT d.path,c.text,d.stamp,bm25(kb_terms,1.0,4.0) AS rank
            FROM kb_terms JOIN kb_chunks c ON c.id=kb_terms.rowid JOIN kb_docs d ON d.id=c.doc
            WHERE kb_terms MATCH ?
            ORDER BY rank, c.id LIMIT ?
        )SQL");
        sql.bind(1, query);
        sql.bind(2, limit);
        std::vector<Match> matches;
        while (sql.row()) matches.push_back({sql.text(0), sql.text(1), sql.text(2), -sql.number(3)});
        return matches;
    }

private:
    void exec(const char* sql) {
        if (sqlite3_exec(db_.get(), sql, nullptr, nullptr, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(db_.get()));
    }
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_{nullptr, sqlite3_close};
};
#endif

}  // namespace knowledge
