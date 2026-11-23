#pragma once

#include "storage.h"

#include <chrono>

namespace storage {

// One file per document keeps checkpoint updates independent of other jobs.
class FileTasks {
public:
    explicit FileTasks(std::string data_dir) : dir_(std::move(data_dir) + "/tasks") {}

    enum Result { OK, INVALID, MISSING, TOO_LARGE, CONFLICT, IO };
    static constexpr size_t max_payload_bytes = 8 * 1024 * 1024;

    Result list(json::Value* result) const {
        if (!ensure_dir(dir_)) return IO;
        FileLock lock(dir_ + "/.lock", LOCK_SH);
        if (!lock) return IO;
        *result = json::Value::make_arr();
        std::error_code ec;
        const std::filesystem::directory_iterator end;
        for (std::filesystem::directory_iterator pos(dir_, ec); !ec && pos != end; pos.increment(ec)) {
            const auto& entry = *pos;
            if (entry.path().extension() != ".json") continue;
            const auto id = entry.path().stem().string();
            if (!valid_id(id) || !entry.is_regular_file(ec)) continue;
            json::Value item;
            if (!read_json(entry.path().string(), &item) || item.s("id") != id) continue;

            auto summary = json::Value::make_obj();
            for (const char* field : {"id", "kind", "title", "updated"}) {
                if (const auto* value = item.get(field)) summary.set(field, *value);
            }
            const auto bytes = entry.file_size(ec);
            if (!ec) summary.set("bytes", json::Value::make_num(static_cast<double>(bytes)));
            result->arr.push_back(std::move(summary));
        }
        if (ec) return IO;
        std::sort(result->arr.begin(), result->arr.end(), [](const json::Value& a, const json::Value& b) {
            const double at = a.n("updated"), bt = b.n("updated");
            return at != bt ? at > bt : a.s("id") < b.s("id");
        });
        return OK;
    }

    Result read(const std::string& id, json::Value* item) const {
        if (!valid_id(id)) return INVALID;
        if (!ensure_dir(dir_)) return IO;
        FileLock lock(dir_ + "/.lock", LOCK_SH);
        if (!lock || !read_json(path(id), item)) return IO;
        if (!item->get("id")) return MISSING;
        return item->s("id") == id ? OK : IO;
    }

    Result apply(const json::Value& change, double* = nullptr) const {
        const auto* upsert = change.get("upsert");
        const auto* remove = change.get("delete");
        if (change.type != json::Value::OBJ || (upsert == nullptr) == (remove == nullptr)) return INVALID;
        if (remove && (remove->type != json::Value::STR || !valid_id(remove->str))) return INVALID;

        json::Value item;
        if (upsert) {
            const auto id = upsert->s("id"), kind = upsert->s("kind"), title = upsert->s("title");
            const auto* payload = upsert->get("payload");
            if (upsert->type != json::Value::OBJ || !valid_id(id) || (kind != "scan" && kind != "translation") ||
                title.empty() || title.size() > 512 || !payload || payload->type != json::Value::OBJ) return INVALID;
            if (json::dump(*payload).size() > max_payload_bytes) return TOO_LARGE;
            const auto updated = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            item = json::Value::make_obj();
            item.set("id", json::Value::make_str(id));
            item.set("kind", json::Value::make_str(kind));
            item.set("title", json::Value::make_str(title));
            item.set("updated", json::Value::make_num(static_cast<double>(updated)));
            item.set("payload", *payload);
        }

        if (!ensure_dir(dir_)) return IO;
        FileLock lock(dir_ + "/.lock");
        if (!lock) return IO;
        if (remove) return unlink(path(remove->str).c_str()) == 0 || errno == ENOENT ? OK : IO;
        return write_json(path(item.s("id")), item) ? OK : IO;
    }

private:
    static bool valid_id(std::string_view id) {
        return !id.empty() && id.size() <= 80 && id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") == id.npos;
    }

    std::string path(const std::string& id) const { return dir_ + "/" + id + ".json"; }
    std::string dir_;
};

}  // namespace storage

#ifdef DUBHE_SQLITE
#include "task_sqlite.h"
#else
namespace storage { using Tasks = FileTasks; }
#endif
