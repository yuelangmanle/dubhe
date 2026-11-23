#pragma once

#include "storage.h"

#include <charconv>
#include <cstdint>
#include <map>

namespace storage {

class Memory {
public:
    explicit Memory(std::string dir) : dir_(std::move(dir)) {}

    enum Result { OK, INVALID, TOO_LARGE, IO };
    static constexpr size_t max_request_bytes = 8 * 1024 * 1024;

    Result read(std::string_view cursor, json::Value* result) const {
        uint64_t since = 0;
        if (!cursor.empty()) {
            const auto [end, error] = std::from_chars(cursor.data(), cursor.data() + cursor.size(), since);
            if (error != std::errc() || end != cursor.data() + cursor.size() || since > max_revision)
                return INVALID;
        }

        if (!ensure_dir(dir_)) return IO;
        FileLock lock(dir_ + "/memory.lock", LOCK_SH);
        Data data;
        if (!lock || !load(&data)) return IO;
        *result = response(data, cursor.empty() || since > data.revision, since);
        return OK;
    }

    Result apply(const json::Value& change, json::Value* result) const {
        const auto* imported = change.get("import");
        const auto* upserted = change.get("upsert");
        const auto* removed = change.get("remove");
        if (change.type != json::Value::OBJ || (!imported && !upserted && !removed) ||
            !valid_items(imported) || !valid_items(upserted)) return INVALID;
        if (removed) {
            if (removed->type != json::Value::ARR || removed->arr.size() > 500) return INVALID;
            for (const auto& id : removed->arr) {
                if (id.type != json::Value::STR || !valid_id(id.str)) return INVALID;
            }
        }

        if (!ensure_dir(dir_)) return IO;
        FileLock lock(dir_ + "/memory.lock");
        Data data;
        if (!lock || !load(&data) || data.revision == max_revision) return IO;
        const uint64_t revision = data.revision + 1;
        bool changed = false;
        auto put = [&](const json::Value* items, bool importing) {
            if (!items) return;
            for (const auto& item : items->arr) {
                const std::string id = item.s("id");
                const auto current = data.rows.find(id);
                if (current != data.rows.end()) {
                    const auto& old = current->second.item;
                    if (importing || old.type == json::Value::NUL ||
                        (old.s("text") == item.s("text") && old.s("scope") == item.s("scope") &&
                         old.n("at") == item.n("at"))) continue;
                }
                json::Value stored = json::Value::make_obj();
                for (const char* field : {"id", "scope", "text", "at"}) stored.set(field, *item.get(field));
                data.rows[id] = {revision, std::move(stored)};
                changed = true;
            }
        };
        put(imported, true);
        put(upserted, false);
        if (removed) {
            for (const auto& id : removed->arr) {
                const auto current = data.rows.find(id.str);
                if (current != data.rows.end() && current->second.item.type == json::Value::NUL) continue;
                data.rows[id.str] = {revision, {}};
                changed = true;
            }
        }
        if (changed) {
            data.revision = revision;
            const std::string encoded = json::dump(encode(data));
            if (encoded.size() > max_store_bytes) return TOO_LARGE;
            if (!write_bytes(path(), encoded)) return IO;
        }
        *result = response(data, true, 0);
        return OK;
    }

private:
    struct Row {
        uint64_t revision;
        json::Value item;
    };
    struct Data {
        uint64_t revision = 0;
        std::map<std::string, Row> rows;
    };
    static constexpr uint64_t max_revision = (uint64_t{1} << 53) - 1;
    static constexpr size_t max_store_bytes = 16 * 1024 * 1024;

    static bool valid_id(std::string_view id) {
        return !id.empty() && id.size() <= 128 &&
               id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") ==
                   std::string_view::npos;
    }

    static bool valid_item(const json::Value& item) {
        const std::string scope = item.s("scope");
        const std::string text = item.s("text");
        const double at = item.n("at", -1);
        return valid_id(item.s("id")) &&
               (scope == "global" || (scope.starts_with("conv:") && valid_id(std::string_view(scope).substr(5)))) &&
               text.size() <= 64 * 1024 && text.find_first_not_of(" \t\r\n") != std::string::npos &&
               text.find('\0') == std::string::npos && at >= 0 && at <= (double)max_revision && std::floor(at) == at;
    }

    static bool valid_items(const json::Value* list) {
        if (!list) return true;
        return list->type == json::Value::ARR && list->arr.size() <= 500 &&
               std::all_of(list->arr.begin(), list->arr.end(), valid_item);
    }

    static bool number(const json::Value& value, const char* field, uint64_t* out) {
        const double n = value.n(field, -1);
        if (n < 0 || n > (double)max_revision || std::floor(n) != n) return false;
        *out = static_cast<uint64_t>(n);
        return true;
    }

    std::string path() const { return dir_ + "/memory.json"; }

    bool load(Data* data) const {
        std::error_code error;
        const auto bytes = std::filesystem::file_size(path(), error);
        if (error == std::errc::no_such_file_or_directory) return true;
        if (error || bytes > max_store_bytes) return false;
        json::Value doc;
        if (!read_json(path(), &doc) || !number(doc, "revision", &data->revision)) return false;
        const auto* rows = doc.get("rows");
        if (!rows || rows->type != json::Value::ARR) return false;
        for (const auto& row : rows->arr) {
            const auto id = row.s("id");
            const auto* item = row.get("item");
            uint64_t revision;
            if (!valid_id(id) || !number(row, "revision", &revision) || revision == 0 ||
                revision > data->revision || !item ||
                (item->type != json::Value::NUL && (!valid_item(*item) || item->s("id") != id)) ||
                !data->rows.emplace(id, Row{revision, *item}).second) return false;
        }
        return true;
    }

    static json::Value encode(const Data& data) {
        auto doc = json::Value::make_obj();
        auto rows = json::Value::make_arr();
        for (const auto& [id, value] : data.rows) {
            auto row = json::Value::make_obj();
            row.set("id", json::Value::make_str(id));
            row.set("revision", json::Value::make_num((double)value.revision));
            row.set("item", value.item);
            rows.arr.push_back(std::move(row));
        }
        doc.set("revision", json::Value::make_num((double)data.revision));
        doc.set("rows", std::move(rows));
        return doc;
    }

    static json::Value response(const Data& data, bool full, uint64_t since) {
        auto result = json::Value::make_obj();
        auto items = json::Value::make_arr();
        auto removed = json::Value::make_arr();
        for (const auto& [id, row] : data.rows) {
            if (!full && row.revision <= since) continue;
            if (row.item.type != json::Value::NUL) items.arr.push_back(row.item);
            else if (!full) removed.arr.push_back(json::Value::make_str(id));
        }
        std::sort(items.arr.begin(), items.arr.end(), [](const json::Value& a, const json::Value& b) {
            const double at = a.n("at"), bt = b.n("at");
            return at != bt ? at > bt : a.s("id") < b.s("id");
        });
        result.set("revision", json::Value::make_num((double)data.revision));
        result.set("items", std::move(items));
        result.set("removed", std::move(removed));
        result.set("full", json::Value::make_bool(full));
        return result;
    }

    std::string dir_;
};

}  // namespace storage
