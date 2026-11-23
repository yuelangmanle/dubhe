#pragma once

#include "json.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

namespace storage {

inline bool ensure_dir(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec && std::filesystem::is_directory(path, ec) && !ec;
}

class FileLock {
public:
    explicit FileLock(const std::string& path, int mode = LOCK_EX) {
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd_ < 0) return;
        int rc;
        do { rc = flock(fd_, mode); } while (rc != 0 && errno == EINTR);
        if (rc != 0) { close(fd_); fd_ = -1; }
    }
    ~FileLock() { if (fd_ >= 0) close(fd_); }
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
    explicit operator bool() const { return fd_ >= 0; }
private:
    int fd_ = -1;
};

inline bool read_json(const std::string& path, json::Value* out) {
    errno = 0;
    std::ifstream file(path, std::ios::binary);
    if (!file) return errno == ENOENT;
    const std::string text((std::istreambuf_iterator<char>(file)), {});
    return !file.bad() && json::parse(text, out) && out->type == json::Value::OBJ;
}

inline bool write_bytes(const std::string& path, const std::string& text, bool replace = true,
                        mode_t mode = 0600) {
    std::string temp = path + ".tmp.XXXXXX";
    const int fd = mkstemp(temp.data());
    if (fd < 0) return false;
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    size_t pos = 0;
    while (pos < text.size()) {
        const ssize_t n = write(fd, text.data() + pos, text.size() - pos);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        pos += (size_t)n;
    }
    const bool saved = pos == text.size() && fchmod(fd, mode) == 0 && fsync(fd) == 0;
    const int closed = close(fd);
    if (!saved || closed != 0 || (replace ? std::rename(temp.c_str(), path.c_str())
                                          : link(temp.c_str(), path.c_str())) != 0) {
        unlink(temp.c_str());
        return false;
    }
    if (!replace) unlink(temp.c_str());
    const auto parent = std::filesystem::path(path).parent_path();
    const int dir = open(parent.empty() ? "." : parent.c_str(), O_RDONLY | O_CLOEXEC);
    if (dir >= 0) { fsync(dir); close(dir); }
    return true;
}

inline bool write_json(const std::string& path, const json::Value& value) {
    return write_bytes(path, json::dump(value));
}

// 按对话 ID 批量提交，在不同进程 / 端口间共用同一份数据。删除标记防止
// 其他旧页面下次把已经删除的本地历史重新导入。
class History {
public:
    explicit History(std::string dir) : dir_(std::move(dir)) {}

    bool read(json::Value* result) const {
        if (!ensure_dir(dir_)) return false;
        FileLock lock(dir_ + "/history.lock", LOCK_SH);
        json::Value current;
        if (!lock || !read_json(dir_ + "/history.json", &current)) return false;
        *result = json::Value::make_obj();
        const auto* items = current.get("items");
        result->set("items", items ? *items : json::Value::make_arr());
        return true;
    }

    enum Result { OK, INVALID, IO };

    Result apply(const json::Value& change) const {
        if (change.type != json::Value::OBJ) return INVALID;
        const auto* upsert = change.get("upsert");
        const auto* remove = change.get("remove");
        const auto* import = change.get("import");
        if (!valid_items(upsert) || !valid_items(import)) return INVALID;
        if (remove) {
            if (remove->type != json::Value::ARR) return INVALID;
            for (const auto& id : remove->arr)
                if (id.type != json::Value::STR || !valid_id(id.str)) return INVALID;
        }

        if (!ensure_dir(dir_)) return IO;
        FileLock lock(dir_ + "/history.lock");
        if (!lock) return IO;
        json::Value old;
        if (!read_json(dir_ + "/history.json", &old)) return IO;
        const auto* previous_items = old.get("items");
        const auto* previous_removed = old.get("removed");
        if (!valid_items(previous_items) ||
            (previous_removed && previous_removed->type != json::Value::ARR)) return IO;
        json::Value items = previous_items ? *previous_items : json::Value::make_arr();
        json::Value removed = previous_removed ? *previous_removed : json::Value::make_arr();
        std::unordered_set<std::string> deleted;
        for (const auto& id : removed.arr) deleted.insert(id.str);
        auto find = [&](const std::string& id) {
            return std::find_if(items.arr.begin(), items.arr.end(),
                                [&](const json::Value& item) { return item.s("id") == id; });
        };
        if (import) {
            for (const auto& item : import->arr) {
                const std::string id = item.s("id");
                if (!deleted.count(id) && find(id) == items.arr.end()) items.arr.push_back(item);
            }
        }
        if (upsert) {
            for (const auto& item : upsert->arr) {
                const std::string id = item.s("id");
                if (deleted.count(id)) continue;
                auto previous = find(id);
                if (previous == items.arr.end()) items.arr.push_back(item);
                else *previous = item;
            }
        }
        if (remove) {
            for (const auto& id : remove->arr) {
                auto previous = find(id.str);
                if (previous != items.arr.end()) items.arr.erase(previous);
                if (deleted.insert(id.str).second) removed.arr.push_back(id);
            }
        }
        json::Value next = json::Value::make_obj();
        next.set("items", std::move(items));
        next.set("removed", std::move(removed));
        return write_json(dir_ + "/history.json", next) ? OK : IO;
    }

private:
    static bool valid_id(const std::string& id) { return !id.empty() && id.size() <= 128; }
    static bool valid_items(const json::Value* items) {
        if (!items) return true;
        if (items->type != json::Value::ARR) return false;
        for (const auto& item : items->arr) {
            const auto* messages = item.get("messages");
            if (!valid_id(item.s("id")) || !messages || messages->type != json::Value::ARR)
                return false;
        }
        return true;
    }
    std::string dir_;
};

}  // namespace storage
