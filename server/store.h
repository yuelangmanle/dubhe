#pragma once

#include "sha256.h"
#include "storage.h"

#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>

extern char** environ;

namespace store {

struct Artifact {
    int64_t bytes = 0;
    const char* file = nullptr;
    const char* url = nullptr;
    const char* sha256 = nullptr;
};

struct Entry {
    const char* id;
    const char* name;
    const char* params;
    const char* quant;
    int64_t bytes;
    const char* file;
    const char* url;
    const char* sha256;
    const char* translation = nullptr;
    bool scan = false;
    bool recommended = false;
    Artifact projector{};
    const char* modality = nullptr;
};

// 文件固定到发布方的 Git revision。size 和 SHA-256 来自对应文件的 LFS 元数据。
inline constexpr Entry catalog[] = {
#include "model_catalog.inc"
};

inline const Entry* translation_file(std::string_view filename) {
    for (const auto& entry : catalog)
        if (entry.translation && filename == entry.file) return &entry;
    return nullptr;
}

inline const Entry* scan_file(std::string_view filename) {
    for (const auto& entry : catalog)
        if (entry.scan && filename == entry.file) return &entry;
    return nullptr;
}

inline const Entry* bundle_file(std::string_view filename) {
    for (const auto& entry : catalog)
        if (entry.projector.file && filename == entry.file) return &entry;
    return nullptr;
}

inline std::filesystem::path install_directory(const std::string& models, const Entry& entry) {
    const auto root = std::filesystem::path(models);
    return entry.projector.file ? root / entry.id : root;
}

inline std::string managed_projector(const std::string& models, const std::filesystem::path& path) {
    if (const auto* bundle = bundle_file(path.filename().string())) {
        const auto dir = install_directory(models, *bundle);
        if (path.parent_path() == dir) return (dir / bundle->projector.file).string();
    }
    return {};
}

inline int64_t file_size(const std::string& path) {
    struct stat info{};
    return lstat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) ? info.st_size : -1;
}

inline bool real_directory(const std::filesystem::path& dir) {
    struct stat info{};
    return lstat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

inline bool gguf_header(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    char header[8];
    return file.read(header, sizeof(header)) && std::memcmp(header, "GGUF", 4) == 0 &&
           (header[4] == 2 || header[4] == 3) && header[5] == 0 &&
           header[6] == 0 && header[7] == 0;
}

inline bool normalize_mirror(std::string* mirror) {
    while (!mirror->empty() && mirror->back() == '/') mirror->pop_back();
    if (mirror->empty()) return true;
    const size_t prefix = mirror->rfind("https://", 0) == 0 ? 8 :
                          mirror->rfind("http://", 0) == 0 ? 7 : 0;
    return prefix && mirror->size() > prefix &&
           mirror->find_first_of("/\\?#@ \r\n\t", prefix) == std::string::npos;
}

inline double physical_memory_bytes() {
#if defined(__APPLE__)
    uint64_t memory = 0;
    size_t bytes = sizeof(memory);
    return sysctlbyname("hw.memsize", &memory, &bytes, nullptr, 0) == 0 ? static_cast<double>(memory) : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    return pages > 0 && page_size > 0 ? static_cast<double>(pages) * page_size : 0;
#endif
}

class Downloads {
public:
    Downloads(std::string models, std::string config, std::span<const Entry> entries = catalog)
        : models_(std::move(models)), config_(std::move(config)), entries_(entries) {}
    ~Downloads() {
        for (auto& [id, job] : jobs_) {
            cancel_job(*job);
            if (job->worker.joinable()) job->worker.join();
        }
    }

    std::string mirror() const {
        json::Value doc;
        if (!storage::read_json(config_, &doc)) return "";
        std::string value = doc.s("mirror");
        return normalize_mirror(&value) ? value : "";
    }

    bool save_mirror(const std::string& value) const {
        const auto dir = std::filesystem::path(config_).parent_path();
        if (!storage::ensure_dir(dir.string())) return false;
        json::Value doc = json::Value::make_obj();
        doc.set("mirror", json::Value::make_str(value));
        return storage::write_json(config_, doc);
    }

    json::Value status() {
        json::Value root = json::Value::make_obj();
        root.set("dir", json::Value::make_str(models_));
        root.set("mirror", json::Value::make_str(mirror()));
        json::Value items = json::Value::make_arr();
        double partial_bytes = 0;
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& entry : entries_) {
            bool installed = true;
            int64_t total = 0, got = 0, partial = 0;
            const bool readable = !entry.projector.file || real_directory(install_directory(models_, entry));
            for (const auto& file : files(entry)) {
                const bool ready = readable && file.installed();
                const auto pending = readable ? std::max<int64_t>(0, file_size(file.part)) : 0;
                installed &= ready;
                total += file.artifact.bytes;
                got += ready ? file.artifact.bytes : std::min(pending, file.artifact.bytes);
                partial += pending;
            }
            std::string state = installed ? "done" : got ? "cancelled" : "idle";
            json::Value item = json::Value::make_obj();
            auto found = jobs_.find(entry.id);
            if (found != jobs_.end()) {
                std::lock_guard<std::mutex> job_lock(found->second->mu);
                state = found->second->state;
                if (state == "done" && !installed) state = "idle";
                if (!found->second->error.empty())
                    item.set("error", json::Value::make_str(found->second->error));
            }
            item.set("id", json::Value::make_str(entry.id));
            item.set("name", json::Value::make_str(entry.name));
            item.set("params", json::Value::make_str(entry.params));
            item.set("quant", json::Value::make_str(entry.quant));
            item.set("translation", json::Value::make_bool(entry.translation != nullptr));
            item.set("scan", json::Value::make_bool(entry.scan));
            item.set("recommended", json::Value::make_bool(entry.recommended));
            item.set("url", json::Value::make_str(entry.url));
            if (entry.modality) item.set("modality", json::Value::make_str(entry.modality));
            item.set("note", json::Value::make_str(entry.translation ? "离线翻译 · 在翻译页使用" :
                entry.scan ? "离线识别 · 在扫描页使用" :
                entry.modality ? (std::string_view(entry.modality) == "image" ?
                    "图片和视频理解 · 含视觉组件" : "音频理解 · 含音频组件") :
                entry.bytes > 16000000000LL ? "权重较大，请预留内存给上下文" : "安装前检查文件完整性"));
            item.set("bytes", json::Value::make_num((double)total));
            item.set("total", json::Value::make_num((double)total));
            item.set("got", json::Value::make_num((double)got));
            item.set("installed", json::Value::make_bool(installed));
            item.set("state", json::Value::make_str(state));
            item.set("can_clear_partial", json::Value::make_bool(partial > 0 && !active(state)));
            partial_bytes += static_cast<double>(partial);
            items.arr.push_back(std::move(item));
        }
        root.set("items", std::move(items));
        root.set("storage", storage_status(partial_bytes));
        return root;
    }

    struct Reply { int status; std::string state; std::string error; };

    Reply download(const std::string& id) {
        auto entry = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& candidate) { return id == candidate.id; });
        if (entry == entries_.end()) return {400, "", "没有这个模型"};
#if defined(__ANDROID__) && __ANDROID_API__ < 28
        return {503, "", "此 Android 命令行构建请通过文件导入模型"};
#endif
        std::lock_guard<std::mutex> lock(mu_);
        auto found = jobs_.find(id);
        if (found != jobs_.end()) {
            const auto& job = found->second;
            {
                std::lock_guard<std::mutex> job_lock(job->mu);
                if (job->state == "cancelling") return {409, job->state, "正在停止下载，稍后再继续"};
                if (job->state == "running" || job->state == "verifying") return {200, job->state, ""};
            }
            if (job->worker.joinable()) job->worker.join();
        }
        const auto dir = install_directory(models_, *entry);
        if (!storage::ensure_dir(dir.string())) return {507, "", "无法创建模型目录"};
        if (entry->projector.file && !real_directory(dir))
            return {409, "", "模型套装目录必须是本地文件夹"};
        const auto artifacts = files(*entry);
        if (std::all_of(artifacts.begin(), artifacts.end(), [](const auto& file) { return file.installed(); }))
            return {200, "done", ""};
        int64_t needed = 0;
        for (const auto& file : artifacts) {
            if (file.installed()) continue;
            if (const auto* error = file.conflict()) return {409, "", error};
            needed += file.artifact.bytes - std::max<int64_t>(0, file_size(file.part));
        }
        struct statvfs space{};
        if (statvfs(dir.c_str(), &space) == 0 &&
            (long double)space.f_bavail * space.f_frsize < (long double)needed)
            return {507, "", "磁盘剩余空间不足"};

        auto job = std::make_shared<Job>();
        const std::string origin = mirror();
        jobs_[id] = job;
        try {
            job->worker = std::thread([job, artifacts, origin] {
                try { execute(*job, artifacts, origin); }
                catch (const std::exception&) {
                    std::lock_guard<std::mutex> lock(job->mu);
                    job->state = "error";
                    job->error = "下载任务无法继续";
                }
            });
        } catch (const std::system_error&) {
            jobs_.erase(id);
            return {503, "", "无法启动下载任务，请稍后重试"};
        }
        return {200, "running", ""};
    }

    std::string cancel(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto found = jobs_.find(id);
        return found == jobs_.end() ? "idle" : cancel_job(*found->second);
    }

    Reply clear_partial(const std::string& id) {
        const auto entry = std::find_if(entries_.begin(), entries_.end(),
            [&](const Entry& candidate) { return id == candidate.id; });
        if (entry == entries_.end()) return {400, "", "没有这个模型"};
        std::lock_guard<std::mutex> lock(mu_);
        auto job = jobs_.find(id);
        if (job != jobs_.end()) {
            {
                std::lock_guard<std::mutex> job_lock(job->second->mu);
                if (active(job->second->state)) return {409, job->second->state, "请先停止下载"};
            }
            if (job->second->worker.joinable()) job->second->worker.join();
        }
        const auto artifacts = files(*entry);
        const bool readable = !entry->projector.file || real_directory(install_directory(models_, *entry));
        if (readable && std::any_of(artifacts.begin(), artifacts.end(), [](const auto& file) { return file_size(file.part) >= 0; })) {
            storage::FileLock file_lock(artifacts.back().part + ".lock", LOCK_EX | LOCK_NB);
            if (!file_lock) return {409, "", "另一 Dubhe 进程可能正在下载此模型"};
            for (const auto& file : artifacts) {
                if (file_size(file.part) >= 0 && unlink(file.part.c_str()) != 0 && errno != ENOENT)
                    return {507, "", "无法删除未完成的下载"};
            }
        }
        if (job != jobs_.end()) jobs_.erase(job);
        return {200, "idle", ""};
    }

    std::optional<Reply> remove_bundle(const std::filesystem::path& path) {
        const auto dir = path.parent_path();
        const auto entry = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& candidate) {
            return candidate.projector.file && dir == install_directory(models_, candidate) &&
                   (path.filename() == candidate.file || path.filename() == candidate.projector.file);
        });
        if (entry == entries_.end()) return std::nullopt;
        if (!real_directory(dir)) return Reply{409, "", "模型套装目录必须是本地文件夹"};
        if (path.filename() == entry->projector.file)
            return Reply{409, "", "请从模型列表删除整套模型"};

        std::lock_guard<std::mutex> lock(mu_);
        auto job = jobs_.find(entry->id);
        if (job != jobs_.end()) {
            {
                std::lock_guard<std::mutex> job_lock(job->second->mu);
                if (active(job->second->state)) return Reply{409, job->second->state, "请先停止下载"};
            }
            if (job->second->worker.joinable()) job->second->worker.join();
        }
        const auto artifacts = files(*entry);
        storage::FileLock file_lock(artifacts.back().part + ".lock", LOCK_EX | LOCK_NB);
        if (!file_lock) return Reply{409, "", "另一 Dubhe 进程可能正在下载此模型"};
        for (auto file = artifacts.rbegin(); file != artifacts.rend(); ++file) {
            if (unlink(file->target.c_str()) != 0 && errno != ENOENT)
                return Reply{400, "", "无法删除模型套装，请检查模型目录权限"};
        }
        if (job != jobs_.end()) jobs_.erase(job);
        return Reply{200, "idle", ""};
    }

private:
    struct DownloadFile {
        Artifact artifact;
        std::string target;
        std::string part;

        bool installed() const { return file_size(target) == artifact.bytes && gguf_header(target); }

        const char* conflict() const {
            struct stat st{};
            if (lstat(target.c_str(), &st) == 0) return "同名文件已存在，请先移走";
            if (lstat(part.c_str(), &st) == 0 && (!S_ISREG(st.st_mode) || st.st_size > artifact.bytes))
                return "现有续传文件无效，请移走后重试";
            return nullptr;
        }
    };

    std::vector<DownloadFile> files(const Entry& entry) const {
        const auto dir = install_directory(models_, entry);
        auto make = [&](Artifact artifact) {
            const auto target = (dir / artifact.file).string();
            return DownloadFile{artifact, target, target + ".part"};
        };
        std::vector<DownloadFile> result;
        if (entry.projector.file) result.push_back(make(entry.projector));
        result.push_back(make({entry.bytes, entry.file, entry.url, entry.sha256}));
        return result;
    }

    static bool active(std::string_view state) {
        return state == "running" || state == "verifying" || state == "cancelling";
    }

    json::Value storage_status(double partial_bytes) const {
        double installed_bytes = 0;
        std::error_code ec;
        const std::filesystem::recursive_directory_iterator end;
        for (std::filesystem::recursive_directory_iterator pos(
                models_, std::filesystem::directory_options::skip_permission_denied, ec);
             !ec && pos != end; pos.increment(ec)) {
            const auto& entry = *pos;
            const auto extension = entry.path().extension().string();
            if (extension != ".gguf" && extension != ".safetensors" && extension != ".npz" && extension != ".bin")
                continue;
            const auto bytes = file_size(entry.path().string());
            if (bytes > 0) installed_bytes += static_cast<double>(bytes);
        }
        double free_bytes = 0;
        for (auto dir = std::filesystem::path(models_); !dir.empty();) {
            const auto space = std::filesystem::space(dir, ec);
            if (!ec) { free_bytes = static_cast<double>(space.available); break; }
            const auto parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
        auto info = json::Value::make_obj();
        info.set("free_bytes", json::Value::make_num(free_bytes));
        info.set("installed_bytes", json::Value::make_num(installed_bytes));
        info.set("partial_bytes", json::Value::make_num(partial_bytes));
        info.set("memory_bytes", json::Value::make_num(physical_memory_bytes()));
        return info;
    }

    struct Job {
        std::mutex mu;
        std::string state = "running";
        std::string error;
        std::atomic<bool> cancelled{false};
        pid_t pid = 0;
        std::thread worker;
    };

    static std::string cancel_job(Job& job) {
        std::lock_guard<std::mutex> lock(job.mu);
        if (job.state == "running" || job.state == "verifying") {
            job.cancelled = true;
            job.state = "cancelling";
            if (job.pid > 0) kill(job.pid, SIGTERM);
        }
        return job.state;
    }

    static int transfer(Job& job, const std::string& url, const std::string& part) {
#if defined(__ANDROID__) && __ANDROID_API__ < 28
        return -1;
#else
        std::vector<std::string> args = {
            "curl", "--disable", "--fail", "--location", "--silent", "--show-error",
            "--proto", "=https,http", "--proto-redir", "=https,http",
            "--connect-timeout", "15", "--speed-limit", "1", "--speed-time", "60",
            "--retry", "2", "--continue-at", "-", "--output", part, "--url", url,
        };
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        {
            std::lock_guard<std::mutex> lock(job.mu);
            if (job.cancelled || posix_spawnp(&job.pid, "curl", nullptr, nullptr, argv.data(), environ) != 0) {
                job.pid = 0;
                return -1;
            }
        }
        auto cancelled_at = std::chrono::steady_clock::time_point{};
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(job.mu);
                int status = 0;
                const pid_t result = waitpid(job.pid, &status, WNOHANG);
                if (result > 0 || (result < 0 && errno != EINTR)) {
                    job.pid = 0;
                    return result > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
                }
                if (job.cancelled) {
                    const auto now = std::chrono::steady_clock::now();
                    if (cancelled_at == std::chrono::steady_clock::time_point{}) cancelled_at = now;
                    if (now - cancelled_at > std::chrono::seconds(2)) kill(job.pid, SIGKILL);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
#endif
    }

    static bool verify(Job& job, const DownloadFile& file) {
        storage::Sha256 hash;
        std::ifstream input(file.part, std::ios::binary);
        std::array<char, 128 * 1024> buffer{};
        while (!job.cancelled) {
            input.read(buffer.data(), buffer.size());
            hash.update(buffer.data(), static_cast<size_t>(input.gcount()));
            if (!input) break;
        }
        return !job.cancelled && input.eof() && !input.bad() &&
               file_size(file.part) == file.artifact.bytes && gguf_header(file.part) &&
               hash.finish() == file.artifact.sha256;
    }

    static void execute(Job& job, const std::vector<DownloadFile>& artifacts, const std::string& origin) {
        auto finish = [&](std::string state, std::string error = {}) {
            std::lock_guard<std::mutex> lock(job.mu);
            job.state = job.cancelled ? "cancelled" : std::move(state);
            job.error = job.cancelled ? "" : std::move(error);
        };
        storage::FileLock file_lock(artifacts.back().part + ".lock", LOCK_EX | LOCK_NB);
        if (!file_lock) { finish("error", "另一个 Dubhe 进程正在下载此模型"); return; }
        std::vector<const DownloadFile*> pending;
        for (const auto& file : artifacts) {
            if (file.installed()) continue;
            if (const auto* error = file.conflict()) { finish("error", error); return; }
            {
                std::lock_guard<std::mutex> lock(job.mu);
                if (job.cancelled) { job.state = "cancelled"; return; }
                job.state = "running";
            }
            if (file_size(file.part) != file.artifact.bytes) {
                std::string url = file.artifact.url;
                constexpr std::string_view official = "https://huggingface.co";
                if (!origin.empty() && url.starts_with(official)) url = origin + url.substr(official.size());
                if (transfer(job, url, file.part) != 0) {
                    finish("error", "下载未完成，已保留可续传文件");
                    return;
                }
            }
            {
                std::lock_guard<std::mutex> lock(job.mu);
                if (job.cancelled) { job.state = "cancelled"; return; }
                job.state = "verifying";
            }
            if (!verify(job, file)) {
                std::lock_guard<std::mutex> lock(job.mu);
                if (job.cancelled) { job.state = "cancelled"; return; }
                unlink(file.part.c_str());
                job.state = "error";
                job.error = "文件完整性校验失败：" + std::string(file.artifact.file);
                return;
            }
            pending.push_back(&file);
        }
        std::lock_guard<std::mutex> lock(job.mu);
        if (job.cancelled) { job.state = "cancelled"; return; }
        // 两份文件均校验后才安装，主权重最后发布。保留续传文件直至全部安装成功。
        for (auto file = pending.begin(); file != pending.end(); ++file) {
            if (link((*file)->part.c_str(), (*file)->target.c_str()) != 0) {
                for (auto linked = pending.begin(); linked != file; ++linked) unlink((*linked)->target.c_str());
                job.state = "error";
                job.error = "模型已验证，但无法安装。请检查同名文件和目录权限";
                return;
            }
        }
        for (const auto* file : pending) unlink(file->part.c_str());
        job.state = "done";
    }

    std::string models_;
    std::string config_;
    std::span<const Entry> entries_;
    std::mutex mu_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
};

}  // namespace store
