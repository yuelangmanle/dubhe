#pragma once

#include "sha256.h"
#include "storage.h"

#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>

extern char** environ;

namespace store {

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
};

// 文件固定到发布方的 Git revision。size 和 SHA-256 来自对应文件的 LFS 元数据。
inline constexpr Entry catalog[] = {
    {"qwen35-9b-q4", "Qwen3.5-9B Instruct", "9B", "Q4_K_M", 5680522464LL,
     "Qwen3.5-9B-Q4_K_M.gguf",
     "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/"
     "3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf",
     "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"},
    {"qwen35-4b-q4", "Qwen3.5-4B Instruct", "4B", "Q4_K_M", 2740937888LL,
     "Qwen3.5-4B-Q4_K_M.gguf",
     "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
     "e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
     "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"},
    {"lfm25-26b-q4", "LFM2.5-2.6B", "2.6B", "Q4_K_M", 1674455040LL,
     "LFM2.5-2.6B-Q4_K_M.gguf",
     "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/"
     "e7caca5d835a3901a8e0d63e94009429bafafdfc/LFM2.5-2.6B-Q4_K_M.gguf",
     "02a8b7e17487d326e46d68ce0ba24211e1b80a14c4cd0597fa73c1cd697f52ed"},
    {"lfm25-12b-q4", "LFM2.5-1.2B Instruct", "1.2B", "Q4_K_M", 730895168LL,
     "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
     "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/"
     "8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
     "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5"},
    {"qwen38-27b-q4", "Qwen3.8-27B", "27B", "UD-Q4_K_M", 16564259744LL,
     "Qwen3.8-27B-UD-Q4_K_M.gguf",
     "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
     "4af51b3542f775b0a8e4b310bfb3f17e01c306c9/Qwen3.8-27B-UD-Q4_K_M.gguf",
     "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529"},
    {"hy-mt2-18b-q4", "Hy-MT2 1.8B · 混元翻译", "1.8B", "Q4_K_M", 1133080448LL,
     "Hy-MT2-1.8B-Q4_K_M.gguf",
     "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/"
     "a0c709d9fac510f2c807aa3af52872340dc37a4a/Hy-MT2-1.8B-Q4_K_M.gguf",
     "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699", "hy"},
    {"index-translate-2b-q4", "Index-Translate 2B · B 站翻译", "2B", "Q4_K_M", 1312164352LL,
     "Index-Translate-2B.Q4_K_M.gguf",
     "https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF/resolve/"
     "3a0eca6ce931d88d00614d427ab3ed8ff0743902/Index-Translate-2B.Q4_K_M.gguf",
     "044b313d29342bd3b2c77cbb64023ca0d209bd9b3247763f9d162767ef2d746a", "index"},
    {"xiaomi-ocr-0-q4", "Xiaomi-OCR-0 模型", "0.8B", "Q4_K_M", 529297664LL,
     "Xiaomi-OCR-0.Q4_K_M.gguf",
     "https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF/resolve/"
     "7bbfa072e0f3d70ac722e26fbccef5b9f64b7121/Xiaomi-OCR-0.Q4_K_M.gguf",
     "1e497ff06b91a3d822c543066df3e9ef69c62c5423be09d739dfa1db55630d3a", nullptr, true},
    {"xiaomi-ocr-0-vision", "Xiaomi-OCR-0 视觉组件", "", "BF16", 207346624LL,
     "Xiaomi-OCR-0.mmproj-bf16.gguf",
     "https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF/resolve/"
     "7bbfa072e0f3d70ac722e26fbccef5b9f64b7121/Xiaomi-OCR-0.mmproj-bf16.gguf",
     "634d6bc95781ebf4a3f06570a59859fa5082d488db540fd474bc6b4061aec90e", nullptr, true},
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

inline int64_t file_size(const std::string& path) {
    struct stat info{};
    return lstat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) ? info.st_size : -1;
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
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& entry : entries_) {
            const std::string path = models_ + "/" + entry.file;
            const bool installed = file_size(path) == entry.bytes && gguf_header(path);
            const auto partial = std::max<int64_t>(0, file_size(path + ".part"));
            std::string state = installed ? "done" : partial ? "cancelled" : "idle";
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
            item.set("note", json::Value::make_str(entry.translation ? "离线翻译 · 在翻译页使用" :
                entry.scan ? "离线识别 · 在扫描页使用" :
                entry.bytes > 16000000000LL ? "权重较大，请预留内存给上下文" : "安装前检查文件完整性"));
            item.set("bytes", json::Value::make_num((double)entry.bytes));
            item.set("total", json::Value::make_num((double)entry.bytes));
            item.set("got", json::Value::make_num((double)(installed ? entry.bytes : partial)));
            item.set("installed", json::Value::make_bool(installed));
            item.set("state", json::Value::make_str(state));
            items.arr.push_back(std::move(item));
        }
        root.set("items", std::move(items));
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
        if (!storage::ensure_dir(models_)) return {507, "", "无法创建模型目录"};
        const std::string target = models_ + "/" + entry->file;
        const std::string part = target + ".part";
        if (file_size(target) == entry->bytes && gguf_header(target)) return {200, "done", ""};
        struct stat st{};
        if (lstat(target.c_str(), &st) == 0) return {409, "", "同名文件已存在，请先移走"};
        if (lstat(part.c_str(), &st) == 0 && (!S_ISREG(st.st_mode) || st.st_size > entry->bytes))
            return {409, "", "现有续传文件无效，请移走后重试"};
        struct statvfs space{};
        const int64_t needed = entry->bytes - std::max<int64_t>(0, file_size(part));
        if (statvfs(models_.c_str(), &space) == 0 &&
            (long double)space.f_bavail * space.f_frsize < (long double)needed)
            return {507, "", "磁盘剩余空间不足"};

        auto job = std::make_shared<Job>();
        std::string url = entry->url;
        const std::string origin = mirror();
        constexpr std::string_view official = "https://huggingface.co";
        if (!origin.empty() && url.starts_with(official)) url = origin + url.substr(official.size());
        const Entry model = *entry;
        jobs_[id] = job;
        try {
            job->worker = std::thread([job, target, part, model, url] {
                try { execute(*job, target, part, model, url); }
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

private:
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

    static void execute(Job& job, const std::string& target, const std::string& part,
                        const Entry& entry, const std::string& url) {
        auto finish = [&](std::string state, std::string error = {}) {
            std::lock_guard<std::mutex> lock(job.mu);
            job.state = job.cancelled ? "cancelled" : std::move(state);
            job.error = job.cancelled ? "" : std::move(error);
        };
        storage::FileLock file_lock(part + ".lock", LOCK_EX | LOCK_NB);
        if (!file_lock) { finish("error", "另一个 Dubhe 进程正在下载此模型"); return; }
        if (file_size(part) != entry.bytes && transfer(job, url, part) != 0) {
            finish("error", "下载未完成，已保留可续传文件");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(job.mu);
            if (job.cancelled) { job.state = "cancelled"; return; }
            job.state = "verifying";
        }
        storage::Sha256 hash;
        std::ifstream input(part, std::ios::binary);
        std::array<char, 128 * 1024> buffer{};
        for (;;) {
            if (job.cancelled) { finish("cancelled"); return; }
            input.read(buffer.data(), buffer.size());
            hash.update(buffer.data(), (size_t)input.gcount());
            if (!input) break;
        }
        std::lock_guard<std::mutex> lock(job.mu);
        if (job.cancelled) { job.state = "cancelled"; return; }
        if (!input.eof() || input.bad() || file_size(part) != entry.bytes ||
            !gguf_header(part) || hash.finish() != entry.sha256) {
            unlink(part.c_str());
            job.state = "error";
            job.error = "文件完整性校验失败，换一个下载源后重试";
            return;
        }
        // link 在同一个文件系统内发布完成的文件，保证已有文件不会被覆盖。
        if (link(part.c_str(), target.c_str()) != 0) {
            job.state = "error";
            job.error = "模型已验证，但无法安装。请检查同名文件和目录权限";
            return;
        }
        unlink(part.c_str());
        job.state = "done";
    }

    std::string models_;
    std::string config_;
    std::span<const Entry> entries_;
    std::mutex mu_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
};

}  // namespace store
