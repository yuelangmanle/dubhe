#pragma once

#include "json.h"
#include <dubhe/dubhe.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <sstream>
#include <thread>

extern char** environ;

// The optional runtimes consume local model files directly. Only the parent exposes
// HTTP to Dubhe clients. Model changes and cancellation explicitly reap the worker.
class ModelWorker {
public:
    ModelWorker(std::string script, std::string kind, std::string path, int ctx, int threads,
                std::string projector = {})
        : script_(std::move(script)), kind_(std::move(kind)), path_(std::move(path)),
          projector_(std::move(projector)), ctx_(ctx), threads_(threads) {}
    ~ModelWorker() { stop(); }

    static std::string script_for(const std::string& web_root) {
        const auto parent = std::filesystem::absolute(web_root).parent_path();
        const auto bundled = parent / "backends/model_backend.py";
        if (access(bundled.c_str(), R_OK) == 0) return bundled.string();
        const auto local = parent / "tools/model_backend.py";
        return access(local.c_str(), R_OK) == 0 ? local.string() : "";
    }

    static std::string llama_runtime(const std::string& script) {
#if defined(__APPLE__)
        if (script.empty()) return "";
        auto locate = [](const std::string& executable) -> std::string {
            if (executable.find('/') != std::string::npos)
                return access(executable.c_str(), X_OK) == 0 ? executable : "";
            const char* path = std::getenv("PATH");
            std::istringstream dirs(path ? path : "");
            for (std::string dir; std::getline(dirs, dir, ':');) {
                if (dir.empty()) continue;
                auto file = std::filesystem::path(dir) / executable;
                if (access(file.c_str(), X_OK) == 0) return file.string();
            }
            return "";
        };
        if (const char* configured = std::getenv("DUBHE_LLAMA_SERVER"); configured && *configured)
            return locate(configured);
        auto parent = std::filesystem::path(script).parent_path().parent_path().parent_path();
        for (const std::string& path : {(parent / "Helpers/llama/llama-server").string(),
                                        std::string("/opt/homebrew/bin/llama-server"),
                                        std::string("/usr/local/bin/llama-server")}) {
            if (auto found = locate(path); !found.empty()) return found;
        }
        return locate("llama-server");
#else
        (void)script;
        return "";
#endif
    }

    bool start(std::string* error, const std::function<bool()>& cancelled = [] { return false; }) {
        if (pid_ > 0) return true;
#if defined(__ANDROID__) || !defined(__APPLE__)
        *error = "此构建未启用可选模型运行时";
        return false;
#else
        std::string runtime;
        if (kind_ == "llama") {
            runtime = llama_runtime(script_);
            if (runtime.empty()) {
                *error = "找不到本机 llama-server。请使用带翻译运行时的安装包，或设置 DUBHE_LLAMA_SERVER。";
                return false;
            }
        }
        int input[2], output[2];
        if (pipe(input) != 0) { *error = "无法建立模型进程通信"; return false; }
        if (pipe(output) != 0) {
            close(input[0]); close(input[1]);
            *error = "无法建立模型进程通信";
            return false;
        }
        for (int fd : {input[0], input[1], output[0], output[1]}) fcntl(fd, F_SETFD, FD_CLOEXEC);
        posix_spawn_file_actions_t actions;
        posix_spawnattr_t attrs;
        posix_spawn_file_actions_init(&actions);
        posix_spawnattr_init(&attrs);
        posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
        posix_spawnattr_setflags(&attrs, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attrs, 0);
        const char* configured = getenv(kind_ == "mlx" ? "DUBHE_MLX_PYTHON" : "DUBHE_BACKEND_PYTHON");
        std::vector<std::string> args{configured && *configured ? configured : "python3", "-u", script_,
            "--backend", kind_, "--model", path_, "--ctx", std::to_string(ctx_),
            "--threads", std::to_string(threads_)};
        if (!runtime.empty()) {
            args.push_back("--llama-server");
            args.push_back(std::move(runtime));
        }
        if (!projector_.empty()) {
            args.push_back("--mmproj");
            args.push_back(projector_);
        }
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        const int rc = posix_spawnp(&pid_, argv[0], &actions, &attrs, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attrs);
        close(input[0]); close(output[1]);
        if (rc != 0) {
            pid_ = 0;
            close(input[1]); close(output[0]);
            *error = "无法启动 Python 模型运行时";
            return false;
        }
        input_ = input[1];
        output_ = output[0];
        fcntl(input_, F_SETFL, O_NONBLOCK);
        fcntl(output_, F_SETFL, O_NONBLOCK);
        json::Value ready;
        if (!read_event(&ready, cancelled, 120) || !ready.flag("ready")) {
            *error = ready.s("error", "模型运行时未就绪，请查看引擎日志");
            stop();
            return false;
        }
        info_ = std::move(ready);
        return true;
#endif
    }

    const json::Value& info() const { return info_; }

    bool generate(const json::Value& request, const std::function<void(const std::string&)>& write,
                  const std::function<bool()>& cancelled, dubhe_stats* stats,
                  std::string* reason, std::string* error) {
        if (!start(error, cancelled)) return false;
        const auto begin = std::chrono::steady_clock::now();
        std::string line = json::dump(request) + "\n";
        for (size_t offset = 0; offset < line.size();) {
            if (!wait_for(input_, POLLOUT, cancelled, begin + std::chrono::seconds(30))) {
                *error = "模型运行时未接收请求";
                stop();
                return false;
            }
            const ssize_t sent = ::write(input_, line.data() + offset, line.size() - offset);
            if (sent < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (sent <= 0) { *error = "模型运行时已退出"; stop(); return false; }
            offset += (size_t)sent;
        }
        auto first = begin;
        for (;;) {
            json::Value event;
            if (!read_event(&event, cancelled, 180)) {
                stop();
                *error = "模型运行时已停止或超时";
                return false;
            }
            if (const auto* failure = event.get("error")) {
                *error = failure->str;
                return false;
            }
            const auto delta = event.s("delta");
            if (!delta.empty()) {
                if (first == begin) first = std::chrono::steady_clock::now();
                write(delta);
            }
            if (!event.flag("done")) continue;
            stats->n_prompt = (int)event.n("prompt_tokens", 0);
            stats->n_generated = (int)event.n("completion_tokens", 0);
            stats->t_gen_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - first).count();
            stats->decode_tps = stats->t_gen_ms > 0 ? stats->n_generated * 1000.0 / stats->t_gen_ms : 0;
            *reason = event.s("finish_reason") == "length" ? "length" : "stop";
            return true;
        }
    }

private:
    using Time = std::chrono::steady_clock;
    static bool wait_for(int fd, short events, const std::function<bool()>& cancelled,
                         Time::time_point deadline) {
        while (!cancelled() && Time::now() < deadline) {
            pollfd p{fd, events, 0};
            int rc = poll(&p, 1, 100);
            if (rc > 0) return (p.revents & events) != 0;
            if (rc < 0 && errno != EINTR) return false;
        }
        return false;
    }

    bool read_event(json::Value* event, const std::function<bool()>& cancelled, int seconds) {
        auto deadline = Time::now() + std::chrono::seconds(seconds);
        for (;;) {
            const auto end = received_.find('\n');
            if (end != std::string::npos) {
                std::string line = received_.substr(0, end);
                received_.erase(0, end + 1);
                return json::parse(line, event) && event->type == json::Value::OBJ;
            }
            if (received_.size() > 1024 * 1024 || !wait_for(output_, POLLIN, cancelled, deadline)) return false;
            char buf[8192];
            const ssize_t n = read(output_, buf, sizeof(buf));
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) return false;
            received_.append(buf, (size_t)n);
        }
    }

    void stop() {
        if (input_ >= 0) close(input_);
        if (output_ >= 0) close(output_);
        input_ = output_ = -1;
        received_.clear();
        if (pid_ <= 0) return;
        kill(-pid_, SIGTERM);
        const auto deadline = Time::now() + std::chrono::seconds(2);
        int status = 0;
        pid_t child = 0;
        while (Time::now() < deadline) {
            child = waitpid(pid_, &status, WNOHANG);
            if (child != 0 && (child > 0 || errno != EINTR)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (child == 0) {
            kill(-pid_, SIGKILL);
            while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
        }
        pid_ = 0;
    }

    std::string script_, kind_, path_, projector_;
    int ctx_, threads_;
    pid_t pid_ = 0;
    int input_ = -1, output_ = -1;
    json::Value info_;
    std::string received_;
};
