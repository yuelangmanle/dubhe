// dubhed.cpp — 本地 HTTP 服务 + iOS 风格 Web UI 后端
//
// 端点：
//   GET  /                         UI
//   GET  /static/*                 静态资源
//   GET  /healthz
//   GET  /v1/models
//   POST /v1/chat/completions      OpenAI 兼容（stream / 非 stream）
//   GET  /api/state                运行时状态（硬件、模型、统计）
//   POST /api/load                 切换模型
//   POST /api/tokenize             调试用
#include <dubhe/dubhe.h>

#include "http.h"
#include "json.h"

#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct AppState {
    dubhe_engine* engine = nullptr;
    std::string model_path;
    std::string model_id;     // 对外暴露的短模型名
    std::string model_dir;
    std::string web_root;
    std::string workspace;
    std::string api_key;      // 为空表示不校验（仅建议本机使用）
    std::string lan_url;      // 局域网可访问的 base url
    bool allow_shell = false;
    std::mutex mu;
    std::atomic<bool> busy{false};
    long long total_requests = 0;
    long long total_tokens = 0;
    double last_decode_tps = 0;
};

AppState g;

// 取本机局域网 IPv4（用于"分享给手机用"）
std::string lan_ip() {
    struct ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return "";
    std::string best;
    for (auto* p = ifs; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        const std::string name = p->ifa_name;
        if (name == "lo0" || name.rfind("utun", 0) == 0 || name.rfind("awdl", 0) == 0) continue;
        char buf[64];
        auto* sin = (struct sockaddr_in*)p->ifa_addr;
        if (!inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) continue;
        const std::string ip = buf;
        if (ip.rfind("169.254.", 0) == 0) continue;
        // 优先 en0（Wi-Fi）
        if (name == "en0") return ip;
        if (best.empty()) best = ip;
    }
    freeifaddrs(ifs);
    return best;
}

// 从路径推导对外的模型 id：去目录、去扩展名
std::string model_id_of(const std::string& path) {
    size_t a = path.find_last_of('/');
    std::string n = (a == std::string::npos) ? path : path.substr(a + 1);
    if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) n.resize(n.size() - 5);
    return n.empty() ? "dubhe" : n;
}

std::string random_key() {
    static const char* cs = "abcdefghijkmnopqrstuvwxyz23456789";
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> d(0, 31);
    std::string s = "sk-dubhe-";
    for (int i = 0; i < 24; i++) s += cs[d(rng)];
    return s;
}

// ---------------------------------------------------------------- 工具函数
std::string read_file(const std::string& path, size_t max_bytes = 256 * 1024) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::string s;
    s.resize(max_bytes);
    f.read(&s[0], (std::streamsize)max_bytes);
    s.resize((size_t)f.gcount());
    return s;
}

bool write_file(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(data.data(), (std::streamsize)data.size());
    return f.good();
}

std::string guess_mime(const std::string& p) {
    auto ends = [&](const char* e) {
        size_t n = std::strlen(e);
        return p.size() >= n && p.compare(p.size() - n, n, e) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".js")) return "application/javascript; charset=utf-8";
    if (ends(".json")) return "application/json; charset=utf-8";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".woff2")) return "font/woff2";
    return "application/octet-stream";
}

// ---------------------------------------------------------------- 工具调用
struct ToolResult {
    bool ok = false;
    std::string output;
};

ToolResult run_tool(const std::string& name, const json::Value& args) {
    ToolResult r;
    if (name == "read_file") {
        std::string p = args.s("path");
        if (p.empty()) { r.output = "缺少 path 参数"; return r; }
        std::string content = read_file(p);
        if (content.empty()) { r.output = "文件为空或无法读取: " + p; return r; }
        r.ok = true;
        r.output = content;
        return r;
    }
    if (name == "list_dir") {
        std::string p = args.s("path", g.workspace);
        std::string cmd = "ls -la \"" + p + "\" 2>&1 | head -80";
        FILE* fp = popen(cmd.c_str(), "r");
        if (!fp) { r.output = "无法执行"; return r; }
        char buf[4096];
        while (fgets(buf, sizeof(buf), fp)) r.output += buf;
        pclose(fp);
        r.ok = true;
        return r;
    }
    if (name == "write_file") {
        std::string p = args.s("path");
        std::string c = args.s("content");
        if (p.empty()) { r.output = "缺少 path 参数"; return r; }
        r.ok = write_file(p, c);
        r.output = r.ok ? ("已写入 " + std::to_string(c.size()) + " 字节到 " + p) : ("写入失败: " + p);
        return r;
    }
    if (name == "shell") {
        if (!g.allow_shell) { r.output = "shell 工具未启用（启动时加 --allow-shell）"; return r; }
        std::string cmd = args.s("command");
        if (cmd.empty()) { r.output = "缺少 command 参数"; return r; }
        FILE* fp = popen((cmd + " 2>&1").c_str(), "r");
        if (!fp) { r.output = "无法执行"; return r; }
        char buf[4096];
        std::string out;
        while (fgets(buf, sizeof(buf), fp)) {
            out += buf;
            if (out.size() > 64 * 1024) break;
        }
        pclose(fp);
        r.ok = true;
        r.output = out;
        return r;
    }
    r.output = "未知工具: " + name;
    return r;
}

const char* kToolPrompt =
    "\n\n你可以调用以下工具来完成任务。需要调用时，输出一个 JSON 块，格式严格如下：\n"
    "<tool_call>{\"name\":\"工具名\",\"arguments\":{...}}</tool_call>\n"
    "一次只调用一个工具。工具结果会以 <tool_result> 的形式返回给你，然后你继续回答。\n"
    "可用工具：\n"
    "- read_file(path): 读取文本文件\n"
    "- list_dir(path): 列出目录内容\n"
    "- write_file(path, content): 写入文件\n"
    "- shell(command): 执行 shell 命令（可能被禁用）\n"
    "不需要工具时直接回答，不要输出 tool_call。";

// 从文本中提取第一个 <tool_call>...</tool_call>
bool extract_tool_call(const std::string& text, std::string* name, json::Value* args,
                       size_t* end_pos) {
    const std::string open = "<tool_call>";
    const std::string close = "</tool_call>";
    size_t a = text.find(open);
    if (a == std::string::npos) return false;
    size_t b = text.find(close, a);
    if (b == std::string::npos) return false;
    std::string body = text.substr(a + open.size(), b - a - open.size());
    json::Value v;
    if (!json::parse(body, &v)) return false;
    *name = v.s("name");
    if (auto* a2 = v.get("arguments")) *args = *a2;
    if (end_pos) *end_pos = b + close.size();
    return !name->empty();
}

// ---------------------------------------------------------------- ChatML
std::string build_prompt(const json::Value& messages, bool chatml, bool with_tools) {
    std::string p;
    if (!chatml) {
        for (auto& m : messages.arr) {
            std::string role = m.s("role");
            std::string content = m.s("content");
            if (role == "system") p += content + "\n\n";
            else if (role == "user") p += "User: " + content + "\n";
            else if (role == "assistant") p += "Assistant: " + content + "\n";
        }
        p += "Assistant: ";
        return p;
    }
    bool has_system = false;
    for (auto& m : messages.arr) {
        std::string role = m.s("role");
        std::string content = m.s("content");
        if (role == "system") {
            has_system = true;
            if (with_tools) content += kToolPrompt;
            p += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == "user") {
            p += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            p += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        } else if (role == "tool") {
            p += "<|im_start|>user\n<tool_result>" + content + "</tool_result><|im_end|>\n";
        }
    }
    if (!has_system && with_tools) {
        p = std::string("<|im_start|>system\n") + kToolPrompt + "<|im_end|>\n" + p;
    }
    p += "<|im_start|>assistant\n";
    return p;
}

std::string json_escape(const std::string& s) {
    json::Value v = json::Value::make_str(s);
    return json::dump(v);
}

// ---------------------------------------------------------------- SSE 流式上下文
// 放在文件作用域，避免在 dubhe_generate 的回调里引用被捕获的 lambda。
struct StreamCtx {
    http::Conn* c;
    std::string* acc;
    const std::string* id;
    long long created;
    bool first = true;
};

void emit_delta(StreamCtx* x, const std::string& delta) {
    std::string s = "data: {\"id\":\"" + *x->id +
                    "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                    std::to_string(x->created) + ",\"choices\":[{\"index\":0,\"delta\":{\"content\":" +
                    json_escape(delta) + "},\"finish_reason\":null}]}\n\n";
    x->c->write_chunk(s);
    x->first = false;
}

void on_piece(void* ud, int32_t /*tok*/, const char* piece) {
    auto* x = (StreamCtx*)ud;
    if (!piece) return;
    x->acc->append(piece);
    emit_delta(x, piece);
}

// /v1/* 的鉴权：设了 key 就必须带 Authorization: Bearer <key>
// 本机回环访问免鉴权——自己电脑上的 UI 不该再让人粘贴一次密钥。
bool authorized(const http::Request& req) {
    if (g.api_key.empty()) return true;
    if (req.remote == "127.0.0.1" || req.remote == "::1" || req.remote.empty()) return true;
    std::string h = req.header("authorization");
    const std::string p = "Bearer ";
    if (h.size() > p.size() && h.compare(0, p.size(), p) == 0) h = h.substr(p.size());
    else if (h.empty()) h = req.param("api_key");
    return h == g.api_key;
}

// /api/state 里的密钥只给本机看
bool is_local(const http::Request& req) {
    return req.remote == "127.0.0.1" || req.remote == "::1" || req.remote.empty();
}

std::string now_id() {
    static std::atomic<long long> n{0};
    return "chatcmpl-" + std::to_string((long long)time(nullptr)) + "-" + std::to_string(n++);
}

}  // namespace

// ---------------------------------------------------------------- handlers
int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 7777;
    std::string model;
    std::string web_root = "web";
    int ctx = 0;
    int threads = 0;
    bool no_gpu = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--port") port = atoi(next().c_str());
        else if (a == "--host") host = next();
        else if (a == "--web") web_root = next();
        else if (a == "-c" || a == "--ctx") ctx = atoi(next().c_str());
        else if (a == "-t" || a == "--threads") threads = atoi(next().c_str());
        else if (a == "--allow-shell") g.allow_shell = true;
        else if (a == "--no-gpu") no_gpu = true;
        else if (a == "--api-key") g.api_key = next();
        else if (a == "--share") {
            // 对外分享：监听所有网卡，并自动生成访问密钥
            host = "0.0.0.0";
            if (g.api_key.empty()) g.api_key = random_key();
        }
        else if (a == "-h" || a == "--help") {
            std::printf(
                "dubhed %s\n\n"
                "用法: dubhed -m <model.gguf> [--port 7777] [--host 127.0.0.1]\n"
                "             [--web web] [-c 4096] [-t N] [--allow-shell] [--no-gpu]\n"
                "             [--share] [--api-key <key>]\n\n"
                "  --share        监听 0.0.0.0 并自动生成 API Key（手机/其他设备可接入）\n"
                "  --api-key K    指定 API Key（/v1/* 需要 Authorization: Bearer K）\n",
                dubhe_version());
            return 0;
        }
    }

    // 监听非本机地址却不设密钥，等于把模型白送出去——自动补一个
    if (host != "127.0.0.1" && host != "localhost" && g.api_key.empty()) {
        g.api_key = random_key();
        std::fprintf(stderr, "提示: 已监听 %s，自动生成 API Key\n", host.c_str());
    }
    if (host != "127.0.0.1" && host != "localhost") {
        const std::string ip = lan_ip();
        if (!ip.empty()) g.lan_url = "http://" + ip + ":" + std::to_string(port);
    }

    g.web_root = web_root;
    g.workspace = ".";
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) g.workspace = cwd;

    dubhe_config cfg{};
    cfg.n_threads = threads;
    cfg.n_ctx = ctx;
    cfg.use_gpu = no_gpu ? 0 : 1;
    g.engine = dubhe_create(&cfg);
    if (!g.engine) { std::fprintf(stderr, "引擎创建失败\n"); return 1; }

    if (!model.empty()) {
        std::fprintf(stderr, "加载模型 %s …\n", model.c_str());
        auto t0 = std::chrono::steady_clock::now();
        if (dubhe_load_model(g.engine, model.c_str()) != DUBHE_OK) {
            std::fprintf(stderr, "加载失败: %s\n", dubhe_last_error());
            return 1;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        g.model_path = model;
        g.model_id = model_id_of(model);
        dubhe_model_info mi{};
        dubhe_model_info_get(g.engine, &mi);
        std::fprintf(stderr, "就绪 %.0f ms · %s · %s · %.2fB\n", (double)ms, mi.arch, mi.quant_desc,
                     (double)mi.n_params / 1e9);
    } else {
        std::fprintf(stderr, "未指定模型，UI 中可加载。\n");
    }

    http::Server srv;

    // ---- 静态资源
    auto serve_file = [](const http::Request& req, http::Conn& c, const std::string& root) {
        std::string p = req.path;
        if (p == "/" || p.empty()) p = "/index.html";
        std::string full = root + p;
        if (full.find("..") != std::string::npos) {
            c.send(400, "application/json", "{\"error\":\"bad path\"}");
            return;
        }
        std::string body = read_file(full, 8 * 1024 * 1024);
        if (body.empty()) {
            c.send(404, "application/json", "{\"error\":\"not found\"}");
            return;
        }
        c.send(200, guess_mime(full), body);
    };

    srv.get("/", [&](const http::Request& req, http::Conn& c) { serve_file(req, c, g.web_root); });
    srv.get("/index.html", [&](const http::Request& req, http::Conn& c) { serve_file(req, c, g.web_root); });
    srv.get("/app.js", [&](const http::Request& req, http::Conn& c) { serve_file(req, c, g.web_root); });
    srv.get("/app.css", [&](const http::Request& req, http::Conn& c) { serve_file(req, c, g.web_root); });
    srv.get("/favicon.svg", [&](const http::Request& req, http::Conn& c) { serve_file(req, c, g.web_root); });

    srv.get("/healthz", [](const http::Request&, http::Conn& c) {
        c.send(200, "application/json", "{\"ok\":true}");
    });

    srv.get("/api/state", [](const http::Request& req, http::Conn& c) {
        json::Value o = json::Value::make_obj();
        o.set("hardware", json::Value::make_str(dubhe_hardware_info()));
        o.set("version", json::Value::make_str(dubhe_version()));
        o.set("model", json::Value::make_str(g.model_path));
        o.set("model_id", json::Value::make_str(g.model_id.empty() ? "dubhe" : g.model_id));
        o.set("loaded", json::Value::make_bool(!g.model_path.empty()));
        o.set("workspace", json::Value::make_str(g.workspace));
        o.set("shell", json::Value::make_bool(g.allow_shell));
        o.set("api_key", json::Value::make_str(is_local(req) ? g.api_key : ""));
        o.set("lan_url", json::Value::make_str(g.lan_url));
        o.set("auth", json::Value::make_bool(!g.api_key.empty()));
        o.set("requests", json::Value::make_num((double)g.total_requests));
        o.set("tokens", json::Value::make_num((double)g.total_tokens));
        o.set("decode_tps", json::Value::make_num(g.last_decode_tps));
        if (!g.model_path.empty()) {
            dubhe_model_info mi{};
            if (dubhe_model_info_get(g.engine, &mi) == DUBHE_OK) {
                json::Value m = json::Value::make_obj();
                m.set("arch", json::Value::make_str(mi.arch));
                m.set("params", json::Value::make_num((double)mi.n_params));
                m.set("quant", json::Value::make_str(mi.quant_desc));
                m.set("n_ctx", json::Value::make_num(mi.n_ctx_train));
                m.set("n_layer", json::Value::make_num(mi.n_layer));
                m.set("n_embd", json::Value::make_num(mi.n_embd));
                m.set("n_vocab", json::Value::make_num(mi.n_vocab));
                m.set("chatml", json::Value::make_bool(mi.has_chat_template != 0));
                m.set("gpu", json::Value::make_bool(mi.is_gpu != 0));
                o.set("info", m);
            }
        }
        c.send(200, "application/json", json::dump(o));
    });

    srv.get("/v1/models", [](const http::Request& req, http::Conn& c) {
        if (!authorized(req)) { c.send(401, "application/json", "{\"error\":{\"message\":\"无效的 API Key\",\"type\":\"invalid_request_error\"}}"); return; }
        json::Value root = json::Value::make_obj();
        root.set("object", json::Value::make_str("list"));
        json::Value arr = json::Value::make_arr();
        json::Value m = json::Value::make_obj();
        m.set("id", json::Value::make_str(g.model_id.empty() ? "dubhe" : g.model_id));
        m.set("object", json::Value::make_str("model"));
        m.set("owned_by", json::Value::make_str("dubhe"));
        arr.arr.push_back(m);
        root.set("data", arr);
        c.send(200, "application/json", json::dump(root));
    });

    // ---- 核心：chat completions
    srv.post("/v1/chat/completions", [](const http::Request& req, http::Conn& c) {
        if (!authorized(req)) { c.send(401, "application/json", "{\"error\":{\"message\":\"无效的 API Key\",\"type\":\"invalid_request_error\"}}"); return; }
        if (g.model_path.empty()) {
            c.send(400, "application/json", "{\"error\":\"未加载模型\"}");
            return;
        }
        json::Value body;
        if (!json::parse(req.body, &body)) {
            c.send(400, "application/json", "{\"error\":\"JSON 解析失败\"}");
            return;
        }
        auto* msgs = body.get("messages");
        if (!msgs || msgs->type != json::Value::ARR) {
            c.send(400, "application/json", "{\"error\":\"缺少 messages\"}");
            return;
        }
        const bool stream = body.flag("stream", false);
        const bool with_tools = body.get("tools") != nullptr || body.flag("dubhe_tools", false);

        dubhe_model_info mi{};
        dubhe_model_info_get(g.engine, &mi);
        const bool chatml = mi.has_chat_template != 0;

        dubhe_sampling sp{};
        sp.max_tokens = (int)body.n("max_tokens", 512);
        sp.temperature = (float)body.n("temperature", 0.7);
        sp.top_k = (int)body.n("top_k", 40);
        sp.top_p = (float)body.n("top_p", 0.95);
        sp.repeat_penalty = (float)body.n("repeat_penalty", 1.1);
        sp.repeat_last_n = 64;

        std::string prompt = build_prompt(*msgs, chatml, with_tools);
        int32_t n = dubhe_tokenize(g.engine, prompt.c_str(), 1, nullptr, 0);
        std::vector<int32_t> toks((size_t)n);
        dubhe_tokenize(g.engine, prompt.c_str(), 1, toks.data(), n);

        const std::string id = now_id();
        long long created = (long long)time(nullptr);

        if (stream) {
            c.begin_stream();
            std::string acc;
            StreamCtx ctx{&c, &acc, &id, created};

            dubhe_stats st{};
            dubhe_generate(g.engine, toks.data(), n, &sp, on_piece, nullptr, &ctx, &st);

            // 工具调用：一轮
            std::string tname;
            json::Value targs;
            size_t endpos = 0;
            if (with_tools && extract_tool_call(acc, &tname, &targs, &endpos)) {
                std::string ev = "data: {\"dubhe\":{\"type\":\"tool_call\",\"name\":" + json_escape(tname) +
                                 ",\"arguments\":" + json::dump(targs) + "}}\n\n";
                c.write_chunk(ev);

                ToolResult tr = run_tool(tname, targs);
                std::string ev2 = "data: {\"dubhe\":{\"type\":\"tool_result\",\"ok\":" +
                                  std::string(tr.ok ? "true" : "false") + ",\"output\":" +
                                  json_escape(tr.output.substr(0, 8000)) + "}}\n\n";
                c.write_chunk(ev2);

                // 第二轮：带上工具结果
                json::Value msgs2 = *msgs;
                json::Value am = json::Value::make_obj();
                am.set("role", json::Value::make_str("assistant"));
                am.set("content", json::Value::make_str(acc));
                msgs2.arr.push_back(am);
                json::Value tm = json::Value::make_obj();
                tm.set("role", json::Value::make_str("tool"));
                tm.set("content", json::Value::make_str(tr.output));
                msgs2.arr.push_back(tm);

                std::string p2 = build_prompt(msgs2, chatml, with_tools);
                int32_t n2 = dubhe_tokenize(g.engine, p2.c_str(), 1, nullptr, 0);
                std::vector<int32_t> t2((size_t)n2);
                dubhe_tokenize(g.engine, p2.c_str(), 1, t2.data(), n2);

                dubhe_stats st2{};
                dubhe_generate(g.engine, t2.data(), n2, &sp, on_piece, nullptr, &ctx, &st2);
                st.n_generated += st2.n_generated;
                st.decode_tps = st2.decode_tps;
            }

            g.last_decode_tps = st.decode_tps;
            g.total_requests++;
            g.total_tokens += st.n_generated;

            std::string done = "data: {\"id\":\"" + id +
                               "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":" +
                               std::to_string(st.n_generated) + ",\"prompt_tokens\":" +
                               std::to_string(st.n_prompt) + ",\"decode_tps\":" +
                               std::to_string(st.decode_tps) + "}}\n\n";
            c.write_chunk(done);
            c.write_chunk("data: [DONE]\n\n");
            c.end_stream();
            return;
        }

        // 非流式
        std::string acc;
        dubhe_stats st{};
        struct Ctx2 { std::string* acc; } ctx2{&acc};
        dubhe_generate(
            g.engine, toks.data(), n, &sp,
            [](void* ud, int32_t, const char* piece) {
                auto* x = (Ctx2*)ud;
                if (piece) x->acc->append(piece);
            },
            nullptr, &ctx2, &st);

        g.last_decode_tps = st.decode_tps;
        g.total_requests++;
        g.total_tokens += st.n_generated;

        json::Value root = json::Value::make_obj();
        root.set("id", json::Value::make_str(id));
        root.set("object", json::Value::make_str("chat.completion"));
        root.set("created", json::Value::make_num((double)created));
        json::Value choices = json::Value::make_arr();
        json::Value ch = json::Value::make_obj();
        ch.set("index", json::Value::make_num(0));
        json::Value msg = json::Value::make_obj();
        msg.set("role", json::Value::make_str("assistant"));
        msg.set("content", json::Value::make_str(acc));
        ch.set("message", msg);
        ch.set("finish_reason", json::Value::make_str("stop"));
        choices.arr.push_back(ch);
        root.set("choices", choices);
        json::Value usage = json::Value::make_obj();
        usage.set("prompt_tokens", json::Value::make_num(st.n_prompt));
        usage.set("completion_tokens", json::Value::make_num(st.n_generated));
        usage.set("decode_tps", json::Value::make_num(st.decode_tps));
        root.set("usage", usage);
        c.send(200, "application/json", json::dump(root));
    });

    std::fprintf(stderr, "\n  Dubhe 已启动 →  http://%s:%d\n", host.c_str(), port);
    if (!g.lan_url.empty()) std::fprintf(stderr, "  局域网网关   %s/v1\n", g.lan_url.c_str());
    if (!g.api_key.empty()) std::fprintf(stderr, "  API Key      %s\n", g.api_key.c_str());
    std::fprintf(stderr, "\n");
    if (!srv.listen(host, port)) return 1;
    srv.run();
    return 0;
}
