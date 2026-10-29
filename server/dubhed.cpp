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
    std::string mmproj_path;  // 视觉/语音塔；空表示不支持多模态
    std::string tmp_dir;      // 多模态请求的落地目录
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

// 从 p 开始截取第一个配平的 {...}（考虑字符串内的括号与转义）
bool first_json_object(const std::string& text, size_t p, std::string* out, size_t* end) {
    if (p >= text.size() || text[p] != '{') return false;
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = p; i < text.size(); i++) {
        const char ch = text[i];
        if (in_str) {
            if (esc) esc = false;
            else if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') in_str = true;
        else if (ch == '{') depth++;
        else if (ch == '}') {
            if (--depth == 0) {
                *out = text.substr(p, i - p + 1);
                if (end) *end = i + 1;
                return true;
            }
        }
    }
    return false;
}

// 从文本中提取第一个工具调用。
// 两种形态都接受：<tool_call>{...}</tool_call> 和裸的 {"name":...}
bool extract_tool_call(const std::string& text, std::string* name, json::Value* args,
                       size_t* end_pos) {
    const std::string open = "<tool_call>";
    const std::string close = "</tool_call>";
    size_t a = text.find(open);
    if (a != std::string::npos) {
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
    // 裸 JSON：模型常常吐出 {"name":...} 后面还跟着一串解释文字，
    // 所以只截取第一个配平的对象来判断。
    size_t p = text.find_first_not_of(" \t\r\n");
    if (p == std::string::npos || text[p] != '{') return false;
    std::string obj;
    size_t obj_end = 0;
    if (!first_json_object(text, p, &obj, &obj_end)) return false;
    json::Value v;
    if (!json::parse(obj, &v)) return false;
    *name = v.s("name");
    if (name->empty()) return false;
    if (auto* a2 = v.get("arguments")) *args = *a2;
    if (end_pos) *end_pos = obj_end;
    return true;
}

// ---------------------------------------------------------------- ChatML
// ---------------------------------------------------------------- 多模态
//
// 走 OpenAI 的 content 数组格式（和手机端 SDK 一致）：
//   {"type":"text","text":"..."}
//   {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
//   {"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}
//   {"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}
//
// video_url 不是 OpenAI 官方字段（官方还没定视频协议），沿用社区的写法。
//
// 为什么落临时文件而不是加一套"从内存编码"的 ABI：塔的编码本来就是几百毫秒
// 起步，多一次几十 KB 的写盘可以忽略；而给 image/audio 各加一对 *_mem 接口
// 意味着两条预处理链路都要重写一遍（ImageIO 走 NSData、AVFoundation 走
// AVAudioFile），出错面远大于收益。
//
// 一次请求只接受一个媒体。引擎侧 dubhe_set_mm_embeds 只保存一段区间，
// 两段特征要两套前缀，等 Omni 那一轮改成多段再放开。

struct MediaPart {
    std::string kind;   // "image" / "audio" / "video"
    std::string path;   // 临时文件
};

std::string now_id();   // 定义在文件末尾

bool b64_decode(const std::string& in, std::string* out) {
    static int8_t tbl[256];
    static bool init = false;
    if (!init) {
        std::memset(tbl, -1, sizeof tbl);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) tbl[(uint8_t)a[i]] = (int8_t)i;
        init = true;
    }
    out->clear();
    out->reserve(in.size() * 3 / 4 + 4);
    int acc = 0, bits = 0;
    for (unsigned char ch : in) {
        if (ch == '=' || ch == '\n' || ch == '\r' || ch == ' ') continue;
        const int8_t v = tbl[ch];
        if (v < 0) return false;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out->push_back((char)((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// 支持 data URL 和本机绝对路径两种写法。
// 不直接抓 http(s)：服务端去访问外部 URL 会变成 SSRF 入口，而且局域网场景下
// 用户手里的图片本来就在本机。
bool load_media(const std::string& url, std::string* bytes, std::string* ext,
                std::string* err) {
    if (url.rfind("data:", 0) == 0) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos) { if (err) *err = "data URL 格式不对"; return false; }
        const std::string head = url.substr(5, comma - 5);
        if (head.find("base64") == std::string::npos) {
            if (err) *err = "只支持 base64 编码的 data URL";
            return false;
        }
        const size_t semi = head.find(';');
        const std::string mime = head.substr(0, semi == std::string::npos ? head.size() : semi);
        *ext = ".bin";
        if (mime == "image/png") *ext = ".png";
        else if (mime == "image/jpeg" || mime == "image/jpg") *ext = ".jpg";
        else if (mime == "image/heic") *ext = ".heic";
        else if (mime == "image/webp") *ext = ".webp";
        else if (mime == "image/gif") *ext = ".gif";
        else if (mime == "audio/wav" || mime == "audio/x-wav") *ext = ".wav";
        else if (mime == "audio/mpeg" || mime == "audio/mp3") *ext = ".mp3";
        else if (mime == "audio/mp4" || mime == "audio/m4a") *ext = ".m4a";
        else if (mime == "audio/flac") *ext = ".flac";
        else if (mime == "audio/aiff") *ext = ".aiff";
        else if (mime == "video/mp4") *ext = ".mp4";
        else if (mime == "video/quicktime") *ext = ".mov";
        else if (mime == "video/webm") *ext = ".webm";
        if (!b64_decode(url.substr(comma + 1), bytes)) {
            if (err) *err = "base64 解码失败";
            return false;
        }
        if (bytes->empty()) { if (err) *err = "data URL 里没有数据"; return false; }
        return true;
    }
    std::ifstream f(url, std::ios::binary);
    if (!f) { if (err) *err = "读不了这个路径: " + url; return false; }
    bytes->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    const size_t dot = url.find_last_of('.');
    *ext = dot == std::string::npos ? ".bin" : url.substr(dot);
    return !bytes->empty();
}

// 把 content 数组摊平成"文本 + 占位符"，媒体落到临时文件。
// 只保留最后一条 user 消息里的媒体（多轮历史里的旧媒体没有对应特征，
// 塞进去只会让模型看到一堆没有内容的占位符）。
json::Value flatten_messages(const json::Value& messages, std::vector<MediaPart>* media,
                             std::string* err) {
    json::Value out = json::Value::make_arr();
    size_t last_user = 0;
    for (size_t i = 0; i < messages.arr.size(); i++)
        if (messages.arr[i].s("role") == "user") last_user = i;

    for (size_t i = 0; i < messages.arr.size(); i++) {
        const auto& m = messages.arr[i];
        json::Value nm = json::Value::make_obj();
        nm.set("role", json::Value::make_str(m.s("role")));
        const json::Value* content = m.get("content");
        if (!content) { nm.set("content", json::Value::make_str("")); out.arr.push_back(nm); continue; }

        if (content->type != json::Value::ARR) {
            // 字符串形式：原样带走。注意不能写成 content->s("")，那是"取名为空
            // 串的子键"，对 STR 类型的 Value 恒返回默认值，会把正文整个吃掉。
            nm.set("content", json::Value::make_str(
                                  content->type == json::Value::STR ? content->str : ""));
            out.arr.push_back(nm);
            continue;
        }
        // 数组形式：拼文本，媒体变占位符
        std::string text;
        for (const auto& part : content->arr) {
            const std::string type = part.s("type");
            if (type == "text") {
                text += part.s("text");
            } else if (type == "image_url" || type == "input_audio" || type == "video_url") {
                const char* ph = (type == "image_url") ? "<|image|>"
                                 : (type == "video_url" ? "<|video|>" : "<|audio|>");
                const std::string kind = (type == "image_url") ? "image"
                                         : (type == "video_url" ? "video" : "audio");
                if (i != last_user) { text += ph; continue; }   // 历史消息里的媒体：只留占位
                if (!media->empty()) {
                    if (err) *err = "一次请求只支持一个媒体（图像 / 视频 / 音频）";
                    return json::Value::make_obj();
                }
                std::string url, b64, fmt;
                if (const json::Value* iu = part.get("image_url")) url = iu->s("url");
                if (const json::Value* vu = part.get("video_url")) {
                    url = vu->s("url");
                    if (url.empty()) url = vu->s("path");
                }
                if (const json::Value* ia = part.get("input_audio")) {
                    b64 = ia->s("data");
                    fmt = ia->s("format");
                    if (!b64.empty() && b64.rfind("data:", 0) != 0)
                        url = "data:audio/" + (fmt.empty() ? std::string("wav") : fmt) +
                              ";base64," + b64;
                }
                if (url.empty()) { if (err) *err = std::string("缺少 ") + type + " 数据"; return json::Value::make_obj(); }

                std::string bytes, ext, lerr;
                if (!load_media(url, &bytes, &ext, &lerr)) { if (err) *err = lerr; return json::Value::make_obj(); }
                std::string dir = g.tmp_dir.empty() ? "/tmp" : g.tmp_dir;
                std::string path = dir + "/dubhe_mm_" + now_id() + ext;
                std::ofstream of(path, std::ios::binary);
                if (!of) { if (err) *err = "写不了临时文件: " + path; return json::Value::make_obj(); }
                of.write(bytes.data(), (std::streamsize)bytes.size());
                of.close();
                media->push_back({kind, path});
                text += ph;
            }
        }
        nm.set("content", json::Value::make_str(text));
        out.arr.push_back(nm);
    }
    return out;
}

// 把 prompt 里单个占位符展开成 n 个，返回展开后的注入起点。
// 和 CLI 里的 expand_mm_token 是同一套逻辑：展开后的下标才是注入位置。
bool expand_placeholder(dubhe_engine* e, std::vector<int32_t>* toks, int n,
                        const char* ph_text, int32_t* at) {
    int32_t ph = -1;
    if (dubhe_tokenize(e, ph_text, 0, &ph, 1) != 1) return false;
    size_t pos = toks->size();
    for (size_t i = 0; i < toks->size(); i++)
        if ((*toks)[i] == ph) { pos = i; break; }
    if (pos == toks->size()) return false;
    toks->erase(toks->begin() + (long)pos);
    toks->insert(toks->begin() + (long)pos, (size_t)n, ph);
    *at = (int32_t)pos;
    return true;
}

// LFM 系的 chat 模板第一行是 `{{- bos_token -}}`，但 GGUF 的 add_bos_token
// 是 false。少了它模型不进入对话状态（详见 docs/02-进度书.md §9.10）。
void prepend_bos(dubhe_engine* e, std::vector<int32_t>* toks) {
    int32_t id = -1;
    if (dubhe_tokenize(e, "<|startoftext|>", 0, &id, 1) == 1)
        toks->insert(toks->begin(), id);
}

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

// 请求级的多模态缓存。工具调用会走第二轮生成，而视觉塔编一张图要十几秒，
// 不能因为要多生成一轮就把同一张图重编一遍。
struct MmCache {
    bool valid = false;
    std::string kind;
    std::vector<float> embeds;
    int32_t n_tokens = 0;
    int32_t dim = 0;
};

// messages → token 序列：摊平 content 数组 → 分词 → 补 BOS → 编码媒体 →
// 展开占位符 → 注入特征。返回 false 时 err 里是可以直接回给客户端的原因。
//
// force_raw：不加 chat 模板，直接续写。语音模型特别吃这一套 —— 见下面的判定。
bool build_tokens(dubhe_engine* e, const json::Value& messages, bool chatml, bool with_tools,
                  bool force_raw, std::vector<int32_t>* toks, MmCache* cache,
                  std::string* err) {
    std::vector<MediaPart> media;
    json::Value flat = flatten_messages(messages, &media, err);
    if (flat.type != json::Value::ARR) return false;

    // 只给了一段音频、一个字都没打 —— 用户要的就是转写。
    // 这时候走 chat 模板反而害事：LFM3-Audio 进了对话状态后会把音频当成
    // "待推理的问题"，输出 "We need to answer the question: …" 这种分析过程，
    // 转写被埋在里面。不加模板直接续写，第一句就是转写正文。
    bool raw = force_raw;
    if (!raw && media.size() == 1 && media[0].kind == "audio") {
        std::string t;
        for (auto& m : flat.arr)
            if (m.s("role") == "user") t = m.s("content");
        for (const char* ph : {"<|audio|>", "<|image|>", "<|video|>"}) {
            size_t p;
            while ((p = t.find(ph)) != std::string::npos) t.erase(p, std::strlen(ph));
        }
        const size_t a = t.find_first_not_of(" \t\r\n");
        raw = (a == std::string::npos);
    }

    std::string prompt;
    if (raw) {
        for (auto& m : flat.arr)
            if (m.s("role") == "user") prompt = m.s("content");
    } else {
        prompt = build_prompt(flat, chatml, with_tools);
    }

    const int32_t n = dubhe_tokenize(e, prompt.c_str(), 1, nullptr, 0);
    if (n <= 0) { if (err) *err = "分词失败"; return false; }
    toks->assign((size_t)n, 0);
    if (dubhe_tokenize(e, prompt.c_str(), 1, toks->data(), n) != n) {
        if (err) *err = "分词失败";
        return false;
    }
    // raw 也必须补 BOS。不补的话模型不知道自己站在序列开头，转写内容是对的，
    // 但一句接一句复读到 max_tokens 也不吐 EOS；补上之后它写完就停。
    prepend_bos(e, toks);

    // 上一轮请求可能留了特征，不主动清掉的话，这一轮纯文本也会被覆盖。
    dubhe_clear_mm_embeds(e);
    if (media.empty()) return true;

    // ---- 编码：一次请求只做一遍，第二轮复用
    if (!cache->valid) {
        const MediaPart& mp = media[0];
        float* emb = nullptr;
        int32_t n_mm = 0, dim = 0;
        int rc;
        if (mp.kind == "image") {
            rc = dubhe_encode_image(e, mp.path.c_str(), &emb, &n_mm, &dim);
        } else if (mp.kind == "video") {
            // 2 fps / 16 帧 / 长边 448：默认值在"看得清动作"和"编得完"之间取的折中
            rc = dubhe_encode_video(e, mp.path.c_str(), 2.0, 16, 448, &emb, &n_mm, &dim);
        } else {
            rc = dubhe_encode_audio(e, mp.path.c_str(), &emb, &n_mm, &dim);
        }
        if (rc != DUBHE_OK) {
            if (err) *err = std::string("媒体编码失败: ") + dubhe_last_error();
            return false;
        }
        cache->kind = mp.kind;
        cache->embeds.assign(emb, emb + (size_t)n_mm * (size_t)dim);
        cache->n_tokens = n_mm;
        cache->dim = dim;
        cache->valid = true;
        dubhe_free_embeds(emb);
    }
    // 特征已经拷进内存，临时文件可以删了（base64 的图动辄几 MB，不删会攒满 /tmp）
    for (auto& mp : media) std::remove(mp.path.c_str());

    const char* ph = (cache->kind == "image") ? "<|image|>"
                     : (cache->kind == "video" ? "<|video|>" : "<|audio|>");
    int32_t at = 0;
    if (!expand_placeholder(e, toks, cache->n_tokens, ph, &at)) {
        // 分清两种失败：词表里压根没这个 token（语言模型不支持该模态），
        // 还是词表有但 prompt 里没写。前者是配置问题，后者是调用方问题。
        int32_t probe = -1;
        const bool known = dubhe_tokenize(e, ph, 0, &probe, 1) == 1;
        if (err) {
            *err = known ? (std::string("prompt 里没有 ") + ph + " 占位符")
                         : (std::string("这个语言模型的词表里没有 ") + ph +
                            "，它不支持该模态（塔能加载，但语言模型接不住）");
        }
        return false;
    }
    if (dubhe_set_mm_embeds(e, cache->embeds.data(), cache->n_tokens, at, cache->dim) != DUBHE_OK) {
        if (err) *err = std::string("特征注入失败: ") + dubhe_last_error();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- SSE 流式上下文
// 放在文件作用域，避免在 dubhe_generate 的回调里引用被捕获的 lambda。
struct StreamCtx {
    http::Conn* c;
    std::string* acc;
    const std::string* id;
    long long created;
    bool first = true;
    // 工具调用判定：模型有时会直接吐 {"name":...} 而不加 <tool_call> 包裹，
    // 那样原始 JSON 会先流到用户眼前。所以先攒几个字符再决定要不要抑制。
    bool decided = false;
    bool suppress = false;
    std::string pending;
};

void emit_delta(StreamCtx* x, const std::string& delta) {
    std::string s = "data: {\"id\":\"" + *x->id +
                    "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                    std::to_string(x->created) + ",\"choices\":[{\"index\":0,\"delta\":{\"content\":" +
                    json_escape(delta) + "},\"finish_reason\":null}]}\n\n";
    x->c->write_chunk(s);
    x->first = false;
}

// 前 16 个字符足以判断"这是不是一段工具调用"
void on_piece(void* ud, int32_t /*tok*/, const char* piece) {
    auto* x = (StreamCtx*)ud;
    if (!piece) return;
    x->acc->append(piece);
    if (x->suppress) return;

    if (x->decided) {
        emit_delta(x, piece);
        return;
    }
    x->pending += piece;
    // 去掉前导空白后再判断
    size_t i = 0;
    while (i < x->pending.size() && (x->pending[i] == ' ' || x->pending[i] == '\n' ||
                                     x->pending[i] == '\r' || x->pending[i] == '\t')) i++;
    const std::string head = x->pending.substr(i);
    if (head.empty()) return;  // 还全是空白，继续等
    const bool looks_tool = head.rfind("<tool_call", 0) == 0 || head.rfind("{\"name\"", 0) == 0 ||
                            head.rfind("{ \"name\"", 0) == 0;
    if (looks_tool) {
        x->suppress = true;
        x->decided = true;
        x->pending.clear();
        return;
    }
    if (head.size() >= 16) {
        x->decided = true;
        const std::string all = x->pending;
        x->pending.clear();
        emit_delta(x, all);
    }
}

// 流结束后把还没吐出去的缓冲补发
void flush_pending(StreamCtx* x) {
    if (!x->decided && !x->pending.empty()) {
        const std::string all = x->pending;
        x->pending.clear();
        x->decided = true;
        emit_delta(x, all);
    }
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
    std::vector<std::string> mmprojs;

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
        else if (a == "--mmproj") mmprojs.push_back(next());
        else if (a == "--tmp-dir") g.tmp_dir = next();
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
                "             [--mmproj <mmproj.gguf>] [--tmp-dir /tmp]\n"
                "             [--share] [--api-key <key>]\n\n"
                "  --mmproj P     加载视觉塔或语音塔（可给多次，Omni 模型两个都要）\n"
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

    if (g.tmp_dir.empty()) g.tmp_dir = "/tmp";

    // 多模态塔加载失败不该让整个服务起不来：纯文本照常可用，只是 UI 上
    // 不显示附件入口（/api/state 的 mm 字段会是 0）。
    // --mmproj 可以给多次：Omni 模型是视觉塔 + 语音塔两个独立文件。
    for (const auto& mp : mmprojs) {
        std::fprintf(stderr, "加载多模态塔 %s …\n", mp.c_str());
        auto t0 = std::chrono::steady_clock::now();
        if (dubhe_load_mmproj(g.engine, mp.c_str()) != DUBHE_OK) {
            std::fprintf(stderr, "多模态塔加载失败: %s（跳过这个塔）\n", dubhe_last_error());
            continue;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        const int caps = dubhe_mm_caps(g.engine);
        std::fprintf(stderr, "  → 就绪 %.0f ms · 现有能力: %s\n", (double)ms,
                     caps == 3 ? "图像/视频 + 语音" : (caps & 2 ? "语音" : "图像/视频"));
        if (!g.mmproj_path.empty()) g.mmproj_path += ",";
        g.mmproj_path += mp;
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
        // 多模态能力：0 无 / 1 图像+视频 / 2 语音 / 3 两者都有
        const int caps = dubhe_mm_caps(g.engine);
        o.set("mm", json::Value::make_num(caps));
        o.set("mm_vision", json::Value::make_bool((caps & 1) != 0));
        o.set("mm_audio", json::Value::make_bool((caps & 2) != 0));
        o.set("mmproj", json::Value::make_str(g.mmproj_path));
        if (!g.model_path.empty()) {
            dubhe_model_info mi{};
            if (dubhe_model_info_get(g.engine, &mi) == DUBHE_OK) {
                json::Value m = json::Value::make_obj();
                m.set("arch", json::Value::make_str(mi.arch));
                m.set("params", json::Value::make_num((double)mi.n_params));
                m.set("quant", json::Value::make_str(mi.quant_desc));
                m.set("n_ctx", json::Value::make_num(mi.n_ctx_limit));
                m.set("n_ctx_train", json::Value::make_num(mi.n_ctx_train));
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

        // 只允许一个媒体：视觉塔和语音塔是两个独立的 mmproj，一次注入两段特征
        // 需要两套前缀，留到 Omni 那一轮再做。
        MmCache mcache;
        std::vector<int32_t> toks;
        std::string berr;
        const bool force_raw = body.flag("dubhe_raw", false);
        if (!build_tokens(g.engine, *msgs, chatml, with_tools, force_raw, &toks, &mcache, &berr)) {
            c.send(400, "application/json",
                   "{\"error\":{\"message\":" + json_escape(berr) +
                       ",\"type\":\"invalid_request_error\"}}");
            return;
        }
        const int32_t n = (int32_t)toks.size();

        const std::string id = now_id();
        long long created = (long long)time(nullptr);

        if (stream) {
            c.begin_stream();
            std::string acc;
            StreamCtx ctx{&c, &acc, &id, created, true, false, false, {}};

            dubhe_stats st{};
            dubhe_generate(g.engine, toks.data(), n, &sp, on_piece, nullptr, &ctx, &st);
            flush_pending(&ctx);

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

                // 第二轮复用第一轮编好的特征：图片不重编，只重新分词
                std::vector<int32_t> t2;
                std::string e2;
                if (!build_tokens(g.engine, msgs2, chatml, with_tools, force_raw, &t2, &mcache, &e2)) {
                    c.write_chunk("data: {\"dubhe\":{\"type\":\"error\",\"message\":" +
                                  json_escape(e2) + "}}\n\n");
                }

                // 第二轮重新判定：这次应该是正常回答了
                ctx.decided = false;
                ctx.suppress = false;
                ctx.pending.clear();
                dubhe_stats st2{};
                if (!t2.empty())
                    dubhe_generate(g.engine, t2.data(), (int32_t)t2.size(), &sp, on_piece, nullptr,
                                   &ctx, &st2);
                flush_pending(&ctx);
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

        // 非流式 + 工具：按 OpenAI 协议把 tool_calls 交给客户端执行，
        // 而不是在服务端偷偷跑一遍再拼一段文本回去。
        std::string tname;
        json::Value targs;
        size_t endpos = 0;
        if (with_tools && extract_tool_call(acc, &tname, &targs, &endpos)) {
            msg.set("content", json::Value::make_str(""));
            json::Value calls = json::Value::make_arr();
            json::Value call = json::Value::make_obj();
            call.set("id", json::Value::make_str("call_" + std::to_string(created)));
            call.set("type", json::Value::make_str("function"));
            json::Value fn = json::Value::make_obj();
            fn.set("name", json::Value::make_str(tname));
            fn.set("arguments", json::Value::make_str(json::dump(targs)));
            call.set("function", fn);
            calls.arr.push_back(call);
            msg.set("tool_calls", calls);
            ch.set("finish_reason", json::Value::make_str("tool_calls"));
        } else {
            msg.set("content", json::Value::make_str(acc));
            ch.set("finish_reason", json::Value::make_str("stop"));
        }
        ch.set("message", msg);
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
