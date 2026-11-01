// dubhed.cpp — 本地 HTTP 服务 + iOS 风格 Web UI 后端
//
// 端点：
//   GET  /                         UI
//   GET  /static/*                 静态资源
//   GET  /healthz
//   GET  /v1/models
//   POST /v1/chat/completions      OpenAI 兼容（stream / 非 stream）
//   GET  /api/state                运行时状态（硬件、模型、统计）
//   GET  /api/models               本地模型清单（扫描 --model-dir）
//   POST /api/load                 切换模型（卸载 → 加载 → 失败自动装回旧的）
//   GET  /api/files                列目录（JSON）
//   GET  /api/files/read           读文本文件
//   GET  /api/tools                工具清单
//   POST /api/tools/run            手动执行一个工具
//   GET  /api/kb                   知识库文档与统计
//   POST /api/kb/add               索引文件/目录
//   POST /api/kb/remove|clear|rag  移除 / 清空 / 开关自动注入
//   POST /api/kb/search            BM25 检索
#include <dubhe/dubhe.h>

#include "http.h"
#include "json.h"

#include <ifaddrs.h>
#include <dirent.h>
#include <limits.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
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
#include <unordered_map>
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

// ---------------------------------------------------------------- 路径
//
// UI 的「文件」和「知识库」两个视图要能浏览工作区，所以这里给一组
// 最小但完备的路径工具。安全边界只有一条：**不许出现 ".."**，并且
// 解析后的绝对路径必须落在白名单根目录之下。用 realpath 而不是字符串
// 比较 —— 符号链接可以把 /tmp/x 指到 /etc，字符串前缀检查拦不住。

bool is_dir_p(const std::string& p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
bool is_file_p(const std::string& p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}
std::string path_base(const std::string& p) {
    const size_t a = p.find_last_of('/');
    return (a == std::string::npos) ? p : p.substr(a + 1);
}
std::string path_dir(const std::string& p) {
    const size_t a = p.find_last_of('/');
    if (a == std::string::npos) return ".";
    if (a == 0) return "/";
    return p.substr(0, a);
}

// 相对路径按工作区解析；~ 展开成家目录
std::string path_abs(const std::string& p) {
    if (p.empty() || p == "." || p == "./") return g.workspace;
    if (p[0] == '~') {
        const char* h = getenv("HOME");
        return std::string(h ? h : "") + p.substr(1);
    }
    if (p[0] == '/') return p;
    return path_join(g.workspace, p);
}

// 白名单根：工作区、模型目录、家目录下的 .dubhe。realpath 之后比较。
bool path_allowed(const std::string& p) {
    if (p.find("..") != std::string::npos) return false;
    char buf[PATH_MAX];
    std::string real = p;
    if (realpath(p.c_str(), buf)) real = buf;
    // 不存在的新文件：退到父目录再判
    if (!is_file_p(p) && !is_dir_p(p)) {
        std::string par = path_dir(p);
        if (realpath(par.c_str(), buf)) real = path_join(buf, path_base(p));
        else return false;
    }
    const char* h = getenv("HOME");
    std::vector<std::string> roots = {g.workspace};
    if (h) {
        roots.push_back(std::string(h));
        roots.push_back(std::string(h) + "/.dubhe");
    }
    if (!g.model_dir.empty()) roots.push_back(g.model_dir);
    if (!g.tmp_dir.empty()) roots.push_back(g.tmp_dir);
    for (const auto& r : roots) {
        if (r.empty()) continue;
        std::string rr = r;
        char rb[PATH_MAX];
        if (realpath(r.c_str(), rb)) rr = rb;
        if (real == rr) return true;
        if (real.size() > rr.size() && real.compare(0, rr.size(), rr) == 0 &&
            real[rr.size()] == '/')
            return true;
    }
    return false;
}

// ---------------------------------------------------------------- 文件浏览
const char* kTextExt[] = {".txt", ".md",  ".markdown", ".json", ".jsonl", ".yaml", ".yml",
                          ".toml", ".csv", ".tsv",      ".c",    ".h",     ".cc",   ".cpp",
                          ".hpp", ".mm",  ".py",        ".js",   ".ts",    ".html", ".css",
                          ".sh",  ".rs",  ".go",        ".java", ".kt",    ".swift", ".log",
                          ".ini", ".conf", ".xml",      ".sql",  ".gitignore"};

bool has_text_ext(const std::string& p) {
    std::string low = p;
    for (auto& c : low) c = (char)std::tolower((unsigned char)c);
    for (const char* e : kTextExt) {
        const size_t n = std::strlen(e);
        if (low.size() >= n && low.compare(low.size() - n, n, e) == 0) return true;
    }
    return false;
}

// UTF-8 边界。中文一个字 3 字节，按字节切块/截断一定会切在字中间，
// 而 json::dump 是原样输出的 —— 半个汉字出去就是非法 UTF-8，前端
// JSON.parse 直接抛错。这两个小函数是唯一可靠的挡板。
size_t utf8_next(const std::string& s, size_t i) {
    while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) i++;
    return i;
}
size_t utf8_back(const std::string& s, size_t i) {
    if (i > s.size()) i = s.size();
    while (i > 0 && i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
    return i;
}

// 读文本：超过上限就截断。二进制文件直接拒绝（用 0 字节判断）。
std::string read_text_file(const std::string& p, size_t max_bytes, bool* truncated) {
    if (truncated) *truncated = false;
    std::ifstream f(p, std::ios::binary);
    if (!f) return "";
    std::string s;
    s.resize(max_bytes);
    f.read(&s[0], (std::streamsize)max_bytes);
    const size_t got = (size_t)f.gcount();
    s.resize(got);
    if (f.peek() != EOF && truncated) *truncated = true;
    // 截断点落在汉字中间的话，退到字符起点
    if (got == max_bytes) s.resize(utf8_back(s, got));
    return s;
}

// ---------------------------------------------------------------- 知识库
//
// 为什么是 BM25 而不是向量检索
// ---------------------------
// 本地没有 embedding 模型。为了做 RAG 硬塞一个 300 MB 的 bge 进来，
// 在 1.7B 的模型旁边等于多背一个 1/4 大小的权重，而收益只是"语义相近"。
// BM25 是零模型、零显存、结果可解释（能指出命中了哪几个词）的，
// 对"哪段话里有这个词"这类问题足够。真要做向量检索，接口形状不用变。
//
// 中文没有空格，用 bigram 切："本地推理" → 本地 / 地推 / 推理。
// 这不是分词，但对 BM25 够用 —— 召回偏高、精度略低，配合长度归一化后
// 排序仍然稳。英文和数字按 [a-z0-9_]+ 切，统一小写。
//
// 索引不落盘，只落"文档清单"。启动时按清单重读文件重建 —— 几十个文件
// 只要几十毫秒，而把倒排表序列化出去既容易和源文件不同步，又得处理
// 版本兼容。源文件才是唯一真源。

struct KbDoc {
    std::string path;
    long long mtime = 0;
    long long bytes = 0;
    int chunks = 0;
    bool missing = false;
};

struct KbChunk {
    int doc = 0;
    int len = 0;          // 词数，BM25 的长度归一化用
    size_t off = 0;       // 在原文里的起点（做 snippet 用）
    std::string text;
};

std::vector<KbDoc> g_kb_docs;
std::vector<KbChunk> g_kb_chunks;
std::unordered_map<std::string, std::vector<std::pair<int, int>>> g_kb_post;  // term → (chunk, tf)
double g_kb_avg = 1.0;
std::mutex g_kb_mu;
bool g_kb_loaded = false;
bool g_kb_rag = true;   // 是否把检索结果自动注入对话

std::vector<std::string> kb_tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string word;
    std::vector<std::string> cjk;
    auto flush_word = [&]() {
        if (!word.empty()) {
            out.push_back(word);
            word.clear();
        }
    };
    auto flush_cjk = [&]() {
        if (cjk.empty()) return;
        if (cjk.size() == 1) {
            out.push_back(cjk[0]);
        } else {
            for (size_t j = 0; j + 1 < cjk.size(); j++) out.push_back(cjk[j] + cjk[j + 1]);
        }
        cjk.clear();
    };
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            flush_cjk();
            if (std::isalnum(c) || c == '_') word += (char)std::tolower(c);
            else flush_word();
            i++;
            continue;
        }
        flush_word();
        const int n = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        if (i + (size_t)n > s.size()) break;
        cjk.push_back(s.substr(i, (size_t)n));
        i += (size_t)n;
    }
    flush_word();
    flush_cjk();
    return out;
}

// 切块：约 700 字节一块、重叠 120 字节。断点优先段落，其次句号。
// 重叠是必要的 —— 一个答案横跨切点时，没有重叠就两边都搜不到。
std::vector<std::pair<size_t, std::string>> kb_split(const std::string& text) {
    std::vector<std::pair<size_t, std::string>> out;
    const size_t kMax = 700, kOverlap = 120;
    size_t i = 0;
    while (i < text.size()) {
        size_t end = std::min(text.size(), i + kMax);
        if (end < text.size()) {
            size_t p = text.rfind('\n', end);
            if (p == std::string::npos || p < i + kMax / 3) p = text.rfind("。", end);
            if (p == std::string::npos || p < i + kMax / 3) p = text.rfind('.', end);
            if (p != std::string::npos && p > i + kMax / 3) end = p + 1;
        }
        size_t a = i, b = end;
        while (a < b && std::isspace((unsigned char)text[a])) a++;
        while (b > a && std::isspace((unsigned char)text[b - 1])) b--;
        // 断点可能落在 '。' 的第 2 字节上，往前退到字符起点
        b = utf8_back(text, b);
        a = utf8_next(text, a);
        if (b > a) out.emplace_back(a, text.substr(a, b - a));
        if (end >= text.size()) break;
        const size_t prev = i;
        i = (end > kOverlap) ? end - kOverlap : end;
        if (i <= prev) i = end;       // 重叠量比前进量还大时退化成不重叠
        if (i <= prev) break;         // 兜底：绝不原地打转
    }
    return out;
}

std::string kb_manifest_path() {
    const char* h = getenv("HOME");
    return std::string(h ? h : "/tmp") + "/.dubhe/kb/docs.tsv";
}

void kb_save_manifest() {
    const std::string p = kb_manifest_path();
    ::mkdir(path_dir(p).c_str(), 0755);
    ::mkdir(path_dir(path_dir(p)).c_str(), 0755);
    std::ofstream f(p, std::ios::trunc);
    if (!f) return;
    for (const auto& d : g_kb_docs)
        f << d.path << '\t' << d.mtime << '\t' << d.bytes << '\n';
}

void kb_load_manifest() {
    std::ifstream f(kb_manifest_path());
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        const size_t a = line.find('\t');
        if (a == std::string::npos) continue;
        const size_t b = line.find('\t', a + 1);
        KbDoc d;
        d.path = line.substr(0, a);
        d.mtime = atoll(line.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1).c_str());
        if (b != std::string::npos) d.bytes = atoll(line.substr(b + 1).c_str());
        g_kb_docs.push_back(std::move(d));
    }
}

// 全量重建。文档数在几十的量级，重建比增量维护简单得多，也不会不同步。
void kb_rebuild() {
    g_kb_chunks.clear();
    g_kb_post.clear();
    double total = 0;
    for (size_t di = 0; di < g_kb_docs.size(); di++) {
        KbDoc& d = g_kb_docs[di];
        d.chunks = 0;
        d.missing = false;
        struct stat st {};
        if (stat(d.path.c_str(), &st) != 0) {
            d.missing = true;
            continue;
        }
        bool tr = false;
        const std::string txt = read_text_file(d.path, 2 * 1024 * 1024, &tr);
        if (txt.empty()) {
            d.missing = true;
            continue;
        }
        for (auto& seg : kb_split(txt)) {
            KbChunk ch;
            ch.doc = (int)di;
            ch.off = seg.first;
            ch.text = seg.second;
            const auto toks = kb_tokenize(ch.text);
            ch.len = (int)toks.size();
            if (ch.len == 0) continue;
            const int ci = (int)g_kb_chunks.size();
            std::unordered_map<std::string, int> tf;
            for (const auto& t : toks) tf[t]++;
            for (const auto& kv : tf) g_kb_post[kv.first].emplace_back(ci, kv.second);
            total += ch.len;
            d.chunks++;
            g_kb_chunks.push_back(std::move(ch));
        }
    }
    g_kb_avg = g_kb_chunks.empty() ? 1.0 : total / (double)g_kb_chunks.size();
}

struct KbHit {
    int chunk = 0;
    double score = 0;
    std::string snippet;
    std::string path;
};

std::vector<KbHit> kb_search(const std::string& q, int topk) {
    std::vector<KbHit> hits;
    if (g_kb_chunks.empty()) return hits;
    auto toks = kb_tokenize(q);
    if (toks.empty()) return hits;
    std::vector<std::string> uniq;
    for (auto& t : toks)
        if (std::find(uniq.begin(), uniq.end(), t) == uniq.end()) uniq.push_back(t);

    const double k1 = 1.2, b = 0.75;
    const int N = (int)g_kb_chunks.size();
    std::unordered_map<int, double> score;
    std::unordered_map<int, std::string> best_term;
    for (const auto& t : uniq) {
        auto it = g_kb_post.find(t);
        if (it == g_kb_post.end()) continue;
        const double df = (double)it->second.size();
        const double idf = std::log(1.0 + (N - df + 0.5) / (df + 0.5));
        for (const auto& [ci, tf] : it->second) {
            const double len = (double)g_kb_chunks[ci].len;
            const double s = idf * ((double)tf * (k1 + 1.0)) /
                             ((double)tf + k1 * (1.0 - b + b * len / g_kb_avg));
            score[ci] += s;
            // snippet 以第一个命中的查询词为中心 —— 查询词顺序本身就是
            // 用户心里的权重顺序，不必再挑 IDF 最大的那个
            if (best_term.find(ci) == best_term.end()) best_term[ci] = t;
        }
    }
    std::vector<std::pair<double, int>> order;
    order.reserve(score.size());
    for (const auto& kv : score) order.emplace_back(kv.second, kv.first);
    std::sort(order.begin(), order.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    if ((int)order.size() > topk) order.resize((size_t)topk);
    for (const auto& [sc, ci] : order) {
        const KbChunk& ch = g_kb_chunks[ci];
        KbHit h;
        h.chunk = ci;
        h.score = sc;
        h.path = g_kb_docs[(size_t)ch.doc].path;
        // snippet：以命中词为中心取前后 140 字节
        std::string sn = ch.text;
        const std::string& term = best_term[ci];
        size_t at = term.empty() ? std::string::npos : sn.find(term);
        if (at != std::string::npos && sn.size() > 300) {
            size_t a = (at > 140) ? at - 140 : 0;
            a = utf8_next(sn, a);
            size_t b = std::min(sn.size(), a + 300);
            b = utf8_back(sn, b);
            sn = (a > 0 ? "…" : "") + sn.substr(a, b - a) + (b < sn.size() ? "…" : "");
        } else if (sn.size() > 300) {
            sn = sn.substr(0, utf8_back(sn, 300)) + "…";
        }
        h.snippet = sn;
        hits.push_back(std::move(h));
    }
    return hits;
}

// 懒加载：第一次用到时才读清单 + 重建。调用方持 g_kb_mu。
void kb_ensure_loaded() {
    if (g_kb_loaded) return;
    kb_load_manifest();
    kb_rebuild();
    g_kb_loaded = true;
}

// 给对话用的上下文块。检索不到就返回空串，调用方据此不加东西。
//
// 门槛 5.0：BM25 分数随语料规模变，但"明显相关"的命中在几百个块上
// 通常 10 以上，而闲聊撞到的词只有 1~3。不设门槛的话每句"你好"都会
// 拖进三段文档 —— 1.7B 这种小模型会当场被带跑，去复述上下文而不是回答。
std::string kb_context_for(const std::string& q, int topk) {
    const auto hits = kb_search(q, topk);
    if (hits.empty() || hits[0].score < 5.0) return "";
    // 措辞很重要。原来写的是"回答时优先依据它们"，结果 1.7B 会把资料当圣旨：
    // 问"今天天气怎么样"，它去复述文档里那句"今天天气很好，我们一起"的
    // 测试样例 —— 检索没错（那句话确实在库里），错在没给它"这跟我无关"
    // 这个选项。现在明确说可能不相关、不相关就当没看见。
    std::string s = "\n\n以下是从本地知识库检索到的片段，可能相关，也可能不相关。\n"
                    "只有确实能回答用户问题时才引用；不相关就忽略，不要复述，"
                    "也不要说「资料里提到」。\n";
    for (size_t i = 0; i < hits.size(); i++) {
        s += "\n[" + std::to_string(i + 1) + "] " + path_base(hits[i].path) + "\n";
        s += hits[i].snippet + "\n";
    }
    return s;
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
    if (ends(".jpg")) return "image/jpeg";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".webmanifest")) return "application/manifest+json; charset=utf-8";
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
        const std::string content = read_file(p);
        if (content.empty()) { r.output = "文件为空或无法读取: " + p; return r; }

        // 分页读。一次把整个文件塞进 prompt 是工具轮里最贵的一步：读一个
        // 10 KB 的 README 就是 3158 个 token，prefill 要十几秒 —— 而用户
        // 问的可能只是"第一行写了什么"。
        //
        // 给 offset/limit 之后，模型可以"先读开头、需要再往下翻"，
        // 结果大小由它自己控制，而不是由文件大小决定。
        // 默认只给前 120 行：这个数不是拍的，是按"工具结果不该吃掉半个
        // 上下文"倒推的 —— README 这种密排文档 120 行大约 1700 token，
        // 在 4096 的上下文里还留得下对话和回答。要更多就用 offset 往下翻。
        const int off = std::max(1, (int)args.n("offset", 1));
        const int lim = std::max(1, (int)args.n("limit", 120));

        std::string out;
        int line_no = 0, taken = 0, total = 0;
        size_t i = 0;
        const size_t n = content.size();
        // 逐行扫。i < n 而不是 i <= n：文件以换行结尾时，最后那个换行后面
        // 不是"多出来的一行"（否则 240 行的 README 会报成 241 行）。
        while (i < n) {
            const size_t e = content.find('\n', i);
            const size_t end = (e == std::string::npos) ? n : e;
            line_no++;
            if (line_no >= off && taken < lim) {
                out.append(content, i, end - i);
                out.push_back('\n');
                taken++;
            }
            if (e == std::string::npos) break;
            i = e + 1;
        }
        total = line_no;
        r.ok = true;
        r.output = out;
        if (taken == 0) {
            r.output = "文件只有 " + std::to_string(total) + " 行，offset=" +
                       std::to_string(off) + " 已经超出范围。";
        } else if (off > 1 || off + taken - 1 < total) {
            // 说清楚"给了哪一段、总共多少"，模型才知道要不要接着翻
            r.output += "\n…（这是第 " + std::to_string(off) + "–" +
                        std::to_string(off + taken - 1) + " 行，全文共 " +
                        std::to_string(total) + " 行）";
            if (off + taken - 1 < total) {
                r.output += "。要看后面的内容，用 offset=" + std::to_string(off + taken) +
                            " 再读一次。";
            }
        }
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

// 工具说明。
//
// 写法上有一条踩过的教训：**不要把工具结果的格式教给模型**。
// 原来这里有一句"工具结果会以 <tool_result> 的形式返回给你"，结果 2B 级的
// 模型（LFM3-VL-2B）学会了这个标签就直接用它编答案 —— 实测 5 个问题里
// 有 2 个输出 <tool_result>{"name":"read_file",...}</tool_result>，把"调用"
// 写进了"结果"标签，服务端一次工具都没执行，用户却看到一段像是真读到了
// 文件的内容（问 README.md 首行，它编出"# 我的家常菜谱"）。
//
// 所以现在只说"怎么调用"，并明确禁止它自己输出 <tool_result>。
// 工具返回块用的标记，跟着模型模板走。Qwen2.5 和 LFM3 都是 <tool_response>，
// 早期通用写法是 <tool_result>；用错了模型不认这是结果，会把同一个工具再要一遍。
static std::string g_tool_open = "<tool_result>";
static std::string g_tool_close = "</tool_result>";

// 工具调用格式有两种，各家模板不一样：
//   0 = JSON      <tool_call>{"name":…,"arguments":{…}}</tool_call>   （Qwen 系）
//   1 = LFM3      <tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n
//                 </function>\n</tool_call>
//
// 这个必须跟模型自己的模板一致。教错了不是"效果差一点"——LFM3 被教成 JSON
// 之后输出的是 "read_file>\npath>\n.gitignore" 这种尖括号全丢的残片，
// 一次都调不到工具，而且看起来像解码器坏了。
static int g_tool_fmt = 0;

std::string tool_prompt() {
    std::string s =
        "\n\n需要读文件、列目录、写文件或执行命令时，调用工具。格式严格如下：\n";
    if (g_tool_fmt == 1) {
        s +=
            "<tool_call>\n<function=工具名>\n<parameter=参数名>\n参数值\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "例如读取 a.txt，你这一轮就只输出：\n"
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n"
            "</function>\n</tool_call>\n";
    } else {
        s +=
            "<tool_call>{\"name\":\"工具名\",\"arguments\":{...}}</tool_call>\n"
            "例如读取 a.txt，你这一轮就只输出：\n"
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}</tool_call>\n";
    }
    s +=
        "一次只调用一个工具。可用工具：\n"
        "- read_file(path, offset, limit): 读取文本文件，默认给前 120 行。\n"
        "  文件很长时先读开头；需要后面的内容再用 offset=121 往下翻。\n"
        "- list_dir(path): 列出目录内容\n"
        "- write_file(path, content): 写入文件\n"
        "- shell(command): 执行 shell 命令（可能被禁用）\n"
        "不需要工具时直接回答。不要描述你打算怎么做，直接输出工具调用。"
        "不要自己写工具返回块（" + g_tool_open + "），那是系统给你的，不是你输出的。";
    return s;
}

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

// LFM3 系的工具调用格式（见模板里的 render_tool_calls）：
//
//   <tool_call>
//   <function=read_file>
//   <parameter=path>
//   .gitignore
//   </parameter>
//   </function>
//   </tool_call>
//
// 不是 JSON。少了这一条分支，LFM3 的调用永远解析不出来 —— 原文被抑制逻辑
// 藏起来，用户看到的就是"已完成 · 0 tokens"加一片空白。
bool extract_lfm_tool_call(const std::string& text, std::string* name, json::Value* args) {
    const size_t a = text.find("<tool_call>");
    if (a == std::string::npos) return false;
    size_t b = text.find("</tool_call>", a);
    if (b == std::string::npos) b = text.size();   // 截断在 max_tokens 上也认
    const std::string body = text.substr(a, b - a);

    const size_t f = body.find("<function=");
    if (f == std::string::npos) return false;
    const size_t fe = body.find('>', f);
    if (fe == std::string::npos) return false;
    *name = body.substr(f + 10, fe - f - 10);
    while (!name->empty() && (name->back() == ' ' || name->back() == '\n' || name->back() == '\r'))
        name->pop_back();
    if (name->empty()) return false;

    *args = json::Value::make_obj();
    size_t p = fe;
    while (true) {
        const size_t k = body.find("<parameter=", p);
        if (k == std::string::npos) break;
        const size_t ke = body.find('>', k);
        if (ke == std::string::npos) break;
        const std::string key = body.substr(k + 11, ke - k - 11);
        size_t vs = ke + 1;
        while (vs < body.size() && (body[vs] == '\n' || body[vs] == '\r')) vs++;
        const size_t ve = body.find("</parameter>", vs);
        if (ve == std::string::npos) break;
        std::string val = body.substr(vs, ve - vs);
        while (!val.empty() &&
               (val.back() == '\n' || val.back() == '\r' || val.back() == ' ')) val.pop_back();
        // 参数值可能是数字/布尔/对象，先按 JSON 解；解不了就当字符串
        json::Value pv;
        if (!val.empty() && json::parse(val, &pv)) args->set(key, pv);
        else args->set(key, json::Value::make_str(val));
        p = ve + 12;
    }
    return true;
}

// 从文本中提取第一个工具调用。
// 两种形态都接受：<tool_call>{...}</tool_call> 和裸的 {"name":...}
bool extract_tool_call(const std::string& text, std::string* name, json::Value* args,
                       size_t* end_pos) {
    // 模型模板是 <function=> 风格时优先按它解析 —— 同一个模型不会两种混用，
    // 但两种都试一遍没有代价，还能兜住模板判断失误的情况。
    if (extract_lfm_tool_call(text, name, args)) {
        const size_t c = text.find("</tool_call>");
        if (end_pos) *end_pos = (c == std::string::npos) ? text.size() : c + 12;
        return true;
    }
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

// 请求里已经带了工具结果（role:"tool"）→ 说明调用方在按 OpenAI 协议自己
// 跑循环：模型出 tool_calls，它执行，再把结果发回来。
//
// 这种情况下服务端**不能再执行一遍**。读文件重跑一次只是浪费，但
// write_file / shell 是有副作用的 —— 客户端写一次、服务端再写一次，
// 表现就是"文件里内容重复了两遍"，而且看日志两边都很正常。
bool has_tool_result(const json::Value& messages) {
    if (messages.type != json::Value::ARR) return false;
    for (const auto& m : messages.arr) {
        if (m.type != json::Value::OBJ) continue;
        if (m.s("role") == "tool") return true;
    }
    return false;
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

// ---------------------------------------------------------------- GGUF 元数据
//
// 只读 KV 段，遇到张量信息之前就停 —— 够拿到 tokenizer.chat_template。
//
// 为什么要读它：各家模板给"工具返回"用的标记不一样，用错了模型就不认这是
// 结果，于是把同一个工具再要一遍。
//   Qwen2.5    <|im_start|>user\n<tool_response>…</tool_response><|im_end|>
//   通用写法    <|im_start|>user\n<tool_result>…</tool_result><|im_end|>
// 实测 Qwen2.5-0.5B：发 <tool_result> 时它第二轮改口要 "/.gitignore"（绝对
// 路径），第三轮再要一次，永远收敛不了；换成它模板里的 <tool_response> 之后
// 一次就答上了。
struct GgufReader {
    FILE* f = nullptr;
    bool v3 = true;   // GGUF v3 起字符串长度是 u64，v2 是 u32

    bool rd(void* p, size_t n) { return std::fread(p, 1, n, f) == n; }
    bool str(std::string* out) {
        uint64_t n = 0;
        if (v3) {
            if (!rd(&n, 8)) return false;
        } else {
            uint32_t n32 = 0;
            if (!rd(&n32, 4)) return false;
            n = n32;
        }
        if (n > (1u << 22)) return false;   // 元数据字符串不该这么大
        out->resize((size_t)n);
        return n == 0 || rd(&(*out)[0], (size_t)n);
    }
    bool skip(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: return fseek(f, 1, SEEK_CUR) == 0;
            case 2: case 3: return fseek(f, 2, SEEK_CUR) == 0;
            case 4: case 5: case 6: return fseek(f, 4, SEEK_CUR) == 0;
            case 10: case 11: case 12: return fseek(f, 8, SEEK_CUR) == 0;
            case 8: { std::string s; return str(&s); }
            case 9: {   // 数组：元素类型 + 个数 + 元素
                uint32_t et = 0;
                uint64_t n = 0;
                if (!rd(&et, 4)) return false;
                if (v3) { if (!rd(&n, 8)) return false; }
                else { uint32_t n32 = 0; if (!rd(&n32, 4)) return false; n = n32; }
                size_t sz = 0;
                switch (et) {
                    case 0: case 1: case 7: sz = 1; break;
                    case 2: case 3: sz = 2; break;
                    case 4: case 5: case 6: sz = 4; break;
                    case 10: case 11: case 12: sz = 8; break;
                    default: break;
                }
                if (sz) return fseek(f, (long)(sz * n), SEEK_CUR) == 0;
                for (uint64_t i = 0; i < n; i++) if (!skip(et)) return false;
                return true;
            }
            default: return false;
        }
    }
};

std::string gguf_meta_string(const std::string& path, const std::string& key) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char magic[4] = {0};
    uint32_t ver = 0;
    uint64_t nt = 0, nkv = 0;
    GgufReader r;
    r.f = f;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
        std::fclose(f);
        return {};
    }
    if (!r.rd(&ver, 4) || !r.rd(&nt, 8) || !r.rd(&nkv, 8)) {
        std::fclose(f);
        return {};
    }
    if (ver < 2 || ver > 3) {
        std::fclose(f);
        return {};
    }
    r.v3 = (ver >= 3);
    for (uint64_t i = 0; i < nkv; i++) {
        std::string k;
        uint32_t t = 0;
        if (!r.str(&k)) break;
        if (!r.rd(&t, 4)) break;
        if (k == key && t == 8) {
            r.str(&out);
            break;
        }
        if (!r.skip(t)) break;
    }
    std::fclose(f);
    return out;
}

// 当前模型的"工具返回"标记。换模型时跟着换 —— 模板变了标记就得变。
// （g_tool_open / g_tool_close 在 tool_prompt 之前就声明了。）

// 思考块。LFM3 的模板里有 enable_thinking，默认 true：
//   <|im_start|>assistant\n<think>      ← 想了再答
//   <|im_start|>assistant\n<think></think>  ← 直接答
// 开着的时候模型会把整段推理写进正文，界面上看到的就是
// "用户要求我使用 read_file 工具……我需要先调用……"，答案被埋在最后。
// 对话产品要的是答案，所以统一走"不想"那条路。
static int g_think_style = 0;   // 0 = 模板没有思考标记；1 = LFM3 系

void refresh_model_markers(const std::string& model_path) {
    g_tool_open = "<tool_result>";
    g_tool_close = "</tool_result>";
    g_think_style = 0;
    g_tool_fmt = 0;
    const std::string tpl = gguf_meta_string(model_path, "tokenizer.chat_template");
    if (tpl.empty()) {
        std::fprintf(stderr, "  模板: 模型没带 chat_template，用通用写法（%s）\n", g_tool_open.c_str());
        return;
    }
    if (tpl.find("<tool_response>") != std::string::npos) {
        g_tool_open = "<tool_response>";
        g_tool_close = "</tool_response>";
    } else if (tpl.find("<tool_result>") != std::string::npos) {
        g_tool_open = "<tool_result>";
        g_tool_close = "</tool_result>";
    }
    if (tpl.find("enable_thinking") != std::string::npos &&
        tpl.find("<think>") != std::string::npos) {
        g_think_style = 1;
    }
    // <function=name> 是 LFM3 系 render_tool_calls 的写法，和 JSON 完全不同
    if (tpl.find("<function=") != std::string::npos) g_tool_fmt = 1;
    std::fprintf(stderr, "  模板: 返回 %s · 调用 %s · 思考块 %s\n", g_tool_open.c_str(),
                 g_tool_fmt ? "<function=> 风格" : "JSON 风格",
                 g_think_style ? "有（已关闭）" : "无");
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
            if (with_tools) content += tool_prompt();
            p += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == "user") {
            p += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            // 历史里的助手轮也要带上"空思考块"。模板在 keep_thinking 为假时
            // 会补一个，我们自己拼 prompt 就得手动补 —— 少了它，模型看到
            // 自己上一轮"没有思考块"，会以为格式错了，接着把推理写出来。
            if (g_think_style && content.rfind("<think>", 0) != 0) content = "<think></think>" + content;
            p += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        } else if (role == "tool") {
            // 光把结果摆上去，小模型会以为"我还没调用"，于是再要一次同样的工具，
            // 循环永远不收敛。明确说一句"已经执行完了，直接用结果回答"，
            // 这一句能把大部分原地打转掐掉。
            p += "<|im_start|>user\n" + g_tool_open + content + g_tool_close +
                 "\n工具已经执行完毕，上面就是返回结果。"
                 "请直接基于这个结果回答，不要再调用同一个工具。<|im_end|>\n";
        }
    }
    if (!has_system && with_tools) {
        p = std::string("<|im_start|>system\n") + tool_prompt() + "<|im_end|>\n" + p;
    }
    p += "<|im_start|>assistant\n";
    if (g_think_style) p += "<think></think>";   // 空块 = 这轮不要思考，直接答
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

// 上一次**真正喂给引擎**的那段 prompt：文本 ↔ token 序列。
//
// 为什么要存 token 而不是只存文本：模型生成的 token 不一定等于"贪婪分词"
// 的结果。实测 LFM3 会把 "：\n\n" 当成一个 token 吐出来，而把同样的文字
// 重新分词得到的是 "：" + "\n\n"。字一模一样，token 边界不同，KV 前缀就
// 对不上 —— 引擎那边的复用逻辑只能整体放弃，长上下文多轮等于没优化。
//
// 所以这里存"当时真的喂进去的那串 token"。下一轮如果新 prompt 的文本以
// 它开头，就直接沿用这串 token，只对剩下的部分分词。这样前缀逐 token
// 完全一致，KV 能整段复用。
//
// 只对纯文本生效：带图/带音频的请求 token 一样但特征不同，KV 不能复用。
struct PromptCache {
    std::mutex mu;
    uint64_t id = 0;          // 每次 set 自增，用来判断"这一轮还算不算同一段"
    std::string text;
    std::vector<int32_t> toks;

    uint64_t set(const std::string& t, const std::vector<int32_t>& k) {
        std::lock_guard<std::mutex> lk(mu);
        text = t;
        toks = k;
        return ++id;
    }
    // 命中就把缓存的 token 拷出来、并告诉调用方吃掉了多长的前缀
    bool splice(const std::string& p, std::vector<int32_t>* out, size_t* used) {
        std::lock_guard<std::mutex> lk(mu);
        if (toks.empty() || text.size() < 8 || p.size() <= text.size()) return false;
        if (p.compare(0, text.size(), text) != 0) return false;
        *out = toks;
        *used = text.size();
        return true;
    }
    uint64_t current_id() {
        std::lock_guard<std::mutex> lk(mu);
        return id;
    }
    // 生成完之后把"这一轮新吐出来的字和 token"接到缓存尾巴上
    void extend(uint64_t want, const std::string& t, const std::vector<int32_t>& k) {
        std::lock_guard<std::mutex> lk(mu);
        if (want != id || t.empty()) return;   // 期间被别的请求改过，别乱接
        text += t;
        toks.insert(toks.end(), k.begin(), k.end());
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        text.clear();
        toks.clear();
        id++;
    }
};
static PromptCache g_pcache;

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

    // 附件必须和已加载的塔配套。
    //
    // 这一条不能靠"注入时维度对不对"来兜底：LFM3 全家的 hidden 都是 2048，
    // 视觉塔的输出能顺利塞进任何一个 LFM3 文本模型，然后模型回一段和图片
    // 毫无关系的话。实测把 LFM3-1.7B-F16 配上 VL 的塔，问"这张图是什么
    // 颜色"，回的是 "They provided a block of text that appears to be
    // some garbled mix…"。HTTP 200、无异常、无日志 —— 比报错难查得多。
    // 所以在这里直接拒掉，把原因说清楚。
    if (!media.empty()) {
        const int caps = dubhe_mm_caps(e);
        for (const auto& m : media) {
            const bool is_audio = (m.kind == "audio");
            if ((caps & (is_audio ? 2 : 1)) == 0) {
                if (err) {
                    *err = is_audio
                        ? "当前模型没有加载语音塔，无法处理音频。请到「模型」视图换一个带语音能力的模型。"
                        : "当前模型没有加载视觉塔，无法处理图片/视频。请到「模型」视图换一个多模态模型。";
                }
                return false;
            }
        }
    }

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
        // ---- 知识库：把检索到的片段拼在**最后一条用户消息前面**。
        // 只在纯文本 + 非 raw 时做 —— 多模态的 prompt 里带 <|image|> 占位符，
        // 往前面塞文本会让占位符和特征对不上位置。
        //
        // 为什么不插成最前面的 system：那样一来每条新问题的检索结果一变，
        // 整个 prompt 的第一个字节就变了，前缀缓存全部作废 —— 多轮对话
        // 里每一轮都要重新 prefill 整段历史（实测 7 秒）。拼在末尾之后，
        // 前面的历史一个字节都不动，缓存照常命中。
        // 顺带这也更符合 RAG 的常规做法：资料紧贴着问题。
        //
        // 带工具时不注入。实测（LFM3-1.7B）两者会打架：知识库里本来就有
        // README，把它的片段塞进 prompt 之后，模型把"资料已经在这儿了"
        // 当成任务，回的是"我们需要执行 read_file……"然后一个工具都不调；
        // 关掉 RAG 同一句话 1.45 秒就正确调用。1.7B 这种规模一次只能抓住
        // 一个意图，而工具模式下"去读真实内容"本来就比"复述旧片段"更对。
        if (g_kb_rag && media.empty() && !with_tools) {
            std::string last_user;
            size_t last_idx = 0;
            bool has_last = false;
            for (size_t i = 0; i < flat.arr.size(); i++) {
                if (flat.arr[i].s("role") == "user") {
                    last_user = flat.arr[i].s("content");
                    last_idx = i;
                    has_last = true;
                }
            }
            if (has_last && !last_user.empty()) {
                std::string ctx;
                {
                    std::lock_guard<std::mutex> lk(g_kb_mu);
                    if (!g_kb_docs.empty()) {
                        kb_ensure_loaded();
                        ctx = kb_context_for(last_user, 3);
                    }
                }
                if (!ctx.empty()) {
                    // kb_context_for 以两个换行开头，去掉后拼在问题前面
                    const std::string body = ctx.substr(ctx.find_first_not_of('\n'));
                    flat.arr[last_idx].set(
                        "content",
                        json::Value::make_str(body + "\n\n" + last_user));
                }
            }
        }
        prompt = build_prompt(flat, chatml, with_tools);
    }

    // 拼出来的 prompt 长什么样，是这类"模型不听话"问题里唯一可靠的证据。
    // 靠猜模板、猜标记会来回改很多轮；打出来看一眼往往一次就定位。
    if (const char* dump = std::getenv("DUBHE_DUMP_PROMPT")) {
        if (dump[0] != '0') {
            std::fprintf(stderr, "\n===== prompt (%zu 字节, chatml=%d tools=%d) =====\n%s\n=====\n",
                         prompt.size(), (int)chatml, (int)with_tools, prompt.c_str());
        }
    }

    const int32_t n = dubhe_tokenize(e, prompt.c_str(), 1, nullptr, 0);
    if (n <= 0) { if (err) *err = "分词失败"; return false; }

    // 前缀复用：新 prompt 的文本以上次喂进去的那段开头时，沿用它的 token，
    // 只给剩下的部分分词。这样前缀逐 token 一致，引擎能整段跳过 prefill。
    bool spliced = false;
    size_t used = 0;
    if (!raw && media.empty() && g_pcache.splice(prompt, toks, &used)) {
        const size_t at = toks->size();
        const std::string rest = prompt.substr(used);
        if (!rest.empty()) {
            const int32_t m = dubhe_tokenize(e, rest.c_str(), 0, nullptr, 0);
            if (m > 0) {
                toks->resize(at + (size_t)m);
                if (dubhe_tokenize(e, rest.c_str(), 0, toks->data() + at, m) != m) {
                    toks->resize(at);
                } else {
                    spliced = true;
                }
            }
        }
        if (!spliced) toks->clear();   // 剩下的部分分不出来，退回整段重分
    }
    if (!spliced) {
        toks->assign((size_t)n, 0);
        if (dubhe_tokenize(e, prompt.c_str(), 1, toks->data(), n) != n) {
            if (err) *err = "分词失败";
            return false;
        }
        // raw 也必须补 BOS。不补的话模型不知道自己站在序列开头，转写内容是对的，
        // 但一句接一句复读到 max_tokens 也不吐 EOS；补上之后它写完就停。
        prepend_bos(e, toks);
    }
    // 记下"这一轮喂进去的到底是什么"，生成完之后再把新吐的字接上来。
    if (!raw && media.empty()) g_pcache.set(prompt, *toks);

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
    // 第二轮（带着工具结果的那一轮）是最终回答，不该再被工具判定拦下来。
    // 实测 2B 模型看到工具结果后会再吐一次 tool_call，抑制逻辑把它整段
    // 吞掉，用户看到的就是"生成中"转很久然后一片空白 —— 比直接显示
    // 原始内容难查得多。
    bool no_suppress = false;
    std::string pending;
    // 思考块。我们已经在 prompt 里把 enable_thinking 关掉了，这里是兜底：
    // 万一模型还是写了 <think>…</think>，整段丢掉，别让推理过程占满气泡。
    bool in_think = false;
    std::string think_buf;
    // 上一片末尾没凑齐的多字节字符。引擎是按 token 吐字节的，一个汉字
    // （3 字节）经常被切在两个 token 之间；半截字节进了 JSON，浏览器那边
    // 就只能解成 U+FFFD —— 中文回复里随机冒出"均���的"这种方块。
    std::string tail;
    // 这一轮真正生成的 token 序列，用来给前缀缓存接尾巴
    std::vector<int32_t> gtoks;
};

// 返回 s 末尾"不完整的多字节序列"占几个字节；0 表示末尾是干净的。
//
// UTF-8 的续字节最高两位是 10，首字节决定了整个字符要几个字节。
// 从后往前找第一个非续字节，看它声明的长度和实际剩下的字节数对不对得上。
static size_t utf8_incomplete_tail(const std::string& s) {
    const size_t n = s.size();
    if (n == 0) return 0;
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        const size_t i = n - back;
        const unsigned char c = (unsigned char)s[i];
        if ((c & 0xC0) == 0x80) continue;          // 续字节，继续往前找首字节
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else return 0;                              // 非法首字节，不拦，交给上层
        return back < need ? back : 0;               // 够长就完整，不够就把这几字节留住
    }
    return 0;
}

void emit_delta(StreamCtx* x, const std::string& delta) {
    std::string s = "data: {\"id\":\"" + *x->id +
                    "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                    std::to_string(x->created) + ",\"choices\":[{\"index\":0,\"delta\":{\"content\":" +
                    json_escape(delta) + "},\"finish_reason\":null}]}\n\n";
    x->c->write_chunk(s);
    x->first = false;
}

// 按 OpenAI 的流式协议把工具调用发出去。
//
// 外部 SDK（openai-python 之类）只认 delta.tool_calls，不认我们的 dubhe 扩展
// 事件。少了这一段，客户端会收到 finish_reason=tool_calls，却一条调用都拿不到，
// 只能干看着 —— 比报错更难查。
void emit_tool_calls_delta(StreamCtx* x, const std::string& name, const json::Value& args) {
    std::string s = "data: {\"id\":\"" + *x->id +
                    "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                    std::to_string(x->created) +
                    ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_" +
                    std::to_string(x->created) + "\",\"type\":\"function\",\"function\":{\"name\":" +
                    json_escape(name) + ",\"arguments\":" + json_escape(json::dump(args)) +
                    "}}]},\"finish_reason\":null}]}\n\n";
    x->c->write_chunk(s);
}

// 前 16 个字符足以判断"这是不是一段工具调用"
void on_piece(void* ud, int32_t tok, const char* piece) {
    auto* x = (StreamCtx*)ud;
    if (!piece) return;
    x->acc->append(piece);
    x->gtoks.push_back(tok);

    // 思考块：整段吞掉，不进正文
    {
        const std::string pc(piece);
        if (pc == "<think>") {
            x->in_think = true;
            x->think_buf.clear();
            return;
        }
        if (pc == "</think>") {
            x->in_think = false;
            x->think_buf.clear();
            return;
        }
        if (x->in_think) {
            x->think_buf += pc;
            return;
        }
    }

    if (x->suppress) return;

    // 先和上一片留下的半截字符拼起来，再把末尾凑不齐的部分留住，
    // 剩下的才是能安全写进 JSON 的。
    std::string s = x->tail + std::string(piece);
    const size_t hold = utf8_incomplete_tail(s);
    if (hold) {
        x->tail.assign(s, s.size() - hold, hold);
        s.resize(s.size() - hold);
    } else {
        x->tail.clear();
    }
    if (s.empty()) return;   // 这一片整个是半截字符，等下一片

    // 这一轮不该抑制（最后一段回答，或者调用方自己管循环）→ 直接吐。
    //
    // 抑制是为了藏住 tool_call 的原始 JSON，前提是"后面还有一轮会说人话"。
    // 一旦没有下一轮，抑制就等于把整条回复吞掉 —— 界面上只剩
    // "已完成 · 0 tokens" 和一片空白，比露出 JSON 难查得多。
    if (x->no_suppress) {
        x->decided = true;
        emit_delta(x, s);
        return;
    }

    if (x->decided) {
        emit_delta(x, s);
        return;
    }
    x->pending += s;
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
        x->tail.clear();
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
    // 思考块没闭合（多半是被 max_tokens 截断在中间）。这时候丢弃等于一个字
    // 都不给，用户看到的是空白气泡 —— 与其这样，不如把它原样吐出来。
    if (x->in_think) {
        x->in_think = false;
        if (!x->think_buf.empty()) {
            const std::string t = x->think_buf;
            x->think_buf.clear();
            x->decided = true;
            emit_delta(x, t);
        }
    }
    if (!x->decided && !x->pending.empty()) {
        const std::string all = x->pending;
        x->pending.clear();
        x->decided = true;
        emit_delta(x, all);
    }
    // 正常回复的最后一片是完整的字符，不会留尾巴。真留了（比如被 max_tokens
    // 截断在半个汉字上）也一起吐出去 —— 让前端自己决定怎么显示，总比丢字好。
    if (!x->tail.empty() && !x->suppress) {
        const std::string t = x->tail;
        x->tail.clear();
        emit_delta(x, t);
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

// ---------------------------------------------------------------- 模型清单
struct ModelEntry {
    std::string path, name;
    long long bytes = 0;
    long long mtime = 0;
    std::string mmproj;   // 同目录配套的塔；空表示纯文本
};

void scan_models(const std::string& dir, std::vector<ModelEntry>* out, int depth) {
    if (depth > 3 || out->size() >= 400) return;
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        const std::string full = path_join(dir, n);
        struct stat st {};
        if (stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            scan_models(full, out, depth + 1);
        } else if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
            // mmproj 是塔，不是模型。以前它会被当成可选模型列出来，点一下就
            // 拿视觉塔去当语言模型加载 —— 必然失败，但用户不知道为什么会失败。
            // 这里直接排除，改成挂到同目录的模型上（见 mmproj 字段）。
            if (n.compare(0, 6, "mmproj") == 0) continue;
            ModelEntry m;
            m.path = full;
            m.name = n.substr(0, n.size() - 5);
            m.bytes = (long long)st.st_size;
            m.mtime = (long long)st.st_mtime;
            out->push_back(std::move(m));
        }
    }
    closedir(d);
}

// 从文件名猜量化类型。顺序要紧：长的先匹配，"Q4_K_M" 必须排在 "Q4_0" 前面，
// 否则 Q4_K_M 会被识别成 Q4_0。
std::string guess_quant(const std::string& n) {
    std::string u = n;
    for (auto& c : u) c = (char)std::toupper((unsigned char)c);
    static const char* kQ[] = {"IQ4_XS", "IQ4_NL", "IQ3_M",  "IQ2_XS", "IQ2_M", "Q8_K",
                               "Q8_0",   "Q6_K",   "Q5_K_M", "Q5_K_S", "Q5_1",  "Q5_0",
                               "Q4_K_M", "Q4_K_S", "Q4_1",   "Q4_0",   "Q3_K_L", "Q3_K_M",
                               "Q3_K_S", "Q2_K",   "BF16",   "F16",    "F32"};
    for (const char* q : kQ)
        if (u.find(q) != std::string::npos) return q;
    return "";
}

// 从文件名猜参数量（"LFM3-1.7B" → 1.7e9）。猜不出来就返回 0，
// 前端会显示"—"而不是编一个数字。
double guess_params(const std::string& n) {
    for (size_t i = 0; i < n.size(); i++) {
        if (n[i] != 'B' && n[i] != 'b') continue;
        size_t j = i;
        while (j > 0 && (std::isdigit((unsigned char)n[j - 1]) || n[j - 1] == '.')) j--;
        if (j == i) continue;
        const std::string num = n.substr(j, i - j);
        const double v = atof(num.c_str());
        if (v > 0.05 && v < 300) return v * 1e9;
    }
    return 0;
}

// 找一个模型"配套"的多模态塔。
//
// 约定：塔的 GGUF 和模型放在同一个目录，文件名以 mmproj 开头。这是
// llama.cpp 和 LM Studio 下载下来的默认布局，本机 ~/.dubhe/models 也是这个形状
// （lfm3-vl-2b/ 下是模型 + mmproj-*.gguf，lfm3-audio-2b/ 同理）。
//
// 只认同目录、不往上层找：上层是 models/ 根，那里放的是别的模型的塔，
// 拿错了就是"维度对得上、语义对不上"的静默垃圾。
std::string find_sibling_mmproj(const std::string& model_path) {
    const std::string dir = path_dir(model_path);
    DIR* d = opendir(dir.c_str());
    if (!d) return "";
    std::string best;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() < 11) continue;
        if (n.compare(0, 6, "mmproj") != 0) continue;
        if (n.compare(n.size() - 5, 5, ".gguf") != 0) continue;
        const std::string full = path_join(dir, n);
        if (!is_file_p(full)) continue;
        // 一个目录里同时有视觉塔和语音塔时，先拿到哪个算哪个：
        // 调用方会按 dim 和 caps 再校验一遍，拿错的那次加载会被回退。
        if (best.empty() || full < best) best = full;
    }
    closedir(d);
    return best;
}

json::Value list_dir_json(const std::string& dir) {
    json::Value root = json::Value::make_obj();
    root.set("path", json::Value::make_str(dir));
    root.set("parent", json::Value::make_str(path_dir(dir)));
    json::Value dirs = json::Value::make_arr();
    json::Value files = json::Value::make_arr();
    DIR* d = opendir(dir.c_str());
    if (!d) {
        root.set("error", json::Value::make_str("无法打开目录"));
        root.set("items", dirs);
        return root;
    }
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.empty() || n[0] == '.') continue;   // 隐藏文件先不列，减少噪声
        const std::string full = path_join(dir, n);
        struct stat st {};
        if (stat(full.c_str(), &st) != 0) continue;
        json::Value o = json::Value::make_obj();
        o.set("name", json::Value::make_str(n));
        o.set("path", json::Value::make_str(full));
        o.set("dir", json::Value::make_bool(S_ISDIR(st.st_mode)));
        o.set("bytes", json::Value::make_num((double)st.st_size));
        o.set("mtime", json::Value::make_num((double)st.st_mtime));
        o.set("text", json::Value::make_bool(S_ISREG(st.st_mode) && has_text_ext(n)));
        if (S_ISDIR(st.st_mode)) dirs.arr.push_back(o);
        else files.arr.push_back(o);
    }
    closedir(d);
    auto by_name = [](const json::Value& a, const json::Value& b) {
        return a.s("name") < b.s("name");
    };
    std::sort(dirs.arr.begin(), dirs.arr.end(), by_name);
    std::sort(files.arr.begin(), files.arr.end(), by_name);
    for (auto& v : files.arr) dirs.arr.push_back(v);
    root.set("items", dirs);
    return root;
}

// ---------------------------------------------------------------- 工具清单
json::Value tools_json() {
    json::Value arr = json::Value::make_arr();
    auto add = [&](const char* name, const char* desc, const char* args, bool on) {
        json::Value o = json::Value::make_obj();
        o.set("name", json::Value::make_str(name));
        o.set("desc", json::Value::make_str(desc));
        o.set("args", json::Value::make_str(args));
        o.set("enabled", json::Value::make_bool(on));
        arr.arr.push_back(o);
    };
    add("read_file", "读取文本文件，默认前 120 行，可用 offset/limit 分页", "path, offset, limit", true);
    add("list_dir", "列出目录内容", "path", true);
    add("write_file", "写入文件（覆盖）", "path, content", true);
    add("shell", "执行 shell 命令", "command", g.allow_shell);
    return arr;
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
    bool no_mmproj = false;
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
        else if (a == "--model-dir") g.model_dir = next();
        else if (a == "--no-gpu") no_gpu = true;
        else if (a == "--api-key") g.api_key = next();
        else if (a == "--mmproj") mmprojs.push_back(next());
        else if (a == "--no-mmproj") no_mmproj = true;
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
                "                 不给的话，模型同目录下的 mmproj*.gguf 会自动挂上\n"
                "  --no-mmproj    不自动挂塔，纯文本启动\n"
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

    // 模型目录默认 ~/.dubhe/models（CLI 下载模型就放这儿）。
    // UI 的「模型」视图靠它扫描，没指定时给个合理默认，否则那个视图永远是空的。
    if (g.model_dir.empty()) {
        const char* home = getenv("HOME");
        if (home) g.model_dir = std::string(home) + "/.dubhe/models";
    }

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
        refresh_model_markers(model);
        dubhe_model_info mi{};
        dubhe_model_info_get(g.engine, &mi);
        std::fprintf(stderr, "就绪 %.0f ms · %s · %s · %.2fB\n", (double)ms, mi.arch, mi.quant_desc,
                     (double)mi.n_params / 1e9);
    } else {
        std::fprintf(stderr, "未指定模型，UI 中可加载。\n");
    }

    if (g.tmp_dir.empty()) g.tmp_dir = "/tmp";

    // 没显式给 --mmproj 时，看模型旁边有没有塔，有就自动挂上。
    // 规则和 /api/load 一致：同一个目录 = 同一套东西。这样
    // `dubhed -m ~/.dubhe/models/lfm3-vl-2b/xxx.gguf` 直接就能看图，
    // 不用再把塔的路径也背一遍。要纯文本启动就加 --no-mmproj。
    if (mmprojs.empty() && !no_mmproj && !model.empty()) {
        const std::string sib = find_sibling_mmproj(model);
        if (!sib.empty()) {
            std::fprintf(stderr, "模型旁边有配套的塔，自动加载 %s（纯文本启动用 --no-mmproj）\n",
                         path_base(sib).c_str());
            mmprojs.push_back(sib);
        }
    }

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
        // 维度对不上时在启动阶段就喊出来。注入阶段也有同样的检查，但那时
        // 用户已经在等回答了；这里至少让他一眼看到"我挂错了塔"。
        {
            dubhe_model_info mi{};
            dubhe_model_info_get(g.engine, &mi);
            const int d = dubhe_mm_dim(g.engine);
            if (d > 0 && mi.n_embd > 0 && d != (int)mi.n_embd) {
                std::fprintf(stderr,
                             "  !! 这个塔输出 %d 维，当前模型要 %d 维 —— 注入会失败，请换配套的塔\n",
                             d, (int)mi.n_embd);
            }
        }
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

    // 其余静态资源走兜底：图标、manifest、字体以后还会加，一条一条注册跟不上。
    // 但只放行白名单后缀 —— 否则 /api/typo 会返回 404 页面而不是 404 JSON，
    // 而且等于把整个 web_root 连同未来可能放进去的源文件都暴露出去。
    srv.set_fallback([&](const http::Request& req, http::Conn& c) {
        if (req.method != "GET") {
            c.send(404, "application/json", "{\"error\":\"not found\"}");
            return;
        }
        static const char* kAllowed[] = {".html", ".css", ".js", ".json", ".svg",
                                         ".png", ".ico", ".webmanifest", ".woff2", ".jpg"};
        std::string p = req.path;
        bool ok = false;
        for (const char* e : kAllowed) {
            size_t n = std::strlen(e);
            if (p.size() > n && p.compare(p.size() - n, n, e) == 0) { ok = true; break; }
        }
        if (!ok) {
            c.send(404, "application/json", "{\"error\":\"not found\"}");
            return;
        }
        serve_file(req, c, g.web_root);
    });

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
        // 真实占用：进程常驻内存 + 模型文件体积。
        // 前端右栏的"内存"条靠这两个值，不做估算 —— 估算出来的数字
        // 一旦和 Activity Monitor 对不上，用户就不信这块面板了。
        {
            struct stat st {};
            double model_bytes = 0;
            if (!g.model_path.empty() && stat(g.model_path.c_str(), &st) == 0) {
                model_bytes += (double)st.st_size;
            }
            if (!g.mmproj_path.empty() && stat(g.mmproj_path.c_str(), &st) == 0) {
                model_bytes += (double)st.st_size;
            }
            o.set("model_bytes", json::Value::make_num(model_bytes));

            struct rusage ru {};
            double rss = 0;
            if (getrusage(RUSAGE_SELF, &ru) == 0) {
#ifdef __APPLE__
                rss = (double)ru.ru_maxrss;            // macOS: 字节
#else
                rss = (double)ru.ru_maxrss * 1024.0;   // Linux: KB
#endif
            }
            o.set("rss_bytes", json::Value::make_num(rss));
        }
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

    // ---- 模型清单：UI 的「模型」视图靠它列出本地所有 gguf。
    // 只 stat 文件、不读 GGUF 头 —— 一个目录几十个模型时读头要好几百毫秒，
    // 而名字里已经带了量化和参数量。真正加载时才读头。
    srv.get("/api/models", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        std::vector<ModelEntry> ms;
        if (!g.model_dir.empty()) scan_models(g.model_dir, &ms, 0);
        std::sort(ms.begin(), ms.end(),
                  [](const ModelEntry& a, const ModelEntry& b) { return a.name < b.name; });
        json::Value root = json::Value::make_obj();
        root.set("dir", json::Value::make_str(g.model_dir));
        root.set("current", json::Value::make_str(g.model_path));
        for (auto& m : ms) m.mmproj = find_sibling_mmproj(m.path);
        json::Value arr = json::Value::make_arr();
        for (const auto& m : ms) {
            json::Value o = json::Value::make_obj();
            o.set("path", json::Value::make_str(m.path));
            o.set("name", json::Value::make_str(m.name));
            o.set("bytes", json::Value::make_num((double)m.bytes));
            o.set("mtime", json::Value::make_num((double)m.mtime));
            o.set("quant", json::Value::make_str(guess_quant(m.name)));
            o.set("params", json::Value::make_num(guess_params(m.name)));
            o.set("loaded", json::Value::make_bool(m.path == g.model_path));
            // 有配套塔就能看图/听音。这里只报"有没有塔"和塔的路径，
            // 不报具体是视觉还是语音 —— 那要读塔的 GGUF 头，而几十个模型
            // 逐个读头是几百毫秒。真正的 caps 在 /api/state 里，加载后自然知道。
            o.set("mmproj", json::Value::make_str(m.mmproj));
            arr.arr.push_back(o);
        }
        root.set("items", arr);
        c.send(200, "application/json", json::dump(root));
    });

    // ---- 切换模型。文档里写了这个端点但一直没实现，UI 上点「切换」
    // 只能弹一句"还在开发中"。补上：卸载 → 加载 → 更新 model_id。
    // 加载失败要把旧模型装回去，否则一次误操作就让服务变成没有模型的状态。
    srv.post("/api/load", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        std::string p = req.param("path");
        if (p.empty() && json::parse(req.body, &body)) p = body.s("path");
        if (p.empty()) {
            c.send(400, "application/json", "{\"error\":\"缺少 path\"}");
            return;
        }
        const std::string full = path_abs(p);
        if (!path_allowed(full) || !is_file_p(full)) {
            c.send(400, "application/json", "{\"error\":\"路径不允许或文件不存在\"}");
            return;
        }
        if (g.busy.load()) {
            c.send(409, "application/json", "{\"error\":\"正在生成，稍后再切换\"}");
            return;
        }
        std::lock_guard<std::mutex> lk(g.mu);
        const std::string old = g.model_path;
        const std::string old_mm = g.mmproj_path;
        const auto t0 = std::chrono::steady_clock::now();

        // 塔必须先卸：塔是跟模型配套的，留着旧的只会让 /api/state 报出
        // 一个"其实对不上"的 mm_vision=true，UI 于是继续露出图片按钮，
        // 用户点下去拿到一段和图片毫无关系的话。
        dubhe_unload_mmproj(g.engine);
        g.mmproj_path.clear();

        dubhe_unload_model(g.engine);
        if (dubhe_load_model(g.engine, full.c_str()) != DUBHE_OK) {
            const std::string err = dubhe_last_error();
            g.model_path.clear();
            g.model_id.clear();
            // 装回旧模型，并把旧模型原来的塔也装回去 —— 否则一次误操作
            // 会让服务从"能看图"退化成"不能看图"，而且没有任何提示。
            if (!old.empty()) {
                if (dubhe_load_model(g.engine, old.c_str()) == DUBHE_OK) {
                    g.model_path = old;
                    g.model_id = model_id_of(old);
                    refresh_model_markers(old);
                    std::string tmp = old_mm;
                    size_t p = 0;
                    while (p <= tmp.size() && !tmp.empty()) {
                        const size_t q = tmp.find(',', p);
                        const std::string one = tmp.substr(p, q == std::string::npos ? std::string::npos : q - p);
                        if (!one.empty() && dubhe_load_mmproj(g.engine, one.c_str()) == DUBHE_OK) {
                            if (!g.mmproj_path.empty()) g.mmproj_path += ",";
                            g.mmproj_path += one;
                        }
                        if (q == std::string::npos) break;
                        p = q + 1;
                    }
                }
            }
            json::Value e = json::Value::make_obj();
            e.set("error", json::Value::make_str("加载失败: " + err));
            e.set("restored", json::Value::make_str(g.model_path));
            e.set("mm", json::Value::make_num(dubhe_mm_caps(g.engine)));
            c.send(500, "application/json", json::dump(e));
            return;
        }

        // ---- 给新模型配塔。
        // 显式传了 mmproj 就用传的；没传就在模型同目录里找 mmproj*.gguf。
        // 找到之后必须过一道维度校验：LFM3 家族的 hidden 都是 2048，
        // 拿错塔不会报错，只会静默输出垃圾，所以这里宁可卸载也不将就。
        std::string want_mm = body.s("mmproj");
        if (want_mm.empty()) want_mm = req.param("mmproj");
        if (want_mm.empty()) want_mm = find_sibling_mmproj(full);
        std::string mm_note;
        if (!want_mm.empty()) {
            if (!is_file_p(want_mm) || !path_allowed(want_mm)) {
                mm_note = "塔文件不存在或路径不允许: " + want_mm;
            } else if (dubhe_load_mmproj(g.engine, want_mm.c_str()) != DUBHE_OK) {
                mm_note = std::string("塔加载失败: ") + dubhe_last_error();
            } else {
                const int need = dubhe_mm_dim(g.engine);
                dubhe_model_info chk{};
                dubhe_model_info_get(g.engine, &chk);
                if (need > 0 && chk.n_embd > 0 && need != (int)chk.n_embd) {
                    dubhe_unload_mmproj(g.engine);
                    mm_note = "塔输出 " + std::to_string(need) + " 维，模型要 " +
                              std::to_string(chk.n_embd) + " 维，已跳过";
                } else {
                    g.mmproj_path = want_mm;
                }
            }
        }

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        g.model_path = full;
        g.model_id = model_id_of(full);
        refresh_model_markers(full);
        // 换了模型，上一段 prompt 的 token 和 KV 全部作废
        g_pcache.clear();
        json::Value o = json::Value::make_obj();
        o.set("ok", json::Value::make_bool(true));
        o.set("model", json::Value::make_str(g.model_path));
        o.set("model_id", json::Value::make_str(g.model_id));
        o.set("ms", json::Value::make_num((double)ms));
        o.set("mm", json::Value::make_num(dubhe_mm_caps(g.engine)));
        o.set("mmproj", json::Value::make_str(g.mmproj_path));
        if (!mm_note.empty()) o.set("mm_note", json::Value::make_str(mm_note));
        dubhe_model_info mi{};
        if (dubhe_model_info_get(g.engine, &mi) == DUBHE_OK) {
            o.set("arch", json::Value::make_str(mi.arch));
            o.set("quant", json::Value::make_str(mi.quant_desc));
            o.set("params", json::Value::make_num((double)mi.n_params));
        }
        c.send(200, "application/json", json::dump(o));
    });

    // ---- 文件浏览
    srv.get("/api/files", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        const std::string full = path_abs(req.param("path"));
        if (!is_dir_p(full)) {
            c.send(404, "application/json", "{\"error\":\"目录不存在\"}");
            return;
        }
        if (!path_allowed(full)) {
            c.send(403, "application/json", "{\"error\":\"不在允许的范围内\"}");
            return;
        }
        c.send(200, "application/json", json::dump(list_dir_json(full)));
    });

    srv.get("/api/files/read", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        const std::string full = path_abs(req.param("path"));
        if (!is_file_p(full)) {
            c.send(404, "application/json", "{\"error\":\"文件不存在\"}");
            return;
        }
        if (!path_allowed(full)) {
            c.send(403, "application/json", "{\"error\":\"不在允许的范围内\"}");
            return;
        }
        bool tr = false;
        std::string txt = read_text_file(full, 512 * 1024, &tr);
        json::Value o = json::Value::make_obj();
        o.set("path", json::Value::make_str(full));
        o.set("name", json::Value::make_str(path_base(full)));
        o.set("bytes", json::Value::make_num((double)txt.size()));
        o.set("truncated", json::Value::make_bool(tr));
        o.set("binary", json::Value::make_bool(txt.find('\0') != std::string::npos));
        o.set("text", json::Value::make_str(txt));
        c.send(200, "application/json", json::dump(o));
    });

    // ---- 工具：清单 + 手动执行（给 UI 的调试面板用，不必绕一圈模型）
    srv.get("/api/tools", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value root = json::Value::make_obj();
        root.set("items", tools_json());
        root.set("workspace", json::Value::make_str(g.workspace));
        c.send(200, "application/json", json::dump(root));
    });

    srv.post("/api/tools/run", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        if (!json::parse(req.body, &body)) {
            c.send(400, "application/json", "{\"error\":\"JSON 解析失败\"}");
            return;
        }
        const std::string name = body.s("name");
        if (name.empty()) {
            c.send(400, "application/json", "{\"error\":\"缺少 name\"}");
            return;
        }
        // arguments 允许是对象，也允许直接平铺在顶层（方便 curl 手测）
        json::Value args = json::Value::make_obj();
        const json::Value* a = body.get("arguments");
        if (a && a->type == json::Value::OBJ) args = *a;
        else args = body;
        // 文件类工具走路径白名单，和 UI 的文件视图同一套边界
        for (const char* k : {"path", "content"}) {
            const std::string v = args.s(k);
            if (!v.empty() && std::strcmp(k, "path") == 0) {
                const std::string full = path_abs(v);
                if (!path_allowed(full)) {
                    c.send(403, "application/json", "{\"error\":\"路径不在允许的范围内\"}");
                    return;
                }
            }
        }
        const ToolResult r = run_tool(name, args);
        json::Value o = json::Value::make_obj();
        o.set("ok", json::Value::make_bool(r.ok));
        o.set("output", json::Value::make_str(r.output));
        c.send(200, "application/json", json::dump(o));
    });

    // ---- 知识库：索引 / 检索 / 开关
    auto kb_snapshot = []() {
        json::Value root = json::Value::make_obj();
        json::Value arr = json::Value::make_arr();
        long long total = 0;
        for (const auto& d : g_kb_docs) {
            json::Value o = json::Value::make_obj();
            o.set("path", json::Value::make_str(d.path));
            o.set("name", json::Value::make_str(path_base(d.path)));
            o.set("mtime", json::Value::make_num((double)d.mtime));
            o.set("bytes", json::Value::make_num((double)d.bytes));
            o.set("chunks", json::Value::make_num((double)d.chunks));
            o.set("missing", json::Value::make_bool(d.missing));
            total += d.bytes;
            arr.arr.push_back(o);
        }
        root.set("docs", arr);
        root.set("n_docs", json::Value::make_num((double)g_kb_docs.size()));
        root.set("n_chunks", json::Value::make_num((double)g_kb_chunks.size()));
        root.set("bytes", json::Value::make_num((double)total));
        root.set("rag", json::Value::make_bool(g_kb_rag));
        root.set("dir", json::Value::make_str(path_dir(kb_manifest_path())));
        return root;
    };

    srv.get("/api/kb", [&](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        std::lock_guard<std::mutex> lk(g_kb_mu);
        if (!g_kb_loaded) {
            kb_load_manifest();
            kb_rebuild();
            g_kb_loaded = true;
        }
        c.send(200, "application/json", json::dump(kb_snapshot()));
    });

    // 递归收集文本文件（索引一个目录时用）
    srv.post("/api/kb/add", [&](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        std::string p = req.param("path");
        if (p.empty() && json::parse(req.body, &body)) p = body.s("path");
        if (p.empty()) {
            c.send(400, "application/json", "{\"error\":\"缺少 path\"}");
            return;
        }
        const std::string full = path_abs(p);
        if (!path_allowed(full)) {
            c.send(403, "application/json", "{\"error\":\"路径不在允许的范围内\"}");
            return;
        }
        std::vector<std::string> files;
        std::function<void(const std::string&, int)> walk = [&](const std::string& d, int dep) {
            if (dep > 4 || files.size() >= 200) return;
            if (is_file_p(d)) {
                if (has_text_ext(d)) files.push_back(d);
                return;
            }
            DIR* dp = opendir(d.c_str());
            if (!dp) return;
            while (dirent* e = readdir(dp)) {
                const std::string n = e->d_name;
                if (n.empty() || n[0] == '.') continue;
                walk(path_join(d, n), dep + 1);
            }
            closedir(dp);
        };
        walk(full, 0);
        if (files.empty()) {
            c.send(400, "application/json", "{\"error\":\"这里没有可索引的文本文件\"}");
            return;
        }
        std::lock_guard<std::mutex> lk(g_kb_mu);
        if (!g_kb_loaded) {
            kb_load_manifest();
            g_kb_loaded = true;
        }
        int added = 0;
        for (const auto& f : files) {
            struct stat st {};
            if (stat(f.c_str(), &st) != 0) continue;
            bool exists = false;
            for (auto& d : g_kb_docs) {
                if (d.path == f) {
                    d.mtime = (long long)st.st_mtime;
                    d.bytes = (long long)st.st_size;
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                KbDoc d;
                d.path = f;
                d.mtime = (long long)st.st_mtime;
                d.bytes = (long long)st.st_size;
                g_kb_docs.push_back(std::move(d));
                added++;
            }
        }
        kb_rebuild();
        kb_save_manifest();
        json::Value o = kb_snapshot();
        o.set("added", json::Value::make_num((double)added));
        c.send(200, "application/json", json::dump(o));
    });

    srv.post("/api/kb/remove", [&](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        std::string p = req.param("path");
        if (p.empty() && json::parse(req.body, &body)) p = body.s("path");
        std::lock_guard<std::mutex> lk(g_kb_mu);
        kb_ensure_loaded();
        const size_t before = g_kb_docs.size();
        g_kb_docs.erase(std::remove_if(g_kb_docs.begin(), g_kb_docs.end(),
                                       [&](const KbDoc& d) { return d.path == p || path_base(d.path) == p; }),
                        g_kb_docs.end());
        kb_rebuild();
        kb_save_manifest();
        json::Value o = kb_snapshot();
        o.set("removed", json::Value::make_num((double)(before - g_kb_docs.size())));
        c.send(200, "application/json", json::dump(o));
    });

    srv.post("/api/kb/clear", [&](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        std::lock_guard<std::mutex> lk(g_kb_mu);
        g_kb_docs.clear();
        kb_rebuild();
        kb_save_manifest();
        c.send(200, "application/json", json::dump(kb_snapshot()));
    });

    srv.post("/api/kb/rag", [&](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        if (json::parse(req.body, &body)) g_kb_rag = body.flag("on", g_kb_rag);
        std::lock_guard<std::mutex> lk(g_kb_mu);
        kb_ensure_loaded();
        c.send(200, "application/json", json::dump(kb_snapshot()));
    });

    srv.post("/api/kb/search", [&](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        if (!json::parse(req.body, &body)) {
            c.send(400, "application/json", "{\"error\":\"JSON 解析失败\"}");
            return;
        }
        const std::string q = body.s("q");
        int k = (int)body.n("k", 6);
        if (k < 1) k = 1;
        if (k > 30) k = 30;
        std::lock_guard<std::mutex> lk(g_kb_mu);
        kb_ensure_loaded();
        const auto hits = kb_search(q, k);
        json::Value root = json::Value::make_obj();
        json::Value arr = json::Value::make_arr();
        for (const auto& h : hits) {
            json::Value o = json::Value::make_obj();
            o.set("path", json::Value::make_str(h.path));
            o.set("name", json::Value::make_str(path_base(h.path)));
            o.set("score", json::Value::make_num(h.score));
            o.set("snippet", json::Value::make_str(h.snippet));
            arr.arr.push_back(o);
        }
        root.set("hits", arr);
        root.set("q", json::Value::make_str(q));
        c.send(200, "application/json", json::dump(root));
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

        // ---- 串行化：整条请求（含多模态编码和工具两轮）放进同一把锁
        //
        // 先澄清一件事：**生成本身早就是串行的**。engine.cpp 的 dubhe_generate
        // 进来就 lock_guard(e->mu)，一直持到出结果，而且开头就 reset_kv()。
        // 所以两个请求同时来不会互相踩 KV，也不会串台 —— 这一点
        // tools/concurrent_check.py 能验证（改动前的二进制同样通过）。
        //
        // 这把锁补的是引擎锁**盖不住**的那一段：
        //
        //   1. build_tokens 会跑视觉/语音编码器，它调 dubhe_encode_image /
        //      dubhe_encode_audio，那些函数只在"赋值给 e->mm_embeds"那一瞬间
        //      加锁。于是 A 编码完、还没开始 generate 时，B 的编码可以把
        //      mm_embeds 覆盖掉 —— A 拿到的就是 B 的图。这条竞态只在
        //      "多模态 + 并发"时出现，纯文本复现不了。
        //   2. 工具调用是两轮 generate，中间引擎锁是松开的。
        //
        // 锁放这里而不是只包 generate，就是为了把上面两段一起罩住。
        // 后来的请求**排队等**而不是返回 409：本地单机场景下"等一下再答"
        // 比"直接失败"符合直觉；排到队时打一行日志，免得以为卡死了。
        std::unique_lock<std::mutex> gen_lk(g.mu, std::defer_lock);
        if (!gen_lk.try_lock()) {
            std::fprintf(stderr, "[dubhed] 生成中，请求排队等待…\n");
            gen_lk.lock();
        }
        g.busy = true;
        struct BusyGuard {
            ~BusyGuard() { g.busy = false; }
        } busy_guard;

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

        // 输入本身就超过上下文时，generate 会一开始就 break：没有报错、没有
        // 正文，界面上是"生成中"转很久然后什么都没有。这里直接说清楚。
        dubhe_model_info lim{};
        dubhe_model_info_get(g.engine, &lim);
        if (lim.n_ctx_limit > 0 && n >= lim.n_ctx_limit) {
            c.send(400, "application/json",
                   "{\"error\":{\"message\":" +
                       json_escape("输入太长：" + std::to_string(n) + " tokens，超过当前上下文上限 " +
                                   std::to_string(lim.n_ctx_limit) +
                                   "。请缩短问题、清空历史，或换一个上下文更长的模型。") +
                       ",\"type\":\"invalid_request_error\"}}");
            return;
        }

        const std::string id = now_id();
        long long created = (long long)time(nullptr);

        if (stream) {
            c.begin_stream();

            // 当前这一轮的正文。每轮单独攒：工具调用要从**本轮**输出里找，
            // 拿整个累积文本去找的话，第一轮的 tool_call 在第二轮还会被再匹配到。
            std::string rtext;
            // 逐字段赋值而不是聚合初始化：这个结构体会加字段（tail、no_suppress
            // 都是后补的），聚合初始化一漏就悄悄错位到相邻字段，编译器只会
            // 给一句 missing field 警告。
            StreamCtx ctx{};
            ctx.c = &c;
            ctx.acc = &rtext;
            ctx.id = &id;
            ctx.created = created;

            dubhe_stats st{};

            // 调用方自己跑循环时服务端一轮都不执行；否则最多跑两轮工具。
            //
            // 上限不是随手定的：2B 级别的模型看到工具结果后经常又吐一次
            // tool_call（尤其结果里有代码/JSON 时），不设上限会来回打转。
            // 两轮足够覆盖"读文件→读目录→回答"这种最常见的组合。
            const bool client_drives = has_tool_result(*msgs);
            const int max_rounds = client_drives ? 0 : 2;
            int rounds = 0;
            std::vector<int32_t> cur = toks;
            std::string last_sig;      // 上一次执行的工具签名，用来发现原地打转
            const char* finish = "stop";
            // 这一轮喂进去的是哪段 prompt（前缀缓存的代号）。生成完要拿它
            // 把"新吐出来的字和 token"接到缓存尾巴上。
            uint64_t pcid = g_pcache.current_id();

            while (!cur.empty()) {
                rtext.clear();
                ctx.acc = &rtext;
                ctx.first = true;
                ctx.decided = false;
                ctx.suppress = false;
                ctx.pending.clear();
                ctx.tail.clear();
                ctx.in_think = false;
                ctx.think_buf.clear();
                ctx.gtoks.clear();
                // 有工具说明时才做抑制判定 —— 否则根本没在找 tool_call，
                // 直接往外吐就是最快的路。
                ctx.no_suppress = !with_tools;

                dubhe_stats s{};
                dubhe_generate(g.engine, cur.data(), (int32_t)cur.size(), &sp, on_piece, nullptr,
                               &ctx, &s);
                flush_pending(&ctx);

                st.n_generated += s.n_generated;
                st.decode_tps = s.decode_tps;
                if (s.n_prompt) st.n_prompt = s.n_prompt;
                // 工具轮的第二轮几乎整段都是上一轮算过的，复用数按轮累加
                // 才有意义：单看最后一轮会小得让人以为缓存没生效。
                st.n_reused += s.n_reused;

                // 把这一轮新吐出来的字和 token 接到前缀缓存尾巴上。
                //
                // 接的是 rtext 而不是 emitted：KV 里装的是引擎吐出来的**全部**
                // token，包括被抑制掉、没发给用户的工具调用原文。拿 emitted
                // 去接的话，工具轮第二轮的 prompt 就和 KV 对不上了 ——
                // 而工具轮恰恰是最需要复用的一轮（3600 token 的 prefill）。
                if (!ctx.gtoks.empty()) g_pcache.extend(pcid, rtext, ctx.gtoks);

                if (!with_tools) break;

                std::string tname;
                json::Value targs;
                size_t endpos = 0;
                if (!extract_tool_call(rtext, &tname, &targs, &endpos)) {
                    // 开头的字符看着像工具调用，原文被藏了，可解析不出来
                    // （模型少个尖括号、JSON 截断在 max_tokens 上，都很常见）。
                    // 藏了又没人接手 —— 用户看到的就是"已完成 · 0 tokens"加一片空白。
                    // 原文吐回去，难看至少能读。
                    if (ctx.suppress) emit_delta(&ctx, rtext);
                    break;
                }

                // 调用方自己在跑循环 → 只把调用报回去，服务端不执行。
                if (client_drives) {
                    emit_tool_calls_delta(&ctx, tname, targs);
                    c.write_chunk("data: {\"dubhe\":{\"type\":\"tool_call\",\"name\":" +
                                  json_escape(tname) + ",\"arguments\":" + json::dump(targs) +
                                  "}}\n\n");
                    finish = "tool_calls";
                    break;
                }
                if (rounds >= max_rounds) {
                    // 到顶了模型还在要工具。把原始 JSON 摆给用户没有意义 ——
                    // 他看不懂，也做不了什么，只能以为程序坏了。说清楚发生了什么，
                    // 已经拿到的工具结果还留在上面的卡片里。
                    emit_delta(&ctx, "（模型反复请求调用工具 " + tname + "，已达 " +
                                         std::to_string(max_rounds) +
                                         " 轮上限，已停止。可以换个问法再试。）");
                    break;
                }

                // 同一个工具、同一组参数，再跑一遍结果一模一样，循环不可能收敛。
                // 2B 级别的模型很容易卡在这里（尤其"读文件→再读一次"），
                // 直接停下比再等一轮十几秒的 prefill 强。
                const std::string sig = tname + "|" + json::dump(targs);
                if (sig == last_sig) {
                    emit_delta(&ctx, "（模型在重复调用同一个工具 " + tname +
                                         "，结果不会变，已停止。可以换个问法再试。）");
                    break;
                }
                last_sig = sig;

                c.write_chunk("data: {\"dubhe\":{\"type\":\"tool_call\",\"name\":" +
                              json_escape(tname) + ",\"arguments\":" + json::dump(targs) +
                              "}}\n\n");

                ToolResult tr = run_tool(tname, targs);
                c.write_chunk("data: {\"dubhe\":{\"type\":\"tool_result\",\"ok\":" +
                              std::string(tr.ok ? "true" : "false") + ",\"output\":" +
                              json_escape(tr.output.substr(0, 8000)) + "}}\n\n");
                rounds++;

                // 下一轮：把这一轮的调用和结果追加进对话。
                //
                // 助手这一轮**必须重建成规范的 <tool_call> 包裹形式**，不能原样
                // 塞回模型输出。模型经常吐裸 JSON（{"name":...}），原样拼回去
                // 它自己都认不出那是刚才的调用，于是把同一个工具再要一遍 ——
                // 实测 Qwen2.5 就是这样：第一轮 .gitignore，第二轮 /.gitignore。
                //
                // 但"重建"会让这段文本和模型刚才真正吐出来的逐字节不同，
                // token 边界一变，第一轮算好的 KV 就整段作废（3600 token 的
                // prefill 是十几秒）。所以分两种情况：模型自己就写对了规范
                // 格式时原样用（KV 能接上），只有它吐裸 JSON 时才重建 ——
                // 那种情况本来也对不上，重建不影响什么。
                json::Value call = json::Value::make_obj();
                call.set("name", json::Value::make_str(tname));
                call.set("arguments", targs);
                json::Value msgs2 = *msgs;
                json::Value am = json::Value::make_obj();
                am.set("role", json::Value::make_str("assistant"));
                std::string call_text;
                if (g_tool_fmt == 1) {
                    call_text = "<tool_call>\n<function=" + tname + ">\n";
                    if (targs.type == json::Value::OBJ) {
                        for (const auto& kv : targs.obj) {
                            std::string v = (kv.second.type == json::Value::STR)
                                                ? kv.second.str
                                                : json::dump(kv.second);
                            call_text += "<parameter=" + kv.first + ">\n" + v + "\n</parameter>\n";
                        }
                    }
                    call_text += "</function>\n</tool_call>";
                } else {
                    call_text = "<tool_call>" + json::dump(call) + "</tool_call>";
                }
                // 模型自己就写成了规范包裹形式 → 原样用它，KV 才能接上。
                // 判据看 rtext 而不是 emitted：rtext 才是 KV 里那段 token 的
                // 文本（工具调用本来就被抑制，emitted 是空的）。
                std::string am_content = call_text;
                {
                    const size_t s0 = rtext.find_first_not_of(" \t\r\n");
                    if (s0 != std::string::npos && rtext.compare(s0, 11, "<tool_call>") == 0) {
                        am_content = rtext;
                    }
                }
                am.set("content", json::Value::make_str(am_content));
                msgs2.arr.push_back(am);
                json::Value tm = json::Value::make_obj();
                tm.set("role", json::Value::make_str("tool"));
                tm.set("content", json::Value::make_str(tr.output));
                msgs2.arr.push_back(tm);

                // 工具结果可能比整个上下文还长（read_file 读一个 10KB 的
                // README 就接近 4096 的上限）。不截的话第二轮 prompt 直接
                // 越过 n_ctx，generate 一开始就 break —— 界面上是"生成中"
                // 转两分钟然后什么都没有，比报错难查得多。
                //
                // 按实际 token 数往回缩，而不是按字符数猜比例：中英混排的
                // 字符/ token 比能差 4 倍。
                dubhe_model_info ctx_mi{};
                dubhe_model_info_get(g.engine, &ctx_mi);
                const int reserve = std::min((int)sp.max_tokens, 512);
                const int room = ctx_mi.n_ctx_limit - reserve;

                std::vector<int32_t> t2;
                std::string e2;
                size_t keep = tr.output.size();
                bool truncated = false;
                for (int attempt = 0; attempt < 6; attempt++) {
                    std::string body = tr.output;
                    if (keep < body.size()) {
                        // 别切在半个汉字上：往回退到 UTF-8 边界
                        size_t cut = keep;
                        while (cut > 0 && ((unsigned char)body[cut] & 0xC0) == 0x80) cut--;
                        body = body.substr(0, cut) +
                               "\n\n…（结果过长，已截断到 " + std::to_string(cut) + " 字节）";
                        truncated = true;
                    }
                    msgs2.arr.back().set("content", json::Value::make_str(body));
                    t2.clear();
                    if (!build_tokens(g.engine, msgs2, chatml, with_tools, force_raw, &t2, &mcache,
                                      &e2)) {
                        c.write_chunk("data: {\"dubhe\":{\"type\":\"error\",\"message\":" +
                                      json_escape(e2) + "}}\n\n");
                        break;
                    }
                    if ((int)t2.size() <= room) break;
                    keep = keep * 3 / 4;
                }
                if (t2.empty()) break;
                if ((int)t2.size() > room) {
                    emit_delta(&ctx, "（工具返回的内容超过了上下文长度，即使截断也放不下。"
                                     "换个小一点的读取目标再试。）");
                    break;
                }
                if (truncated) {
                    c.write_chunk("data: {\"dubhe\":{\"type\":\"note\",\"message\":" +
                                  json_escape("工具返回的内容较长，已截断到 " +
                                              std::to_string(t2.size()) + " tokens 以内") +
                                  "}}\n\n");
                }
                cur = std::move(t2);
                // 第二轮重新拼了 prompt，前缀缓存的代号换了 —— 这一轮生成完
                // 要往"新的那一段"上接，不能再用第一轮的代号。
                pcid = g_pcache.current_id();
            }

            g.last_decode_tps = st.decode_tps;
            g.total_requests++;
            g.total_tokens += st.n_generated;

            std::string done = "data: {\"id\":\"" + id +
                               "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" +
                               std::string(finish) + "\"}],\"usage\":{\"completion_tokens\":" +
                               std::to_string(st.n_generated) + ",\"prompt_tokens\":" +
                               std::to_string(st.n_prompt) + ",\"decode_tps\":" +
                               std::to_string(st.decode_tps) + ",\"reused_tokens\":" +
                               std::to_string(st.n_reused) + "}}\n\n";
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
        usage.set("reused_tokens", json::Value::make_num(st.n_reused));
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
