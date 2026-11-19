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
#include "calc.h"

#include <ifaddrs.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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
    // ---- 权限位
    //
    // 这几项以前只有"启动参数决定、之后就定死"这一种形态。用户想开终端
    // 就得重启进程、还得记住 --allow-shell 这个只有源码里才知道的名字 ——
    // 实际上等于没有权限设置。现在它们可以在运行期从 UI 改，改完立刻生效。
    //
    // 默认值改了：shell / net 默认**开**。理由是这是个本机单用户工具，
    // "模型说我没有执行命令的权限、而界面上又找不到地方能开"是用户实际
    // 报上来的问题；真正的风险由 sandbox 兜（写盘只允许落在工作区与 /tmp，
    // 命令有超时和输出上限）。要收紧就在设置里关掉，或者启动时用
    // --no-shell / --no-net 强制关。
    bool allow_shell = true;    // 执行 shell 命令
    bool allow_write = true;    // 写文件 / 改文件
    bool allow_net = true;      // 联网（web_search / fetch_url）
    bool sandbox = true;        // shell 用 seatbelt 限制写盘范围
    // 一次提问里最多跑几轮工具。
    //
    // 原来是写死的 2。2 轮只够"读文件 → 回答"，而实际任务经常是
    // "grep 找位置 → 读那一处 → 再读一个相关文件 → 改 → 回答"，第三轮就被
    // 掐断，用户拿到的是"已达 2 轮上限"加一坨没消化完的材料。上限本身是必要
    // 的（2B 模型看到工具结果会反复要工具），但 2 太紧；改成可调，默认 6，
    // 同时保留"同一个工具同样的参数连调两次就停"的原地打转保护。
    int max_tool_rounds = 6;
    // 纯算式短路（见 try_arith）。默认开：算术有唯一正确答案，交给模型算
    // 只会得到"自信的错"。真要看模型怎么算，用 --no-arith 关掉。
    int no_arith_fastpath = 0;
    // 写域默认只到当前项目文件夹。读可以看整个家目录（模型权重就在
    // ~/.dubhe 下，读不到等于残废），但"能读"和"能改"不该是同一个范围 ——
    // 这是 DeepSeek Harness 的 workspace-write 默认值，也是用户把文件夹
    // 绑成项目文件夹时预期的语义。要放开就在设置里打开。
    bool allow_write_outside = false;
    int  shell_timeout_ms = 20000;
    // ---- 逐次审批
    //
    // 工具循环里遇到 gate() 的 Ask，要停下来等用户点"允许/拒绝"。
    //
    // 这里**必须**用独立的互斥量，不能借用 g.mu：整条 chat 请求从头到尾
    // 压在 g.mu 上（请求级 workspace 就是靠这个串行化），等待期间如果还
    // 需要 g.mu，/api/approve 永远拿不到锁 —— 表现是点"允许"毫无反应，
    // 一直等到超时。这个坑不写下来下次一定还会踩。
    std::mutex ask_mu;
    std::condition_variable ask_cv;
    std::map<std::string, int> ask_result;   // id → 1 允许 / 2 拒绝
    std::atomic<long long> ask_seq{0};
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
    // 文件名加成。
    //
    // 用户问"ADR-0003 决定的是什么"，期望的是那个文件本身；而 BM25 只看
    // 块的正文 —— "0003" 根本不在正文里，于是检索回来的是 ADR **模板**
    // 那几篇（正文里到处是 ADR 这个词）。实测就是这样：问 ADR-0003，
    // 命中 docs/adr/README.md，1.7B 拿着模板照猫画虎编了一份"ADR-0.003"。
    //
    // 文件名短、信息密度高，命中它比命中正文更能说明"就是这篇"。所以给
    // "查询词出现在文件名里"加一份固定分。6.0 是按"一条明显相关的命中
    // 在几百个块上通常 10 分以上"估的：够把对的推上来，又不至于让文件名
    // 里的一个泛词（readme）压过正文里真正的多处命中。
    std::unordered_map<int, double> doc_bonus;
    // 查询里的纯数字词（0003 / 2119）。这类词只在**文件名**里出现，正文里
    // 一个都没有 —— 见下面"编号命中"的注释。
    std::vector<std::string> nums;
    for (const auto& t : uniq) {
        if (t.size() < 2) continue;
        bool digits = true;
        for (char ch : t)
            if (!std::isdigit((unsigned char)ch)) { digits = false; break; }
        if (digits) nums.push_back(t);
    }
    auto bonus_of = [&](int di) {
        auto it = doc_bonus.find(di);
        if (it != doc_bonus.end()) return it->second;
        double add = 0;
        const auto nt = kb_tokenize(path_base(g_kb_docs[(size_t)di].path));
        for (const auto& t : uniq) {
            if (t.size() < 2) continue;
            if (std::find(nt.begin(), nt.end(), t) != nt.end()) add += 6.0;
        }
        // 编号命中。
        //
        // 文件名加成是"每个命中词 +6"，可 ADR 系列每一篇的文件名里都有 "adr"：
        // 问 ADR-0003 时，ADR-0003.md 拿 12 分，ADR-0004.md 也拿 6 分 —— 中间
        // 这 6 分差距常常小于正文 BM25 的波动，于是排第一的还是错的那篇。
        // 数字部分才是真正的区分点：文件名里带 0003 的那一篇就是用户要的那一篇。
        // +40 是"压过一切正文分差"的量级（这套语料正文分在 5~26 之间），
        // 而且只有"文件名里真的有这个编号"时才触发，普通提问一分都不受影响。
        for (const auto& num : nums) {
            if (std::find(nt.begin(), nt.end(), num) != nt.end()) { add += 40.0; break; }
        }
        doc_bonus.emplace(di, add);
        return add;
    };
    for (auto& kv : score) kv.second += bonus_of(g_kb_chunks[(size_t)kv.first].doc);
    for (const auto& kv : score) order.emplace_back(kv.second, kv.first);
    std::sort(order.begin(), order.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    if ((int)order.size() > topk) order.resize((size_t)topk);
    // 按查询长度归一。
    //
    // BM25 的原始分**不能跨查询比较**：一个 20 个汉字的问题切出十几个 bigram，
    // 每个都能撞上几个常见词，加起来自然高；两个字的词只有一两个 bigram，
    // 就算命中的是决定性术语，分也上不去。实测：
    //
    //   "帮我把这段话翻译成英文：今天天气不错"  19.72  ← 和文档毫无关系
    //   "monorepo"                              11.72  ← 真该命中
    //   "你好"                                   7.07  ← 不该命中
    //
    // 只看分数的话，闲聊排在正经提问前面。除以 sqrt(查询词数) 之后：
    // 19.72/4.4=4.5、11.72/1=11.7、7.07/1=7.1 —— 顺序才和直觉一致。
    const double norm = std::sqrt((double)std::max<size_t>(1, uniq.size()));
    for (const auto& [sc, ci] : order) {
        const KbChunk& ch = g_kb_chunks[ci];
        KbHit h;
        h.chunk = ci;
        h.score = sc / norm;
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
// 门槛 7.5，用的是**按查询长度归一之后**的分数（见 kb_search 末尾）。
//
// 归一之前这道线是 8.0，而它拦不住真正该拦的东西 —— 一个长问题切出十几个
// bigram，随便撞几个常见词就能到 19 分：
//
//   "帮我把这段话翻译成英文：今天天气不错"  19.72  ← 和文档毫无关系，
//                                                   1.7B 会照着文档答非所问
//   "monorepo"                              11.72  ← 真该命中
//   "你好"                                   7.07  ← 不该命中
//
// 归一之后（除以 sqrt(查询词数)）同一批查询变成 4.5 / 11.7 / 7.1，7.5 这道线
// 才真的分得开"问项目"和"问别的"。
std::string kb_context_for(const std::string& q, int topk) {
    // 多取一些再挑。
    //
    // 同一篇文档常常有两三个块一起命中，直接把 top3 塞进去就会变成
    // "三条全是同一篇"：问 ADR-0003，回来的两条是 ADR-0004 的开头两段，
    // 真正那一篇被挤到看不见。这里改成每篇最多两块，先把不同的文档凑齐。
    const auto hits = kb_search(q, std::max(topk * 3, 8));
    if (hits.empty() || hits[0].score < 7.5) return "";
    std::vector<const KbHit*> picked;
    std::unordered_map<std::string, int> per_doc;
    for (const auto& h : hits) {
        if (h.score < 7.5) break;               // 排好序的，后面的只会更低
        if (per_doc[h.path] >= 2) continue;
        per_doc[h.path]++;
        picked.push_back(&h);
        if ((int)picked.size() >= topk) break;
    }
    if (picked.empty()) return "";
    // 措辞很重要。原来写的是"回答时优先依据它们"，结果 1.7B 会把资料当圣旨：
    // 问"今天天气怎么样"，它去复述文档里那句"今天天气很好，我们一起"的
    // 测试样例 —— 检索没错（那句话确实在库里），错在没给它"这跟我无关"
    // 这个选项。现在明确说可能不相关、不相关就当没看见。
    std::string s = "\n\n以下是从本地知识库检索到的片段，可能相关，也可能不相关。\n"
                    "只有确实能回答用户问题时才引用；不相关就忽略，不要复述，"
                    "也不要说「资料里提到」。\n";
    for (size_t i = 0; i < picked.size(); i++) {
        s += "\n[" + std::to_string(i + 1) + "] " + path_base(picked[i]->path) + "\n";
        s += picked[i]->snippet + "\n";
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
    // 被门禁挡下来（不是工具执行失败）。前端据此把卡片画成"未执行"，
    // 而不是把一句权限提示渲染成命令输出。
    bool denied = false;
    // ---- 三层分离里的另外两层
    //
    // output 只服务模型（拼进 prompt 的自然语言）。card + meta 服务界面：
    // card 决定画哪种卡片，meta 是回放卡片要用的结构化事实。
    //
    // 这是从 DeepSeek Harness 的 presentCall/presentResult 抄的。原来只有
    // 一个 output 字符串，前端只能把它整块塞进 <pre>，于是 shell、读文件、
    // 改文件、搜索四种东西长成一个样 —— 用户看不出模型到底做了什么。
    std::string card = "generic";   // generic|terminal|diff|read|search|web
    json::Value meta = json::Value::make_obj();
};

// ---------------------------------------------------------------- 进程执行
//
// 为什么不用 popen：popen 没有超时、没有 cwd、拿不到退出码，而且命令是交给
// /bin/sh 在**当前进程的会话**里跑的 —— 模型写一个 `sleep 600` 就把整个
// 请求挂死，写一个 `rm -rf` 就直接删用户文件，中间没有任何一道闸。
//
// 这里改成 fork + setpgid + 非阻塞读 + 截止时间：超时整组 kill，输出超过
// 上限就截断并标注。进程组是关键 —— 只 kill 子进程的话，`sh -c "a | b"`
// 里被管道拉起来的 b 会活下来，继续往管道里写，读端永远等不到 EOF。

std::string home_dir() {
    const char* h = getenv("HOME");
    return h ? std::string(h) : std::string("/");
}

// 单引号包裹：shell 里单引号内没有任何元字符，唯一要处理的是单引号本身。
// 模型生成的命令里出现 `$(...)`、反引号、分号的概率很高，直接拼进去就是
// 命令注入 —— 这里的调用方本来就打算执行任意命令，但**不该**在执行前
// 被意外展开一次（比如命令里带 $PATH 会被服务端的环境替换掉）。
std::string shell_quote(const std::string& s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\'') o += "'\\''";
        else o += c;
    }
    o += "'";
    return o;
}


struct ProcOut {
    int  exit_code = -1;
    bool timed_out = false;
    bool truncated = false;
    std::string out;
};

// seatbelt 到底套不套得上：探一次，全局缓存。
//
// 为什么值得单独探一次：`sandbox-exec` 在外层已经有沙箱约束的进程里会直接
// 报 `sandbox_apply: Operation not permitted` 并以 71 退出 —— 命令**根本没
// 执行**。原来的做法是"先套沙箱跑一遍，看到 71 再裸跑一遍"，也就是每条命令
// 都要 fork 两次；更糟的是 `/api/state` 里 `sandbox` 一直写的是配置值 `true`，
// 界面上显示"已隔离"，实际早就退化成裸跑了 —— 这种"以为有保护其实没有"比
// 明确说没有更危险。
//
// 探测本身用最小 profile 跑 `/bin/true`：套得上退出码 0，套不上 71。
// 用 once_flag 保证只探一次，多个 shell 工具并发进来也只 fork 一次。
bool seatbelt_works() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        const int rc = std::system(
            "/usr/bin/sandbox-exec -p '(version 1)(allow default)' /bin/true "
            ">/dev/null 2>&1");
        ok = (rc == 0);
        if (!ok) {
            std::fprintf(stderr,
                         "[dubhe] seatbelt 不可用（sandbox_apply 被拒绝），"
                         "shell 将裸跑，写盘限制不生效\n");
        }
    });
    return ok;
}

// 实际生效的沙箱状态 = 用户开了 且 本机套得上。给 /api/state 和工具用同一个口径。
bool sandbox_active() { return g.sandbox && seatbelt_works(); }

ProcOut run_proc(const std::string& cmd, const std::string& cwd,
                 int timeout_ms, size_t max_bytes, bool sandbox_writes) {
    ProcOut r;
    int fds[2];
    if (pipe(fds) != 0) { r.out = "无法创建管道"; return r; }

    // 沙箱：只允许写工作区与临时目录。
    //
    // sandbox-exec 在 macOS 上早已标记废弃，但它是系统自带的、唯一不需要
    // 额外安装的写盘隔离手段。deny default 太脆（连 dylib 都读不到），
    // 所以这里保留全盘可读、只收紧写 —— 对"模型把用户文件改坏"这一类
    // 事故是有效的，对"模型读到敏感文件"无效。想连读也收紧就自己改 profile。
    std::string wrapped = cmd;
    if (sandbox_writes) {
        char tmpl[] = "/tmp/dubhe-sb-XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd >= 0) {
            const std::string prof =
                "(version 1)\n(allow default)\n"
                "(deny file-write*)\n"
                "(allow file-write* (subpath \"" + g.workspace + "\"))\n"
                "(allow file-write* (subpath \"" + (g.tmp_dir.empty() ? "/tmp" : g.tmp_dir) + "\"))\n"
                "(allow file-write* (subpath \"/private/tmp\"))\n"
                "(allow file-write* (subpath \"/private/var/folders\"))\n";
            if (write(fd, prof.data(), prof.size()) == (ssize_t)prof.size()) {
                close(fd);
                wrapped = "/usr/bin/sandbox-exec -f " + std::string(tmpl) + " /bin/sh -c " +
                          shell_quote(cmd);
            } else {
                close(fd);
                unlink(tmpl);
                wrapped = "/bin/sh -c " + shell_quote(cmd);
            }
        } else {
            wrapped = "/bin/sh -c " + shell_quote(cmd);
        }
    } else {
        wrapped = "/bin/sh -c " + shell_quote(cmd);
    }

    // 环境变量只留必要的。把服务端进程的全部环境（包括 --api-key、各种
    // token）交给模型生成的命令，等于把密钥交给一段自己看不懂的文本。
    // 这些必须在 fork 之前拼好 —— 子进程里不能分配内存（见下面 execve 处）。
    const std::string home = home_dir();
    std::vector<std::string> envs = {
        "PATH=/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG=en_US.UTF-8",
        "TERM=dumb",
        "DUBHE_SANDBOXED=1",
    };
    if (!home.empty()) envs.push_back("HOME=" + home);
    std::vector<char*> sh_argv;
    sh_argv.push_back(const_cast<char*>("/bin/sh"));
    sh_argv.push_back(const_cast<char*>("-c"));
    sh_argv.push_back(const_cast<char*>(wrapped.c_str()));
    sh_argv.push_back(nullptr);
    std::vector<char*> envp;
    envp.reserve(envs.size() + 1);
    for (auto& e : envs) envp.push_back(const_cast<char*>(e.c_str()));
    envp.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); r.out = "fork 失败"; return r; }
    if (pid == 0) {
        // 子进程：独立进程组，方便整组回收
        setpgid(0, 0);
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);   // 错误也要能看到，否则失败信息全丢
        close(fds[1]);
        if (!cwd.empty()) { if (chdir(cwd.c_str()) != 0) { /* 落回继承的 cwd */ } }
        // 注意：fork 之后**不能**再做堆分配。这是个多线程服务，别的线程
        // 可能正握着 malloc 的锁，子进程一分配就会永久卡住。所以环境数组
        // 在 fork 之前就拼好（见上面的 envs/envp）。
        execve("/bin/sh", sh_argv.data(), envp.data());
        _exit(127);
    }

    setpgid(pid, pid);   // 与子进程里的 setpgid 双写，避免竞态
    close(fds[1]);
    fcntl(fds[0], F_SETFL, fcntl(fds[0], F_GETFL, 0) | O_NONBLOCK);

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 20000);
    char buf[8192];
    bool eof = false, killed = false;
    int status = 0;
    while (true) {
        while (!eof) {
            const ssize_t n = read(fds[0], buf, sizeof(buf));
            if (n > 0) {
                if (r.out.size() < max_bytes) {
                    const size_t room = max_bytes - r.out.size();
                    r.out.append(buf, (size_t)std::min((size_t)n, room));
                    if ((size_t)n > room) r.truncated = true;
                } else {
                    r.truncated = true;
                }
                continue;
            }
            if (n == 0) { eof = true; break; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            eof = true;   // EINTR 之类，当作结束
            break;
        }
        const pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) break;
        if (!killed && std::chrono::steady_clock::now() > deadline) {
            kill(-pid, SIGKILL);
            kill(pid, SIGKILL);
            killed = true;
            r.timed_out = true;
        }
        struct timespec ts { 0, 2 * 1000 * 1000 };   // 2ms
        nanosleep(&ts, nullptr);
    }
    // 把管道里剩下的读干净（进程已退出，这里不会阻塞太久）
    while (true) {
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n <= 0) break;
        if (r.out.size() < max_bytes) {
            const size_t room = max_bytes - r.out.size();
            r.out.append(buf, (size_t)std::min((size_t)n, room));
            if ((size_t)n > room) r.truncated = true;
        } else { r.truncated = true; }
    }
    close(fds[0]);
    if (WIFEXITED(status)) r.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) r.exit_code = 128 + WTERMSIG(status);
    if (sandbox_writes) { /* 临时 profile 由系统回收，/tmp 下的残留无害 */ }
    return r;
}

// ---------------------------------------------------------------- HTML → 文本
//
// 抓网页给模型看，不能把原始 HTML 直接塞进去：一个普通的新闻页有 200 KB
// 标签、5 KB 正文，全塞进去既超上下文又全是噪声。这里做最小可用的抽取：
// 去掉 script/style/注释，块级标签转换行，其余标签删掉，实体还原。
std::string html_unescape(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '&') { o += s[i]; continue; }
        const size_t sc = s.find(';', i);
        if (sc == std::string::npos || sc - i > 8) { o += s[i]; continue; }
        const std::string e = s.substr(i + 1, sc - i - 1);
        if (e == "amp") o += '&';
        else if (e == "lt") o += '<';
        else if (e == "gt") o += '>';
        else if (e == "quot") o += '"';
        else if (e == "apos" || e == "#39") o += '\'';
        else if (e == "nbsp") o += ' ';
        else if (!e.empty() && e[0] == '#') {
            const long cp = (e.size() > 1 && (e[1] == 'x' || e[1] == 'X'))
                                ? strtol(e.c_str() + 2, nullptr, 16)
                                : strtol(e.c_str() + 1, nullptr, 10);
            // UTF-8 编码一个码点
            if (cp > 0 && cp < 0x80) o += (char)cp;
            else if (cp >= 0x80 && cp < 0x800) {
                o += (char)(0xC0 | (cp >> 6));
                o += (char)(0x80 | (cp & 0x3F));
            } else if (cp >= 0x800 && cp < 0x10000) {
                o += (char)(0xE0 | (cp >> 12));
                o += (char)(0x80 | ((cp >> 6) & 0x3F));
                o += (char)(0x80 | (cp & 0x3F));
            } else if (cp >= 0x10000 && cp < 0x110000) {
                o += (char)(0xF0 | (cp >> 18));
                o += (char)(0x80 | ((cp >> 12) & 0x3F));
                o += (char)(0x80 | ((cp >> 6) & 0x3F));
                o += (char)(0x80 | (cp & 0x3F));
            } else o += '?';
        } else { o += s[i]; continue; }
        i = sc;
    }
    return o;
}

std::string html_to_text(const std::string& html) {
    std::string s = html;
    // 丢掉不可能含正文的整段
    auto drop_block = [&](const char* open, const char* close) {
        for (;;) {
            const size_t a = s.find(open);
            if (a == std::string::npos) break;
            const size_t b = s.find(close, a);
            if (b == std::string::npos) { s.erase(a); break; }
            s.erase(a, b + std::strlen(close) - a);
        }
    };
    drop_block("<script", "</script>");
    drop_block("<style", "</style>");
    drop_block("<!--", "-->");
    drop_block("<svg", "</svg>");

    std::string o;
    o.reserve(s.size() / 2);
    bool in_tag = false;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (in_tag) {
            if (c == '>') in_tag = false;
            continue;
        }
        if (c == '<') {
            // 块级标签留一个换行，否则整页会挤成一行
            const char* blocks[] = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"};
            for (const char* b : blocks) {
                const size_t n = std::strlen(b);
                if (s.compare(i + 1, n, b) == 0) { o += '\n'; break; }
            }
            in_tag = true;
            continue;
        }
        o += c;
    }
    o = html_unescape(o);
    // 折叠空白：连续空格/空行压掉，保留段落
    std::string t;
    t.reserve(o.size());
    int nl = 0;
    for (size_t i = 0; i < o.size(); i++) {
        const char c = o[i];
        if (c == '\n') {
            if (++nl <= 2) t += '\n';
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r') {
            if (!t.empty() && t.back() != ' ' && t.back() != '\n') t += ' ';
            continue;
        }
        nl = 0;
        t += c;
    }
    return t;
}

// DuckDuckGo 的 HTML 端点。选它是因为**不需要 API Key** —— 一个本地
// 单用户工具没理由为了搜一次就要求用户去注册。
//
// 结果同时给两份：拼成文本给模型读，结构化成 WebHit 给界面画。
// 只给文本的话，前端要自己拿正则去啃"    https://…"这种缩进约定，
// 哪天文本格式为了模型好读调一下，界面就跟着瞎 —— 而这两件事本来
// 就不该绑在一起。
struct WebHit {
    std::string title;
    std::string url;
    std::string snippet;
};

std::vector<WebHit> web_search_hits(const std::string& q, int n) {
    std::string qs;
    for (unsigned char c : q) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.') qs += (char)c;
        else if (c == ' ') qs += '+';
        else { char b[8]; std::snprintf(b, sizeof(b), "%%%02X", c); qs += b; }
    }
    const std::string url = "https://html.duckduckgo.com/html/?q=" + qs;
    ProcOut po = run_proc("curl -sSL --compressed -m 20 -A 'Mozilla/5.0 (Macintosh)' " +
                              shell_quote(url),
                          g.workspace, 25000, 2 * 1024 * 1024, false);
    std::vector<WebHit> hits;
    if (po.out.empty()) return hits;

    // 解析结果块：标题在 class="result__a" 的 <a> 里，摘要在 class="result__snippet"
    size_t p = 0;
    while ((int)hits.size() < n) {
        const size_t a = po.out.find("result__a", p);
        if (a == std::string::npos) break;
        const size_t gt = po.out.find('>', a);
        const size_t end = po.out.find("</a>", gt);
        if (gt == std::string::npos || end == std::string::npos) break;
        std::string title = html_to_text(po.out.substr(gt + 1, end - gt - 1));
        // 真实地址藏在 href 的 uddg= 参数里。
        // 必须**向后**找 href，并且限定在这个 <a> 标签的 '>' 之前：
        // DuckDuckGo 的属性顺序是 class 在前、href 在后，用 rfind 往前找
        // 会拿到上一条结果的链接（实测第二条结果的 URL 就是第一条的）。
        std::string href;
        const size_t hs = po.out.find("href=\"", a);
        if (hs != std::string::npos && hs < gt) {
            const size_t he = po.out.find('"', hs + 6);
            if (he != std::string::npos) {
                std::string raw = po.out.substr(hs + 6, he - hs - 6);
                const size_t u = raw.find("uddg=");
                if (u != std::string::npos) {
                    std::string enc = raw.substr(u + 5);
                    const size_t amp = enc.find('&');
                    if (amp != std::string::npos) enc.resize(amp);
                    href = http::Request::url_decode(enc);
                } else href = raw;
            }
        }
        std::string snip;
        const size_t sa = po.out.find("result__snippet", end);
        if (sa != std::string::npos) {
            const size_t sgt = po.out.find('>', sa);
            const size_t send = po.out.find("</a>", sgt);
            if (sgt != std::string::npos && send != std::string::npos)
                snip = html_to_text(po.out.substr(sgt + 1, send - sgt - 1));
        }
        if (title.empty()) { p = end; continue; }
        hits.push_back(WebHit{title, href, snip});
        p = (sa == std::string::npos) ? end : sa + 16;
    }
    return hits;
}

// 前置声明：字符串替换的实现在后面（跟算术短路放一起），这里先用。
static void str_replace_all(std::string* s, const std::string& from, const std::string& to);

// 把"今天/最近"换成具体日期再搜。
//
// 为什么非换不可：搜索引擎不认识"今天"。实测搜"今天新闻"，DuckDuckGo 回的
// 是今日头条、Google 新闻、BBC 中文三个**门户首页** —— 模型再老实，也只能
// 把这三个首页标题抄一遍，用户拿到的是"新闻网站的目录"而不是新闻。
// 换成 "2026年11月18日 新闻"，才可能搜到当天真正发出来的稿子。
// 只在查询没写明年份时替换，用户自己写了日期就不动。
std::string localize_query(std::string q) {
    bool has_year = false;
    for (size_t i = 0; i + 4 <= q.size(); i++) {
        if (std::isdigit((unsigned char)q[i]) && std::isdigit((unsigned char)q[i + 1]) &&
            std::isdigit((unsigned char)q[i + 2]) && std::isdigit((unsigned char)q[i + 3])) {
            const int y = std::atoi(q.substr(i, 4).c_str());
            if (y >= 2000 && y <= 2099) { has_year = true; break; }
        }
    }
    if (has_year) return q;

    std::time_t t = std::time(nullptr);
    std::tm tm {};
    localtime_r(&t, &tm);
    char day[40], mon[40];
    std::snprintf(day, sizeof(day), "%d年%d月%d日", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday);
    std::snprintf(mon, sizeof(mon), "%d年%d月", tm.tm_year + 1900, tm.tm_mon + 1);

    bool changed = false;
    for (const char* k : {"今天", "今日"}) {
        if (q.find(k) != std::string::npos) { str_replace_all(&q, k, day); changed = true; }
    }
    if (!changed) {
        for (const char* k : {"最近", "最新"}) {
            if (q.find(k) != std::string::npos) { str_replace_all(&q, k, mon); changed = true; }
        }
    }
    return q;
}

// 抓一页正文，压成一段纯文本。
std::string fetch_page_text(const std::string& url, size_t limit) {
    const std::string cmd =
        "curl -sSL --compressed -m 12 -A 'Mozilla/5.0 (Dubhe)' " + shell_quote(url);
    ProcOut po = run_proc(cmd, g.workspace, 15000, 512 * 1024, false);
    std::string t = html_to_text(po.out);

    // 丢掉导航栏。
    //
    // html_to_text 只做标签剥离，菜单、面包屑、版权信息全留着。实测抓新浪的
    // 新闻页，前 400 字是"新浪首页 / 体育 / 财经 / 娱乐 / 博客 / 图片…"，
    // 真正的正文被挤出 1200 字的预算 —— 模型读到的还是噪声，等于白抓。
    // 新闻段落长且成句，导航项短且不成句，按行长度切一刀就能滤掉绝大部分。
    {
        std::string kept;
        size_t pos = 0;
        while (pos <= t.size()) {
            size_t nl = t.find('\n', pos);
            if (nl == std::string::npos) nl = t.size();
            std::string line = t.substr(pos, nl - pos);
            size_t a = line.find_first_not_of(" \t\r");
            size_t b = line.find_last_not_of(" \t\r");
            if (a != std::string::npos) {
                line = line.substr(a, b - a + 1);
                if (line.size() >= 24) kept += line + "\n";   // 24 字节 ≈ 8 个汉字
            }
            if (nl == t.size()) break;
            pos = nl + 1;
        }
        if (!kept.empty()) t = kept;
    }

    while (t.find("\n\n\n") != std::string::npos) str_replace_all(&t, "\n\n\n", "\n\n");
    if (t.size() > limit) {
        t.resize(limit);
        // 别切在半个 UTF-8 字符上
        while (!t.empty() && (t.back() & 0xC0) == 0x80) t.pop_back();
        t += "…";
    }
    return t;
}

// 给模型读的那一份。编号是必要的：模型要在回答里标"据 [2]"，
// 没有编号它只能含糊地说"有资料显示"。
std::string hits_to_text(const std::vector<WebHit>& hits) {
    std::string out;
    for (size_t i = 0; i < hits.size(); i++) {
        out += "[" + std::to_string(i + 1) + "] " + hits[i].title + "\n";
        if (!hits[i].url.empty()) out += "    " + hits[i].url + "\n";
        if (!hits[i].snippet.empty()) out += "    " + hits[i].snippet + "\n";
    }
    return out;
}

// ---------------------------------------------------------------- 工具门禁
//
// 策略集中在这一个函数里，不散落在每个工具内部。
//
// 这条是从 DeepSeek Harness 的 `tools/pre-execute` 抄的。它官方文档里有一句
// 话正好戳中 Dubhe 原来的做法：「尽量不要把部署策略内建到工具中」。而我们
// 之前是反例 —— allow_write 的判断在 write_file 和 edit_file 里各写一遍，
// allow_net 在 fetch_url 和 web_search 里各写一遍，加一个新工具就多一个
// 漏判点，而漏判的表现是"权限没生效"，不是报错，没人会发现。
//
// 三态而不是布尔：现在只会用到 Allow / Deny，但 Ask 的位置先留出来。将来
// 前端要做"这条命令要跑，确认吗"的时候（DSH 的 Web UI 就是这么做的），
// 只需要在 run_tool 里加一个分支，不用再改一遍九个工具。
enum class Verdict { Allow, Deny, Ask };

struct GateOut {
    Verdict v = Verdict::Deny;
    std::string reason;   // 给模型看的，必须能指导它下一步怎么做
    std::string detail;   // 给用户看的（审批卡片上那一行）
};

static GateOut gate_deny(const std::string& why) {
    GateOut o;
    o.v = Verdict::Deny;
    o.reason = why;
    return o;
}
static GateOut gate_ok() {
    GateOut o;
    o.v = Verdict::Allow;
    return o;
}
static GateOut gate_ask(const std::string& why, const std::string& detail) {
    GateOut o;
    o.v = Verdict::Ask;
    o.reason = why;
    o.detail = detail;
    return o;
}

// p 是否落在 root 之下（按路径段比，不是按字符串前缀 —— 否则 /a/bc 会被
// 判成在 /a/b 里面）。
static bool under_root(const std::string& p, const std::string& root) {
    if (root.empty() || p.size() < root.size()) return false;
    if (p.compare(0, root.size(), root) != 0) return false;
    return p.size() == root.size() || p[root.size()] == '/';
}

GateOut gate(const std::string& name, const json::Value& args, bool approved = false) {
    // 白名单，不是黑名单：新增工具忘了登记，表现是"这个工具用不了"，
    // 而不是"它绕过了所有权限"。fail-closed 的全部意义就在这里。
    static const char* kKnown[] = {"read_file", "list_dir", "write_file", "edit_file",
                                   "grep",      "glob",     "fetch_url",  "web_search",
                                   "kb_search", "shell",    "calc",
                                    // 模型训练里见过的同义工具名。不登记的话
                                    // gate 会返回"未知工具"，模型连着试三四轮
                                    // 编出来的名字，最后干脆自己心算一堆乱码。
                                    "calculate", "calculator", "math", "bash", "terminal"};
    bool known = false;
    for (const char* k : kKnown) {
        if (name == k) {
            known = true;
            break;
        }
    }
    if (!known) return gate_deny("未知工具: " + name);

    const bool is_write = (name == "write_file" || name == "edit_file");
    const bool has_path = (name == "read_file" || name == "list_dir" || name == "write_file" ||
                           name == "edit_file" || name == "grep" || name == "glob");
    // 只有这三个工具"没有 path 就没法干活"。grep/glob/list_dir 的 path 可省
    // （默认工作区），shell 和联网工具根本不看 path —— 早先这里写成
    // "除了 grep/glob/list_dir 都要 path"，结果 shell 被自己的门禁挡了。
    const bool path_required = (name == "read_file" || name == "write_file" ||
                                name == "edit_file");

    // ---- 权限位
    if (is_write && !g.allow_write)
        return gate_deny("写文件权限已关闭（设置 → 权限 → 允许写文件）");
    if (name == "shell" && !g.allow_shell)
        return gate_deny("shell 工具未启用。在界面「设置 → 权限」里打开「允许执行命令」即可。");
    if ((name == "fetch_url" || name == "web_search") && !g.allow_net)
        return gate_deny("联网权限已关闭（设置 → 权限 → 允许联网）");

    // ---- 必填键
    //
    // 放在门禁里而不是各工具里：工具自己判的话，模型只会看到一句
    // "缺少 path 参数"，不知道该怎么办。这里统一说清楚缺哪个键。
    if (path_required && args.s("path").empty()) return gate_deny("缺少 path 参数");
    if (name == "shell" && args.s("command").empty()) return gate_deny("缺少 command 参数");

    // ---- 不可逆命令的硬闸
    //
    // 这不是沙箱，是最后一道闸。理由是：seatbelt 在本机套不上的时候（见
    // shell 分支的退出码 71 降级），写盘限制其实是**没生效**的，而 1.7B 级的
    // 模型迟早会吐出一条 `rm -rf ~`。
    //
    // 故意做得很窄：只挡"一旦执行就不可逆"的那几种，不做通用命令审计。
    // 通用审计要么拦不住（拼个变量就绕了），要么误伤正常命令，两头都不讨好。
    // 真要放开 sudo 这类，用户在设置里关掉沙箱就是在表达"我知道风险"。
    if (name == "shell") {
        std::string c = args.s("command");
        // 折叠空白：`rm   -rf   /` 和 `rm -rf /` 是同一件事
        std::string flat;
        flat.reserve(c.size());
        bool sp = false;
        for (char ch : c) {
            if (ch == ' ' || ch == '\t' || ch == '\n') {
                if (!sp) flat.push_back(' ');
                sp = true;
            } else {
                flat.push_back(ch);
                sp = false;
            }
        }
        static const char* kDanger[] = {
            "rm -rf /",   "rm -rf ~",  "rm -rf *",   "rm -fr /",   "rm -fr ~",
            "mkfs",       "dd if=",    "dd of=/dev", ":(){",       "> /dev/disk",
            "diskutil erase", "chmod -R 777 /",     "chown -R /",
        };
        for (const char* d : kDanger) {
            if (flat.find(d) != std::string::npos)
                return gate_deny(std::string("这条命令被拦下了（匹配到 \"") + d +
                                 "\"）。它会删掉或覆盖不可恢复的东西，"
                                 "换个范围明确的做法：删目录就写全路径，"
                                 "覆盖文件就用 write_file。");
        }
        // sudo 不硬拒 —— 用户让装个软件包是正常需求。但也不该静默执行，
        // 问一次，答案只对这一次有效。
        if (flat.find("sudo ") != std::string::npos) {
            if (approved) return gate_ok();
            return gate_ask("这条命令要用 sudo 提权，需要你确认。",
                            "sudo：" + args.s("command"));
        }
    }
    if (name == "grep" && args.s("pattern").empty()) return gate_deny("缺少 pattern 参数");
    if (name == "glob" && args.s("pattern").empty()) return gate_deny("缺少 pattern 参数");
    if (name == "fetch_url" && args.s("url").empty()) return gate_deny("缺少 url 参数");
    if (name == "web_search" && args.s("query").empty()) return gate_deny("缺少 query 参数");
    if (name == "edit_file" && args.s("old").empty()) return gate_deny("缺少 old 参数");

    // ---- 路径域
    if (has_path) {
        const std::string p = path_abs(args.s("path", g.workspace));
        if (p.find("..") != std::string::npos) return gate_deny("路径里不能有 ..");
        if (!path_allowed(p)) return gate_deny("路径不在允许范围内: " + p);
        if (is_write && !g.allow_write_outside) {
            bool ok = under_root(p, g.workspace);
            if (!ok) {
                const char* h = getenv("HOME");
                // ~/.dubhe 是应用自己的目录（配置、临时落地），写它不算越界
                if (h && under_root(p, std::string(h) + "/.dubhe")) ok = true;
            }
            if (!ok) {
                // 写到项目外：不是直接拒，而是问一次。
                //
                // 这是三态里 Ask 唯一的用处 —— 用户可能确实要写 /tmp 下的
                // 一个脚本，一刀切拒绝会逼他去设置里开全局开关，然后忘掉。
                // 问一次，答案只对这一次有效。
                if (approved) return gate_ok();
                return gate_ask(
                    "这次操作需要用户确认：要写到项目文件夹之外。",
                    "写到项目文件夹之外\n" + p);
            }
        }
    }
    return gate_ok();
}

ToolResult run_tool(const std::string& name, const json::Value& args, bool approved = false) {
    ToolResult r;
    const GateOut gt = gate(name, args, approved);
    if (gt.v == Verdict::Deny || gt.v == Verdict::Ask) {
        // Ask 走到这里说明调用方没做审批（比如 /api/tools/run 手测）。
        // 按拒绝处理 —— 绝不能在没人应答的情况下放行。
        r.denied = true;
        r.output = gt.v == Verdict::Ask ? ("需要确认后才能执行：" + gt.reason) : gt.reason;
        // note 是给卡片看的。output 是给模型看的，里面常带"换个范围明确的做法"
        // 这类指导语 —— 原样摆在卡片上，用户会觉得程序在对他讲黑话。
        r.meta.set("note", json::Value::make_str(gt.reason));
        return r;
    }
    // 9B（MiMo）实测会发明 calculate / calculator，0.5B 反而老老实实写 calc。
    // 与其纠正模型，不如在服务端认这些别名 —— 反正它们要的都是同一件事。
    if (name == "calc" || name == "calculate" || name == "calculator" || name == "math") {
        // 模型经常把参数名写错成 expression / query / input，都收下。
        // 它甚至会把整个 JSON 片段塞进来，这里也试着捞一下。
        std::string e = args.s("expr");
        if (e.empty()) e = args.s("expression");
        if (e.empty()) e = args.s("query");
        if (e.empty()) e = args.s("command");
        if (e.empty()) {
            r.output = "缺少 expr 参数：把要算的式子放在 expr 里，例如 expr=\"376*49\"";
            return r;
        }
        const dubhe_calc::CalcResult c = dubhe_calc::calc_eval(e);
        r.ok = c.ok;
        // output 是模型唯一能看到的东西。**只给结果，不给过程**——
        // 过程里带函数名、引号、括号，小模型会把它当成让它继续调用的指令。
        r.output = c.ok ? (e + " = " + c.text)
                        : ("计算失败：" + c.error + "。表达式: " + e);
        r.card = "calc";
        r.meta.set("kind", json::Value::make_str("calc"));
        r.meta.set("expr", json::Value::make_str(e));
        if (c.ok) r.meta.set("value", json::Value::make_str(c.text));
        return r;
    }
    if (name == "read_file") {
        // 相对路径按项目文件夹解析。以前这里直接用原始字符串交给 read_file()，
        // 于是 "a.txt" 会相对**服务进程的 cwd** 去找 —— 一旦用户把项目文件夹
        // 绑到别处，模型读到的就是另一个同名文件，而且不会报错。
        const std::string p = path_abs(args.s("path"));
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
        r.card = "read";
        r.meta.set("path", json::Value::make_str(p));
        r.meta.set("offset", json::Value::make_num(off));
        r.meta.set("lines", json::Value::make_num(taken));
        r.meta.set("total", json::Value::make_num(total));
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
        // 不再 shell 出去调 ls：一是没开 shell 权限时它就不该依赖 shell，
        // 二是 `ls -la "路径"` 里路径带双引号、反引号就能注入。
        std::string p = path_abs(args.s("path", g.workspace));
        DIR* d = opendir(p.c_str());
        if (!d) { r.output = "无法打开目录: " + p; return r; }
        std::vector<std::string> dirs, files;
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            struct stat st {};
            if (stat(path_join(p, n).c_str(), &st) != 0) continue;
            char line[1024];
            if (S_ISDIR(st.st_mode)) {
                std::snprintf(line, sizeof(line), "%s/", n.c_str());
                dirs.push_back(line);
            } else {
                std::snprintf(line, sizeof(line), "%s\t%lld B", n.c_str(), (long long)st.st_size);
                files.push_back(line);
            }
        }
        closedir(d);
        std::sort(dirs.begin(), dirs.end());
        std::sort(files.begin(), files.end());
        r.ok = true;
        r.output = p + "\n";
        for (auto& x : dirs) r.output += "  " + x + "\n";
        for (auto& x : files) r.output += "  " + x + "\n";
        if (dirs.empty() && files.empty()) r.output += "  （空目录）\n";
        r.card = "search";
        r.meta.set("shape", json::Value::make_str("paths"));
        r.meta.set("path", json::Value::make_str(p));
        r.meta.set("count", json::Value::make_num((double)(dirs.size() + files.size())));
        return r;
    }
    if (name == "write_file") {
        const std::string c = args.s("content");
        const std::string p = path_abs(args.s("path"));
        r.ok = write_file(p, c);
        r.output = r.ok ? ("已写入 " + std::to_string(c.size()) + " 字节到 " + p)
                        : ("写入失败: " + p);
        r.card = "diff";
        r.meta.set("path", json::Value::make_str(p));
        r.meta.set("is_new", json::Value::make_bool(true));
        r.meta.set("old", json::Value());              // 新文件：没有旧内容
        r.meta.set("new", json::Value::make_str(c.substr(0, 8000)));
        r.meta.set("bytes", json::Value::make_num((double)c.size()));
        return r;
    }
    if (name == "edit_file") {
        // 精确串替换。为什么不是"整文件重写"：小模型重写整文件时会把
        // 没看过的部分凭记忆补一遍，改一个函数顺手删掉另外三个。
        const std::string old_s = args.s("old");
        const std::string new_s = args.s("new");
        const std::string p = path_abs(args.s("path"));
        std::string body = read_file(p, 4 * 1024 * 1024);
        if (body.empty()) { r.output = "读不到文件: " + p; return r; }
        size_t at = body.find(old_s);
        if (at == std::string::npos) {
            r.output = "文件里找不到这段原文。请先用 read_file 确认当前内容（注意缩进和空格）。";
            return r;
        }
        const size_t second = body.find(old_s, at + old_s.size());
        if (second != std::string::npos) {
            r.output = "这段原文在文件里出现了不止一次，替换位置有歧义。"
                       "请多带几行上下文让它唯一。";
            return r;
        }
        body.replace(at, old_s.size(), new_s);
        r.ok = write_file(p, body);
        r.output = r.ok ? ("已替换 1 处，文件现在是 " + std::to_string(body.size()) + " 字节")
                        : ("写入失败: " + p);
        r.card = "diff";
        r.meta.set("path", json::Value::make_str(p));
        r.meta.set("is_new", json::Value::make_bool(false));
        r.meta.set("old", json::Value::make_str(old_s.substr(0, 4000)));
        r.meta.set("new", json::Value::make_str(new_s.substr(0, 4000)));
        return r;
    }
    if (name == "grep") {
        // 递归文本搜索。用 /usr/bin/grep 而不是自己写：它已经处理了二进制
        // 检测、编码、正则方言，自己写只会更慢更错。
        const std::string pat = args.s("pattern");
        std::string base = path_abs(args.s("path", g.workspace));
        int limit = (int)args.n("limit", 60);
        if (limit <= 0) limit = 60;
        std::string cmd = "grep -rnI --exclude-dir=.git --exclude-dir=node_modules "
                          "--exclude-dir=build --exclude-dir=.build "
                          "-m " + std::to_string(limit) + " -e " + shell_quote(pat) + " " +
                          shell_quote(base) + " 2>&1 | head -" + std::to_string(limit);
        ProcOut po = run_proc(cmd, g.workspace, 15000, 48 * 1024, false);
        r.ok = true;
        r.output = po.out.empty() ? ("没有匹配: " + pat) : po.out;
        int hits = 0;
        for (char ch : po.out) if (ch == '\n') hits++;
        r.card = "search";
        r.meta.set("shape", json::Value::make_str("matches"));
        r.meta.set("pattern", json::Value::make_str(pat));
        r.meta.set("path", json::Value::make_str(base));
        r.meta.set("total", json::Value::make_num(hits));
        r.meta.set("truncated", json::Value::make_bool(hits >= limit));
        return r;
    }
    if (name == "glob") {
        const std::string pat = args.s("pattern");
        std::string base = path_abs(args.s("path", g.workspace));
        // find -name 只支持 glob，不支持 **。这里把 ** 折成"任意深度"，
        // 也就是去掉路径前缀里的 **/，剩下的交给 -name。
        std::string name_pat = pat;
        const size_t sl = name_pat.find_last_of('/');
        if (sl != std::string::npos) name_pat = name_pat.substr(sl + 1);
        std::string cmd = "find " + shell_quote(base) +
                          " -maxdepth 6 -type f -name " + shell_quote(name_pat) +
                          " 2>/dev/null | head -200";
        ProcOut po = run_proc(cmd, g.workspace, 15000, 48 * 1024, false);
        r.ok = true;
        r.output = po.out.empty() ? ("没有匹配的文件: " + pat) : po.out;
        int hits = 0;
        for (char ch : po.out) if (ch == '\n') hits++;
        r.card = "search";
        r.meta.set("shape", json::Value::make_str("paths"));
        r.meta.set("pattern", json::Value::make_str(pat));
        r.meta.set("path", json::Value::make_str(base));
        r.meta.set("count", json::Value::make_num(hits));
        r.meta.set("truncated", json::Value::make_bool(hits >= 200));
        return r;
    }
    if (name == "kb_search") {
        // 知识库原来只做"自动注入"：用户提问时服务端偷偷检索一次塞进 system。
        // 模型自己**没有**查文档的手段 —— 它想问"ADR-0003 怎么说的"，只能靠
        // 注入那一把，注入没命中就只能编。这个工具把同一个检索器开给它自己用，
        // 区别是它可以换个词再试一次。
        const std::string q = args.s("query");
        if (q.empty()) { r.output = "缺少 query 参数"; return r; }
        int k = (int)args.n("k", 5);
        if (k < 1) k = 1;
        if (k > 20) k = 20;
        std::vector<KbHit> hits;
        {
            std::lock_guard<std::mutex> lk(g_kb_mu);
            kb_ensure_loaded();
            hits = kb_search(q, k);
        }
        if (hits.empty()) {
            r.ok = true;
            r.output = "知识库里没有匹配的片段。可以换个更具体的关键词再试。";
        } else {
            std::string s;
            for (size_t i = 0; i < hits.size(); i++) {
                s += "[" + std::to_string(i + 1) + "] " + path_base(hits[i].path) + "\n";
                s += hits[i].snippet + "\n\n";
            }
            r.ok = true;
            r.output = s;
        }
        r.card = "search";
        r.meta.set("shape", json::Value::make_str("kb"));
        r.meta.set("pattern", json::Value::make_str(q));
        r.meta.set("count", json::Value::make_num((double)hits.size()));
        return r;
    }
    if (name == "fetch_url") {
        const std::string url = args.s("url");
        if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
            r.output = "只支持 http/https";
            return r;
        }
        // 走系统 curl：本工程不依赖 libcurl，而自己实现 TLS 不现实。
        // -L 跟随跳转，-m 限时，--compressed 省流量。
        std::string cmd = "curl -sSL --compressed -m 20 -A 'Mozilla/5.0 (Dubhe)' " +
                          shell_quote(url);
        ProcOut po = run_proc(cmd, g.workspace, 25000, 512 * 1024, false);
        r.ok = po.exit_code == 0;
        r.output = html_to_text(po.out);
        if (r.output.empty()) r.output = "抓取失败或内容为空";
        r.card = "web";
        r.meta.set("kind", json::Value::make_str("fetch"));
        r.meta.set("url", json::Value::make_str(url));
        r.meta.set("bytes", json::Value::make_num((double)r.output.size()));
        return r;
    }
    if (name == "web_search") {
        const std::string asked = args.s("query");
        const std::string q = localize_query(asked);
        const std::vector<WebHit> hits = web_search_hits(q, (int)args.n("n", 6));
        r.ok = !hits.empty();
        if (!r.ok) {
            r.output = "搜索没有返回结果（可能被限流，或本机无法访问外网）";
        } else {
            r.output = hits_to_text(hits);

            // 顺手把前几条的正文抓回来。
            //
            // 为什么工具自己要做这件事：1.7B 这类模型拿到"标题 + 链接 + 摘要"
            // 之后，绝大多数情况直接把列表抄一遍当答案 —— 它不会想到"再调一次
            // fetch_url 把正文读进来"。实测原话是"帮我查一下今天一些新闻"，
            // 它调了两次 web_search，最后输出三个新闻网站的首页地址和简介。
            // 与其教它编排（教不会），不如一次给全：标题、链接，外加正文摘录。
            const int read = (int)args.n("read", 2);
            std::string extra;
            for (int i = 0; i < (int)hits.size() && i < read; i++) {
                if (hits[i].url.empty()) continue;
                std::string body = fetch_page_text(hits[i].url, 1200);
                if (body.empty()) continue;
                extra += "\n【" + hits[i].title + "】网页正文摘录：\n" + body + "\n";
            }
            if (!extra.empty()) {
                r.output += "\n" + extra +
                            "\n（以上摘录由搜索工具自动抓取。归纳要点即可，"
                            "需要细节再用 fetch_url 打开某一篇。）";
            }
        }
        r.card = "web";
        r.meta.set("kind", json::Value::make_str("search"));
        r.meta.set("query", json::Value::make_str(asked));
        if (q != asked) r.meta.set("localized", json::Value::make_str(q));
        r.meta.set("count", json::Value::make_num((double)hits.size()));
        // 结构化的一份给界面画可点的来源列表。只给文本的话，前端得按
        // "    https://…" 这种缩进约定去啃，格式一调界面就跟着坏。
        json::Value arr = json::Value::make_arr();
        for (const WebHit& h : hits) {
            json::Value o = json::Value::make_obj();
            o.set("t", json::Value::make_str(h.title));
            o.set("u", json::Value::make_str(h.url));
            o.set("s", json::Value::make_str(h.snippet.substr(0, 240)));
            arr.arr.push_back(o);
        }
        r.meta.set("results", arr);
        return r;
    }
    if (name == "shell" || name == "bash" || name == "terminal") {
        const std::string cmd = args.s("command");
        const std::string cwd = args.s("cwd", g.workspace);
        const bool sb = sandbox_active();   // 探过一次就缓存，不再每条命令 fork 两次
        ProcOut po = run_proc(cmd, cwd.empty() ? g.workspace : cwd,
                              g.shell_timeout_ms, 96 * 1024, sb);
        // 探测说套得上、真跑时却还是 71 —— 说明这个 profile 本身被拒（例如
        // workspace 路径里有沙箱解析不了的东西）。这时候命令没执行，退一步
        // 裸跑，并把"未隔离"记到 meta 里让界面显示。
        bool no_sandbox = !sb;
        if (sb && po.exit_code == 71 && po.out.find("sandbox_apply") != std::string::npos) {
            po = run_proc(cmd, cwd.empty() ? g.workspace : cwd,
                          g.shell_timeout_ms, 96 * 1024, false);
            no_sandbox = true;
        }
        r.ok = (po.exit_code == 0) && !po.timed_out;
        r.output = po.out;
        // 不再把这句提示塞进 output。output 是模型眼里的 stdout，前面挂一段
        // 中文告警会让它把告警当成命令输出的一部分（实测它会引用"当前环境
        // 不支持…"来回答"这条命令输出了什么"）。隔离状态走 meta，界面画徽章。
        if (po.timed_out) {
            r.output += "\n[超时] 命令超过 " + std::to_string(g.shell_timeout_ms / 1000) +
                        " 秒被终止。拆小一点，或者只跑必要的那一步。";
        }
        if (po.truncated) r.output += "\n[截断] 输出超过 96 KB，只保留了前面部分。";
        if (po.out.empty() && r.ok) r.output = "（命令执行成功，没有输出）";
        else if (!po.out.empty()) r.output += "\n[退出码 " + std::to_string(po.exit_code) + "]";
        r.card = "terminal";
        r.meta.set("command", json::Value::make_str(cmd));
        r.meta.set("cwd", json::Value::make_str(cwd.empty() ? g.workspace : cwd));
        r.meta.set("exit", json::Value::make_num(po.exit_code));
        r.meta.set("timed_out", json::Value::make_bool(po.timed_out));
        r.meta.set("truncated", json::Value::make_bool(po.truncated));
        r.meta.set("no_sandbox", json::Value::make_bool(no_sandbox));
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
        "\n\n以下事情**必须调用工具，不许猜、不许心算**：\n"
        "- 任何算术（加减乘除、百分比、幂、日期差、单位换算）→ calc\n"
        "- 当前时间、日期、星期几、系统信息 → shell\n"
        "- 可能记错的事实、新闻、价格、版本号 → web_search\n"
        "算错比慢一点糟得多。哪怕你觉得“这个我会算”，也必须调工具。\n\n"
        "需要读文件、列目录、写文件或执行命令时，调用工具。格式严格如下：\n";
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
        // 这里原来有"工具名必须严格从列表里选"这类提醒，实测 0.5B 把它
        // 理解成"写出工具名就行"，输出 calc("376*49") 这种伪函数调用，
        // 连 <tool_call> 都不吐了。名字纠正放在服务端做别名，别教模型。
        "一次只调用一个工具。可用工具：\n"
        "- calc(expr): 精确算术，支持 + - * / % ^ 和括号，以及 sqrt/abs/round/floor/ceil/pow/min/max。\n"
        "  例：expr=\"376*49\"。**所有算术都用它，不要自己算。**\n"
        "- read_file(path, offset, limit): 读取文本文件，默认给前 120 行。\n"
        "  文件很长时先读开头；需要后面的内容再用 offset=121 往下翻。\n"
        "- list_dir(path): 列出目录内容\n"
        "- grep(pattern, path): 在目录里递归搜索文本，pattern 是正则\n"
        "- glob(pattern, path): 按文件名找文件，例如 *.md\n"
        "- write_file(path, content): 写入文件（覆盖）\n"
        "- edit_file(path, old, new): 把文件里一段原文替换掉。old 必须唯一。\n"
        "- shell(command, cwd): 执行 shell 命令\n"
        "- kb_search(query, k): 在本地知识库（项目文档）里检索相关片段。\n"
        "  问“这个项目怎么设计的”这类问题时先用它，比 grep 更懂语义。\n"
        "- web_search(query, n, read): 联网搜索。返回标题、链接、摘要，"
        "并自动附上前 read 条（默认 2）的网页正文摘录 —— 直接据此归纳就行，"
        "不要只把标题和链接抄一遍。\n"
        "  查新闻/时事时把用户说的“今天”换成年月日再搜，"
        "例如 query=\"2026年11月18日 新闻\"。\n"
        "- fetch_url(url): 抓取网页正文\n"
        "不需要工具时直接回答。不要描述你打算怎么做，直接输出工具调用。"
        "不要自己写工具返回块（" + g_tool_open + "），那是系统给你的，不是你输出的。"
        // 实测 1.7B 会把"我知道工作目录是什么"讲成"我执行了 pwd，返回…"。
        // 工作目录确实在下面那句里给了它，所以它没编内容，只是把"知道"
        // 说成了"做过"。用户看到的是"它跑了命令"，而界面上一条工具卡都没有。
        "没有真的调用过工具时，不要说\"我已经执行了\"，只说你知道什么。"
        // 截图里工具返回 2026-11-03，模型最终回答写 2026-01-03 —— 数字
        // 就在上下文里，它还是自己编了一个。泛泛说"基于结果回答"不够，
        // 必须明确"逐字照抄、不许改写"，小模型才可能照做。
        "工具返回里的数字、日期、时间、路径必须**逐字照抄**到回答中，"
        "一个字符都不许改；不确定就原样引用，不要转述。";
    if (!g.workspace.empty()) {
        s += "\n当前工作目录是 " + g.workspace + "，相对路径都以它为根。";
    }
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

    // 工具名必须是个标识符。
    //
    // 不校验的后果比"调用失败"难看得多：模型写坏一个 <function= ，比如漏掉
    // 结尾的 '>'，那"名字"就会一路吃到下一个 '>' —— 实测抓到过一个 1850 字符的
    // "工具名"，内容是模型的整段自言自语。这条调用当然过不了白名单，于是界面
    // 上多出一张失败卡片，卡片标题就是那 1850 个字：用户看到的是一大段乱码，
    // 而且会误以为是程序崩了。
    // 工具名只可能是 snake_case（read_file / web_search / calc），见到别的
    // 字符就说明这段不是合法的调用，当普通文本处理。
    if (name->size() > 32) return false;
    for (char ch : *name) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' ||
                        ch == '-' || ch == '.';
        if (!ok) return false;
    }

    *args = json::Value::make_obj();
    size_t p = fe;
    while (true) {
        const size_t k = body.find("<parameter=", p);
        if (k == std::string::npos) break;
        const size_t ke = body.find('>', k);
        if (ke == std::string::npos) break;
        const std::string key = body.substr(k + 11, ke - k - 11);
        // 参数名同理：不是标识符就说明这一段本来就不是合法调用，跳过它，
        // 别把半句话当成参数名塞进 JSON（界面上会显示成一串乱码键）。
        if (key.empty() || key.size() > 32) break;
        bool key_ok = true;
        for (char ch : key) {
            if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                  (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')) { key_ok = false; break; }
        }
        if (!key_ok) break;
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
                std::string url, fmt;
                if (const json::Value* iu = part.get("image_url")) url = iu->s("url");
                if (const json::Value* vu = part.get("video_url")) {
                    url = vu->s("url");
                    if (url.empty()) url = vu->s("path");
                }
                if (const json::Value* ia = part.get("input_audio")) {
                    // 两种写法都得认：
                    //   data 是**完整 data URL**（前端就是发这个）
                    //   data 是**裸 base64**（OpenAI 官方的形状，format 另给）
                    // 早先这里只处理第二种，于是前端发来的音频一律报
                    // "缺少 input_audio 数据" —— 界面上录完音点发送，得到的是
                    // 一句 400，看起来像"语音功能没做"。
                    const std::string d = ia->s("data");
                    fmt = ia->s("format");
                    if (d.rfind("data:", 0) == 0) url = d;
                    else if (!d.empty())
                        url = "data:audio/" + (fmt.empty() ? std::string("wav") : fmt) +
                              ";base64," + d;
                    if (url.empty()) {
                        if (const json::Value* p = ia->get("path")) url = p->str;
                    }
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
// 上面的注释说"对话产品要的是答案，所以统一不想"——那是 2026-11-03 的决定。
// 现在做成可配：DUBHE_THINKING=1 时保留 <think> 开洞，让模型先推理再答。
// 对外行为不变（默认仍是关闭），只是给了难题一条出路。
static bool g_think_enabled = false;

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
                 g_think_style ? (g_think_enabled ? "有（已开启）" : "有（已关闭）") : "无");
}

// 历史里的思考块和工具调用原文，回灌前必须剥掉。
//
// 不剥会有连锁反应，而且一步比一步难看：
//   1. 模型上一轮写出来的推理被原样塞回 prompt，它看到"这就是我的输出风格"，
//      下一轮推理更长 —— 1.7B 这种小模型尤其收不住，几轮之后整条回复都是
//      自我复述，正事一个字没有。
//   2. 下面给历史助手轮补 <think></think> 的那段逻辑，碰到正文里已经含有
//      <think> 的内容，会拼成 "<think></think>推理…</think>" 这种畸形结构。
//      模型随后就学会了在正文里乱吐 </think> —— 界面上看到的那堆标签就是这么来的。
// 从"整段原始输出"里取出真正的答案。
//
// 提示词末尾已经替模型开了 <think>，所以输出形如
//   推理……</think>答案
// 流式通道有 feed_think 逐片分流，非流式没有，只能事后切一刀。
//
// 没有 </think> 时要分两种情况，不能一律当成"没想"：
//   · 撞了 max_tokens → 推理还没写完，这段是草稿，扔掉；
//   · 正常结束（EOS）  → 模型压根没进思考块，直接给的答案，原样保留。
std::string strip_think_block(const std::string& s, bool truncated, bool think_model) {
    const size_t k = s.rfind("</think>");
    if (k != std::string::npos) return s.substr(k + 8);
    if (think_model && truncated) return std::string();
    return s;
}

std::string scrub_history(const std::string& s) {
    std::string r = s;
    auto cut = [&r](const std::string& open, const std::string& close) {
        for (;;) {
            const size_t a = r.find(open);
            if (a == std::string::npos) break;
            const size_t b = r.find(close, a + open.size());
            if (b == std::string::npos) { r.erase(a); break; }   // 没闭合，连尾巴一起丢
            r.erase(a, b + close.size() - a);
        }
    };
    cut("<think>", "</think>");
    cut("<tool_call>", "</tool_call>");
    cut("<function=", "</function>");
    // 上面 cut 没兜住的孤立结束标签
    for (const char* t : {"</think>", "</tool_call>", "</function>"}) {
        for (size_t p = r.find(t); p != std::string::npos; p = r.find(t, p)) r.erase(p, strlen(t));
    }
    // 剥完常留下成串空行，压一压，别把 prompt 撑长
    std::string out;
    out.reserve(r.size());
    int blank = 0;
    for (char ch : r) {
        if (ch == '\n') {
            if (++blank > 2) continue;
        } else {
            blank = 0;
        }
        out.push_back(ch);
    }
    const size_t b = out.find_first_not_of(" \t\n\r");
    if (b == std::string::npos) return std::string();
    const size_t e = out.find_last_not_of(" \t\n\r");
    return out.substr(b, e - b + 1);
}

// ------------------------------------------------------------ 纯算术短路
//
// 为什么要在"进模型"之前拦一道：让 1.7B 自己做 887×7737，它会把数字读错、
// 把中间步骤写错，最后给出一个很自信的错误答案；让它调用 calc，它又经常
// 写不出合法的参数（实测 expr 会写成 "8*87*7*73*7*...?"，还会把数字拆成
// "8 8 7"）。而这类问题根本不需要模型 —— 表达式是确定的，答案也是确定的。
//
// 只认"整句就是一个算式"的情况。带解释、带上下文的问法一律交回模型：
// 抢答比答错更让人恼火。

static void str_replace_all(std::string* s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    for (size_t at = s->find(from); at != std::string::npos; at = s->find(from, at + to.size()))
        s->replace(at, from.size(), to);
}

static bool ends_with(const std::string& s, const std::string& t) {
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

static bool starts_with(const std::string& s, const std::string& t) {
    return s.size() >= t.size() && s.compare(0, t.size(), t) == 0;
}

// 取最后一条 user 的纯文本。多模态消息里 content 是数组，只拼其中的文本片段。
std::string last_user_text(const json::Value& messages) {
    std::string last;
    for (auto& m : messages.arr) {
        if (m.s("role") != "user") continue;
        const json::Value* c = m.get("content");
        if (!c) continue;
        if (c->type == json::Value::STR) {
            last = c->str;
        } else if (c->type == json::Value::ARR) {
            std::string t;
            for (auto& p : c->arr) {
                if (p.type == json::Value::OBJ && p.s("type") == "text") t += p.s("text");
            }
            if (!t.empty()) last = t;
        }
    }
    return last;
}

bool try_arith(const std::string& q, std::string* out) {
    std::string s = q;
    if (s.empty() || s.size() > 200) return false;

    // 全角符号 / 数学符号
    str_replace_all(&s, "×", "*");
    str_replace_all(&s, "✕", "*");
    str_replace_all(&s, "÷", "/");
    str_replace_all(&s, "（", "(");
    str_replace_all(&s, "）", ")");
    str_replace_all(&s, "＝", "=");
    // 中文动词（先长后短，"乘以" 必须先于 "乘" 替换）
    str_replace_all(&s, "乘以", "*");
    str_replace_all(&s, "除以", "/");
    str_replace_all(&s, "加上", "+");
    str_replace_all(&s, "减去", "-");
    str_replace_all(&s, "乘", "*");
    str_replace_all(&s, "除", "/");
    str_replace_all(&s, "加", "+");
    str_replace_all(&s, "减", "-");

    // 剥掉问句外壳。反复剥，因为 "请问 3+6 等于多少？" 是两层。
    //
    // 后面那几个是"元指令"（直接给结果/不要解释）。它们不影响这是一道算式，
    // 但会让 try_arith 的字符校验失败，于是整句被推给模型 —— 实测 9B 拿到
    // "887乘以7737等于多少？直接给结果，不要解释" 会想满 1500 token 还不给答案。
    const char* tails[] = {"等于多少", "等于几", "是多少", "结果是多少", "等于", "结果",
                           "多少", "=", "?", "？", "。", "！", "!", "，", ",",
                           "直接给结果", "直接回答", "直接说结果", "不要解释", "不用解释",
                           "只给结果", "只给答案", "只输出结果", "只输出答案"};
    const char* heads[] = {"请问", "帮我算一下", "帮我算算", "帮我算", "请计算", "计算一下",
                           "计算", "算一下", "算算"};
    for (bool changed = true; changed;) {
        changed = false;
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n')) {
            s.pop_back();
            changed = true;
        }
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n')) {
            s.erase(0, 1);
            changed = true;
        }
        // 必须按**最长**匹配剥。
        // 按数组顺序剥会出事："…等于多少？直接给结果" 里 "结果" 是后缀，
        // 先被剥掉就剩 "…直接给"，再也没法还原，整句被判定成"不是算式"。
        // 实测就是这么漏的 —— 明明是个纯算式却要走模型。
        {
            size_t best = 0;
            for (const char* t : tails) {
                const size_t n = strlen(t);
                if (n > best && ends_with(s, t)) best = n;
            }
            if (best) { s.erase(s.size() - best); changed = true; }
        }
        for (const char* h : heads) {
            if (starts_with(s, h)) { s.erase(0, strlen(h)); changed = true; }
        }
    }

    // 校验：只允许数字、运算符、括号、小数点。出现任何别的字符（汉字、字母、
    // 单位）就说明这不是"整句一个算式"，交回模型。
    bool has_digit = false, has_op = false;
    int digits = 0;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') { has_digit = true; digits++; continue; }
        if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '%' || ch == '^') {
            has_op = true;
            continue;
        }
        if (ch == '.' || ch == '(' || ch == ')' || ch == ' ') continue;
        return false;
    }
    if (!has_digit || !has_op || digits < 2) return false;

    const dubhe_calc::CalcResult r = dubhe_calc::calc_eval(s);
    if (!r.ok) return false;
    *out = s + " = " + r.text;
    return true;
}

// 现在几点 / 今天星期几。这类问题同样是确定性的 —— 机器上有表，没必要问模型。
//
// 只认短句：长句里出现"今天"往往是在讲故事（"今天我去了趟超市"），
// 抢答会答非所问。所以限制长度 + 必须命中"星期几/几号/几点"这种明确的问法。
bool try_clock(const std::string& q, std::string* out) {
    std::string s;
    for (char ch : q) {
        if (ch == ' ' || ch == '\t' || ch == '\n') continue;
        s.push_back(ch);
    }
    // 收尾的问号/句号。中文标点是三字节，不能写成 char 字面量比较。
    static const char* kTail[] = {"?", "？", "。", "!", "！", "~", " "};
    for (bool changed = true; changed && !s.empty();) {
        changed = false;
        for (const char* t : kTail) {
            const size_t n = strlen(t);
            if (s.size() >= n && s.compare(s.size() - n, n, t) == 0) {
                s.resize(s.size() - n);
                changed = true;
            }
        }
    }
    if (s.empty() || s.size() > 24) return false;

    static const char* kDate[] = {"星期几", "周几", "几号", "今天的日期", "今天是什么日子",
                                  "今天几", "现在几号", "日期"};
    static const char* kTime[] = {"现在几点", "当前时间", "现在的时间", "几点钟", "现在时间"};
    bool date = false, time = false;
    for (const char* k : kDate) if (s.find(k) != std::string::npos) { date = true; break; }
    for (const char* k : kTime) if (s.find(k) != std::string::npos) { time = true; break; }
    if (!date && !time) return false;

    std::time_t t = std::time(nullptr);
    std::tm tm {};
    localtime_r(&t, &tm);
    static const char* wd[] = {"日", "一", "二", "三", "四", "五", "六"};
    char buf[96];
    if (date && !time) {
        std::snprintf(buf, sizeof(buf), "%d年%d月%d日 星期%s", tm.tm_year + 1900,
                      tm.tm_mon + 1, tm.tm_mday, wd[tm.tm_wday]);
        *out = std::string("今天是 ") + buf;
    } else if (time && !date) {
        std::snprintf(buf, sizeof(buf), "%02d:%02d", tm.tm_hour, tm.tm_min);
        *out = std::string("现在是 ") + buf;
    } else {
        std::snprintf(buf, sizeof(buf), "%d年%d月%d日 星期%s %02d:%02d", tm.tm_year + 1900,
                      tm.tm_mon + 1, tm.tm_mday, wd[tm.tm_wday], tm.tm_hour, tm.tm_min);
        *out = std::string("现在是 ") + buf;
    }
    return true;
}

// 一条消息里塞了几个互不相干的小问题（"今天是星期几，887乘以7737等于多少"）
// —— 逐个看能不能确定性回答；**只要有一段答不了，整条交回模型**。
// 半抢答比不答更糟：用户会以为后面那半句被吞了。
bool try_quick_answer(const std::string& q, std::string* out) {
    std::vector<std::string> segs;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) segs.push_back(cur);
        cur.clear();
    };
    for (size_t i = 0; i < q.size();) {
        // 中文标点按 UTF-8 三字节匹配，英文标点单字节
        if (q.compare(i, 3, "，") == 0 || q.compare(i, 3, "；") == 0 ||
            q.compare(i, 3, "、") == 0) {
            flush();
            i += 3;
            continue;
        }
        if (q[i] == ',' || q[i] == ';' || q[i] == '\n') {
            flush();
            i++;
            continue;
        }
        const bool conn = q.compare(i, 6, "还有") == 0 || q.compare(i, 6, "以及") == 0;
        if (conn) { flush(); i += 6; continue; }
        cur.push_back(q[i]);
        i++;
    }
    flush();
    if (segs.empty() || segs.size() > 3) return false;

    std::string joined;
    for (auto& s : segs) {
        // 纯元指令的小段（"不要解释""直接给结果"）直接丢掉，不要它拖累整句。
        // 实测 "887乘以7737等于多少？直接给结果，不要解释" 被逗号切成两段后，
        // 第二段答不了，整句就退回模型 —— 明明是个算式。
        {
            static const char* kMeta[] = {"直接给结果", "直接回答", "不要解释", "不用解释",
                                          "只给结果", "只给答案", "只输出", "请", "谢谢"};
            bool meta = false, has_digit = false;
            for (const char* mk : kMeta) if (s.find(mk) != std::string::npos) { meta = true; break; }
            for (char c : s) if (c >= '0' && c <= '9') { has_digit = true; break; }
            if (meta && !has_digit) continue;
        }
        std::string a;
        if (try_arith(s, &a) || try_clock(s, &a)) {
            // 用中文逗号接起来：markdown 会把单个换行当成同一段，
            // 拼出来就成了"星期三887*7737"，两句话黏在一起。
            if (!joined.empty()) joined += "，";
            joined += a;
            continue;
        }
        return false;
    }
    if (joined.empty()) return false;
    *out = joined;
    return true;
}

// 当前时间，给模型当"现在"用。
//
// 模型的知识停在训练那一刻，它不知道今天几号。于是"今天的新闻""最近发布了
// 什么"这类问题，它只能猜 —— 实测它会搜"今天新闻"，搜索引擎回一堆门户首页，
// 它就把首页标题抄一遍当答案。
std::string now_stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm {};
    localtime_r(&t, &tm);
    static const char* wd[] = {"日", "一", "二", "三", "四", "五", "六"};
    // 用中文格式而不是 2026-11-18：实测 1.7B 会把连字符日期读错 ——
    // 给它 "2026-11-18"，它转手搜成"2026年1月8日"，搜回来的全是 1 月的新闻。
    // 中文年月日它不用做格式换算，直接抄就行。
    char b[80];
    std::snprintf(b, sizeof(b), "%d年%d月%d日 %02d:%02d 星期%s", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, wd[tm.tm_wday]);
    return b;
}

// 本次请求是否跳过思考通道。
//
// 用 thread_local 而不是参数一路传：build_tokens 已经被四个地方调用，
// 每加一个开关就要改四遍，漏一处就是"某个路径行为不一致"这种最难查的 bug。
// 每个请求在自己的线程上跑，进 handler 时设置、结束自然回收。
static thread_local bool t_no_think = false;

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
    // 时间戳只贴在**最后一条** user 上。
    //
    // 放在 system 里最自然，但 system 是整段 prompt 的开头：它一变，前缀缓存
    // 全部作废，多轮对话每轮都要重新 prefill 整段历史（实测 7 秒）。
    // 贴在最后一条问题前面，前面一个字节都不动，缓存照常命中。
    size_t last_user = std::string::npos;
    for (size_t i = 0; i < messages.arr.size(); i++)
        if (messages.arr[i].s("role") == "user") last_user = i;

    for (size_t mi = 0; mi < messages.arr.size(); mi++) {
        auto& m = messages.arr[mi];
        std::string role = m.s("role");
        std::string content = m.s("content");
        if (role == "system") {
            has_system = true;
            if (with_tools) content += tool_prompt();
            p += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == "user") {
            // 让模型知道"今天"是哪天：新闻、版本、最近发生了什么都要用到。
            if (mi == last_user) content = "（当前时间：" + now_stamp() + "）\n" + content;
            p += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            content = scrub_history(content);
            if (content.empty()) continue;   // 整轮都是工具调用/推理，没什么可回灌的
            // 历史里的助手轮统一补一个**空**思考块：历史不该回灌推理（那是
            // 上一轮的草稿，越长越带偏），但格式必须完整 —— 少了它，模型
            // 看到自己上一轮"没有思考块"，会以为格式错了，接着把推理写出来。
            if (g_think_style)
                content = "<think></think>" + content;
            p += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        } else if (role == "tool" || role == "tool_denied") {
            // 光把结果摆上去，小模型会以为"我还没调用"，于是再要一次同样的工具，
            // 循环永远不收敛。明确说一句"已经执行完了，直接用结果回答"，
            // 这一句能把大部分原地打转掐掉。
            //
            // 但被拒绝的那次**没有执行**，不能这么说。原来说了，模型就
            // 顺着"已执行"编出"已写入成功" —— 用户看到一条成功消息，
            // 磁盘上什么都没有，这比报错严重得多。
            if (role == "tool_denied") {
                p += "<|im_start|>user\n" + g_tool_open + content + g_tool_close +
                     "\n这次工具没有执行，用户拒绝了。不要重试同一个调用，"
                     "也不要假装它成功了。直接告诉用户「这一步被拒绝、没有执行」，"
                     "然后问他想怎么继续。<|im_end|>\n";
            } else {
                p += "<|im_start|>user\n" + g_tool_open + content + g_tool_close +
                     "\n工具已经执行完毕，上面就是返回结果。"
                     "请直接基于这个结果回答，不要再调用同一个工具。"
                     "回答里的数字/日期/时间必须与上面的返回值**逐字一致**，"
                     "不许改写、不许凭记忆另写一个。<|im_end|>\n";
            }
        }
    }
    if (!has_system && with_tools) {
        p = std::string("<|im_start|>system\n") + tool_prompt() + "<|im_end|>\n" + p;
    }
    p += "<|im_start|>assistant\n";
    // 思考通道**永远打开**，不再取决于用户的开关。
    //
    // 之前的做法是"关掉思考"就塞一个空的 <think></think>，指望模型直接作答。
    // 实测 LFM3-1.7B 根本不照办：它照样要想，只是把整段推理写进正文 ——
    // 用户看到的就是几屏英文自言自语加最后一句答案，比开着思考还难读。
    //
    // 既然它无论如何都要想，那就让它在自己的通道里想：同样的 token 数、
    // 同样的速度，但推理会被 </think> 收走，正文干净。界面上那个开关
    // 改成"要不要把这段推理显示出来"，而不是"要不要想"。
    if (g_think_style && !t_no_think) p += "<think>";
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
                    // 先 ensure 再判空。原来的顺序反了：g_kb_docs 在进程刚起来
                    // 时是空的，而"判空"挡在 ensure 前面 —— 于是第一次对话
                    // 一定注入不了；更糟的是只要没人点开"知识库"那一页，
                    // g_kb_docs 就永远是空的，RAG 从头到尾一次都不生效。
                    // 实测：重启后直接问，prompt 里没有知识库片段；先访问一次
                    // /api/kb 再问，片段就出现了。
                    kb_ensure_loaded();
                    if (!g_kb_docs.empty()) ctx = kb_context_for(last_user, 3);
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
    // 末尾没凑齐的半个标签（"<thi"、"</thin"）。引擎按 token 吐字节，标签
    // 边界和 token 边界毫无关系，所以必须跨片拼接 —— 见 feed_think 的注释。
    std::string think_scan;
    // 用户这次要了"深度思考"：推理过程不当垃圾丢掉，而是单独推给界面。
    bool think_emit = false;
    // 这一轮真正给用户看到了多少字节。用来区分"模型什么都没说"和"只是我们
    // 把内容藏起来了"——前者要交代一句，后者不该再补废话。
    size_t visible = 0;
    // 上一片末尾没凑齐的多字节字符。引擎是按 token 吐字节的，一个汉字
    // （3 字节）经常被切在两个 token 之间；半截字节进了 JSON，浏览器那边
    // 就只能解成 U+FFFD —— 中文回复里随机冒出"均���的"这种方块。
    std::string tail;
    // 这一轮真正生成的 token 序列，用来给前缀缓存接尾巴
    std::vector<int32_t> gtoks;
};

// 引擎每吐一个 token 问一次"还要不要继续"。
//
// 客户端关页面 / 按停止之后，继续生成纯属浪费：它占着 g.mu（整条请求的
// 串行锁），后面的请求全排队。实测一次 90 秒的审批等待被一个已经关掉的
// 标签页占着，新请求 40 秒收不到一个字节 —— 用户看到的就是"卡死"。
//
// 注意 ud 是 dubhe_generate 里那个**共用的** void*，也就是 StreamCtx*，
// 不是 http::Conn*。当初直接把它当 Conn* 用，读到的 fd 是垃圾值，
// peer_gone() 一律返回"断了"，于是每个请求都立刻 finish_reason=stop、
// 0 个 token —— 现象是"模型突然一个字都不说了"，报错日志一条都没有。
static int conn_gone_cb(void* ud) {
    auto* x = (StreamCtx*)ud;
    return (x && x->c && x->c->peer_gone()) ? 1 : 0;
}

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
    x->visible += delta.size();
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

// 思考过程的增量推送。只有用户显式开了"深度思考"才会走到这里。
void emit_think_delta(StreamCtx* x, const std::string& s) {
    if (s.empty()) return;
    x->c->write_chunk("data: {\"dubhe\":{\"type\":\"think\",\"text\":" + json_escape(s) + "}}\n\n");
}

// 把一片 token 拆成"可见正文"和"思考内容"，结果写进 out。
//
// 为什么不能拿整片去和 "<think>" 比较：BPE 分词根本不管标签边界，模型吐出来
// 的常常是 "</think><tool_call>"、"让我们调用: </think>" 这样一长片。按整片
// 比较的话标签一条都匹配不上，推理过程和 </think> 就原样流进气泡 —— 用户
// 看到的是一大段自我复述外加一堆看不懂的标签，这就是"回复乱七八糟"的主因。
//
// 所以逐字符扫，末尾几个字符先扣住，等下一片补全再判断。每片最多扣 7 字节。
void feed_think(StreamCtx* x, const std::string& piece, std::string* out) {
    const std::string OPEN = "<think>", CLOSE = "</think>";
    std::string s = x->think_scan + piece;
    x->think_scan.clear();

    size_t i = 0;
    while (i < s.size()) {
        const std::string& tag = x->in_think ? CLOSE : OPEN;
        const size_t k = s.find(tag, i);
        if (k == std::string::npos) {
            // 尾部可能是被切断的半个标签，扣住等下一片。
            const size_t n = s.size() - i;
            size_t keep = 0;
            for (size_t L = std::min(n, tag.size() - 1); L > 0; --L) {
                if (s.compare(s.size() - L, L, tag, 0, L) == 0) { keep = L; break; }
            }
            if (x->in_think) {
                x->think_buf.append(s, i, n - keep);
                if (x->think_emit) emit_think_delta(x, s.substr(i, n - keep));
            } else {
                out->append(s, i, n - keep);
            }
            x->think_scan.assign(s, s.size() - keep, keep);
            return;
        }
        if (x->in_think) {
            x->think_buf.append(s, i, k - i);
            if (x->think_emit) emit_think_delta(x, s.substr(i, k - i));
            x->in_think = false;
        } else {
            out->append(s, i, k - i);
            x->think_buf.clear();
            x->in_think = true;
        }
        i = k + tag.size();
    }
}

// 前 16 个字符足以判断"这是不是一段工具调用"
void on_piece(void* ud, int32_t tok, const char* piece) {
    auto* x = (StreamCtx*)ud;
    if (!piece) return;
    x->acc->append(piece);
    x->gtoks.push_back(tok);

    // 思考块：拆出来，正文继续往下走
    std::string vis;
    feed_think(x, std::string(piece), &vis);
    if (vis.empty()) return;

    if (x->suppress) return;

    // 先和上一片留下的半截字符拼起来，再把末尾凑不齐的部分留住，
    // 剩下的才是能安全写进 JSON 的。
    std::string s = x->tail + vis;
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
    // 思考块没闭合（多半是被 max_tokens 截断在推理中间）。
    //
    // 原来这里的做法是"原样吐出来，总比空白强"。但那等于把模型最没价值的
    // 一段自言自语塞进气泡 —— 实测 1.7B 模型很容易一路推理到 max_tokens，
    // 用户看到的就是几屏"也许用户想问…"的复述，误以为程序坏了。
    // 现在改成：推理只在用户开了深度思考时才有资格露脸，否则丢掉，
    // 但这一轮如果什么都没剩下，得明确说一句为什么。
    if (x->in_think) {
        x->in_think = false;
        const bool had = !x->think_buf.empty();
        x->think_buf.clear();
        // 不管推理给不给用户看，只要这一轮一个字答案都没剩下，就得交代
        // 一句为什么。否则界面上要么是空气泡，要么是一个思考块下面空着 ——
        // 两种都像"程序坏了"。
        if (had && x->visible == 0 && !x->suppress) {
            x->c->write_chunk(
                "data: {\"dubhe\":{\"type\":\"note\",\"message\":"
                "\"模型把全部输出预算都用在了内部推理上，没有留下答案。"
                "可以把最大生成长度调大，或者换个更明确的问法。\"}}\n\n");
        }
    }
    // 末尾那个"可能是半个标签"的残留：既然流已经结束，就不是标签，是正文。
    if (!x->think_scan.empty() && !x->suppress && !x->in_think) {
        emit_delta(x, x->think_scan);
        x->think_scan.clear();
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
    bool mlx = false;     // 苹果 MLX 格式（目录 + safetensors），本引擎跑不了
};

// 一个目录是不是 MLX 模型：看有没有 *.safetensors。
// MLX 的模型是**目录**，不是单个文件，所以按文件扫的代码永远看不到它。
bool has_safetensors(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    bool found = false;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        // ".safetensors" 是 12 个字符（点 + 11 个字母）。写成 11 会永远匹配
        // 不上，表现就是"MLX 模型一个都看不见" —— 而代码看着完全正常。
        if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) {
            found = true;
            break;
        }
    }
    closedir(d);
    return found;
}

// 目录里权重文件的总字节。只统计顶层，不递归 —— MLX 的分片就在顶层。
long long weights_bytes(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return 0;
    long long total = 0;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        const bool w = (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) ||
                       (n.size() > 4 && n.compare(n.size() - 4, 4, ".bin") == 0);
        if (!w) continue;
        struct stat st {};
        if (stat(path_join(dir, n).c_str(), &st) == 0) total += (long long)st.st_size;
    }
    closedir(d);
    return total;
}

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
            // MLX 模型是一个目录。这里把它当成一条清单项收下，然后不再往里翻 ——
            // 里面的 safetensors 分片不是"可以点开加载的东西"。
            // 为什么要列出来而不是继续装作没有：用户把 MLX 模型放进模型目录，
            // 打开清单找不到它，第一反应是"这软件不支持苹果格式"，而真实情况
            // 是"不支持，但至少该让你看见它在哪、为什么用不了"。
            if (has_safetensors(full)) {
                ModelEntry m;
                m.path = full;
                m.name = n;
                m.bytes = weights_bytes(full);
                m.mtime = (long long)st.st_mtime;
                m.mlx = true;
                out->push_back(std::move(m));
                continue;
            }
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
//
// FP16 / FP32 也要排在 F16 / F32 前面：HuggingFace 官方仓库的文件名是
// "qwen2.5-0.5b-instruct-fp16.gguf"，而 "FP16" 里并不含连续的 "F16"
// （是 F、P、1、6），只认 F16 的话这份文件会被判成"未知格式"。
std::string guess_quant(const std::string& n) {
    std::string u = n;
    for (auto& c : u) c = (char)std::toupper((unsigned char)c);
    static const char* kQ[] = {"IQ4_XS", "IQ4_NL", "IQ3_M",  "IQ2_XS", "IQ2_M", "Q8_K",
                               "Q8_0",   "Q6_K",   "Q5_K_M", "Q5_K_S", "Q5_1",  "Q5_0",
                               "Q4_K_M", "Q4_K_S", "Q4_1",   "Q4_0",   "Q3_K_L", "Q3_K_M",
                               "Q3_K_S", "Q2_K",   "BF16",   "FP16",   "FP32",   "F16",
                               "F32"};
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
    add("calc", "精确计算四则运算、幂、括号和常用函数（expr），算术一律用它", "expr", true);
    add("read_file", "读取文本文件，默认前 120 行，可用 offset/limit 分页", "path, offset, limit", true);
    add("list_dir", "列出目录内容", "path", true);
    add("grep", "在目录里递归搜索文本（正则）", "pattern, path, limit", true);
    add("glob", "按文件名通配符找文件", "pattern, path", true);
    add("write_file", "写入文件（覆盖）", "path, content", g.allow_write);
    add("edit_file", "精确替换文件里的一段原文（只改一处，要求唯一）", "path, old, new", g.allow_write);
    add("shell", "执行 shell 命令（有超时与写盘沙箱）", "command, cwd", g.allow_shell);
    add("web_search", "联网搜索；会自动抓取前几条正文摘录", "query, n, read", g.allow_net);
    add("kb_search", "在本地知识库（项目文档）里检索相关片段", "query, k", true);
    add("fetch_url", "抓取网页并转成纯文本", "url", g.allow_net);
    return arr;
}


}  // namespace

// 上下文压缩用的累积回调。放在这里而不是 lambda：dubhe_generate 收的是
// 裸函数指针，带捕获的 lambda 转不过去。
struct SumCtx {
    std::string out;
};
static void sum_piece_cb(void* ud, int32_t tok, const char* piece) {
    auto* x = (SumCtx*)ud;
    if (piece && *piece) x->out += piece;
}

// ---------------------------------------------------------------- handlers
int main(int argc, char** argv) {
    // SIGPIPE 必须忽略。
    //
    // 这是本机 SSE 服务的头号杀手：用户切到别的对话、点停止、或者刷新页面，
    // 前端会中止 fetch，socket 那头就没了；服务端还在按 token 往这条已经
    // 关闭的连接写数据，内核直接给进程一个 SIGPIPE —— 整个引擎当场死掉，
    // 而日志里只有"退出码 13（被信号终止）"这一行，看起来像玄学崩溃。
    // 实测就是这样：切一次对话 = 崩一次。
    // 忽略之后 send() 会返回 -1/EPIPE，write_all 里已经把它标成 broken_，
    // 连接自己回收，进程继续服务其它请求。
    ::signal(SIGPIPE, SIG_IGN);

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
        else if (a == "--no-shell") g.allow_shell = false;
        else if (a == "--no-net") g.allow_net = false;
        else if (a == "--no-sandbox") g.sandbox = false;
        else if (a == "--no-write") g.allow_write = false;
        else if (a == "--model-dir") g.model_dir = next();
        else if (a == "--no-gpu") no_gpu = true;
        else if (a == "--thinking") g_think_enabled = true;
        else if (a == "--no-arith") g.no_arith_fastpath = 1;
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
                "             [--mmproj <mmproj.gguf>] [--tmp-dir /tmp] [--thinking]\n"
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
    // DUBHE_THINKING=1 让支持思考块的模型先推理再答；默认仍关闭
    if (const char* th = std::getenv("DUBHE_THINKING")) g_think_enabled = std::string(th) == "1";
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

        // 静态资源加版本戳（用文件 mtime 当版本号）。
        //
        // 为什么非做不可：桌面壳子加载的是 http://127.0.0.1:<port>/，端口不变、
        // URL 不变。只要 WebKit 的磁盘缓存里还留着上一次的 app.js，它就会直接
        // 用旧的 —— 表现是"代码改了、界面没变"，而且 index.html 和 app.js
        // 还可能一个新一个旧（我在这版上就踩了：按钮出现了但没绑定任何逻辑，
        // 旧的标签泄漏 bug 也照旧，从画面上完全看不出是缓存问题）。
        // 加上 mtime 之后每次构建都是新 URL，缓存自然失效，也不用手工维护版本号。
        if (p == "/index.html") {
            auto stamp = [&](const std::string& attr, const std::string& file) {
                struct stat st {};
                const long long v =
                    (::stat((root + file).c_str(), &st) == 0) ? (long long)st.st_mtime : 0;
                const std::string from = attr + file + "\"";
                const std::string to = attr + file + "?v=" + std::to_string(v) + "\"";
                for (size_t at = body.find(from); at != std::string::npos;
                     at = body.find(from, at + to.size())) {
                    body.replace(at, from.size(), to);
                }
            };
            stamp("href=\"", "/app.css");
            stamp("src=\"", "/app.js");
            // 版本号也注进去，别让前端再手工维护一份（两份迟早对不上，
            // 而"到底跑的是哪一版"正是排查时第一个要确认的事）。
            const std::string ver = std::to_string(DUBHE_VERSION_MAJOR) + "." +
                                    std::to_string(DUBHE_VERSION_MINOR) + "." +
                                    std::to_string(DUBHE_VERSION_PATCH);
            const std::string head = "<head>";
            const size_t at = body.find(head);
            if (at != std::string::npos) {
                body.insert(at + head.size(),
                            "\n<meta name=\"dubhe-version\" content=\"" + ver + "\">");
            }
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
        o.set("write", json::Value::make_bool(g.allow_write));
        o.set("net", json::Value::make_bool(g.allow_net));
        o.set("sandbox", json::Value::make_bool(sandbox_active()));
        // 配置值单列一个字段：sandbox=false 且 sandbox_cfg=true，说明用户开了
        // 但本机套不上（外层沙箱/CI 容器），界面据此给一句解释而不是静默失效。
        o.set("sandbox_cfg", json::Value::make_bool(g.sandbox));
        o.set("write_outside", json::Value::make_bool(g.allow_write_outside));
        o.set("shell_timeout_s", json::Value::make_num(g.shell_timeout_ms / 1000.0));
        o.set("tool_rounds", json::Value::make_num(g.max_tool_rounds));
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
                // 这台机器装得下的上限。前端拿它去画"上下文用量条"的满格，
                // 拿 n_ctx 当"这次能用多少"，两者不再混为一谈。
                m.set("n_ctx_ceiling", json::Value::make_num(mi.n_ctx_ceiling));
                m.set("n_layer", json::Value::make_num(mi.n_layer));
                m.set("n_embd", json::Value::make_num(mi.n_embd));
                m.set("n_vocab", json::Value::make_num(mi.n_vocab));
                m.set("chatml", json::Value::make_bool(mi.has_chat_template != 0));
                m.set("gpu", json::Value::make_bool(mi.is_gpu != 0));
                // 这个模型的对话模板里到底有没有思考通道。没有的话界面上
                // 那个「深度思考」按钮要灰掉 —— 放着能点却毫无反应，
                // 比直接告诉用户"这个模型不支持"更让人困惑。
                m.set("think", json::Value::make_bool(g_think_style != 0));
                o.set("info", m);
            }
        }
        c.send(200, "application/json", json::dump(o));
    });

    // ---- 模型清单：UI 的「模型」视图靠它列出本地所有 gguf。
    // 只 stat 文件、不读 GGUF 头 —— 一个目录几十个模型时读头要好几百毫秒，
    // 而名字里已经带了量化和参数量。真正加载时才读头。
    // ---------------------------------------------------------------- 模型商店
    //
    // 只列引擎真能加载的架构（llama 家族 / lfm2 / lfm3 / qwen35），并且每个
    // 链接的 content-length 都验证过。清单里出现一个下下来跑不起来的模型，
    // 比商店空着更伤信任 —— 用户会花二十分钟下载，然后得到一个"加载失败"。
    // 设计取舍见 docs/adr/ADR-0006-模型商店.md。
    struct StoreEntry {
        const char* id;
        const char* name;
        const char* params;
        const char* quant;
        long long bytes;
        const char* file;
        const char* url;
        const char* note;
    };
    static const StoreEntry kStore[] = {
        {"qwen35-9b-q4", "Qwen3.5-9B Instruct", "9B", "Q4_K_M", 5680522464LL,
         "Qwen3.5-9B-Q4_K_M.gguf",
         "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
         "最新一代，引擎有专属内核（Gated DeltaNet）；24GB 上约 10–12 tok/s"},
        {"qwen35-4b-q4", "Qwen3.5-4B Instruct", "4B", "Q4_K_M", 2740937888LL,
         "Qwen3.5-4B-Q4_K_M.gguf",
         "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
         "质量与速度的平衡点，比 1.7B 明显聪明，速度还有 20 tok/s 上下"},
        {"lfm25-26b-q4", "LFM2.5-2.6B", "2.6B", "Q4_K_M", 1674455040LL,
         "LFM2.5-2.6B-Q4_K_M.gguf",
         "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf",
         "LiquidAI 新一代端侧模型，多语言好、延迟低"},
        {"lfm25-12b-q4", "LFM2.5-1.2B Instruct", "1.2B", "Q4_K_M", 730895168LL,
         "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
         "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/"
         "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
         "最小最快，适合分类、打标、翻译这类高频杂活"},
        // 千问 3.8 目前只有 27B 一个尺寸（另两个是 MoE 巨兽）。架构名仍然是
        // qwen35，引擎能加载；但 16.5 GB 塞进 24GB 机器后剩不下多少给 KV，
        // 上下文会很短、速度掉到个位数。列出来是为了有据可查，不是推荐。
        {"qwen38-27b-q4", "Qwen3.8-27B（大，24GB 吃紧）", "27B", "UD-Q4_K_M",
         16564259744LL, "Qwen3.8-27B-UD-Q4_K_M.gguf",
         "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf",
         "最新旗舰，但 16.5 GB 在 24GB 机器上只剩很少上下文，约 4–5 tok/s；"
         "想试可以，不建议日常用"},
    };

    struct DlJob {
        std::string id, file;
        long long total = 0;
        std::string state = "running";   // running / done / error / cancelled
        std::string error;
        pid_t pid = 0;                   // curl 的进程号，取消时要 kill 它
        std::thread th;
    };
    auto dl_mu = std::make_shared<std::mutex>();
    auto dl_jobs = std::make_shared<std::map<std::string, std::shared_ptr<DlJob>>>();

    // 下载源。国内直连 huggingface.co 常常只有几百 KB/s，一个 4.7GB 的模型
    // 要下两个小时。换成 hf-mirror.com 这类镜像能快一个数量级。
    // 存在 ~/.dubhe/store.json 里，重启后仍然生效。
    auto mirror = std::make_shared<std::string>();
    {
        const std::string cfg = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.dubhe/store.json";
        FILE* f = std::fopen(cfg.c_str(), "rb");
        if (f) {
            std::string s;
            char buf[512];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
            std::fclose(f);
            json::Value v;
            if (json::parse(s, &v)) *mirror = v.s("mirror");
        }
    }
    auto save_mirror = [mirror]() {
        const std::string cfg = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.dubhe/store.json";
        json::Value v = json::Value::make_obj();
        v.set("mirror", json::Value::make_str(*mirror));
        FILE* f = std::fopen(cfg.c_str(), "wb");
        if (f) { std::fputs(json::dump(v).c_str(), f); std::fclose(f); }
    };


    srv.get("/api/store", [dl_mu, dl_jobs, mirror](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value root = json::Value::make_obj();
        root.set("dir", json::Value::make_str(g.model_dir));
        root.set("mirror", json::Value::make_str(*mirror));
        json::Value arr = json::Value::make_arr();
        for (const auto& e : kStore) {
            json::Value o = json::Value::make_obj();
            const std::string target = g.model_dir + "/" + e.file;
            o.set("id", json::Value::make_str(e.id));
            o.set("name", json::Value::make_str(e.name));
            o.set("params", json::Value::make_str(e.params));
            o.set("quant", json::Value::make_str(e.quant));
            o.set("bytes", json::Value::make_num((double)e.bytes));
            o.set("note", json::Value::make_str(e.note));
            o.set("installed", json::Value::make_bool(is_file_p(target)));
            {
                std::lock_guard<std::mutex> lk(*dl_mu);
                auto it = dl_jobs->find(e.id);
                if (it != dl_jobs->end()) {
                    auto& j = *it->second;
                    o.set("state", json::Value::make_str(j.state));
                    o.set("total", json::Value::make_num((double)j.total));
                    long long got = 0;
                    struct stat st {};
                    if (::stat((target + ".part").c_str(), &st) == 0) got = (long long)st.st_size;
                    o.set("got", json::Value::make_num((double)got));
                    if (!j.error.empty())
                        o.set("error", json::Value::make_str(j.error));
                }
            }
            arr.arr.push_back(o);
        }
        root.set("items", arr);
        c.send(200, "application/json", json::dump(root));
    });

    srv.post("/api/store/download", [dl_mu, dl_jobs, mirror](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        std::string id = req.param("id");
        if (id.empty() && json::parse(req.body, &body)) id = body.s("id");
        const StoreEntry* e = nullptr;
        for (const auto& s : kStore) if (id == s.id) { e = &s; break; }
        if (!e) {
            c.send(400, "application/json", "{\"error\":\"没有这个模型\"}");
            return;
        }
        if (g.model_dir.empty()) {
            c.send(400, "application/json", "{\"error\":\"模型目录未配置\"}");
            return;
        }
        const std::string target = g.model_dir + "/" + e->file;
        const std::string part = target + ".part";
        {
            std::lock_guard<std::mutex> lk(*dl_mu);
            auto it = dl_jobs->find(id);
            if (it != dl_jobs->end() && it->second->state == "running") {
                c.send(200, "application/json", "{\"ok\":true,\"state\":\"running\"}");
                return;
            }
            auto job = std::make_shared<DlJob>();
            job->id = id;
            job->file = e->file;
            job->total = e->bytes;
            std::string url = e->url;
            // 镜像只在 host 层面替换，路径一个字节都不动 —— 镜像站用的
            // 就是同一套路径，改别的反而会 404。
            static const std::string kHf = "https://huggingface.co";
            if (!mirror->empty() && url.rfind(kHf, 0) == 0)
                url = *mirror + url.substr(kHf.size());
            job->th = std::thread([job, url, part, target, dl_mu]() {
                // 用 fork/exec 而不是 std::system：
                // 取消下载必须能拿到 curl 的 pid 去 kill，而 system() 只返回
                // 一个退出码 —— 拿到退出码的时候 curl 已经跑完了，停不下来。
                pid_t pid = fork();
                if (pid == 0) {
                    // -C - 断点续传；下载完再改名，避免半个文件被扫进清单。
                    execl("/usr/bin/curl", "curl", "-fL", "--retry", "3",
                          "--retry-delay", "2", "-C", "-", "-o", part.c_str(),
                          url.c_str(), (char*)nullptr);
                    _exit(127);
                }
                if (pid < 0) {
                    job->state = "error";
                    job->error = "无法启动下载进程";
                    return;
                }
                job->pid = pid;
                int st = 0;
                while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
                {
                    std::lock_guard<std::mutex> lk2(*dl_mu);
                    if (job->state == "cancelled") {
                        // 用户取消：半截 .part 留着，下次「重试」能接着下。
                        return;
                    }
                }
                if (WIFEXITED(st) && WEXITSTATUS(st) == 0) {
                    if (std::rename(part.c_str(), target.c_str()) == 0) {
                        job->state = "done";
                    } else {
                        job->state = "error";
                        job->error = "下载完成但改名失败";
                    }
                } else {
                    job->state = "error";
                    job->error = "下载失败（网络或磁盘），已保留 .part 可续传";
                }
            });
            job->th.detach();
            (*dl_jobs)[id] = job;
        }
        c.send(200, "application/json", "{\"ok\":true,\"state\":\"running\"}");
    });

    // 导入：把用户自己的 .gguf 软链进模型目录。
    // 不复制 —— 一个 5GB 的文件复制一份，磁盘直接少 5GB。
    srv.post("/api/models/import", [](const http::Request& req, http::Conn& c) {
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
        if (!is_file_p(full)) {
            c.send(400, "application/json", "{\"error\":\"文件不存在\"}");
            return;
        }
        const std::string name = full.substr(full.find_last_of('/') + 1);
        const std::string link = g.model_dir + "/" + name;
        std::error_code ec;
        std::filesystem::create_symlink(full, link, ec);
        if (ec) {
            c.send(400, "application/json",
                   "{\"error\":\"建链接失败：" + json_escape(ec.message()) + "\"}");
            return;
        }
        json::Value r = json::Value::make_obj();
        r.set("ok", json::Value::make_bool(true));
        r.set("name", json::Value::make_str(name));
        c.send(200, "application/json", json::dump(r));
    });

    // 取消下载。kill 掉 curl，把状态标成 cancelled 并保留 .part ——
    // 用户点了取消之后又想要，还能接着下，不用从头来。
    srv.post("/api/store/cancel", [dl_mu, dl_jobs](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        std::string id = req.param("id");
        if (id.empty() && json::parse(req.body, &body)) id = body.s("id");
        {
            std::lock_guard<std::mutex> lk(*dl_mu);
            auto it = dl_jobs->find(id);
            if (it != dl_jobs->end() && it->second->state == "running") {
                it->second->state = "cancelled";
                if (it->second->pid > 0) ::kill(it->second->pid, SIGTERM);
                c.send(200, "application/json", "{\"ok\":true,\"state\":\"cancelled\"}");
                return;
            }
        }
        c.send(200, "application/json", "{\"ok\":true,\"state\":\"idle\"}");
    });

    // 删除模型。只允许删模型目录里的文件，且不能删当前正在加载的那个 ——
    // 删了它，引擎手里还攥着已映射的权重，之后任何一次切换都会失败。
    srv.post("/api/models/delete", [](const http::Request& req, http::Conn& c) {
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
        if (g.model_dir.empty() || full.rfind(path_abs(g.model_dir) + "/", 0) != 0) {
            c.send(400, "application/json", "{\"error\":\"只能删模型目录里的文件\"}");
            return;
        }
        if (full == path_abs(g.model_path)) {
            c.send(400, "application/json",
                   "{\"error\":\"这是当前加载的模型，先切到别的再删\"}");
            return;
        }
        if (!is_file_p(full)) {
            c.send(404, "application/json", "{\"error\":\"文件不存在\"}");
            return;
        }
        if (std::remove(full.c_str()) != 0) {
            c.send(400, "application/json", "{\"error\":\"删除失败，可能没有权限\"}");
            return;
        }
        c.send(200, "application/json", "{\"ok\":true}");
    });

    // 下载源。空字符串 = 官方 huggingface.co。
    srv.get("/api/store/mirror", [mirror](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value r = json::Value::make_obj();
        r.set("mirror", json::Value::make_str(*mirror));
        c.send(200, "application/json", json::dump(r));
    });
    srv.post("/api/store/mirror", [mirror, save_mirror](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        std::string m;
        if (json::parse(req.body, &body)) m = body.s("mirror");
        if (!m.empty() && m.rfind("http", 0) != 0) {
            c.send(400, "application/json", "{\"error\":\"镜像地址要以 http 开头\"}");
            return;
        }
        while (!m.empty() && m.back() == '/') m.pop_back();
        *mirror = m;
        save_mirror();
        json::Value r = json::Value::make_obj();
        r.set("ok", json::Value::make_bool(true));
        r.set("mirror", json::Value::make_str(*mirror));
        c.send(200, "application/json", json::dump(r));
    });

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
        for (auto& m : ms) if (!m.mlx) m.mmproj = find_sibling_mmproj(m.path);
        json::Value arr = json::Value::make_arr();
        for (const auto& m : ms) {
            json::Value o = json::Value::make_obj();
            o.set("path", json::Value::make_str(m.path));
            o.set("name", json::Value::make_str(m.name));
            o.set("bytes", json::Value::make_num((double)m.bytes));
            o.set("mtime", json::Value::make_num((double)m.mtime));
            // MLX 条目不猜量化也不猜参数量：目录名是 "lfm3-mlx-4bit"，
            // guess_params 会把 "4bit" 里的 4 当成 4B，界面上就出现一个
            // "4.0B · 928 MB" 的自相矛盾条目（4B 的 4bit 模型不可能只有 928 MB）。
            o.set("quant", json::Value::make_str(m.mlx ? "" : guess_quant(m.name)));
            o.set("params", json::Value::make_num(m.mlx ? 0 : guess_params(m.name)));
            o.set("loaded", json::Value::make_bool(m.path == g.model_path));
            // 有配套塔就能看图/听音。这里只报"有没有塔"和塔的路径，
            // 不报具体是视觉还是语音 —— 那要读塔的 GGUF 头，而几十个模型
            // 逐个读头是几百毫秒。真正的 caps 在 /api/state 里，加载后自然知道。
            o.set("mmproj", json::Value::make_str(m.mmproj));
            // MLX 条目只报"看得见、用不了"，不假装能加载。loadable=false
            // 的项前端会画成灰的并给一句原因，而不是让用户点下去等一个报错。
            o.set("mlx", json::Value::make_bool(m.mlx));
            o.set("loadable", json::Value::make_bool(!m.mlx));
            if (m.mlx) {
                o.set("note", json::Value::make_str(
                    "MLX 格式（safetensors 目录），当前引擎只加载 GGUF。"
                    "想跑它需要 MLX 运行时，这里只把它列出来免得你以为丢了。"));
            }
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
        // MLX 是目录 + safetensors，引擎只认 GGUF。在这里挡下来并说清楚，
        // 否则会走到 dubhe_load_model，报一句"magic 不是 GGUF"，用户完全
        // 看不懂那是什么意思。
        if (is_dir_p(full)) {
            c.send(400, "application/json",
                   "{\"error\":\"这是目录形式的模型（MLX / safetensors），"
                   "当前引擎只加载 GGUF 单文件。\"}");
            return;
        }
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
    // 运行期权限开关。
    //
    // 这些位以前只能由启动参数决定，改了要重启进程 —— 用户实际体验就是
    // "没有权限设置"：模型说"我没有执行命令的能力"，界面上也找不到任何
    // 地方能开。现在放进 /api/settings，UI 改完立刻生效，下一次工具调用
    // 就用新值。
    //
    // 只接受本机请求：把 shell 权限开放给局域网里任何一台机器，等于把
    // 这台 Mac 的终端交出去。
    srv.post("/api/settings", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req)) {
            c.send(403, "application/json", "{\"error\":\"权限设置只允许本机修改\"}");
            return;
        }
        json::Value body;
        if (!json::parse(req.body, &body)) {
            c.send(400, "application/json", "{\"error\":\"JSON 解析失败\"}");
            return;
        }
        {
            std::lock_guard<std::mutex> lk(g.mu);
            if (body.get("shell")) g.allow_shell = body.flag("shell", g.allow_shell);
            if (body.get("write")) g.allow_write = body.flag("write", g.allow_write);
            if (body.get("net")) g.allow_net = body.flag("net", g.allow_net);
            if (body.get("sandbox")) g.sandbox = body.flag("sandbox", g.sandbox);
            if (body.get("write_outside"))
                g.allow_write_outside = body.flag("write_outside", g.allow_write_outside);
            const double t = body.n("shell_timeout_s", 0);
            if (t >= 1 && t <= 600) g.shell_timeout_ms = (int)(t * 1000);
            const double tr = body.n("tool_rounds", 0);
            if (tr >= 1 && tr <= 24) g.max_tool_rounds = (int)tr;
            // 工作目录：必须是已存在的目录，且落在白名单根之内
            const std::string wd = body.s("workspace");
            if (!wd.empty()) {
                const std::string full = path_abs(wd);
                if (!is_dir_p(full)) {
                    c.send(400, "application/json", "{\"error\":\"目录不存在\"}");
                    return;
                }
                if (!path_allowed(full)) {
                    c.send(403, "application/json", "{\"error\":\"目录不在允许的范围内\"}");
                    return;
                }
                g.workspace = full;
            }
        }
        json::Value o = json::Value::make_obj();
        o.set("shell", json::Value::make_bool(g.allow_shell));
        o.set("write", json::Value::make_bool(g.allow_write));
        o.set("net", json::Value::make_bool(g.allow_net));
        o.set("sandbox", json::Value::make_bool(sandbox_active()));
        // 配置值单列一个字段：sandbox=false 且 sandbox_cfg=true，说明用户开了
        // 但本机套不上（外层沙箱/CI 容器），界面据此给一句解释而不是静默失效。
        o.set("sandbox_cfg", json::Value::make_bool(g.sandbox));
        o.set("write_outside", json::Value::make_bool(g.allow_write_outside));
        o.set("shell_timeout_s", json::Value::make_num(g.shell_timeout_ms / 1000.0));
        o.set("tool_rounds", json::Value::make_num(g.max_tool_rounds));
        o.set("workspace", json::Value::make_str(g.workspace));
        c.send(200, "application/json", json::dump(o));
    });

    // ---- 逐次审批的应答口
    //
    // 只碰 ask_mu，绝不碰 g.mu —— 等待中的那条 chat 请求正压在 g.mu 上，
    // 这里要是也去抢 g.mu，就成了"用户点了允许、服务端却在等自己"。
    srv.post("/api/approve", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req)) {
            c.send(403, "application/json", "{\"error\":\"审批只允许本机操作\"}");
            return;
        }
        json::Value body;
        if (!json::parse(req.body, &body)) {
            c.send(400, "application/json", "{\"error\":\"JSON 解析失败\"}");
            return;
        }
        const std::string id = body.s("id");
        if (id.empty()) {
            c.send(400, "application/json", "{\"error\":\"缺少 id\"}");
            return;
        }
        const bool allow = body.flag("allow", false);
        {
            std::lock_guard<std::mutex> lk(g.ask_mu);
            // 没人认领的授权会一直留在这张表里（chat 那条路等完就删，但
            // /api/tools/run 拿到 id 之后可能再也不来第二次）。id 是单调
            // 递增的 ask<N>，std::map 按字典序排，超了就从最小的删起。
            while (g.ask_result.size() > 64) g.ask_result.erase(g.ask_result.begin());
            g.ask_result[id] = allow ? 1 : 2;
        }
        g.ask_cv.notify_all();
        c.send(200, "application/json",
               std::string("{\"ok\":true,\"allow\":") + (allow ? "true" : "false") + "}");
    });

    srv.get("/api/tools", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value root = json::Value::make_obj();
        root.set("items", tools_json());
        root.set("workspace", json::Value::make_str(g.workspace));
        root.set("shell", json::Value::make_bool(g.allow_shell));
        root.set("write", json::Value::make_bool(g.allow_write));
        root.set("net", json::Value::make_bool(g.allow_net));
        root.set("sandbox", json::Value::make_bool(sandbox_active()));
        root.set("sandbox_cfg", json::Value::make_bool(g.sandbox));
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
        // arguments 允许是对象，也允许直接平铺在顶层（方便 curl 手测）。
        //
        // 顺带认一下 "args"：内部事件（tool_call）里这个字段叫 args，
        // 照着手写一遍 curl 就是 args，然后拿到一句"缺少 path 参数" ——
        // 明明传了 path，看起来像门禁坏了，实际是名字没对上。
        json::Value args = json::Value::make_obj();
        const json::Value* a = body.get("arguments");
        if (!a) a = body.get("args");
        if (a && a->type == json::Value::OBJ) args = *a;
        else args = body;
        // 路径与权限一律交给 gate()。这里原来还有一段自己的 path_allowed 检查，
        // 结果是同一个越界路径在两条路上返回两种错误形状 —— 模型看到的和手测
        // 看到的不是一回事，排查时最容易把人带偏。
        //
        // 门禁说"要问一次"的时候，这个接口不能只说一句"需要确认后才能执行"
        // 就收摊：调用方（curl、脚本、第三方客户端）没有任何地方能点"允许"，
        // 这条路走到一半就断了。给它一个能接上的闭环：
        //   1) 本次**不执行**，返回 need_approval + ask.id
        //   2) POST /api/approve {id, allow:true}
        //   3) 带 "approve":"<id>" 再调一次，这一次真的执行
        // 一次授权只对一次调用有效 —— 消费掉就删，不给"批准一次、反复用"。
        bool approved = false;
        {
            const GateOut gt = gate(name, args, false);
            if (gt.v == Verdict::Ask) {
                const json::Value* av = body.get("approve");
                const std::string aid =
                    (av && av->type == json::Value::STR) ? av->str : std::string();
                if (!aid.empty()) {
                    std::lock_guard<std::mutex> lk(g.ask_mu);
                    auto it = g.ask_result.find(aid);
                    if (it != g.ask_result.end()) {
                        approved = (it->second == 1);
                        g.ask_result.erase(it);
                    }
                }
                if (!approved) {
                    const std::string nid = "ask" + std::to_string(++g.ask_seq);
                    json::Value ask = json::Value::make_obj();
                    ask.set("id", json::Value::make_str(nid));
                    ask.set("name", json::Value::make_str(name));
                    ask.set("detail", json::Value::make_str(gt.detail));
                    ask.set("reason", json::Value::make_str(gt.reason));
                    json::Value o = json::Value::make_obj();
                    o.set("ok", json::Value::make_bool(false));
                    o.set("need_approval", json::Value::make_bool(true));
                    o.set("ask", ask);
                    o.set("output", json::Value::make_str(
                        "这次调用需要用户确认，还没有执行。确认后带上 approve 再调一次。"));
                    c.send(200, "application/json", json::dump(o));
                    return;
                }
            }
        }
        const ToolResult r = run_tool(name, args, approved);
        json::Value o = json::Value::make_obj();
        o.set("ok", json::Value::make_bool(r.ok));
        o.set("denied", json::Value::make_bool(r.denied));
        o.set("card", json::Value::make_str(r.card));
        o.set("meta", r.meta);
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

    // 上下文压缩：把一段较早的对话压成要点。
    //
    // 为什么放在服务端：只有这里有模型。前端能做的只是"截断"，那等于
    // 把用户前半小时说过的话直接扔掉；压缩至少把结论留下来。
    //
    // 调用方（前端）负责决定"压哪一段"，这里只管"把给的这段压成一段话"。
    // 这样服务端不需要知道会话结构，也不用持久化任何对话 —— 隐私面上
    // 更干净：对话始终只存在浏览器里。
    srv.post("/api/summarize", [](const http::Request& req, http::Conn& c) {
        if (!is_local(req) && !authorized(req)) {
            c.send(401, "application/json", "{\"error\":\"未授权\"}");
            return;
        }
        json::Value body;
        if (!json::parse(req.body, &body)) {
            c.send(400, "application/json", "{\"error\":\"JSON 解析失败\"}");
            return;
        }
        const json::Value* msgs = body.get("messages");
        if (!msgs || msgs->type != json::Value::ARR || msgs->arr.empty()) {
            c.send(400, "application/json", "{\"error\":\"缺少 messages\"}");
            return;
        }
        std::string dialog;
        for (auto& m : msgs->arr) {
            const std::string role = m.s("role");
            const std::string ct = m.s("content");
            if (ct.empty()) continue;
            dialog += (role == "user" ? "用户：" : "助手：");
            dialog += ct;
            dialog += "\n";
        }
        if (dialog.empty()) {
            c.send(400, "application/json", "{\"error\":\"没有可压缩的正文\"}");
            return;
        }
        // 只压最近这一段：太长的输入在 1-2B 模型上既慢又容易中途跑偏
        const size_t CAP = 20000;
        if (dialog.size() > CAP) dialog = dialog.substr(dialog.size() - CAP);

        const std::string ask =
            "把下面这段对话压缩成中文要点，供之后继续对话时参考。要求：\n"
            "1. 保留事实、数字、结论、用户明确提出的要求和偏好；\n"
            "2. 删掉寒暄、重复和已经作废的方案；\n"
            "3. 用短句分条写，不要复述原文，不要加评论；\n"
            "4. 总长不超过 300 字。\n\n对话记录：\n" + dialog + "\n\n压缩结果：";

        std::unique_lock<std::mutex> lk(g.mu, std::defer_lock);
        if (!lk.try_lock()) {
            c.send(409, "application/json", "{\"error\":\"引擎正在生成，稍后再试\"}");
            return;
        }
        dubhe_model_info mi{};
        dubhe_model_info_get(g.engine, &mi);

        json::Value one = json::Value::make_arr();
        json::Value m0 = json::Value::make_obj();
        m0.set("role", json::Value::make_str("user"));
        m0.set("content", json::Value::make_str(ask));
        one.arr.push_back(m0);

        MmCache mc;
        std::vector<int32_t> toks;
        std::string berr;
        if (!build_tokens(g.engine, one, mi.has_chat_template != 0, false, false,
                          &toks, &mc, &berr)) {
            c.send(500, "application/json", "{\"error\":" + json_escape("分词失败: " + berr) + "}");
            return;
        }
        dubhe_sampling sp{};
        sp.max_tokens = 420;
        sp.temperature = 0.2f;
        sp.top_k = 20;
        sp.top_p = 0.9f;
        sp.repeat_penalty = 1.05f;
        sp.repeat_last_n = 64;

        SumCtx sc;
        dubhe_stats st{};
        dubhe_generate(g.engine, toks.data(), (int32_t)toks.size(), &sp, sum_piece_cb,
                       nullptr, &sc, &st);

        // 摘要短得离谱时，补一段原文摘录兜底。
        //
        // 1-2B 模型经常把"压缩成要点"理解成"说一句总结"：实测 20 条历史压出来
        // 17 个字 —— 而这 17 个字要顶替掉前面十几轮对话。这种压缩比截断更糟：
        // 截断至少留着原文，压缩是把原文**删了**换一行字。
        //
        // 所以加一道下限：输入够长、输出却短到不可能是有效摘要时，把对话开头
        // 原样附在后面。宁可摘要里混着原文，也不能让上下文凭空少掉一大块。
        std::string out = sc.out;
        if (dialog.size() > 600 && out.size() < 80) {
            size_t cut = std::min<size_t>(dialog.size(), 600);
            cut = utf8_back(dialog, cut);
            out += "\n（模型给出的摘要过短，附对话开头原文）\n";
            out += dialog.substr(0, cut);
            out += "…";
        }
        json::Value o = json::Value::make_obj();
        o.set("summary", json::Value::make_str(out));
        o.set("tokens", json::Value::make_num((double)st.n_generated));
        o.set("fellback", json::Value::make_bool(out.size() != sc.out.size()));
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
        bool with_tools = body.get("tools") != nullptr || body.flag("dubhe_tools", false);

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
            // 排队期间不能一声不吭。
            //
            // 本地单机场景下"等一下再答"比"直接失败"符合直觉，但排队时客户端
            // 收到的第一个字节还没有 —— 界面上和"卡死"完全一样，用户会以为
            // 坏了，去点停止、刷新、甚至重启服务。所以先把 SSE 头发出去，再发
            // 一个 queued 事件，界面就有话可说。
            if (stream) {
                c.begin_stream();
                c.write_chunk("data: {\"dubhe\":{\"type\":\"queued\"}}\n\n");
            }
            // 分片等锁，而不是 lock() 一次等到底：
            //   1) 每片看一眼对端还在不在 —— 排队时关掉页面的人，不该在轮到
            //      它之后还让引擎跑一遍没人看的结果；
            //   2) 每 5 秒发一个 SSE 注释心跳，免得中间的代理或浏览器把这条
            //      一直没数据的连接掐掉。
            auto last_ping = std::chrono::steady_clock::now();
            for (;;) {
                if (gen_lk.try_lock()) break;
                if (c.peer_gone()) {
                    std::fprintf(stderr, "[dubhed] 排队中客户端已断开，放弃\n");
                    c.end_stream();
                    return;
                }
                const auto now = std::chrono::steady_clock::now();
                if (stream && now - last_ping > std::chrono::seconds(5)) {
                    last_ping = now;
                    c.write_chunk(": keep-alive\n\n");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
        }
        g.busy = true;
        struct BusyGuard {
            ~BusyGuard() { g.busy = false; }
        } busy_guard;

        // 请求级的"深度思考"。界面上那个开关按每条消息发过来，覆盖启动时的
        // 全局默认值（--thinking / DUBHE_THINKING），请求结束还原。
        //
        // 为什么敢用全局量：整条请求串行化在 g.mu 上（上面刚拿到锁），同一
        // 时刻只有一个请求在拼 prompt。而且这个 scope 声明在锁之后，
        // 析构（还原）发生在锁释放之前，不会漏给下一个请求。
        struct ThinkScope {
            bool old = false;
            bool active = false;
            ~ThinkScope() { if (active) g_think_enabled = old; }
        } think_scope;
        if (body.get("thinking")) {
            const bool want = body.flag("thinking", false);
            if (want != g_think_enabled) {
                think_scope.old = g_think_enabled;
                think_scope.active = true;
                g_think_enabled = want;
            }
        }

        // 跳过思考通道：翻译、改写、抽取这类"照着做就行"的任务用。
        //
        // 和 thinking=false 的区别：那个只是"想完了不给你看"，模型照样要想，
        // 时间照样花；no_think 是连 `<think>` 都不替它开，让它直接出结果。
        // 万一它还是自己写了 <think>，扫描器照样会把那段收进通道丢掉 ——
        // 所以不会出现"推理漏进译文"。
        t_no_think = body.get("no_think") && body.flag("no_think", false);

        // 提前失败的统一出口。
        //
        // 排队那条路上头已经发出去了（见上），此时 c.send(400) 会因为
        // sent_header_ 直接返回，客户端拿到一条空流 —— 比报错还难查。
        auto fail = [&](const std::string& msg) {
            if (c.header_sent()) {
                c.write_chunk("data: {\"dubhe\":{\"type\":\"error\",\"message\":" +
                              json_escape(msg) + "}}\n\n");
                c.write_chunk("data: [DONE]\n\n");
                c.end_stream();
            } else {
                c.send(400, "application/json",
                       "{\"error\":{\"message\":" + json_escape(msg) +
                           ",\"type\":\"invalid_request_error\"}}");
            }
        };

        // 请求级工作目录。
        //
        // 前端每个对话可以绑定一个"项目文件夹"，工具的相对路径就以它为根。
        // 改的是 g.workspace 这个全局量，看起来危险 —— 但整条请求是串行化
        // 在 g.mu 上的（上面刚拿到锁），同一时刻只可能有一个请求在跑，
        // 而且用 RAII 恢复，异常路径也不会漏。换成请求上下文对象当然更干净，
        // 但那要把 workspace 一路传进 run_tool / path_abs 的每个调用点，
        // 改动面大得多，收益只有"并发时不互相影响"—— 而这里本来就不并发。
        struct WsScope {
            std::string old;
            bool active = false;
            ~WsScope() { if (active) g.workspace = old; }
        } ws_scope;
        {
            const std::string want = body.s("workspace");
            if (!want.empty()) {
                const std::string full = path_abs(want);
                if (is_dir_p(full) && path_allowed(full) && full != g.workspace) {
                    ws_scope.old = g.workspace;
                    ws_scope.active = true;
                    g.workspace = full;
                }
            }
        }

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
            fail(berr);
            return;
        }
        const int32_t n = (int32_t)toks.size();

        // 输入本身就超过上下文时，generate 会一开始就 break：没有报错、没有
        // 正文，界面上是"生成中"转很久然后什么都没有。这里直接说清楚。
        dubhe_model_info lim{};
        dubhe_model_info_get(g.engine, &lim);
        if (lim.n_ctx_limit > 0 && n >= lim.n_ctx_limit) {
            fail("输入太长：" + std::to_string(n) + " tokens，超过当前上下文上限 " +
                 std::to_string(lim.n_ctx_limit) +
                 "。请缩短问题、清空历史，或换一个上下文更长的模型。");
            return;
        }

        const std::string id = now_id();
        long long created = (long long)time(nullptr);

        // ---- 纯算式直接给答案，不经过模型 ----
        //
        // 这一条对所有模型都生效，不只是小模型：算术没有"风格"可言，正确答案
        // 只有一个。让模型算，最坏的结果是它算错了还说得很有道理。
        // 命中条件很窄（整句就是一个算式），所以不会抢走"解释一下为什么"这类问题。
        {
            std::string ans;
            if (g.no_arith_fastpath == 0 && try_quick_answer(last_user_text(*msgs), &ans)) {
                if (stream) {
                    if (!c.header_sent()) c.begin_stream();
                    c.write_chunk("data: {\"id\":\"" + id +
                                  "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                                  std::to_string(created) +
                                  ",\"choices\":[{\"index\":0,\"delta\":{\"content\":" +
                                  json_escape(ans) + "},\"finish_reason\":null}]}\n\n");
                    c.write_chunk("data: {\"id\":\"" + id +
                                  "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
                                  "\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{"
                                  "\"completion_tokens\":0,\"prompt_tokens\":0,"
                                  "\"decode_tps\":0,\"reused_tokens\":0}}\n\n");
                    c.write_chunk("data: [DONE]\n\n");
                    c.end_stream();
                } else {
                    json::Value root = json::Value::make_obj();
                    root.set("id", json::Value::make_str(id));
                    root.set("object", json::Value::make_str("chat.completion"));
                    root.set("created", json::Value::make_num((double)created));
                    json::Value choices = json::Value::make_arr();
                    json::Value ch = json::Value::make_obj();
                    ch.set("index", json::Value::make_num(0));
                    json::Value msg = json::Value::make_obj();
                    msg.set("role", json::Value::make_str("assistant"));
                    msg.set("content", json::Value::make_str(ans));
                    ch.set("message", msg);
                    ch.set("finish_reason", json::Value::make_str("stop"));
                    choices.arr.push_back(ch);
                    root.set("choices", choices);
                    json::Value usage = json::Value::make_obj();
                    usage.set("prompt_tokens", json::Value::make_num(0));
                    usage.set("completion_tokens", json::Value::make_num(0));
                    usage.set("decode_tps", json::Value::make_num(0));
                    usage.set("reused_tokens", json::Value::make_num(0));
                    root.set("usage", usage);
                    c.send(200, "application/json", json::dump(root));
                }
                g.total_requests++;
                return;
            }
        }

        // 大输入先报一声。
        //
        // 预填充是**一个字节都不吐**的一段：5 万 token 在 1.7B 上大约要一分钟，
        // 这段时间界面上只有"生成中…"，用户以为卡死，去点停止、刷新。
        // 这里先把 SSE 头和一个 note 发出去，界面就有话可说。
        if (stream && n > 8000) {
            c.begin_stream();   // 幂等：已经发过头就直接返回
            c.write_chunk("data: {\"dubhe\":{\"type\":\"note\",\"message\":" +
                          json_escape("输入约 " + std::to_string(n) +
                                      " tokens，正在做预填充，要等一会儿") +
                          "}}\n\n");
        }

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
            // 开了深度思考就把推理过程推给界面（单独一条 think 事件，
            // 由前端折进"思考过程"块）；关着就照旧当垃圾丢掉。
            ctx.think_emit = g_think_enabled && g_think_style;

            dubhe_stats st{};

            // 调用方自己跑循环时服务端一轮都不执行；否则最多跑 g.max_tool_rounds 轮。
            //
            // 上限本身是必要的：2B 级别的模型看到工具结果后经常又吐一次
            // tool_call（尤其结果里有代码/JSON 时），不设上限会来回打转。
            // 但原来写死 2 太紧 —— "grep 定位 → 读那处 → 读相关文件 → 改 →
            // 回答"这种最普通的组合第三轮就被掐断。现在默认 6，可调，
            // 并且保留下面那个"同样的调用连做两次就停"的保护。
            const bool client_drives = has_tool_result(*msgs);
            const int max_rounds = client_drives ? 0 : std::max(1, g.max_tool_rounds);
            int rounds = 0;
            std::vector<int32_t> cur = toks;
            // 这一轮请求里跑过的所有工具签名。用"见过没有"而不是"和上一次比"，
            // 因为模型打转的方式是 A B A B 交替，不是 A A A。
            std::vector<std::string> seen_sigs;
            const char* finish = "stop";
            // 每一轮都从 msgs 出发重新拼（工具结果追加到副本上），所以它必须
            // 活到下一轮 —— 轮次用完时要用"上一轮那份含工具结果的对话"去开
            // 最后一个回答轮。
            json::Value msgs2 = *msgs;
            // 这一轮喂进去的是哪段 prompt（前缀缓存的代号）。生成完要拿它
            // 把"新吐出来的字和 token"接到缓存尾巴上。
            uint64_t pcid = g_pcache.current_id();

            // "最后一轮只回答"。
            //
            // 两种情况下都会走到：轮次用完了，或者发现模型在原地打转。两处
            // 原来各写各的 —— 轮次那处会补一轮不带工具的回答，打转那处只
            // emit 一句"已停止"就 break，用户拿到的是一句告警加零结论。
            // 抽成一个 lambda，两条路径都以"拿到回答"收尾。
            //
            // 返回 false 表示连这一轮都拼不出来（上下文放不下），调用方只能
            // 如实说停在哪。
            auto final_answer_round = [&](const std::string& why) -> bool {
                json::Value fin = msgs2;
                json::Value note = json::Value::make_obj();
                note.set("role", json::Value::make_str("user"));
                note.set("content", json::Value::make_str(why));
                fin.arr.push_back(note);
                std::vector<int32_t> t3;
                std::string e3;
                if (!build_tokens(g.engine, fin, chatml, false, force_raw, &t3, &mcache, &e3) ||
                    t3.empty()) {
                    return false;
                }
                cur = std::move(t3);
                with_tools = false;   // 下一轮不再找 tool_call，也不再抑制
                pcid = g_pcache.current_id();
                return true;
            };

            while (!cur.empty()) {
                rtext.clear();
                ctx.acc = &rtext;
                ctx.first = true;
                ctx.decided = false;
                ctx.suppress = false;
                ctx.pending.clear();
                ctx.tail.clear();
                // 提示词末尾已经替模型打开了 <think>（见 build_prompt），
                // 所以这条流是**从思考块内部**开始的 —— 模型不会再吐一个
                // <think> 出来，扫描器必须一开始就处于 in_think。
                // 原来这里写 false，结果整段推理被当成正文直接吐出去，
                // 就是"关了开关反而更难读"的原因。
                ctx.in_think = (g_think_style != 0 && !t_no_think);
                ctx.think_buf.clear();
                ctx.think_scan.clear();
                ctx.visible = 0;
                ctx.gtoks.clear();
                // 有工具说明时才做抑制判定 —— 否则根本没在找 tool_call，
                // 直接往外吐就是最快的路。
                ctx.no_suppress = !with_tools;

                dubhe_stats s{};
                dubhe_generate(g.engine, cur.data(), (int32_t)cur.size(), &sp, on_piece,
                               conn_gone_cb, &ctx, &s);
                flush_pending(&ctx);

                // 客户端走了就别再往下走：不要跑工具（可能是个写文件/shell），
                // 也不要再开下一轮生成。这一步省下的不只是算力 —— 工具是有
                // 副作用的，为一个已经关掉的页面去改文件，用户下次打开会
                // 看到一个自己没批准过的改动。
                if (c.peer_gone()) break;

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
                    // （模型少个尖括号、工具名写坏、JSON 截断在 max_tokens 上，
                    // 都很常见）。藏了又没人接手 —— 用户看到的就是
                    // "已完成 · 0 tokens"加一片空白。
                    //
                    // 但"原样吐回去"也不行：吐回去的是
                    // <tool_call><function=… 这种程序内部结构，用户读不懂，
                    // 只会觉得软件坏了（上一版就是这么被投诉的）。
                    // 剥掉标签，剩下的才是模型真说过的人话；一句都没有，
                    // 就明说这次为什么没答案。
                    if (ctx.suppress) {
                        const std::string vis = scrub_history(rtext);
                        emit_delta(&ctx, vis.empty()
                            ? "（模型这一轮想调用工具，但调用格式不合法，没有产生可显示的回答。换个问法再试一次。）"
                            : vis);
                    }
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
                    // 轮次用完了，模型还在要工具。
                    //
                    // 直接说一句"已达上限"最省事，但用户拿到的是**没有结论**
                    // 的回答：目录列了、文件读了，就是不告诉你结果。实测
                    // "先列 docs 再读 README 前五行"正好两轮用光，最后吐出一段
                    // 裸 <tool_call>，用户什么也没得到。
                    //
                    // 所以再开一轮**不带工具说明**的生成：把上一轮那份含工具
                    // 结果的对话原样摆回去，末尾补一句"工具次数用完了，直接用
                    // 上面的结果回答"。这一轮的 prompt 前缀和上一轮完全一致，
                    // 只是尾巴多了一句，KV 前缀缓存还能接上，代价很小。
                    if (final_answer_round(
                            "工具调用次数已经用完。不要再调用任何工具，"
                            "直接用上面拿到的结果回答问题；拿不到的部分就说没拿到。")) {
                        continue;
                    }
                    // 连这一步都拼不出来（上下文放不下），只能如实说停在哪。
                    emit_delta(&ctx, "（模型反复请求调用工具 " + tname + "，已达 " +
                                         std::to_string(max_rounds) +
                                         " 轮上限，已停止。可以换个问法再试。）");
                    break;
                }

                // 同一个工具、同一组参数，再跑一遍结果一模一样，循环不可能收敛。
                //
                // 原来是"和上一次比"（last_sig），只挡得住 A A A 这种连发。
                // 实测 1.7B 更常见的是 A B A B 来回切：list_dir → read_file →
                // list_dir → read_file，连续两次永远不相同，守卫一次都没触发，
                // 六轮全烧在这种乒乓上。改成"和这一轮请求里所有跑过的比"，
                // 一旦出现重复就直接进"只回答"那一轮。
                const std::string sig = tname + "|" + json::dump(targs);
                if (std::find(seen_sigs.begin(), seen_sigs.end(), sig) != seen_sigs.end()) {
                    if (final_answer_round(
                            "这个工具和参数你前面已经调用过了，结果不会变。"
                            "不要再调用工具，直接用已经拿到的结果回答。")) {
                        continue;
                    }
                    emit_delta(&ctx, "（模型在重复调用同一个工具 " + tname +
                                         "，结果不会变，已停止。可以换个问法再试。）");
                    break;
                }
                seen_sigs.push_back(sig);

                c.write_chunk("data: {\"dubhe\":{\"type\":\"tool_call\",\"name\":" +
                              json_escape(tname) + ",\"arguments\":" + json::dump(targs) +
                              "}}\n\n");

                // 需要用户点头的调用：发一张审批卡，然后在这里等。
                //
                // 等待有上限（90 秒）。超时按拒绝处理 —— 宁可让模型收到
                // "用户没回应"，也不能让一条没人确认的 sudo 在后台跑起来。
                bool approved = false;
                const GateOut gt = gate(tname, targs, false);
                if (gt.v == Verdict::Ask) {
                    {
                        const std::string aid = "ask" + std::to_string(++g.ask_seq);
                        json::Value d = json::Value::make_obj();
                        d.set("type", json::Value::make_str("tool_ask"));
                        d.set("id", json::Value::make_str(aid));
                        d.set("name", json::Value::make_str(tname));
                        d.set("detail", json::Value::make_str(gt.detail));
                        json::Value ev = json::Value::make_obj();
                        ev.set("dubhe", d);
                        c.write_chunk("data: " + json::dump(ev) + "\n\n");

                        std::unique_lock<std::mutex> lk(g.ask_mu);
                        const auto deadline = std::chrono::steady_clock::now() +
                                              std::chrono::seconds(90);
                        // 一次 wait_until(90s) 是错的：用户在这 90 秒里关掉页面，
                        // 这条请求仍然压在 g.mu 上，后面所有请求陪着等。切成 1 秒
                        // 一片，每片顺便看一眼连接还在不在。
                        bool answered = false;
                        while (std::chrono::steady_clock::now() < deadline) {
                            if (g.ask_cv.wait_for(lk, std::chrono::seconds(1), [&] {
                                    return g.ask_result.count(aid) > 0;
                                })) {
                                answered = true;
                                break;
                            }
                            if (c.peer_gone()) break;
                        }
                        int verdict = 0;
                        if (answered) {
                            verdict = g.ask_result[aid];
                            g.ask_result.erase(aid);
                        } else {
                            // 没人答（超时或对端走了）：把这条 id 清掉，免得
                            // ask_result 随着一次次的等待慢慢涨。
                            g.ask_result.erase(aid);
                        }
                        lk.unlock();
                        approved = (verdict == 1);
                        // 把"用户怎么答的"也发回去，前端那张卡要变灰。
                        json::Value d2 = json::Value::make_obj();
                        d2.set("type", json::Value::make_str("tool_ask_done"));
                        d2.set("id", json::Value::make_str(aid));
                        d2.set("allow", json::Value::make_bool(approved));
                        d2.set("timeout", json::Value::make_bool(!answered));
                        json::Value ev2 = json::Value::make_obj();
                        ev2.set("dubhe", d2);
                        c.write_chunk("data: " + json::dump(ev2) + "\n\n");
                    }
                }

                // 等审批的这段时间里用户可能已经把页面关了。这时候再往下跑
                // 工具就是"替一个不在场的人动文件"，直接收手。
                if (c.peer_gone()) break;

                ToolResult tr;
                if (gt.v == Verdict::Ask && !approved) {
                    // 用户拒绝 / 超时没答。这里必须自己造结果，不能交给
                    // run_tool —— 它会把 gate 的 Ask 转成一句"需要确认后
                    // 才能执行"，模型读到的像是"流程还没走完"，于是重试同
                    // 一个调用。要明确告诉它这条路走不通。
                    tr.denied = true;
                    tr.output = "用户没有允许这次操作，所以它没有执行，什么都没有发生。"
                                "不要重试同一个调用，也不要声称它成功了。"
                                "直接告诉用户这一步被拒绝、没有执行，然后问他想怎么继续。";
                    tr.meta.set("note", json::Value::make_str("你拒绝了这一步，它没有执行。"));
                } else {
                    tr = run_tool(tname, targs, approved);
                }
                {
                    // 事件体走 json::dump 而不是手拼字符串：card/meta 里可能有
                    // 引号、换行、反斜杠（命令本身、diff 的 old/new 都有），
                    // 手拼一次漏转义就是前端 JSON.parse 失败 —— 而失败的样子
                    // 是"工具卡片不出现"，看不出是转义问题。
                    json::Value d = json::Value::make_obj();
                    d.set("type", json::Value::make_str("tool_result"));
                    d.set("ok", json::Value::make_bool(tr.ok));
                    d.set("denied", json::Value::make_bool(tr.denied));
                    d.set("card", json::Value::make_str(tr.card));
                    d.set("meta", tr.meta);
                    d.set("output", json::Value::make_str(tr.output.substr(0, 8000)));
                    json::Value ev = json::Value::make_obj();
                    ev.set("dubhe", d);
                    c.write_chunk("data: " + json::dump(ev) + "\n\n");
                }
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
                msgs2 = *msgs;
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
                // 被拒绝的那一轮用另一个角色名。用 "tool" 的话，
                // build_tokens 会照常加上"工具已经执行完毕，上面就是返回结果"，
                // 而这条根本没执行 —— 小模型看到这句会当成"已经拿到了结果"，
                // 接着编一个"已写入成功"的回答（实测 LFM3-1.7B 就这样）。
                tm.set("role", json::Value::make_str(tr.denied ? "tool_denied" : "tool"));
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
            // 非流式这条路把**原始输出**整段吐回去，其中包括
            // "<推理…</think>答案" 这种结构 —— 流式通道有 feed_think 分流，
            // 这里没有，于是同一条请求 stream=true 干净、stream=false 满屏
            // 自言自语。scrub 一下，两条路给的东西保持一致。
            const bool truncated = sp.max_tokens > 0 && st.n_generated >= sp.max_tokens;
            std::string shown = scrub_history(strip_think_block(acc, truncated, g_think_style != 0));
            if (shown.empty() && !acc.empty()) {
                shown = "（模型把全部输出预算都用在了内部推理上，没有留下答案。"
                        "可以把最大生成长度调大，或打开「思考过程」看看它想了什么。）";
            }
            msg.set("content", json::Value::make_str(shown));
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
