// http.h — 极简 HTTP/1.1 服务（阻塞 socket + 线程/连接）
//
// 只服务本机单用户场景：不做 keep-alive 复用、不做 TLS、不做 chunked 请求体。
// 支持分块响应（SSE 流式输出）。
#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace http {

struct Request {
    std::string method;
    std::string path;
    std::string query;
    std::string body;
    std::string remote;   // 对端 IP（用于判断是否本机访问）
    std::map<std::string, std::string> headers;

    // HTTP 头名大小写不敏感
    std::string header(const std::string& k) const {
        auto it = headers.find(k);
        if (it != headers.end()) return it->second;
        auto low = [](std::string s) {
            for (auto& c : s) c = (char)std::tolower((unsigned char)c);
            return s;
        };
        const std::string want = low(k);
        for (auto& kv : headers)
            if (low(kv.first) == want) return kv.second;
        return "";
    }
    std::string param(const std::string& k) const {
        // 从 query 中取 ?a=1&b=2，并做 percent-decode。
        //
        // 解码是必须的：前端用 encodeURIComponent 编码路径，/Users/a b/c.md
        // 会变成 %2FUsers%2Fa%20b%2Fc.md。不解码的话服务端拿到的是一个
        // 以 %2F 开头的"相对路径"，stat 必然失败 —— 而且报的是"文件不存在"，
        // 看起来像权限问题，极难往编码上想。
        //
        // 只处理 %XX，不把 '+' 当空格：query 里 '+' 表示空格是表单编码的
        // 约定，而这里的参数基本都是文件路径，'+' 是合法的文件名字符。
        size_t p = 0;
        while (p < query.size()) {
            size_t eq = query.find('=', p);
            size_t amp = query.find('&', p);
            if (amp == std::string::npos) amp = query.size();
            if (eq != std::string::npos && eq < amp) {
                if (query.substr(p, eq - p) == k) return url_decode(query.substr(eq + 1, amp - eq - 1));
            }
            p = amp + 1;
        }
        return "";
    }

    static std::string url_decode(const std::string& s) {
        auto hex = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        std::string o;
        o.reserve(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '%' && i + 2 < s.size()) {
                const int h = hex(s[i + 1]), l = hex(s[i + 2]);
                if (h >= 0 && l >= 0) {
                    o += (char)(h * 16 + l);
                    i += 2;
                    continue;
                }
            }
            o += s[i];
        }
        return o;
    }
};

class Conn {
public:
    explicit Conn(int fd) : fd_(fd) {}

    void send(int status, const std::string& content_type, const std::string& body,
              const std::vector<std::pair<std::string, std::string>>& extra = {}) {
        if (sent_header_) return;
        std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
        h += "Content-Type: " + content_type + "\r\n";
        h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        h += "Cache-Control: no-store\r\n";
        for (auto& e : extra) h += e.first + ": " + e.second + "\r\n";
        h += "Connection: close\r\n\r\n";
        write_all(h);
        write_all(body);
        sent_header_ = true;
    }

    // SSE / 分块
    void begin_stream(const std::string& content_type = "text/event-stream") {
        if (sent_header_) return;
        std::string h = "HTTP/1.1 200 OK\r\n";
        h += "Content-Type: " + content_type + "\r\n";
        h += "Transfer-Encoding: chunked\r\n";
        h += "Cache-Control: no-cache\r\n";
        h += "Connection: close\r\n\r\n";
        write_all(h);
        sent_header_ = true;
        chunked_ = true;
    }

    void write_chunk(const std::string& data) {
        if (data.empty()) return;
        char len[32];
        std::snprintf(len, sizeof(len), "%zx\r\n", data.size());
        write_all(len);
        write_all(data);
        write_all("\r\n");
    }

    void end_stream() {
        if (!chunked_) return;
        write_all("0\r\n\r\n");
        chunked_ = false;
    }

    bool broken() const { return broken_; }

    // 响应头是不是已经发出去了。
    //
    // 排队等锁时我们会**先**发 SSE 头（否则客户端在排队期间一个字节都收不到，
    // 界面上和卡死一样）。此后任何提前失败都不能再走 send() —— 它看到
    // sent_header_ 就直接返回，客户端拿到一条空流，比报错更难查。
    bool header_sent() const { return sent_header_; }

    // 对端还在不在。
    //
    // broken_ 只有"写失败"才会置位。可流式生成里最要命的一种情况是
    // **用户关掉页面或按了停止**：连接被关掉，我们这边却还没写过任何东西
    // （比如正卡在等审批），broken_ 一直是 false，于是服务端继续占着全局
    // 生成锁，后面的请求全排队 —— 表现就是"发了消息没反应"。
    //
    // MSG_PEEK + 非阻塞探一次：
    //   >0     有数据（客户端又发了东西）→ 活着
    //   ==0    EOF，对端关了写端        → 断了
    //   <0     EAGAIN/EWOULDBLOCK       → 活着，只是没数据
    //          其它 errno                → 断了
    bool peer_gone() const {
        if (broken_) return true;
        char b = 0;
        const ssize_t n = ::recv(fd_, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n > 0) return false;
        if (n == 0) return true;
        return !(errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR);
    }

private:
    void write_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::send(fd_, s.data() + off, s.size() - off, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { broken_ = true; return; }
            off += (size_t)n;
        }
    }
    static const char* reason(int s) {
        switch (s) {
            case 200: return "OK";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 401: return "Unauthorized";
            case 403: return "Forbidden";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 409: return "Conflict";
            case 413: return "Content Too Large";
            case 431: return "Request Header Fields Too Large";
            case 500: return "Internal Server Error";
            case 503: return "Service Unavailable";
            case 507: return "Insufficient Storage";
            default:  return "OK";
        }
    }

    int fd_;
    bool sent_header_ = false;
    bool chunked_ = false;
    std::atomic<bool> broken_{false};
};

class Server {
public:
    using Handler = std::function<void(const Request&, Conn&)>;
    using Filter = std::function<bool(const Request&, Conn&)>;

    void get(const std::string& p, Handler h) { routes_["GET " + p] = std::move(h); }
    void post(const std::string& p, Handler h) { routes_["POST " + p] = std::move(h); }

    // 未命中任何路由时调用。用于静态资源通配，避免为每个文件注册一条路由。
    void set_fallback(Handler h) { fallback_ = std::move(h); }
    void set_filter(Filter f) { filter_ = std::move(f); }

    bool listen(const std::string& host, int port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        fcntl(fd_, F_SETFD, FD_CLOEXEC);
        int one = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) return false;
        if (::bind(fd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
            std::fprintf(stderr, "bind 失败: %s:%d (端口被占用？)\n", host.c_str(), port);
            return false;
        }
        if (::listen(fd_, 64) != 0) return false;
        return true;
    }

    void run() {
        running_ = true;
        while (running_) {
            sockaddr_in cli{};
            socklen_t len = sizeof(cli);
            int cfd = ::accept(fd_, (sockaddr*)&cli, &len);
            if (cfd < 0) {
                if (!running_) break;
                continue;
            }
            int one = 1;
            fcntl(cfd, F_SETFD, FD_CLOEXEC);
            setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            const timeval timeout{15, 0};
            setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            char ipbuf[64] = {0};
            inet_ntop(AF_INET, &cli.sin_addr, ipbuf, sizeof(ipbuf));
            const std::string peer = ipbuf;
            if (active_.fetch_add(1) >= 64) {
                active_.fetch_sub(1);
                Conn(cfd).send(503, "application/json", "{\"error\":\"连接已满，请稍后重试\"}");
                ::close(cfd);
                continue;
            }
            try {
                std::thread([this, cfd, peer] {
                    handle(cfd, peer);
                    ::close(cfd);
                    active_.fetch_sub(1);
                }).detach();
            } catch (const std::system_error&) {
                active_.fetch_sub(1);
                ::close(cfd);
            }
        }
    }

    void stop() {
        running_ = false;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

private:
    void handle(int cfd, const std::string& peer) {
        Conn conn(cfd);
        Request req;
        req.remote = peer;
        if (!read_request(cfd, &req, conn)) return;
        try {
            if (filter_ && !filter_(req, conn)) return;
            auto it = routes_.find(req.method + " " + req.path);
            if (it != routes_.end()) it->second(req, conn);
            else if (fallback_) fallback_(req, conn);
            else conn.send(404, "application/json", "{\"error\":\"not found\"}");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[dubhed] HTTP 请求失败: %s\n", e.what());
            conn.send(500, "application/json", "{\"error\":\"服务端无法处理该请求\"}");
        }
        conn.end_stream();
    }

    static bool read_request(int fd, Request* req, Conn& conn) {
        static constexpr size_t kHeadersLimit = 64 * 1024;
        static constexpr size_t kBodyLimit = 32 * 1024 * 1024;
        std::string raw;
        char buffer[8192];
        auto receive = [&](size_t max_read) {
            ssize_t n;
            do { n = recv(fd, buffer, max_read, 0); } while (n < 0 && errno == EINTR);
            if (n > 0) raw.append(buffer, (size_t)n);
            return n > 0;
        };
        auto bad_request = [&] {
            conn.send(400, "application/json", "{\"error\":\"请求头或正文长度无效\"}");
            return false;
        };

        size_t header_end;
        while ((header_end = raw.find("\r\n\r\n")) == std::string::npos) {
            if (raw.size() >= kHeadersLimit) {
                conn.send(431, "application/json", "{\"error\":\"请求头过大\"}");
                return false;
            }
            if (!receive(sizeof(buffer))) return false;
        }
        if (header_end > kHeadersLimit) {
            conn.send(431, "application/json", "{\"error\":\"请求头过大\"}");
            return false;
        }

        const size_t line_end = raw.find("\r\n");
        const size_t first_space = raw.find(' ');
        const size_t second_space = raw.find(' ', first_space + 1);
        if (first_space >= line_end || second_space >= line_end || second_space <= first_space + 1)
            return bad_request();
        req->method = raw.substr(0, first_space);
        const std::string target = raw.substr(first_space + 1, second_space - first_space - 1);
        if (target.front() != '/') return bad_request();
        const size_t question = target.find('?');
        req->path = target.substr(0, question);
        if (question != std::string::npos) req->query = target.substr(question + 1);

        for (size_t pos = line_end + 2; pos < header_end;) {
            const size_t end = raw.find("\r\n", pos);
            const size_t colon = raw.find(':', pos);
            if (colon == pos || colon >= end) return bad_request();
            std::string key = raw.substr(pos, colon - pos);
            for (auto& ch : key) ch = (char)std::tolower((unsigned char)ch);
            std::string value = raw.substr(colon + 1, end - colon - 1);
            const size_t start = value.find_first_not_of(" \t");
            const size_t last = value.find_last_not_of(" \t");
            value = start == std::string::npos ? "" : value.substr(start, last - start + 1);
            const bool inserted = req->headers.emplace(key, value).second;
            if (!inserted && (key == "host" || key == "content-length" || key == "authorization"))
                return bad_request();
            pos = end + 2;
        }

        if (!req->header("transfer-encoding").empty()) return bad_request();
        const auto length = req->headers.find("content-length");
        size_t bytes = 0;
        if (length != req->headers.end()) {
            const auto& value = length->second;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), bytes);
            if (result.ec != std::errc() || result.ptr != value.data() + value.size()) return bad_request();
        }
        if (bytes > kBodyLimit) {
            conn.send(413, "application/json", "{\"error\":\"请求正文超过 32 MiB，请缩小附件\"}");
            return false;
        }
        const size_t body_start = header_end + 4;
        const size_t body_end = body_start + bytes;
        while (raw.size() < body_end) {
            if (!receive(std::min(sizeof(buffer), body_end - raw.size()))) return false;
        }
        req->body = raw.substr(body_start, bytes);
        return true;
    }

    int fd_ = -1;
    std::atomic<int> active_{0};
    Filter filter_;
    std::atomic<bool> running_{false};
    std::map<std::string, Handler> routes_;
    Handler fallback_;
};

}  // namespace http
