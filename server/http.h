// http.h — 极简 HTTP/1.1 服务（阻塞 socket + 线程/连接）
//
// 只服务本机单用户场景：不做 keep-alive 复用、不做 TLS、不做 chunked 请求体。
// 支持分块响应（SSE 流式输出）。
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cerrno>
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
        h += "Access-Control-Allow-Origin: *\r\n";
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
        h += "Access-Control-Allow-Origin: *\r\n";
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
            if (n <= 0) { broken_ = true; return; }
            off += (size_t)n;
        }
    }
    static const char* reason(int s) {
        switch (s) {
            case 200: return "OK";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 500: return "Internal Server Error";
            default:  return "OK";
        }
    }

    int fd_;
    bool sent_header_ = false;
    bool chunked_ = false;
    bool broken_ = false;
};

class Server {
public:
    using Handler = std::function<void(const Request&, Conn&)>;

    void get(const std::string& p, Handler h) { routes_["GET " + p] = std::move(h); }
    void post(const std::string& p, Handler h) { routes_["POST " + p] = std::move(h); }

    // 未命中任何路由时调用。用于静态资源通配，避免为每个文件注册一条路由。
    void set_fallback(Handler h) { fallback_ = std::move(h); }

    bool listen(const std::string& host, int port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
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
            setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            char ipbuf[64] = {0};
            inet_ntop(AF_INET, &cli.sin_addr, ipbuf, sizeof(ipbuf));
            const std::string peer = ipbuf;
            std::thread([this, cfd, peer] { handle(cfd, peer); }).detach();
        }
    }

    void stop() {
        running_ = false;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

private:
    void handle(int cfd, const std::string& peer) {
        std::string raw;
        char buf[8192];
        size_t header_end = std::string::npos;
        size_t content_len = 0;
        bool have_len = false;

        while (true) {
            ssize_t n = ::recv(cfd, buf, sizeof(buf), 0);
            if (n <= 0) { ::close(cfd); return; }
            raw.append(buf, (size_t)n);
            if (header_end == std::string::npos) {
                header_end = raw.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    // 解析 Content-Length
                    std::string head = raw.substr(0, header_end);
                    size_t p = head.find("Content-Length:");
                    if (p == std::string::npos) p = head.find("content-length:");
                    if (p != std::string::npos) {
                        content_len = (size_t)std::strtoul(head.c_str() + p + 15, nullptr, 10);
                        have_len = true;
                    }
                }
            }
            if (header_end != std::string::npos) {
                size_t body_have = raw.size() - (header_end + 4);
                if (!have_len || body_have >= content_len) break;
            }
            if (raw.size() > 32u * 1024 * 1024) break;  // 32MB 上限
        }

        Request req;
        req.remote = peer;
        size_t line_end = raw.find("\r\n");
        std::string line = raw.substr(0, line_end);
        {
            size_t sp1 = line.find(' ');
            size_t sp2 = line.find(' ', sp1 + 1);
            if (sp1 != std::string::npos) {
                req.method = line.substr(0, sp1);
                std::string target = (sp2 == std::string::npos) ? line.substr(sp1 + 1)
                                                                : line.substr(sp1 + 1, sp2 - sp1 - 1);
                size_t q = target.find('?');
                if (q == std::string::npos) {
                    req.path = target;
                } else {
                    req.path = target.substr(0, q);
                    req.query = target.substr(q + 1);
                }
            }
        }
        size_t p = line_end + 2;
        while (p < header_end) {
            size_t e = raw.find("\r\n", p);
            if (e == std::string::npos || e > header_end) break;
            std::string h = raw.substr(p, e - p);
            size_t c = h.find(':');
            if (c != std::string::npos) {
                std::string k = h.substr(0, c), v = h.substr(c + 1);
                while (!v.empty() && v[0] == ' ') v.erase(v.begin());
                req.headers[k] = v;
            }
            p = e + 2;
        }
        if (header_end != std::string::npos && have_len) {
            size_t start = header_end + 4;
            req.body = raw.substr(start, content_len);
        }

        Conn conn(cfd);
        auto it = routes_.find(req.method + " " + req.path);
        if (it == routes_.end()) {
            // 兜底：静态资源用一条通配路由接住，不必为每个 .png / .svg 单独注册。
            // 没有兜底时仍然返回 404 JSON，行为与以前一致。
            if (fallback_) {
                try {
                    fallback_(req, conn);
                } catch (const std::exception& e) {
                    conn.send(500, "application/json",
                              std::string("{\"error\":\"") + e.what() + "\"}");
                }
            } else {
                conn.send(404, "application/json", "{\"error\":\"not found\"}");
            }
        } else {
            try {
                it->second(req, conn);
            } catch (const std::exception& e) {
                conn.send(500, "application/json",
                          std::string("{\"error\":\"") + e.what() + "\"}");
            }
        }
        conn.end_stream();
        ::close(cfd);
    }

    int fd_ = -1;
    std::atomic<bool> running_{false};
    std::map<std::string, Handler> routes_;
    Handler fallback_;
};

}  // namespace http
