// json.h — 极简 JSON（够用即可，不追求完整规范）
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace json {

struct Value;
using Object = std::vector<std::pair<std::string, Value>>;  // 保序
using Array = std::vector<Value>;

struct Value {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    Array arr;
    Object obj;

    static Value make_obj() { Value v; v.type = OBJ; return v; }
    static Value make_arr() { Value v; v.type = ARR; return v; }
    static Value make_str(std::string s) { Value v; v.type = STR; v.str = std::move(s); return v; }
    static Value make_num(double d) { Value v; v.type = NUM; v.num = d; return v; }
    static Value make_bool(bool x) { Value v; v.type = BOOL; v.b = x; return v; }

    const Value* get(const std::string& k) const {
        if (type != OBJ) return nullptr;
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    void set(const std::string& k, Value v) {
        if (type != OBJ) { type = OBJ; }
        for (auto& kv : obj) if (kv.first == k) { kv.second = std::move(v); return; }
        obj.emplace_back(k, std::move(v));
    }
    std::string s(const std::string& k, const std::string& def = "") const {
        auto* v = get(k);
        return (v && v->type == STR) ? v->str : def;
    }
    double n(const std::string& k, double def = 0) const {
        auto* v = get(k);
        return (v && v->type == NUM) ? v->num : def;
    }
    bool flag(const std::string& k, bool def = false) const {
        auto* v = get(k);
        return (v && v->type == BOOL) ? v->b : def;
    }
};

// ---------------------------------------------------------------- 解析
class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}
    bool parse(Value* out) {
        skip();
        if (!value(out)) return false;
        skip();
        return p_ >= s_.size();
    }
    const std::string& error() const { return err_; }

private:
    void skip() { while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\r' || s_[p_] == '\t')) p_++; }
    bool eat(char c) { skip(); if (p_ < s_.size() && s_[p_] == c) { p_++; return true; } return false; }
    bool fail(const char* m) { if (err_.empty()) err_ = m; return false; }

    bool value(Value* v) {
        skip();
        if (p_ >= s_.size()) return fail("unexpected end");
        char c = s_[p_];
        if (c == '{') return object(v);
        if (c == '[') return array(v);
        if (c == '"') { v->type = Value::STR; return string(&v->str); }
        if (c == 't' || c == 'f') return boolean(v);
        if (c == 'n') return null_value(v);
        return number(v);
    }
    bool null_value(Value* v) {
        if (s_.compare(p_, 4, "null") == 0) { p_ += 4; v->type = Value::NUL; return true; }
        return fail("bad literal");
    }
    bool boolean(Value* v) {
        if (s_.compare(p_, 4, "true") == 0) { p_ += 4; v->type = Value::BOOL; v->b = true; return true; }
        if (s_.compare(p_, 5, "false") == 0) { p_ += 5; v->type = Value::BOOL; v->b = false; return true; }
        return fail("bad bool");
    }
    bool number(Value* v) {
        size_t start = p_;
        if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) p_++;
        while (p_ < s_.size() && (isdigit((unsigned char)s_[p_]) || s_[p_] == '.' || s_[p_] == 'e' ||
                                  s_[p_] == 'E' || s_[p_] == '-' || s_[p_] == '+')) p_++;
        if (p_ == start) return fail("bad number");
        v->type = Value::NUM;
        v->num = std::strtod(s_.substr(start, p_ - start).c_str(), nullptr);
        return true;
    }
    bool string(std::string* out) {
        if (!eat('"')) return fail("expected string");
        out->clear();
        while (p_ < s_.size()) {
            char c = s_[p_++];
            if (c == '"') return true;
            if (c != '\\') { *out += c; continue; }
            if (p_ >= s_.size()) break;
            char e = s_[p_++];
            switch (e) {
                case '"': *out += '"'; break;
                case '\\': *out += '\\'; break;
                case '/': *out += '/'; break;
                case 'b': *out += '\b'; break;
                case 'f': *out += '\f'; break;
                case 'n': *out += '\n'; break;
                case 'r': *out += '\r'; break;
                case 't': *out += '\t'; break;
                case 'u': {
                    if (p_ + 4 > s_.size()) return fail("bad \\u");
                    unsigned cp = (unsigned)std::strtoul(s_.substr(p_, 4).c_str(), nullptr, 16);
                    p_ += 4;
                    // surrogate pair
                    if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 6 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
                        unsigned lo = (unsigned)std::strtoul(s_.substr(p_ + 2, 4).c_str(), nullptr, 16);
                        p_ += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: *out += e; break;
            }
        }
        return fail("unterminated string");
    }
    static void append_utf8(std::string* o, unsigned cp) {
        if (cp < 0x80) *o += (char)cp;
        else if (cp < 0x800) { *o += (char)(0xC0 | (cp >> 6)); *o += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { *o += (char)(0xE0 | (cp >> 12)); *o += (char)(0x80 | ((cp >> 6) & 0x3F)); *o += (char)(0x80 | (cp & 0x3F)); }
        else { *o += (char)(0xF0 | (cp >> 18)); *o += (char)(0x80 | ((cp >> 12) & 0x3F)); *o += (char)(0x80 | ((cp >> 6) & 0x3F)); *o += (char)(0x80 | (cp & 0x3F)); }
    }
    bool array(Value* v) {
        if (!eat('[')) return fail("expected [");
        v->type = Value::ARR;
        skip();
        if (eat(']')) return true;
        while (true) {
            Value item;
            if (!value(&item)) return false;
            v->arr.push_back(std::move(item));
            skip();
            if (eat(',')) continue;
            if (eat(']')) return true;
            return fail("expected , or ]");
        }
    }
    bool object(Value* v) {
        if (!eat('{')) return fail("expected {");
        v->type = Value::OBJ;
        skip();
        if (eat('}')) return true;
        while (true) {
            std::string key;
            if (!string(&key)) return false;
            if (!eat(':')) return fail("expected :");
            Value item;
            if (!value(&item)) return false;
            v->obj.emplace_back(std::move(key), std::move(item));
            skip();
            if (eat(',')) continue;
            if (eat('}')) return true;
            return fail("expected , or }");
        }
    }

    const std::string& s_;
    size_t p_ = 0;
    std::string err_;
};

inline bool parse(const std::string& text, Value* out) {
    Parser p(text);
    return p.parse(out);
}

// ---------------------------------------------------------------- 序列化
//
// 字符串原样输出，但 >= 0x80 的字节要先验一遍 UTF-8 —— 前端是 JSON.parse，
// 收到半个汉字（比如按字节截断的 snippet，或者 GBK 编码的文本文件）会直接
// 抛异常，整个响应就废了。非法序列替换成 U+FFFD，宁可显示一个"�"。
inline void escape_to(std::string* o, const std::string& s) {
    *o += '"';
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"': *o += "\\\""; break;
            case '\\': *o += "\\\\"; break;
            case '\n': *o += "\\n"; break;
            case '\r': *o += "\\r"; break;
            case '\t': *o += "\\t"; break;
            case '\b': *o += "\\b"; break;
            case '\f': *o += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    *o += b;
                } else if (c < 0x80) {
                    *o += (char)c;
                } else {
                    const int n = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 0;
                    bool ok = n > 0 && i + (size_t)n <= s.size();
                    if (ok) {
                        for (int k = 1; k < n; k++) {
                            if (((unsigned char)s[i + (size_t)k] & 0xC0) != 0x80) { ok = false; break; }
                        }
                    }
                    if (!ok) {
                        *o += "\xEF\xBF\xBD";
                    } else {
                        *o += s.substr(i, (size_t)n);
                        i += (size_t)n - 1;
                    }
                }
        }
    }
    *o += '"';
}

inline void dump_to(std::string* o, const Value& v) {
    switch (v.type) {
        case Value::NUL: *o += "null"; break;
        case Value::BOOL: *o += v.b ? "true" : "false"; break;
        case Value::NUM: {
            if (std::isfinite(v.num) && v.num == (long long)v.num && std::fabs(v.num) < 1e15) {
                char b[32]; std::snprintf(b, sizeof(b), "%lld", (long long)v.num); *o += b;
            } else {
                char b[64]; std::snprintf(b, sizeof(b), "%.6g", v.num); *o += b;
            }
            break;
        }
        case Value::STR: escape_to(o, v.str); break;
        case Value::ARR: {
            *o += '[';
            for (size_t i = 0; i < v.arr.size(); i++) { if (i) *o += ','; dump_to(o, v.arr[i]); }
            *o += ']';
            break;
        }
        case Value::OBJ: {
            *o += '{';
            for (size_t i = 0; i < v.obj.size(); i++) {
                if (i) *o += ',';
                escape_to(o, v.obj[i].first);
                *o += ':';
                dump_to(o, v.obj[i].second);
            }
            *o += '}';
            break;
        }
    }
}

inline std::string dump(const Value& v) {
    std::string s;
    s.reserve(256);
    dump_to(&s, v);
    return s;
}

}  // namespace json
