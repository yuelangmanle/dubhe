// calc.h — 给模型用的安全算术求值器
//
// 为什么不用 eval / system("python3 -c ...")：
//   模型生成的表达式会直接进这里，哪怕有白名单也挡不住奇怪的边界情况；
//   而且拉起一个 Python 进程要几十毫秒，算术工具本该是微秒级的。
//
// 这里是一个递归下降解析器，只认数字、四则、括号和一张固定的函数表。
// 任何不认识的字符都在 parse 阶段报错，不存在"执行代码"的可能。
//
// 语法（优先级从低到高）：
//   expr    := term (('+' | '-') term)*
//   term    := unary (('*' | '/' | '%') unary)*
//   unary   := ('+' | '-')? power
//   power   := primary ('^' unary)?        // 右结合，2^3^2 = 512
//   primary := number | '(' expr ')' | func '(' expr (',' expr)? ')' | const
//
// 用法：
//   CalcResult r = calc_eval("376*49");
//   r.ok ? r.value : r.error
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

namespace dubhe_calc {

struct CalcResult {
    bool ok = false;
    double value = 0;
    std::string error;
    std::string text;       // 已经格式化好、可以直接塞进 prompt 的结果
};

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    bool parse(double* out, std::string* err) {
        skip();
        if (eof()) { set_err(err, "表达式是空的"); return false; }
        const double v = expr();
        if (!ok_) { set_err(err, error_); return false; }
        skip();
        if (!eof()) { set_err(err, "有多余的内容: " + s_.substr(pos_)); return false; }
        if (!std::isfinite(v)) { set_err(err, "结果不是有限数（可能除以 0 或溢出）"); return false; }
        *out = v;
        return true;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;
    bool ok_ = true;
    std::string error_;

    void fail(const std::string& msg) {
        if (ok_) { ok_ = false; error_ = msg; }
    }
    static void set_err(std::string* err, const std::string& msg) { if (err) *err = msg; }
    bool eof() const { return pos_ >= s_.size(); }
    char peek() const { return eof() ? '\0' : s_[pos_]; }

    void skip() {
        while (!eof() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                          s_[pos_] == '\r' || s_[pos_] == '\n')) pos_++;
    }
    bool eat(char c) {
        skip();
        if (peek() == c) { pos_++; return true; }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("缺少 '") + c + "'");
    }

    double expr() {
        double v = term();
        for (;;) {
            skip();
            const char c = peek();
            if (c != '+' && c != '-') break;
            pos_++;
            const double rhs = term();
            if (!ok_) return 0;
            v = (c == '+') ? v + rhs : v - rhs;
        }
        return v;
    }

    double term() {
        double v = unary();
        for (;;) {
            skip();
            const char c = peek();
            if (c != '*' && c != '/' && c != '%') break;
            pos_++;
            const double rhs = unary();
            if (!ok_) return 0;
            if ((c == '/' || c == '%') && rhs == 0) { fail("除数不能为 0"); return 0; }
            if (c == '*') v *= rhs;
            else if (c == '/') v /= rhs;
            else v = std::fmod(v, rhs);
        }
        return v;
    }

    double unary() {
        skip();
        if (peek() == '+') { pos_++; return unary(); }
        if (peek() == '-') { pos_++; return -unary(); }
        return power();
    }

    double power() {
        double base = primary();
        if (!ok_) return 0;
        skip();
        if (peek() != '^') return base;
        pos_++;
        const double e = unary();          // 右结合
        if (!ok_) return 0;
        if (base < 0 && std::fmod(e, 1.0) != 0) { fail("负数不能开偶次方"); return 0; }
        const double v = std::pow(base, e);
        if (!std::isfinite(v)) { fail("幂运算溢出"); return 0; }
        return v;
    }

    // 读一个标识符（函数名 / 常量名）
    std::string ident() {
        skip();
        const size_t start = pos_;
        while (!eof() && ((s_[pos_] >= 'a' && s_[pos_] <= 'z') ||
                          (s_[pos_] >= 'A' && s_[pos_] <= 'Z'))) pos_++;
        return s_.substr(start, pos_ - start);
    }

    double primary() {
        skip();
        if (eof()) { fail("表达式不完整"); return 0; }

        if (peek() == '(') {
            pos_++;
            const double v = expr();
            expect(')');
            return v;
        }

        const char c = peek();
        if ((c >= '0' && c <= '9') || c == '.') {
            size_t start = pos_;
            while (!eof() && s_[pos_] >= '0' && s_[pos_] <= '9') pos_++;
            if (!eof() && s_[pos_] == '.') {
                pos_++;
                while (!eof() && s_[pos_] >= '0' && s_[pos_] <= '9') pos_++;
            }
            if (!eof() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
                const size_t save = pos_;
                pos_++;
                if (!eof() && (s_[pos_] == '+' || s_[pos_] == '-')) pos_++;
                const size_t digs = pos_;
                while (!eof() && s_[pos_] >= '0' && s_[pos_] <= '9') { pos_++; }
                if (pos_ == digs) pos_ = save;
            }
            const std::string num = s_.substr(start, pos_ - start);
            try { return std::stod(num); }
            catch (...) { fail("数字格式不对: " + num); return 0; }
        }

        // 下面必须是标识符：函数名或常量
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
            fail(std::string("不认识的字符: '") + c + "'");
            return 0;
        }
        const std::string name = ident();
        if (name == "pi" || name == "PI") return 3.14159265358979323846;
        if (name == "e"  || name == "E")  return 2.71828182845904523536;

        if (!eat('(')) { fail("不认识的标识符: " + name); return 0; }
        const double a = expr();
        double b = 0;
        bool two = false;
        if (eat(',')) { b = expr(); two = true; }
        expect(')');
        if (!ok_) return 0;

        if (name == "abs")   return std::fabs(a);
        if (name == "sqrt")  { if (a < 0) { fail("sqrt 的参数不能是负数"); return 0; } return std::sqrt(a); }
        if (name == "floor") return std::floor(a);
        if (name == "ceil")  return std::ceil(a);
        if (name == "round") return std::round(a);
        if (name == "exp")   return std::exp(a);
        if (name == "ln")    { if (a <= 0) { fail("ln 的参数必须大于 0"); return 0; } return std::log(a); }
        if (name == "log10") { if (a <= 0) { fail("log10 的参数必须大于 0"); return 0; } return std::log10(a); }
        if (name == "sin")   return std::sin(a);
        if (name == "cos")   return std::cos(a);
        if (name == "tan")   return std::tan(a);
        if (name == "pow")   { if (!two) { fail("pow 需要两个参数"); return 0; } return std::pow(a, b); }
        if (name == "min")   { if (!two) { fail("min 需要两个参数"); return 0; } return std::fmin(a, b); }
        if (name == "max")   { if (!two) { fail("max 需要两个参数"); return 0; } return std::fmax(a, b); }
        fail("不支持的函数: " + name);
        return 0;
    }
};

inline std::string format(double v) {
    if (v == 0) return "0";
    // %.12g 在 376×49 这种整数上是精确的；大数会转科学计数法，
    // 但至少不会把 18424 写成 18423.999999999996。
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.12g", v);
    return std::string(buf);
}

}  // namespace detail

inline CalcResult calc_eval(const std::string& expression) {
    CalcResult r;
    // 超长表达式基本都是模型跑飞了，直接挡掉，别让它吃掉一个工具轮
    if (expression.size() > 512) {
        r.error = "表达式太长（上限 512 字符）";
        return r;
    }
    detail::Parser p(expression);
    double v = 0;
    if (!p.parse(&v, &r.error)) return r;
    r.ok = true;
    r.value = v;
    r.text = detail::format(v);
    return r;
}

}  // namespace dubhe_calc
