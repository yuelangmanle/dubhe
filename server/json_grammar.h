#pragma once

#include "json.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

// 将 JSON 结构编为字节语法。采样前过滤候选 token，避免生成完才检查格式。
namespace json_grammar {

class Program {
    enum Op { Accept, Byte, Split, Call, Return };
    using Bytes = std::bitset<256>;
    struct Instruction { Op op; int next, alternate; Bytes bytes; };

public:
    explicit Program(const json::Value& schema) {
        constexpr int accept = 0;
        code_.push_back({Accept, 0, 0, {}});
        start_ = compile(schema, accept, 0);
    }

    static json::Value object_format() {
        json::Value schema = json::Value::make_obj();
        schema.set("type", json::Value::make_str("object"));
        return schema;
    }

    class Cursor {
    public:
        explicit Cursor(const Program& program) : program_(program) {
            add(current_, State{program_.start_});
        }
        bool complete() const {
            return std::any_of(current_.begin(), current_.end(), [&](const State& state) {
                return program_.code_[state.pc].op == Accept;
            });
        }
        bool allows(std::string_view token) {
            if (token.empty()) return false;
            const auto first = (unsigned char)token.front();
            if (std::none_of(current_.begin(), current_.end(), [&](const State& state) {
                    const auto& op = program_.code_[state.pc];
                    return op.op == Byte && op.bytes[first];
                })) return false;
            scratch_ = current_;
            for (const unsigned char byte : token) {
                step(scratch_, next_, byte);
                if (next_.empty()) return false;
                scratch_.swap(next_);
            }
            return true;
        }
        void consume(std::string_view token) {
            for (const unsigned char byte : token) {
                step(current_, next_, byte);
                current_.swap(next_);
            }
        }

    private:
        struct State {
            int pc;
            size_t depth = 0;
            std::array<int, 32> stack{};

            bool operator==(const State& other) const {
                return pc == other.pc && depth == other.depth &&
                    std::equal(stack.begin(), stack.begin() + depth, other.stack.begin());
            }
        };

        void add(std::vector<State>& out, State state) {
            const auto& op = program_.code_[state.pc];
            switch (op.op) {
                case Accept:
                case Byte:
                    if (std::find(out.begin(), out.end(), state) == out.end()) out.push_back(state);
                    return;
                case Split:
                    state.pc = op.alternate;
                    add(out, state);
                    break;
                case Call:
                    if (state.depth == state.stack.size()) return;
                    state.stack[state.depth++] = op.alternate;
                    break;
                case Return:
                    if (state.depth) {
                        state.pc = state.stack[--state.depth];
                        add(out, state);
                    }
                    return;
            }
            state.pc = op.next;
            add(out, state);
        }
        void step(const std::vector<State>& from, std::vector<State>& to, unsigned char byte) {
            to.clear();
            for (auto state : from) {
                const auto& op = program_.code_[state.pc];
                if (op.op != Byte || !op.bytes[byte]) continue;
                state.pc = op.next;
                add(to, state);
            }
        }

        const Program& program_;
        std::vector<State> current_, scratch_, next_;
    };

private:
    int emit(Op op, int next = 0, int alternate = 0, Bytes bytes = {}) {
        if (code_.size() >= 50000) throw std::invalid_argument("JSON Schema 太大");
        code_.push_back({op, next, alternate, bytes});
        return (int)code_.size() - 1;
    }
    static Bytes range(int first, int last) {
        Bytes bytes;
        for (int i = first; i <= last; i++) bytes.set(i);
        return bytes;
    }
    static Bytes chars(std::string_view text) {
        Bytes bytes;
        for (unsigned char ch : text) bytes.set(ch);
        return bytes;
    }
    int byte(Bytes values, int next) { return emit(Byte, next, 0, values); }
    int either(int a, int b) { return emit(Split, a, b); }
    int literal(std::string_view text, int next) {
        for (size_t pos = text.size(); pos > 0; --pos) {
            const auto ch = (unsigned char)text[pos - 1];
            next = byte(range(ch, ch), next);
        }
        return next;
    }
    int spaces(int next) {
        // 至多一个可选空格。避免模型把预算花在不影响 JSON 的换行和缩进上。
        return either(next, literal(" ", next));
    }
    int digits(int next) {
        const int loop = either(next, 0);
        const int digit = byte(range('0', '9'), loop);
        code_[loop].alternate = digit;
        return digit;
    }
    int number(int next, bool integer) {
        if (!integer) {
            const int e_digits = digits(next);
            const int exponent = byte(chars("eE"), either(e_digits, byte(chars("+-"), e_digits)));
            next = either(next, exponent);
            next = either(next, literal(".", digits(next)));
        }
        const int rest = either(next, digits(next));
        const int real = either(literal("0", next), byte(range('1', '9'), rest));
        return either(real, literal("-", real));
    }
    int string_atom(int next) {
        const auto cont = range(0x80, 0xbf);
        const int c1 = byte(cont, next);
        const int c2 = byte(cont, c1);
        const int c3 = byte(cont, c2);
        int ascii = byte(range(0x20, 0x21) | range(0x23, 0x5b) | range(0x5d, 0x7f), next);
        int utf8 = byte(range(0xc2, 0xdf), c1);
        utf8 = either(utf8, byte(range(0xe1, 0xec) | range(0xee, 0xef), c2));
        utf8 = either(utf8, byte(range(0xe0, 0xe0), byte(range(0xa0, 0xbf), c1)));
        utf8 = either(utf8, byte(range(0xed, 0xed), byte(range(0x80, 0x9f), c1)));
        utf8 = either(utf8, byte(range(0xf1, 0xf3), c3));
        utf8 = either(utf8, byte(range(0xf0, 0xf0), byte(range(0x90, 0xbf), c2)));
        utf8 = either(utf8, byte(range(0xf4, 0xf4), byte(range(0x80, 0x8f), c2)));
        const auto hex = range('0', '9') | range('a', 'f') | range('A', 'F');
        const int last_two = byte(hex, byte(hex, next));
        const int unicode = either(byte(hex & ~chars("dD"), byte(hex, last_two)),
                                   byte(chars("dD"), byte(range('0', '7'), last_two)));
        const int escape = literal("\\", either(byte(chars("\"\\/bfnrt"), next), literal("u", unicode)));
        return either(ascii, either(utf8, escape));
    }
    int string(int next, int minimum = 0, int maximum = -1) {
        const int close = literal("\"", next);
        int body = close;
        if (maximum < 0) {
            body = either(close, 0);
            const int atom = string_atom(body);
            code_[body].alternate = atom;
        } else {
            for (int i = minimum; i < maximum; i++) body = either(close, string_atom(body));
        }
        for (int i = 0; i < minimum; i++) body = string_atom(body);
        return literal("\"", body);
    }
    int array(int item, int next, int minimum, int maximum) {
        const int close = spaces(literal("]", next));
        if (maximum == 0) return literal("[", close);
        auto entry = [&](int rest) { return spaces(emit(Call, item, rest)); };
        auto tail = [&](int rest) { return spaces(literal(",", entry(rest))); };
        int rest = close;
        const int required = std::max(1, minimum);
        if (maximum < 0) {
            rest = either(close, 0);
            const int more = tail(rest);
            code_[rest].alternate = more;
        } else {
            for (int i = required; i < maximum; i++) rest = either(close, tail(rest));
        }
        for (int i = 1; i < required; i++) rest = tail(rest);
        const int body = entry(rest);
        return literal("[", minimum == 0 ? either(close, body) : body);
    }
    int any() {
        if (any_ >= 0) return any_;
        any_ = either(0, 0);
        if (return_ < 0) return_ = emit(Return);
        const int next = return_;
        const int a = array(any_, next, 0, -1);
        const int close = spaces(literal("}", next));
        const int more = either(close, 0);
        const int value = spaces(emit(Call, any_, more));
        const int member = spaces(string(spaces(literal(":", value))));
        const int again = spaces(literal(",", member));
        code_[more].alternate = again;
        any_object_ = literal("{", either(close, member));
        const int primitive = either(string(next), either(number(next, false),
            either(literal("null", next), either(literal("true", next), literal("false", next)))));
        code_[any_].next = primitive;
        code_[any_].alternate = either(a, any_object_);
        return any_;
    }
    static int bound(const json::Value& schema, const char* key, int fallback, int maximum) {
        const auto* value = schema.get(key);
        if (!value) return fallback;
        if (value->type != json::Value::NUM || !(value->num >= 0 && value->num <= maximum) ||
            value->num != std::floor(value->num)) throw std::invalid_argument(std::string("无效的 ") + key);
        return (int)value->num;
    }
    static bool is_integer(const json::Value& value) {
        if (value.type != json::Value::NUM) return false;
        if (value.str.empty()) return std::floor(value.num) == value.num;
        const size_t exponent_at = value.str.find_first_of("eE");
        const auto mantissa = std::string_view(value.str).substr(0, exponent_at);
        const size_t nonzero = mantissa.find_last_of("123456789");
        if (nonzero == std::string_view::npos) return true;
        const size_t point = mantissa.find('.');
        const size_t integer_digits = point == std::string_view::npos ? mantissa.size() : point;
        const auto needed = (int64_t)nonzero - (int64_t)integer_digits +
            (point == std::string_view::npos || nonzero < point ? 1 : 0);
        const long long exponent = exponent_at == std::string::npos ? 0 :
            std::strtoll(value.str.c_str() + exponent_at + 1, nullptr, 10);
        return exponent >= needed;
    }
    static bool matches_type(const json::Value& value, const std::string& type) {
        return type.empty() ||
            (type == "object" && value.type == json::Value::OBJ) ||
            (type == "array" && value.type == json::Value::ARR) ||
            (type == "string" && value.type == json::Value::STR) ||
            (type == "null" && value.type == json::Value::NUL) ||
            (type == "boolean" && value.type == json::Value::BOOL) ||
            (type == "number" && value.type == json::Value::NUM) ||
            (type == "integer" && is_integer(value));
    }
    int compile(const json::Value& schema, int next, int depth) {
        if (depth > 24 || schema.type != json::Value::OBJ) throw std::invalid_argument("JSON Schema 嵌套或类型无效");
        const auto* options = schema.get("anyOf");
        const auto* enum_values = schema.get("enum");
        const auto* constant = schema.get("const");
        const std::string type = schema.s("type");
        static constexpr std::string_view supported[] = {
            "$schema", "title", "description", "default", "examples", "type", "anyOf", "enum", "const",
            "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
            "minLength", "maxLength",
        };
        for (const auto& [key, value] : schema.obj) {
            if (std::find(std::begin(supported), std::end(supported), key) == std::end(supported))
                throw std::invalid_argument("不支持的 JSON Schema 字段 " + key);
            const bool annotation = key == "$schema" || key == "title" || key == "description" ||
                key == "default" || key == "examples";
            if (!annotation && key != "anyOf" && options)
                throw std::invalid_argument("anyOf 请放在独立的子结构中");
            if (!annotation && key != "enum" && key != "const" && key != "type" &&
                (enum_values || constant))
                throw std::invalid_argument("enum/const 的子结构仅支持类型和说明");
        }
        if (return_ < 0) return_ = emit(Return);
        if (schema.get("type") && type.empty()) throw std::invalid_argument("JSON Schema type 必须是类型名，多个类型请用 anyOf");
        if (options || enum_values || constant) {
            int first = -1;
            auto include = [&](const json::Value& value) {
                const int rule = options ? compile(value, next, depth + 1) : literal(json::dump(value), next);
                first = first < 0 ? rule : either(first, rule);
            };
            if (constant && enum_values) throw std::invalid_argument("请为该字段选择 const 或 enum");
            if (constant && matches_type(*constant, type)) include(*constant);
            if (const auto* values = options ? options : enum_values) {
                if (values->type != json::Value::ARR || values->arr.size() > 64)
                    throw std::invalid_argument("JSON Schema 选项应为不超过 64 项的数组");
                for (const auto& option : values->arr)
                    if (options || matches_type(option, type)) include(option);
            }
            if (first < 0) throw std::invalid_argument("JSON Schema 没有可选值");
            return first;
        }
        const bool object_fields = schema.get("properties") || schema.get("required") || schema.get("additionalProperties");
        const bool array_fields = schema.get("items") || schema.get("minItems") || schema.get("maxItems");
        const bool string_fields = schema.get("minLength") || schema.get("maxLength");
        if ((object_fields && type != "object") || (array_fields && type != "array") ||
            (string_fields && type != "string"))
            throw std::invalid_argument("JSON Schema 字段与 type 不匹配");
        if (type.empty()) return emit(Call, any(), next);
        if (type == "null") return literal("null", next);
        if (type == "boolean") return either(literal("true", next), literal("false", next));
        if (type == "integer" || type == "number") return number(next, type == "integer");
        if (type == "string") {
            const int min = bound(schema, "minLength", 0, 256);
            const int max = bound(schema, "maxLength", -1, 256);
            if (max >= 0 && max < min) throw std::invalid_argument("maxLength 小于 minLength");
            return string(next, min, max);
        }
        if (type == "array") {
            const int min = bound(schema, "minItems", 0, 256);
            const int max = bound(schema, "maxItems", -1, 256);
            if (max >= 0 && max < min) throw std::invalid_argument("maxItems 小于 minItems");
            const auto* item = schema.get("items");
            return array(item ? compile(*item, return_, depth + 1) : any(), next, min, max);
        }
        if (type != "object") throw std::invalid_argument("不支持的 JSON Schema 类型 " + type);

        const auto* props = schema.get("properties");
        const auto* required = schema.get("required");
        const auto* extra = schema.get("additionalProperties");
        if (!props && (!required || (required->type == json::Value::ARR && required->arr.empty())) &&
            (!extra || (extra->type == json::Value::BOOL && extra->b))) {
            // 复用通用对象入口，仅允许顶层为对象。
            any();
            return emit(Call, any_object_, next);
        }
        if ((props && (props->type != json::Value::OBJ || props->obj.size() > 64)) ||
            (required && required->type != json::Value::ARR) ||
            (extra && extra->type != json::Value::BOOL))
            throw std::invalid_argument("JSON Schema 对象的属性定义无效");
        if (extra && extra->b)
            throw std::invalid_argument("当前约束输出的已声明对象字段不支持 additionalProperties: true");
        if (required) {
            for (const auto& name : required->arr)
                if (name.type != json::Value::STR || !props || !props->get(name.str))
                    throw std::invalid_argument("required 中的字段必须有 properties 定义");
        }
        int empty = spaces(literal("}", next)), populated = empty;
        if (props) {
            for (auto it = props->obj.rbegin(); it != props->obj.rend(); ++it) {
                const auto& [name, prop] = *it;
                const bool mandatory = required && std::any_of(required->arr.begin(), required->arr.end(),
                    [&](const json::Value& key) { return key.str == name; });
                const int value = compile(prop, return_, depth + 1);
                const int head = spaces(literal(json::dump(json::Value::make_str(name)),
                    spaces(literal(":", spaces(emit(Call, value, populated))))));
                const int with_comma = spaces(literal(",", head));
                empty = mandatory ? head : either(empty, head);
                populated = mandatory ? with_comma : either(populated, with_comma);
            }
        }
        return literal("{", empty);
    }

    std::vector<Instruction> code_;
    int start_ = 0, return_ = -1, any_ = -1, any_object_ = -1;
};

}  // namespace json_grammar
