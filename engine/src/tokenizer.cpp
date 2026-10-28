// tokenizer.cpp — GPT-2 字节级 BPE + SentencePiece 兜底
#include "internal.h"

#include <algorithm>
#include <unordered_map>

namespace dubhe {

namespace {

std::string utf8_of(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// 取一个 UTF-8 字符的字节数
int utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

enum CharClass { C_SPACE, C_LETTER, C_DIGIT, C_PUNCT };

CharClass classify(const char* s, int len) {
    uint8_t c = (uint8_t)s[0];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return C_SPACE;
    if (c < 0x80) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return C_LETTER;
        if (c >= '0' && c <= '9') return C_DIGIT;
        return C_PUNCT;
    }
    (void)len;
    return C_LETTER;  // 非 ASCII 一律视作字母（CJK / 拉丁扩展 / emoji）
}

}  // namespace

void Tokenizer::build_byte_map() {
    // GPT-2 bytes_to_unicode：可打印字节映射成自身码位（注意是"码位"，
    // 161..255 必须按 UTF-8 编码写出，否则与 GGUF 里的词表字符串对不上），
    // 其余字节按出现顺序映射到 256+n。
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if (printable) {
            byte_to_char_[b] = utf8_of((uint32_t)b);
        } else {
            byte_to_char_[b] = utf8_of((uint32_t)(256 + n));
            n++;
        }
    }
    char_to_byte_.clear();
    char_to_byte_.reserve(256);
    for (int b = 0; b < 256; b++) char_to_byte_.emplace(byte_to_char_[b], (uint8_t)b);
}

bool Tokenizer::init(const GgufFile& f, std::string* err) {
    build_byte_map();

    auto* toks = f.get("tokenizer.ggml.tokens");
    if (!toks || toks->arr_str.empty()) {
        if (err) *err = "GGUF 缺少 tokenizer.ggml.tokens";
        return false;
    }
    id_to_token_ = toks->arr_str;
    token_to_id_.clear();
    token_to_id_.reserve(id_to_token_.size() * 2);
    for (int32_t i = 0; i < (int32_t)id_to_token_.size(); i++)
        token_to_id_.emplace(id_to_token_[i], i);

    std::string model = f.get_str("tokenizer.ggml.model", "gpt2");
    is_spm_ = (model == "llama" || model == "spm");

    if (auto* m = f.get("tokenizer.ggml.merges"); m) {
        for (auto& s : m->arr_str) {
            auto sp = s.find(' ');
            if (sp == std::string::npos) continue;
            merges_.emplace(std::make_pair(s.substr(0, sp), s.substr(sp + 1)), (int)merges_.size());
        }
    }

    // 特殊 token：type 3(CONTROL) / 4(USER_DEFINED)
    if (auto* tt = f.get("tokenizer.ggml.token_type"); tt) {
        for (int32_t i = 0; i < (int32_t)tt->arr_i32.size() && i < (int32_t)id_to_token_.size(); i++) {
            int t = tt->arr_i32[i];
            if (t == 3 || t == 4) specials_.push_back(i);
        }
    }
    // 长 token 优先匹配
    std::sort(specials_.begin(), specials_.end(), [&](int32_t a, int32_t b) {
        return id_to_token_[a].size() > id_to_token_[b].size();
    });

    chat_template_ = f.get_str("tokenizer.chat_template", "");

    bos_ = f.has("tokenizer.ggml.bos_token_id")
               ? (int32_t)f.get_u32("tokenizer.ggml.bos_token_id", 0) : -1;
    eos_ = f.has("tokenizer.ggml.eos_token_id")
               ? (int32_t)f.get_u32("tokenizer.ggml.eos_token_id", 0) : -1;
    add_bos_ = f.get_u32("tokenizer.ggml.add_bos_token", 0) != 0;

    auto it = token_to_id_.find("<|im_start|>");
    if (it != token_to_id_.end()) im_start_ = it->second;
    it = token_to_id_.find("<|im_end|>");
    if (it != token_to_id_.end()) im_end_ = it->second;

    return true;
}

int32_t Tokenizer::token_id(const std::string& piece) const {
    auto it = token_to_id_.find(piece);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::vector<int32_t> Tokenizer::bpe(const std::string& piece) const {
    // 关键：GPT-2 字节级 BPE 的初始符号是"单个字节"，不是"单个 UTF-8 字符"。
    // 一个汉字 = 3 个字节 = 3 个初始符号，再由 merges 逐级合并。
    std::vector<std::string> syms;
    syms.reserve(piece.size());
    for (unsigned char c : piece) syms.push_back(byte_to_char_[c]);
    if (syms.empty()) return {};

    // 快速路径：整段就是一个 token
    if (auto it = token_to_id_.find(piece); it != token_to_id_.end() && syms.size() > 1) {
        return {it->second};
    }

    while (syms.size() > 1) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = merges_.find({syms[i], syms[i + 1]});
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        const std::string a = syms[best_i], b = syms[best_i + 1];
        const std::string merged = a + b;
        std::vector<std::string> next;
        next.reserve(syms.size());
        for (size_t i = 0; i < syms.size();) {
            if (i + 1 < syms.size() && syms[i] == a && syms[i + 1] == b) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(syms[i]);
                i += 1;
            }
        }
        syms.swap(next);
    }

    std::vector<int32_t> out;
    out.reserve(syms.size());
    for (auto& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) out.push_back(it->second);
    }
    return out;
}

// 按 Qwen2 / GPT-2 的 pre-tokenizer 正则做近似切分：
//   (?i:'s|'t|...) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//    ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
void Tokenizer::pretokenize(const std::string& text, std::vector<std::string>* out) const {
    const size_t n = text.size();
    auto cl = [&](size_t p) {
        int l = utf8_len((uint8_t)text[p]);
        if (p + (size_t)l > n) l = 1;
        return classify(text.data() + p, l);
    };
    auto step = [&](size_t p) {
        int l = utf8_len((uint8_t)text[p]);
        if (p + (size_t)l > n) l = 1;
        return (size_t)l;
    };

    size_t i = 0;
    while (i < n) {
        CharClass c0 = cl(i);

        if (c0 == C_SPACE) {
            size_t j = i;
            while (j < n && cl(j) == C_SPACE) j += step(j);
            if (j < n) {
                CharClass c1 = cl(j);
                if (c1 == C_LETTER || c1 == C_DIGIT) {
                    // 空格串：前面的空格各自成块，最后一个空格并入后词
                    if (j - i > 1) out->emplace_back(text.substr(i, j - 1 - i));
                    size_t k = j;
                    if (c1 == C_LETTER) {
                        while (k < n && cl(k) == C_LETTER) k += step(k);
                    } else {
                        k += step(j);
                    }
                    out->emplace_back(text.substr(j - 1, k - (j - 1)));
                    i = k;
                    continue;
                }
                if (c1 == C_PUNCT && j - i == 1) {
                    // " ?[^\s\p{L}\p{N}]+[\r\n]*"
                    size_t k = j;
                    while (k < n && cl(k) == C_PUNCT) k += step(k);
                    while (k < n && (text[k] == '\n' || text[k] == '\r')) k++;
                    out->emplace_back(text.substr(i, k - i));
                    i = k;
                    continue;
                }
            }
            out->emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (c0 == C_LETTER) {
            size_t k = i;
            while (k < n && cl(k) == C_LETTER) k += step(k);
            out->emplace_back(text.substr(i, k - i));
            i = k;
            continue;
        }

        if (c0 == C_DIGIT) {
            // Qwen 用 \p{N}（单个数字），合并交给 merges 完成
            size_t l = step(i);
            out->emplace_back(text.substr(i, l));
            i += l;
            continue;
        }

        // C_PUNCT：先试「一个标点 + 连续字母」
        {
            size_t l = step(i);
            if (i + l < n && cl(i + l) == C_LETTER) {
                size_t k = i + l;
                while (k < n && cl(k) == C_LETTER) k += step(k);
                out->emplace_back(text.substr(i, k - i));
                i = k;
                continue;
            }
            size_t k = i;
            while (k < n && cl(k) == C_PUNCT) k += step(k);
            while (k < n && (text[k] == '\n' || text[k] == '\r')) k++;
            out->emplace_back(text.substr(i, k - i));
            i = k;
        }
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_special) const {
    std::vector<int32_t> out;
    if (add_special && add_bos_ && bos_ >= 0) out.push_back(bos_);

    const size_t n = text.size();
    size_t i = 0;
    std::string pending;  // 普通文本缓冲，攒够一个 piece 就 BPE

    auto flush = [&]() {
        if (pending.empty()) return;
        std::vector<std::string> pieces;
        pretokenize(pending, &pieces);
        for (auto& pc : pieces) {
            auto ids = bpe(pc);
            out.insert(out.end(), ids.begin(), ids.end());
        }
        pending.clear();
    };

    while (i < n) {
        // 1) 特殊 token 精确匹配
        bool matched = false;
        for (int32_t sid : specials_) {
            const std::string& s = id_to_token_[sid];
            if (s.empty() || s.size() > n - i) continue;
            if (text.compare(i, s.size(), s) == 0) {
                flush();
                out.push_back(sid);
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // 2) 普通文本：攒到下一个特殊 token 前，再统一 pre-tokenize
        size_t j = i;
        while (j < n) {
            bool hit = false;
            for (int32_t sid : specials_) {
                const std::string& s = id_to_token_[sid];
                if (!s.empty() && s.size() <= n - j && text.compare(j, s.size(), s) == 0) {
                    hit = true;
                    break;
                }
            }
            if (hit) break;
            j++;
        }
        pending.append(text, i, j - i);
        flush();
        i = j;
    }
    flush();
    return out;
}

std::string Tokenizer::decode_piece(int32_t id) const {
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return {};
    const std::string& s = id_to_token_[id];

    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        int l = utf8_len((uint8_t)s[i]);
        if (i + (size_t)l > s.size()) l = 1;
        auto it = char_to_byte_.find(s.substr(i, (size_t)l));
        if (it == char_to_byte_.end()) return s;  // 特殊 token（含非字节映射字符）原样返回
        out.push_back((char)it->second);
        i += (size_t)l;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (auto id : ids) out += decode_piece(id);
    return out;
}

}  // namespace dubhe
