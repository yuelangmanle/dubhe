#pragma once

#include <cmath>

#include "model_worker.h"
#include "store.h"

namespace scan {

constexpr size_t max_image_bytes = 8 * 1024 * 1024;

inline std::string_view trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    return first == text.npos ? std::string_view{} : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

inline bool valid_dimensions(std::string_view bytes, bool png) {
    auto byte = [&](size_t index) { return static_cast<uint32_t>(static_cast<unsigned char>(bytes[index])); };
    auto u16 = [&](size_t index) { return (byte(index) << 8) | byte(index + 1); };
    auto u32 = [&](size_t index) { return (u16(index) << 16) | u16(index + 2); };
    auto bounded = [](uint32_t width, uint32_t height) {
        return width && height && width <= 4096 && height <= 4096 && width * height <= 12000000;
    };
    if (png) {
        return bytes.size() >= 33 && bytes.starts_with("\x89PNG\r\n\x1a\n") &&
            u32(8) == 13 && bytes.substr(12, 4) == "IHDR" && bounded(u32(16), u32(20));
    }
    if (!bytes.starts_with("\xff\xd8")) return false;
    for (size_t pos = 2; pos + 4 <= bytes.size();) {
        if (byte(pos++) != 0xff) return false;
        while (pos < bytes.size() && byte(pos) == 0xff) ++pos;
        if (pos + 3 > bytes.size()) return false;
        const uint32_t marker = byte(pos++);
        if (marker == 0xda || marker == 0xd9) return false;  // Scan data before a frame header.
        const uint32_t size = u16(pos);
        if (size < 2 || size > bytes.size() - pos) return false;
        // The SOF markers store dimensions. DHT, JPG and DAC do not.
        if (marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc)
            return size >= 8 && bounded(u16(pos + 5), u16(pos + 3));
        pos += size;
    }
    return false;
}

inline bool valid_image(std::string_view image) {
    constexpr std::string_view png = "data:image/png;base64,";
    constexpr std::string_view jpeg = "data:image/jpeg;base64,";
    const bool is_png = image.starts_with(png);
    if (!is_png && !image.starts_with(jpeg)) return false;
    image.remove_prefix(is_png ? png.size() : jpeg.size());
    if (image.empty() || image.size() % 4 || image.size() > ((max_image_bytes + 2) / 3) * 4) return false;

    const size_t pad = image.ends_with("==") ? 2 : image.ends_with("=") ? 1 : 0;
    const size_t bytes = image.size() / 4 * 3 - pad;
    if (bytes > max_image_bytes) return false;
    static constexpr auto alphabet = [] {
        std::array<int, 256> table{};
        table.fill(-1);
        constexpr std::string_view base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < base64.size(); ++i) table[static_cast<unsigned char>(base64[i])] = static_cast<int>(i);
        return table;
    }();
    std::string decoded;
    decoded.reserve(bytes);
    uint32_t bits = 0;
    size_t pending = 0;
    for (const unsigned char ch : image.substr(0, image.size() - pad)) {
        const auto value = alphabet[ch];
        if (value < 0) return false;
        bits = (bits << 6) | static_cast<uint32_t>(value);
        pending += 6;
        if (pending >= 8) {
            pending -= 8;
            decoded.push_back(static_cast<char>((bits >> pending) & 0xff));
        }
    }
    return (bits & ((1u << pending) - 1)) == 0 && valid_dimensions(decoded, is_png);
}

// Xiaomi-OCR-0 emits OTSL tags. The grid walk retains rectangular spans and
// treats cell contents as text, including any markup that appears in a photo.
enum class CellTag { Filled, Empty, Left, Up, Both, NewRow };
struct Cell {
    CellTag tag;
    std::string_view text;
};

inline std::string_view otsl_tag_at(std::string_view text) {
    for (const std::string_view tag : {"<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"})
        if (text.starts_with(tag)) return tag;
    return {};
}

inline std::string escape_cell(std::string_view text) {
    std::string out;
    for (const char ch : text) {
        switch (ch) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '\n': out += "<br>"; break;
        case '\r': break;
        default: out += ch;
        }
    }
    return out;
}

inline std::string otsl_table(std::string_view input) {
    input = trim(input);
    if (otsl_tag_at(input).empty()) return {};
    std::vector<std::vector<Cell>> rows(1);
    size_t width = 0;
    bool has_cell = false;
    while (!input.empty()) {
        const auto tag = otsl_tag_at(input);
        if (tag.empty()) return {};
        size_t next = input.find('<', tag.size());
        while (next != input.npos && otsl_tag_at(input.substr(next)).empty())
            next = input.find('<', next + 1);
        const auto text = trim(input.substr(tag.size(), next == input.npos ? next : next - tag.size()));
        const auto kind = tag == "<fcel>" ? CellTag::Filled : tag == "<ecel>" ? CellTag::Empty :
            tag == "<lcel>" ? CellTag::Left : tag == "<ucel>" ? CellTag::Up :
            tag == "<xcel>" ? CellTag::Both : CellTag::NewRow;
        if (kind != CellTag::Filled && !text.empty()) return {};
        if (kind == CellTag::NewRow) {
            if (!rows.back().empty()) rows.emplace_back();
        } else {
            has_cell |= kind == CellTag::Filled || kind == CellTag::Empty;
            rows.back().push_back({kind, kind == CellTag::Filled ? text : std::string_view{}});
            width = std::max(width, rows.back().size());
        }
        if (width > 128 || rows.size() > 257) return {};
        if (next == input.npos) break;
        input.remove_prefix(next);
    }
    if (rows.back().empty()) rows.pop_back();
    if (!has_cell || rows.empty() || width * rows.size() > 8192) return {};
    for (auto& row : rows) row.resize(width, {CellTag::Empty, {}});
    std::vector<std::vector<bool>> covered(rows.size(), std::vector<bool>(width));
    std::string out = "<table>";
    for (size_t row = 0; row < rows.size(); ++row) {
        out += "<tr>";
        for (size_t col = 0; col < width; ++col) {
            if (covered[row][col]) continue;
            const auto& cell = rows[row][col];
            size_t across = 1, down = 1;
            if (cell.tag == CellTag::Filled || cell.tag == CellTag::Empty) {
                auto tag_at = [&](size_t r, size_t c, CellTag edge) {
                    return !covered[r][c] && (rows[r][c].tag == edge || rows[r][c].tag == CellTag::Both);
                };
                while (col + across < width && tag_at(row, col + across, CellTag::Left)) ++across;
                while (row + down < rows.size()) {
                    size_t k = 0;
                    while (k < across && tag_at(row + down, col + k, CellTag::Up)) ++k;
                    if (k != across) break;
                    ++down;
                }
            }
            for (size_t r = row; r < row + down; ++r)
                for (size_t c = col; c < col + across; ++c) covered[r][c] = true;
            out += "<td";
            if (across > 1) out += " colspan=\"" + std::to_string(across) + "\"";
            if (down > 1) out += " rowspan=\"" + std::to_string(down) + "\"";
            out += ">" + escape_cell(cell.text) + "</td>";
        }
        out += "</tr>";
    }
    return out + "</table>";
}

inline std::string document_tables(std::string_view document) {
    // Scan complete lines so that a table does not swallow neighboring prose.
    std::string result;
    auto line_at = [&](size_t start) {
        const auto end = document.find('\n', start);
        return end == document.npos ? document.size() : end + 1;
    };
    std::string_view ordinary_fence;
    for (size_t start = 0; start < document.size();) {
        size_t end = line_at(start);
        const auto line = trim(document.substr(start, end - start));
        std::string table;
        const auto fence = line.starts_with("```") || line.starts_with("~~~") ? line.substr(0, 3) : std::string_view{};
        if (ordinary_fence.empty() && !fence.empty() && trim(line.substr(3)) == "otsl") {
            size_t close = end;
            while (close < document.size()) {
                const size_t next = line_at(close);
                if (trim(document.substr(close, next - close)) == fence) {
                    table = otsl_table(document.substr(end, close - end));
                    if (!table.empty()) end = next;
                    break;
                }
                close = next;
            }
            if (table.empty()) ordinary_fence = fence;
        } else if (ordinary_fence.empty() && !otsl_tag_at(line).empty()) {
            while (end < document.size()) {
                const size_t next = line_at(end);
                if (otsl_tag_at(trim(document.substr(end, next - end))).empty()) break;
                end = next;
            }
            table = otsl_table(document.substr(start, end - start));
        } else if (!fence.empty()) {
            if (ordinary_fence.empty()) ordinary_fence = fence;
            else if (line == ordinary_fence) ordinary_fence = {};
        }
        if (table.empty()) result.append(document.substr(start, end - start));
        else result += "\n" + table + "\n\n";
        start = end;
    }
    return result;
}

class Service {
    static constexpr auto q4_file = "Xiaomi-OCR-0.Q4_K_M.gguf";
    static constexpr auto bf16_file = "Xiaomi-OCR-0.BF16.gguf";
    static constexpr auto vision_file = "Xiaomi-OCR-0.mmproj-bf16.gguf";

public:
    Service(std::string model_dir, std::string script)
        : model_dir_(std::move(model_dir)), script_(std::move(script)) {}

    json::Value options() const {
        auto result = json::Value::make_obj();
        auto files = json::Value::make_arr();
        bool vision_ready = false, q4_ready = false, bf16_ready = false;
        for (const auto& entry : store::catalog) {
            if (!entry.scan) continue;
            const bool present = installed(entry);
            const std::string_view filename = entry.file;
            if (filename == vision_file) vision_ready = present;
            if (filename == q4_file) q4_ready = present;
            if (filename == bf16_file) bf16_ready = present;
            auto file = json::Value::make_obj();
            file.set("id", json::Value::make_str(entry.id));
            file.set("name", json::Value::make_str(entry.name));
            file.set("path", json::Value::make_str(model_path(entry)));
            file.set("bytes", json::Value::make_num(static_cast<double>(entry.bytes)));
            file.set("installed", json::Value::make_bool(present));
            files.arr.push_back(std::move(file));
        }
        result.set("runtime_available", json::Value::make_bool(!ModelWorker::llama_runtime(script_).empty()));
        result.set("installed", json::Value::make_bool(vision_ready && (q4_ready || bf16_ready)));
        result.set("files", std::move(files));
        return result;
    }

    struct Result {
        int status;
        std::string text;
        std::string kind = {};
    };

    Result recognize(const json::Value& body, const std::function<bool()>& client_left) {
        const auto mode = body.s("mode");
        std::string instruction;
        std::string kind = "text";
        int max_tokens = 2048;
        auto schema = json::Value::make_obj();
        if (mode == "document") {
            instruction = "Convert the main body of this page to Markdown. Use reading order, omit running headers and "
                "footers, and represent formulas as LaTeX and actual tables as OTSL. "
                "Keep figure captions. Do not convert charts, plots or illustrations into tables or infer hidden values.";
            kind = "markdown";
            max_tokens = 6144;
        } else if (mode == "text") {
            instruction = "Extract the text in the image.";
        } else if (mode == "table") {
            instruction = "Parse the table in the image into OTSL.";
            kind = "markdown";
            max_tokens = 4096;
        } else if (mode == "formula") {
            instruction = "Identify the formula in the image and transcribe it in LaTeX format.";
            kind = "latex";
            max_tokens = 1024;
        } else if (mode == "extract") {
            const auto* fields = body.get("fields");
            if (!fields || fields->type != json::Value::ARR || fields->arr.empty() || fields->arr.size() > 20)
                return {400, "请提供 1 到 20 个需要提取的字段"};
            for (const auto& field : fields->arr) {
                const auto key = trim(field.str);
                if (field.type != json::Value::STR || key.empty() || key.size() > 96 ||
                    key.find_first_of("\r\n\t") != key.npos || schema.get(std::string(key)))
                    return {400, "请填写不同的字段名称，每个不超过 96 字节，且不能换行"};
                schema.set(std::string(key), json::Value::make_str(""));
            }
            instruction = "Extract key information in the image\n\n"
                "Please output the key information in JSON format according to the following schema:\n{\n";
            for (size_t i = 0; i < schema.obj.size(); ++i) {
                if (i) instruction += ",\n";
                instruction += "    " + json::dump(json::Value::make_str(schema.obj[i].first)) + ": \"\"";
            }
            instruction += "\n}\n\nCopy values only from the image. If a requested value is missing or unreadable, "
                "leave it as an empty string. Do not guess.";
            kind = "json";
        } else if (mode == "ask") {
            const auto question = body.s("question");
            if (trim(question).empty() || question.size() > 2048)
                return {400, "请输入一个关于图片的问题，长度不超过 2048 字节"};
            instruction = "Answer based only on the image. If the image does not show the answer, say so.\n" + question;
        } else {
            return {400, "请选择有效的扫描模式"};
        }

        const auto language = body.s("language");
        if (!language.empty() && language != "auto") {
            static const std::pair<std::string_view, std::string_view> languages[] = {
                {"zh", "Chinese"}, {"en", "English"}, {"mixed", "Chinese and English"},
                {"ja", "Japanese"}, {"ko", "Korean"},
            };
            for (const auto& [code, name] : languages) {
                if (language != code) continue;
                instruction += "\nExpected source language: ";
                instruction += name;
                instruction += ". Preserve the original text and language. Do not translate.";
                break;
            }
        }
        if (const auto* limit = body.get("max_tokens")) {
            if (limit->type != json::Value::NUM || limit->num < 512 || limit->num > 8192 ||
                std::floor(limit->num) != limit->num)
                return {400, "每页输出上限应为 512 到 8192 个 token"};
            max_tokens = static_cast<int>(limit->num);
        }

        const auto* image = body.get("image");
        if (!image || image->type != json::Value::STR || !valid_image(image->str))
            return {400, "扫描只接受 JPEG 或 PNG，最大 8 MiB、边长 4096 像素、总计 1200 万像素"};

        const auto variant = body.s("variant");
        if (!variant.empty() && variant != "q4" && variant != "bf16") return {400, "请选择有效的小米 OCR 版本"};
        const auto* projector = store::scan_file(vision_file);
        const auto* model = store::scan_file(variant == "bf16" ? bf16_file : q4_file);
        std::unique_lock lock(mu_, std::defer_lock);
        if (!lock.try_lock_for(std::chrono::milliseconds(200)))
            return {409, "正在扫描另一张图片，请等待完成或先停止"};
        if (client_left()) return {499, "扫描已停止"};
        if (variant.empty() && !installed(*model)) model = store::scan_file(bf16_file);
        if (!installed(*model) || !installed(*projector))
            return {404, "请先下载小米 OCR 模型和视觉组件"};
        if (ModelWorker::llama_runtime(script_).empty())
            return {503, "找不到扫描运行时，请使用带 llama.cpp 的 Mac 安装包"};

        if (!worker_ || worker_model_ != model) {
            worker_.reset();
            worker_ = std::make_unique<ModelWorker>(script_, "llama", model_path(*model), 16384, 4, model_path(*projector));
            worker_model_ = model;
        }
        std::string output, reason, error;
        dubhe_stats stats{};
        constexpr size_t max_output_bytes = 256 * 1024;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(4);
        auto cancelled = [&] {
            return client_left() || output.size() > max_output_bytes || std::chrono::steady_clock::now() >= deadline;
        };
        if (!worker_->generate(request(*image, std::move(instruction), max_tokens, schema),
                               [&](const std::string& piece) { output += piece; },
                               cancelled, &stats, &reason, &error)) {
            if (client_left()) return {499, "扫描已停止"};
            if (std::chrono::steady_clock::now() >= deadline) return {504, "扫描耗时过长，请缩小识别区域"};
            if (output.size() > max_output_bytes) return {422, "结果超出长度上限，请缩小识别区域"};
            return {502, error.empty() ? "扫描运行失败" : std::move(error)};
        }
        if (reason == "length" || output.size() > max_output_bytes)
            return {422, "结果达到长度上限。可提高输出上限、缩小识别区域，或使用单独的表格和公式模式"};
        auto decoded = trim(output);
        if (mode == "document" && decoded.starts_with("<md>") && decoded.ends_with("</md>"))
            decoded = trim(decoded.substr(4, decoded.size() - 9));
        std::string result(decoded);
        if (result.empty()) return {422, "未识别到内容，请检查图片清晰度或调整选取范围"};
        if (kind == "markdown") result = document_tables(result);
        if (kind == "json") {
            json::Value parsed;
            auto raw = std::string_view(result);
            if (raw.starts_with("```") && raw.ends_with("```")) {
                const auto newline = raw.find('\n');
                if (newline != raw.npos && newline < raw.size() - 3)
                    raw = trim(raw.substr(newline + 1, raw.size() - newline - 4));
            }
            if (!json::parse(std::string(raw), &parsed) || parsed.type != json::Value::OBJ)
                return {422, "模型未能整理成 JSON，请选择较清晰的区域后重试"};
            for (auto& [key, value] : schema.obj) {
                const auto* field = parsed.get(key);
                if (field && field->type == json::Value::STR) value = *field;
            }
            result = json::dump(schema);
        }
        return {200, std::move(result), std::move(kind)};
    }

    bool remove_file(const std::string& path, const std::function<void()>& remove) {
        if (!store::scan_file(std::filesystem::path(path).filename().string())) {
            remove();
            return true;
        }
        std::unique_lock lock(mu_, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        worker_.reset();
        remove();
        return true;
    }

private:
    std::string model_path(const store::Entry& item) const {
        return (std::filesystem::path(model_dir_) / item.file).string();
    }

    bool installed(const store::Entry& item) const {
        const auto path = model_path(item);
        return store::file_size(path) == item.bytes && store::gguf_header(path);
    }

    static json::Value request(const json::Value& image, std::string instruction, int max_tokens,
                               const json::Value& schema) {
        auto address = json::Value::make_obj();
        address.set("url", image);
        auto picture = json::Value::make_obj();
        picture.set("type", json::Value::make_str("image_url"));
        picture.set("image_url", std::move(address));
        auto text = json::Value::make_obj();
        text.set("type", json::Value::make_str("text"));
        text.set("text", json::Value::make_str(std::move(instruction)));
        auto content = json::Value::make_arr();
        content.arr.push_back(std::move(picture));
        content.arr.push_back(std::move(text));
        auto message = json::Value::make_obj();
        message.set("role", json::Value::make_str("user"));
        message.set("content", std::move(content));
        auto messages = json::Value::make_arr();
        messages.arr.push_back(std::move(message));
        auto job = json::Value::make_obj();
        job.set("messages", std::move(messages));
        job.set("no_think", json::Value::make_bool(true));
        job.set("temperature", json::Value::make_num(0));
        job.set("max_tokens", json::Value::make_num(max_tokens));
        if (!schema.obj.empty()) {
            auto fields = json::Value::make_arr();
            for (const auto& [key, value] : schema.obj) fields.arr.push_back(json::Value::make_str(key));
            job.set("ocr_fields", std::move(fields));
        }
        return job;
    }

    std::string model_dir_, script_;
    std::timed_mutex mu_;
    std::unique_ptr<ModelWorker> worker_;
    const store::Entry* worker_model_ = nullptr;
};

}  // namespace scan
