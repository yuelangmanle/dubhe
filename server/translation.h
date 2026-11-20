#pragma once

#include "model_worker.h"
#include "store.h"

namespace translation {

inline constexpr const char* default_model = "hy-mt2-18b-q4";

inline std::string_view language_name(std::string_view code) {
    if (code == "zh") return "简体中文";
    if (code == "en") return "英语";
    if (code == "ja") return "日语";
    if (code == "ko") return "韩语";
    if (code == "fr") return "法语";
    if (code == "de") return "德语";
    return "";
}

class Service {
public:
    Service(std::string model_dir, std::string script)
        : model_dir_(std::move(model_dir)), script_(std::move(script)) {}

    json::Value options() const {
        auto result = json::Value::make_obj();
        auto models = json::Value::make_arr();
        for (const auto& item : store::catalog) {
            if (!item.translation) continue;
            auto model = json::Value::make_obj();
            model.set("id", json::Value::make_str(item.id));
            model.set("name", json::Value::make_str(item.name));
            model.set("installed", json::Value::make_bool(installed(item)));
            models.arr.push_back(std::move(model));
        }
        result.set("models", std::move(models));
        result.set("default_model", json::Value::make_str(default_model));
        result.set("runtime_available", json::Value::make_bool(!ModelWorker::llama_runtime(script_).empty()));
        return result;
    }

    struct Result {
        int status;
        std::string text;
    };

    Result translate(const json::Value& body, const std::function<bool()>& cancelled) {
        const auto id = body.s("model", default_model);
        const store::Entry* model = nullptr;
        for (const auto& entry : store::catalog)
            if (entry.translation && id == entry.id) model = &entry;
        if (!model) return {400, "未知的翻译模型"};
        const std::string source_code = body.s("source", "auto");
        const auto source = language_name(source_code), target = language_name(body.s("target"));
        if (target.empty() || (source_code != "auto" && source.empty()))
            return {400, "请选择有效的原文和目标语言"};
        const std::string text = body.s("text");
        if (text.size() > 12288 || text.find_first_not_of(" \t\r\n") == std::string::npos)
            return {400, "单段原文不能为空，且不能超过 12 KB"};

        std::vector<std::pair<std::string, std::string>> glossary;
        if (const auto* terms = body.get("glossary")) {
            if (terms->type != json::Value::ARR || terms->arr.size() > 64)
                return {400, "术语表最多支持 64 项"};
            for (const auto& term : terms->arr) {
                auto from = term.s("source"), to = term.s("target");
                if (from.empty() || to.empty() || from.size() > 256 || to.size() > 256 ||
                    from.find_first_of("\r\n") != std::string::npos ||
                    to.find_first_of("\r\n") != std::string::npos)
                    return {400, "每个术语需要原文和译文，且应写在同一行"};
                if (text.find(from) != std::string::npos)
                    glossary.emplace_back(std::move(from), std::move(to));
            }
        }
        const auto request = prompt(*model, source, target, text, glossary, body.flag("formatted"));

        std::unique_lock lock(mu_, std::defer_lock);
        while (!cancelled() && !lock.try_lock_for(std::chrono::milliseconds(100))) {}
        if (!lock.owns_lock() || cancelled()) return {499, "翻译已取消"};
        if (!installed(*model)) return {404, "请先在模型商店下载所选翻译模型"};
        if (script_.empty() || ModelWorker::llama_runtime(script_).empty())
            return {503, "找不到本机翻译运行时。请使用带翻译功能的 Mac 安装包，或安装 llama.cpp"};
        const auto path = model_path(*model);
        if (cached_path_ != path) {
            worker_.reset();
            worker_ = std::make_unique<ModelWorker>(script_, "llama", path, 8192, 4);
            cached_path_ = path;
        }

        std::string output, reason, error;
        dubhe_stats stats{};
        if (!worker_->generate(request, [&](const std::string& piece) { output += piece; },
                               cancelled, &stats, &reason, &error)) {
            if (cancelled()) return {499, "翻译已取消"};
            return {502, error.empty() ? "翻译运行失败" : std::move(error)};
        }
        if (reason == "length") return {422, "译文达到长度上限，请缩短这一段原文后重试"};
        const auto start = output.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return {502, "模型没有输出译文"};
        return {200, output.substr(start, output.find_last_not_of(" \t\r\n") - start + 1)};
    }

    bool remove_file(const std::string& path, const std::function<void()>& remove) {
        const auto filename = std::filesystem::path(path).filename().string();
        if (!store::translation_file(filename)) { remove(); return true; }
        std::unique_lock lock(mu_, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        if (path == cached_path_) {
            worker_.reset();
            cached_path_.clear();
        }
        remove();
        return true;
    }

private:
    std::string model_path(const store::Entry& item) const {
        return (std::filesystem::path(model_dir_) / item.file).string();
    }

    bool installed(const store::Entry& item) const {
        const auto path = model_path(item);
        std::error_code error;
        return std::filesystem::file_size(path, error) == static_cast<uintmax_t>(item.bytes) &&
            store::gguf_header(path);
    }

    static json::Value prompt(const store::Entry& model, std::string_view source,
                              std::string_view target, const std::string& text,
                              const std::vector<std::pair<std::string, std::string>>& glossary,
                              bool formatted) {
        const bool hy = std::string_view(model.translation) == "hy";
        const std::string format = formatted
            ? "原样保留所有占位符，保持数量和顺序。" : "";
        const std::string direction = "将以下" + std::string(source) + "文本翻译为" + std::string(target);
        std::string instruction;
        if (hy) {
            if (!glossary.empty()) {
                instruction = "参考下面的翻译：\n";
                for (const auto& [from, to] : glossary) instruction += from + " 翻译成 " + to + "\n";
            }
            if (formatted) instruction += format + "\n";
            instruction += direction + "，注意只需要输出翻译后的结果，不要额外解释：\n" + text;
        } else if (glossary.empty() && !formatted) {
            instruction = "请" + direction + "，直接输出翻译结果，不要进行任何解释。\n\n" + text;
        } else {
            instruction = "请" + direction + "，并且严格遵循所有约束要求。\n\n【源文】\n" +
                text + "\n\n【约束要求】\n";
            int n = 1;
            if (formatted) instruction += std::to_string(n++) + ". 【硬性要求】" + format + "\n";
            if (!glossary.empty()) {
                instruction += std::to_string(n) + ". 【硬性要求】术语对照：";
                for (const auto& [from, to] : glossary) instruction += from + "→" + to + "；";
            }
            instruction += "\n\n只输出译文，不要有任何额外说明。";
        }

        auto message = json::Value::make_obj();
        message.set("role", json::Value::make_str("user"));
        message.set("content", json::Value::make_str(instruction));
        auto messages = json::Value::make_arr();
        messages.arr.push_back(std::move(message));
        auto request = json::Value::make_obj();
        request.set("messages", std::move(messages));
        request.set("no_think", json::Value::make_bool(true));
        request.set("temperature", json::Value::make_num(hy ? 0.2 : 0));
        request.set("top_p", json::Value::make_num(hy ? 0.6 : 1.0));
        request.set("top_k", json::Value::make_num(hy ? 20 : 0));
        request.set("repeat_penalty", json::Value::make_num(hy ? 1.05 : 1.0));
        request.set("max_tokens", json::Value::make_num(std::clamp<size_t>(text.size() * 2 + 48, 192, 2048)));
        return request;
    }

    std::string model_dir_, script_, cached_path_;
    std::timed_mutex mu_;
    std::unique_ptr<ModelWorker> worker_;
};

}  // namespace translation
