package dev.dubhe.mobile

internal const val translationLimit = 20_000

internal enum class Language(val label: String) {
    Auto("自动检测"), Chinese("简体中文"), English("英语"), Japanese("日语"),
    Korean("韩语"), French("法语"), German("德语"), Spanish("西班牙语"),
}

internal data class TranslationState(
    val source: String = "",
    val from: Language = Language.Auto,
    val to: Language = Language.Chinese,
    val result: String = "",
    val preferDedicated: Boolean = true,
    val inputLabel: String? = null,
)

internal fun translationPrompt(text: String, from: Language, to: Language, dedicated: Boolean): String {
    val source = if (from == Language.Auto) "" else from.label
    val instruction = "将以下${source}文本翻译为${to.label}，注意只需要输出翻译后的结果，不要额外解释：\n$text"
    return if (dedicated) "<｜hy_begin▁of▁sentence｜><｜hy_User｜>$instruction<｜hy_Assistant｜>" else instruction
}

/** Keep paragraph separators and split long paragraphs at a nearby sentence boundary. */
internal fun translationBlocks(text: String): List<String> = buildList {
    for (paragraph in Regex("\\n(?:[ \\t]*\\n)+").findAll(text).let { separators ->
        var start = 0
        buildList {
            for (separator in separators) {
                add(text.substring(start, separator.range.first))
                add(separator.value)
                start = separator.range.last + 1
            }
            add(text.substring(start))
        }
    }) {
        if (paragraph.isBlank()) {
            add(paragraph)
            continue
        }
        var start = 0
        while (paragraph.length - start > 600) {
            val limit = start + 600
            val preferred = paragraph.substring(start + 300, limit).indexOfLast { it in "。.!?！？；;\n" }
            var end = if (preferred >= 0) start + 301 + preferred else limit
            if (paragraph[end - 1].isHighSurrogate()) end--
            add(paragraph.substring(start, end))
            add("\n")
            start = end
        }
        add(paragraph.substring(start))
    }
}
