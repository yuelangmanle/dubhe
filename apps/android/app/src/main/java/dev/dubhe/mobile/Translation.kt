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
    val glossary: String = "",
)

internal fun normalizeTranslationSource(text: String): String = text.trim().replace("\r\n", "\n").replace('\r', '\n')

internal fun readTranslationGlossary(text: String): List<Pair<String, String>> {
    require(text.length <= 1200) { "常用术语最多 1200 个字符。" }
    val terms = text.lineSequence().filter { it.isNotBlank() }.map { line ->
        val fields = line.split('=', '＝', limit = 2).map { it.trim() }
        require(fields.size == 2 && fields.all { it.isNotEmpty() && it.length <= 100 }) {
            "每行填写一个「原文 = 译法」。"
        }
        fields[0] to fields[1]
    }.toList()
    require(terms.size <= 20 && terms.distinctBy { it.first.lowercase() }.size == terms.size) {
        "最多保存 20 个不重复的术语。"
    }
    return terms
}

internal fun translationPrompt(
    text: String, from: Language, to: Language, dedicated: Boolean, glossary: List<Pair<String, String>> = emptyList(),
): String {
    val source = if (from == Language.Auto) "" else from.label
    val applicable = glossary.filter { text.contains(it.first, ignoreCase = true) }
    val instruction = buildString {
        append("将以下${source}文本翻译为${to.label}，注意只需要输出翻译后的结果，不要额外解释")
        if (applicable.isNotEmpty()) {
            append("。请使用指定译法：\n")
            for ((word, meaning) in applicable) append("$word = $meaning\n")
            append("以下是原文")
        }
        append("：\n$text")
    }
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
