package dev.dubhe.mobile

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI

internal object ReleaseUpdates {
    const val page = "https://github.com/yuelangmanle/dubhe/releases/latest"
    private const val api = "https://api.github.com/repos/yuelangmanle/dubhe/releases/latest"
    private val tagPattern = Regex("""v(\d+)\.(\d+)\.(\d+)""")

    fun newerVersion(current: String): String? {
        val connection = URI(api).toURL().openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 10_000
            connection.readTimeout = 10_000
            connection.instanceFollowRedirects = false
            connection.setRequestProperty("Accept", "application/vnd.github+json")
            connection.setRequestProperty("User-Agent", "Dubhe Android")
            val status = connection.responseCode
            if (status == 404) error("GitHub 上暂时没有可查询的正式版本。")
            if (status != 200) {
                if (connection.getHeaderField("X-RateLimit-Remaining") == "0") {
                    error("GitHub 的查询次数用完了，稍后再试。")
                }
                error("GitHub 暂时无法返回发布信息。")
            }
            val release = connection.inputStream.bufferedReader(Charsets.UTF_8).use { JSONObject(it.readText()) }
            return newerVersion(current, release)
        } finally {
            connection.disconnect()
        }
    }

    internal fun newerVersion(current: String, release: JSONObject): String? {
        val tag = release.getString("tag_name")
        val offered = version(tag)
        val installed = version("v$current")
        val difference = offered.indices.firstOrNull { offered[it] != installed[it] } ?: return null
        if (offered[difference] < installed[difference]) return null
        val version = tag.drop(1)
        val assets = release.getJSONArray("assets")
        if ((0 until assets.length()).none { assets.getJSONObject(it).getString("name") == "Dubhe-$version-Android-arm64.apk" }) {
            error("这个版本还未上传 Android 安装包。")
        }
        return version
    }

    private fun version(tag: String): List<Int> =
        tagPattern.matchEntire(tag)?.groupValues?.drop(1)?.map { it.toInt() }
            ?: error("GitHub 上的版本号无法识别。")
}
