package dev.dubhe.mobile

import java.io.InputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URI
import java.security.MessageDigest
import javax.net.ssl.HttpsURLConnection
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONObject

internal data class AndroidRelease(
    val version: String,
    val notes: String,
    val size: Long,
    val sha256: String,
    val downloadUrl: URI,
)

internal class UpdateException(message: String) : Exception(message)

internal object ReleaseUpdates {
    private const val repository = "https://github.com/yuelangmanle/dubhe"
    const val page = "$repository/releases/latest"
    private const val api = "https://api.github.com/repos/yuelangmanle/dubhe/releases/latest"
    private val tagPattern = Regex("""v(\d+)\.(\d+)\.(\d+)""")
    private val digestPattern = Regex("sha256:([a-fA-F0-9]{64})")
    private val assetHosts = setOf("release-assets.githubusercontent.com", "objects.githubusercontent.com")

    fun check(current: String): AndroidRelease? {
        val connection = connect(URI(api))
        try {
            connection.setRequestProperty("Accept", "application/vnd.github+json")
            when (connection.responseCode) {
                HttpURLConnection.HTTP_OK -> Unit
                HttpURLConnection.HTTP_NOT_FOUND -> throw UpdateException("GitHub 上暂时没有可查询的正式版本。")
                else -> if (connection.getHeaderField("X-RateLimit-Remaining") == "0") {
                    throw UpdateException("GitHub 的查询次数用完了，稍后再试。")
                } else {
                    throw UpdateException("GitHub 暂时无法返回发布信息。")
                }
            }
            val release = connection.inputStream.bufferedReader(Charsets.UTF_8).use { JSONObject(it.readText()) }
            return parse(current, release)
        } finally {
            connection.disconnect()
        }
    }

    internal fun parse(current: String, release: JSONObject): AndroidRelease? {
        val tag = release.optString("tag_name")
        val offered = version(tag)
        val installed = version("v$current")
        val difference = offered.indices.firstOrNull { offered[it] != installed[it] } ?: return null
        if (offered[difference] < installed[difference]) return null
        if (release.optBoolean("draft") || release.optBoolean("prerelease")) {
            throw UpdateException("GitHub 上暂时没有可查询的正式版本。")
        }

        val version = tag.drop(1)
        val name = "Dubhe-$version-Android-arm64.apk"
        val assets = release.optJSONArray("assets")
        val asset = (0 until (assets?.length() ?: 0)).asSequence()
            .mapNotNull { assets?.optJSONObject(it) }
            .filter { it.optString("name") == name && it.optString("state") == "uploaded" }
            .singleOrNull() ?: throw UpdateException("这个版本还未提供 Android 安装包。")
        val size = asset.optLong("size")
        val digest = digestPattern.matchEntire(asset.optString("digest"))?.groupValues?.get(1)
        if (size <= 0 || digest == null) throw UpdateException("这个版本缺少安装包的校验信息。")

        val url = URI("$repository/releases/download/$tag/$name")
        if (asset.optString("browser_download_url") != url.toString()) {
            throw UpdateException("这个版本的安装包下载地址无法验证。")
        }
        return AndroidRelease(version, (release.opt("body") as? String).orEmpty().trim(), size, digest.lowercase(), url)
    }

    suspend fun download(release: AndroidRelease, output: OutputStream, progress: (Int) -> Unit) {
        val connection = openAsset(release.downloadUrl)
        try {
            val contentLength = connection.contentLengthLong
            if (contentLength >= 0 && contentLength != release.size) {
                throw UpdateException("下载的安装包大小与发布信息不符。")
            }
            connection.inputStream.use { input -> verifyDownload(release, input, output, progress) }
        } finally {
            connection.disconnect()
        }
    }

    internal suspend fun verifyDownload(
        release: AndroidRelease,
        input: InputStream,
        output: OutputStream,
        progress: (Int) -> Unit,
    ) {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        var received = 0L
        var percent = 0
        while (true) {
            currentCoroutineContext().ensureActive()
            val count = input.read(buffer)
            if (count < 0) break
            received += count
            if (received > release.size) throw UpdateException("下载的安装包大小与发布信息不符。")
            digest.update(buffer, 0, count)
            output.write(buffer, 0, count)
            val next = (received.toDouble() / release.size * 100).toInt()
            if (next > percent) {
                percent = next
                progress(percent)
            }
        }
        if (received != release.size) throw UpdateException("下载的安装包大小与发布信息不符。")
        if (digest.digest().joinToString("") { "%02x".format(it) } != release.sha256) {
            throw UpdateException("安装包校验失败，请重新下载。")
        }
    }

    private fun openAsset(original: URI): HttpsURLConnection {
        var url = original
        repeat(6) {
            val connection = connect(url)
            connection.setRequestProperty("Accept-Encoding", "identity")
            val status = try {
                connection.responseCode
            } catch (error: Exception) {
                connection.disconnect()
                throw error
            }
            if (status == HttpURLConnection.HTTP_OK) return connection
            val location = try {
                connection.getHeaderField("Location")
            } finally {
                connection.disconnect()
            }
            if (status !in setOf(301, 302, 303, 307, 308) || location.isNullOrBlank()) {
                throw UpdateException("GitHub 暂时无法提供这个安装包，请稍后重试。")
            }
            url = url.resolve(location)
            if (url.scheme != "https" || url.host !in assetHosts || url.userInfo != null ||
                url.fragment != null || url.port !in setOf(-1, 443)
            ) {
                throw UpdateException("安装包的下载地址无法验证。")
            }
        }
        throw UpdateException("安装包的下载地址发生了太多次跳转。")
    }

    private fun connect(url: URI): HttpsURLConnection =
        (url.toURL().openConnection() as HttpsURLConnection).apply {
            connectTimeout = 10_000
            readTimeout = 20_000
            instanceFollowRedirects = false
            setRequestProperty("User-Agent", "Dubhe Android")
        }

    private fun version(tag: String): List<Int> {
        val parts = tagPattern.matchEntire(tag)?.groupValues?.drop(1)?.mapNotNull { it.toIntOrNull() }
        if (parts?.size != 3) throw UpdateException("版本号无法识别。")
        return parts
    }
}
