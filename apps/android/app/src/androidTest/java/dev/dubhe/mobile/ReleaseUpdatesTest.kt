package dev.dubhe.mobile

import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.ByteArrayOutputStream
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ReleaseUpdatesTest {
    private val name = "Dubhe-1.10.0-Android-arm64.apk"
    private val url = "https://github.com/yuelangmanle/dubhe/releases/download/v1.10.0/$name"
    private val digest = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    @Test
    fun selectsNewerVersionAndExactApkWithItsMetadata() {
        val release = ReleaseUpdates.parse("1.9.9", release())!!
        assertEquals("1.10.0", release.version)
        assertEquals("- 修复问题", release.notes)
        assertEquals(3L, release.size)
        assertEquals(digest.removePrefix("sha256:"), release.sha256)
        assertEquals(url, release.downloadUrl.toString())
        assertNull(ReleaseUpdates.parse("1.10.0", release()))
        assertNull(ReleaseUpdates.parse("2.0.0", release()))
    }

    @Test
    fun refusesUnverifiableOrWrongAssets() {
        val invalid = listOf(
            asset().put("browser_download_url", url.replace("https", "http")),
            asset().put("browser_download_url", url.replace("github.com", "other.example")),
            asset().put("name", "Dubhe-1.10.0-Android-x86.apk"),
            asset().put("state", "new"),
            asset().put("digest", JSONObject.NULL),
            asset().put("size", 0),
        )
        for (apk in invalid) {
            assertThrows(UpdateException::class.java) { ReleaseUpdates.parse("1.9.9", release(apk)) }
        }
    }

    @Test
    fun verifiesBytesSizeAndDigestBeforeTheApkCanBeUsed() {
        val release = ReleaseUpdates.parse("1.9.9", release())!!
        val output = ByteArrayOutputStream()
        val progress = mutableListOf<Int>()
        runBlocking { ReleaseUpdates.verifyDownload(release, "abc".byteInputStream(), output, progress::add) }
        assertArrayEquals("abc".toByteArray(), output.toByteArray())
        assertEquals(100, progress.last())
        for (invalid in listOf("ab", "abcd", "xyz")) {
            assertThrows(UpdateException::class.java) {
                runBlocking {
                    ReleaseUpdates.verifyDownload(release, invalid.byteInputStream(), ByteArrayOutputStream()) {}
                }
            }
        }
    }

    private fun release(apk: JSONObject = asset()) = JSONObject()
        .put("tag_name", "v1.10.0")
        .put("body", "\n- 修复问题\n")
        .put("assets", JSONArray().put(JSONObject().put("name", "Dubhe-1.10.0-macOS-arm64.dmg")).put(apk))

    private fun asset() = JSONObject()
        .put("name", name)
        .put("state", "uploaded")
        .put("size", 3L)
        .put("digest", digest)
        .put("browser_download_url", url)
}
