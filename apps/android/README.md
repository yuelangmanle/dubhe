# Dubhe for Android

Dubhe 在 arm64 Android 12 及以上设备上读取本地 GGUF。应用使用 Kotlin 和 Compose，推理在独立的 `:engine` 进程中运行。

可安装的 APK 在 [GitHub Releases](https://github.com/yuelangmanle/dubhe/releases/latest)。应用右上角的「更新」提供检查更新和本地更新日志。

macOS 发布机从 `~/Library/Application Support/Dubhe/AndroidSigning` 读取 Android 发布密钥和密码。后续发布复用这套文件，发布脚本会核对 APK 的证书指纹。此前的调试 APK 使用不同证书。

## 构建和安装

准备 JDK 17、Android SDK 36 和 NDK 27.2.12479018。用 `ANDROID_HOME` 或 `apps/android/local.properties` 指定 Android SDK。

在仓库根目录运行：

```sh
./apps/android/gradlew -p apps/android :app:assembleDebug :dubhekit:assembleRelease
adb install -r apps/android/app/build/outputs/apk/debug/app-debug.apk
```

APK 使用本机调试签名，可以安装测试。AAR 在 `sdk/kotlin/build/outputs/aar/dubhekit-release.aar`。

默认构建使用 CPU。加上 `-PdubheVulkan=true` 后，Vulkan 构建会尝试用 GPU 执行支持的 Q4_K 矩阵乘。没有可用驱动、模型不符合要求或计算失败时，推理回退到 CPU。应用会在模型名称下显示实际后端。

## 使用

1. 在 Dubhe 中选「选择 GGUF 文件」，从系统文件选择器导入模型。
2. 输入消息并发送，回复逐段出现。点「停止」可以中止生成，已显示的内容会保留。
3. 使用配套的视觉或语音模型时，先选「导入多模态组件」加入对应的 `mmproj` GGUF，再选图片、视频或音频作为当前消息的附件。

应用把模型复制到私有目录，重新打开时恢复上一次成功导入的模型。导入会额外占用一份模型大小的存储。更换语言模型后，请选与新模型配套的 `mmproj`。

一次消息可带一个文件。视频默认每秒抽两帧，最多四帧。音频最长 60 秒。解码格式取决于设备的系统解码器。在附件编码期间请求停止会在编码返回后生效。应用在绑定期间使用独立推理进程，并通过 Binder 批量发送文本。

Android 报告严重过热时，应用请求停止当前生成，并提示降温后再试。

## 验证

```sh
./apps/android/gradlew -p apps/android :app:assembleDebug :app:assembleDebugAndroidTest
adb install -r apps/android/app/build/outputs/apk/debug/app-debug.apk
adb install -r apps/android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w dev.dubhe.mobile.test/androidx.test.runner.AndroidJUnitRunner
```

在调试设备的应用私有目录 `files/smoke.gguf` 放一个小型 ChatML GGUF，就能验证真实生成、取消和再次生成。多模态用例从 `files/smoke-media/` 读取 `vision.gguf`、`vision-mmproj.gguf`、`audio.gguf`、`audio-mmproj.gguf`、`red.png`、`red.mp4`、`hello.wav` 和 `hello.m4a`。测试夹具由本地提供，缺失时对应测试跳过。
