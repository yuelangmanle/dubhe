# DubheKit

Swift 和 Kotlin SDK 直接使用 Dubhe 的 C ABI。公开头文件 `sdk/include/dubhe.h` 链到引擎的同一份头文件。

## Swift

`sdk/swift` 是可在本地引用的 Swift Package，提供 `DubheKit` 库。它在 macOS 13 及以上编译仓库里的 C++ 和 Metal 源码。通过 `.package(path: "sdk/swift")` 加入 SwiftPM 项目，或在 Xcode 中添加本地软件包。

```swift
import DubheKit
import Foundation

let engine = try DubheEngine(contextSize: 2048)
defer { engine.close() }
let model = try await engine.loadModel(at: URL(fileURLWithPath: "/path/to/model.gguf"))
let result = try await engine.generate("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n") { text in
    print(text, terminator: "")
}
```

运行验证：

```sh
DUBHE_TEST_MODEL=/path/to/small-chatml.gguf swift test --package-path sdk/swift
```

## Kotlin

在仓库根目录运行 `./apps/android/gradlew -p apps/android :dubhekit:assembleRelease`，产物是 `sdk/kotlin/build/outputs/aar/dubhekit-release.aar`。AAR 支持 Android 12 及以上的 arm64 设备。使用本地 AAR 时，同时加入 `org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0`。

```kotlin
import dev.dubhe.sdk.DubheEngine
import java.io.File

// 在协程里运行
DubheEngine(contextSize = 2048).use { engine ->
    val model = engine.loadModel(File("/path/to/model.gguf"))
    val result = engine.generate("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n") { text ->
        print(text)
    }
}
```

Kotlin SDK 还能加载配套的 `mmproj`。先调用 `loadProjector`，再给 `generate` 传入 `Attachment`。提示词中要有对应的 `AttachmentKind.token`，SDK 会把占位符替换成附件特征。

两个 SDK 都在专用工作线程上运行模型调用。可以从另一线程调用 `cancelGeneration()` 请求停止，或取消等待生成的 Swift Task 和 Kotlin 协程。文本回调运行在工作线程，更新 UI 时请切回主线程。通过 `close()` 释放引擎。提示词应使用模型对应的模板，示例使用 ChatML。
