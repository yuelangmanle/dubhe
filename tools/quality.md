# Dubhe 质量检查

## 本机模型

生成验收文件不需要联网、模型或额外 Python 包。所有内容为合成资料。先让本机 Dubhe 服务运行，在另一窗口执行以下命令。

```bash
python3 tools/quality_eval.py fixtures build/quality
python3 tools/quality_eval.py evaluate build/quality --vision | tee build/quality/report.json
python3 tools/quality_eval.py chat --repeat 3 | tee build/quality/chat.json
```

服务端走 `127.0.0.1`。端口不同时传 `--port`。可用 `--suite ocr` 或 `--suite translation` 单独测一项。只有安装并可用的模型才进入评测。查询报告时，`skipped` 表示缺少模型或运行环境。校对结果看每个用例的输出和缺项。各类语料检查的是关键数字、专名、术语、阅读顺序、跨列表头和公式，适合发现回归。手写用例用数个不规则笔划模拟编号，实际手写体仍要用已授权的样例另测。

`--vision` 在 Mac 上添加系统文字识别的比较，结构化表格和公式只对支持相应输出的引擎测。对话统计从发出请求到第一段可见正文，包含排队和准备时间。仅当请求计数为零时，将首轮标为服务启动后的第一次生成。期间的操作系统磁盘缓存与是否已加载模型由具体测试环境决定。比较两份报告时使用同一设备、权重和设置。

## 多页任务验收

`build/quality/resume-20.pdf` 有 20 页，文字可选中。各页带有不同的英文页标 `DUBHE-PAGE-01` 至 `DUBHE-PAGE-20`。Mac 可将它转换成只有图像的扫描件。

```bash
swiftc tools/quality_images.swift -o build/quality/quality-images
build/quality/quality-images scan-copy build/quality/resume-20.pdf build/quality/resume-20-scan.pdf
```

1. 打开图像版，从第 1 页扫描至第 20 页。等到第 8 页完成后退出并重开。检查最近任务、页码和已完成结果，继续时从第 9 页处理。
2. 放大原页，选择包含编号的小块重扫。校对并编辑当页，同时确认其他页仍有原来的结果。
3. 送去翻译，停止后重开再继续。导出双语文件，对照 PDF 页标并确认各页只出现一次。
4. 用文字版再跑一遍。对比是否无需识别就能取出文字，以及导出的页次是否一致。

## Android 安装与设备验收

使用一台允许安装应用的 arm64 模拟器或测试手机。在 Apple Silicon 上，可以在 Android Studio 的设备管理器创建 arm64 手机镜像。先用 `adb devices -l` 确认序列号。调试和正式 APK 的签名可能不同，分别使用对应测试设备或者空白模拟器。

```bash
./apps/android/gradlew -p apps/android :app:assembleDebug :app:assembleDebugAndroidTest
bash tools/quality_android.sh 模拟器序列号 test
bash tools/quality_android.sh 测试设备序列号 upgrade 已签名旧版.apk 已签名新版.apk
```

调试测试会运行 instrumentation。覆盖安装保留应用数据，并确认版本变化。真机可记录以下项目。

| 场景 | 观察项 |
| --- | --- |
| 安装后第一次对话 | 设备、Android 版本、内存、模型及量化、启动到能输入、发送到出现首字 |
| 短对话与长对话 | 分别连续三次，记录首字、卡顿、停止后能否再次生成 |
| 连续使用 | 记录开始和结束的内存、耗时、热状态变化以及过热后的反馈。可用 `adb shell dumpsys meminfo dev.dubhe.mobile` 和 `adb shell dumpsys thermalservice` 查看 |
| 切后台和重启 | 已生成的译文、任务状态、聊天与模型选择是否能找回 |
| 覆盖安装前后 | 留一条可辨认的会话、一条未完成翻译和已安装模型。对照新版本是否仍加载它们 |
| 应用内更新 | 看到新版、下载进度、首次授权、系统安装确认。按取消后可重新进入 |
| 首次使用 | 给新用户一张截图和一份多页 PDF，观察在哪一步停下来及最终文件是否能使用 |

常规 CI 在 Linux 构建 Android APK 和测试包，在 Mac 和 Linux 检查合成语料。arm64 APK 的安装与 instrumentation 用设备或 Apple Silicon 上的 arm64 模拟器跑。
