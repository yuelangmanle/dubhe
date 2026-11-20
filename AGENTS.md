# Dubhe 发布

- 每次发布客户端更新时，提高 `engine/include/dubhe/dubhe.h` 的版本号。macOS、Android 和引擎共用这一个版本。
- 在 `CHANGELOG.md` 顶部新增该版本，写明用户能看到的变化、修复和兼容性影响。
- 验证两个客户端的安装包。使用 `bash tools/release_clients.sh publish` 推送当前版本和标签，并把 macOS DMG 和签名 Android APK 一起发布到 GitHub Releases。
- Android 的后续版本沿用同一发布签名。签名文件和密码留在本机。
