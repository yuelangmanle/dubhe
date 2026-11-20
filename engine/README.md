# Dubhe Engine

原生引擎读取 GGUF，供 CLI、HTTP 服务与 SDK 使用。公开 API 位于 \`include/dubhe/dubhe.h\`。运行 \`make library\` 得到 \`build/libdubhe.a\`。

- \`src/gguf.cpp\` 与 \`src/tokenizer.cpp\` 处理模型权重和分词。
- \`src/model.cpp\` 与 \`src/engine.cpp\` 处理模型执行、生成、前缀复用及取消。
- Apple Silicon 支持 Metal 与 NEON CPU。Android 支持 NEON CPU，可选 Vulkan Q4_K 矩阵乘。
- Metal 多序列接口使用分开的 KV 槽位。HTTP 服务可将多个普通文本请求合为一个小批。
- 视觉与音频使用和语言权重配套的 mmproj。Mac 走系统媒体解码。Android SDK 用 Android 解码器提供像素或 PCM。

macOS HTTP 服务可用独立运行时打开 MLX 目录和部分 Gemma/MoE GGUF。原生 C 接口面向 Dubhe 自己实现的 GGUF 路径。

改动执行内核后运行 \`make test\` 和 \`make check\`。有多模态模型时也运行相应探针。需离线检查 Metal 源码时可用 \`make shader-check\`，它要求 Xcode 的 Metal Toolchain 组件。
