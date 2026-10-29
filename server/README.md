# server/ · dubhed

本地 HTTP 服务 + iOS 风格 Web UI 后端，OpenAI-compatible。

## 启动

```bash
# 纯文本
./build/dubhed -m model.gguf

# 带视觉塔（图像 / 视频）
./build/dubhed -m LFM3-VL-2B-Q4_K_M.gguf --mmproj mmproj-LFM3-VL-2B-Q8_0.gguf

# 带语音塔
./build/dubhed -m LFM3-Audio-2B-Q4_K_M.gguf --mmproj mmproj-LFM3-Audio-2B-Q8_0.gguf

# 双塔（Omni 是视觉 + 语音两个 mmproj，可以给多次）
./build/dubhed -m model.gguf \
    --mmproj mmproj-vision.gguf --mmproj mmproj-audio.gguf

# 分享给手机 / 局域网（自动生成 API Key）
./build/dubhed -m model.gguf --share
```

其他参数：`--port` `--host` `--web` `-c/--ctx` `-t/--threads` `--allow-shell`
`--no-gpu` `--api-key` `--tmp-dir`。

## 接口

```
GET  /                        Web UI
GET  /app.js /app.css /favicon.svg
GET  /healthz
GET  /api/state               运行时状态（硬件、模型、多模态能力、统计）
POST /api/load                切换模型
POST /api/tokenize            调试用
GET  /v1/models
POST /v1/chat/completions     OpenAI 兼容（stream / 非 stream / tools）
```

## 多模态

`content` 走 OpenAI 的数组格式：

```jsonc
{"type":"text","text":"..."}
{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}   // 社区写法
{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}
```

- `url` 支持 data URL 和本机绝对路径；**不抓 http(s)**，避免把网关变成 SSRF 入口。
- 一次请求只接受一个媒体（`dubhe_set_mm_embeds` 只保存一段区间）。
- 只给音频、不给文字时自动走续写（转写模式）；`"dubhe_raw": true` 可显式强制。
- 塔能加载不等于模态能用 —— 占位 token 在**语言模型词表**里。VL 模型挂音频塔
  会返回"这个语言模型的词表里没有 `<|audio|>`"。

`/api/state` 里 `mm` 是能力位掩码（0 无 / 1 视觉 / 2 语音 / 3 两者），
`mm_vision` / `mm_audio` 是它的布尔展开，UI 靠它决定露出哪些附件入口。

## 约束

- 复用引擎，不复制逻辑；HTTP 层只做协议转换与限流。
- 单进程多线程，不引异步框架。
- 只监听 127.0.0.1，默认无鉴权；对外必须显式 `--share` 或 `--host` + `--api-key`。
- 流式响应走 chunked，不做 SSE 二次缓冲（避免首 token 被拖慢）。
- 媒体落临时文件（`--tmp-dir`，默认 `/tmp`），编码完立刻删。
