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
`--no-gpu` `--api-key` `--tmp-dir` `--model-dir`。

`--model-dir` 默认 `~/.dubhe/models`，UI 的「模型」视图扫这个目录。

## 接口

```
GET  /                        Web UI
GET  /app.js /app.css /favicon.svg
GET  /healthz
GET  /api/state               运行时状态（硬件、模型、多模态能力、统计）
GET  /api/models              本地模型清单（扫描 --model-dir）
POST /api/load                切换模型（卸载 → 加载 → 失败自动装回旧的）
GET  /api/files?path=         列目录（JSON，目录在前）
GET  /api/files/read?path=    读文本文件（512 KB 上限）
GET  /api/tools               工具清单 + 是否启用
POST /api/tools/run           手动执行一个工具，不必绕一圈模型
GET  /api/kb                  知识库文档清单与统计
POST /api/kb/add              索引文件或目录（递归，文本后缀白名单，≤200 个）
POST /api/kb/remove|clear|rag 移除 / 清空 / 开关自动注入
POST /api/kb/search           BM25 检索
GET  /v1/models
POST /v1/chat/completions     OpenAI 兼容（stream / 非 stream / tools）
```

`/api/*` 在设了 `--api-key` 时只允许本机免鉴权，外部访问要带 key。

### 路径安全

文件与工具接口统一走一套边界：参数里不许出现 `..`，`realpath` 之后必须
落在白名单根（工作区 / 家目录 / 模型目录 / `--tmp-dir`）之下。用 realpath
而不是字符串前缀比较 —— 符号链接可以把 `/tmp/x` 指到 `/etc`，前缀检查拦不住。

### 知识库

检索是 BM25，不是向量检索。本地没有 embedding 模型，为了 RAG 硬塞一个
300 MB 的 bge 进 1.7B 的模型旁边不划算；BM25 零模型、零显存、结果可解释。

- 中文没有空格，按 **bigram** 切词（`本地推理` → 本地 / 地推 / 推理）。
  这不是分词，召回偏高、精度略低，配长度归一化后排序仍然稳。
- 切块约 700 字节、重叠 120 字节。重叠是必要的：答案横跨切点时，没有
  重叠两边都搜不到。
- 索引不落盘，只落文档清单（`~/.dubhe/kb/docs.tsv`），启动时重读文件重建。
  源文件才是唯一真源，倒排表序列化出去只会不同步。
- 命中会自动前置成一条 system 注入对话，门槛是 top1 分数 ≥ 5.0。不设门槛
  的话每句"你好"都会拖进三段文档，1.7B 会当场被带跑。

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
