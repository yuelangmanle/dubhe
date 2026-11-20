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
`--no-gpu` `--api-key` `--tmp-dir` `--model-dir` `--data-dir` `--parallel`。

`--data-dir`（也可设 `DUBHE_DATA_DIR`）默认是 `~/.dubhe`，用于保存历史、设置和知识库清单。未指定 `--model-dir` 时，「模型」视图扫描数据目录下的 `models`。

## 接口

```
GET  /                        Web UI
GET  /app.js /app.css /favicon.svg
GET  /healthz
GET  /api/state               运行时状态（硬件、模型、多模态能力、统计）
GET  /api/models              本地模型清单（扫描 --model-dir）
GET  /api/history             读取本机对话
POST /api/history             按对话更新、移除或导入旧记录
GET  /api/store               可下载的模型和进度
POST /api/store/download|cancel 下载或取消
GET  /api/translate           离线翻译模型和安装状态
POST /api/translate           使用独立模型翻译文字
POST /api/load                切换模型（卸载 → 加载 → 失败自动装回旧的）
POST /api/unload              卸载当前聊天模型
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

局域网客户端通过 `Authorization: Bearer <key>` 调用 `/v1/models` 和 `/v1/chat/completions`。文件、工具、历史和模型管理需要从本机打开。局域网的 `/api/state` 只返回基本服务状态。

### 结构化响应

原生 GGUF 文本对话可在 `response_format` 指定 `{"type":"json_object"}`，或使用 `type: "json_schema"` 并在 `json_schema.schema` 提供结构。支持基本类型、`properties`、`required`、数组及长度、字符串长度、`const`、`enum` 和 `anyOf`。声明字段的对象只生成这些字段，请显式指定 `additionalProperties: false`。超出支持范围会报错。

### Metal 并跑

`--parallel` 接受 1–8，默认 4。支持多序列的 Metal 模型可将短时间内到达的普通文本请求组成一批。`--parallel 1` 始终逐个处理。多模态、工具、思考、约束输出和超过批量上下文容量的请求走单路。

### macOS 可选模型运行时

MLX 权重通过安装了 `mlx-lm` 的 Python 读取。启动服务时可用 `DUBHE_MLX_PYTHON` 指定该解释器。Gemma 和部分 MoE GGUF 使用本机的 `llama-server`，可用 `DUBHE_LLAMA_SERVER` 指定可执行文件路径。直接启动 App 可执行文件时，App 会把这些环境变量传给服务。加载后支持普通和流式文本对话。

### 离线翻译

Hy-MT2 1.8B 与 Index-Translate 2B 通过模型商店下载。macOS 翻译使用独立的 llama.cpp 进程，聊天模型可同时使用。服务接受目标语言、原文和可选术语表。例如 `{"model":"hy-mt2-18b-q4","source":"auto","target":"en","text":"水文年","glossary":[{"source":"水文年","target":"water year"}]}`。翻译适配器需要本机 Python 3，优先查找安装包中的 llama.cpp。

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

- `url` 支持 data URL，本机请求也可使用绝对路径。
- 一次请求只接受一个媒体（`dubhe_set_mm_embeds` 只保存一段区间）。
- 向支持转写的 LFM3 音频模型只发音频时，服务自动构造转写提示。`"dubhe_raw": true` 仍可显式选择原始续写。
- 塔能加载不等于模态能用 —— 占位 token 在**语言模型词表**里。VL 模型挂音频塔
  会返回"这个语言模型的词表里没有 `<|audio|>`"。

`/api/state` 里 `mm` 是能力位掩码（0 无 / 1 视觉 / 2 语音 / 3 两者），
`mm_vision` / `mm_audio` 是它的布尔展开，UI 靠它决定露出哪些附件入口。

**塔必须和模型配套。** 请求里带了图片/音频但对应能力位为 0 时直接返回 400，
不会把特征硬塞给一个不认识的模型 —— LFM3 全家的 hidden 都是 2048，维度
对得上，但语义对不上，结果是"200 + 一段和图片毫无关系的话"，比报错难查得多。

`POST /api/load` 换模型时会先卸掉旧的塔，再按"同目录 = 同一套东西"的约定
自动挂上同目录的 `mmproj*.gguf`（可用 `mmproj` 参数显式指定）。响应里的
`mm` / `mmproj` / `mm_note` 说明最终状态；塔加载失败或维度对不上时
`mm_note` 会写明原因。命令行启动同理：不给 `--mmproj` 就自动找，要纯文本
启动加 `--no-mmproj`。

## 约束

- 复用引擎，不复制逻辑；HTTP 层只做协议转换与限流。
- 单进程多线程，不引异步框架。
- 默认只监听 127.0.0.1。显式启用 `--share` 或绑定外部网卡时需要推理密钥，省略密钥会自动生成。
- 流式响应使用 chunked SSE。批量生成由各请求线程分别写入连接。
- 媒体落临时文件（`--tmp-dir`，默认 `/tmp`），编码完立刻删。
