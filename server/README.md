# server/ · dubhed

本地 HTTP 服务，OpenAI-compatible。M2 交付。

## 接口（目标）

```
GET  /v1/models
POST /v1/chat/completions      # 支持 stream: true
POST /v1/completions
GET  /healthz
GET  /metrics                  # Prometheus 文本格式
```

## 约束

- 复用引擎，不复制逻辑；HTTP 层只做协议转换与限流。
- 单进程多线程，不引异步框架；用 cpp-httplib。
- 只监听 127.0.0.1，默认无鉴权；如需对外必须显式 `--host` + `--token`。
- 流式响应走 chunked，不做 SSE 二次缓冲（避免首 token 被拖慢）。
