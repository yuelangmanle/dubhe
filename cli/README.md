# cli/ · dubhe

命令行入口。单二进制，无运行时依赖。

## 子命令（M0 目标）

```
dubhe inspect -m model.gguf          # 打印模型元信息（层数/量化/tokenizer）
dubhe run     -m model.gguf -p "..." # 单次生成，--stats 打印 tok/s
dubhe chat    -m model.gguf          # 交互式对话
dubhe bench   -m model.gguf --suite chat-1 --concurrency 8
```

## 约束

- 不引第三方 CLI 框架，参数解析自己写（保持零依赖）。
- 每个子命令都要有 `--json` 输出，供 bench 与 CI 消费。
- 输出人类可读版与机器可读版必须来自同一份数据，不允许两套逻辑。
