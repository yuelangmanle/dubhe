# bench/ · DubheBench

性能与正确性基准套件。规格见 DUBHE-DEV-001 §6。

> 这是项目的"裁判"，不是"啦啦队"。它的第一职责是让团队在真实 workload 上被打脸，而不是产出好看的图表。

## 组成

```
client/         OpenAI-compatible 并发压测客户端（唯一打所有引擎的入口）
workloads/      Chat-1 · LongPrefill · Agent-N · Rewrite · Structured · Soak
datasets/       固定 prompt 集（含 hash），禁止临时改题
schema/         run manifest 与 result 的 JSON Schema
baselines.lock  竞品冻结 commit + 版本 + 配置
run.sh          一键跑单引擎单 workload
report/         生成标准图表与对比表
```

## 铁律

1. **同权重、同量化、同 tokenizer、同 prompt、同输出长度**，否则数字作废。
2. 每 case ≥5 warmup + ≥10 measured；报 median 与 p95，附 95% bootstrap CI。
3. 首次运行 / autotune 后 / 热缓存**分开统计**。
4. 必须同时报 throughput 与 tail latency，不允许用吞吐掩盖 TTFT。
5. **反作弊**：不得通过降低输出质量、缩短 context、换 tokenizer、串行化对手请求制造优势。
6. 每次 run 落一份 manifest（git sha、编译器、OS build、device、autotune DB hash、model hash…）。

## 状态

M0：schema 与客户端开发中（任务 #7 / #8）。
