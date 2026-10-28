# CHANGELOG

本项目从立项（2026-10-29）起记录所有对外可见的变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

分类：`新增` `变更` `废弃` `移除` `修复` `安全` `文档` `基准`

---

## [Unreleased]

### 文档
- 新增《Aether 项目开发书（总纲）》v1.0：产品定义、分层架构、进程模型、Git 治理、M0–M6 里程碑、质量门禁、合规与商业边界。
- 新增《Aether 项目进度书》v1.0：W0 立项记录、M0 任务分解（20 项 / 约 1,244 h）、里程碑看板。
- 新增《推理引擎开发书索引》：章节索引、9 条已生效结论、Markdown 迁移计划。
- 新增 ADR-0001 ～ ADR-0006（monorepo、不 fork 单引擎、三图 IR、文档即代码、独立进程、许可证策略）。
- 归档《Aether 自适应本地 AI 推理引擎开发书》v1.0 至 `docs/references/`。

### 新增
- 初始化仓库骨架：`engine/` `sdk/` `apps/` `tools/` `bench/` `docs/`。

---

## 版本规划（预告，非承诺）

| 版本 | 对应里程碑 | 内容 |
| --- | --- | --- |
| 0.1.0 | M1 | 引擎可加载 GGUF / MLX，单路对话可用 |
| 0.5.0 | M2 | 并发调度、state 管理、AetherKit 0.5、Desktop Alpha |
| 0.8.0 | M3 | Android Beta、热控 |
| 0.9.0 | M4 | NPU 插件、Studio 微调闭环 |
| 1.0.0 | M6 | 公开发布 |
