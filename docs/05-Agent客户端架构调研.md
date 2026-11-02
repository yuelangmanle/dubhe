# Agent 客户端架构调研与移植判断

编号：DUBHE-DEV-005
状态：调研完成，结论已落到工程动作
调研对象：DeepSeek Harness（`dsh`）／Cordis、Qwen Code、OpenHands、Continue、Cline
调研时间：2026-11-02

---

## 0. 一句话结论

DSH 的**插件框架本身搬不动**（Node/TypeScript 运行时，Dubhe 是 C++ 单进程），但它的**三个设计决策可以直接搬，而且都指向 Dubhe 现在的真实缺陷**：

| 决策 | DSH 的做法 | Dubhe 现状 | 能否搬 |
| --- | --- | --- | --- |
| 策略与工具分离 | `tools/pre-execute` 事件，工具里不写权限判断 | 每个工具内部散落 `if (!g.allow_write)` | **能，且本周就该做** |
| 输出值 / 模型视图 / UI 卡片三层分离 | `output.schema` → `output.render` → `presentCall/presentResult` | 一个 `std::string` 同时当三者用 | **能，改返回结构即可** |
| 注册即副作用，卸载可逆 | `ctx.effect()` 返回 disposer | 无注册表，工具是硬编码 if 链 | 部分能，代价大于收益 |

**不能搬的**：Cordis 本身、`dsh.bundle` 的 npm 分发模型、PTC mode（`run_code`）。
理由在 §5，逐条给了。

---

## 1. DSH 是什么

| 项 | 值 |
| --- | --- |
| 仓库 | `github.com/deepseek-ai/deepseek-harness` |
| 定位（About 原文） | `DeepSeek Harness: Everything is a Plugin.` |
| 许可 | **MIT**（依赖许可另见 `THIRD_PARTY_NOTICES.md`） |
| 状态 | developer preview，README 明写 `THERE WILL BE COMPATIBILITY-BREAKING CHANGES.` |
| 底层框架 | [Cordis](https://github.com/cordiverse/cordis)，MIT，9.5k star，论文 [arXiv:2608.25512](https://arxiv.org/abs/2608.25512) |
| 启动 | `npx @deepseek-ai/dsh web` → `http://127.0.0.1:3080` |
| 文档 | `deepseek-harness.github.io/deepseek-harness` |

规模数据（抓取当日页面）：269.4k stars / 32.7k forks / 24,922 commits。这个体量说明它不是玩具，但也意味着**它的抽象是为「一个团队长期演进」设计的**，不是为「一个人两周写完」设计的。

---

## 2. Cordis：插件树的五个概念

DSH 的全部架构都建立在这五个概念上（来自官方 Cordis 入门页）：

| 概念 | 原文表述 | 翻译成工程语言 |
| --- | --- | --- |
| **插件 = 实现 Service 的对象** | 函数带 `apply(ctx)`，或 `Service` 子类 | 一个模块注册一个能力 |
| **上下文 = 服务的容器** | 服务占据稳定的 `ctx.<key>`（`ctx.tools`、`ctx.llm`、`ctx.sessions`） | 按 key 查服务，不 import 具体实现 |
| **`inject` 声明依赖** | 声明所需服务后，等这些服务就绪才启动 | 加载顺序由依赖表达，不手写启动序列 |
| **类型化事件通信** | `emit` / `waterfall` / `parallel` / `serial` / `bail` 五种分发 | 见下表 |
| **注册是可逆副作用** | 通过 `ctx.effect()` / `ctx.on()` 安装，reload 和 teardown 时撤销 | 每个注册必须返回 disposer |

### 五种分发模式（这是 Cordis 最有价值的部分）

| 模式 | 是否 await | 顺序 | 有返回值 | 语义 |
| --- | --- | --- | --- | --- |
| `emit` | 否 | 注册顺序 | 否 | 纯广播，观察 |
| `waterfall` | 否 | 注册顺序 | 是 | **环绕中间件** |
| `parallel` | 是 | 并行 | 否 | 扇出 |
| `serial` | 是 | 注册顺序 | 是 | 按序执行 |
| `bail` | 否 | 注册顺序 | 是 | 停在首个非空返回 |

`waterfall` 的语义值得抄：监听器签名是 `(...args, next)`，**调用 `next()` 才执行下游，不调就直接短路**。官方文档明确说「对于单决策事件，短路是设计意图；策略监听器在拥有决策权时可以不调用 `next()` 直接返回」。

这一句话就是权限门禁的全部理论基础：**拒绝 = 不调 next**。

---

## 3. DSH 的工具模型：三层分离

这是最值得 Dubhe 抄的部分。DSH 里一个工具的定义长这样：

```ts
ctx.tools.register(defineTool({
  name: 'read_file',
  description: 'Read a file from disk.',          // 模型看到的
  parameters: {
    path:  { type: 'string', required: true, description: 'Absolute path' },
    limit: { type: 'number' },                     // 默认可选
  },
  output: {
    schema: { type: 'string' },                    // 规范值
    render: (_args, value) => [{ type: 'text', text: value }],  // 模型视图
  },
  async execute(args, exec) {
    return readFile(args.path, { encoding: 'utf8', signal: exec.signal });
  },
}))
```

三层各自独立：

| 层 | 职责 | 谁消费 |
| --- | --- | --- |
| `execute` 返回值（`output.schema` 约束的**规范 JSON 值**） | 程序化 API | PTC 模式下的 `await tools.<name>(args)` |
| `output.render(args, value)` | 模型看到的自然语言 | 拼进 prompt |
| `presentCall` / `presentResult` 返回的**卡片意图** | UI 渲染 | 客户端 |

卡片意图是一个联合体，官方定义了七种：

| `card` | 场景 | 关键字段 |
| --- | --- | --- |
| `generic` | 默认 | `title`、`kind`（图标）、`rawInput`、`locations` |
| `terminal` | shell 命令 | `title`（命令本身）、`description`、`cwd` |
| `diff` | 写文件 | `diffs: [{ path, oldText, newText }]`，新文件 `oldText: null` |
| `read` | 读文件完成态 | `path`、`offset`、`lines`（保留文件行号）、`totalLines`、`lang` |
| `search` | grep/glob 完成态 | `shape: 'matches' \| 'paths'`、`truncated`、`total` |
| `web` | 联网 | `kind: 'search' \| 'fetch'`、结构化来源 |
| （`terminal` 完成态） | 命令跑完 | 原始输出 + 退出元数据 |

三条硬规则（原文要求，违反会出问题）：

1. **展示函数必须是纯函数。** 实时流式输出和会话日志回放都会跑它，所以不能做 I/O、不能读会话状态、不能用时钟或随机数。
2. **UI 格式不许进模型结果。** ` ```console ` 围栏、diff、相对化路径不该仅为服务 UI 而进入规范值。
3. **展示路径软校验。** 参数格式错就让包装器返回 `undefined` 走通用回退，**绝不能让展示导致回放崩溃**。

> 对照 Dubhe：现在 `run_tool` 返回 `{ ok, output }`，`output` 同时是模型视图和 UI 视图。前端拿到 `output` 只能整块塞进 `<pre>`，所以工具调用卡片永远长得一样——这正是用户说的「工具太少并且调用失败」里「调用失败」的观感来源之一：看不出它到底读了什么、改了什么。

---

## 4. 权限模型

### 4.1 DSH 的权限

| 项 | 值 |
| --- | --- |
| 默认模式 | `workspace-write`（只能写工作区） |
| 自动批准 | `--full-auto` / `--yolo` |
| 解除沙箱 | `--sandbox danger-full-access` |
| 审批交互 | 需要审批的操作，Web UI 先弹窗问 |

官方 `SAFETY.md` 原文（照抄，不做转述）：

> Sandboxing, approval prompts, and permission controls can reduce risk, but they do not guarantee isolation or prevent damage. Even correctly enforced restrictions cannot protect resources that the project is allowed to access.
>
> Do not rely on DeepSeek Harness as the sole security control for untrusted workloads.

以及：

> DeepSeek Harness is experimental developer-preview software. It has not undergone a security audit and must not be treated as secure or production-ready.

**这条声明本身是设计资产**：它把「我们不保证隔离」写在最显眼的位置，从而把责任边界划清。Dubhe 的 `README` 也应该有这么一段，而不是等用户被 `rm -rf` 之后才解释。

### 4.2 六个扩展点（按执行顺序）

| 扩展点 | 作用 | 失败行为 |
| --- | --- | --- |
| `tools/pre-execute` | 允许／拒绝／询问策略 | **fail-closed**：监听器抛异常即视为拒绝 |
| `ctx.tools.guard()` | 最终单调拒绝，后续监听器**无法撤销** | 永久拒绝 |
| `tools/execute` | 加截止时间、重试、埋点 | 包装原调用 |
| `tools/post-execute` | 替换展示内容或返回值、阻止结果、附加模型可见上下文 | 可改可挡 |
| `tools/result` | 只观测不可变结果，**不能改** | 只读 |
| `output.presentationMeta` | 从规范值派生可回放的卡片数据，持久化在 `tool/result` 上 | 投影失败则卡片走通用回退 |

官方还明写了一条工程纪律：

> 尽量不要把部署策略内建到工具中。使用 `tools/pre-execute` 实现可扩展的允许／拒绝／询问策略。

> 对照 Dubhe：**这条正好说反了我们现在的做法。** `run_tool` 里每个工具自己 `if (!g.allow_write) { ... }`、`if (!g.allow_net)`、`if (!g.allow_shell)`，权限判断散在 4 个地方，加一个新工具就多一个漏判风险。这是 §6 要改的第一件事。

---

## 5. 横向对比：其他国产／开源 Agent 客户端

| 项目 | 形态 | 架构核心 | 扩展点 | 插件开源 | 许可 |
| --- | --- | --- | --- | --- | --- |
| **DSH** | CLI + Web UI（3080） | Cordis 插件树，一切皆插件 | `ctx.tools` / `ctx.llm` / `ctx.sessions` + 6 个工具钩子 | **是**，MIT | MIT |
| **Qwen Code** | CLI（终端） | fork 自 Gemini CLI，针对 Qwen3-Coder 改解析器与工具支持 | Skills（v0.6.0 起实验性）、MCP；`@qwen-code/webui` 共享 UI 包 | 是 | Apache-2.0 |
| **OpenHands** | Web + Docker | 事件流（event stream）+ Docker 沙箱，agent 与运行时分离 | 自定义 agent / runtime / tool | 是 | MIT |
| **Continue** | IDE 扩展 | 把自己定位成 IDE 扩展框架，配置驱动 | `config.yaml` 自定义模型与上下文提供方 | 是 | Apache-2.0 |
| **Cline** | VS Code 扩展 | 绑定 VS Code，每步操作人工审查 | 无独立插件系统，靠 VS Code 生态 | 部分 | Apache-2.0 |

三条观察：

1. **「插件系统」有三种做法，不是一种。**
   - 进程内插件树（DSH/Cordis）：最强，代价是要有稳定的服务边界和事件契约。
   - 配置文件驱动（Continue）：最轻，能配模型和上下文，不能加工具。
   - 复用宿主生态（Cline/Qwen Code 用 MCP）：零成本拿到别人的工具，代价是受制于 MCP 的表达力。

2. **国产里只有 DSH 把「框架」抽出来单独立项**（Cordis 是独立仓库、独立论文、独立 star 数）。Qwen Code 是 fork 改造，OpenHands 是自研事件流，都没有可独立复用的插件内核。

3. **沙箱强度与体验成反比，且没人真的解决了它。** OpenHands 靠 Docker（重，但真隔离）；DSH 靠 `workspace-write` + 审批（轻，官方承认不保证）；Dubhe 靠 seatbelt（轻，但本机 `sandbox_apply` 直接 `Operation not permitted`，只能降级裸跑）。**三家都在同一个坑里，区别只是坑的深度。**

---

## 6. 移植判断：逐条落到 Dubhe

Dubhe 的现实约束：C++17 单进程 `dubhed`（3647 行）+ 无构建步骤的 `app.js`（约 2400 行）+ WKWebView 外壳。没有 Node 运行时，没有 TypeScript，工具是 `run_tool` 里的一条 if 链。

### 6.1 能直接搬（本周做）

| 项 | 怎么做 | 落点 |
| --- | --- | --- |
| **策略与工具分离** | 加 `gate(name, args)`，在 `run_tool` 入口调一次；工具内部删掉所有 `if (!g.allow_*)`；未登记的工具有一个算一个，**默认拒绝**（fail-closed） | `server/dubhed.cpp` |
| **三层分离** | `ToolResult` 加 `card` 字段（`generic`/`terminal`/`diff`/`read`/`search`/`web`）与 `meta`（JSON），`output` 只留给模型 | `server/dubhed.cpp` + `web/app.js` |
| **卡片纯函数纪律** | `card` 由服务端在 `run_tool` 返回前算好，前端只做映射，不做 I/O、不读时钟 | `web/app.js` |

### 6.2 改形后搬（代价可控）

| 项 | DSH 形态 | Dubhe 形态 | 为什么改 |
| --- | --- | --- | --- |
| **waterfall 短路** | `ctx.waterfall('tools/pre-execute', ...)` | `gate()` 返回 `Allow/Deny/Ask` 三态，`Deny` 直接短路 | 单决策事件不需要通用中间件，三态枚举就够 |
| **`defineTool` 的 schema→类型推导** | TS 类型推导出 `InferArgs` | 手写 `args.s()` / `args.n()` 取值 | C++ 没有结构化类型推导；但可以在 `gate` 里做必填键校验，效果接近 |
| **`ctx.effect()` 可逆注册** | 返回 disposer | 工具表改成 `std::vector<Reg>` + 析构清理 | 只有做热插拔才需要，现在硬编码 if 链更简单 |

### 6.3 搬不动 / 不必搬

| 项 | 判断 | 理由 |
| --- | --- | --- |
| **Cordis 本身** | 搬不动 | 依赖 Node 的 Proxy、AsyncLocalStorage、TS 声明合并。C++ 里复刻等于重写一个 DI 容器，收益抵不上 |
| **`dsh.bundle` + `dsh plugin add`** | 搬不动 | 前提是有 npm 生态。Dubhe 的「插件」只能是**编译期注册的 C++ 函数**，或者**外部进程 + JSON-RPC**（后者等价于 MCP） |
| **PTC mode（`run_code`）** | 不必搬 | 它让模型写代码去调工具，前提是模型能写出正确代码。Dubhe 现在跑的 LFM3-1.7B Q4_K 连 `<tool_call>` 标签都会漏闭合（见 §7 的实测），PTC 只会放大失败率 |
| **Profile 机制（web/headless）** | 部分搬 | Dubhe 已经有 `--serve` / CLI 两种形态，等价物已存在，不需要抽象成 profile |

### 6.4 如果将来真要做插件系统

只有一条路是现实的：**外部进程 + JSON-RPC over stdio**，协议对齐 MCP。这样能直接吃下 MCP 生态（Qwen Code、Cline、DSH 都在用），而不用自己定义一套。

代价估算：`dubhed` 侧约 300 行（spawn 子进程、JSON-RPC 帧、能力发现、超时与崩溃隔离），前端零改动（工具列表由 `/api/tools` 动态返回，现在已经是这个形状）。

**但现在不做。** 理由：9 个内置工具还没打磨完（`edit_file` 的歧义拒绝、`shell` 的沙箱降级都是这两周才补的），先接 MCP 只会让「调用失败」的排查面翻倍。

---

## 7. 参考：DSH 值得抄的三句话

1. 「对于单决策事件，短路是设计意图。」——权限门禁不需要通用中间件，一个三态返回值就够。
2. 「尽量不要把部署策略内建到工具中。」——这是 Dubhe 现在最明显的架构债。
3. 「Sandboxing … do not guarantee isolation.」——写在 README 最前面，比写在错误日志里有用。

---

## 附：抓取来源

| 内容 | URL |
| --- | --- |
| DSH 仓库 | `https://github.com/deepseek-ai/deepseek-harness` |
| SAFETY.md | `https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md` |
| Cordis 入门（五概念、五种分发） | `https://deepseek-harness.github.io/deepseek-harness/reference/cordis-primer` |
| 第一个插件（`apply(ctx)`、`inject`、`ctx.effect`） | `https://deepseek-harness.github.io/deepseek-harness/develop/basic/` |
| 开发一个工具（`defineTool`） | `https://deepseek-harness.github.io/deepseek-harness/develop/basic/tool` |
| 工具编写参考（六扩展点、卡片联合体、三条硬规则） | `https://deepseek-harness.github.io/deepseek-harness/reference/cookbook/adding-a-tool` |
| 使用 Web UI（工作区选择） | `https://deepseek-harness.github.io/deepseek-harness/guide/quickstart` |
| Cordis 仓库 | `https://github.com/cordiverse/cordis` |
| Qwen Code | `https://github.com/QwenLM/qwen-code` |
