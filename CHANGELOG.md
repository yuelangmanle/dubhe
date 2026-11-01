# CHANGELOG

从立项（2026-10-29）起记录所有变更。版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

分类：`新增` `变更` `废弃` `移除` `修复` `安全` `文档` `基准`

---

## [0.15.2] · 2026-11-01 · 知识库和工具调用会打架

### 修复

- **知识库注入让前缀缓存每轮失效**。检索结果原来插成**最前面**的一条
  system，于是每条新问题一换，整个 prompt 的第一个字节就变了 —— 前缀
  缓存全部作废，多轮对话每轮重新 prefill 整段历史（实测 7 秒）。
  改成拼在最后一条用户消息前面：前面的历史一个字节都不动，缓存照常命中。
  四轮对话实测 60/80、144/168、225/249 全部命中，首字 0.06–0.11 s。

- **带工具时不注入知识库**。实测（LFM3-1.7B）两者会打架：知识库里本来
  就有 README，把它的片段塞进 prompt 之后，模型把"资料已经在这儿了"
  当成任务，回的是"我们需要执行 read_file……"然后一个工具都不调；
  关掉 RAG 同一句话 1.45 秒就正确调用。1.7B 这种规模一次只能抓住一个
  意图，而工具模式下"去读真实内容"本来就比"复述旧片段"更对。

---

## [0.15.1] · 2026-11-01 · read_file 支持分页

### 新增

- **`read_file(path, offset, limit)`**。原来一次返回整个文件（上限 256 KB），
  读一个 10 KB 的 README 就是 3158 个 token，工具轮第二轮的 prefill 要
  十几秒 —— 而用户问的可能只是"第一行写了什么"。

  现在默认只给前 120 行，结果末尾写明"这是第 1–120 行，全文共 240 行，
  要看后面的内容用 offset=121 再读一次"。模型可以自己决定读多少。
  越界会明确说"文件只有 N 行"。

  实测工具轮首字：12.94 s → **8.47 s**，且不再触发上下文截断。

### 修复

- 分页的行数统计差一行（240 行的文件报成 241 行）。文件以换行结尾时，
  最后那个换行后面不是"多出来的一行"。

---

## [0.15.0] · 2026-11-01 · 多轮对话不再每轮重算整段历史

### 新增

- **KV 前缀复用**。每一轮请求都会把整段历史重新发一遍，而真正新增的
  往往只有用户刚说的十几个字；原来的做法是 `reset_kv()` 之后整段重算。
  现在引擎记住"KV 里躺着哪串 token"，新 prompt 以它开头就跳过 prefill。

  实测（2214 token 的历史，问第二句）：

  | | 首字延迟 |
  |---|---|
  | 修复前 | 8.19 s |
  | 修复后 | **0.23 s** |

  只认"整段 KV 都是新 prompt 的前缀"这一种情况，不做"最长公共前缀"：
  LFM3 的 ShortConv 是滚动状态，没法回退到中间位置，一旦分叉只能整体重来。
  多模态不参与复用 —— 同一串 token id 配不同的图，算出来的 KV 完全不同。

- **服务端按 token 缓存 prompt 前缀**。光靠文本比对不够：模型生成的
  token 不一定等于"贪婪分词"的结果，实测 LFM3 会把 `：\n\n` 当成一个
  token 吐出来，而重新分词得到 `：` + `\n\n`。字一模一样，token 边界
  一变 KV 就整段作废。所以服务端记住"上一次真正喂进去的那串 token"，
  下一轮按文本前缀直接沿用。

- **工具轮也用上了复用**。第二轮原来把助手轮重建成规范 `<tool_call>`
  格式，和 KV 里的原文对不上。现在模型自己写对了规范格式就原样用，
  只有它吐裸 JSON（那种情况本来也对不上）才重建。

- `dubhe_stats.n_reused` / 流式 `usage.reused_tokens`：这一轮跳过了多少
  个 token 的 prefill。多轮对话里这个数接近 prompt 长度就说明缓存生效了。

### 修复

- **知识库检索把闲聊带跑偏**。注入语原来写"回答时优先依据它们"，1.7B
  会当圣旨：问"今天天气怎么样"，它去复述文档里那句"今天天气很好，我们
  一起"的测试样例。检索没错（那句话确实在库里），错在没给模型"这跟我
  无关"这个选项。改成"可能相关，也可能不相关；不相关就忽略，不要复述"。

### 变更

- `make check` 增加两项：**前缀复用对拍**（复用得到的结果必须和冷启动
  逐 token 一致，分叉的 prompt 必须不复用）和**分词往返检查**（encode →
  decode → 再 encode 是否同一串 token，这是前缀复用的前提）。

---

## [0.14.1] · 2026-11-01 · 一个 const 让工具轮永远停在「生成中」

### 修复

- **工具卡片一到达，气泡就再也长不出正文**。`send()` 里 `node` 和 `body`
  声明成 `const`，而 `tool_result` 事件要整块重绘气泡（把工具卡片画上去），
  重绘那行 `node = fresh` 直接抛 `TypeError: Assignment to constant variable`。
  这个异常被外层 `catch` 吞掉，`finally` 又拿着**已经脱离文档**的旧 `node`
  去 `replaceWith` —— 等于什么也没做。于是界面上只剩一个永远写着
  「生成中…」的空气泡，控制台一句错都不报。改 `let`，并在收尾时确认
  节点还挂在文档里、替换的必须是当前活着的那个。

- **真空轮次给一句人话**。既没有正文也没有工具卡片时，原来渲染成
  「已完成 · 0 tokens」的空气泡；现在明确写清可能是上下文放不下。

- **工具结果超过上下文时按 token 往回缩**。`read_file` 读一个 10KB 的
  README 就接近 4096 上限，不截的话第二轮 prompt 越过 `n_ctx`，
  `generate` 一开始就 break —— 没有报错、没有正文，只有转圈。
  现在按实际 token 数逐次回缩（不切半个汉字），并推一条 `note` 事件
  说明截断到了多少。

- **输入本身超上下文时直接 400**，把 tokens 数和上限一起写进错误里，
  而不是静默返回空回复。

---


### 修复

- **解码器把 USER_DEFINED 特殊 token 当噪声丢了**。`is_special()` 同时覆盖
  CONTROL(type 3) 和 USER_DEFINED(type 4)，而 `<tool_call>`、`<function=`、
  `<parameter=`、`<think>`、`<tool_response>` 在 LFM3 词表里全是 type 4。
  模型完整吐出来的工具调用被一段段剔掉，只剩
  `read_file>\npath>\n.gitignore` —— 看着像模型不会用工具，其实是解码器
  在吃它。改成只丢 CONTROL（llama.cpp 的约定：CONTROL 是结构标记，
  USER_DEFINED 是内容）。编码仍然两种都整段匹配。

- **工具调用格式教错了**。LFM3 的模板输出
  `<tool_call>\n<function=read_file>\n<parameter=path>\nv\n</parameter>...`，
  不是 JSON。原来给所有模型灌同一段 JSON 说明，模型照着写就产出半 JSON
  半 XML 的残片。现在按模型自带模板分两套。

- **工具返回的标记不对**。Qwen2.5 和 LFM3 的模板都用 `<tool_response>`，
  我们发 `<tool_result>`。模型不认这个块是结果，把同一个工具再要一遍。
  启动和换模型时读 GGUF 的 `tokenizer.chat_template` 来认标记。

- **思考块默认开着**。LFM3 模板 `enable_thinking` 默认 true，生成提示是
  `<|im_start|>assistant\n<think>`，模型把整段推理写进正文。改走空块
  `<think></think>`；历史里的助手轮也补空块。

- **传 `role:"tool"` 的请求会再执行一遍工具**。客户端自己跑循环时服务端
  不该插手 —— `write_file` / `shell` 是真副作用。现在只把 `tool_calls`
  报回去，并按协议发 `delta.tool_calls`。

- **第二轮正文被抑制逻辑整段吞掉**，界面显示"已完成 · 0 tokens"。
  抑制的前提是"后面还有一轮会说人话"，现在只有确实有人接手时才抑制。

### 新增

- 工具循环上限 2 轮 + 同参数重复检测，到顶时说一句人话而不是把原始 JSON
  摆给用户。
- `DUBHE_DUMP_PROMPT=1`：拼出来的 prompt 直接打到 stderr。这一轮的四个
  问题全靠它确认。
- `tools/toolconv.py`：工具调用收敛测试，判据是服务端发了几次
  `tool_result`（不是回复里有没有关键词 —— 模型编一段也能骗过关键词）。

### 变更

- 前端工具说明默认关闭，并做 localStorage 迁移（旧默认值写进了存储，
  光改默认值救不了老用户）。Agent 视图改成单次开启，不写进设置。
- Agent 视图说明文案：原来那句"2B 级模型工具调用不可靠"已经不成立。

### 基准

| 模型 | 修复前 | 修复后 |
|---|---|---|
| LFM3-1.7B-Q4_K_M | 0 次调用，输出残片 | 1 次调用，读到真实内容，3/3 |
| Qwen2.5-0.5B | 重复调用，永不收敛 | 1 次调用收敛 |

---

## [0.13.2] · 2026-10-31 · 换模型不换塔，图片被静默喂给不认识的模型

### 修复

- **切换模型后多模态状态不跟着变**。从 LFM3-VL-2B 切到 LFM3-1.7B-F16
  之后，`/api/state` 仍报 `mm_vision=true`，`mmproj` 还指着旧的视觉塔。
  图片按钮亮着，用户点下去拿到的是 "They provided a block of text that
  appears to be some garbled mix…" —— HTTP 200、无异常、无日志。

  原有的维度检查拦不住：LFM3 全家的 hidden 都是 2048，视觉塔的特征能顺利
  注入任何一个 LFM3 文本模型。维度对得上，语义对不上。

  `/api/load` 现在先卸塔，再按"同目录 = 同一套东西"自动找配套的
  `mmproj*.gguf`；找不到就诚实报 `mm=0`。加载失败回滚时旧塔一起装回。

- **`build_tokens` 缺能力位时直接 400**。带图/带音频但对应塔没加载时，
  返回"当前模型没有加载视觉塔，无法处理图片/视频。请到「模型」视图换一个
  多模态模型。"，而不是让模型自由发挥。

- **流式中文乱码**。引擎按 token 吐字节，一个汉字常被切在两个 token 之间；
  半截字节进 JSON 就是非法 UTF-8，界面上随机出现 `均���的`。现在吐之前把
  结尾凑不齐的多字节序列留住，和下一片拼上再发。已验证流式与非流式逐字一致。

- **空气泡**。模型把整段输出写成 `<tool_result>{...}`（并没有真的调工具）时，
  `stripToolCalls` 一剥什么都不剩，只剩"已完成 · 9 tokens"和一片空白。
  现在剥完是空且没有工具卡片可看时把原文露出来；带附件时也不再下发工具说明。

- **`/api/models` 把 mmproj 当模型列出来**。点一下必然失败且原因不明。
  改成从清单里摘掉，挂到同目录模型上作为「多模态」标记。

### 新增

- `dubhe_unload_mmproj()` / `dubhe_mm_dim()`。塔是跟模型配套的，不是跟引擎
  配套的 —— 换模型的一方必须能显式卸塔。
- `/api/load` 接受 `mmproj` 参数；响应新增 `mm` / `mmproj` / `mm_note`。
- `--no-mmproj`。不给 `--mmproj` 时，模型同目录下的 `mmproj*.gguf` 会自动
  挂上；维度对不上时启动阶段就打印警告。
- `tools/ui_switch.mjs`：在界面上来回切模型，盯住附件入口有没有跟着变。

### 基准

| | 修前 | 修后 |
|---|---|---|
| 缺塔时发图 | 200 + 一段无关的话 | **400 + 明确原因（0.0 s）** |
| 切到纯文本模型后附件按钮 | 仍可见 | 收起 |
| 流式中文 | 随机出现 `均���的` | 无替换字符 |

---

## [0.13.1] · 2026-10-30 · 多模态 10x，以及把前端"走"一遍而不是"看"一遍

### 基准

同一张 1024×1024 PNG 走 `/v1/chat/completions`：

| | 耗时 | 图像 token |
|---|---:|---:|
| 修前 | 75.3 s | 1045 |
| 修后 | **7.6 s** | 277 |

描述内容没有变差（"四角星 + 三个交叠的椭圆环"）。

### 修复

- **大图没有像素上限**。mmproj 自带的 `clip.vision.image_max_pixels` 是
  16.7M（4096²），等于不设防；而视觉塔跑在 CPU 上，耗时随 patch 数近似
  平方增长（512² → 5 s / 768² → 23 s / 1024² → 75 s）。补了
  `DUBHE_MM_MAX_PIXELS`，默认 512²，图像与视频两条路径都压。

### 新增

- `tools/ui_walk.mjs`：CDP 走查——每个视图点开、收集 console 错误与 4xx/5xx、
  真发一条消息确认流式走通、截图。
- `tools/ui_probe.mjs`：针对性探查——API 面板、附件按钮在无多模态模型时的
  禁用状态、窄屏（390×844）布局与横向溢出。
- `tools/ui_mm.mjs`：通过界面走一遍多模态（选图 → 缩略图 → 发送 → 回复）。
  附件这条路有三段容易断（FileReader、缩略图、`image_url` 拼接），curl
  测不出来。

### 验证

- 6 个视图全部渲染，无 console 错误、无 4xx/5xx。
- 纯文本流式：115.9 tok/s、TTFT 0.95 s，仪表与系统日志同步。
- 多模态：1024² 图 12.6 s 走完（含界面轮询间隔），回复正确。
- 窄屏 390×844：无横向溢出，侧栏收成抽屉，输入框可用。

---

## [0.13.0] · 2026-10-30 · 真正的瓶颈是 Q5_0：解码 1.5x

### 基准

同一台机器、新旧配置交替采样 5 轮（decode，qwen2.5-0.5b-q4_k_m）：

| 配置 | 中位 | 最好 |
|---|---:|---:|
| 旧（v1 内核 + 注意力切分 96） | 87.6 tok/s | 89.7 |
| 新（v2 内核 + 注意力切分 6） | **130.9 tok/s** | **136.1** |

prefill 不受影响（436 → 434 tok/s）。CPU 参考 30.7 tok/s，GPU 领先 4.4x。

### 新增

- `mv_q5_0_v2`：解码 matvec 换成"lane 认领整块"，一个 warp 一次吃 32 块
  = 704 字节（原来一个 warp 一轮只吃一个 22 字节的块）。单内核 1.93x
  （nb=28）/ 2.40x（nb=64）。
- `tools/gguf_sizes.py`：拆 GGUF 张量表，按量化类型/张量角色汇总字节。
  靠它才发现这个"Q4_K_M"里 55% 的字节是 Q5_0、只有 6% 是 Q4_K。
- `tools/bench_best.py`：反复跑 bench 取最优值。这台机器有常驻负载，
  decode 会在 ±10% 之间跳，均值不可用。
- `DUBHE_SKIP=<位掩码>`：在真实前向里按组跳过 gemm / 逐元素 / 注意力，
  用来给时间归因。配套必须设 `DUBHE_NO_FUSED_CHECK=1`。
- `DUBHE_MV_V2=0`、`DUBHE_ATTN_DIV`：A/B 开关。
- `make mvbench` 支持 q5_0 / q4_0，并用生产内核当基准做对拍。

### 变更

- 注意力 split-K 的段数公式 `nlen_max / 96` → `nlen_max / 6`。原来 64 token
  上下文只切 1 段，整张网格只有 4 个 threadgroup = 16 个 warp，而 GPU 有
  10 个核。单这一项 +8.8%。

### 基准结论（否定）

- **算子融合减少 dispatch 数没有意义**：约 200 个逐元素 dispatch 合计只占
  0.07 ms（decode 的 0.8%）。
- 权重布局重排对 72 MB 张量无效（0.998x），现有布局已到 96.1 GB/s 峰值。

### 修复

- `mvbench` 的 `cases` 表把 `{"重排布局", true, 0}` 无条件塞进去，非 Q4_K
  类型下 `ps_new` 为 nil、`w_new` 为野指针，第二轮段错误（exit=139）。
- `mvbench` 的 stdout 改为无缓冲，否则崩溃时攒着的输出全丢。

---

## [0.12.1] · 2026-10-30 · 解码天花板量清楚：两把尺子 + 两条否定结论

这一轮没有提速。产出是**把解码还剩多少空间量清楚**，以及两个可复用的
测量工具 —— 顺手否掉了两条看起来很有希望的路，省得后面再走。

### 基准

| 形状（1 GiB 只读，25 轮取最小） | 带宽 |
|---|---:|
| lane 连续、512 字节对齐（`stream`） | 98.3 GB/s |
| 仿 mv_q4_k：warp 分 4 块 × 每块 8 lane（`block4`） | **74.8 GB/s** |
| 整行 16 字节单元连续（`row`） | 100.1 GB/s |

mv_q4_k 实测 68 GB/s = `block4` 天花板的 **94%**。前几轮"再抠一条指令"
没有收益是必然的。

### 新增

- `tools/membw.mm` / `make membw` —— 只读带宽天花板，按访问形状分档。
- `tools/mvbench.mm` / `make mvbench` —— 单内核吞吐 + CPU 对拍，秒级出数。
- `DUBHE_MV_R` 环境变量：MV_R（一个 simdgroup 同时算几行）改为着色器编译前
  注入，C++ 侧读同一个值算派发网格。为"按模型调参"留口子。

### 变更

- MV_R 扫描结果：1→59.2 / 2→62.3 / 4→61.6 / 8→52.4 tok/s，**2 仍是最优**，
  默认值不变。

### 记录（否掉的假设，别再试）

- **整行单元连续映射**：纯读能到 100 GB/s，真实 decode **0.80x**。
  `144 = 9×16`，一行 `nb×9` 个单元，32 条 lane 除不尽，nb=8 时 96 个槽
  里 24 个空转 → ALU 与 x 读取抬高 50%。旧映射的 32 个数据单元正好等于
  32 条 lane，利用率本来就是满的。
- **一条 lane 吃 2 个块摊薄 x 读取**：不成立，x 按块索引，跨块不是同一段。

### 修复

- 换映射时暴露的越界隐患：`o == 0` 的 lane 踩在块头上，必须把 `k/g` 夹到 0
  且 `msk` 一并挡掉，否则 `k4_scale_min` 会读到行外（不崩，但数值错）。
- `/v1/chat/completions` 把整条请求（含多模态编码、工具两轮）放进同一把锁，
  并置 `g.busy`。补的是引擎锁盖不住的两段：`build_tokens` 里
  `e->mm_embeds` 的赋值窗口（并发带图时 A 会拿到 B 的图）、以及工具两轮之间
  引擎锁松开的空档。生成本身本来就是串行的（`dubhe_generate` 全程持
  `e->mu`），这一点已用改动前后的二进制各跑一遍验证过。

### 新增（续）

- `tools/concurrent_check.py` —— 并发一致性回归测试。判据不是"返回 200"，
  而是"回答对得上自己的问题、且没混进别人的内容"。改动前后都通过，
  它守的是"以后别把引擎锁挪走"这条性质。

## [0.12.0] · 2026-10-30 · F16/Q8_0 的 MMA 内核 · 五个视图接真数据

两件事：把 tensor core 铺满所有量化格式，把 6 个空壳导航项做成真界面。

### 基准

| 1.7B | prefill 前 | prefill 后 | decode |
|---|---:|---:|---|
| F16 | 53.7 | **392.5（7.3x）** | 22.2（不变） |
| Q8_0 | 188.4 | **387.5（2.06x）** | 35.4（不变） |

F16 折合 1.22 TFLOP/s，追到 Q4_K 的 72%。之前 F16 只有 Q4_K 的 1/9，
而它每个元素的活比 Q4_K **少** —— 差的全在"没用上 `simdgroup_matrix`"，
和量化格式无关。

### 新增

- **`mm3_f16` / `mm3_q8_0` MMA prefill 内核**。结构同 `mm3_q4_k`，只换反量化段：
  - F16：反量化整个不存在，2 条 `uint4` 读 + 2 条 `uint4` 写。
  - Q8_0：一个线程管半个 32 元素块，scale 每块读一次；16 字节用 8 条 `ushort`
    读（块跨度 34 ≡ 2 mod 4，`uint4` 落不到边界上，这是格式决定的）。
- **9 个服务端接口**：`/api/models`、`/api/load`（真的实现，之前只有文档）、
  `/api/files`、`/api/files/read`、`/api/tools`、`/api/tools/run`、
  `/api/kb`、`/api/kb/{add,remove,clear,rag,search}`。
- **五个真视图**：模型（能切换，失败自动装回旧的）/ 工具（展开式试运行）/
  Agent（任务丢给工具循环 + 真实轨迹聚合）/ 文件（浏览 + 预览 + 转对话）/
  知识库（索引 + BM25 检索 + 命中高亮）。
- **知识库**：BM25 而非向量检索 —— 本地没有 embedding 模型，硬塞一个 300 MB
  的 bge 进 1.7B 旁边不划算。中文按 bigram 切词，切块 700 字节 / 重叠 120 字节，
  索引不落盘只落清单。命中自动前置成 system 注入对话，门槛 top1 ≥ 5.0。
- 地址栏同步 `#models` 这类 hash，刷新不丢位置，也能直接分享。
- `--model-dir` 参数，默认 `~/.dubhe/models`。

### 修复

- **`http::Request::param()` 不做 URL 解码**。前端 `encodeURIComponent` 编码的
  路径原样传到服务端，`stat` 必然失败，而报的是"文件不存在" —— 看起来像
  权限问题。文件预览点了没反应就是这个。
- **按字节截断切在汉字中间**。snippet 取前后 140 字节、切块在 `。` 后 `p+1`
  字节都会切断 3 字节的汉字；`json::escape_to` 又原样输出 >= 0x80 的字节，
  于是整个响应变成非法 UTF-8，前端 `JSON.parse` 直接抛异常。
  两头都堵：切点用 `utf8_next` / `utf8_back` 对齐，序列化时再验一遍、
  非法序列换 U+FFFD。
- `.views` 的入场动画带 `opacity`，某些时机停在半透明、底下的对话透出来。
  改成只做位移，`opacity` 显式给死 1。
- `#viewBack` 用 `i-chevron`（朝下）旋转 180° 得到朝上，方向错了。换 `chevron-right`。
- 路径显示成 `/Users/…/Dubhe/.`。

### 文档

- `docs/02-进度书.md` §9.18：MMA 折算成 FLOP/s 的对比表、知识库的选型理由、
  以及三个只有真点一遍才会暴露的坑。
- `server/README.md`：补全端点、路径安全边界、知识库设计说明。

## [0.11.0] · 2026-10-29 · 三栏工作台 · 图标系统 · App 图标

界面从"一栏聊天页 + 抽屉"改成三栏工作台。这不是加宽屏幕适配，
是承认**找东西 / 说话 / 看机器状态是三件事**，不该挤在同一列里。

### 新增

- **三栏工作台**：`.desk` 用 CSS Grid `nav-w | 1fr | insp-w`，gap 13px。
  三块是各自独立浮起的玻璃板（`.pane`），不是一整块被竖线切开的白板 ——
  后者在视觉上仍然读作"一个页面"，只是多了两条线。
  - 左：品牌 / 搜索（⌘K）/ 7 项导航 / 工作空间会话列表 / 资源条
  - 中：模型胶囊 / 运行状态徽标 / 对话 / 两行输入区
  - 右：环形仪表 / 内存与显存实测条 / 模型信息六行 / 工具调用列表 / 系统日志
- **右栏四个数据块全部接实测数据**，无一处估算：
  - `rss_bytes` ← `getrusage(RUSAGE_SELF)`（macOS 单位是字节，Linux 是 KB，
    这个差异不处理会差 1024 倍）
  - `model_bytes` ← `stat()` 模型文件 + mmproj 文件体积
  - 仪表固定量程 `TPS_MAX = 60`。**自适应量程是错的** —— 它会让 8 tok/s
    和 55 tok/s 看起来一样满。
- **图标 sprite 从 16 枚扩到约 70 枚**，按 12 类分组（导航 / 快捷操作 /
  对话工具栏 / 模型管理 / 消息类型 / 文件操作 / 工具与功能 / 运行控制 /
  用户账户 / 状态指示 / 硬件监控 / 视图布局）。统一 24 格、1.7 描边、
  圆头圆角、纯 `currentColor`。
- **App 图标三档**：`icon.svg`（≥128px，三道环 + 边缘高光）/
  `icon-sm.svg`（≤96px，两道环、描边加倍、星芒放大）/
  `favicon.svg`（16–32px，只剩一个环加一个点）。
  另出 `icon-maskable.svg`（Android 自适应图标，底色满铺不切角）、
  `site.webmanifest`、PNG 切图 `icon-180/192/512`、`icon-maskable-512`。
- `assets/branding/android/ic_launcher_foreground.svg` /
  `ic_launcher_background.svg`：自适应图标两层，前景内容缩进中心 66dp 安全圆。
- 对话态：助手消息改为 26px 头像 + 内容列网格，状态行显示
  「已完成 · 486 tokens · 24.6 tok/s」（真实计时，不是假数字）。
- 系统日志块（右栏深色 `<pre>`）+ 一键复制。

### 变更

- `--brand-solid` 独立于 `--brand-1`。暗色模式下 `--brand-1` 是浅绿
  `#9DBFAE`，拿它做背景会变成一块亮斑。实心底色单独取
  `#2E4238`（亮）/ `#3E5749`（暗），`nav-item.on`、`.mp-badge`、
  `.avatar`、`.send-btn` 全部改用。
- 背景改三层：`.sky` 天空渐变（上冷下暖，给山脊方向性光照）+ 三条平滑
  曲线 SVG 山脊（远淡近深，blur 1.6px）+ `body::before` 光晕。
  光晕半径从 66%/60% 收到 58%/54%/44% 且只在上方 —— 原来的尺寸把山脊糊住了。
- 底色挂 `html`，`body` 保持透明。`body::before` 是 `z-index:-2` 的定位
  元素，会排到 body 背景之上，两者都设在 body 上会打架。
- 断点从 760px 改 860px，判据是"右栏还站不站得住"，不是设备型号：
  1180（栏宽收缩）→ 1024（右栏改浮层）→ 860（左栏改抽屉、单列）→
  560（工具栏只留图标）。
- 顶栏量化标签改从 `model_id` 正则提取。原来的 `quant` 串是
  `"F32+Q4_K+Q6_K"`，取尾项拿到的是输出层，显示成 `Q6_K`。
- `.kv-stack` 取代 `#modelKv` 上的 `.kv` —— `.kv` 是 `display:flex`，
  把六行模型信息压成了一行。
- 环形仪表轨道色 `--fill-strong`（`rgba(90,110,95,.16)`）在浅底上等于
  不可见，改 `rgba(40,60,45,.15)`。

### 修复

- **静态资源路由是逐条枚举的**，新加的图标、manifest 全部 404。
  `http::Server` 加 `set_fallback`，未命中路由时交给兜底处理；
  `dubhed` 用它接静态资源，但只放行后缀白名单（`.html/.css/.js/.json/
  .svg/.png/.ico/.webmanifest/.woff2/.jpg`）—— 否则 `/api/typo` 会返回
  404 页面而不是 404 JSON，且等于把整个 web_root 暴露出去。
- `guess_mime` 补 `.webmanifest`（`application/manifest+json`）与 `.jpg`。
- 移动端 390px 工具栏文字竖排换行：`≤560px` 下 `.cbtn span` 隐藏、
  `#newChat2` 隐藏、`.cmode` 收小。
- 消息气泡内图片右侧留白：实测 `img=190x119` 在 `bub=401` 的气泡里，
  `img.thumb { width: 100% }` 修掉。

### 文档

- 本文。
- `docs/02-进度书.md` §9.13：三栏工作台与图标系统。

---

## [0.10.1] · 2026-10-29 · 品牌色改墨绿（去 AI 味）

### 变更

- **品牌色从紫蓝渐变改为墨绿 + 暖纸白**。原配色（`#6B5CFF` → `#3D9BFF`
  渐变 + 深空底 + 彩虹色散）是很典型的"AI 生成感"：第一眼抓人，第二眼廉价。
  新配色只用三个值：
  - 墨绿 `#2E4238`（图形本体）
  - 纸白 `#FAF8F3`（底色，暖白不是纯白）
  - 雾绿 `#A7C0B2`（极小面积）
- 背景光晕改为雾绿 / 淡豆绿 / 米黄 / 浅青，低饱和高明度。
- 暗色模式底色 `#0A0A12` → `#171D19`（深绿灰，不是墨黑），
  光晕单独提亮，hero 标题单独给亮渐变（`--brand-grad` 同时被发送按钮使用，
  不能直接改亮）。
- 所有带紫相的投影与遮罩统一换成墨绿相。
- 错误色 `#FF5A5F` → 陶土红 `#C4553F`。

### 新增

- `docs/03-品牌视觉与生图提示词.md`：视觉锚点 + 三条简短提示词 +
  负面词 + 验收 + Android 分层切图规格。
- `assets/branding/ref-icon-v1.png`：方向参考图（有平台水印，不可直接用）。

### 文档

- 本文。

---

## [0.10.0] · 2026-10-29 · Web UI 视觉与交互重构（Dubhe Glass）

把 Web UI 从"能用的聊天界面"重做成一套有自己材质语言的产品界面。
这一版同时是 Android 客户端（A 阶段 APK 壳）的视觉底稿。

### 新增

- **矢量图标集**（`web/index.html` 的 `<symbol>` sprite，16 枚）：统一 24 格 /
  1.7 描边 / 圆头圆角，全部 `currentColor`。替换掉原来的
  `☰ ⚙ ＋ ⇄ ⚡ ✕ ♪ ▶ 📎` 文字符号 —— emoji 在安卓与 iOS 上字形完全不同，
  `⚙` 在部分安卓 ROM 上还是彩色的。
  - 品牌标识重画：行星 + 环 + 卫星。环分两段画，下半弧盖到行星前面，
    环是"绕着"行星而不是"贴在"后面。原双轨道版本在 27px 下第二条会糊成脏影。
  - 工具调用图标从扳手换成终端（更贴合"读文件 / 执行命令"）。
  - 附件按钮从加号换成回形针。
  - 图标校验板 `/tmp/icons.html`（从 sprite 自动生成）用于逐个放大核对。
- **Dubhe Glass 材质语言**（`web/app.css`）：
  - *光跟随*：面板上的柔光位置由 `--mx` / `--my` 驱动，指针移到哪光跟到哪
    （`.lit` / `.edge`）。这是界面上唯一由用户输入直接驱动的光。
  - *四边厚度*：顶受光面 / 底反光 / 左右侧壁四层 `inset`，替代原来的单条顶高光。
  - *边缘色散*：边界泛出极淡冷暖偏移（.05 量级），模拟玻璃拆光。
- **非线性动效体系**：`--ease` / `--spring` / `--snap` 三档语义。
  两条 spring 由阻尼振子采样生成（ζ=.62/ω=16、ζ=.5/ω=24），
  经 `@supports (animation-timing-function: linear(0,1))` 升级，老内核回落
  到 `cubic-bezier(.34,1.56,.64,1)`。
- **三档视觉档位**（设置 → 外观，`body[data-glass]`）：
  `完整` 真实模糊 / `精简` 纵向光衰渐变模拟玻璃 / `省电` 纯材质。
  三档共用同一套排版与层次，低档不是"关掉特效"。
  自适应按 UA、核数、`deviceMemory`、`prefers-reduced-transparency` 挑默认值，
  用户手动选过即持久化、不再自动覆盖。
- **移动端侧栏抽屉**：遮罩层（点空白关闭）+ 左滑关闭手势 + 选中会话自动关闭。
- **停止生成**：生成中发送按钮变为停止方块，走 `AbortController`，保留已生成内容。
- **iOS 风格分段控件**：选中块以弹簧曲线滑过去，不是瞬间跳。

### 变更

- 背景光晕从"四团小半径高饱和色块"改为"大半径低饱和色域"
  （半径 56–64%，alpha .40–.58）。原来的做法像几盏彩色灯，会抢内容注意力。
- 空态改为 hero：标题 `clamp(40px, 6.2vw, 62px)`、字距 `-.028em`，
  logo 做成玻璃徽章，`:has()` 垂直居中。
- 顶栏加投影把自己从内容上抬起来 —— 它背后没有内容滚过，单靠
  `backdrop-filter` 会"隐形"。
- 输入区上方加雾化渐变收口，消息滚到底不再被硬切一刀。
- 触摸目标放大到 44px 起（`.icon-btn`、`.conv`、`.side-item`）。

### 修复

- **移动端横向溢出**：诊断时发现 `html` 宽度是 500 而非 390 ——
  是 macOS Chrome 有 ~500px 最小窗口宽度，之前的"移动端截图"其实是
  500px 视口裁到 390。改用 CDP `Emulation.setDeviceMetricsOverride` 复核，
  真实 390px 下 `scrollWidth == innerWidth == 390`，布局本来就是对的。
  截图验证从此走 CDP，不再用 `--window-size`。
- **`backdrop-filter` 在安卓 WebView 上是软件渲染**：大面积模糊会让中低端机
  滚动掉帧。改为滚动容器一律不加模糊，面板按档位降级。
- 深色模式下 `:hover` 写死 `#fff` 会把文字吃掉（`.model-chip`、`.new-btn`、`.chip`）。

### 文档

- 本文。`docs/02-进度书.md` 补 §9.13。

---

## [0.9.0] · 2026-10-29 · Android 交叉编译打通（M4 起步）

### 新增

- **Android arm64-v8a 交叉编译**：`make PLATFORM=android` 走 NDK toolchain，
  产物在 `build-android-<abi>/`，与本机 `build/` 完全隔离。
  `ANDROID_NDK` / `ANDROID_ABI` / `ANDROID_API` 均可覆盖。
- **`engine/src/parallel.h`**：并行 for 的跨平台封装。Apple 走 GCD
  （`dispatch_apply_f`），其它平台走常驻线程池 + 原子任务分发。
  线程数可用 `DUBHE_THREADS` 覆盖。
- **`engine/src/sysinfo.h`**：CPU 名 / 内存 / 核数的跨平台查询。
  Apple 走 sysctl，其它平台读 `/proc/cpuinfo` 与 `/proc/meminfo`。

### 变更

- `model.cpp` 不再包含 `<dispatch/dispatch.h>`；三处 `dispatch_apply_f`
  改为 `parallel_for`。
- `engine.cpp` 不再包含 `<sys/sysctl.h>`。

### 修复

- **交叉编译产物覆盖本机二进制**：`BIN := $(BUILD)/dubhe` 在文件顶部展开，
  而 `BUILD := build-android-<abi>` 在平台分支里才赋值，导致 Android 产物直接
  盖掉 `build/dubhe`。改为平台分支之后再定义 `BIN`/`SRV`，`BUILD ?= build` 兜底。

### 基准（Android 13 模拟器，arm64-v8a / 3 核 / 2 GB，**非真机**）

| 项 | 值 |
| --- | --- |
| 0.5B Q4_K_M decode | 19.8–25.5 tok/s |
| 0.5B Q4_K_M prefill | 14.6–22.6 tok/s |
| 1 线程 vs 3 线程 | 25.5 vs 22.8 tok/s（模拟器虚拟核无真并行，多线程反而亏） |

模拟器数据只能证明"能跑且输出正确"，性能必须真机重测。

### 已知限制

- Android 上图像/音频解码仍是 stub（要接 `ImageDecoder` / `MediaCodec`），
  多模态暂不可用。
- 没有 GPU 后端，纯 CPU。
- 只验了 `arm64-v8a`，`armeabi-v7a` 分支未实测。
- 未上真机。

---

## [0.8.0] · 2026-10-29 · 三模态接产品面

### 新增

- **HTTP 多模态**：`/v1/chat/completions` 支持 OpenAI 的 content 数组 ——
  `image_url`、`video_url`（社区写法，官方还没定视频协议）、`input_audio`。
  data URL 和本机绝对路径都收；不抓 http(s)，避免把网关变成 SSRF 入口。
- **`--mmproj` 可给多次**：Omni 模型是视觉塔 + 语音塔两个独立 mmproj，
  引擎按 `clip.has_audio_encoder` 自动分流，网关只要允许重复参数。
- **`dubhe_mm_caps(e)`**：返回能力位掩码（1 视觉 / 2 语音），`/api/state`
  的 `mm` / `mm_vision` / `mm_audio` 由它驱动，UI 据此决定露出哪些附件入口。
- **Web UI 附件**：输入框左侧 `+` 按钮，`accept` 按服务端能力动态拼；
  选中的媒体先显示预览 chip，发送时展开成 content 数组。
  历史消息只存 192px 缩略图（约 10 KB），不存原始 data URL。
- **纯音频自动转写模式**：最后一条 user 消息只有音频、没有文字时走续写而不是
  chat 模板。LFM3-Audio 进对话状态会把音频当"待推理的问题"，输出整段分析过程，
  转写被埋在引号里；走续写第一句就是正文。也可用 `dubhe_raw: true` 显式强制。

### 修复

- **纯文本请求正文被吃光**（回归）：`flatten_messages` 里用
  `content->s("")` 取字符串型 content，那是"取名为空串的子键"，恒返回默认值，
  导致 prompt 只剩 9 个模板 token，模型以为用户发了空消息。改用 `content->str`。
- **raw 模式无限复读**：不补 BOS 时模型不知道自己站在序列开头，转写内容对但
  一句接一句重复到 `max_tokens` 也不吐 EOS。raw 和 chat 模板两条路都补 BOS。
- **模态不匹配的报错**：区分"词表里没有这个占位 token"（语言模型不支持该模态）
  和"prompt 里没写占位符"（调用方问题）。原来只说后者，把配置问题误导成调用方问题。

### 基准

| 输入 | 端到端 | prompt token | decode |
| --- | ---: | ---: | ---: |
| 图像 448×448 | 6.8 s | 165 | 49.6 tok/s |
| 视频 8 帧 448 | 13.1 s | 804 | 52.0 tok/s |
| 语音 3.45 s | 4.3 s | 44 | 59.1 tok/s |
| 纯文本 | 0.5 s | 17 | 61.5 tok/s |

decode 与纯文本一致 —— 嵌入注入没有额外开销。

### 已知限制

- 一次请求只接受一个媒体。`dubhe_set_mm_embeds` 只保存一段区间，两段特征要
  两套前缀，等 Omni 那一轮改成多段。
- 塔能加载不等于模态能用：占位 token 在语言模型词表里。VL 模型挂音频塔会得到
  "这个语言模型的词表里没有 `<|audio|>`"。
- raw 转写偶尔把同一句重复一遍。流式下无法回退，暂不处理。
- `/api/load` 切换模型不会重挂 mmproj。

---

## [0.7.0] · 2026-10-29 · 视频与语音

### 新增

- **视频输入**（`--video`）：`AVAssetImageGenerator` 按 fps 均匀抽帧
  （`appliesPreferredTrackTransform` 处理竖拍旋转，奇数帧补帧），
  `pack_pixels2(rgb0, rgb1, …)` 把相邻两帧叠进一个 temporal patch。
  `--video-fps` / `--video-frames` / `--video-side` 控制抽帧与缩放。
- **语音输入**（`--audio`）：任意格式（wav/aiff/mp3/m4a/flac）经
  AVFoundation 解成 16 kHz 单声道 → log-mel → 17 层语音塔 → 2048 维嵌入。
  占位符 `<|audio|>`，按 `ceil(帧数/8)` 展开。
- **log-mel 前端**（`engine/src/mel.*`）：周期 Hann、`center=False`、400 点
  DFT（表驱动，窗已折进 cos/sin 表）、power 谱、128 个 Slaney mel 滤波器
  （Slaney 刻度 + 面积归一化）、`log10` 下限 −12、仿射 `(x+4)/4`。
- **语音塔**（`engine/src/audiomm.*`）：三层 stride-2 的 3×3 卷积降采样
  （128 mel → 16，T → T/8）→ 17 层 LFM2（conv/conv/attn ×5 + attn ×2，
  门控因果深度卷积宽 3、GQA 8/4 头、head_dim 96、滑窗 32 因果注意力）
  → embedding_norm → 投影器（LayerNorm → 768→2048 → GELU → 2048→2048）。
- **音频解码**（`engine/src/audiofile.*`）：AVAudioFile + AVAudioConverter。
- **共用核函数**（`engine/src/mm_math.h`）：量化 matmul、NEON 点积、
  RMSNorm/LayerNorm、SiLU/GELU、逐行并行，视觉塔与语音塔共用。
- **工具**：`audio_oracle.py`（HF 的 `Lfm3AudioEncoder` 真跑一遍当参考）、
  `audio_diff.py`（逐层对拍）、`audprobe`（逐层 dump）。

### 修复

- **跨模态漏 BOS**：LFM 系的 chat 模板第一行是 `{{- bos_token -}}`
  （`<|startoftext|>`），但 GGUF 里 `tokenizer.ggml.add_bos_token = false`。
  少了它模型不进入对话状态，而是把提示词当成待续写正文 —— 语音上表现为
  "转写对了、然后复读指令、然后胡说"，视觉上只是"答得糙"，所以一直没被发现。
  `run`（非 `--raw`）与 `chat` 现在都会补。
- **`AVAudioFile` 的 EOF 不是错误**：读到末尾返回 NO + `eofErr(-39)`，
  当成错误处理会让任何正常音频在最后一帧报错。

### 基准

- 语音：mel 前端 23 ms / 3.45 s 音频（实时率 ~150×），整塔 ~0.2 s（~17×）。
- 视频：8 帧 512×512（784 token）编码 28 s；prefill 88.8 tok/s。
- 精度：语音塔与 HF 逐层对拍 22 个检查点，相对误差 ≤6.2e-6，cos = 1.000000。

---

## [0.6.0] · 2026-10-29 · 多模态第一步：LFM3-VL 视觉塔

### 新增

- **mmproj 加载器**（`engine/src/mmproj.*`）：读 `clip.vision.*` 元数据，按实际
  存在的张量判定每层是 conv 还是 attention（`attn_q` → A，`conv2d_in_proj` → C），
  不靠外部配置猜。
- **图像预处理**（`engine/src/image.*`）：macOS 走 ImageIO（PNG/JPEG/HEIC/WebP
  + EXIF 方向），自写 Catmull-Rom 双三次缩放（与 PIL BICUBIC 对齐），
  `smart_resize` 与 HF 等价，`pack_pixels` 按 merge-block 序摊平。
- **视觉塔前向**：patchify → RMSNorm → 轴向 2D RoPE → 14 层 conv/attn 交替
  → 投影器（LayerNorm → 4 patch 拼 4096 → GELU → 2048）。
- **嵌入注入 ABI**：`dubhe_load_mmproj` / `dubhe_encode_image` /
  `dubhe_set_mm_embeds`。注入按**绝对位置**覆盖 token 嵌入。
- **CLI**：`dubhe run -m vl.gguf --mmproj mmproj.gguf --image pic.png -p '<|image|>…'`
- **工具**：`vl_oracle.py`（跑 HF 自己的 forward 当权威参考）、`vl_ref.py`
  （numpy 逐行转写）、`vl_diff.py`（逐层对拍）、`gguf_read.py`（共享读取器）、
  `vlprobe --dump`（逐层 dump）、`DUBHE_MM_TRACE=1`（分阶段计时）。

### 修复（四个静默 bug，都不报错、只让结果悄悄错）

- 门控卷积漏了**输出门**：`out_proj(C * conv)`，中间那段是门。
- RoPE 坐标用了 merge-block 号而不是 **patch 号**，等于把图当成缩小一倍的网格。
- `Gguf.get()` 对 4D 张量用 `reshape(-1, ne[0])`，只对 2D 成立；`patch_embd`
  是 `[16,16,3,1024]`，会得到 `(1536,16)` —— 形状刚好能乘，结果全错。
- patchify 的 matmul 没传步长：每 patch 1536 维只取前 768，却按 768 跨步，
  第 t 个 patch 跨进了相邻 patch。纯白区域看不出来，带细节的图才炸。

### 基准

| 项 | 结果 |
|---|---|
| 与 HF 官方 forward 对拍 | 16 个检查点全部一致，最大相对误差 **1.7e-5**，余弦 1.000000 |
| 640×640（400 token）视觉塔 | 24.9 s → **13.5 s**（1.84x） |
| 384×384（144 token）视觉塔 | ~8 s → **3.5 s** |
| matmul 吞吐 | 标量 22.4 → NEON **48 GFLOP/s**（515 GFLOP/图） |
| 端到端（含模型加载） | 红圆 → *a red circle*；蓝方块+绿长条 → *a blue square and a green rectangle* |

### 说明

- 视觉塔的 matmul 在 CPU 上已贴着 NEON 吞吐上限，再快只能上 Metal。
- 剩下的 2.3 s 主要是 softmax 的 2.87 亿次 `expf`。
- `make engine` 只构建 `build/dubhe`，探针要 `make tools` —— 这一轮因此跑了
  几次旧二进制，白追一个假 bug。

---

## [0.5.0] · 2026-10-29 · prefill 二维分块 GEMM + 权重预取

### 基准（LFM3-1.7B Q4_K_M，601 token prompt，6 轮交错配对取比值中位数）

| 版本 | prefill | decode（对照） | prefill 相对 |
|---|---:|---:|---:|
| HEAD `47b0e27` | 72.9 tok/s | 29.2 tok/s | 1.00x |
| + 二维分块 | 94.6 tok/s | 29.9 tok/s | 1.30x |
| + 权重预取 | **102.2 tok/s** | 32.4 tok/s | **1.42x** |

裸内核单层 GEMM（LFM3-1.7B 真实形状）：422 → 672–688 GFLOP/s。

> 绝对 tok/s 与 0.4.0 的读数**不可比**：那批是另一段窗口、机器负载不同。
> 本机 load 在几分钟内能从 2 漂到 15，只有同一批交错运行的比值可用。
> 0.4.0 里"prefill 66 vs MLX 495（7.5x）"是跨会话读数，已作废；
> 同会话配对后是 **6.11x**（102 / 716）。

### 新增
- **`mm2_*` 分块 GEMM 内核族**（f32 / f16 / q8_0 / q4_k / q6_k）：一个
  threadgroup = 8 个 simdgroup × 4 行 = 32 行输出，K 方向按 256 元素分片，
  激活读放大 ÷32。lane 固定负责 8 个连续元素，scale 解包退化成每块一次。
- **权重软件预取**：解量化拆成 `LD_*`（纯加载）/ `CV_*`（纯换算），在算当前
  K 片之前先发下一片的加载，把 600 拍的 DRAM 往返藏进 256 拍的计算里。
- `tools/ab.py`：交错 A/B/C 基准，**逐轮配对算比值再取中位数**，并用
  decode 做内部对照（decode 走 `mv_*`，改动只碰 prefill 时它的比值必须
  ≈1.00x，偏离多少就是残余漂移）。
- `tools/sweep_mm.py`：分块参数扫描。MSL 是字符串，`-D` 传不进去，每个
  变体必须改头文件重编译 —— 脚本负责改、编、跑、`finally` 还原。
- `tools/mma_probe.mm`：float4 FMA / half4 FMA / 8×8×8 MMA 三者对比。
- `tools/mm2_check.mm`：`mm2_*` 与旧 `mm_*`、与 CPU 参考实现三方对拍，
  外加 `--bench` 裸内核吞吐。
- Makefile 新增 `shader-check`（离线 `xcrun metal` 编译 MSL 字符串）与
  `mm2` 目标。

### 变更
- 激活**不经过 threadgroup memory**，直接读 device 内存。违反直觉但实测
  更快（560 → 653 GFLOP/s）：每个 lane 只读自己那 8 个元素，tgmem 唯一
  作用是"把搬运摊给 256 线程"，而每片一次 barrier 要等 400 拍、计算只有
  280 拍。去掉后静态 tgmem 归零、占用率上升。
- `engine/src/metal_kernels.h` 末尾新增 `namespace dubhe_msl` 常量真源：
  MSL 里的 `#define` 在字符串里，C++ 侧看不见，派发参数与工具都从这里取。

### 修复
- **Makefile 的 `engine` 假目标**：仓库里有同名目录，`make engine` 被当成
  "已存在、无先决条件"的文件目标，直接报 `Nothing to be done` 而**不重编
  任何东西**。一次参数 A/B 因此比了两个完全相同的二进制。已加
  `engine: $(BIN)` 别名。
- **跨会话指标作废**：进度书里 prefill `0.13`（66 / 495）、decode `0.54`
  （47.2 / 87.1）都是跨会话相除，已替换为同会话配对值（0.16 / 0.37），
  并在指标表里保留一行"已作废"说明，避免再被引用。

### 已实测否掉的方案（写下来避免重走）
| 方案 | 实测 |
|---|---|
| `MM_SG=6/8` | 236 / 218 GFLOP/s（基线 672）。越过 64 寄存器线，溢出 |
| `BMAX=16/32` | 405 / 351，每 token 反而慢 1.7x。acc 被 `MM_SG×BMAX` 锁死 |
| threadgroup memory 暂存 | 560（无 tgmem 653）。barrier 等 L2 往返不划算 |
| 上述 + 双缓冲 | 518。tgmem 翻倍把每 core 的 threadgroup 数从 4 压到 2 |
| `simdgroup_matrix` 8×8×8 MMA | 2.45 TFLOP/s **低于** float4 FMA 的 2.84（苹果是标量模拟） |
| fp16 FMA | 3.11 vs 2.84 TFLOP/s，只有 1.19x |

---

## [0.4.0] · 2026-10-29 · LFM3 tri-hybrid + 量化 matvec 重写

### 基准（LFM3-1.7B，M4 24GB，同一会话背靠背，load≈4）

| 后端 | 权重 | 体积 | decode | 有效带宽 |
|---|---|---:|---:|---:|
| 天枢 Metal | Q4_K_M | 1.10 GB | **47.2 tok/s** | 52 GB/s |
| 天枢 Metal | Q8_0 | 1.85 GB | 33.5 tok/s | 62 GB/s |
| 天枢 Metal | F16 | 3.47 GB | 20.8 tok/s | 72 GB/s |
| 天枢 CPU | Q4_K_M | 1.10 GB | 10.7 tok/s | 12 GB/s |
| 苹果 MLX | 4bit | 0.97 GB | 87.1 tok/s | 85 GB/s |
| 苹果 MLX | 8bit | 1.90 GB | 47.9 tok/s | 91 GB/s |

prefill（601 token）：天枢 66 tok/s，MLX 495 tok/s。完整矩阵见
[`docs/references/LFM3-精度与后端实测.md`](docs/references/LFM3-精度与后端实测.md)。

### 新增
- **LFM3 tri-hybrid 架构支持**（CPU + Metal 双路径）：13 个 ShortConv 层、
  6 个全局注意力层、5 个滑窗注意力层，外加每 head 的 QK RMSNorm、
  sigmoid 输出门、zero-tail partial RoPE、sandwich norm。层类型直接读
  GGUF 的 `head_count_kv` / `sliding_window_pattern` 两张逐层表。
- **Metal 融合内核 4 个**：`k_headnorm` / `k_gate_mul` / `k_conv`+`k_conv_shift`
  / `k_mul`，全部编进同一个 command buffer，每 token 仍只同步一次。
- `tools/tokprobe.cpp`：分词 + 贪心生成探针，用于区分"输出不同"是分词
  差异还是前向差异。
- `tools/mlx_lfm3.py`：补上 mlx-lm 缺失的 lfm3 实现，于是 Liquid 官方的
  `LFM3-*-MLX-4bit/8bit` 能在苹果 MLX 上跑 —— 既是独立参考实现，
  也是"苹果的版本"的速度基准。
- `tools/bench_matrix.py`：精度 × 后端全矩阵，输出 markdown 表格。
- `DUBHE_VERBOSE=1`：打印逐层结构表（层类型 / rope base / 角度 / 门）。
- `DUBHE_F32_REF=1`：强制 CPU 参考路径走纯 f32，把对拍噪声底从 1e-1 降到 1e-5。

### 变更
- **量化 matvec 重写**：Q4_K / Q6_K 从"一个 lane 在每个 32 元素段各取一个
  元素"改成"一个 lane 固定负责一个 scale 组的 8 个元素"。scale 查询从
  每块 8 次降到 1 次，指令数从 10 条/元素降到约 3 条；8 个量化字节用一条
  `uint2` / 4 条 `ushort` 向量读。decode **13.8 → 47.2 tok/s（3.4x）**。
- mv 内核改为"一个 simdgroup 同时算 `MV_R` 行"（`MV_R=2`）。
- 层结构里新增 `rope_angles_l`。

### 修复
- **滑窗层的 RoPE 角度**：HF 的 `rope_parameters` 是按层类型分开配的，
  `sliding_attention` 没有 `partial_rotary_factor`（缺省 1.0 → 64/64 全量），
  早先按全局一个值套所有层，5 个滑窗层的 q/k 有一半维度该转没转。
- `k_mul` 把字节偏移当元素下标（残差流 rel=0.23）、行跨度写死成 `n`
  （nrow=1 精确、nrow=2 崩）、`k_conv` 写 y 不带行偏移（多行全写第 0 行）。
- ShortConv 历史缓冲改为**先卷积、后移位**（反过来会让核整体偏移一位）。

### 基准
- ANE 实测：batch=1 无优势（脉动阵列空转），batch≥32 快 2–3 倍，
  `conv_in` 在 batch=128 上比 GPU 快 **9.7 倍**。结论：ANE 属于 prefill。

---

## [0.3.0] · 2026-10-29 · 底层三处结构性优化

### 基准（Qwen2.5-0.5B-Instruct Q4_K_M，M4 24GB，背靠背交替测量）

| 场景 | 上一版 | 本版 | 倍数 |
|---|---|---|---|
| 短上下文 prefill | 87–92 tok/s | **382 tok/s** | **4.3x** |
| 短上下文 decode | 78–79 tok/s | **79–84 tok/s** | 持平 |
| 631 tok prefill | 47 tok/s | **276–318 tok/s** | **6.0x** |
| 631 tok decode | 33–35 tok/s | **47–52 tok/s** | **1.5x** |

### 新增
- **批处理 prefill**（`mm_*` 内核 + `fwd_batch`）：权重读一遍喂 8 个 token，
  算术强度 ×8。prefill 内 `out_w` 只对最后一行计算（它是全模型最大的单个
  矩阵乘，占权重流量 36%）。
- **激活转置布局** `[n_in][BMAX]`：一个权重值的 8 次 fma 从 8 条标量加载
  变成 2 条 `float4`。转置时对 `j >= nrow` 填 0，内核里彻底去掉分支。
- **split-K 注意力**（`k_attn_split` + `k_attn_comb`）：长上下文把 key 区间
  切成 nsp 段并行算局部 (m, l, acc) 再归并。nsp 随上下文长度增长，上限 8。
- `make optest`：7 个 Metal 融合算子逐个与 C 参考实现对拍
  （`tools/gpu_optest.mm`），覆盖 rows=1/3、nsp=1/4、多种 head_dim。
- `metal::unbind_weights()`：卸载模型只释放权重绑定，不再拆全局 device /
  library（否则同进程里其它模型一起失效）。
- `dubhe_model_info.backend` 字段；`make check` 覆盖 GPU/CPU 双路径。

### 变更
- **matvec 并行粒度**：一线程一行 → 一个 simdgroup 一行（32 lane 分摊行内
  元素 + `simd_sum` 归约）。ffn_down 的线程数从 896 → 28672。
- Q4_K 的 min 项折进逐元素 fma，省掉一遍 x 求和。
- 注意力在 `nsp == 1`（短上下文）时走融合版内核——split 版在这个区间是
  纯亏的，实测会拖低 decode 25%。

### 修复
- **公共 lambda 漏绑 buffer**：`add`/`silu` 收进只 `setPipeline` 的公共
  lambda 后，`k_add` 读到上一次 `gemm` 的绑定，`d` 指向权重 mmap。症状是
  "输出等于输入 embedding"。
- **原地 rmsnorm 的 aliasing**：最后一层 norm 改为写入独立的 `g_bfinal`。

### 文档
- 进度书 v2.3：§9.6 记录三处优化、两个 bug、以及"绝对数字不可信，只信
  交替对比"的测量方法。

---

## [0.2.0] · 2026-10-29 · Metal GPU 后端

### 基准（Qwen2.5-0.5B-Instruct Q4_K_M，M4 24GB，本机）
| 指标 | CPU (NEON) | Metal GPU | 倍数 |
|---|---|---|---|
| decode | 20.6 tok/s | **60.9 tok/s** | 2.96x |
| prefill | 21.3 tok/s | **70.0 tok/s** | 3.29x |
| 与纯 f32 基准的 logits 差 | 3e-2（Q8 激活量化） | **1e-4** | — |

### 新增
- **Metal 计算后端**（`engine/src/metal.mm` + `metal_kernels.h`）：
  - **运行时编译 MSL**（`newLibraryWithSource:`）。绕开缺失的离线 Metal
    Toolchain（`xcodebuild -downloadComponent MetalToolchain`），任何装了
    Xcode CLT 的机器都能跑。
  - **权重零拷贝**：整块 GGUF mmap 直接包成 `MTLStorageModeShared` buffer，
    每张量用字节偏移定位。内存占用与 CPU 路径相同，不翻倍。
  - 量化内核：`mv_f32/f16/q4_0/q5_0/q8_0/q4_k/q6_k`（一线程一行，按 GGUF
    原始量化格式直读，不解包）。
  - 融合算子：`k_rmsnorm` / `k_rope` / `k_attn`（GQA + 在线 softmax）/
    `k_silu_mul` / `k_add` / `k_bias` / `k_copy`。
  - **整层前向编进一个 command buffer**，每 token 只同步一次。
  - **加载期交叉校验**：用真实权重跑 GPU vs 纯 f32 基准，相对误差 > 1e-3
    就整体退回 CPU（避免"能跑但胡言乱语"的静默错误）。
- `dubhe_model_info.backend`：暴露计算后端与校验误差。
- `DUBHE_F32_REF=1`：强制 CPU 走纯 f32 点积（校验 GPU 数值时必须关掉 Q8 激活
  量化，否则量的是 CPU 自己的近似误差）。
- `make gpu`：一条命令对比 CPU / GPU 吞吐。`make check` 现在两条路径都跑。

### 关键发现
- 本机 **单次 command buffer commit+wait 延迟 201 µs**。0.5B 模型单 token 有
  169 次 matvec，逐算子同步 = 34 ms，把 GPU 算力全埋掉。实测逐 matvec 同步
  的版本只比 CPU 快 8%。改成整层一个 command buffer 后才有 3x。
- 本机 GPU 实测读带宽 47 GB/s、CPU 17 GB/s（真机 M4 应分别在 100 GB/s 量级），
  环境受限，绝对数值不可当真机结论，加速比可信。

### 修复
- `k_attn` 的在线 softmax 在最大值更新时只缩放了分母 `l`，**忘了同步缩放已累加
  的 `acc`**——flash-attention 最经典的坑。表现为部分 head 输出错、部分正确
  （数据相关），输出乱码但程序不崩。
- MSL 里 `half` 是内建 16 位浮点类型名，不能当变量名。

---

## [0.1.0] · 2026-10-29 · M0 交付

### 新增
- **引擎**：GGUF v2/v3 解析（mmap 零拷贝）、11 种量化类型内核、
  GPT-2 字节级 BPE + Qwen2 pre-tokenize、Llama/Qwen2 前向
  （RMSNorm → QKV → RoPE → GQA → SwiGLU）、C ABI（`dubhe.h` 0.1.0）。
- **NEON 整数点积**：量化激活（Q8_0）+ `vdotq_s32`，覆盖 Q4_0/Q4_1/Q5_0/
  Q5_1/Q8_0/Q4_K/Q6_K。实测 1.4–3.8x，端到端 7.7 → 20.0 tok/s。
- **CLI**：`inspect` / `run` / `chat` / `bench`。
- **服务**：`dubhed`，OpenAI 兼容 `/v1/chat/completions`（含 SSE 流式）、
  `/v1/models`、`/api/state`。
- **Web UI**：iOS 风格单页应用（会话管理、Markdown + 代码块、工具卡片、
  设置面板、API 网关面板）。
- **工具调用**：`read_file` / `list_dir` / `write_file` / `shell`，
  模型输出 `<tool_call>` → 执行 → 结果回灌。
- **API 网关**：`--share` 监听 0.0.0.0 并自动生成 `sk-dubhe-…`；
  局域网需 `Authorization: Bearer`，本机回环免鉴权。
- **自检**：`tests/test_quant.cpp`（位解包 + 整数路径一致性）、
  `tools/selftest.cpp`、`tools/ggufdump.cpp`、`tools/ui_shot.mjs`。
- `bench/matvec_bench.cpp`：按真实模型形状测各类型吞吐。

### 修复（都是会让结果静默错掉的真实 bug）
- 字节映射表对 161–255 的字节用了原始字节而非其 UTF-8 编码，导致中文切不开。
- BPE 初始符号按 UTF-8 字符而非字节切分，merges 永不命中。
- Makefile 缺头文件依赖跟踪，改结构体后部分目标文件不重编 → ABI 错位段错误。
- KV cache 按训练上下文预分配（805 MB），改为按需翻倍。
- EOS / `<|im_end|>` 被当正文输出；HTTP 头名大小写敏感导致鉴权失效。

### 变更
- **首期范围扩大**：UI 从"推迟到 M5"提前到 M0（产品形态要求）。
- `dubhed` 从 M2 提前到 M0。

---

## [Unreleased] · v2.0 计划基线

### 变更（重要：v1 计划作废）

- **定名**：项目由代号 Aether 定名为 **天枢 / Dubhe**（北斗第一星 + 中枢；Dubhe 为该星天文名）。
- **开发模式**：由"8 人团队 × 63 周"改为 **AI 主开发 + 人类验证**，里程碑按 Round 计（27–43 轮，5–8 个月）。
- **首期范围收敛**：只交付引擎 + CLI + 本地 HTTP，**UI 推迟到 M5**。
- **许可证**：由 open-core（引擎 Apache-2.0）改为 **全闭源**，保留 4 条开源复审条件。
- **进程模型**：ADR-0005 引擎独立进程 **推迟到 M5**，M0–M4 单进程。

### 文档
- 重写《总项目开发书》v2.0：新增 §7 开发模式（协作协议）、§9 需你提供的环境信息、按 Round 排的里程碑。
- 重写《进度书》v2.0：M0 任务重切（9 项）、阻塞项 B1–B4、轮次记录模板。
- 新增 ADR-0007：AI 主开发 + 人类验证。
- 重写 ADR-0006：全闭源策略与"闭源 ≠ 不能用开源"的关键澄清。
- 新增 `LICENSE`（Proprietary）与 `THIRD_PARTY_NOTICES.md`（含排除清单）。

### 新增
- 仓库骨架：`engine/` `sdk/` `cli/` `server/` `apps/` `tools/` `bench/` `docs/`。
- `tools/docs/build_docx.sh` + 表格列宽预处理器（Markdown → docx，备用）。

---

## 版本规划

| 版本 | 里程碑 | 内容 |
| --- | --- | --- |
| 0.1.0 | M0 | GGUF 加载 + CPU reference + `dubhe run` 出字 |
| 0.2.0 | M1 | Metal 加速，B=1 追上 llama.cpp |
| 0.3.0 | M2 | 并发调度 + `dubhed` |
| 0.4.0 | M3 | MLX 直读 + 长上下文 + 结构化输出 |
| 0.5.0 | M4 | Android（CPU + Vulkan） |
| 1.0.0 | M5 | 双端 App + 打包发布 |

---

## v0.0.1-docs · 2026-10-29

### 文档
- 首版《总项目开发书》《进度书》《推理引擎开发书索引》。
- ADR-0001..0006 首批落档。
- 归档引擎开发书 v1.0 docx 至 `docs/references/`（正文仍用旧代号，属历史原貌）。
