/* Dubhe Web UI */
(() => {
  'use strict';

  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, txt) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  };

  // 图标统一走 index.html 里的 <symbol> 集，不在这里拼 SVG 路径
  const icon = (name, cls) =>
    `<svg class="${cls || 'i'}" viewBox="0 0 24 24" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const iconNode = (name, cls) => {
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('class', cls || 'i');
    s.setAttribute('viewBox', '0 0 24 24');
    s.setAttribute('aria-hidden', 'true');
    const u = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    u.setAttribute('href', '#i-' + name);
    s.appendChild(u);
    return s;
  };

  const STORE_KEY = 'dubhe.conv.v1';
  const SET_KEY = 'dubhe.settings.v1';
  // 用户在设置面板里亲手拨过"启用工具调用"的开关。
  // 有这个标记才认存储里的 tools 值；没有就说明那个 true 是旧默认值留下的。
  const TOOLS_TOUCHED = 'dubhe.tools.touched';

  const state = {
    convs: [],
    current: null,
    streaming: false,
    server: null,
    // 工具说明默认关。2B 级模型带工具说明会"讲计划"或"编造结果"，
    // 普通对话被污染的成本远大于偶尔能自动读个文件的收益。
    // 要用就走 Agent 视图（单次开启），或者在设置里显式打开。
    tools: false,
    // 联网搜索：只在本次会话有效。打开后每个问题前面会加一句"先搜再答"，
    // 并临时打开工具（否则模型没有 web_search 可用，开关等于摆设）。
    web: false,
    attach: null,   // 待发送的图片/音频/视频（一次只允许一个）
    logs: [],       // 右栏系统日志：只记真实事件，不编
    gen: null,      // 本轮生成统计 { tokens, tps, total, ttft }
    q: '',          // 左栏搜索关键词
    abort: null,    // 生成中的 AbortController，用于"停止"
    view: null,     // 当前打开的功能视图（null = 对话）
    fsPath: '',     // 文件视图当前目录
    memory: [],     // 记忆条目（见 loadMemory）
  };

  const settings = Object.assign(
    { temperature: 0.7, top_p: 0.95, max_tokens: 1024, system: '', tools: false },
    JSON.parse(localStorage.getItem(SET_KEY) || '{}')
  );
  // 迁移：老版本的默认值是 tools:true，而且它随其它设置一起被存进了
  // localStorage。光把默认值改成 false 救不了老用户 —— Object.assign
  // 会被存储里的 true 盖掉。没拨过开关就说明这个 true 不是用户选的。
  if (!localStorage.getItem(TOOLS_TOUCHED)) settings.tools = false;
  // 输入框旁的工具栏按钮切的是 state.tools（本次会话），设置面板切的是
  // settings.tools（持久偏好）。两个开关如果各判各的，就会出现"按钮亮着
  // 但没生效"这种没法解释的状态。启动时以偏好为准，之后按钮独立。
  state.tools = settings.tools;

  // ---------------------------------------------------------------- 存储
  //
  // localStorage 只有 5 MB 左右，写满时 setItem 抛 QuotaExceededError。
  // 这里原来的 catch 是空的 —— 于是"对话突然不再保存"，重启后丢一片，
  // 界面上一个字都不提示。工具结果动辄 8 KB（read_file 的正文、diff 的
  // old/new），一次会话几十条就上千 KB，这个上限不是理论上的。
  //
  // 两道处理：
  //   1) 存之前把只对"看"有用的东西截短：工具 output 留 1500 字（回放时
  //      卡片照样画得出来，meta 里的 path/command/diff 全部保留），
  //      搜索结果的摘要不留。
  //   2) 还是写不进去，就从最老的对话开始丢（绝不丢当前打开的那条），
  //      丢到能写进去为止，并且明确告诉用户丢了哪一条 —— 静默丢数据
  //      是所有处理方式里最坏的一种。
  const TOOL_OUT_KEEP = 1500;

  function slimTool(t) {
    const o = { name: t.name, ok: t.ok, card: t.card };
    if (t.ms) o.ms = Math.round(t.ms);
    if (t.denied) o.denied = 1;
    if (t.output) {
      const s = String(t.output);
      o.output = s.slice(0, TOOL_OUT_KEEP);
      if (s.length > TOOL_OUT_KEEP) o.trunc = 1;
    }
    if (t.meta) {
      const m = Object.assign({}, t.meta);
      if (Array.isArray(m.results)) m.results = m.results.map((r) => ({ t: r.t, u: r.u }));
      o.meta = m;
    }
    return o;
  }

  function slimConv(c) {
    const out = Object.assign({}, c);
    out.messages = (c.messages || []).map((m) => {
      if (!m.tools || !m.tools.length) return m;
      const mm = Object.assign({}, m);
      mm.tools = m.tools.map(slimTool);
      return mm;
    });
    return out;
  }

  function saveConvs() {
    for (let guard = 0; guard < 24; guard++) {
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify(state.convs.map(slimConv)));
        return;
      } catch (e) {
        const idx = state.convs.findIndex((c) => c !== state.current);
        if (idx < 0) {
          log('对话没能保存到本地：存储空间不足', 'err');
          return;
        }
        const gone = state.convs.splice(idx, 1)[0];
        log('本地存储满了，已丢弃最早的对话「' + (gone.title || gone.id) + '」', 'err');
      }
    }
    log('对话没能保存到本地：存储空间不足', 'err');
  }
  function loadConvs() {
    try { state.convs = JSON.parse(localStorage.getItem(STORE_KEY) || '[]'); } catch (e) { state.convs = []; }
    if (!Array.isArray(state.convs)) state.convs = [];
  }
  function saveSettings() { localStorage.setItem(SET_KEY, JSON.stringify(settings)); }

  // ---------------------------------------------------------------- 记忆
  //
  // 两个作用域：
  //   conv:<id> —— 只属于这一个对话。默认作用域，写进来的东西不会跑到别的
  //                对话里去（"不同对话会串记忆"这条问题的正面解法）。
  //   global    —— 显式选择才写，所有对话共享。
  //
  // 为什么存在浏览器而不是服务端：对话本身就只存在这里（localStorage），
  // 记忆是对话的附属物，放到服务端反而会出现"对话删了、记忆还在"的错位。
  const MEM_KEY = 'dubhe.memory.v1';
  function loadMemory() {
    try {
      const a = JSON.parse(localStorage.getItem(MEM_KEY) || '[]');
      return Array.isArray(a) ? a : [];
    } catch (e) { return []; }
  }
  function saveMemory() {
    try { localStorage.setItem(MEM_KEY, JSON.stringify(state.memory)); } catch (e) {}
  }
  function convScope() {
    return 'conv:' + (state.current ? state.current.id : 'none');
  }
  function addMemory(text, scope) {
    // 存之前把 markdown 的强调标记去掉：记忆是纯文本背景，留着 ** 和 ```
    // 只会让模型在回答里也学着加，界面上看着也脏。
    const t = String(text || '')
      .replace(/\*\*|__|`{1,3}/g, '')
      .replace(/^\s*[-*+]\s+/gm, '')
      .replace(/^#+\s+/gm, '')
      .trim();
    if (!t) return null;
    const m = { id: 'm' + Date.now() + Math.random().toString(36).slice(2, 6), scope, text: t, at: Date.now() };
    state.memory.unshift(m);
    if (state.memory.length > 500) state.memory.length = 500;
    saveMemory();
    return m;
  }
  // 本对话能看到的记忆：自己的 + 全局的
  function memoryForConv(conv) {
    const sc = 'conv:' + (conv ? conv.id : 'none');
    return state.memory.filter((m) => m.scope === sc || m.scope === 'global');
  }

  // ---------------------------------------------------------------- 上下文压缩
  //
  // 触发条件是"消息条数"，不是"token 数"：前端拿不到准确 token 数，
  // 而按条数触发的好处是行为可预测、不需要额外请求。每条消息的长短差异
  // 由 KEEP_RECENT 和摘要本身的长度上限兜住。
  const COMPRESS_AT = 16;   // 未折叠的消息超过这个数就压
  const KEEP_RECENT = 8;    // 无论如何保留最近这么多条原文

  async function maybeCompress(conv) {
    const folded = conv.folded || 0;
    const total = conv.messages.length;
    if (total - folded <= COMPRESS_AT) return;
    const end = total - KEEP_RECENT;
    if (end <= folded) return;
    const slice = conv.messages.slice(folded, end)
      .filter((m) => m.content && (m.role === 'user' || m.role === 'assistant'))
      .map((m) => ({ role: m.role, content: m.content }));
    if (slice.length < 4) return;
    log('上下文压缩中（' + slice.length + ' 条）…');
    try {
      const r = await vpost('/api/summarize', { messages: slice });
      const sum = (r && r.summary ? String(r.summary) : '').trim();
      if (!sum) { log('压缩没拿到摘要，本轮按原样发', 'err'); return; }
      conv.summary = (conv.summary ? conv.summary + '\n' : '') + sum;
      conv.folded = end;
      saveConvs();
      log('已压缩 ' + slice.length + ' 条 → 摘要 ' + sum.length + ' 字');
      renderMessages();
    } catch (e) {
      // 压缩失败不该挡住这一轮对话：原样发出去，下次再试
      log('压缩失败：' + e.message, 'err');
    }
  }

  function newConv() {
    // 新对话默认继承上一个对话绑定的项目文件夹 —— 像 Codex 那样"开个新
    // 会话继续在同一个项目里干活"，而不是每次都要重新选一遍目录。
    const inherit = state.current && state.current.cwd;
    const c = { id: 'c' + Date.now(), title: '新对话', messages: [], at: Date.now() };
    if (inherit) c.cwd = inherit;
    state.convs.unshift(c);
    state.current = c;
    saveConvs();
    renderConvList();
    renderMessages();
    renderCwd();
    $('#convTitle').textContent = '新对话';
    $('#input').focus();
    return c;
  }

  // ---------------------------------------------------------------- 项目文件夹
  //
  // 一个对话 = 一个项目文件夹。工具的相对路径以它为根，服务端每次请求
  // 单独收（POST body 里的 workspace），所以两个对话可以各自待在各自的
  // 目录里，不会互相看见对方读的是哪个项目。
  function renderCwd() {
    const chip = $('#cwdChip');
    const name = $('#cwdName');
    if (!chip || !name) return;
    const c = state.current;
    const p = (c && c.cwd) || (state.server && state.server.workspace) || '';
    if (!p) {
      name.textContent = '未绑定';
      chip.classList.add('unbound');
      chip.title = '本对话还没有绑定项目文件夹';
      return;
    }
    chip.classList.remove('unbound');
    const seg = p.split('/').filter(Boolean);
    name.textContent = seg.length ? seg[seg.length - 1] : p;
    chip.title = (c && c.cwd ? '项目文件夹：' : '默认工作目录：') + p;
  }

  // ---------------------------------------------------------------- Markdown
  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
  }

  function inline(s) {
    return s
      .replace(/`([^`\n]+)`/g, (m, c) => `<code>${c}</code>`)
      .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  // 代码块外壳。
  //
  // 三种语言可以直接跑：html / htm / svg。其余只给复制。
  // 判断放在这里而不是渲染后扫 DOM：流式渲染时同一个块会被反复重建，
  // 扫 DOM 的做法要么漏、要么每次都得重挂事件。
  const RUNNABLE_LANG = { html: 1, htm: 1, svg: 1, xml: 1 };
  function codeBlockHtml(lang, code) {
    const raw = (lang || '').trim().toLowerCase();
    const first = raw.split(/[\s,{]/)[0] || '';
    const label = first || 'text';
    const runnable = !!RUNNABLE_LANG[first];
    return '<div class="codeblock" data-lang="' + label + '">'
      + '<div class="cb-head"><span class="cb-lang">' + label + '</span>'
      + '<span class="cb-actions">'
      + (runnable ? '<button class="cb-btn run" data-run>运行</button>' : '')
      + '<button class="cb-btn" data-copy>复制</button>'
      + '</span></div>'
      // tabindex 让 pre 可聚焦 —— 只有聚焦的元素方向键才滚它自己
      + '<pre tabindex="0"><code>' + code + '</code></pre></div>';
  }

  function renderMarkdown(src) {
    const lines = escapeHtml(src).split('\n');
    let html = '';
    let inCode = false, codeLang = '', codeBuf = [];
    let listType = null;

    const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];
      const line = raw.replace(/\s+$/, '');

      if (/^```/.test(line)) {
        if (!inCode) { inCode = true; codeLang = line.slice(3).trim(); codeBuf = []; closeList(); }
        else {
          inCode = false;
          html += codeBlockHtml(codeLang, codeBuf.join('\n'));
        }
        continue;
      }
      if (inCode) { codeBuf.push(raw); continue; }

      if (!line.trim()) { closeList(); continue; }

      const h = line.match(/^(#{1,3})\s+(.*)$/);
      if (h) { closeList(); html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; continue; }

      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ul) {
        if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; }
        html += `<li>${inline(ul[1])}</li>`;
        continue;
      }
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ol) {
        if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; }
        html += `<li>${inline(ol[1])}</li>`;
        continue;
      }
      const bq = line.match(/^>\s?(.*)$/);
      if (bq) { closeList(); html += `<blockquote>${inline(bq[1])}</blockquote>`; continue; }

      closeList();
      html += `<p>${inline(line)}</p>`;
    }
    if (inCode && codeBuf.length) html += codeBlockHtml(codeLang, codeBuf.join('\n'));
    closeList();
    return html;
  }

  // 隐藏模型输出的 tool_call 原文
  function stripToolCalls(s) {
    return s.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').replace(/<tool_result>[\s\S]*?<\/tool_result>/g, '').trim();
  }

  // 流式渲染的"安全切点"：返回一个下标，之前的文本已经定型、不会再变。
  //
  // 两个安全位置：空行之后，以及 ``` 围栏闭合之后。切在空行上不会把段落
  // 劈成两半；切在围栏之后不会把代码块拆开（拆开的话每帧都在重建它，
  // 代码块内部的滚动位置就保不住）。
  // 返回 0 表示还没有任何定型内容，整段都算尾巴。
  function stableCut(s) {
    const lines = s.split('\n');
    let off = 0, safe = 0, inFence = false;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      off += line.length + 1;
      if (/^```/.test(line)) {
        inFence = !inFence;
        if (!inFence) safe = off;      // 围栏刚闭合，整块都定型了
        continue;
      }
      if (inFence) continue;
      if (!line.trim()) safe = off;
    }
    return safe;
  }

  // 助手正文的 HTML。
  //
  // 为什么要单独一个函数：小模型（尤其带图的场景）有时会把整段输出包进
  // <tool_call>/<tool_result> 里，但并没有真的调用工具。stripToolCalls
  // 一剥，什么都不剩，界面上就是"已完成 · 9 tokens"加一片空白 ——
  // 用户完全不知道发生了什么，也没法判断是不是自己用错了。
  // 所以剥完是空、又没有工具卡片可看的时候，把原文露出来：难看，但诚实。
  function bodyHtml(content, hasTools) {
    const raw = (content || '').trim();
    const visible = stripToolCalls(raw);
    if (visible) return renderMarkdown(visible);
    if (!raw || hasTools) return '';
    return '<div class="raw-note">模型这段输出被识别成工具调用，没有可显示的正文字。原文：</div>'
      + '<pre class="raw-out">' + escapeHtml(raw.slice(0, 2000)) + '</pre>';
  }

  // ---------------------------------------------------------------- 渲染
  function renderConvList() {
    const list = $('#convList');
    list.innerHTML = '';
    if (!state.convs.length) {
      list.appendChild(el('div', 'conv-empty', '还没有对话。点上面的 ＋ 开始。'));
      return;
    }
    const q = (state.q || '').trim().toLowerCase();
    const items = q ? state.convs.filter((c) => (c.title || '').toLowerCase().includes(q)) : state.convs;
    if (!items.length) {
      list.appendChild(el('div', 'conv-empty', '没有匹配的对话'));
      return;
    }
    for (const c of items) {
      const active = state.current && c.id === state.current.id;
      const d = el('div', 'conv' + (active ? ' active' : ''));
      const ci = el('span', 'ci');
      ci.appendChild(iconNode('chat', 'i'));
      d.appendChild(ci);
      d.appendChild(el('span', 'conv-title', c.title));
      d.onclick = () => {
        state.current = c;
        renderConvList();
        renderMessages();
        renderTools();
        renderCwd();
        $('#convTitle').textContent = c.title;
        if (window.innerWidth <= 860) closeSidebar();
      };
      list.appendChild(d);
    }
  }

  // ---- 工具卡片
  //
  // 服务端在 tool_result 里带 card + meta，这里只做映射：不读时钟、不做 I/O、
  // 不看全局状态。原因是同一段数据要在两个时刻被画出来 —— 实时流进来的时候，
  // 以及以后从会话记录回放的时候。只要它依赖了"现在几点"或"当前工作目录"，
  // 回放出来的卡片就和当时看到的不一样（DSH 把这条写成硬规则：展示函数必须是
  // 纯函数）。
  //
  // card 缺失（老会话记录、或工具没声明）一律走 generic，不抛错。
  const CARD_ICON = {
    terminal: 'terminal', diff: 'code', read: 'txt',
    search: 'search', web: 'globe', generic: 'terminal',
  };

  const baseName = (p) => String(p || '').split('/').filter(Boolean).pop() || '';
  const hostOf = (u) => {
    const m = String(u || '').match(/^https?:\/\/([^/?#]+)/i);
    return m ? m[1] : String(u || '');
  };

  // 一行摘要：卡片头部右边那行小字。写不出东西就返回空串，不留空位。
  function cardSub(t, meta) {
    if (t.denied) return '未执行';
    if (t.ok === false) return '失败';
    const card = t.card || 'generic';
    if (card === 'read' && meta.total) {
      const a = meta.offset || 1, b = a + (meta.lines || 0) - 1;
      return a + '–' + b + ' / ' + meta.total + ' 行';
    }
    if (card === 'terminal') {
      if (meta.timed_out) return '超时';
      if (meta.exit != null) return '退出 ' + meta.exit;
    }
    if (card === 'diff') return meta.is_new ? '新建' : '已改';
    if (card === 'search') {
      if (meta.truncated) return '已截断';
      if (meta.shape === 'kb') return (meta.count || 0) + ' 段';
      return meta.shape === 'matches' ? (meta.total || 0) + ' 处' : (meta.count || 0) + ' 个';
    }
    if (card === 'web') return meta.kind === 'fetch' ? '抓取' : (meta.count || 0) + ' 条';
    return '';
  }

  // 卡片主标题 = "它到底动了什么"，不是工具名。用户不关心工具叫 read_file，
  // 只关心读了哪个文件、跑了哪条命令。
  function cardTitle(t, meta) {
    const card = t.card || 'generic';
    if (card === 'terminal') return meta.command || t.name;
    if (card === 'diff' || card === 'read') return baseName(meta.path) || t.name;
    if (card === 'search') return meta.pattern || baseName(meta.path) || t.name;
    if (card === 'web') return meta.kind === 'fetch' ? hostOf(meta.url) : ('“' + (meta.query || '') + '”');
    return t.name;
  }

  // ---------------------------------------------------------------- 引用联动
  //
  // 模型回答联网内容时会写 [1] [2]，卡片上也有编号 1..6 的来源 —— 但两边原来
  // 谁也不认谁：正文里的 [2] 只是一段普通文字，用户得自己数到卡片第二行再点。
  // 这里把 [n] 换成指向第 n 条来源的链接，点开的同时把卡片里那一行闪一下，
  // "这句话是从哪来的"一步就能落地。
  //
  // 只在这条消息**真的带了结构化搜索结果**时才做。别的时候 [1] 可能是
  // Markdown 脚注、是数组下标、是正则里的字符类 —— 一律不能动。

  // 一条消息里可能搜了两次，两套编号都从 1 开始，混起来必然错位。
  // 取最后一次搜索：回答里引的几乎总是最后那次的结果。
  function citeTableOf(tools) {
    const list = tools || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const t = list[i];
      if (t.card !== 'web' || !t.meta || t.meta.kind !== 'search') continue;
      const hits = t.meta.results;
      if (Array.isArray(hits) && hits.length) return hits;
    }
    return null;
  }

  // 这些容器里的 [1] 一定不是引用：代码、链接文字、内联代码。
  const CITE_SKIP = { PRE: 1, CODE: 1, A: 1, SCRIPT: 1, STYLE: 1 };

  // 点到 [2]：把卡片里第 2 条来源滚进视野并闪一下。
  // 用 closest('.msg') 而不是闭包里存节点：消息重绘（reswap）之后旧节点已经
  // 不在文档里了，闭包存的那份会把高亮闪在一个看不见的地方。
  function flashCite(a, i) {
    const msg = a.closest ? a.closest('.msg') : null;
    if (!msg) return;
    const row = msg.querySelectorAll('.cite-table .web-item')[i];
    if (!row) return;
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    row.classList.remove('flash');
    void row.offsetWidth;   // 强制回流，否则连点两次第二次不会重放动画
    row.classList.add('flash');
    clearTimeout(row._ft);
    row._ft = setTimeout(() => row.classList.remove('flash'), 1400);
  }

  function linkCites(root, hits) {
    if (!root || !hits || !hits.length) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue || n.nodeValue.indexOf('[') < 0) return NodeFilter.FILTER_SKIP;
        if (CITE_SKIP[n.parentNode && n.parentNode.nodeName]) return NodeFilter.FILTER_SKIP;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const texts = [];
    while (walker.nextNode()) texts.push(walker.currentNode);

    for (const node of texts) {
      const s = node.nodeValue;
      // [1] / [1][3] / [1, 3] / [1，3]。中文逗号也得认 —— 模型写中文回答时
      // 经常用全角，只认半角会让一半引用点不动。
      const re = /\[(\d{1,2}(?:\s*[,，]\s*\d{1,2})*)\]/g;
      let m, last = 0, frag = null;
      while ((m = re.exec(s))) {
        const nums = m[1].split(/[,，]/).map((x) => parseInt(x, 10));
        // 有一个越界就整段不动。拆一半的结果是"[1, 99]"变成"[1, 99" ——
        // 比不处理更难看，而且会让用户以为模型引了个不存在的来源。
        if (nums.some((n) => !(n >= 1 && n <= hits.length))) continue;
        if (!frag) frag = document.createDocumentFragment();
        frag.appendChild(document.createTextNode(s.slice(last, m.index)));
        nums.forEach((n, k) => {
          if (k) {
            const sepCh = (m[1].match(/[,，]/) || [','])[0];
            frag.appendChild(document.createTextNode(sepCh === '，' ? '，' : ', '));
          }
          const h = hits[n - 1];
          const a = document.createElement('a');
          a.className = 'cite';
          a.href = h.u || '#';
          a.target = '_blank';
          a.rel = 'noreferrer noopener';
          a.textContent = '[' + n + ']';
          a.title = (h.t || h.u || '') + (h.u ? '\n' + h.u : '');
          a.onclick = () => flashCite(a, n - 1);
          frag.appendChild(a);
        });
        last = m.index + m[0].length;
      }
      if (!last) continue;
      frag.appendChild(document.createTextNode(s.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  function toolCard(t) {
    const meta = t.meta || {};
    const card = t.card || 'generic';

    // 审批卡：这条调用被门禁要求确认，服务端正在等（最多 90 秒）。
    // 它和普通工具卡的区别是"还没执行"，所以不能显示输出，只能显示
    // 要做什么 + 两个按钮。
    if (t.askId) {
      const wait = t.askPending;
      const box2 = el('div', 'tool-card ask' +
        (wait ? '' : (t.askAllow ? ' allowed' : ' denied')));
      const head = el('div', 'tool-head');
      const ic = el('span', 'ic');
      ic.appendChild(iconNode(wait ? 'shield' : (t.askAllow ? 'check' : 'x'), 'i'));
      head.appendChild(ic);
      head.appendChild(el('span', 'tool-name', t.name));
      head.appendChild(el('span', 'tool-sub',
        wait ? '等待确认' : (t.askAllow ? '已允许' : (t.askTimeout ? '超时未答' : '已拒绝'))));
      box2.appendChild(head);
      if (t.askDetail) box2.appendChild(el('div', 'ask-detail', t.askDetail));
      if (wait) {
        const acts = el('div', 'ask-acts');
        const yes = el('button', 'ask-btn yes', '允许');
        yes.onclick = () => { yes.disabled = true; no.disabled = true; approve(t.askId, true); };
        const no = el('button', 'ask-btn', '拒绝');
        no.onclick = () => { yes.disabled = true; no.disabled = true; approve(t.askId, false); };
        acts.appendChild(yes);
        acts.appendChild(no);
        box2.appendChild(acts);
      }
      return box2;
    }

    const box = el('div', 'tool-card' + (t.ok === false ? ' failed' : '') + (t.denied ? ' denied' : ''));

    const head = el('div', 'tool-head');
    const ic = el('span', 'ic');
    ic.appendChild(iconNode(t.denied ? 'shield' : (t.ok === false ? 'x' : (CARD_ICON[card] || 'terminal')), 'i'));
    head.appendChild(ic);
    head.appendChild(el('span', 'tool-name', cardTitle(t, meta)));
    const sub = cardSub(t, meta);
    // 沙箱没生效这件事必须让用户看见。
    //
    // 原来服务端把这句提示塞进命令的 stdout，模型会把"当前环境不支持沙箱"
    // 当成命令输出的一部分拿去引用；而 /api/state 里又一直写着 sandbox:true，
    // 界面上一切正常。结果是"以为有写盘隔离，其实裸跑"。现在隔离状态走 meta，
    // 在这里画一个琥珀色小标，只有真的没隔离时才出现。
    // 放在 sub 之前：sub 有 margin-left:auto，会把自己和后面的东西一起推到右边。
    if (card === 'terminal' && meta.no_sandbox && !t.denied) {
      head.appendChild(el('span', 'tool-chip warn', '未隔离'));
    }
    if (sub) head.appendChild(el('span', 'tool-sub', sub));
    box.appendChild(head);

    // diff 卡片单独画：并排的"旧/新"两段比一整块 pre 有用得多 ——
    // 尤其 edit_file 只改一处的时候，用户要确认的是"改的是不是这一处"。
    if (card === 'diff' && (meta.old || meta.new) && !t.denied) {
      const d = el('div', 'tool-diff');
      if (meta.old) {
        const del = el('pre', 'diff-del', String(meta.old));
        d.appendChild(del);
      }
      if (meta.new) d.appendChild(el('pre', 'diff-add', String(meta.new)));
      box.appendChild(d);
      return box;
    }

    if (t.output) {
      // 联网搜索结果单独画成来源列表：一堆 "[1] 标题 / 缩进 URL / 摘要" 的
      // 纯文本在卡片里既占地方又点不了，而这类结果的价值恰恰在于"能点开核一下"。
      // 只有服务端给了结构化 results 才走这条路，老对话里存的是纯文本，仍然走下面。
      const hits = card === 'web' && meta.kind === 'search' ? meta.results : null;
      if (Array.isArray(hits) && hits.length) {
        const list = el('div', 'web-list');
        hits.forEach((h, i) => {
          const row = el('div', 'web-item');
          const a = document.createElement('a');
          a.className = 'web-title';
          a.href = h.u || '#';
          a.target = '_blank';
          a.rel = 'noreferrer noopener';
          a.textContent = h.t || h.u || '(无标题)';
          row.appendChild(el('span', 'web-idx', String(i + 1)));
          const mid = el('div', 'web-mid');
          mid.appendChild(a);
          if (h.u) mid.appendChild(el('div', 'web-host', hostOf(h.u)));
          if (h.s) mid.appendChild(el('div', 'web-snip', String(h.s).slice(0, 160)));
          row.appendChild(mid);
          list.appendChild(row);
        });
        box.appendChild(list);
        return box;
      }
      // 去掉只对模型有用的尾巴。服务端在 output 里写"…（这是第 1–3 行，全文共
      // 245 行）"和"[退出码 0]"，是为了让模型知道要不要接着翻、命令有没有成功；
      // 卡片头上已经有"1–3 / 245 行"和"退出 0"了，正文再写一遍是重复。
      // 只改展示，不动 t.output —— 回放和复制拿到的还是原文。
      //
      // 被拒的那条整个换成 note：output 里写的是"不要重试同一个调用，也不要
      // 声称它成功了"这类**对模型说的话**，摆在卡片上就是程序在跟用户讲黑话。
      let text = t.denied && meta.note ? String(meta.note) : String(t.output);
      if (card === 'read') text = text.replace(/\n…（这是第 [\s\S]*$/, '');
      if (card === 'terminal') text = text.replace(/\n\[退出码 -?\d+\]$/, '');
      const tb = el('pre', 'tool-body');
      tb.textContent = text.slice(0, 1200);
      box.appendChild(tb);
      // 存盘时工具输出被截过（见 saveConvs）。回放时要说清楚，否则用户会
      // 以为工具当时只返回了这么多 —— 那等于替他改动记忆。
      // 这句必须挂在 pre **外面**：pre 里的正文自己就截到 1200 字，写进
      // text 末尾等于写进了看不见的地方。
      if (t.trunc) {
        box.appendChild(el('div', 'tool-note',
          '本地只留了前 ' + TOOL_OUT_KEEP + ' 字，完整结果只在那次请求里'));
      }
    }
    return box;
  }

  // 把"允许/拒绝"送回服务端。服务端那条 chat 请求正在等这个答案。
  // 送不到（网络断、服务重启）时不要静默：用户会以为点了没反应，
  // 实际上是 90 秒后按拒绝处理 —— 那也得让他知道。
  async function approve(id, allow) {
    try {
      await vpost('/api/approve', { id, allow });
      log(allow ? '已允许这次操作' : '已拒绝这次操作');
    } catch (e) {
      log('审批没送到：' + e.message + '（90 秒后会自动按拒绝处理）', 'err');
    }
  }

  // 同一条消息第二次、第三次画出来时不要再放"升起"动画。
  //
  // .msg 上挂着 animation: rise .62s —— 那是给"新消息出现"用的。可工具结果
  // 一到、审批卡一变，整个气泡都会被 messageNode 重画一次，动画跟着重放：
  // 长回答 + 两次工具调用 = 屏幕上闪三下，而且每次重画都会把滚动位置顶一下。
  // 用户说的"生成时界面闪烁"有一半就是它。
  //
  // 用 WeakSet 而不是在消息对象上打标记：state.convs 整个被 JSON 序列化进
  // localStorage，多一个字段就多存一份垃圾。
  const drawn = new WeakSet();
  function markDrawn(node, m) {
    if (drawn.has(m)) node.classList.add('no-rise');
    else drawn.add(m);
    return node;
  }

  function messageNode(m) {
    if (m.role === 'user') {
      const wrap = el('div', 'msg user');
      const b = el('div', 'bubble');
      if (m.att) {
        if (m.att.kind === 'image' && m.att.thumb) {
          const im = new Image();
          im.className = 'thumb';
          im.src = m.att.thumb;
          im.alt = m.att.name;
          b.appendChild(im);
        } else {
          // 图标名必须是 index.html 里真实存在的 <symbol>：这里原来写的是
          // 'audio' / 'video' / 'file'，而 symbol 集里只有 msg-audio /
          // msg-video / msg-file —— 于是附件卡片永远是一个空方块。
          const kind = m.att.kind === 'audio' ? 'msg-audio'
            : (m.att.kind === 'video' ? 'msg-video' : 'msg-file');
          const f = el('div', 'mfile');
          f.appendChild(iconNode(kind, 'i mfile-i'));
          f.appendChild(el('span', '', m.att.name));
          b.appendChild(f);
        }
      }
      if (m.content) b.appendChild(document.createTextNode(m.content));
      wrap.appendChild(b);
      return markDrawn(wrap, m);
    }

    // 助手消息：头像 + 内容列。头像不是装饰 —— 它让"谁在说话"不用读文字就分得清。
    const wrap = el('div', 'msg assistant');
    const av = el('div', 'avatar');
    av.appendChild(iconNode('logo', 'i'));
    wrap.appendChild(av);
    const col = el('div', 'col');
    wrap.appendChild(col);

    const st = el('div', 'msg-status');
    if (m.meta && (m.meta.tokens || m.meta.tps)) {
      st.appendChild(el('span', 'dot-ok'));
      st.appendChild(el('span', '', '已完成'));
      if (m.meta.tokens) {
        st.appendChild(el('span', 'sp', '·'));
        st.appendChild(el('span', '', m.meta.tokens + ' tokens'));
      }
      if (m.meta.tps) {
        st.appendChild(el('span', 'sp', '·'));
        st.appendChild(el('span', '', m.meta.tps.toFixed(1) + ' tok/s'));
      }
      col.appendChild(st);
    } else if (m.streaming) {
      // 排队、预填充、生成是三件不同的事，都不出字，但用户该知道自己在等什么。
      st.appendChild(el('span', '', m.queued ? '排队中…'
        : (m.noteText || '生成中…')));
      col.appendChild(st);
    }

    const body = el('div', 'body');
    body.innerHTML = bodyHtml(m.content, m.tools && m.tools.length);
    col.appendChild(body);

    // 服务端在流里报的错。不显示出来的话，这一轮就是一个永远转着
    // "生成中…"、一个字都没有的空气泡 —— 比报错难查得多。
    if (m.errText) {
      const e = el('div', 'fake-tool', '⚠️ ' + m.errText);
      col.insertBefore(e, body);
    }

    // 模型编造工具结果。
    //
    // 小模型（2B 级）看到工具说明后，经常不去调工具，而是直接吐一段
    // <tool_result>{"content":"..."}</tool_result> —— 内容看着像真的读到了
    // 文件，其实是编的。实测让它"读 README.md 第一行"，它编出了
    // {"content":"# README\n..."}，服务端一次工具都没执行。
    // 这比空白危险得多：用户会当真。所以在气泡上直接标出来。
    if (!(m.tools && m.tools.length) && /<tool_result>/.test(m.content || '')) {
      const warn = el('div', 'fake-tool');
      warn.textContent = '注意：这段内容里的工具结果是模型编造的，实际没有执行任何工具。';
      col.insertBefore(warn, body);
    }

    if (m.tools && m.tools.length) {
      // 引用联动得先定"编号的出处"：把选中的那张来源卡标出来（.cite-table），
      // 点正文里的 [2] 才知道该闪哪一行。
      const tbl = citeTableOf(m.tools);
      let tblNode = null;
      for (const t of m.tools) {
        const cardNode = toolCard(t);
        if (tbl && t.meta && t.meta.results === tbl) tblNode = cardNode;
        col.appendChild(cardNode);
      }
      if (tbl) {
        if (tblNode) tblNode.classList.add('cite-table');
        linkCites(body, tbl);
      }
    }

    if (m.meta) {
      const meta = el('div', 'meta');
      if (m.meta.tps) meta.appendChild(el('span', '', m.meta.tps.toFixed(1) + ' tok/s'));
      if (m.meta.tokens) meta.appendChild(el('span', '', m.meta.tokens + ' tokens'));
      col.appendChild(meta);
    }

    // 助手消息的操作行：只放真正会用到的那一两个。
    // 「记住」把这条回答的要点写进本对话的记忆，下次请求自动带上 ——
    // 这是记忆系统唯一需要用户动手的地方，所以放在消息上而不是设置里。
    if (!m.streaming && (m.content || '').trim()) {
      const acts = el('div', 'msg-acts');
      const memBtn = el('button', 'act-btn', '记住');
      memBtn.title = '把这条回答存进本对话的记忆';
      memBtn.onclick = () => {
        const pick = String(m.content).trim().slice(0, 300);
        addMemory(pick, convScope());
        memBtn.textContent = '已记住';
        memBtn.disabled = true;
      };
      acts.appendChild(memBtn);
      const globBtn = el('button', 'act-btn', '记住（全局）');
      globBtn.title = '存进全局记忆，所有对话都能看到';
      globBtn.onclick = () => {
        const pick = String(m.content).trim().slice(0, 300);
        addMemory(pick, 'global');
        globBtn.textContent = '已记住';
        globBtn.disabled = true;
      };
      acts.appendChild(globBtn);
      col.appendChild(acts);
    }
    return markDrawn(wrap, m);
  }

  function renderMessages() {
    const box = $('#messages');
    box.innerHTML = '';
    const c = state.current;
    if (!c || !c.messages.length) {
      const empty = $('#emptyState');
      if (empty) box.appendChild(empty);
      return;
    }
    for (const m of c.messages) box.appendChild(messageNode(m));
    if (c.summary && c.folded) box.insertBefore(compressNote(c), box.firstChild);
    scrollBottom(true);
  }

  // 折叠提示：告诉用户"前面那段已经被压缩了"，并且摘要随时可以点开看。
  // 不做成折叠掉就消失 —— 用户翻历史时会以为消息丢了。
  function compressNote(c) {
    const wrap = el('div', 'compress-note');
    const head = el('div', 'cn-head');
    head.appendChild(iconNode('scissors', 'i'));
    head.appendChild(el('b', null, '已压缩 ' + (c.folded || 0) + ' 条早期消息'));
    const toggle = el('button', 'act-btn', '看摘要');
    head.appendChild(toggle);
    wrap.appendChild(head);
    const bodyEl = el('div', 'cn-body hidden', c.summary || '');
    wrap.appendChild(bodyEl);
    toggle.onclick = () => {
      const hidden = bodyEl.classList.toggle('hidden');
      toggle.textContent = hidden ? '看摘要' : '收起';
    };
    return wrap;
  }

  // ---------------------------------------------------------------- 滚动跟随
  //
  // 规则只有一条：**贴底才跟随**。
  //
  // 旧实现每帧都算一次"离底 < 220px 就贴底"，看起来很合理，实际有两个毛病：
  //   1) 220px 太宽。用户往上滑半屏想重读一段，仍然被判定为"在底部"，
  //      下一帧就被拽回去 —— 表现就是"滑不上去"。
  //   2) 判定用的是 scrollTop，而流式输出时 scrollHeight 每帧都在长，
  //      程序自己滚出来的位置也会被当成用户意图，两边互相打架。
  // 现在把"要不要跟随"变成一个有状态的开关：用户主动上滑就关掉，滑回底部
  // 才重新打开。程序滚动期间用一个短标记忽略自己触发的 scroll 事件。
  const stick = { on: true };
  let progScroll = false;
  const STICK_GAP = 72;   // 离底多少像素以内算"在底部"

  function atBottom(box) {
    return box.scrollHeight - box.scrollTop - box.clientHeight < STICK_GAP;
  }

  function scrollBottom(force) {
    const box = $('#messages');
    if (!box) return;
    if (force) stick.on = true;
    if (!stick.on) return;
    progScroll = true;
    box.scrollTop = box.scrollHeight;
    requestAnimationFrame(() => { progScroll = false; });
  }

  function bindScrollStick() {
    const box = $('#messages');
    if (!box || box.dataset.stickBound) return;
    box.dataset.stickBound = '1';
    box.addEventListener('scroll', () => {
      // 自己滚出来的事件不算用户意图，否则跟随开关会被自己关掉
      if (progScroll) return;
      stick.on = atBottom(box);
    }, { passive: true });
    // 滚轮/触摸立刻表态，不用等 scroll 事件排队 —— 差一帧就会被拽回去
    box.addEventListener('wheel', (e) => {
      if (e.deltaY < 0) stick.on = false;
    }, { passive: true });
    box.addEventListener('touchstart', () => { progScroll = false; }, { passive: true });
  }

  // ---------------------------------------------------------------- 服务端
  function fmtCtx(n) {
    if (!n) return '—';
    return n >= 1024 ? Math.round(n / 1024) + 'K' : String(n);
  }
  function parseSysRam(hw) {
    const m = /([\d.]+)\s*GB/.exec(hw || '');
    return m ? parseFloat(m[1]) : 0;
  }
  const GB = 1073741824;

  async function refreshState() {
    try {
      const r = await fetch('/api/state');
      const s = await r.json();
      const first = !state.server;
      state.server = s;
      renderHead(s);
      renderModelInfo(s);
      renderMeters(s);
      renderRes(s);
      renderCwd();
      $('#hwInfo').textContent = s.hardware || '';
      // 只有加载了视觉塔/语音塔才露出对应入口
      const att = $('#btnAttach');
      if (att) att.hidden = !(s.mm_vision || s.mm_audio);
      const bi = $('#btnImage');
      if (bi) {
        bi.disabled = !s.mm_vision;
        bi.classList.toggle('dim', !s.mm_vision);
      }
      const mi = $('#btnMic');
      if (mi) {
        mi.disabled = !s.mm_audio;
        mi.classList.toggle('dim', !s.mm_audio);
        mi.title = s.mm_audio ? '语音输入（点一下开始，再点结束）' : '当前模型没有语音塔';
      }
      const bw = $('#btnWeb');
      if (bw) {
        bw.disabled = !s.net;
        bw.classList.toggle('dim', !s.net);
        bw.title = s.net ? '联网搜索（先搜再答）' : '联网权限已关闭（设置 → 权限）';
        if (!s.net && state.web) { state.web = false; bw.classList.remove('on'); }
      }
      if (first) {
        log(s.loaded ? 'Model loaded successfully.' : 'No model loaded.', s.loaded ? '' : 'err');
        if (s.info && s.info.gpu) log('Using Metal (GPU) ...');
      }
      return s;
    } catch (e) {
      $('#mpName').textContent = '服务未连接';
      $('#mpTags').innerHTML = '';
      $('#liveBadge').classList.remove('on');
      $('#liveBadge').querySelector('span').textContent = '未连接';
      return null;
    }
  }

  // ---------------------------------------------------------------- 右栏
  // 量化标签取 model_id 里的主量化标记（Q4_K_M），而不是 info.quant 串。
  // info.quant 是逐层拼起来的，比如 "F32+Q4_K+Q6_K" —— 尾项 Q6_K 是输出层，
  // 拿它当标签等于说这个模型是 Q6_K，和实际主体差两档。
  // 顶栏胶囊和右栏模型信息共用这一个函数，两处不能各写一套。
  function quantLabel(s) {
    const mid = (s && s.model_id) || '';
    // 先认量化档，再认"未量化"。
    //
    // 未量化的写法大小写不定：LFM3 那份叫 -F16，HuggingFace 官方的 qwen 叫
    // -fp16（而且 "FP16" 里没有连续的 "F16"，是 F、P、1、6）。所以这两类
    // 分开匹配：量化档保持大小写敏感（带 i 会让 "qwen2.5" 里的 q2 被当成
    // 量化标记），未量化的那一组忽略大小写。
    const qm = /(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*)/.exec(mid);
    if (qm) return qm[1];
    const uq = /(BF16|FP16|F16|FP32|F32)/i.exec(mid);
    if (uq) return uq[1].toUpperCase();
    // model_id 里没有量化标记时退回 quant 串：跳过 F32/F16 这类"未量化"的边角层，
    // 取第一个真正的量化块。全都未量化才显示 F32。
    const parts = (((s && s.info && s.info.quant) || '')).split('+').filter(Boolean);
    return parts.find((x) => !/^(F32|F16|FP32|FP16|BF16)$/.test(x)) || parts[0] || '';
  }

  function renderHead(s) {
    const name = $('#mpName');
    const tags = $('#mpTags');
    const live = $('#liveBadge');
    if (!name) return;
    tags.innerHTML = '';
    if (s && s.loaded && s.info) {
      name.textContent = s.model_id || '已加载';
      const i = s.info;
      const bits = [
        (i.params / 1e9).toFixed(2) + 'B',
        quantLabel(s),
        fmtCtx(i.n_ctx),
      ].filter(Boolean);
      for (const t of bits) tags.appendChild(el('span', '', t));
      live.classList.add('on');
      live.querySelector('span').textContent = '本地运行 · 已加载';
    } else {
      name.textContent = s ? '未加载模型' : '服务未连接';
      live.classList.remove('on');
      live.querySelector('span').textContent = s ? '空闲' : '未连接';
    }
  }

  function renderModelInfo(s) {
    const kv = $('#modelKv');
    const nm = $('#mcName');
    const tg = $('#mcTag');
    if (!kv) return;
    kv.innerHTML = '';
    if (!s || !s.info) {
      nm.textContent = '未加载';
      tg.textContent = '空闲';
      tg.className = 'tag off';
      return;
    }
    const i = s.info;
    nm.textContent = s.model_id || i.arch;
    tg.textContent = '本地';
    tg.className = 'tag';
    const rows = [
      ['参数量', (i.params / 1e9).toFixed(2) + 'B'],
      ['量化', quantLabel(s)],
      ['架构', i.arch],
      ['层数', String(i.n_layer)],
      ['上下文', fmtCtx(i.n_ctx)],
      ['词表', i.n_vocab >= 1000 ? Math.round(i.n_vocab / 1000) + 'K' : String(i.n_vocab)],
    ];
    for (const [k, v] of rows) {
      const row = el('div', 'kv');
      row.appendChild(el('span', 'k', k));
      row.appendChild(el('span', 'v', v));
      kv.appendChild(row);
    }
  }

  // 内存条只显示实测值（进程 RSS / 系统内存），不做估算。
  // 估算值一旦和活动监视器对不上，这块面板就没人信了。
  function renderMeters(s) {
    const box = $('#meters');
    if (!box) return;
    const rows = [];
    const sysGB = parseSysRam(s && s.hardware);
    const rssGB = s && s.rss_bytes ? s.rss_bytes / GB : 0;
    if (sysGB && rssGB) {
      rows.push({
        label: '内存',
        pct: Math.min(100, (rssGB / sysGB) * 100),
        val: rssGB.toFixed(1) + ' / ' + sysGB.toFixed(0) + ' GB',
      });
    }
    const maxTok = settings.max_tokens || 1024;
    const genTok = state.gen && state.gen.tokens ? state.gen.tokens : 0;
    rows.push({
      label: '生成',
      pct: Math.min(100, (genTok / maxTok) * 100),
      val: genTok + ' / ' + maxTok,
    });

    box.innerHTML = '';
    for (const r of rows) {
      const m = el('div', 'meter');
      m.appendChild(el('span', 'm-label', r.label));
      const bar = el('span', 'm-bar');
      const fill = el('i');
      if (r.pct >= 88) fill.className = 'warn';
      fill.style.width = r.pct.toFixed(1) + '%';
      bar.appendChild(fill);
      m.appendChild(bar);
      m.appendChild(el('span', 'm-val', r.val));
      box.appendChild(m);
    }
    if (s && s.info) {
      const bits = [];
      bits.push(s.info.gpu ? 'Metal' : 'CPU');
      bits.push('上下文 ' + fmtCtx(s.info.n_ctx));
      if (s.info.chatml) bits.push('ChatML');
      box.appendChild(el('div', 'meter-note', bits.join(' · ')));
    }
  }

  // 左栏底部的资源条：和右栏内存条同源，只是换了呈现方式。
  function renderRes(s) {
    const txt = $('#resTxt');
    const bar = $('#resBar');
    if (!txt || !bar) return;
    const sysGB = parseSysRam(s && s.hardware);
    const rssGB = s && s.rss_bytes ? s.rss_bytes / GB : 0;
    if (!sysGB || !rssGB) {
      txt.textContent = '—';
      bar.style.width = '0';
      return;
    }
    txt.textContent = rssGB.toFixed(1) + ' / ' + sysGB.toFixed(0) + ' GB';
    bar.style.width = Math.min(100, (rssGB / sysGB) * 100).toFixed(1) + '%';
  }

  // 环形仪表：满量程 60 tok/s。固定量程比自适应好 ——
  // 自适应会让"快"和"慢"看起来一样，环形就失去参照意义了。
  const TPS_MAX = 60;
  const GAUGE_ARC = 235.6;
  function setGauge(tps, tokens, total, ttft) {
    const fill = $('#gaugeFill');
    if (!fill) return;
    const pct = Math.max(0, Math.min(1, (tps || 0) / TPS_MAX));
    fill.setAttribute('stroke-dasharray', (pct * GAUGE_ARC).toFixed(1) + ' 314.2');
    $('#gTps').textContent = tps ? tps.toFixed(1) : '—';
    $('#gTok').textContent = tokens ? tokens.toLocaleString() : '0';
    $('#gTime').textContent = total ? total.toFixed(1) + ' s' : '—';
    $('#gTtft').textContent = ttft ? ttft.toFixed(2) + ' s' : '—';
  }

  function log(msg, kind) {
    const d = new Date();
    const t = [d.getHours(), d.getMinutes(), d.getSeconds()]
      .map((n) => String(n).padStart(2, '0')).join(':');
    state.logs.push({ t, msg, kind: kind || '' });
    if (state.logs.length > 80) state.logs.shift();
    renderLogs();
  }
  function renderLogs() {
    const box = $('#logBox');
    if (!box) return;
    box.innerHTML = '';
    for (const l of state.logs) {
      const line = el('div');
      line.appendChild(el('span', 't', '[' + l.t + '] '));
      line.appendChild(el('span', l.kind === 'err' ? 'e' : '', l.msg));
      box.appendChild(line);
    }
    box.scrollTop = box.scrollHeight;
  }

  function renderTools() {
    const box = $('#toolList');
    if (!box) return;
    const c = state.current;
    let tools = [];
    if (c) for (const m of c.messages) if (m.tools) tools = tools.concat(m.tools);
    const cnt = $('#toolCount');
    if (cnt) cnt.textContent = String(tools.length);
    box.innerHTML = '';
    if (!tools.length) {
      box.appendChild(el('div', 'tool-empty', '本轮对话还没有调用工具'));
      return;
    }
    for (const t of tools.slice(-8)) {
      const row = el('div', 'tool-row' + (t.ok === false ? ' fail' : '') + (t.denied ? ' denied' : ''));
      const ti = el('span', 'ti');
      ti.appendChild(iconNode(t.denied ? 'shield' : (t.ok === false ? 'x' : 'terminal'), 'i'));
      row.appendChild(ti);
      row.appendChild(el('span', 'tn', t.name));
      const td = el('span', 'td');
      if (t.denied) {
        td.appendChild(el('span', '', '未执行'));
      } else if (t.ok === false) {
        td.appendChild(el('span', '', '失败'));
      } else {
        td.appendChild(iconNode('check', 'i'));
        if (t.ms) td.appendChild(el('span', '', (t.ms / 1000).toFixed(1) + 's'));
      }
      row.appendChild(td);
      box.appendChild(row);
    }
  }

  // ---------------------------------------------------------------- 附件
  // 服务端一次只接受一个媒体（视觉塔和语音塔是两个独立的 mmproj，同时注入
  // 需要两套前缀），所以这里也做成单选。
  function pickAttachment(only) {
    const s = state.server || {};
    const acc = [];
    if (only === 'image') {
      acc.push('image/*');
    } else if (only === 'file') {
      acc.push('*/*');
    } else {
      if (s.mm_vision) acc.push('image/*', 'video/*');
      if (s.mm_audio) acc.push('audio/*');
    }
    const inp = $('#fileInput');
    inp.accept = acc.length ? acc.join(',') : '*/*';
    inp.value = '';
    inp.click();
  }

  // 文本/代码类扩展名。为什么要专门列一张表：File.type 在桌面 App 和
  // 部分 WebView 里经常是空串（尤其是 .md / .py / .go 这些系统没有注册
  // 关联的类型），只看 mime 会把它们一律当成二进制。
  const TEXT_EXT = new Set([
    'txt', 'md', 'markdown', 'json', 'jsonl', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf',
    'csv', 'tsv', 'log', 'xml', 'html', 'htm', 'css', 'scss', 'js', 'mjs', 'cjs', 'ts',
    'tsx', 'jsx', 'vue', 'svelte', 'py', 'rb', 'go', 'rs', 'c', 'h', 'cc', 'cpp', 'hpp',
    'm', 'mm', 'swift', 'kt', 'java', 'sh', 'zsh', 'bash', 'sql', 'proto', 'gradle',
    'cmake', 'mk', 'dockerfile', 'gitignore', 'env', 'patch', 'diff', 'srt', 'vtt',
  ]);

  function looksText(file) {
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (TEXT_EXT.has(ext)) return true;
    const m = file.type || '';
    if (m.startsWith('text/')) return true;
    return ['application/json', 'application/xml', 'application/javascript',
            'application/x-sh', 'application/x-yaml'].includes(m);
  }

  function readAttachment(file) {
    // 文本/代码：直接读成字符串塞进上下文。
    // 原来这里不分类型，非音视频一律当 image 走 data URL —— 拖一个 .md
    // 进来会变成 image_url，服务端当场拒掉，用户看到的就是"上传不了文件"。
    if (looksText(file)) {
      return new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onerror = () => reject(new Error('读取失败'));
        fr.onload = () => {
          let t = String(fr.result || '');
          const CAP = 120 * 1024;
          if (t.length > CAP) {
            t = t.slice(0, CAP) + `\n…（文件较大，只截取了前 ${Math.round(CAP / 1024)} KB，` +
                                 `原文共 ${Math.round(file.size / 1024)} KB）`;
          }
          resolve({ kind: 'text', name: file.name, mime: file.type || 'text/plain',
                    text: t, bytes: file.size });
        };
        fr.readAsText(file);
      });
    }
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onerror = () => reject(new Error('读取失败'));
      fr.onload = () => {
        const dataUrl = String(fr.result || '');
        const mime = file.type || 'application/octet-stream';
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        let kind = 'image';
        if (mime.startsWith('audio/') || ['wav', 'mp3', 'm4a', 'flac', 'aiff', 'aif', 'ogg'].includes(ext)) kind = 'audio';
        else if (mime.startsWith('video/') || ['mp4', 'mov', 'webm', 'm4v'].includes(ext)) kind = 'video';
        else if (!mime.startsWith('image/') && ext !== 'png' && ext !== 'jpg' && ext !== 'jpeg'
                 && ext !== 'webp' && ext !== 'gif' && ext !== 'bmp') {
          // 既不是文本、也不是音视频、也不是常见图片 —— 硬发过去只会得到
          // 一句服务端错误。在这里就拒掉，理由说清楚。
          reject(new Error('不支持这种文件类型：' + (file.type || ext || '未知')));
          return;
        }
        resolve({ kind, name: file.name, mime, fmt: mime.split('/')[1] || 'bin', dataUrl });
      };
      fr.readAsDataURL(file);
    });
  }

  // 缩略图：原图几 MB，直接塞进 localStorage 会撑爆配额（历史记录要留着）
  function makeThumb(dataUrl) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onerror = () => resolve(null);
      img.onload = () => {
        const S = 192;
        const s = Math.min(1, S / Math.max(img.width, img.height));
        const c = document.createElement('canvas');
        c.width = Math.max(1, Math.round(img.width * s));
        c.height = Math.max(1, Math.round(img.height * s));
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        try { resolve(c.toDataURL('image/jpeg', 0.62)); } catch (e) { resolve(null); }
      };
      img.src = dataUrl;
    });
  }

  function renderAttach() {
    const bar = $('#attachBar');
    if (!bar) return;
    bar.innerHTML = '';
    const a = state.attach;
    if (!a) { bar.classList.add('hidden'); return; }
    bar.classList.remove('hidden');
    const chip = el('div', 'attach-chip');
    if (a.kind === 'image' && a.thumb) {
      const im = new Image();
      im.src = a.thumb;
      im.alt = a.name;
      chip.appendChild(im);
    } else {
      const ph = el('div', 'ph');
      ph.appendChild(iconNode(a.kind === 'audio' ? 'msg-audio'
                             : a.kind === 'text' ? 'msg-file' : 'msg-video', 'i'));
      chip.appendChild(ph);
    }
    chip.appendChild(el('div', 'an', a.kind === 'text' && a.bytes
      ? a.name + ' · ' + fmtBytes(a.bytes) : a.name));
    const x = el('button', 'ax');
    x.setAttribute('aria-label', '移除附件');
    x.appendChild(iconNode('x', 'i'));
    x.title = '移除';
    x.onclick = () => { state.attach = null; renderAttach(); };
    chip.appendChild(x);
    bar.appendChild(chip);
  }

  // 把附件塞进 OpenAI 的 content 数组
  function attachmentParts(a) {
    if (a.kind === 'image') return { type: 'image_url', image_url: { url: a.dataUrl } };
    if (a.kind === 'video') return { type: 'video_url', video_url: { url: a.dataUrl } };
    // data 直接给完整 data URL，服务端会原样用
    return { type: 'input_audio', input_audio: { data: a.dataUrl, format: a.fmt } };
  }

  // 附件收口。三种媒体各要一座塔，没有塔的时候现在就在这儿拦下来。
  //
  // 以前是发出去让服务端回 400（"当前模型没有视觉塔"），用户看到的是
  // 一条红色错误、附件已经贴在上面了，得自己删掉再换模型 —— 而现在
  // 拖进来的一瞬间就知道不行。
  async function useAttachment(a) {
    const s = state.server || {};
    if ((a.kind === 'image' || a.kind === 'video') && !s.mm_vision) {
      alert('当前模型没有视觉塔，看不了' + (a.kind === 'video' ? '视频' : '图片') +
            '。换 lfm3-vl 这类带 mmproj 的模型再试。');
      return;
    }
    if (a.kind === 'audio' && !s.mm_audio) {
      alert('当前模型没有语音塔，听不了音频。换 lfm3-audio 这类带 mmproj 的模型再试。');
      return;
    }
    if (a.kind === 'image') a.thumb = await makeThumb(a.dataUrl);
    state.attach = a;
    renderAttach();
  }

  // ---------------------------------------------------------------- 语音输入
  //
  // 录成 16 kHz 单声道 WAV，而不是拿 MediaRecorder 直接录一段 webm/opus。
  // 服务端解码走 AVFoundation 的 AVAudioFile（engine/src/audiofile.mm），它认
  // wav / mp3 / m4a / flac / aiff / caf，**不认 webm** —— 而 Chromium 系
  // 的 MediaRecorder 默认就给 webm。发过去只会得到一句解码失败，用户看到的
  // 就是"语音输入用不了"。
  //
  // 自己取 PCM 再写 44 字节 WAV 头是二十行的事，换来的是不依赖浏览器的编码器
  // 选什么。采样率也自己降到 16 kHz：语音塔的 mel 就是按 16 kHz 算的，
  // 交给 AVFoundation 去重采样只是多一次误差。
  const rec = { on: false, ctx: null, stream: null, proc: null, chunks: [], t0: 0, timer: 0 };

  function downsampleTo(buf, from, to) {
    if (to >= from) return buf;
    const ratio = from / to;
    const n = Math.floor(buf.length / ratio);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // 取区间平均而不是最近邻：直接抽点会把 8 kHz 以上的成分折回可听频段，
      // 录出来的声音发闷、带金属味，语音识别率会掉。
      const a = Math.floor(i * ratio);
      const b = Math.min(buf.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = a; j < b; j++) sum += buf[j];
      out[i] = b > a ? sum / (b - a) : 0;
    }
    return out;
  }

  function pcmToWav(pcm, rate) {
    const n = pcm.length;
    const ab = new ArrayBuffer(44 + n * 2);
    const dv = new DataView(ab);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
    str(0, 'RIFF');
    dv.setUint32(4, 36 + n * 2, true);
    str(8, 'WAVE');
    str(12, 'fmt ');
    dv.setUint32(16, 16, true);       // PCM 头长度
    dv.setUint16(20, 1, true);        // 1 = PCM
    dv.setUint16(22, 1, true);        // 单声道
    dv.setUint32(24, rate, true);
    dv.setUint32(28, rate * 2, true); // 字节率
    dv.setUint16(32, 2, true);        // 块对齐
    dv.setUint16(34, 16, true);       // 位深
    str(36, 'data');
    dv.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const v = Math.max(-1, Math.min(1, pcm[i]));
      dv.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
    return ab;
  }

  function b64Of(ab) {
    const u8 = new Uint8Array(ab);
    let s = '';
    // 一次 32 KB：再大 String.fromCharCode.apply 会爆栈
    for (let i = 0; i < u8.length; i += 32768) {
      s += String.fromCharCode.apply(null, u8.subarray(i, i + 32768));
    }
    return btoa(s);
  }

  async function startRec() {
    const s = state.server || {};
    if (!s.mm_audio) {
      log('当前模型没有加载语音塔，语音输入用不了。换 lfm3-audio 这类带 mmproj 的模型再试。', 'err');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const AC = window.AudioContext || window.webkitAudioContext;
      const ctx = new AC();
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      rec.chunks = [];
      proc.onaudioprocess = (e) => {
        if (!rec.on) return;
        rec.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      src.connect(proc);
      // ScriptProcessor 不接 destination 就不会被拉 —— 它的输出是静音，
      // 所以这样接不会把麦克风回放出来。
      proc.connect(ctx.destination);
      rec.on = true; rec.ctx = ctx; rec.stream = stream; rec.proc = proc;
      rec.t0 = performance.now();
      const btn = $('#btnMic');
      if (btn) { btn.classList.add('rec'); btn.title = '再点一下结束录音'; }
      log('开始录音…再点一下麦克风结束');
      // 60 秒硬上限：WAV 不压缩，1 分钟 16 kHz 单声道约 1.9 MB，
      // 再长 base64 进请求体就开始难看了。
      rec.timer = setTimeout(() => stopRec(true), 60000);
    } catch (e) {
      log('打不开麦克风：' + (e.message || e), 'err');
    }
  }

  function stopRec(auto) {
    if (!rec.on) return;
    rec.on = false;
    clearTimeout(rec.timer);
    const ms = performance.now() - rec.t0;
    try { rec.proc.disconnect(); } catch (e) {}
    try { rec.stream.getTracks().forEach((t) => t.stop()); } catch (e) {}
    const rate = rec.ctx ? rec.ctx.sampleRate : 16000;
    try { rec.ctx.close(); } catch (e) {}
    const btn = $('#btnMic');
    if (btn) { btn.classList.remove('rec'); btn.title = '语音输入'; }

    let n = 0;
    for (const c of rec.chunks) n += c.length;
    if (!n) { log('没录到声音', 'err'); return; }
    const all = new Float32Array(n);
    let off = 0;
    for (const c of rec.chunks) { all.set(c, off); off += c.length; }
    const wav = pcmToWav(downsampleTo(all, rate, 16000), 16000);
    const name = '语音 ' + (ms / 1000).toFixed(1) + 's.wav';
    state.attach = {
      kind: 'audio', name, mime: 'audio/wav', fmt: 'wav',
      dataUrl: 'data:audio/wav;base64,' + b64Of(wav), bytes: wav.byteLength,
    };
    renderAttach();
    log('已录音 ' + (ms / 1000).toFixed(1) + 's · ' + (wav.byteLength / 1024).toFixed(0) +
        ' KB' + (auto ? '（到 60 秒自动停了）' : ''));
  }

  // ---------------------------------------------------------------- 联网搜索开关
  //
  // 打开之后在用户问题前面加一句"先搜再答"，让模型自己去调 web_search。
  // 为什么不在这里直接把搜索结果塞进 prompt：那样模型会把搜索摘要当成
  // 用户给的材料，不再引用来源；而走工具调用，卡片上能看到它搜了什么、
  // 搜到几条，用户能判断这个答案值不值得信。
  function webOn() { return !!state.web; }

  // ---------------------------------------------------------------- 发送
  // opts.tools 可以单次覆盖工具开关。
  //
  // Agent 视图需要一次带工具的请求，但它不该把这个选择写进 settings ——
  // 否则从 Agent 里跑完一次，之后所有普通对话都会带上工具说明。实测
  // 2B 级模型带着工具说明问"今天天气怎么样"，回的是"我需要调用工具…"，
  // 一个一次性动作污染了长期配置。
  async function send(text, opts = {}) {
    if (state.streaming) return;
    const content = (text != null ? text : $('#input').value).trim();
    const att = state.attach;
    if (!content && !att) return;

    if (!state.current) newConv();
    const conv = state.current;

    // 压缩放在最前面：它可能要先跑一次生成（压旧对话），期间不该把
    // 用户这条消息已经塞进列表里 —— 否则压缩的切片边界会把新消息也算进去。
    await maybeCompress(conv);

    // 文本/代码附件：内容直接并进这一条 user 消息，不走 image_url。
    //
    // 为什么用围栏包起来而不是直接拼：模型需要一眼看出"这是文件内容、
    // 到哪结束"，否则它会顺着文件最后一行的语气继续往下写。文件名放在
    // 围栏的语言位上也顺便让界面把它渲染成一个带标题的代码块。
    let wire = content;
    if (att && att.kind === 'text') {
      const head = att.name.replace(/[`\s]/g, '_');
      wire = '```' + head + '\n' + att.text + '\n```' + (content ? '\n\n' + content : '');
    }

    const userMsg = { role: 'user', content };
    if (att) userMsg.att = { kind: att.kind, name: att.name, thumb: att.thumb, mime: att.mime, bytes: att.bytes };
    conv.messages.push(userMsg);
    if (conv.title === '新对话') {
      const t = content || att.name;
      conv.title = t.slice(0, 24) + (t.length > 24 ? '…' : '');
      $('#convTitle').textContent = conv.title;
    }
    state.attach = null;
    renderAttach();
    $('#input').value = '';
    autoGrow();

    const ai = { role: 'assistant', content: '', tools: [], meta: null, streaming: true };
    conv.messages.push(ai);
    state.gen = { tokens: 0, tps: 0, total: 0, ttft: 0 };
    setGauge(0, 0, 0, 0);
    renderMeters(state.server);
    log('Running inference ...');

    const box = $('#messages');
    const empty = $('#emptyState');
    if (empty && empty.parentNode) empty.remove();
    box.appendChild(messageNode(userMsg));
    // node / body 必须是 let：工具卡片到达时要整块重绘，重绘后这两个引用
    // 得指向新节点。原来写成 const，重绘那行会抛 TypeError，而且是被
    // 外层 catch 吞掉的那种 —— 界面上只剩一个永远"生成中"的空气泡，
    // 控制台一句错都不报，查起来极难。
    let node = messageNode(ai);
    box.appendChild(node);
    let body = node.querySelector('.body');
    body.classList.add('typing');
    scrollBottom(true);

    state.streaming = true;
    setSendEnabled(false);
    saveConvs();
    renderConvList();

    // 只发"没被折叠"的那一段 + 摘要。折叠掉的原文还在 conv.messages 里
    // （界面上照样能往上翻），但不再占上下文 —— 这就是压缩真正省下来的东西。
    const live = conv.messages.slice(conv.folded || 0);
    const msgs = live
      .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
      .map((m) => {
        if (m === userMsg && att && att.kind !== 'text') {
          const parts = [attachmentParts(att)];
          if (content) parts.push({ type: 'text', text: content });
          return { role: 'user', content: parts };
        }
        // 历史消息里的附件只留一行说明：服务端只认最后一条 user 的媒体，
        // 而且占位符没配特征会让模型看到空 token。
        const c = m === userMsg ? wire : (m.content || '');
        return { role: m.role, content: m.att ? (c ? c + '\n' : '') + `[${m.att.kind}: ${m.att.name}]` : c };
      });
    // 记忆 → 背景。放在 system 里而不是拼进最后一条 user：拼进 user 会让
    // 模型把记忆当成"用户刚说的话"，回答时反复复述。
    const mem = memoryForConv(conv);
    if (mem.length) {
      msgs.unshift({
        role: 'system',
        content: '以下是用户希望你记住的背景信息（不要逐条复述，只在相关时自然使用）：\n'
          + mem.map((m, i) => (i + 1) + '. ' + m.text).join('\n'),
      });
    }
    if (conv.summary) {
      msgs.unshift({
        role: 'system',
        content: '以下是本对话更早部分的摘要（原文已折叠，不要假装记得原文细节）：\n' + conv.summary,
      });
    }
    if (settings.system) msgs.unshift({ role: 'system', content: settings.system });

    // 联网搜索开关。
    //
    // 只说"先搜再答"不够 —— 小模型会跳过工具直接编。所以要求它把搜到的
    // 东西落进回答里，并明确禁止编造来源。实测 LFM3-1.7B 在带这句时会先调
    // 一次 web_search；不带就 3 次里只有 1 次真的搜。
    if (state.web) {
      msgs.unshift({
        role: 'system',
        content: '用户打开了联网搜索。回答前先用 web_search 工具查一次，'
          + '把查到的内容作为依据，并在回答里说明信息来自哪里。'
          + '如果搜索没有返回结果，就直接说没搜到，不要凭记忆编造来源或数字。',
      });
    }

    const t0 = performance.now();
    let tFirst = 0;

    try {
      state.abort = new AbortController();
      // 带附件时关掉工具说明。
      //
      // 小模型（LFM3-VL-2B 这类 2B 级）看到工具说明 + 一张图，很容易把
      // 整段输出写成 <tool_result>{...}，但它并没有真的调用工具 ——
      // 结果就是界面上一条"已完成 9 tokens"的空气泡。图片问答本来也不该
      // 走工具循环，关掉对双方都更省。真要图文 + 工具，用 API 显式传 tools。
      // 联网搜索开着就必须带工具说明，否则上面那段 system 让模型"用
      // web_search"，而请求里根本没有这个工具，它只能编。
      const withTools = ((opts.tools !== undefined ? opts.tools : state.tools) || state.web) && !att;
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: state.abort.signal,
        body: JSON.stringify({
          messages: msgs,
          stream: true,
          temperature: settings.temperature,
          top_p: settings.top_p,
          max_tokens: settings.max_tokens,
          dubhe_tools: withTools,
          // 本对话绑定的项目文件夹。服务端在请求作用域内把它当成工具的根，
          // 请求结束就恢复 —— 所以两个对话各读各的项目，不会串。
          workspace: conv.cwd || undefined,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let raf = null;
      let genTick = 0;

      // 流式渲染拆成"已定型"和"还在长"两段。
      //
      // 旧实现每来一个 token 就把整段 markdown 重新渲染一遍写回 innerHTML。
      // 三个后果都是用户能直接看到的：
      //   1) 闪烁 —— 整块 DOM 每 20ms 重建一次，代码块的 hover/过渡全部重来；
      //   2) 代码块滚不动 —— 重建后 scrollTop 归零，刚滚到的那行立刻弹回顶部；
      //   3) 选中不了文字 —— 选区随 DOM 一起被丢掉。
      // 现在按"安全边界"（空行或已闭合的 ``` 围栏）切开：前面那段只在变长时
      // 重绘一次，后面那段才是每帧都换的。切点绝不会落在代码块中间，所以
      // 已经闭合的代码块只渲染一次，滚动位置自然保住。
      // 用 let：工具卡片到达时整个气泡会被换成新节点，这两段要跟着搬过去。
      let headEl = document.createElement('div');
      let tailEl = document.createElement('div');
      headEl.className = 'md-head';
      tailEl.className = 'md-tail';
      body.appendChild(headEl);
      body.appendChild(tailEl);
      let lastHead = null;

      // 整块重绘气泡。工具结果和审批卡到达时都走这里。
      //
      // 抽成函数是因为它要改四个外层 let（node / body / headEl / tailEl）
      // 和一个 lastHead。原来这段是内联在 tool_result 分支里的，加审批卡
      // 时又要抄一遍 —— 抄漏一处的症状是"后面的增量写进了已经不在 DOM
      // 里的旧节点"，界面上看着像卡住，控制台一句错都不报。
      const reswap = () => {
        const fresh = messageNode(ai);
        node.replaceWith(fresh);
        node = fresh;
        body = node.querySelector('.body');
        body.classList.add('typing');
        headEl = null;
        tailEl = null;
        body.appendChild((headEl = document.createElement('div')));
        body.appendChild((tailEl = document.createElement('div')));
        headEl.className = 'md-head';
        tailEl.className = 'md-tail';
        lastHead = null;
        paint();
      };

      const paintNow = () => {
        const raw = (ai.content || '').trim();
        const visible = stripToolCalls(raw);
        if (!visible) {
          // 全被工具调用吃掉了：露原文，别留空气泡
          if (lastHead !== '') { headEl.innerHTML = ''; lastHead = ''; }
          tailEl.innerHTML = (!raw || (ai.tools && ai.tools.length)) ? ''
            : '<div class="raw-note">模型这段输出被识别成工具调用，没有可显示的正文字。原文：</div>'
              + '<pre class="raw-out">' + escapeHtml(raw.slice(0, 2000)) + '</pre>';
          return;
        }
        const cut = stableCut(visible);
        const head = cut > 0 ? visible.slice(0, cut) : '';
        const tail = visible.slice(cut);
        if (head !== lastHead) { headEl.innerHTML = renderMarkdown(head); lastHead = head; }
        tailEl.innerHTML = renderMarkdown(tail);
        // 流式期间也要补引用：来源卡是先到的（tool_result），文字是后到的，
        // 所以每一帧画完都得重新扫一遍 —— 新吐出来的那几个 [n] 上一帧还不存在。
        // 没有搜索卡片的消息在这里直接返回，不产生任何额外开销。
        const tbl = citeTableOf(ai.tools);
        if (tbl) { linkCites(headEl, tbl); linkCites(tailEl, tbl); }
      };

      const paint = () => {
        // DOM 写操作全部收进 rAF：一个 token 一次布局，而不是一个 SSE
        // 分片一次。原来 innerHTML 写在 rAF 外面，等于每片都强制同步布局。
        if (raf) return;
        raf = requestAnimationFrame(() => {
          raf = null;
          paintNow();
          scrollBottom();
        });
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\n\n');
        buf = parts.pop();
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') continue;
          let j;
          try { j = JSON.parse(data); } catch (e) { continue; }

          if (j.dubhe) {
            if (j.dubhe.type === 'tool_call') {
              ai.tools.push({ name: j.dubhe.name, args: j.dubhe.arguments, ok: true, t0: performance.now() });
              log('Tool call: ' + j.dubhe.name);
              renderTools();
            } else if (j.dubhe.type === 'queued') {
              // 引擎一次只答一个，第二个请求在服务端排队。排队时它一个字节都
              // 还没发，界面和"卡死"长得一模一样 —— 所以服务端会先推这个事件，
              // 这里把状态行改成"排队中"，让用户知道是等别人，不是坏了。
              ai.queued = true;
              log('引擎正在回答另一个请求，已排队');
              reswap();
            } else if (j.dubhe.type === 'error') {
              // 服务端在流里报的错。原来前端只认 tool_* 三种事件，这类消息
              // 直接被丢掉 —— 表现是气泡永远"生成中…"，正文一个字都没有。
              ai.errText = j.dubhe.message || '生成失败';
              log('生成失败：' + ai.errText, 'err');
              reswap();
            } else if (j.dubhe.type === 'note') {
              // 服务端的"交代"，比如"输入 5 万 token，正在预填充"。还没出字
              // 的时候把它顶到状态行上 —— 大输入预填充要一分钟，只写日志
              // 等于没说。
              log(j.dubhe.message || '');
              if (!ai.content) { ai.noteText = j.dubhe.message || ''; reswap(); }
            } else if (j.dubhe.type === 'tool_ask') {
              // 需要用户点头的调用。卡片上两个按钮，点完 POST /api/approve。
              // 服务端在等（最多 90 秒），所以这张卡必须显眼、必须可点。
              //
              // 服务端先发 tool_call 再发 tool_ask，所以这里要**合并到上一条**
              // 而不是新增一条 —— 否则界面上会出现两张卡：一张空的 write_file，
              // 一张审批卡，看起来像调用了两次。
              const last = ai.tools[ai.tools.length - 1];
              const patch = {
                askId: j.dubhe.id, askDetail: j.dubhe.detail,
                askPending: true, ok: true,
              };
              if (last && last.name === j.dubhe.name && !last.output && !last.askId) {
                Object.assign(last, patch);
              } else {
                ai.tools.push(Object.assign({ name: j.dubhe.name, t0: performance.now() }, patch));
              }
              log('需要确认：' + (j.dubhe.detail || j.dubhe.name));
              renderTools();
              reswap();
              scrollBottom();
            } else if (j.dubhe.type === 'tool_ask_done') {
              const t = ai.tools.find((x) => x.askId === j.dubhe.id);
              if (t) {
                t.askPending = false;
                t.askAllow = j.dubhe.allow === true;
                t.askTimeout = j.dubhe.timeout === true;
              }
              renderTools();
              reswap();
              scrollBottom();
            } else if (j.dubhe.type === 'tool_result') {
              const last = ai.tools[ai.tools.length - 1];
              if (last) {
                last.output = j.dubhe.output;
                last.ok = j.dubhe.ok;
                // denied 和「执行失败」是两件事：前者是门禁没让它跑，
                // 后者是跑了但出错。卡片上必须分开，否则用户会去查命令
                // 本身，而真正该做的是去设置里开权限。
                last.denied = j.dubhe.denied === true;
                last.card = j.dubhe.card || 'generic';
                last.meta = j.dubhe.meta || {};
                last.ms = performance.now() - (last.t0 || performance.now());
                // 审批过的那条：结果一到就换成普通卡片。留着 askId 会一直
                // 画审批卡（"已允许"），而用户此刻想看的是"到底写进去了什么"。
                if (last.askId) { delete last.askId; last.askPending = false; }
              }
              renderTools();
              reswap();
            }
            continue;
          }
          const delta = j.choices && j.choices[0] && j.choices[0].delta;
          if (delta && delta.content) {
            if (!tFirst) {
              tFirst = performance.now();
              state.gen.ttft = (tFirst - t0) / 1000;
              log('First token in ' + state.gen.ttft.toFixed(2) + ' s');
            }
            // 排到队了 / 预填充完了：状态行换回"生成中"。只在这一帧重画一次，
            // 不能每来一个 token 都 reswap —— 那等于把整块 DOM 每秒重建几十次。
            if (ai.queued || ai.noteText) {
              ai.queued = false;
              delete ai.noteText;
              reswap();
            }
            ai.content += delta.content;
            paint();
            // 生成中每 12 帧刷一次仪表，别让 DOM 写操作拖慢解码
            if (++genTick % 12 === 0) {
              const el2 = (performance.now() - t0) / 1000;
              state.gen.total = el2;
              setGauge(state.gen.tokens / Math.max(el2, 0.001), state.gen.tokens, el2, state.gen.ttft);
            }
          }
          if (j.usage) {
            ai.meta = { tps: j.usage.decode_tps, tokens: j.usage.completion_tokens };
            state.gen.tokens = j.usage.completion_tokens || 0;
            state.gen.tps = j.usage.decode_tps || 0;
          }
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        // 用户主动停止：保留已生成的部分，不报错
        if (!ai.content) ai.content = '（已停止）';
      } else {
        ai.content = (ai.content || '') + `\n\n> ⚠️ 请求失败：${e.message}`;
      }
    } finally {
      state.streaming = false;
      state.abort = null;
      ai.streaming = false;
      delete ai.queued;    // 状态行用的临时标记，不该跟着会话存下去
      delete ai.noteText;
      setSendEnabled(true);
      const total = (performance.now() - t0) / 1000;
      const tokens = (ai.meta && ai.meta.tokens) || state.gen.tokens || 0;
      state.gen = {
        tokens,
        tps: (ai.meta && ai.meta.tps) || (tokens / Math.max(total, 0.001)),
        total,
        ttft: state.gen.ttft || 0,
      };
      setGauge(state.gen.tps, tokens, total, state.gen.ttft);
      if (tokens) {
        log('Response generated (' + tokens + ' tokens, '
            + state.gen.tps.toFixed(1) + ' tok/s).');
      } else {
        log('Stopped.', 'err');
      }
      // 收尾一律替换"当前还挂在文档里"的那个节点。
      // 工具卡片到达时整块重绘过，早期抓到的引用已经脱离文档；对它
      // replaceWith 等于什么都没做 —— 结果是气泡永远停在"生成中…"，
      // 正文一个字都渲染不出来。
      if (!node.isConnected) box.appendChild(node);
      const finalNode = messageNode(ai);
      node.replaceWith(finalNode);
      node = finalNode;
      body = node.querySelector('.body');
      body.classList.remove('typing');
      // "已完成 · 0 tokens"的空气泡比报错更让人困惑：真空了就直说。
      // 已经有 errText 的时候不要盖掉 —— 那才是真正的原因，"模型直接收尾了"
      // 会把用户引到错误的方向。
      if (!String(ai.content || '').trim() && !(ai.tools && ai.tools.length) && !ai.errText) {
        ai.content = '（这一轮没有任何输出。可能是上下文放不下，或者模型直接收尾了，换个问法再试。）';
        const nn = messageNode(ai);
        node.replaceWith(nn);
        node = nn;
        body = node.querySelector('.body');
        body.classList.remove('typing');
      }
      saveConvs();
      // 收尾不强制回底。用户生成到一半往上滑去读前面的内容，这时候把他
      // 拽回底部，等于"我看了半天的那段又没了"—— stick.on 为真时它照样
      // 会贴底，为假说明是他自己选择停在上面的。
      scrollBottom();
      renderTools();
      renderMeters(state.server);
      refreshState();
    }
  }

  // 生成中按钮不能变灰：它要变成"停止"。禁用掉的话，用户只能刷新页面，
  // 这是流式对话里最反直觉的一个坑。
  function setSendEnabled(on) {
    const b = $('#send');
    b.disabled = false;
    b.classList.toggle('stop', !on);
    b.title = on ? '发送' : '停止生成';
    b.innerHTML = on ? ICON_SEND : ICON_STOP;
  }

  // ---------------------------------------------------------------- 面板
  function openSheet(title, builder) {
    $('#sheetTitle').textContent = title;
    const body = $('#sheetBody');
    body.innerHTML = '';
    builder(body);
    $('#sheet').classList.remove('hidden');
    $('#mask').classList.remove('hidden');
  }
  function closeSheet() {
    $('#sheet').classList.add('hidden');
    $('#mask').classList.add('hidden');
  }

  function row(parent, label, desc, control) {
    const r = el('div', 'row');
    const left = el('div');
    left.appendChild(el('div', 'label', label));
    if (desc) left.appendChild(el('div', 'desc', desc));
    r.appendChild(left);
    r.appendChild(control);
    parent.appendChild(r);
    return r;
  }

  function slider(parent, label, desc, key, min, max, step, fmt) {
    const wrap = el('div');
    wrap.style.display = 'flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '10px';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = min; input.max = max; input.step = step; input.value = settings[key];
    const val = el('span', 'value', fmt(settings[key]));
    input.oninput = () => { settings[key] = parseFloat(input.value); val.textContent = fmt(settings[key]); saveSettings(); };
    wrap.appendChild(input);
    wrap.appendChild(val);
    row(parent, label, desc, wrap);
  }

  function toggle(parent, label, desc, get, set, dis) {
    const sw = el('label', 'switch');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = get();
    cb.onchange = () => set(cb.checked);
    // dis：这一项当前**不可能**生效，别让它看起来能拨。
    //
    // 命令沙箱就是这种情况：本机 sandbox-exec 套不上时，开关拨到 ON 也没用，
    // 刷新一下又弹回 OFF —— 用户会以为"开关坏了"，反复点。直接置灰更诚实。
    if (dis) {
      cb.disabled = true;
      cb.checked = false;
      sw.classList.add('off');
      sw.title = '本机不支持，无法开启';
    }
    sw.appendChild(cb);
    sw.appendChild(el('span'));
    row(parent, label, desc, sw);
  }

  // 分段控件：选中块是滑过去的，不是跳过去的
  function seg(parent, options, get, set, desc) {
    const wrap = el('div', 'seg');
    const thumb = el('div', 'seg-thumb');
    wrap.appendChild(thumb);

    const btns = [];
    const paint = () => {
      const cur = get();
      let idx = 0;
      btns.forEach((b, i) => {
        const on = b.dataset.v === cur;
        b.classList.toggle('on', on);
        if (on) idx = i;
      });
      const b = btns[idx];
      if (!b || !b.offsetWidth) return;
      thumb.style.width = b.offsetWidth + 'px';
      thumb.style.transform = `translateX(${b.offsetLeft}px)`;
    };

    options.forEach((o) => {
      const b = el('button', '', o.label);
      b.dataset.v = o.value;
      b.onclick = () => { set(o.value); paint(); };
      btns.push(b);
      wrap.appendChild(b);
    });

    parent.appendChild(wrap);
    if (desc) parent.appendChild(el('div', 'desc', desc));
    requestAnimationFrame(paint);
    window.addEventListener('resize', paint);
    return wrap;
  }

  // ---------------------------------------------------------------- 权限
  //
  // 服务端把权限做成了运行期可改的位（POST /api/settings），这里负责把它
  // 变成开关。改完立刻生效 —— 不需要重启进程，也不需要记住启动参数。
  async function setPerm(key, val) {
    try {
      const r = await vpost('/api/settings', { [key]: val });
      state.server = Object.assign({}, state.server || {}, r);
      const shown = typeof val === 'boolean' ? (val ? '开' : '关') : val;
      log('权限已更新：' + key + ' = ' + shown);
      renderMeters(state.server);
    } catch (e) {
      log('权限更新失败：' + e.message, 'err');
    }
  }

  function shortPath(p) {
    const home = (state.server && state.server.home) || '';
    let s = String(p || '');
    if (home && s.startsWith(home)) s = '~' + s.slice(home.length);
    if (s.length <= 42) return s;
    return s.slice(0, 18) + '…' + s.slice(-21);
  }

  // 目录选择器。复用 /api/files，所以看到的目录树和「文件」视图完全一致，
  // 不会出现"设置里能选、文件视图里看不到"这种两套边界的问题。
  async function pickDir(start, onPick) {
    let cur = start || '.';
    const render = async () => {
      let d;
      try {
        d = await vapi('/api/files?path=' + encodeURIComponent(cur));
      } catch (e) {
        log('读目录失败：' + e.message, 'err');
        return;
      }
      openSheet('选择文件夹', (body) => {
        body.appendChild(el('div', 'desc', '当前：' + d.path));
        // 两个动作按钮放在列表**上面**。
        //
        // 原来放在列表下面，目录一多就被推到折叠线以外 —— 看得见的只有
        // 右上角那个"完成"，而它只是关闭。用户点了"完成"以为选好了，
        // 实际什么都没变，回来发现项目文件夹还是旧的。
        const acts = el('div', 'dirpick-acts');
        const use = el('button', 'vbtn', '用这个文件夹');
        use.onclick = () => { closeSheet(); onPick(d.path); };
        acts.appendChild(use);
        const dflt = el('button', 'vbtn ghost', '用默认目录');
        dflt.onclick = () => { closeSheet(); onPick(null); };
        acts.appendChild(dflt);
        body.appendChild(acts);
        const card = el('div', 'vcard');
        const mk = (label, onClick, iconName) => {
          const r = el('div', 'vrow tap');
          const ic = el('div', 'vico');
          ic.appendChild(iconNode(iconName || 'folder'));
          r.appendChild(ic);
          const m = el('div', 'vmain');
          m.appendChild(el('div', 'vname', label));
          r.appendChild(m);
          r.onclick = onClick;
          card.appendChild(r);
        };
        mk('上一级', () => { cur = d.parent; render(); }, 'arrow-up');
        d.items.filter((i) => i.dir).forEach((i) => mk(i.name + '/', () => { cur = i.path; render(); }));
        body.appendChild(card);
      });
    };
    render();
  }

  function openSettings() {
    openSheet('设置', (body) => {
      body.appendChild(el('div', 'group-title', '模型'));
      if (state.server && state.server.info) {
        const i = state.server.info;
        const box = el('div');
        const kv = (k, v) => { const d = el('div', 'kv'); d.appendChild(el('span', 'k', k)); d.appendChild(el('span', 'v', v)); box.appendChild(d); };
        kv('架构', i.arch);
        kv('参数量', (i.params / 1e9).toFixed(2) + ' B');
        // 量化这一行用 quantLabel 而不是原始 quant 串：后者是逐层拼起来的
        // （"F32+F16"），看着像"这个模型一半是 F32"，其实只是几个边角层
        // 没量化。顶栏胶囊用的就是这个函数，两处必须一致。
        kv('量化', quantLabel(state.server));
        kv('层数 / 维度', `${i.n_layer} / ${i.n_embd}`);
        kv('词表', String(i.n_vocab));
        // 三个数分清楚：n_ctx 是这次能用的，n_ctx_ceiling 是这台机器装得下的，
        // n_ctx_train 是模型训练时见过的。三者不同时全列出来，否则用户会以为
        // "模型支持 256K，为什么 118K 就报满"。
        kv('上下文', fmtCtx(i.n_ctx)
          + (i.n_ctx_ceiling > i.n_ctx ? ' / 上限 ' + fmtCtx(i.n_ctx_ceiling) : '')
          + (i.n_ctx_train > (i.n_ctx_ceiling || i.n_ctx)
            ? ' / 训练 ' + fmtCtx(i.n_ctx_train) : ''));
        kv('Chat 模板', i.chatml ? 'ChatML' : '无');
        body.appendChild(box);
      } else {
        body.appendChild(el('div', 'desc', '尚未加载模型。用命令行启动：dubhed -m <model.gguf>'));
      }

      body.appendChild(el('div', 'group-title', '外观'));
      seg(body, [
        { value: 'full', label: '完整' },
        { value: 'lite', label: '精简' },
        { value: 'off', label: '省电' },
      ], glassLevel, (v) => setGlass(v, true),
        '完整最通透；精简省 GPU、观感接近；省电纯材质。三档排版与层次一致。');

      body.appendChild(el('div', 'group-title', '生成参数'));
      slider(body, '温度', '越高越随机，0 为确定性输出', 'temperature', 0, 1.5, 0.05, (v) => v.toFixed(2));
      slider(body, 'Top-P', '核采样阈值', 'top_p', 0.1, 1, 0.01, (v) => v.toFixed(2));
      slider(body, '最大输出', '单次回复的 token 上限', 'max_tokens', 128, 4096, 64, (v) => String(v));

      body.appendChild(el('div', 'group-title', '工具'));
      toggle(body, '启用工具调用', '允许模型读取文件、列目录、执行命令', () => settings.tools, (v) => { settings.tools = v; state.tools = v; localStorage.setItem(TOOLS_TOUCHED, '1'); saveSettings(); });

      // ---- 权限
      //
      // 这一段以前是一行只读文字"服务端 Shell：未启用（启动加 --allow-shell）"。
      // 用户看到的是"模型说它没有执行命令的能力"，而界面上唯一能做的
      // 事情是重启进程加一个只有源码里才知道的参数。现在改成开关，
      // 直接打 /api/settings，下一次工具调用就生效。
      body.appendChild(el('div', 'group-title', '权限'));
      const srv = () => state.server || {};
      toggle(body, '执行命令', '允许模型运行 shell 命令。有超时和输出上限。',
        () => !!srv().shell, (v) => setPerm('shell', v));
      toggle(body, '写文件', '允许 write_file / edit_file 改动磁盘上的文件。',
        () => srv().write !== false, (v) => setPerm('write', v));
      // 读域和写域分开。读可以看整个家目录（模型权重就在 ~/.dubhe 下），
      // 但写默认只落在当前项目文件夹里 —— 这是 Codex / DeepSeek Harness 的
      // workspace-write 语义，也是把文件夹绑成项目之后最符合直觉的边界。
      //
      // 关着不等于"写不了"：越界的那一次会在对话里弹审批卡，用户点允许
      // 才写。所以这里的开关是"要不要每次都问"，不是"允不允许"—— 说明
      // 文字必须这么写，否则用户会以为关掉之后模型彻底写不了外面。
      toggle(body, '写到项目外', '关着时，越界的写入会先问你一次（允许只对那一次有效）；打开后不再问。',
        () => srv().write_outside === true, (v) => setPerm('write_outside', v));
      toggle(body, '联网', '允许 web_search 搜索、fetch_url 抓网页。',
        () => !!srv().net, (v) => setPerm('net', v));
      const sbOff = srv().sandbox_cfg && srv().sandbox === false;
      toggle(body, '命令沙箱', '命令只能写工作区与 /tmp。关掉就是完全放开。',
        () => srv().sandbox !== false, (v) => setPerm('sandbox', v), sbOff);
      // 开关"开着但不生效"是最坏的一种状态：用户以为有隔离，其实命令在裸跑。
      // 服务端把配置值和实际探测结果分成两个字段，这里把差异说出来。
      if (sbOff) {
        const warn = el('div', 'perm-warn',
          '本机的 sandbox-exec 套不上（外层已有沙箱或系统限制），命令是未隔离执行的。' +
          '写盘限制目前不生效 —— 模型自己的 write_file 仍然只写项目内。');
        body.appendChild(warn);
      }

      // 命令超时：给几档就够，没必要拉滑块
      const tmo = el('div', 'seg');
      [10, 20, 60, 180].forEach((sec) => {
        const b = el('button', 'seg-item' + ((srv().shell_timeout_s || 20) === sec ? ' on' : ''), sec + 's');
        b.onclick = async () => {
          await setPerm('shell_timeout_s', sec);
          tmo.querySelectorAll('.seg-item').forEach((x) => x.classList.remove('on'));
          b.classList.add('on');
        };
        tmo.appendChild(b);
      });
      row(body, '命令超时', '超过这个时间整组进程会被杀掉', tmo);

      // 工具轮数。
      //
      // 这条以前根本不在界面上，服务端写死 2 —— 一个"grep 找位置 → 读那一处
      // → 读相关文件 → 改 → 回答"的任务第三轮就被掐断，用户看到的是
      // "已达 2 轮上限"。轮数越多越慢（每轮都要重做一次 prefill），所以
      // 给几档让用户按任务复杂度自己选，而不是替他定死。
      const rnd = el('div', 'seg');
      [2, 4, 6, 12].forEach((n) => {
        const b = el('button', 'seg-item' + ((srv().tool_rounds || 6) === n ? ' on' : ''), String(n));
        b.onclick = async () => {
          await setPerm('tool_rounds', n);
          rnd.querySelectorAll('.seg-item').forEach((x) => x.classList.remove('on'));
          b.classList.add('on');
        };
        rnd.appendChild(b);
      });
      row(body, '工具轮数', '一次提问里最多连续跑几轮工具。任务复杂就调大，但每轮都要重算一遍输入。', rnd);

      // 工作目录 = 工具眼里的"根"。相对路径全部相对它解析。
      const wd = el('div');
      wd.style.display = 'flex';
      wd.style.alignItems = 'center';
      wd.style.gap = '8px';
      const wdPath = el('span', 'value', shortPath(srv().workspace || '—'));
      wdPath.title = srv().workspace || '';
      const wdBtn = el('button', 'vbtn ghost', '选择…');
      wdBtn.onclick = () => pickDir(srv().workspace || '~', (p) => {
        // 全局工作目录不接受 null（那会让服务端失去一个合法的根）
        if (p) setPerm('workspace', p);
      });
      wd.appendChild(wdPath);
      wd.appendChild(wdBtn);
      row(body, '工作目录', '模型读写文件和执行命令的根目录', wd);

      body.appendChild(el('div', 'group-title', '系统提示词'));
      const ta = el('textarea', 'field');
      ta.rows = 4;
      ta.placeholder = '例如：你是一个严谨的工程助手，回答简洁。';
      ta.value = settings.system;
      ta.oninput = () => { settings.system = ta.value; saveSettings(); };
      body.appendChild(ta);
    });
  }

  function openApi() {
    openSheet('API 网关', (body) => {
      const s = state.server || {};
      const origin = s.lan_url || location.origin;
      body.appendChild(el('div', 'desc', s.auth
        ? '已开启鉴权。其他设备（手机、平板、同局域网的电脑）用下面的地址和密钥接入，本机访问免鉴权。'
        : '本机服务已兼容 OpenAI 协议。若要让手机接入，用 --share 启动会自动生成密钥。'));

      body.appendChild(el('div', 'group-title', 'Base URL'));
      body.appendChild(codeBox(origin + '/v1'));
      if (s.auth) {
        body.appendChild(el('div', 'group-title', 'API Key'));
        body.appendChild(codeBox(s.api_key || '（仅本机可见）'));
      }
      body.appendChild(el('div', 'group-title', '模型 ID'));
      body.appendChild(codeBox(s.model_id || s.model || 'dubhe'));
      body.appendChild(el('div', 'group-title', 'cURL 示例'));
      body.appendChild(codeBox(
        `curl ${origin}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n` +
        (s.auth && s.api_key ? `  -H "Authorization: Bearer ${s.api_key}" \\\n` : '') +
        `  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'`
      ));
      body.appendChild(el('div', 'group-title', '运行时'));
      const kv = (k, v) => { const d = el('div', 'kv'); d.appendChild(el('span', 'k', k)); d.appendChild(el('span', 'v', v)); body.appendChild(d); };
      kv('硬件', s.hardware || '—');
      kv('请求数', String(s.requests || 0));
      kv('累计 token', String(s.tokens || 0));
      kv('最近解码', (s.decode_tps || 0).toFixed(1) + ' tok/s');
      kv('版本', s.version || '—');
    });
  }

  function codeBox(text) {
    const d = el('div', 'code-box', text);
    const b = el('button', 'cp', '复制');
    b.style.cssText = 'position:absolute;top:7px;right:7px;border:none;background:var(--fill);color:var(--secondary);font-size:11px;padding:3px 7px;border-radius:5px;cursor:pointer';
    b.onclick = () => { navigator.clipboard.writeText(text); b.textContent = '已复制'; setTimeout(() => (b.textContent = '复制'), 1200); };
    d.appendChild(b);
    return d;
  }

  // ---------------------------------------------------------------- 交互
  function autoGrow() {
    const t = $('#input');
    t.style.height = 'auto';
    t.style.height = Math.min(t.scrollHeight, 180) + 'px';
  }

  // ---------------------------------------------------------------- 侧栏抽屉
  // 移动端的抽屉必须满足三件事，缺一件用户就会觉得"卡"：
  //   1. 点空白处能关（遮罩）
  //   2. 能左滑关（手势）
  //   3. 选中会话后自己关（否则用户要再点一次 ☰）
  function syncScrim() {
    const nav = $('#sidebar').classList.contains('open');
    const insp = $('#inspector').classList.contains('show');
    const scrim = $('#scrim');
    if (scrim) scrim.classList.toggle('show', nav || insp);
  }

  function closeSidebar() {
    $('#sidebar').classList.remove('open');
    $('#inspector').classList.remove('show');
    syncScrim();
  }

  function toggleInspector(force) {
    const ins = $('#inspector');
    const want = force != null ? force : !ins.classList.contains('show');
    ins.classList.toggle('show', want);
    if (want && window.innerWidth <= 860) $('#sidebar').classList.remove('open');
    syncScrim();
  }

  function bindSidebar() {
    const sb = $('#sidebar');
    let sx = 0, sy = 0, drag = false;

    sb.addEventListener('touchstart', (e) => {
      if (window.innerWidth > 860) return;
      sx = e.touches[0].clientX; sy = e.touches[0].clientY;
      drag = true; sb.style.transition = 'none';
    }, { passive: true });

    sb.addEventListener('touchmove', (e) => {
      if (!drag) return;
      const dx = e.touches[0].clientX - sx;
      const dy = e.touches[0].clientY - sy;
      // 只认明确的左滑，竖向滑动交给列表自己滚
      if (dx > 0 || Math.abs(dy) > Math.abs(dx)) return;
      sb.style.transform = `translateX(${Math.max(dx, -320)}px)`;
    }, { passive: true });

    sb.addEventListener('touchend', (e) => {
      if (!drag) return;
      drag = false;
      sb.style.transition = '';
      const dx = e.changedTouches[0].clientX - sx;
      sb.style.transform = '';
      if (dx < -70) closeSidebar();
    });
  }

  // ---------------------------------------------------------------- 光跟随
  // 指针在玻璃上移动时，把坐标写进 --mx / --my，CSS 的 radial-gradient 跟着走。
  // 只在有精确指针的设备上挂（触摸设备没有 hover，跟着手指跑反而更耗电）。
  const FINE_POINTER = window.matchMedia('(pointer: fine)');

  function bindLight() {
    document.querySelectorAll('.chip, .icon-btn, .send-btn, .conv, .nav-item, .model-pick, .suggest button')
      .forEach((n) => n.classList.add('lit'));
    document.querySelectorAll('.composer, .sheet').forEach((n) => n.classList.add('edge'));
    if (!FINE_POINTER.matches) return;

    let raf = null, host = null, mx = 0, my = 0;

    document.addEventListener('pointermove', (ev) => {
      const h = ev.target.closest && ev.target.closest('.lit, .edge');
      if (!h) return;
      const r = h.getBoundingClientRect();
      mx = ev.clientX - r.left; my = ev.clientY - r.top;
      host = h;
      h.classList.add('on');
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = null;
        if (!host) return;
        host.style.setProperty('--mx', mx + 'px');
        host.style.setProperty('--my', my + 'px');
      });
    }, { passive: true });

    document.addEventListener('pointerout', (ev) => {
      const h = ev.target.closest && ev.target.closest('.lit, .edge');
      if (h && !h.contains(ev.relatedTarget)) h.classList.remove('on');
    }, { passive: true });
  }

  // ---------------------------------------------------------------- 视觉档位
  // 三档材质。自适应只负责挑默认值；用户手动选过就永久记住，不再自动覆盖。
  // 关键点：低档不是"关掉特效"，是换一套更省电的材质 —— 见 app.css 的档位段。
  const GLASS_KEY = 'dubhe.glass';

  function autoGlass() {
    if (window.matchMedia('(prefers-reduced-transparency: reduce)').matches) return 'lite';
    const ua = navigator.userAgent;
    const cores = navigator.hardwareConcurrency || 4;
    const mem = navigator.deviceMemory || 4;
    // 安卓 WebView 的大面积 backdrop-filter 是软件渲染，中低端机扛不住
    if (/Android/i.test(ua)) return (cores <= 4 || mem <= 3) ? 'lite' : 'full';
    if (cores <= 2) return 'off';
    return 'full';
  }

  function glassLevel() {
    return localStorage.getItem(GLASS_KEY) || autoGlass();
  }

  function setGlass(level, persist) {
    document.body.dataset.glass = level;
    if (persist) { try { localStorage.setItem(GLASS_KEY, level); } catch (e) {} }
  }

  const ICON_SEND = icon('send');
  const ICON_STOP = icon('stop');

  // ==========================================================================
  // 功能视图：模型 / 工具 / Agent / 文件 / 知识库
  //
  // 五个视图都是真实数据 + 真实操作，没有一个是占位页：
  // 模型能真的切换（卸载 + 加载），文件能真的读，工具能真的执行，
  // 知识库能真的索引和检索。后端没做的能力（比如自动多步规划）就不假装有。
  // ==========================================================================

  async function vapi(path, opts) {
    const r = await fetch(path, opts);
    const text = await r.text();
    let j = null;
    try { j = JSON.parse(text); } catch (e) { /* 非 JSON 响应 */ }
    if (!r.ok) throw new Error((j && (j.error || j.message)) || 'HTTP ' + r.status);
    return j || {};
  }
  const vpost = (p, body) =>
    vapi(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });

  function fmtBytes(b) {
    if (!b) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0, v = b;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (v >= 10 || i === 0 ? v.toFixed(0) : v.toFixed(1)) + ' ' + u[i];
  }

  function fmtAgo(t) {
    if (!t) return '';
    const d = new Date(t * 1000);
    const s = (Date.now() - d.getTime()) / 1000;
    if (s < 90) return '刚刚';
    if (s < 3600) return Math.floor(s / 60) + ' 分钟前';
    if (s < 86400) return Math.floor(s / 3600) + ' 小时前';
    if (s < 86400 * 30) return Math.floor(s / 86400) + ' 天前';
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  }

  function setViewHead(title, sub, actions) {
    $('#viewTitle').textContent = title;
    $('#viewSub').textContent = sub || '';
    const box = $('#viewActions');
    box.innerHTML = '';
    (actions || []).forEach((a) => {
      const b = el('button', 'vbtn' + (a.pri ? ' pri' : '') + (a.ghost ? ' ghost' : ''));
      if (a.icon) b.appendChild(iconNode(a.icon));
      b.appendChild(document.createTextNode(a.label));
      if (a.on) b.onclick = a.on;
      box.appendChild(b);
    });
  }

  const VIEWS = {
    models: { title: '模型', render: () => viewModels() },
    tools: { title: '工具', render: () => viewTools() },
    agent: { title: 'Agent', render: () => viewAgent() },
    files: { title: '文件', render: () => viewFiles(state.fsPath) },
    kb: { title: '知识库', render: () => viewKb() },
    memory: { title: '记忆', render: () => viewMemory() },
  };

  // ---------------------------------------------------------------- 记忆视图
  function viewMemory() {
    const body = $('#viewBody');
    body.innerHTML = '';
    const conv = state.current;
    const mine = state.memory.filter((m) => m.scope === convScope());
    const glob = state.memory.filter((m) => m.scope === 'global');
    setViewHead('记忆', mine.length + ' 条本对话 · ' + glob.length + ' 条全局', [
      {
        label: '新建',
        on: () => {
          const t = prompt('要记住什么？（只对本对话生效）');
          if (t) { addMemory(t, convScope()); viewMemory(); }
        },
      },
    ]);

    body.appendChild(el('div', 'desc',
      '记忆会作为背景信息注入本对话的每一次请求。默认只属于当前对话；'
      + '标成「全局」的才会出现在所有对话里 —— 这就是对话之间不串记忆的边界。'));

    const section = (title, items, scope) => {
      const sec = el('div', 'vsec');
      const head = el('div', 'vsec-head');
      head.appendChild(el('h3', null, title));
      head.appendChild(el('span', 'n', items.length + ' 条'));
      sec.appendChild(head);
      if (!items.length) {
        sec.appendChild(vEmpty('还没有记忆', scope === 'global'
          ? '全局记忆对所有对话可见，适合放"我叫什么、我做什么"这类稳定事实。'
          : '在对话里点助手消息下的「记住」，或者点右上角「新建」。'));
        body.appendChild(sec);
        return;
      }
      const card = el('div', 'vcard');
      items.forEach((m) => {
        const row = el('div', 'vrow');
        const ico = el('div', 'vico');
        ico.appendChild(iconNode('think'));
        row.appendChild(ico);
        const main = el('div', 'vmain');
        main.appendChild(el('div', 'vname', m.text));
        main.appendChild(el('div', 'vmeta',
          new Date(m.at).toLocaleString() + (m.scope === 'global' ? ' · 全局' : ' · 本对话')));
        row.appendChild(main);
        const del = el('button', 'vbtn ghost', '删除');
        del.onclick = () => {
          state.memory = state.memory.filter((x) => x.id !== m.id);
          saveMemory();
          viewMemory();
        };
        row.appendChild(del);
        card.appendChild(row);
      });
      sec.appendChild(card);
      body.appendChild(sec);
    };

    section('本对话', mine, convScope());
    section('全局', glob, 'global');
  }

  function openView(name) {
    const v = VIEWS[name];
    if (!v) return;
    state.view = name;
    $('#views').hidden = false;
    $('#viewBody').innerHTML = '';
    setViewHead(v.title, '', []);
    // 地址栏带上是哪个视图：刷新不丢位置，也能直接分享 /#kb
    try { history.replaceState(null, '', '#' + name); } catch (e) {}
    document.querySelectorAll('.nav-item[data-view]').forEach((x) =>
      x.classList.toggle('on', x.dataset.view === name));
    v.render();
  }

  function closeView() {
    state.view = null;
    $('#views').hidden = true;
    try { history.replaceState(null, '', location.pathname + location.search); } catch (e) {}
    document.querySelectorAll('.nav-item[data-view]').forEach((x) =>
      x.classList.toggle('on', x.dataset.view === 'chat'));
  }

  function vEmpty(title, desc) {
    const b = el('div', 'vempty');
    b.appendChild(el('b', null, title));
    if (desc) b.appendChild(document.createTextNode(desc));
    return b;
  }

  // ---------------------------------------------------------------- 代码预览
  //
  // 模型吐出来的 HTML 片段直接就能看，不用"复制 → 存文件 → 打开"。
  //
  // 沙箱是硬约束：sandbox 只给 allow-scripts，**不给** allow-same-origin。
  // 少了这一条，预览页和主站同源，模型随手写个 fetch('/api/load') 就能
  // 操作本机引擎、读 localStorage 里的对话。给 allow-scripts 而不给
  // same-origin，iframe 拿到的是不透明源，脚本照跑，但碰不到我们。
  let pv = null;
  function ensurePreview() {
    if (pv) return pv;
    const mask = el('div', 'pv-mask');
    const panel = el('div', 'pv-panel');
    const head = el('div', 'pv-head');
    head.appendChild(el('span', 'pv-title', '运行预览'));
    head.appendChild(el('span', 'pv-hint', '沙箱内运行 · 不联网、不碰本站数据'));
    const acts = el('div', 'cb-actions');
    const btnNew = el('button', 'cb-btn', '新窗口');
    const btnClose = el('button', 'cb-btn', '关闭');
    acts.appendChild(btnNew);
    acts.appendChild(btnClose);
    head.appendChild(acts);
    const frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-scripts allow-modals allow-forms allow-popups');
    frame.setAttribute('referrerpolicy', 'no-referrer');
    frame.setAttribute('title', '预览');
    panel.appendChild(head);
    panel.appendChild(frame);
    document.body.appendChild(mask);
    document.body.appendChild(panel);
    mask.style.display = 'none';
    panel.style.display = 'none';
    const close = () => {
      mask.style.display = 'none';
      panel.style.display = 'none';
      frame.srcdoc = '';
    };
    btnClose.onclick = close;
    mask.onclick = close;
    btnNew.onclick = () => {
      const w = window.open('', '_blank');
      if (!w) return;
      w.document.open();
      w.document.write(frame.srcdoc || '');
      w.document.close();
    };
    pv = { mask, panel, frame, close };
    return pv;
  }

  function openPreview(code, lang) {
    const p = ensurePreview();
    p.frame.srcdoc = wrapPreviewDoc(code, lang);
    p.mask.style.display = '';
    p.panel.style.display = '';
  }

  // SVG 直接扔进 iframe 会被当成 HTML 解析，得包一层；纯 HTML 片段缺
  // <meta charset> 时中文会乱码（srcdoc 继承的是父页编码，但不保证）。
  function wrapPreviewDoc(code, lang) {
    const l = String(lang || '').toLowerCase();
    if (l === 'svg' || /^\s*<svg[\s>]/i.test(code)) {
      return '<!doctype html><meta charset="utf-8">'
        + '<style>html,body{margin:0;height:100%;display:grid;place-items:center;background:#fff}'
        + 'svg{max-width:96vw;max-height:96vh}</style>' + code;
    }
    if (/<html[\s>]/i.test(code)) return code;
    return '<!doctype html><meta charset="utf-8">'
      + '<style>body{margin:0;font:15px/1.6 -apple-system,system-ui,sans-serif;padding:18px}</style>'
      + code;
  }

  // ---------------------------------------------------------------- 模型
  async function viewModels() {
    const body = $('#viewBody');
    body.innerHTML = '';
    body.appendChild(el('div', 'vempty', '正在扫描模型目录…'));
    let d;
    try {
      d = await vapi('/api/models');
    } catch (e) {
      body.innerHTML = '';
      body.appendChild(el('div', 'vnote bad', '读不到模型清单：' + e.message));
      return;
    }
    setViewHead('模型', d.items.length + ' 个 · ' + d.dir, [
      { label: '刷新', icon: 'restart', on: () => viewModels() },
    ]);
    body.innerHTML = '';
    if (!d.items.length) {
      body.appendChild(vEmpty('模型目录是空的', d.dir + ' 下没有 .gguf 文件。把模型放到这里就能在界面上切换。'));
      return;
    }
    const card = el('div', 'vcard');
    // 速度预估。
    //
    // 解码是**带宽瓶颈**：每出一个 token 就要把整个权重读一遍，所以
    //   tok/s ≈ 内存带宽 ÷ 每 token 要读的字节数（≈ 模型文件大小）
    // 与其写死一个数字，不如拿"当前已加载模型的实测 tok/s × 体积比"外推 ——
    // 换台机器、换个内存带宽，这个比例关系仍然成立。
    //
    // 这件事值得摆在界面上，是因为"十几 tok/s 太慢"这个观感有一半来自
    // 选错了量化档：F16 是 3.5 GB，Q4_K_M 是 1.1 GB，同一台机器差三倍。
    const anchor = d.items.find((x) => x.loaded && x.bytes);
    const anchorTps = (state.server && state.server.decode_tps) || 0;
    d.items.forEach((m) => {
      const row = el('div', 'vrow' + (m.loaded ? ' on' : ''));
      const ico = el('div', 'vico');
      ico.appendChild(iconNode('cube'));
      row.appendChild(ico);
      // 后端已经把 mmproj 从模型清单里摘掉了（它是塔，不能单独加载）。
      // 这里保留一道同名判断只是兜底：万一指向的是别的目录结构，
      // 也不会给一个点了必然失败的"加载"按钮。
      const isMm = /^mmproj/i.test(m.name);
      const main = el('div', 'vmain');
      const nm = el('div', 'vname', m.name);
      if (m.loaded) nm.appendChild(el('span', 'vtag on', '当前'));
      // 同目录下有配套的塔 → 这个模型能看图/听音。切换过去时后端会自动挂上。
      // 已加载的也要标：它同时是"当前"和"多模态"，两个标签不冲突。
      if (m.mmproj) nm.appendChild(el('span', 'vtag', '多模态'));
      if (isMm) nm.appendChild(el('span', 'vtag', '多模态塔'));
      // MLX 条目：列出来但明说不能加载。装作没有的代价是用户以为
      // "这软件不支持苹果格式"，而实际上原因和出路都在 note 里。
      if (m.mlx) nm.appendChild(el('span', 'vtag', 'MLX'));
      main.appendChild(nm);
      const bits = [];
      if (m.quant) bits.push(m.quant);
      // MLX 目录名里的位宽（lfm3-mlx-4bit）没有走 guess_quant，补一下 ——
      // 一行 "MLX · 0.9 GB" 看不出是 4bit 还是 8bit，而这两个速度差一倍。
      else if (m.mlx) {
        const q = /(\d+)\s*bit/i.exec(m.name || '');
        if (q) bits.push(q[1] + 'bit');
      }
      if (m.params) bits.push((m.params / 1e9).toFixed(1) + 'B');
      bits.push(fmtBytes(m.bytes));
      bits.push(fmtAgo(m.mtime));
      if (!m.mlx && anchor && anchorTps > 0 && m.bytes) {
        bits.push(m.loaded ? '实测 ' + anchorTps.toFixed(0) + ' tok/s'
                           : '≈' + Math.round(anchorTps * anchor.bytes / m.bytes) + ' tok/s');
      }
      main.appendChild(el('div', 'vmeta', bits.join(' · ')));
      if (m.note) main.appendChild(el('div', 'vnote', m.note));
      row.appendChild(main);
      if (m.mlx) {
        row.classList.add('off');
      } else if (!m.loaded && !isMm) {
        const b = el('button', 'vbtn', '加载');
        b.onclick = async () => {
          b.disabled = true;
          b.textContent = '加载中…';
          try {
            const r = await vpost('/api/load', { path: m.path });
            b.textContent = '已加载 ' + Math.round(r.ms) + ' ms';
            await refreshState();
            // 塔加载失败（文件坏了、维度和模型对不上）不该静默吞掉：
            // 结果是"模型能跑但看不了图"，不说清楚用户会以为是图片的问题。
            if (r.mm_note) {
              const note = el('div', 'vnote bad', '模型已切换，但' + r.mm_note);
              card.parentNode.insertBefore(note, card);
              setTimeout(() => note.remove(), 8000);
            }
            viewModels();
          } catch (e) {
            b.disabled = false;
            b.textContent = '加载';
            const note = el('div', 'vnote bad', '加载失败：' + e.message);
            card.parentNode.insertBefore(note, card);
            setTimeout(() => note.remove(), 6000);
          }
        };
        row.appendChild(b);
      }
      card.appendChild(row);
    });
    body.appendChild(card);
    body.appendChild(el('div', 'vnote',
      '切换模型会先卸载当前权重再加载新的，通常 1~2 秒。加载失败会自动装回原来的模型，不会把服务变成没有模型的状态。'));
    body.appendChild(el('div', 'vnote',
      '速度是按"内存带宽 ÷ 模型体积"估的，拿当前模型的实测值外推 —— 同一台机器上，'
      + 'F16 的体积是 Q4_K_M 的三倍，速度也就只有三分之一。觉得慢先换小量化档，换内核没用。'));
  }

  // ---------------------------------------------------------------- 工具
  async function viewTools() {
    const body = $('#viewBody');
    body.innerHTML = '';
    body.appendChild(el('div', 'vempty', '正在读取工具清单…'));
    let d;
    try {
      d = await vapi('/api/tools');
    } catch (e) {
      body.innerHTML = '';
      body.appendChild(el('div', 'vnote bad', '读不到工具清单：' + e.message));
      return;
    }
    setViewHead('工具', d.items.filter((t) => t.enabled).length + ' / ' + d.items.length + ' 已启用', [
      { label: '刷新', icon: 'restart', on: () => viewTools() },
    ]);
    body.innerHTML = '';
    body.appendChild(el('div', 'vnote', '工作区：' + d.workspace));

    const card = el('div', 'vcard');
    d.items.forEach((t) => {
      const wrap = el('div');
      const row = el('div', 'vrow tap');
      const ico = el('div', 'vico');
      ico.appendChild(iconNode(t.name === 'shell' ? 'terminal' : 'nodes'));
      row.appendChild(ico);
      const main = el('div', 'vmain');
      const nm = el('div', 'vname', t.name);
      nm.appendChild(el('span', 'vtag' + (t.enabled ? ' on' : ' warn'), t.enabled ? '已启用' : '未启用'));
      main.appendChild(nm);
      main.appendChild(el('div', 'vmeta', t.desc + ' · 参数 ' + t.args));
      row.appendChild(main);
      const caret = el('div', 'vmeta', '试运行');
      row.appendChild(caret);
      wrap.appendChild(row);

      // 展开区：参数输入 + 执行 + 输出。默认收起，避免一屏全是输入框。
      const panel = el('div');
      panel.style.display = 'none';
      panel.style.padding = '0 13px 12px';
      const inp = el('input');
      inp.type = 'text';
      inp.placeholder = t.name === 'shell' ? 'command，例如 ls -la' : 'path，例如 README.md';
      const ibox = el('div', 'vinput');
      ibox.appendChild(iconNode('terminal'));
      ibox.appendChild(inp);
      const run = el('button', 'vbtn pri', '运行');
      run.style.marginLeft = '8px';
      const bar = el('div');
      bar.style.display = 'flex';
      bar.style.alignItems = 'center';
      bar.appendChild(ibox);
      ibox.style.flex = '1 1 auto';
      bar.appendChild(run);
      panel.appendChild(bar);
      const out = el('pre', 'vcode');
      out.style.display = 'none';
      out.style.marginTop = '9px';
      panel.appendChild(out);
      wrap.appendChild(panel);

      row.onclick = () => {
        const open = panel.style.display !== 'none';
        panel.style.display = open ? 'none' : 'block';
        caret.textContent = open ? '试运行' : '收起';
        if (!open) inp.focus();
      };
      run.onclick = async (ev) => {
        ev.stopPropagation();
        const val = inp.value.trim();
        if (!val) { inp.focus(); return; }
        run.disabled = true;
        run.textContent = '运行中…';
        out.style.display = 'block';
        out.textContent = '…';
        try {
          const args = t.name === 'shell' ? { command: val } : { path: val };
          const r = await vpost('/api/tools/run', { name: t.name, arguments: args });
          out.textContent = r.output || '(无输出)';
          if (!r.ok) out.style.borderColor = 'var(--red)';
        } catch (e) {
          out.textContent = '执行失败：' + e.message;
        }
        run.disabled = false;
        run.textContent = '运行';
      };
      card.appendChild(wrap);
    });
    body.appendChild(card);
  }

  // ---------------------------------------------------------------- Agent
  //
  // 不做"假的多步规划面板"：1.7B 级模型拆出来的计划基本没法用，画一个漂亮
  // 的计划树只会骗人。这里做的是真实的东西 —— 一个把任务直接丢给带工具
  // 循环的对话的入口，加上从历史会话里真实聚合出来的工具调用轨迹。
  async function viewAgent() {
    const body = $('#viewBody');
    body.innerHTML = '';
    setViewHead('Agent', '任务交给带工具的对话执行', []);

    const card = el('div', 'vcard');
    card.style.padding = '13px';
    card.appendChild(el('div', 'vname', '要做什么'));
    card.appendChild(el('div', 'vmeta', '会切回对话并临时打开工具调用，让模型自己决定读哪个文件、跑哪条命令。'));

    // 这段限制是实测出来的，不是免责声明。工具调用能不能成，主要看模型：
    // LFM3-1.7B 稳定一次调用就收敛；更小的模型会把计划说出来而不真调。
    // 写"模型自己决定"会让人以为工具坏了 —— 把边界摆在入口处更诚实。
    const lim = el('div', 'vnote');
    lim.style.marginTop = '9px';
    lim.textContent =
      '工具调用取决于模型。模型只把计划说出来、或一直重复调同一个工具时，' +
      '这里会给出提示并停下，不会空转。要确定能跑通，去「工具」视图直接试运行 —— ' +
      '那条路径不依赖模型判断。';
    card.appendChild(lim);

    const inp = el('textarea');
    inp.rows = 3;
    inp.placeholder = '例如：读一下 engine/src/metal.mm，告诉我 prefill 的派发逻辑在哪几行';
    inp.style.cssText =
      'width:100%;margin:10px 0;padding:9px 11px;border-radius:11px;border:1px solid var(--glass-line);' +
      'background:var(--glass-strong);font:inherit;font-size:13px;color:var(--text);outline:0;resize:vertical;box-sizing:border-box';
    card.appendChild(inp);

    const go = el('button', 'vbtn pri', '在对话中执行');
    go.onclick = () => {
      const t = inp.value.trim();
      if (!t) { inp.focus(); return; }
      closeView();
      // 单次开启，不写进 settings —— 从这里跑完一次不该让之后所有
      // 普通对话都背上工具说明。
      send(t, { tools: true });
    };
    card.appendChild(go);
    body.appendChild(card);

    // ---- 真实轨迹：把历史会话里所有工具调用按时间倒序摊平
    const runs = [];
    for (const c of state.convs) {
      for (const m of c.messages || []) {
        for (const t of m.tools || []) runs.push({ t, conv: c.title, at: c.at });
      }
    }
    runs.sort((a, b) => (b.at || 0) - (a.at || 0));

    const sec = el('div', 'vsec');
    const head = el('div', 'vsec-head');
    head.appendChild(el('h3', null, '最近的工具调用'));
    head.appendChild(el('span', 'n', runs.length + ' 次'));
    sec.appendChild(head);

    if (!runs.length) {
      sec.appendChild(vEmpty('还没有工具调用', '在对话里问一个需要读文件的问题，这里就会出现轨迹。'));
    } else {
      const list = el('div', 'vcard');
      runs.slice(0, 20).forEach(({ t, conv }) => {
        const row = el('div', 'vrow');
        const ico = el('div', 'vico');
        ico.appendChild(iconNode('nodes'));
        row.appendChild(ico);
        const main = el('div', 'vmain');
        const nm = el('div', 'vname', t.name + (t.ok === false ? '（失败）' : ''));
        main.appendChild(nm);
        const arg = t.args ? JSON.stringify(t.args).slice(0, 70) : '';
        main.appendChild(el('div', 'vmeta', arg));
        row.appendChild(main);
        row.appendChild(el('div', 'vmeta', conv || ''));
        list.appendChild(row);
      });
      sec.appendChild(list);
    }
    body.appendChild(sec);
  }

  // ---------------------------------------------------------------- 文件
  async function viewFiles(path) {
    const body = $('#viewBody');
    body.innerHTML = '';
    body.appendChild(el('div', 'vempty', '正在读目录…'));
    let d;
    try {
      d = await vapi('/api/files?path=' + encodeURIComponent(path || '.'));
    } catch (e) {
      body.innerHTML = '';
      setViewHead('文件', path || '', [{ label: '返回对话', on: closeView }]);
      body.appendChild(el('div', 'vnote bad', '读不到目录：' + e.message));
      return;
    }
    state.fsPath = d.path;
    setViewHead('文件', d.items.length + ' 项', [
      { label: '上一级', icon: 'folder', on: () => viewFiles(d.parent) },
    ]);
    body.innerHTML = '';
    body.appendChild(el('div', 'vpath', d.path));

    if (!d.items.length) {
      body.appendChild(vEmpty('这个目录是空的', ''));
      return;
    }
    const card = el('div', 'vcard');
    d.items.forEach((it) => {
      const row = el('div', 'vrow tap');
      const ico = el('div', 'vico');
      ico.appendChild(iconNode(it.dir ? 'folder' : it.text ? 'msg-file' : 'cube'));
      row.appendChild(ico);
      const main = el('div', 'vmain');
      main.appendChild(el('div', 'vname', it.name + (it.dir ? '/' : '')));
      const bits = [];
      if (!it.dir) bits.push(fmtBytes(it.bytes));
      bits.push(fmtAgo(it.mtime));
      if (!it.dir && !it.text) bits.push('非文本');
      main.appendChild(el('div', 'vmeta', bits.join(' · ')));
      row.appendChild(main);
      row.onclick = () => {
        if (it.dir) viewFiles(it.path);
        else if (it.text) viewFileText(it.path);
        else {
          const n = el('div', 'vnote bad', '这是二进制文件，界面上不预览。');
          card.parentNode.insertBefore(n, card);
          setTimeout(() => n.remove(), 3500);
        }
      };
      card.appendChild(row);
    });
    body.appendChild(card);
  }

  async function viewFileText(path) {
    const body = $('#viewBody');
    body.innerHTML = '';
    body.appendChild(el('div', 'vempty', '正在读文件…'));
    let d;
    try {
      d = await vapi('/api/files/read?path=' + encodeURIComponent(path));
    } catch (e) {
      body.innerHTML = '';
      body.appendChild(el('div', 'vnote bad', '读不到文件：' + e.message));
      return;
    }
    setViewHead(d.name, fmtBytes(d.bytes) + (d.truncated ? ' · 已截断' : ''), [
      { label: '上一级', icon: 'folder', on: () => viewFiles(path.replace(/\/[^/]*$/, '') || '/') },
      {
        label: '问这个文件',
        pri: true,
        on: () => {
          closeView();
          send('请读一下 ' + path + '，然后概括它讲了什么');
        },
      },
    ]);
    body.innerHTML = '';
    body.appendChild(el('div', 'vpath', d.path));
    if (d.binary) {
      body.appendChild(el('div', 'vnote bad', '这个文件里有二进制内容，不预览。'));
      return;
    }
    const pre = el('pre', 'vcode', d.text || '(空文件)');
    body.appendChild(pre);
  }

  // ---------------------------------------------------------------- 知识库
  async function viewKb() {
    const body = $('#viewBody');
    body.innerHTML = '';
    body.appendChild(el('div', 'vempty', '正在读取知识库…'));
    let d;
    try {
      d = await vapi('/api/kb');
    } catch (e) {
      body.innerHTML = '';
      body.appendChild(el('div', 'vnote bad', '读不到知识库：' + e.message));
      return;
    }
    const rag = { on: d.rag };
    setViewHead('知识库', d.n_docs + ' 篇 · ' + d.n_chunks + ' 块 · ' + fmtBytes(d.bytes), []);
    body.innerHTML = '';

    // ---- 自动注入开关
    const sw = el('button', 'vbtn' + (d.rag ? ' pri' : ''), d.rag ? '自动注入：开' : '自动注入：关');
    sw.onclick = async () => {
      rag.on = !rag.on;
      try {
        const r = await vpost('/api/kb/rag', { on: rag.on });
        sw.className = 'vbtn' + (r.rag ? ' pri' : '');
        sw.textContent = r.rag ? '自动注入：开' : '自动注入：关';
      } catch (e) { /* 保持原状 */ }
    };

    const bar = el('div');
    bar.style.cssText = 'display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap';
    bar.appendChild(sw);

    const inp = el('input');
    inp.type = 'text';
    inp.placeholder = '路径，如 docs';
    const ibox = el('div', 'vinput');
    ibox.style.flex = '1 1 220px';
    ibox.appendChild(iconNode('folder'));
    ibox.appendChild(inp);
    const add = el('button', 'vbtn', '索引');
    add.onclick = async () => {
      const p = inp.value.trim();
      if (!p) { inp.focus(); return; }
      add.disabled = true;
      add.textContent = '索引中…';
      try {
        const r = await vpost('/api/kb/add', { path: p });
        inp.value = '';
        viewKb();
        return;
      } catch (e) {
        const n = el('div', 'vnote bad', '索引失败：' + e.message);
        body.insertBefore(n, bar);
        setTimeout(() => n.remove(), 6000);
      }
      add.disabled = false;
      add.textContent = '索引';
    };
    bar.appendChild(ibox);
    bar.appendChild(add);
    body.appendChild(bar);

    // ---- 检索
    const q = el('input');
    q.type = 'text';
    q.placeholder = '搜一句试试，比如「Q6_K 移位写错」';
    const qbox = el('div', 'vinput');
    qbox.appendChild(iconNode('search'));
    qbox.appendChild(q);
    body.appendChild(qbox);
    const results = el('div');
    results.style.marginTop = '12px';
    body.appendChild(results);

    const doSearch = async () => {
      const text = q.value.trim();
      if (!text) return;
      results.innerHTML = '';
      results.appendChild(el('div', 'vempty', '检索中…'));
      let r;
      try {
        r = await vpost('/api/kb/search', { q: text, k: 6 });
      } catch (e) {
        results.innerHTML = '';
        results.appendChild(el('div', 'vnote bad', '检索失败：' + e.message));
        return;
      }
      results.innerHTML = '';
      if (!r.hits.length) {
        results.appendChild(vEmpty('没有命中', '换个说法，或者先把相关目录索引进来。'));
        return;
      }
      const card = el('div', 'vcard');
      r.hits.forEach((h) => {
        const box = el('div', 'khit');
        const hd = el('div', 'khit-head');
        hd.appendChild(el('b', null, h.name));
        hd.appendChild(el('span', 'khit-score', h.score.toFixed(1)));
        box.appendChild(hd);
        const sn = el('div', 'khit-snip');
        // 把命中的查询词高亮出来 —— 检索是 BM25，能说清"为什么这条排前面"
        let html = escapeHtml(h.snippet);
        text.split(/\s+/).filter((w) => w.length >= 2).forEach((w) => {
          const safe = escapeHtml(w).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          html = html.replace(new RegExp(safe, 'gi'), (m) => '<mark>' + m + '</mark>');
        });
        sn.innerHTML = html;
        box.appendChild(sn);
        card.appendChild(box);
      });
      results.appendChild(card);
    };
    q.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); doSearch(); }
    });
    qbox.style.cursor = 'text';
    qbox.onclick = () => q.focus();

    // ---- 文档清单
    const sec = el('div', 'vsec');
    sec.style.marginTop = '16px';
    const head = el('div', 'vsec-head');
    head.appendChild(el('h3', null, '已索引'));
    head.appendChild(el('span', 'n', d.n_docs + ' 篇'));
    if (d.n_docs) {
      const clr = el('button', 'vbtn ghost', '清空');
      clr.style.marginLeft = 'auto';
      clr.onclick = async () => {
        if (!confirm('清空知识库索引？源文件不会被删除。')) return;
        try { await vpost('/api/kb/clear', {}); viewKb(); } catch (e) {}
      };
      head.appendChild(clr);
    }
    sec.appendChild(head);

    if (!d.docs.length) {
      sec.appendChild(vEmpty('还没有索引任何东西', '把 docs、README.md 这类目录或文件加进来，之后对话会自动参考它们。'));
    } else {
      const list = el('div', 'vcard');
      d.docs.forEach((doc) => {
        const row = el('div', 'vrow');
        const ico = el('div', 'vico');
        ico.appendChild(iconNode('book'));
        row.appendChild(ico);
        const main = el('div', 'vmain');
        const nm = el('div', 'vname', doc.name);
        if (doc.missing) nm.appendChild(el('span', 'vtag warn', '已删除'));
        main.appendChild(nm);
        main.appendChild(el('div', 'vmeta', doc.chunks + ' 块 · ' + fmtBytes(doc.bytes) + ' · ' + fmtAgo(doc.mtime)));
        row.appendChild(main);
        const rm = el('button', 'vbtn ghost', '移除');
        rm.onclick = async () => {
          try { await vpost('/api/kb/remove', { path: doc.path }); viewKb(); } catch (e) {}
        };
        row.appendChild(rm);
        list.appendChild(row);
      });
      sec.appendChild(list);
    }
    body.appendChild(sec);

    body.appendChild(el('div', 'vnote',
      '检索用 BM25（词频 + 逆文档频率），不是向量检索 —— 本地没有 embedding 模型，' +
      '而为了 RAG 硬塞一个 300 MB 的模型进来不划算。中文按二元组切词，' +
      '所以「本地推理」会被拆成 本地 / 地推 / 推理，召回偏高但排序仍然稳。'));
  }

  function bind() {
    setGlass(glassLevel(), false);
    bindLight();
    bindSidebar();

    $('#input').addEventListener('input', autoGrow);
    $('#input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
    });
    // 软键盘弹起会压缩视口，把最新一条消息顶出去。等键盘落定再回到底部。
    // 手机上软键盘弹出会改视口高度，底部可能被顶出屏幕。但这里不能强制
    // 贴底：用户正往上翻看历史时点一下输入框，就被拽回最新一条，
    // 会让人以为"输入框一点就丢失阅读位置"。只在本来就贴底时才补一下。
    $('#input').addEventListener('focus', () => setTimeout(() => scrollBottom(false), 340));
    if (window.visualViewport) {
      // 同理，键盘收起/弹出时不强制回底，只跟随
      window.visualViewport.addEventListener('resize', () => scrollBottom(false));
    }
    bindScrollStick();

    const cwdChip = $('#cwdChip');
    if (cwdChip) {
      cwdChip.onclick = () => {
        if (!state.current) newConv();
        const cur = state.current.cwd || (state.server && state.server.workspace) || '~';
        pickDir(cur, (p) => {
          if (p) state.current.cwd = p;
          else delete state.current.cwd;
          saveConvs();
          renderCwd();
          log(p ? '项目文件夹 → ' + p : '项目文件夹已解绑，跟随默认目录');
        });
      };
    }

    $('#send').onclick = () => {
      if (state.streaming) { if (state.abort) state.abort.abort(); return; }
      send();
    };
    $('#newChat').onclick = () => { newConv(); closeSidebar(); };
    const nc2 = $('#newChat2');
    if (nc2) nc2.onclick = () => { newConv(); closeSidebar(); };
    $('#toggleSidebar').onclick = () => {
      $('#sidebar').classList.add('open');
      $('#inspector').classList.remove('show');
      syncScrim();
    };
    const navClose = $('#navClose');
    if (navClose) navClose.onclick = closeSidebar;
    const insBtn = $('#toggleInspector');
    if (insBtn) insBtn.onclick = () => toggleInspector();
    const insClose = $('#inspClose');
    if (insClose) insClose.onclick = () => toggleInspector(false);
    $('#scrim').onclick = closeSidebar;

    // 输入栏按钮
    const bf = $('#btnFile');
    if (bf) bf.onclick = () => pickAttachment('file');
    const bim = $('#btnImage');
    if (bim) bim.onclick = () => { if (!bim.disabled) pickAttachment('image'); };
    const bm = $('#btnMode');
    if (bm) bm.onclick = openSettings;
    const mib = $('#modelInfoBtn');
    if (mib) mib.onclick = openSettings;
    const cl = $('#copyLog');
    if (cl) cl.onclick = () => {
      const txt = state.logs.map((l) => '[' + l.t + '] ' + l.msg).join('\n');
      navigator.clipboard.writeText(txt);
      log('Logs copied to clipboard.');
    };

    // 左栏搜索
    const si = $('#searchInput');
    if (si) {
      si.addEventListener('input', () => {
        state.q = si.value;
        renderConvList();
      });
    }
    $('#openSettings').onclick = openSettings;
    $('#openSettings2').onclick = openSettings;
    $('#modelChip').onclick = openSettings;
    $('#openApi').onclick = openApi;
    $('#sheetClose').onclick = closeSheet;
    $('#sheetSave').onclick = closeSheet;
    $('#mask').onclick = closeSheet;
    $('#btnTools').onclick = () => {
      state.tools = !state.tools;
      $('#btnTools').classList.toggle('on', state.tools);
    };
    // 按钮的亮灭必须跟着 state.tools 走。这里原来是硬编码的 true，
    // 工具关掉之后按钮还亮着 —— 用户点一下变成"关"，实际什么都没变。
    $('#btnTools').classList.toggle('on', state.tools);

    $('#btnAttach').onclick = pickAttachment;

    // 麦克风：点一下开始，再点一下结束。disabled 由 refreshState 按 mm_audio 控制。
    $('#btnMic').onclick = () => { rec.on ? stopRec(false) : startRec(); };

    // 联网搜索：只切一个本地开关，不改服务端权限。真正干活的还是 web_search
    // 工具，所以关掉「联网」权限时这个按钮也没意义 —— 一起联动。
    $('#btnWeb').onclick = () => {
      const s = state.server || {};
      if (!s.net) { log('联网权限已关闭（设置 → 权限 → 联网）', 'err'); return; }
      state.web = !state.web;
      $('#btnWeb').classList.toggle('on', state.web);
      log(state.web ? '联网搜索已开：下一个问题会先搜再答' : '联网搜索已关');
    };
    $('#fileInput').onchange = async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      if (f.size > 64 * 1024 * 1024) { alert('文件太大，上限 64 MB'); return; }
      try {
        await useAttachment(await readAttachment(f));
      } catch (err) {
        alert('读取文件失败：' + err.message);
      }
    };

    // 拖文件进窗口。桌面 App 里这条比点按钮更常用 —— 从 Finder 直接拖过来
    // 比"点附件 → 在面板里翻目录 → 选文件"快得多。
    let dragDepth = 0;
    const hasFiles = (e) => !!(e.dataTransfer &&
      Array.from(e.dataTransfer.types || []).includes('Files'));
    const takeFile = async (f) => {
      if (!f) return;
      if (f.size > 64 * 1024 * 1024) { alert('文件太大，上限 64 MB'); return; }
      try {
        await useAttachment(await readAttachment(f));
      } catch (err) {
        alert('读取文件失败：' + err.message);
      }
    };
    window.addEventListener('dragenter', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth++;
      document.body.classList.add('dropping');
    });
    window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
    window.addEventListener('dragleave', (e) => {
      if (!hasFiles(e)) return;
      if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dropping'); }
    });
    window.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth = 0;
      document.body.classList.remove('dropping');
      takeFile(e.dataTransfer.files && e.dataTransfer.files[0]);
    });

    document.querySelectorAll('.chip').forEach((c) => {
      c.onclick = () => send(c.dataset.p);
    });

    // 左栏导航：对话 + 五个真实视图。视图盖在对话区上，切回来时
    // 滚动位置和草稿都还在。
    document.querySelectorAll('.nav-item[data-view]').forEach((b) => {
      b.onclick = () => {
        document.querySelectorAll('.nav-item[data-view]').forEach((x) => x.classList.remove('on'));
        b.classList.add('on');
        if (b.dataset.view === 'chat') closeView();
        else openView(b.dataset.view);
      };
    });
    const vback = $('#viewBack');
    if (vback) vback.onclick = closeView;

    document.addEventListener('click', (e) => {
      // 先判"运行"：它和"复制"在同一行，两者的 closest 都会命中自己那颗按钮，
      // 顺序反了会把点击当成复制处理掉。
      const run = e.target.closest('[data-run]');
      if (run) {
        const box = run.closest('.codeblock');
        const code = box && box.querySelector('code');
        if (code) openPreview(code.textContent, box.dataset.lang);
        return;
      }
      const btn = e.target.closest('[data-copy]');
      if (btn) {
        const box = btn.closest('.codeblock') || btn.parentNode;
        const code = box.querySelector('code');
        if (!code) return;
        navigator.clipboard.writeText(code.textContent);
        btn.textContent = '已复制';
        setTimeout(() => (btn.textContent = '复制'), 1200);
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (pv && pv.panel.style.display !== 'none') { pv.close(); return; }
        closeSheet(); closeSidebar(); if (state.view) closeView();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const si = $('#searchInput');
        if (si && window.innerWidth > 860) si.focus();
        else { $('#sidebar').classList.add('open'); syncScrim(); if (si) si.focus(); }
      }
    });
  }

  // ---------------------------------------------------------------- 启动
  async function boot() {
    loadConvs();
    state.memory = loadMemory();
    bind();
    await refreshState();
    if (state.convs.length) {
      state.current = state.convs[0];
      $('#convTitle').textContent = state.current.title;
      renderConvList();
      renderMessages();
    } else {
      renderConvList();
    }
    setSendEnabled(true);
    renderLogs();
    renderTools();
    setGauge(0, 0, 0, 0);
    // 地址栏 hash 直接打开对应视图（刷新不丢位置）
    const h = location.hash.replace('#', '');
    if (h && VIEWS[h]) openView(h);
    setInterval(refreshState, 8000);
    // 宽屏不需要抽屉遮罩；窗口从窄变宽时清掉残留状态
    window.addEventListener('resize', () => {
      if (window.innerWidth > 1024) {
        $('#inspector').classList.remove('show');
        syncScrim();
      }
      if (window.innerWidth > 860) {
        $('#sidebar').classList.remove('open');
        syncScrim();
      }
    });
  }

  boot();
})();
