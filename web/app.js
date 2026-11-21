/* Dubhe Web UI */
(() => {
  'use strict';

  // 构建标记。界面出了问题的时候，"跑的是不是这一版的代码"必须先能确认 ——
  // 这一版就吃过亏：旧实例在内存里跑着 0.25.0 的 app.js，但窗口里新按钮
  // 照样显示（那是 index.html 提供的），于是所有排查都建立在一个错误前提上。
  // 版本号由服务端注入 index.html（见 serve_file），这里只负责读。
  window.__dubheBuild =
    (document.querySelector('meta[name="dubhe-version"]') || {}).content || 'dev';

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
  const THINK_KEY = 'dubhe.think.v1';

  function readPreference(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function writePreference(key, value) {
    try { localStorage.setItem(key, value); } catch (e) {}
  }

  // ---------------------------------------------------------------- 原生外壳通道
  //
  // 桌面版（apps/macos）把这份 UI 塞进 WKWebView，用一条只认几个 type 的
  // 小通道把"这一轮生成完了"这类事件告诉外壳 —— 外壳据此弹通知、点 Dock。
  //
  // 浏览器里 window.webkit 是 undefined，这里静默退化成空操作；
  // 不要在这里做任何"是不是 App"的分支判断，页面本身要完全一致。
  const shellPost = (payload) => {
    try {
      const h = window.webkit && window.webkit.messageHandlers
                && window.webkit.messageHandlers.shell;
      if (h) h.postMessage(payload);
    } catch (e) { /* 通道不在就当没这回事 */ }
  };

  // 通知正文不能用原始 markdown：代码块、表格、标题符号在通知里就是一堆乱码。
  // 去掉围栏块和行内标记，再把空白压平，超过 160 字截断。
  function shellPostTurnEnd(ai, tokens) {
    // 0 token 的"完成"不是完成：请求失败、被中断、或者根本没跑起来。
    // 这也顺带挡住了一种偶发的幽灵消息（页面重载时队列里残留的那一条），
    // 症状是"一开 App 就弹一条空通知"。
    if (!tokens) return;
    let t = String(ai.content || '')
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`[^`]*`/g, ' ')
      .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/^\s{0,3}#{1,6}\s*/gm, '')
      .replace(/[*_~#>|]+/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (!t) return;
    if (t.length > 160) t = t.slice(0, 160) + '…';
    shellPost({ type: 'turnEnd', body: t });
  }
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
    // 联网搜索：只在本次会话有效，按问题需要调用搜索工具。
    web: false,
    // 关掉时，支持直答的模型会跳过思考。其他模型只隐藏推理过程。
    think: readPreference(THINK_KEY) === '1',
    drafts: new Map(),
    logs: [],       // 右栏系统日志：只记真实事件，不编
    gen: null,      // 本轮生成统计 { tokens, tps, total, ttft }
    q: '',          // 左栏搜索关键词
    abort: null,    // 生成中的 AbortController，用于"停止"
    // 正在生成的那个对话对象。停止按钮只在这个对话里出现 —— 切到别的
    // 对话还显示停止键，按下去停的却是另一条，没人能猜到。
    genConv: null,
    // 排队发送。引擎一次只答一个，所以"在别的对话里点发送"只能排队 ——
    // 但排上之后用户就不必守着这条回答结束再手动发一次。
    queue: [],
    view: null,     // 当前打开的功能视图（null = 对话）
    fsPath: '',     // 文件视图当前目录
    memory: [],     // 记忆条目（见 loadMemory）
  };

  const settings = { temperature: 0.7, top_p: 0.95, max_tokens: 1024, system: '', tools: false };
  try { Object.assign(settings, JSON.parse(readPreference(SET_KEY) || '{}')); } catch (e) {}
  // 迁移：老版本的默认值是 tools:true，而且它随其它设置一起被存进了
  // localStorage。光把默认值改成 false 救不了老用户 —— Object.assign
  // 会被存储里的 true 盖掉。没拨过开关就说明这个 true 不是用户选的。
  if (!readPreference(TOOLS_TOUCHED)) settings.tools = false;
  // 输入框旁的工具栏按钮切的是 state.tools（本次会话），设置面板切的是
  // settings.tools（持久偏好）。两个开关如果各判各的，就会出现"按钮亮着
  // 但没生效"这种没法解释的状态。启动时以偏好为准，之后按钮独立。
  state.tools = settings.tools;

  // ---------------------------------------------------------------- 存储
  const HISTORY_IMPORTED = 'dubhe.history.imported.v1';
  const HISTORY_PENDING = 'dubhe.history.pending.v1:';
  const pendingHistory = new Map();
  const fallbackFailures = new Set();
  const isLocalPage = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
  let legacyHistory = [];
  let historyImported = false;
  let historyConnected = false;
  let historyWriting = false;
  let historyRetry;
  let historyError = '';
  // 保留旧记录中用于提示截断的字数。
  const TOOL_OUT_KEEP = 1500;

  function validConvs(items) {
    return Array.isArray(items) ? items.filter((c) =>
      c && typeof c.id === 'string' && c.id && Array.isArray(c.messages)) : [];
  }

  function recoverConvs(items) {
    return validConvs(items).map((c) => ({
      ...c,
      title: typeof c.title === 'string' ? c.title : '新对话',
      messages: c.messages.filter(Boolean).map((m) => {
        if (!m.streaming) return m;
        const next = { ...m, streaming: false };
        if (m.role !== 'assistant') return next;
        next.content = (m.content || '').trim()
          ? m.content + '\n\n> ⚠️ 上次的生成被中断了，这条回答可能不完整。'
          : '（上次的生成被中断了，重新提问即可。）';
        return next;
      }),
    }));
  }

  function orderedConvs(records) {
    return Array.from(records).sort((a, b) => (b.at || 0) - (a.at || 0));
  }

  function loadFallbackHistory() {
    const records = new Map();
    try {
      legacyHistory = validConvs(JSON.parse(readPreference(STORE_KEY) || '[]'));
      for (const c of recoverConvs(legacyHistory)) records.set(c.id, c);
    } catch (e) { /* 保留原来的历史数据 */ }
    historyImported = readPreference(HISTORY_IMPORTED) === '1';
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (!key?.startsWith(HISTORY_PENDING)) continue;
        const raw = localStorage.getItem(key);
        try {
          const [c] = recoverConvs([JSON.parse(raw)]);
          if (!c || key !== HISTORY_PENDING + c.id) continue;
          records.set(c.id, c);
          pendingHistory.set(c.id, { conv: c, raw });
        } catch (e) { /* 留下看不懂的原始记录以便恢复 */ }
      }
    } catch (e) { /* 浏览器可能禁用了存储，服务端仍可保存 */ }
    state.convs = orderedConvs(records.values());
  }

  function snapshotConv(c) {
    const messages = c.messages.map(({ queued, noteText, stopped, thinkTouched, ...m }) => {
      if (m.tools) m.tools = m.tools.map(({ askId, askPending, t0, ...tool }) => tool);
      return m;
    });
    return { ...c, messages };
  }

  function renderHistoryNotice() {
    const notice = $('#historyNotice');
    let text = '';
    if (!isLocalPage || historyError) {
      if (fallbackFailures.size) text = '有聊天记录还没保存，请导出备份。';
      else if (!isLocalPage) text = '聊天记录保存在当前浏览器。';
      else if (pendingHistory.size) text = '写入历史失败，聊天记录暂存在浏览器，稍后会重试。';
      else text = '读取历史失败，当前显示浏览器中的记录，稍后会重试。';
    }
    notice.textContent = text;
    notice.title = historyError;
    notice.hidden = !text;
  }

  function saveConv(conv) {
    const c = snapshotConv(conv);
    const raw = JSON.stringify(c);
    pendingHistory.set(c.id, { conv: c, raw });
    try {
      localStorage.setItem(HISTORY_PENDING + c.id, raw);
      fallbackFailures.delete(c.id);
    } catch (e) {
      fallbackFailures.add(c.id);
    }
    renderHistoryNotice();
    syncHistory();
  }

  function mergeHistory(items) {
    const previous = new Map(state.convs.map((c) => [c.id, c]));
    const records = new Map();
    for (const c of recoverConvs(items)) {
      const old = previous.get(c.id);
      if (old && pendingHistory.has(c.id)) continue;
      if (!old) {
        records.set(c.id, c);
        continue;
      }
      // 草稿、排队任务和目录选择器可能仍引用这个对话对象。
      Object.keys(old).forEach((key) => { delete old[key]; });
      Object.assign(old, c);
      records.set(c.id, old);
    }
    for (const [id, entry] of pendingHistory) records.set(id, previous.get(id) || entry.conv);
    state.convs = orderedConvs(records.values());
    const current = state.current;
    const draft = draftFor();
    if (current && records.has(current.id)) {
      if (!pendingHistory.has(current.id)) {
        $('#convTitle').textContent = current.title;
        renderMessages();
        renderTools();
        renderCwd();
      }
      renderConvList();
    } else if (current || !(draft.text || draft.att || draft.loading)) {
      activateConv(state.convs[0] || null);
    } else {
      renderConvList();
    }
  }

  async function syncHistory() {
    if (!isLocalPage || historyWriting || historyRetry) return;
    historyWriting = true;
    let failure = '';
    try {
      if (!historyConnected) {
        if (!historyImported && legacyHistory.length) {
          const r = await vpost('/api/history', { import: legacyHistory });
          if (!r.ok) throw new Error('未能导入旧记录');
          historyImported = true;
          writePreference(HISTORY_IMPORTED, '1');
        }
        const r = await vapi('/api/history');
        if (!Array.isArray(r.items)) throw new Error('历史数据格式有误');
        mergeHistory(r.items);
        historyConnected = true;
      }
      for (const [id, entry] of pendingHistory) {
        try {
          const r = await vpost('/api/history', { upsert: [entry.conv] });
          if (!r.ok) throw new Error('未能写入记录');
        } catch (e) {
          if (e.status !== 400 && e.status !== 413) throw e;
          // 单条记录格式或长度不合要求时，仍然保存其他对话。
          failure = e.message;
          continue;
        }
        // 同一对话可能在上传期间继续生成，只确认这次发送的版本。
        const saved = pendingHistory.get(id) === entry;
        if (saved) {
          pendingHistory.delete(id);
          fallbackFailures.delete(id);
        }
        try {
          const key = HISTORY_PENDING + id;
          if (saved && localStorage.getItem(key) === entry.raw) localStorage.removeItem(key);
        } catch (e) {}
      }
    } catch (e) {
      failure = e.message || '连接中断';
    } finally {
      historyWriting = false;
      historyError = failure;
      if (historyError) {
        historyRetry = setTimeout(() => {
          historyRetry = null;
          syncHistory();
        }, 4000);
      }
      renderHistoryNotice();
      if (pendingHistory.size && !historyRetry) syncHistory();
    }
  }

  function exportHistory() {
    const text = JSON.stringify({ items: state.convs.map(snapshotConv) }, null, 2);
    if (window.webkit?.messageHandlers?.shell) {
      shellPost({ type: 'historyExport', text });
      return;
    }
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const link = el('a');
    link.href = url;
    link.download = 'Dubhe-history.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function saveSettings() { writePreference(SET_KEY, JSON.stringify(settings)); }

  // ---------------------------------------------------------------- 记忆
  //
  // 两个作用域：
  //   conv:<id> —— 只属于这一个对话。默认作用域，写进来的东西不会跑到别的
  //                对话里去（"不同对话会串记忆"这条问题的正面解法）。
  //   global    —— 显式选择才写，所有对话共享。
  //
  // 记忆保存在当前浏览器，按对话 ID 或全局作用域读取。
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

  async function maybeCompress(conv, total, signal) {
    const folded = conv.folded || 0;
    if (total - folded <= COMPRESS_AT) return;
    const end = total - KEEP_RECENT;
    if (end <= folded) return;
    const slice = conv.messages.slice(folded, end)
      .filter((m) => m.content && (m.role === 'user' || m.role === 'assistant'))
      .map((m) => ({ role: m.role, content: m.content }));
    if (slice.length < 4) return;
    log('上下文压缩中（' + slice.length + ' 条）…');
    try {
      const r = await vpost('/api/summarize', { messages: slice }, signal);
      signal.throwIfAborted();
      const sum = (r && r.summary ? String(r.summary) : '').trim();
      if (!sum) { log('压缩没拿到摘要，本轮按原样发', 'err'); return; }
      conv.summary = (conv.summary ? conv.summary + '\n' : '') + sum;
      conv.folded = end;
      saveConv(conv);
      log('已压缩 ' + slice.length + ' 条 → 摘要 ' + sum.length + ' 字');
      if (state.current === conv) renderMessages();
    } catch (e) {
      if (signal.aborted) throw e;
      // 压缩失败不该挡住这一轮对话：原样发出去，下次再试
      log('压缩失败：' + e.message, 'err');
    }
  }

  function draftFor(conv = state.current) {
    const id = conv ? conv.id : '';
    if (!state.drafts.has(id)) state.drafts.set(id, { text: '', att: null, loading: null });
    return state.drafts.get(id);
  }

  function activateConv(c, focus = false) {
    const input = $('#input');
    draftFor().text = input.value;
    state.current = c;
    input.value = draftFor().text;
    autoGrow();
    renderAttach();
    renderQueue();
    renderConvList();
    renderMessages();
    renderTools();
    renderCwd();
    refreshComposer();
    $('#convTitle').textContent = c ? c.title : '新对话';
    if (focus) input.focus();
  }

  function newConv(keepDraft = false, focus = true) {
    // 新对话默认继承上一个对话绑定的项目文件夹 —— 像 Codex 那样"开个新
    // 会话继续在同一个项目里干活"，而不是每次都要重新选一遍目录。
    const inherit = state.current && state.current.cwd;
    const id = 'c' + crypto.getRandomValues(new Uint32Array(4)).join('-');
    const c = { id, title: '新对话', messages: [], at: Date.now() };
    if (inherit) c.cwd = inherit;
    state.convs.unshift(c);
    if (keepDraft && !state.current) state.drafts.set(c.id, draftFor());
    activateConv(c, focus);
    if (keepDraft) state.drafts.delete('');
    saveConv(c);
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

  function renderMarkdown(src, { codeActions = true, tables = false } = {}) {
    const lines = escapeHtml(src).split('\n');
    let html = '';
    let fence = '', codeLang = '', codeBuf = [];
    let listType = null;

    const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };
    const codeBlock = () => codeActions ? codeBlockHtml(codeLang, codeBuf.join('\n'))
      : '<pre><code>' + codeBuf.join('\n') + '</code></pre>';
    const cells = line => line.trim().replace(/^\||\|$/g, '').split(/(?<!\\)\|/).map(cell => cell.trim());

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];
      const line = raw.replace(/\s+$/, '');

      if (fence) {
        if (line.trimStart().startsWith(fence) && [...line.trim()].every(c => c === fence[0])) {
          html += codeBlock();
          fence = '';
        } else codeBuf.push(raw);
        continue;
      }
      const opener = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
      if (opener) {
        [, fence, codeLang] = opener;
        codeBuf = [];
        closeList();
        continue;
      }

      if (!line.trim()) { closeList(); continue; }

      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { closeList(); html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; continue; }

      if (tables && line.includes('|') && lines[i + 1]?.includes('|')) {
        const headings = cells(line);
        const rule = cells(lines[i + 1]);
        if (headings.length === rule.length && rule.every(cell => /^:?-{3,}:?$/.test(cell))) {
          closeList();
          html += '<table><thead><tr>' + headings.map(cell => '<th>' + inline(cell) + '</th>').join('') + '</tr></thead><tbody>';
          i++;
          while (lines[i + 1]?.includes('|') && cells(lines[i + 1]).length === headings.length) {
            html += '<tr>' + cells(lines[++i]).map(cell => '<td>' + inline(cell) + '</td>').join('') + '</tr>';
          }
          html += '</tbody></table>';
          continue;
        }
      }

      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ul) {
        if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; }
        html += `<li>${inline(ul[1])}</li>`;
        continue;
      }
      const ol = line.match(/^\s*(\d{1,9})[.)]\s+(.*)$/);
      if (ol) {
        if (listType !== 'ol') { closeList(); html += '<ol start="' + Number(ol[1]) + '">'; listType = 'ol'; }
        html += `<li>${inline(ol[2])}</li>`;
        continue;
      }
      const bq = line.match(/^>\s?(.*)$/);
      if (bq) { closeList(); html += `<blockquote>${inline(bq[1])}</blockquote>`; continue; }

      closeList();
      html += `<p>${inline(line)}</p>`;
    }
    if (fence && codeBuf.length) html += codeBlock();
    closeList();
    return html;
  }

  // 隐藏模型输出的工具调用/思考标记原文。
  //
  // 服务端现在会把它们从流里摘掉，这里再兜一层，理由有两个：
  //   1. 旧对话（0.25.0 之前存的）正文里本来就带着这些标签，重新打开时
  //      不该继续看到它们；
  //   2. 别家客户端/API 直接写进来的 messages 也可能带。
  // 剥完顺手压掉剩下的空行，否则气泡里会留一大片空白。
  function stripToolCalls(s) {
    let r = String(s || '');
    const cut = (open, close) => {
      for (;;) {
        const a = r.indexOf(open);
        if (a < 0) break;
        const b = r.indexOf(close, a + open.length);
        if (b < 0) { r = r.slice(0, a); break; }
        r = r.slice(0, a) + r.slice(b + close.length);
      }
    };
    cut('<tool_call>', '</tool_call>');
    cut('<tool_result>', '</tool_result>');
    cut('<function=', '</function>');
    r = r.replace(/<\/?(?:think|tool_call|tool_result|function|parameter)[^>]*>/g, '');
    return r.replace(/\n{3,}/g, '\n\n').trim();
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
      list.appendChild(el('div', 'conv-empty', '还没有对话。在下方输入框直接提问就会新建一条。'));
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
      const waiting = state.queue.filter((job) => job.conv === c).length;
      if (waiting) d.appendChild(el('span', 'conv-waiting', '排队 ' + waiting));
      d.onclick = () => {
        activateConv(c);
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
    if (card === 'calc') return '精确计算';
    return '';
  }

  // 工具名/命令标题的显示上限。
  //
  // 服务端已经把"整段话被当成工具名"挡在解析层了，这里再兜一层：卡片标题
  // 是一行长字，超过几十个字符就该截断，否则它会横向撑开整张卡片。
  function toolLabel(s) {
    const t = String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
    return t.length > 60 ? t.slice(0, 60) + '…' : t;
  }

  // 卡片主标题 = "它到底动了什么"，不是工具名。用户不关心工具叫 read_file，
  // 只关心读了哪个文件、跑了哪条命令。
  function cardTitle(t, meta) {
    const card = t.card || 'generic';
    if (card === 'terminal') return meta.command || t.name;
    if (card === 'diff' || card === 'read') return baseName(meta.path) || t.name;
    if (card === 'search') return meta.pattern || baseName(meta.path) || t.name;
    if (card === 'web') return meta.kind === 'fetch' ? hostOf(meta.url) : ('“' + (meta.query || '') + '”');
    if (card === 'calc') return meta.expr || t.name;
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
      head.appendChild(el('span', 'tool-name', toolLabel(t.name)));
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
    head.appendChild(el('span', 'tool-name', toolLabel(cardTitle(t, meta))));
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
      // 旧历史可能截过工具输出。回放时要说清楚，否则用户会
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

  // 消息 id。用途只有一个：流式生成期间用户切到别的对话再切回来，
  // DOM 会被 renderMessages 整表重建，闭包里那个 node 引用就悬空了 ——
  // 靠 id 才能在重建后的列表里把同一条消息找回来，接着往里写字。
  let msgSeq = 0;
  function newMsgId() {
    return 'm' + Date.now().toString(36) + '_' + (++msgSeq);
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
      if (m.id) wrap.dataset.mid = m.id;
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

    // 深度思考的推理过程。
    //
    // 默认折起来：用户打开这个开关是为了答案更准，不是想看几屏自言自语。
    // 但流式期间必须展开 —— 推理往往比答案长得多，那几十秒里正文一个字都
    // 没有，收起状态下界面和卡死没有区别。用户手动开合的状态记在 m 上，
    // 这样后面 reswap 重建节点时不会把人家展开的块又收回去。
    if (m.think) {
      const d = document.createElement('details');
      d.className = 'think-box';
      // 三态：用户手动开合过（thinkOpen 有值）就听他的；没碰过的话，
      // 流式期间展开、结束后收起。原来是 `m.streaming ? true : ...`，
      // 于是"收起来看正文"这个动作每来一段推理就被推翻一次。
      d.open = (m.thinkOpen === undefined) ? !!m.streaming : !!m.thinkOpen;
      d.addEventListener('toggle', () => {
        m.thinkTouched = true;
        m.thinkOpen = d.open;
      });
      const s = document.createElement('summary');
      s.textContent = '思考过程 · ' + m.think.length + ' 字';
      d.appendChild(s);
      d.appendChild(el('div', 'think-txt', m.think));
      col.appendChild(d);
    }

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
    if (m.id) wrap.dataset.mid = m.id;
    return markDrawn(wrap, m);
  }

  let emptyState;
  function renderMessages() {
    const box = $('#messages');
    box.innerHTML = '';
    const c = state.current;
    if (!c || !c.messages.length) {
      if (emptyState) box.appendChild(emptyState);
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
  // 离底多少像素以内算"在底部"。
  //
  // 原来是 72 —— 太松了。用户在底部往上滑个几十像素，仍然被判定成"在底部"，
  // 下一段文字一到就把他拽回去；反复滑就反复弹，表现就是"在答案底部上下
  // 滑动会跳"。12px 只够吸收亚像素误差和滚轮惯性，不会吃掉用户的上滑。
  const STICK_GAP = 12;

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
      // 自己滚到底产生的事件不算用户意图；但只要位置不在底部，无论是不是
      // 自己滚出来的，都说明内容被翻上去了 → 停止跟随。
      // 原来是一刀切 `if (progScroll) return;`：用户在那一帧里的上滑会被
      // 一起吞掉，下一段文字到达时又把他拽回底部 —— 这就是"底部上下滑会跳"。
      if (progScroll && atBottom(box)) return;
      // 内容还不到一屏时，"在底部"是恒真的 —— 不能拿它当"用户想跟随"的
      // 证据。否则就是这个现象：用户往上滑了一下（其实也没地方可滑），
      // 下一个 scroll 事件立刻把跟随打开，然后生成一开始，画面就被一路
      // 拽到底部，看起来像"上下滑动时不停跳闪"。
      if (box.scrollHeight <= box.clientHeight + 8) return;
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
      const s = await vapi('/api/state');
      const first = !state.server;
      state.server = s;
      $('#app').hidden = s.share_only === true;
      $('#shareNotice').hidden = !s.share_only;
      if (s.share_only) {
        $('#shareModel').textContent = s.loaded ? ('已加载 ' + (s.model_id || '模型')) : '未加载模型';
        return s;
      }
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
        bw.title = s.net ? '联网搜索（需要查证时使用）' : '联网权限已关闭（设置 → 权限）';
        if (!s.net && state.web) { state.web = false; bw.classList.remove('on'); }
      }
      const bt = $('#btnThink');
      if (bt) bt.title = s.info?.direct_answer
        ? '思考过程（关闭后直接回答）' : '显示模型的推理过程';
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

  // 顶栏胶囊里已经单独挂了量化标签（Q4_K_M / Q8_0 …），模型名里再带一遍
  // 就是十来个字符的纯浪费 —— 窄窗口下正是它把名字挤成 "Qwen2..."。
  // 名字只保留"是什么模型"，"量化成什么样"交给旁边的标签。
  function shortModelName(id) {
    const full = String(id || '');
    const cut = full.replace(
      /[-_.](?:UD-)?(?:(?:IQ|Q)\d+(?:_[A-Za-z0-9]+)*|F16|F32|BF16|FP16|fp16|f16)$/i, '');
    return cut || full;
  }

  function renderHead(s) {
    const name = $('#mpName');
    const tags = $('#mpTags');
    const live = $('#liveBadge');
    if (!name) return;
    tags.innerHTML = '';
    if (s && s.loaded && s.info) {
      name.textContent = shortModelName(s.model_id) || '已加载';
      name.title = s.model_id || '';
      const i = s.info;
      const bits = [
        (i.params / 1e9).toFixed(2) + 'B',
        quantLabel(s),
        fmtCtx(i.n_ctx),
      ].filter(Boolean);
      for (const t of bits) tags.appendChild(el('span', '', t));
      live.classList.add('on');
      live.querySelector('.live-s').textContent = '已加载';
    } else {
      name.textContent = s ? '未加载模型' : '服务未连接';
      live.classList.remove('on');
      live.querySelector('.live-s').textContent = s ? '空闲' : '未连接';
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
    nm.textContent = shortModelName(s.model_id) || i.arch;
    nm.title = s.model_id || '';
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
  let filePickerDraft;
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
    filePickerDraft = draftFor();
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
    const draft = draftFor();
    const a = draft.att;
    if (!a && !draft.loading) { bar.classList.add('hidden'); return; }
    bar.classList.remove('hidden');
    if (draft.loading) bar.appendChild(el('span', 'attach-chip', '正在读取附件…'));
    if (!a) return;
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
    x.onclick = () => {
      draft.att = null;
      renderAttach();
      refreshComposer();
    };
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
  async function attachFile(file, draft) {
    if (file.size > 23 * 1024 * 1024) {
      alert('文件太大，上限 23 MB');
      return;
    }
    const loading = {};
    draft.loading = loading;
    renderAttach();
    refreshComposer();
    try {
      const a = await readAttachment(file);
      if (draft.loading !== loading) return;
      if (!checkAttachment(a)) return;
      if (a.kind === 'image') a.thumb = await makeThumb(a.dataUrl);
      if (draft.loading === loading) draft.att = a;
    } catch (err) {
      if (draft.loading === loading) alert('读取文件失败：' + err.message);
    } finally {
      if (draft.loading === loading) {
        draft.loading = null;
        if (draft === draftFor()) {
          renderAttach();
          refreshComposer();
        }
      }
    }
  }

  function checkAttachment(a) {
    const s = state.server || {};
    if ((a.kind === 'image' || a.kind === 'video') && !s.mm_vision) {
      alert('当前模型没有视觉塔，看不了' + (a.kind === 'video' ? '视频' : '图片') +
            '。换 lfm3-vl 这类带 mmproj 的模型再试。');
      return false;
    }
    if (a.kind === 'audio' && !s.mm_audio) {
      alert('当前模型没有语音塔，听不了音频。换 lfm3-audio 这类带 mmproj 的模型再试。');
      return false;
    }
    return true;
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
    const draft = draftFor();
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
      rec.on = true; rec.ctx = ctx; rec.stream = stream; rec.proc = proc; rec.draft = draft;
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
    const draft = rec.draft;
    rec.draft = null;
    draft.att = {
      kind: 'audio', name, mime: 'audio/wav', fmt: 'wav',
      dataUrl: 'data:audio/wav;base64,' + b64Of(wav), bytes: wav.byteLength,
    };
    if (draft === draftFor()) {
      renderAttach();
      refreshComposer();
    }
    log('已录音 ' + (ms / 1000).toFixed(1) + 's · ' + (wav.byteLength / 1024).toFixed(0) +
        ' KB' + (auto ? '（到 60 秒自动停了）' : ''));
  }

  // ---------------------------------------------------------------- 联网搜索开关
  //
  // 打开之后在用户问题前面加一句"先搜再答"，让模型自己去调 web_search。
  // 为什么不在这里直接把搜索结果塞进 prompt：那样模型会把搜索摘要当成
  // 用户给的材料，不再引用来源；而走工具调用，卡片上能看到它搜了什么、
  // 搜到几条，用户能判断这个答案值不值得信。
  // ---------------------------------------------------------------- 发送
  function renderQueue() {
    const box = $('#queuedMessages');
    box.replaceChildren();
    for (const job of state.queue) {
      if (job.conv !== state.current) continue;
      const row = el('div', 'queue-row');
      row.appendChild(el('span', 'queue-state', '排队中'));
      const label = el('span', 'queue-text', [job.content, job.att?.name].filter(Boolean).join(' · '));
      label.title = label.textContent;
      row.appendChild(label);
      const cancel = el('button', 'act-btn', '取消');
      cancel.onclick = () => {
        const i = state.queue.indexOf(job);
        if (i < 0) return;
        state.queue.splice(i, 1);
        renderQueue();
        renderConvList();
        refreshComposer();
      };
      row.appendChild(cancel);
      box.appendChild(row);
    }
    box.hidden = !box.childElementCount;
  }

  function send(text, opts = {}) {
    const input = $('#input');
    const fromComposer = text == null;
    const draft = draftFor();
    if (fromComposer && draft.loading) return;
    const content = (fromComposer ? input.value : text).trim();
    const att = fromComposer ? draft.att : null;
    if (!content && !att) return;

    const conv = state.current || newConv(fromComposer, fromComposer);
    state.queue.push({
      conv, content, att,
      workspace: conv.cwd || state.server?.workspace,
      web: state.web, think: state.think,
      tools: opts.tools ?? state.tools,
      settings: { ...settings },
      memory: memoryForConv(conv).map((m) => m.text),
    });
    if (fromComposer) {
      draft.text = input.value = '';
      draft.att = null;
      autoGrow();
      renderAttach();
    }
    renderQueue();
    renderConvList();
    drainQueue();
  }

  async function drainQueue() {
    if (state.streaming || !state.queue.length) return;
    const job = state.queue.shift();
    state.streaming = true;
    state.genConv = job.conv;
    state.abort = new AbortController();
    renderQueue();
    renderConvList();
    refreshComposer();
    try {
      await runJob(job, state.abort.signal);
    } catch (e) {
      log('发送失败：' + e.message, 'err');
    } finally {
      state.streaming = false;
      state.genConv = null;
      state.abort = null;
      refreshComposer();
      drainQueue();
    }
  }

  async function runJob(job, signal) {
    const { conv, content, att, settings: opts } = job;
    let wire = content;
    if (att?.kind === 'text') {
      const head = att.name.replace(/[`\s]/g, '_');
      wire = '```' + head + '\n' + att.text + '\n```' + (content ? '\n\n' + content : '');
    }

    const previousCount = conv.messages.length;
    const userMsg = { role: 'user', content, id: newMsgId() };
    if (att) userMsg.att = { kind: att.kind, name: att.name, thumb: att.thumb, mime: att.mime, bytes: att.bytes };
    const ai = { role: 'assistant', content: '', tools: [], meta: null, streaming: true,
      noteText: '正在准备回答…', id: newMsgId() };
    conv.messages.push(userMsg, ai);
    if (conv.title === '新对话') {
      const t = content || att.name;
      conv.title = t.slice(0, 24) + (t.length > 24 ? '…' : '');
      if (state.current === conv) $('#convTitle').textContent = conv.title;
    }

    state.gen = { tokens: 0, tps: 0, total: 0, ttft: 0 };
    setGauge(0, 0, 0, 0);
    renderMeters(state.server);

    const box = $('#messages');
    let node = messageNode(ai);
    let body = node.querySelector('.body');
    body.classList.add('typing');
    if (state.current === conv) {
      emptyState?.remove();
      box.append(messageNode(userMsg), node);
      scrollBottom(true);
    }
    saveConv(conv);
    renderConvList();

    const t0 = performance.now();
    let tFirst = 0;
    const updateGauge = () => {
      const elapsed = (performance.now() - t0) / 1000;
      const decoding = tFirst ? Math.max((performance.now() - tFirst) / 1000, 0.001) : 0;
      state.gen.total = elapsed;
      setGauge(decoding > 0 ? Math.max(0, state.gen.tokens - 1) / decoding : 0,
        state.gen.tokens, elapsed, state.gen.ttft);
    };
    const clock = setInterval(updateGauge, 500);

    try {
      // 保留队列位置直到压缩和生成都结束。摘要只看这次发送前的历史。
      await maybeCompress(conv, previousCount, signal);
      signal.throwIfAborted();
      log('Running inference ...');

      const live = conv.messages.slice(conv.folded || 0);
      const msgs = live
        .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
        .map((m) => {
          if (m === userMsg && att && att.kind !== 'text') {
            const parts = [attachmentParts(att)];
            if (content) parts.push({ type: 'text', text: content });
            return { role: 'user', content: parts };
          }
          // 历史附件只留说明，服务端只对最后一条用户消息处理媒体。
          const c = m === userMsg ? wire : (m.content || '');
          return { role: m.role, content: m.att ? (c ? c + '\n' : '') + `[${m.att.kind}: ${m.att.name}]` : c };
        });
      if (job.memory.length) {
        msgs.unshift({
          role: 'system',
          content: '以下是用户希望你记住的背景信息（不要逐条复述，只在相关时自然使用）：\n'
            + job.memory.map((t, i) => (i + 1) + '. ' + t).join('\n'),
        });
      }
      if (conv.summary) {
        msgs.unshift({
          role: 'system',
          content: '以下是本对话更早部分的摘要（原文已折叠，不要假装记得原文细节）：\n' + conv.summary,
        });
      }
      if (opts.system) msgs.unshift({ role: 'system', content: opts.system });
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({
          messages: msgs,
          stream: true,
          temperature: opts.temperature,
          top_p: opts.top_p,
          max_tokens: opts.max_tokens,
          dubhe_tools: job.tools && !att,
          dubhe_search: job.web && !att,
          thinking: job.think,
          workspace: job.workspace || undefined,
        }),
      });
      if (!resp.ok) throw new Error(await responseError(resp));

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let completed = false;
      let raf = null;

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
        // 用户正在看别的对话：先把引用接回来；接不回来就只更新 ai 上的状态，
        // 反正 reswap 的意义是"让当前 DOM 反映最新状态"，没有 DOM 就跳过。
        if (!node.isConnected && !rebind()) return;
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
        // 节点换了，缓存的那几个思考块的引用全部作废
        thinkBox = thinkText = thinkSum = null;
        paint();
      };

      // 思考过程走"原地追加"，不走 reswap。
      //
      // reswap 是整块重建消息节点。工具卡片那种"结构变了"的更新用它没问题，
      // 但推理是每几十毫秒来一段的连续文本，重建有三个副作用，用户全都看得见：
      //   1) <details> 的 open 被重置 → 手动收起，下一段推理又把它弹开；
      //   2) 新节点会重播 .msg 的入场动画 → 整个气泡在"跳闪"；
      //   3) 每帧重建整棵子树，滚动位置跟着抖，上下滑根本滑不住。
      // 所以这里节点只建一次，之后只 append 文本，并用 rAF 合并写入。
      let thinkBox = null, thinkText = null, thinkSum = null;
      let thinkBuf = '', thinkRaf = 0;

      const ensureThink = () => {
        if (thinkBox && thinkBox.isConnected) return;
        const exist = node.querySelector('.think-box');
        if (exist) {
          thinkBox = exist;
          thinkSum = exist.querySelector('summary');
          thinkText = exist.querySelector('.think-txt');
          return;
        }
        thinkBox = el('details', 'think-box');
        thinkSum = el('summary', '', '思考过程');
        thinkText = el('div', 'think-txt');
        thinkBox.appendChild(thinkSum);
        thinkBox.appendChild(thinkText);
        // 流式期间默认展开（否则那几十秒里正文一个字都没有，看着像卡死），
        // 但用户一旦手动收过，就一直听他的。
        thinkBox.open = ai.thinkOpen !== false;
        thinkBox.addEventListener('toggle', () => {
          ai.thinkTouched = true;
          ai.thinkOpen = thinkBox.open;
        });
        if (ai.think) thinkText.textContent = ai.think;
        thinkSum.textContent = '思考过程 · ' + (ai.think || '').length + ' 字';
        thinkBuf = '';   // 已经在上面一次性画全了
        body.parentNode.insertBefore(thinkBox, body);
      };

      const flushThink = () => {
        if (!thinkText) return;
        if (thinkBuf) {
          thinkText.appendChild(document.createTextNode(thinkBuf));
          thinkBuf = '';
        }
        thinkSum.textContent = '思考过程 · ' + thinkText.textContent.length + ' 字';
      };

      const appendThink = (t) => {
        if (!t) return;
        ai.think = (ai.think || '') + t;
        thinkBuf += t;
        if (thinkRaf) return;
        thinkRaf = requestAnimationFrame(() => {
          thinkRaf = 0;
          if (!ai.streaming) return;
          if (!node.isConnected && !rebind()) return;
          ensureThink();
          flushThink();
          if (stick.on) scrollBottom();
        });
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
          if (!ai.streaming) return;
          // 用户切到别的对话去了：这个 node 已经不在文档里，写进去也没人看。
          // 先把引用接回列表里的新节点；接不回来就这一帧不画，ai.content
          // 仍然在长，切回来时 renderMessages 会一次性画出最新的。
          if (!node.isConnected && !rebind()) return;
          paintNow();
          scrollBottom();
        });
      };

      // 把 node/body 重新绑到"当前 DOM 里那条消息"上。
      //
      // 为什么需要：切走对话会触发 renderMessages 整表重建，闭包里的 node
      // 变成游离节点。之后所有 delta 都写进这个孤儿 —— 表现就是用户说的
      // "切回来字幕不动了，再切一次又冒出一段"。
      const rebind = () => {
        if (node && node.isConnected) return true;
        if (!ai.id) return false;
        const fresh = document.querySelector('#messages .msg[data-mid="' + ai.id + '"]');
        if (!fresh) return false;   // 正在看别的对话
        node = fresh;
        body = node.querySelector('.body');
        if (!body) return false;
        body.classList.add('typing');
        // messageNode 已经把内容整段写进 .body 了。这里清掉，重建成
        // "稳定头 + 尾部"两段式，接着往里面追加（否则内容会写两遍）。
        body.innerHTML = '';
        headEl = document.createElement('div');
        tailEl = document.createElement('div');
        headEl.className = 'md-head';
        tailEl.className = 'md-tail';
        body.appendChild(headEl);
        body.appendChild(tailEl);
        lastHead = null;
        thinkBox = thinkText = thinkSum = null;   // 思考块的引用也随节点作废
        paintNow();
        return true;
      };

      const setStage = (label) => {
        if (!ai.queued && ai.noteText === label) return false;
        ai.queued = false;
        if (label) ai.noteText = label;
        else delete ai.noteText;
        reswap();
        return true;
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
          if (data === '[DONE]') { completed = true; continue; }
          let j;
          try { j = JSON.parse(data); } catch (e) { continue; }

          if (j.dubhe) {
            if (j.dubhe.type === 'tool_call') {
              ai.tools.push({ name: j.dubhe.name, args: j.dubhe.arguments, ok: true, t0: performance.now() });
              log('Tool call: ' + j.dubhe.name);
              setStage(j.dubhe.name === 'web_search' ? '正在联网搜索…' : '正在使用工具…');
              renderTools();
            } else if (j.dubhe.type === 'think') {
              // 推理过程逐段追加到已有的折叠块里，不重建节点 —— 重建会把
              // 用户收起的块重新弹开，还会让气泡一直重播入场动画。
              setStage('正在思考…');
              appendThink(j.dubhe.text || '');
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
              if (!ai.content) setStage(j.dubhe.message || '');
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
              setStage('等待确认…');
              scrollBottom();
            } else if (j.dubhe.type === 'tool_ask_done') {
              const t = ai.tools.find((x) => x.askId === j.dubhe.id);
              if (t) {
                t.askPending = false;
                t.askAllow = j.dubhe.allow === true;
                t.askTimeout = j.dubhe.timeout === true;
              }
              renderTools();
              if (!setStage(t?.askAllow ? '正在使用工具…' : '正在准备回答…')) reswap();
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
              if (!setStage('正在准备回答…')) reswap();
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
            if (ai.queued || ai.noteText) setStage(null);
            ai.content += delta.content;
            // 每片算一个 token。这个服务端是按 token 逐个 on_piece 推的，
            // 所以计数是准的；最后那份 usage 还会再校正一次。
            //
            // 不加这一行的话，生成过程中右栏永远是"已生成 0 / 0 tok/s" ——
            // 明明字在往外冒，仪表却是死的，看着就像界面卡住了。
            state.gen.tokens = (state.gen.tokens || 0) + 1;
            paint();
          }
          if (j.usage) {
            ai.meta = { tps: j.usage.decode_tps, tokens: j.usage.completion_tokens };
            state.gen.tokens = j.usage.completion_tokens || 0;
            state.gen.tps = j.usage.decode_tps || 0;
          }
        }
      }
      // 流结束了，把 rAF 里还没落盘的那一小段补上（通常最后一个 chunk
      // 正好卡在一帧里，不补的话推理末尾会少几个字）。
      flushThink();
      if (!completed && !ai.errText) {
        throw new Error('回答中断，已收到的内容仍保留在对话里。');
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        // 用户主动停止：保留已生成的部分，不报错
        ai.stopped = true;
        if (!ai.content) ai.content = '（已停止）';
      } else {
        // 引擎中途没了的时候，WebKit 报的就是一句 "Load failed"，用户完全
        // 无法把它和"网络断了""服务没起来"区分开。翻译成人话，并且说清楚
        // 下一步该干什么 —— 自动重启一般 1～3 秒，重发就行。
        const raw = e.message || '未知错误';
        const msg = /load failed|failed to fetch|network ?error|networkconnectionlost/i.test(raw)
          ? '与本地引擎的连接中断了（引擎可能刚意外退出，客户端正在自动重启）。等几秒重新发送即可。'
          : raw;
        ai.errText = msg;
      }
    } finally {
      clearInterval(clock);
      ai.streaming = false;
      delete ai.queued;    // 状态行用的临时标记，不该跟着会话存下去
      delete ai.noteText;
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
      if (!String(ai.content || '').trim() && !(ai.tools && ai.tools.length) && !ai.errText) {
        ai.content = '（这一轮没有任何输出。可能是上下文放不下，或者模型直接收尾了，换个问法再试。）';
      }
      if (state.current === conv) {
        // 切离再切回时，页面已重画过；只替换现在属于这条回答的节点。
        const shown = box.querySelector('[data-mid="' + ai.id + '"]');
        if (shown) shown.replaceWith(messageNode(ai));
        scrollBottom();
        renderTools();
      }
      // 桌面版：这一轮真的生成完了，告诉外壳一声（外壳决定要不要弹通知）。
      // 用户主动停止的不算 —— 那说明他正盯着屏幕。
      if (!ai.stopped && !ai.errText) shellPostTurnEnd(ai, tokens);
      delete ai.stopped;          // 只用于刚才那个判断，不该跟着会话存下去
      saveConv(conv);
      renderMeters(state.server);
      refreshState();
    }
  }

  function refreshComposer() {
    const b = $('#send');
    const d = draftFor();
    const busyHere = state.streaming && state.genConv === state.current;
    const hasDraft = Boolean($('#input').value.trim() || d.att || d.loading);
    const stopOnly = busyHere && !hasDraft;
    $('#stopGeneration').hidden = !busyHere || stopOnly;
    b.disabled = Boolean(d.loading);
    b.classList.toggle('stop', stopOnly);
    b.title = stopOnly ? '停止生成' : state.streaming ? '加入队列' : '发送';
    b.setAttribute('aria-label', b.title);
    if (b.dataset.action !== String(stopOnly)) {
      b.innerHTML = stopOnly ? ICON_STOP : ICON_SEND;
      b.dataset.action = String(stopOnly);
    }
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

  function toggle(parent, label, desc, get, set) {
    const sw = el('label', 'switch');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = get();
    cb.onchange = () => set(cb.checked);
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
      if (window.webkit?.messageHandlers?.shell) {
        body.appendChild(el('div', 'group-title', '版本与更新'));
        const actions = el('div', 'tbar');
        for (const [label, type] of [['检查更新', 'checkForUpdates'], ['更新日志', 'showChangelog']]) {
          const button = el('button', 'vbtn', label);
          button.onclick = () => shellPost({ type });
          actions.appendChild(button);
        }
        body.appendChild(actions);
      }
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
      slider(body, '最大输出', '单次回复的 token 上限', 'max_tokens', 128, 16384, 128, (v) => String(v));

      body.appendChild(el('div', 'group-title', '工具'));
      toggle(body, '启用工具调用', '允许模型读取文件、列目录、执行命令', () => settings.tools, (v) => { settings.tools = v; state.tools = v; writePreference(TOOLS_TOUCHED, '1'); saveSettings(); });

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
      toggle(body, '命令隔离', '启用后，命令只能写入工作区和 /tmp。',
        () => srv().sandbox_cfg !== false, async (v) => { await setPerm('sandbox', v); openSettings(); });
      if (sbOff) {
        const warn = el('div', 'perm-warn',
          '本机暂时无法隔离命令。保持隔离开启时，引擎会拒绝执行命令。');
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
    return readPreference(GLASS_KEY) || autoGlass();
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

  function apiError(data) {
    const error = data?.error;
    return (typeof error === 'string' ? error : error?.message) || data?.message || '';
  }

  async function responseError(r) {
    const data = await r.json().catch(() => ({}));
    if (r.status === 413) return '内容超出服务器限制，请缩短消息或换更小的附件。';
    return apiError(data) || 'HTTP ' + r.status;
  }

  async function vapi(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      const error = new Error(await responseError(r));
      error.status = r.status;
      throw error;
    }
    return r.json().catch(() => ({}));
  }
  const vpost = (p, body, signal) =>
    vapi(p, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}), signal });

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
    translate: { title: '翻译', render: () => viewTranslate() },
    scan: { title: '扫描', render: () => viewScan() },
    tasks: { title: '任务与存储', render: () => viewTasks() },
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
    $('#app').dataset.view = name;
    $('#views').hidden = false;
    $('#viewBody').innerHTML = '';
    setViewHead(v.title, '', []);
    // 地址栏带上是哪个视图：刷新不丢位置，也能直接分享 /#kb
    try { history.replaceState(null, '', '#' + name); } catch (e) {}
    document.querySelectorAll('.nav-item[data-view]').forEach((x) => {
      const selected = x.dataset.view === name;
      x.classList.toggle('on', selected);
      const advanced = selected && x.closest('.advanced-nav');
      if (advanced) advanced.open = true;
    });
    v.render();
  }

  function closeView() {
    state.view = null;
    delete $('#app').dataset.view;
    $('#views').hidden = true;
    try { history.replaceState(null, '', location.pathname + location.search); } catch (e) {}
    document.querySelectorAll('.nav-item[data-view]').forEach((x) =>
      x.classList.toggle('on', x.dataset.view === 'chat'));
  }

  async function saveLocalTask(kind, id, title, payload, status) {
    if (!isLocalPage || state.server?.share_only) return;
    try {
      await vpost('/api/tasks', { upsert: { kind, id, title: title.slice(0, 100), payload } });
      status.textContent = '已保存在本机';
    } catch (e) {
      status.textContent = '任务未保存 · ' + e.message;
    }
  }

  async function fingerprintFile(file) {
    const hash = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    return Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, '0')).join('');
  }

  async function viewTasks() {
    const body = $('#viewBody');
    const section = el('section', 'vsec');
    const storage = el('section', 'vsec');
    const list = el('div', 'vcard');
    section.append(el('h3', 'task-section-title', '最近任务'), list);
    body.replaceChildren(section, storage);
    setViewHead('任务与存储', '', [
      { label: '新建扫描', on: () => openView('scan') },
      { label: '新建翻译', on: () => openView('translate') },
    ]);
    const visible = () => state.view === 'tasks' && section.isConnected;
    list.appendChild(vEmpty('正在读取本机任务'));
    try {
      const data = await vapi('/api/tasks');
      if (!visible()) return;
      const tasks = Array.isArray(data.tasks) ? data.tasks : [];
      list.replaceChildren();
      if (!tasks.length) list.appendChild(vEmpty('还没有任务', '开始扫描或翻译后，结果会在这里保存。'));
      for (const task of tasks) {
        const row = el('div', 'vrow task-row');
        const main = el('div', 'vmain');
        const kind = task.kind === 'scan' ? '扫描' : '翻译';
        main.append(el('div', 'vname', task.title), el('div', 'vmeta',
          kind + ' · ' + fmtAgo(task.updated / 1000) + ' · ' + fmtBytes(task.bytes)));
        const resume = el('button', 'vbtn', '打开');
        resume.setAttribute('aria-label', '打开任务 ' + task.title);
        resume.onclick = async () => {
          resume.disabled = true;
          try {
            const { task: entry } = await vapi('/api/tasks?id=' + encodeURIComponent(task.id));
            if (!visible()) return;
            openView(task.kind === 'scan' ? 'scan' : 'translate');
            const pane = task.kind === 'scan' ? scanPane : translationPane;
            pane.restoreTask(entry);
          } catch (e) {
            main.appendChild(el('div', 'vnote bad', '打不开任务 · ' + e.message));
          } finally {
            resume.disabled = false;
          }
        };
        const remove = el('button', 'vbtn ghost', '移除');
        remove.setAttribute('aria-label', '移除任务 ' + task.title);
        remove.onclick = async () => {
          if (scanPane?.usesTask(task.id) || translationPane?.usesTask(task.id)) {
            main.appendChild(el('div', 'vnote', '请先在任务页面停止并清空当前任务。'));
            return;
          }
          if (!confirm('移除这个任务及其进度？原文件会保留。')) return;
          remove.disabled = true;
          try {
            await vpost('/api/tasks', { delete: task.id });
            if (visible()) viewTasks();
          } catch (e) {
            main.appendChild(el('div', 'vnote bad', '移除失败 · ' + e.message));
            remove.disabled = false;
          }
        };
        row.append(main, resume, remove);
        list.appendChild(row);
      }
    } catch (e) {
      if (visible()) list.replaceChildren(vEmpty('读不到本机任务', e.message));
    }
    try {
      const data = await vapi('/api/store');
      if (!visible() || !data.storage) return;
      const usage = data.storage;
      const card = el('div', 'vcard');
      const usageRow = el('div', 'vrow');
      const main = el('div', 'vmain');
      main.append(el('div', 'vname', '模型目录'),
        el('div', 'vmeta', '剩余 ' + fmtBytes(usage.free_bytes) + ' · 已下载 ' +
          fmtBytes(usage.installed_bytes) + ' · 未完成 ' + fmtBytes(usage.partial_bytes)));
      const models = el('button', 'vbtn', '管理模型');
      models.onclick = () => openView('models');
      usageRow.append(main, models);
      card.appendChild(usageRow);
      for (const item of (data.items || []).filter(item => item.can_clear_partial)) {
        const row = el('div', 'vrow task-row');
        const label = el('div', 'vmain');
        label.append(el('div', 'vname', item.name), el('div', 'vmeta', '未完成的下载'));
        const clear = el('button', 'vbtn ghost', '清理');
        clear.onclick = async () => {
          if (!confirm('清理 ' + item.name + ' 的未完成下载？')) return;
          clear.disabled = true;
          try {
            await vpost('/api/store/clear-partial', { id: item.id });
            if (visible()) viewTasks();
          } catch (e) {
            label.appendChild(el('div', 'vnote bad', e.message));
            clear.disabled = false;
          }
        };
        row.append(label, clear);
        card.appendChild(row);
      }
      storage.replaceChildren(el('h3', 'task-section-title', '设备存储'), card);
    } catch (e) {
      if (visible()) storage.replaceChildren(vEmpty('读不到存储信息', e.message));
    }
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

  // 可下载区。清单来自 /api/store —— 服务端内置、链接验证过、架构与引擎匹配。
  // 下载是后台 curl（断点续传），这里只负责画进度和轮询。
  const storeViews = new WeakMap();
  const activeTransfer = (m) => ['running', 'verifying', 'cancelling'].includes(m.state);
  async function viewStore(host) {
    if (!host.isConnected || state.view !== 'models') return;
    let view = storeViews.get(host);
    if (!view) {
      view = { request: 0, timer: null, active: new Set(), list: el('div'), mirror: null };
      host.appendChild(view.list);
      storeViews.set(host, view);
    }
    clearTimeout(view.timer);
    const request = ++view.request;
    let d;
    try {
      d = await vapi('/api/store');
    } catch (e) {
      if (request === view.request && view.active.size) view.timer = setTimeout(() => viewStore(host), 1200);
      return;
    }
    if (request !== view.request || !host.isConnected || state.view !== 'models') return;
    const installed = d.items?.some(m => m.installed && view.active.has(m.id));
    view.list.replaceChildren();
    if (!d.items || !d.items.length) return;
    view.active = new Set(d.items.filter(activeTransfer).map((m) => m.id));

    const heading = el('div', 'vtitle store-heading', '模型商店');
    if (d.storage) {
      const space = el('span', 'vmeta store-space', '可用 ' + fmtBytes(d.storage.free_bytes));
      heading.appendChild(space);
    }
    const sources = el('a', 'vbtn ghost', '下载来源');
    sources.href = 'https://github.com/yuelangmanle/dubhe/blob/main/docs/model-downloads.md';
    sources.target = '_blank';
    sources.rel = 'noopener noreferrer';
    heading.appendChild(sources);
    view.list.appendChild(heading);
    const card = el('div', 'vcard');
    const scannerFiles = d.items.filter(m => m.scan);
    if (scannerFiles.length) {
      const row = el('div', 'vrow');
      const symbol = el('div', 'vico');
      symbol.appendChild(iconNode('scan'));
      const description = el('div', 'vmain');
      const title = el('div', 'vname', 'Xiaomi-OCR-0');
      if (scannerFiles.every(m => m.installed)) title.append(el('span', 'vtag on', '已下载'));
      description.append(title, el('div', 'vmeta', '本地文档识别 · ' +
        fmtBytes(scannerFiles.reduce((sum, m) => sum + m.bytes, 0))));
      const open = el('button', 'vbtn', '去扫描');
      open.onclick = () => openView('scan');
      row.append(symbol, description, open);
      card.appendChild(row);
    }
    d.items.filter(m => !m.scan).forEach((m) => {
      const row = el('div', 'vrow');
      const ico = el('div', 'vico');
      ico.appendChild(iconNode('cube'));
      row.appendChild(ico);

      const main = el('div', 'vmain');
      const nm = el('div', 'vname', m.name);
      nm.appendChild(el('span', 'vtag', m.params + ' · ' + m.quant));
      if (m.recommended) nm.appendChild(el('span', 'vtag on', '适合先试'));
      if (m.installed) nm.appendChild(el('span', 'vtag on', '已下载'));
      main.appendChild(nm);
      main.appendChild(el('div', 'vmeta', m.note + ' · ' + fmtBytes(m.installed ? m.got || m.bytes : m.bytes)));
      if (!m.installed && d.storage?.free_bytes < Math.max(0, m.bytes - (m.got || 0))) {
        main.appendChild(el('div', 'vnote', '可用空间不足'));
      }
      row.appendChild(main);

      if (m.installed) {
        if (m.translation) {
          const open = el('button', 'vbtn', '去翻译');
          open.onclick = () => chooseTranslation(m.id);
          row.appendChild(open);
        } else {
          row.appendChild(el('div', 'vmeta', '已就绪'));
        }
        card.appendChild(row);
        return;
      }

      if (activeTransfer(m)) {
        if (m.state === 'running') {
          const got = m.got || 0;
          const pct = m.total ? Math.min(100, Math.round((got / m.total) * 100)) : 0;
          const bar = el('div', 'pbar');
          const fill = el('div', 'pfill');
          fill.style.width = pct + '%';
          bar.appendChild(fill);
          main.appendChild(el('div', 'vmeta', '下载中 ' + pct + '% · ' + fmtBytes(got) + ' / ' + fmtBytes(m.total)));
          main.appendChild(bar);
        } else {
          main.appendChild(el('div', 'vmeta', m.state === 'verifying' ? '正在验证模型文件…' : '正在取消…'));
        }
        if (m.state !== 'cancelling') {
          const cx = el('button', 'vbtn', '取消');
          cx.onclick = async () => {
            cx.disabled = true;
            try {
              await vpost('/api/store/cancel', { id: m.id });
              viewStore(host);
            } catch (e) {
              cx.disabled = false;
              if (main.isConnected) main.appendChild(el('div', 'vnote bad', e.message));
            }
          };
          row.appendChild(cx);
        }
        card.appendChild(row);
        return;
      }

      const action = m.state === 'error' ? '重试' : m.state === 'cancelled' ? '继续' : '下载';
      const b = el('button', 'vbtn', action);
      b.onclick = async () => {
        b.disabled = true;
        b.textContent = '启动…';
        try {
          await vpost('/api/store/download', { id: m.id });
          view.active.add(m.id);
          viewStore(host);
        } catch (e) {
          if (e.status === 409 && e.message.includes('正在停止')) {
            b.textContent = '等待下载状态…';
            viewStore(host);
            return;
          }
          b.disabled = false;
          b.textContent = action;
          const note = el('div', 'vnote bad', '下载失败：' + e.message);
          if (row.isConnected) row.before(note);
          setTimeout(() => note.remove(), 6000);
        }
      };
      row.appendChild(b);
      if (m.state === 'error' && m.error) {
        main.appendChild(el('div', 'vnote bad', m.error));
      }
      card.appendChild(row);
    });
    view.list.appendChild(card);

    if (!view.mirror) view.mirror = createMirrorSetting(host);
    view.mirror.refresh(d.mirror || '');

    if (view.active.size) view.timer = setTimeout(() => viewStore(host), 1200);
    if (installed) viewModels(false);
  }

  function createMirrorSetting(host) {
    const mrow = el('div', 'vrow');
    const mm = el('div', 'vmain');
    mm.appendChild(el('div', 'vname', '下载源'));
    const inp = document.createElement('input');
    inp.className = 'vsel';
    inp.setAttribute('aria-label', '模型下载源');
    inp.style.width = '100%';
    inp.placeholder = 'https://hf-mirror.com（留空 = 官方 huggingface.co）';
    mm.appendChild(inp);
    mm.appendChild(el('div', 'vmeta', '只替换域名，路径不变。改完对之后开始的下载生效。'));
    mrow.appendChild(mm);
    const save = el('button', 'vbtn', '保存');
    const status = el('div', 'vmeta');
    status.setAttribute('role', 'status');
    mm.appendChild(status);
    inp.oninput = () => { status.textContent = ''; };
    let savedValue = '';
    const refresh = (value) => {
      if (!save.disabled && inp.value === savedValue && document.activeElement !== inp) inp.value = value;
      savedValue = value;
    };
    save.onclick = async () => {
      const value = inp.value.trim();
      save.disabled = true;
      status.textContent = '';
      try {
        const result = await vpost('/api/store/mirror', { mirror: value });
        savedValue = result.mirror ?? value;
        if (inp.value.trim() === value) inp.value = savedValue;
        status.textContent = '已保存';
        viewStore(host);
      } catch (e) {
        status.textContent = '保存失败 · ' + e.message;
      } finally {
        save.disabled = false;
      }
    };
    mrow.appendChild(save);
    host.appendChild(mrow);

    return { refresh };
  }

  // ---------------------------------------------------------------- 模型
  function confirmModelDeletion(name) {
    const dialog = el('dialog', 'delete-confirm');
    const title = el('h3', '', '删除模型？');
    title.id = 'delete-model-title';
    dialog.setAttribute('aria-labelledby', title.id);
    const form = el('form', 'delete-confirm-actions');
    form.method = 'dialog';
    const cancel = el('button', 'vbtn', '取消');
    cancel.value = 'cancel';
    cancel.autofocus = true;
    const remove = el('button', 'vbtn', '删除');
    remove.value = 'delete';
    form.append(cancel, remove);
    dialog.append(title, el('p', '', name + ' 将从模型目录移除。'), form);
    document.body.appendChild(dialog);
    return new Promise(resolve => {
      dialog.addEventListener('close', () => {
        resolve(dialog.returnValue === 'delete');
        dialog.remove();
      }, { once: true });
      dialog.showModal();
    });
  }

  let modelsRequest = 0;
  async function viewModels(refreshStore = true) {
    const body = $('#viewBody');
    let local = body.querySelector('.local-models');
    let storeBox = body.querySelector('.model-store');
    if (!local) {
      local = el('div', 'local-models');
      storeBox = el('div', 'model-store');
      local.appendChild(el('div', 'vempty', '正在扫描模型目录…'));
      body.append(local, storeBox);
    }
    const request = ++modelsRequest;
    let d;
    try {
      d = await vapi('/api/models');
    } catch (e) {
      if (request !== modelsRequest || !local.isConnected || state.view !== 'models') return;
      local.replaceChildren(el('div', 'vnote bad', '读不到模型清单：' + e.message));
      return;
    }
    if (request !== modelsRequest || !local.isConnected || state.view !== 'models') return;
    const actions = [{ label: '刷新', icon: 'restart', on: () => viewModels() }];
    actions.push({ label: '任务与存储', on: () => openView('tasks') });
    if (d.current) actions.push({
      label: '卸载当前模型',
      on: async () => {
        try {
          await vpost('/api/unload');
          shellPost({ type: 'modelUnloaded' });
          await refreshState();
          if (state.view === 'models') await viewModels();
        } catch (e) {
          if (local.isConnected) local.prepend(el('div', 'vnote bad', e.message));
        }
      },
    });
    setViewHead('模型', d.items.length + ' 个 · ' + d.dir, actions);
    local.replaceChildren();
    if (!d.items.length) {
      local.appendChild(vEmpty('模型目录是空的', d.dir + ' 下没有 .gguf 文件。把模型放到这里就能在界面上切换。'));
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
      if (m.translation) {
        const use = el('button', 'vbtn', '去翻译');
        use.onclick = () => chooseTranslation(m.translation);
        row.appendChild(use);
      } else if (m.loadable === false) {
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
            if (!card.isConnected || state.view !== 'models') return;
            await viewModels();
            // 塔加载失败（文件坏了、维度和模型对不上）不该静默吞掉：
            // 结果是"模型能跑但看不了图"，不说清楚用户会以为是图片的问题。
            if (r.mm_note && state.view === 'models') {
              const note = el('div', 'vnote bad', '模型已切换，但' + r.mm_note);
              local.prepend(note);
              setTimeout(() => note.remove(), 8000);
            }
          } catch (e) {
            b.disabled = false;
            b.textContent = '加载';
            const note = el('div', 'vnote bad', '加载失败：' + e.message);
            if (card.isConnected) card.before(note);
            setTimeout(() => note.remove(), 6000);
          }
        };
        row.appendChild(b);

      }
      if (!m.loaded && !m.mlx && !isMm) {
        const del = el('button', 'vbtn', '删除');
        del.onclick = async () => {
          del.disabled = true;
          try {
            if (!await confirmModelDeletion(m.name)) return;
            await vpost('/api/models/delete', { path: m.path });
            if (card.isConnected && state.view === 'models') await viewModels();
          } catch (e) {
            const note = el('div', 'vnote bad', '删除失败：' + e.message);
            if (card.isConnected) card.before(note);
            setTimeout(() => note.remove(), 6000);
          } finally {
            del.disabled = false;
          }
        };
        row.appendChild(del);
      }
      card.appendChild(row);
    });
    if (d.items.length) local.appendChild(card);
    if (refreshStore) viewStore(storeBox);
  }

  // ---------------------------------------------------------------- 翻译
  //
  const LANGS = [
    ['zh', '简体中文'], ['en', 'English'], ['ja', '日本語'],
    ['ko', '한국어'], ['fr', 'Français'], ['de', 'Deutsch'],
  ];

  // 围栏和缩进代码原样保留。普通文字按段落拆，长 Markdown 块只在整行边界拆。
  function translationParts(text, max = 1200) {
    const lines = text.match(/[^\n]*\n|[^\n]+$/g) || [];
    const parts = [];
    const literal = (value) => { if (value) parts.push({ value, translate: false }); };
    const add = (value) => {
      const content = value.trim();
      if (!content) {
        literal(value);
        return;
      }
      const start = value.indexOf(content);
      literal(value.slice(0, start));
      parts.push({ value: content, translate: true });
      literal(value.slice(start + content.length));
    };
    const dequote = (line) => line.replace(/^(?: {0,3}>[ \t]?)+/, '').replace(/\n$/, '');
    const listPrefix = /^ {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+/;
    const heading = (line) => /^ {0,3}#{1,6}(?:[ \t]|$)/.test(dequote(line));
    const list = (line) => listPrefix.test(dequote(line));
    const code = (line) => /^(?: {4}|\t)/.test(dequote(line));
    const fence = (line) => {
      const m = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(dequote(line).replace(listPrefix, ''));
      if (m && (m[1][0] !== '`' || !m[2].includes('`'))) return m[1];
      return '';
    };
    const mathBlock = line => /^[ \t]*([$][$]|\\\[)[ \t]*$/.exec(dequote(line))?.[1];
    const table = (i) => {
      if (!lines[i]?.includes('|') || !lines[i + 1]) return false;
      const rule = lines[i + 1].trim().replace(/^\||\|$/g, '');
      return rule.split('|').every((cell) => /^\s*:?-+:?\s*$/.test(cell));
    };
    const boundary = (i) => !lines[i].trim() || fence(lines[i]) || mathBlock(lines[i]) ||
      code(lines[i]) || heading(lines[i]) || table(i);

    function split(value, structured) {
      while (value.trimEnd().length > max) {
        let cut = value.lastIndexOf('\n', max) + 1;
        if (!cut || !value.slice(0, cut).trim()) {
          if (structured) {
            const next = value.indexOf('\n', max);
            if (next < 0) break;
            cut = next + 1;
          } else {
            const head = value.slice(0, max);
            const mark = Array.from(head.matchAll(/[ \t]+|[。！？.!?](?=\s|$)/g)).at(-1);
            cut = mark && mark.index > max / 2 ? mark.index + mark[0].length : max;
            // 避免在两个 UTF-16 代理项中间切开一个字符。
            if (/[\uD800-\uDBFF]/.test(value[cut - 1])) cut--;
          }
        }
        add(value.slice(0, cut));
        value = value.slice(cut);
      }
      add(value);
    }

    let i = 0;
    while (i < lines.length) {
      const start = i++;
      const line = lines[start];
      const marker = fence(line);
      const math = mathBlock(line);
      if (marker) {
        const closer = new RegExp('^ {0,3}' + marker[0] + '{' + marker.length + ',}[ \t]*$');
        while (i < lines.length && !closer.test(dequote(lines[i++]))) {}
        literal(lines.slice(start, i).join(''));
      } else if (math) {
        const end = math === '$$' ? '$$' : '\\]';
        while (i < lines.length && dequote(lines[i++]).trim() !== end) {}
        literal(lines.slice(start, i).join(''));
      } else if (!line.trim() || code(line)) {
        literal(line);
      } else if (table(start)) {
        i++;
        while (i < lines.length && lines[i].includes('|') && !heading(lines[i]) && !fence(lines[i])) i++;
        split(lines.slice(start, i).join(''), true);
      } else {
        const isHeading = heading(line);
        const isList = list(line);
        // 标题和紧随的正文放在同一段，保留原有空行。
        if (isHeading) {
          let next = i;
          while (next < lines.length && !lines[next].trim()) next++;
          if (next < lines.length && !boundary(next) && !list(lines[next])) i = next;
        }
        while (i < lines.length && !boundary(i) && (isList || !list(lines[i]))) i++;
        split(lines.slice(start, i).join(''), isList || isHeading);
      }
    }
    return parts;
  }

  const TRANSLATION_PREFS = 'dubhe.translation.v1';
  let translationPrefs = { model: 'hy-mt2-18b-q4', source: 'auto', target: 'zh', glossary: '', ocr: 'vision' };
  try {
    const saved = JSON.parse(readPreference(TRANSLATION_PREFS));
    if (saved && typeof saved === 'object') translationPrefs = { ...translationPrefs, ...saved };
  } catch (e) {}
  let translationPane;
  let nativeTranslation;
  let documentRequestId = 0;
  const documentRequests = new Map();

  window.dubheDocumentComplete = (id, result) => {
    const pending = documentRequests.get(id);
    if (!pending) return;
    documentRequests.delete(id);
    if (result.error) pending.reject(new Error(result.error));
    else pending.resolve(result);
  };

  function nativeDocumentRequest(payload) {
    return new Promise((resolve, reject) => {
      const handler = window.webkit?.messageHandlers?.shell;
      if (!handler) { reject(new Error('请在 Dubhe Mac App 中处理 PDF 或 Word。')); return; }
      const id = ++documentRequestId;
      documentRequests.set(id, { resolve, reject });
      try {
        handler.postMessage({ type: 'translationDocument', id, ...payload });
      } catch (e) {
        documentRequests.delete(id);
        reject(e);
      }
    });
  }

  function documentBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(',', 2)[1]);
      reader.onerror = () => reject(reader.error || new Error('读不到文件。'));
      reader.readAsDataURL(file);
    });
  }

  function setTranslationPrefs(changes) {
    translationPrefs = { ...translationPrefs, ...changes };
    writePreference(TRANSLATION_PREFS, JSON.stringify(translationPrefs));
  }

  function chooseTranslation(model) {
    setTranslationPrefs({ model });
    openView('translate');
  }

  function translationGlossary(value) {
    const pairs = String(value || '').split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    if (pairs.length > 64) throw new Error('术语表最多支持 64 行。');
    return pairs.map((line, i) => {
      const pair = /^(.*?)\s*(?:→|=)\s*(.+)$/.exec(line);
      if (!pair || !pair[1].trim() || !pair[2].trim()) {
        throw new Error('术语表第 ' + (i + 1) + ' 行请填写「原文 = 译文」。');
      }
      return { source: pair[1].trim(), target: pair[2].trim() };
    });
  }

  // Protect literal spans before sending text to the model. A changed or
  // missing marker is a failed segment, never a successfully translated file.
  const protectedTranslationSpans = new RegExp([
    /^[ \t]*\|?[ \t]*:?-{3,}:?(?:[ \t]*\|[ \t]*:?-{3,}:?)+[ \t]*\|?[ \t]*(?=\r?$)/.source,
    /^[ \t]*(?:#{1,6}[ \t]+|(?:>[ \t]?)+|(?:[-+*]|\d+[.)])[ \t]+)/.source,
    /`+[^`\r\n]+`+|\[\[@[^\]\r\n]+\]\]|\[@[^\]\r\n]+\]|\[\d+(?:[ ,;–−-]+\d+)*\]/.source,
    /\([^()\r\n]+?et al\.,?[ \t]*(?:19|20)\d\d[a-z]?[^()\r\n]*\)/.source,
    /[$][$][\s\S]+?[$][$]|\$(?!\s)[^$\r\n]+?(?<!\s)\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]/.source,
    /\]\((?:[^()\r\n]|\([^()\r\n]*\))*\)|https?:\/\/[^\s<>()|\]]+|<[^<>\r\n]+>/.source,
    /\{\{[^{}\r\n]+\}\}|\$\{[^{}\r\n]+\}|\{(?:[A-Za-z_][\w.-]*|\d+)\}|%[sdif]|\\[^\r\n]/.source,
    /[+−-]?\d[\d,.]*(?:[eE][+−-]?\d+)?(?:[ \t]*(?:(?:[µμkcm]?[mgsL]|Pa|kPa|MPa|Hz|W|J|h|°[CF])[²³]?(?:\/(?:[µμkcm]?[mgsL]|h|day)[²³]?)?|[%％‰])(?![\p{L}]))?/.source,
    /\r?\n|\||!?\[|\]|⟦\d+⟧/.source,
  ].join('|'), 'gmu');

  function protectTranslation(value) {
    const literals = [];
    const masked = value.replace(protectedTranslationSpans, span => '⟦' + (literals.push(span) - 1) + '⟧');
    const restore = (translated) => {
      let index = 0;
      let valid = true;
      const output = translated.replace(/⟦(\d+)⟧/g, (_, id) => {
        if (id !== String(index) || index >= literals.length) valid = false;
        return literals[index++] || '';
      });
      return valid && index === literals.length ? output : null;
    };
    return { masked, literals, restore };
  }

  function prepareTranslation(text) {
    const parts = translationParts(text);
    return { parts, total: parts.filter(p => p.translate).length, next: 0, finished: 0, result: '' };
  }

  async function translateDocument(draft, prefs, signal, progress, render, checkpoint) {
    const glossary = translationGlossary(prefs.glossary);
    const request = async (text, formatted) => {
      const data = await vpost('/api/translate', {
        model: prefs.model, source: prefs.source, target: prefs.target, text, formatted, glossary,
      }, signal);
      signal.throwIfAborted();
      if (typeof data.text !== 'string' || !data.text.trim()) {
        throw new Error('第 ' + (draft.finished + 1) + ' 段没有完整译文。请重试或换一个模型。');
      }
      return data.text.trim();
    };
    while (draft.next < draft.parts.length) {
      signal.throwIfAborted();
      const p = draft.parts[draft.next];
      if (!p.translate) {
        draft.result += p.value;
        draft.next++;
        continue;
      }
      const part = (draft.finished + 1) + ' / ' + draft.total + ' 段';
      progress(part);
      const fragment = protectTranslation(p.value);
      const reply = /[\p{L}]/u.test(fragment.masked)
        ? await request(fragment.masked, !!fragment.literals.length) : fragment.masked;
      let piece = fragment.restore(reply);
      if (piece === null) {
        // Translate the text between markers separately if the model moved or
        // dropped one. The original punctuation, values and layout stay local.
        progress(part + ' · 恢复格式');
        piece = '';
        const spans = fragment.masked.split(/⟦\d+⟧/g);
        for (let i = 0; i < spans.length; i++) {
          signal.throwIfAborted();
          const span = spans[i];
          if (/[\p{L}]/u.test(span)) {
            const translated = await request(span.trim(), false);
            piece += span.match(/^\s*/)[0] + translated.replace(/\r?\n/g, ' ') + span.match(/\s*$/)[0];
          } else piece += span;
          if (i < fragment.literals.length) piece += fragment.literals[i];
        }
      }
      draft.result += piece;
      draft.finished++;
      draft.next++;
      render(draft.result);
      if (checkpoint) await checkpoint();
    }
    render(draft.result);
  }

  async function translateNative(text, source) {
    nativeTranslation?.abort();
    if (!text?.trim()) {
      shellPost({ type: 'overlay', title: '翻译', text: source === 'ocr'
        ? '选区里没有识别到文字。可以框选更清晰的区域。'
        : '没有读到选中的文字。请重新选中文字，或用截图翻译。' });
      return;
    }
    const job = new AbortController();
    nativeTranslation = job;
    shellPost({ type: 'overlay', title: '翻译中', text: '正在翻译…', source: text });
    try {
      const draft = prepareTranslation(text);
      await translateDocument(draft, translationPrefs, job.signal, () => {}, () => {});
      if (!job.signal.aborted) shellPost({ type: 'overlay', title: '译文', text: draft.result, source: text });
    } catch (e) {
      if (!job.signal.aborted) shellPost({ type: 'overlay', title: '翻译失败', text: e.message, source: text });
    } finally {
      if (nativeTranslation === job) nativeTranslation = null;
    }
  }

  window.dubheOcr = (text) => translateNative(text, 'ocr');
  window.dubheOCRLanguage = () => translationPrefs.source;
  window.dubheOCRProvider = () => translationPrefs.ocr === 'xiaomi' ? 'xiaomi' : 'vision';
  window.dubheText = (text) => translateNative(text, 'selection');
  window.dubheCancelTranslation = () => nativeTranslation?.abort();
  window.dubheTranslateLocalSelection = () => {
    const field = document.activeElement;
    const text = typeof field?.selectionStart === 'number'
      ? field.value.slice(field.selectionStart, field.selectionEnd)
      : window.getSelection()?.toString();
    translateNative(text, 'selection');
    return true;
  };

  function viewTranslate() {
    setViewHead('翻译', '', [
      { label: '最近任务', on: () => openView('tasks') },
      { label: '模型商店', on: () => openView('models') },
    ]);
    translationPane ??= createTranslationPane();
    $('#viewBody').appendChild(translationPane.body);
    translationPane.refreshModels();
  }

  function translationBlocks(preview) {
    return Array.from(preview.children).flatMap(node => {
      const kind = node.localName;
      if (kind === 'ul' || kind === 'ol') {
        return Array.from(node.children).map((item, index) => ({
          kind: 'li', text: (kind === 'ol' ? (node.start + index) + '. ' : '• ') + item.textContent,
        }));
      }
      if (kind === 'table') return [{
        kind, text: Array.from(node.rows).map(row =>
          Array.from(row.cells).map(cell => cell.textContent).join(' │ ')).join('\n'),
      }];
      return [{ kind: kind === 'pre' ? 'code' : kind, text: node.textContent }];
    });
  }

  function createTranslationPane() {
    const body = el('div');
    let catalog;
    let catalogRequest = 0;
    let modelError = '';
    let job = null;
    let work = null;
    let taskId = null;

    const card = el('div', 'vcard');

    // 模型 + 目标语言
    const row1 = el('div', 'vrow translation-settings');
    const m1 = el('div', 'vmain');
    m1.appendChild(el('div', 'vname', '翻译模型'));
    const selM = document.createElement('select');
    selM.className = 'vsel';
    selM.setAttribute('aria-label', '翻译模型');
    m1.appendChild(selM);
    const modelStatus = el('div', 'vmeta');
    m1.appendChild(modelStatus);
    row1.appendChild(m1);

    function languageField(title, key, options) {
      const label = el('label', 'vmain');
      label.appendChild(el('span', 'vname', title));
      const select = el('select', 'vsel');
      for (const [value, name] of options) {
        const option = el('option', '', name);
        option.value = value;
        select.appendChild(option);
      }
      select.value = translationPrefs[key];
      select.onchange = () => {
        setTranslationPrefs({ [key]: select.value });
        updateControls();
      };
      label.appendChild(select);
      row1.appendChild(label);
      return select;
    }
    const selS = languageField('原文语言', 'source', [['auto', '自动识别'], ...LANGS]);
    const selL = languageField('目标语言', 'target', LANGS);
    card.appendChild(row1);
    body.appendChild(card);

    // 原文
    const ta = document.createElement('textarea');
    ta.className = 'field';
    ta.rows = 12;
    ta.setAttribute('aria-label', '原文');
    ta.placeholder = '粘贴要翻译的文字，或读取文件。';
    ta.oninput = () => updateControls();
    const workspace = el('div', 'translation-workspace');
    const srcCard = el('section', 'vcard translation-card');
    srcCard.append(el('h3', '', '原文'), ta);
    workspace.appendChild(srcCard);
    const terms = el('details', 'vcard translation-glossary');
    terms.appendChild(el('summary', '', '术语表'));
    const glossary = el('textarea', 'field');
    glossary.rows = 3;
    glossary.setAttribute('aria-label', '术语表');
    glossary.placeholder = '每行一项，例如 留一流域验证 = leave-one-basin-out validation';
    glossary.value = translationPrefs.glossary;
    glossary.oninput = () => {
      setTranslationPrefs({ glossary: glossary.value });
      updateControls();
    };
    const presetKey = 'dubhe.glossary-presets.v1';
    let presets = [];
    try {
      const saved = JSON.parse(readPreference(presetKey));
      if (Array.isArray(saved)) presets = saved;
    } catch (e) {}
    const presetTools = el('div', 'tbar glossary-presets');
    const presetName = el('input', 'vsel');
    presetName.placeholder = '术语表名称';
    presetName.maxLength = 40;
    presetName.setAttribute('aria-label', '术语表名称');
    const presetSelect = el('select', 'vsel');
    presetSelect.setAttribute('aria-label', '已保存的术语表');
    const savePreset = el('button', 'vbtn', '保存术语表');
    const loadPreset = el('button', 'vbtn', '使用');
    const removePreset = el('button', 'vbtn ghost', '删除');
    const presetStatus = el('span', 'vmeta');
    presetStatus.setAttribute('role', 'status');
    function showPresets(current = '') {
      const empty = el('option', '', '选择术语表');
      empty.value = '';
      empty.disabled = true;
      presetSelect.replaceChildren(empty, ...presets.map(({name}) => {
        const option = el('option', '', name);
        option.value = name;
        return option;
      }));
      presetSelect.value = presets.some(item => item.name === current) ? current : '';
      loadPreset.disabled = removePreset.disabled = Boolean(job || !presetSelect.value);
    }
    presetSelect.onchange = () => { loadPreset.disabled = removePreset.disabled = !presetSelect.value; };
    savePreset.onclick = () => {
      const name = presetName.value.trim();
      if (!name || !glossary.value.trim()) {
        presetStatus.textContent = '先填写术语表和名称。';
        return;
      }
      if (glossary.value.length > 40_000) {
        presetStatus.textContent = '术语表过长，请先精简。';
        return;
      }
      const index = presets.findIndex(item => item.name === name);
      if (index < 0 && presets.length >= 30) {
        presetStatus.textContent = '术语表已达 30 份，可先删除一份。';
        return;
      }
      if (index >= 0 && !confirm('更新已有的术语表「' + name + '」？')) return;
      if (index < 0) presets.push({ name, text: glossary.value });
      else presets[index].text = glossary.value;
      writePreference(presetKey, JSON.stringify(presets));
      showPresets(name);
      presetStatus.textContent = '已保存术语表';
    };
    loadPreset.onclick = () => {
      const selected = presets.find(item => item.name === presetSelect.value);
      if (!selected) return;
      glossary.value = selected.text;
      glossary.oninput();
      presetName.value = selected.name;
      presetStatus.textContent = '已使用「' + selected.name + '」';
    };
    removePreset.onclick = () => {
      const name = presetSelect.value;
      if (!name || !confirm('删除术语表「' + name + '」？')) return;
      presets = presets.filter(item => item.name !== name);
      writePreference(presetKey, JSON.stringify(presets));
      showPresets();
      presetStatus.textContent = '已删除术语表';
    };
    presetTools.append(presetName, savePreset, presetSelect, loadPreset, removePreset, presetStatus);
    showPresets();
    terms.append(glossary, presetTools);

    const bar = el('div', 'tbar');
    const fileBtn = el('button', 'vbtn', '读文件');
    const clearFile = el('button', 'vbtn ghost', '清空工作区');
    clearFile.id = 'translationClear';
    const sample = el('button', 'vbtn ghost', '试用示例');
    const relink = el('button', 'vbtn ghost', '重新选择原文件');
    let relinking = false;
    clearFile.hidden = true;
    let sourceFile = null;
    let fileName = '';
    let exportedParts = null;
    let readingFile = false;
    let exporting = false;
    let outputReady = false;
    const inputFile = document.createElement('input');
    inputFile.type = 'file';
    inputFile.accept = '.txt,.md,.markdown,.csv,.log';
    if (window.webkit?.messageHandlers?.shell) inputFile.accept += ',.docx,.pdf';
    inputFile.style.display = 'none';
    clearFile.onclick = () => {
      sourceFile = exportedParts = null;
      includeSource.checked = false;
      fileName = '';
      work = null;
      taskId = null;
      savedStatus.textContent = '';
      ta.value = '';
      setOutput('');
      copyBtn.disabled = true;
      clearFile.hidden = true;
      prog.textContent = '';
      updateControls();
    };
    relink.onclick = () => { relinking = true; inputFile.click(); };
    sample.onclick = () => {
      acceptScan('Our field survey began after the rain. We measured the river level at three locations.', '英文试用示例');
      prog.textContent = '已填入示例，选择目标语言后即可翻译。';
    };
    fileBtn.onclick = () => { relinking = false; inputFile.click(); };
    inputFile.onchange = async () => {
      const restoreOriginal = relinking;
      relinking = false;
      const file = inputFile.files?.[0];
      inputFile.value = '';
      if (!file || readingFile || job) return;
      const format = /\.(docx|pdf)$/i.exec(file.name)?.[1].toLowerCase();
      readingFile = true;
      updateControls();
      prog.textContent = '正在读取文件…';
      try {
        let next = null;
        let content;
        if (format) {
          if (file.size > 16 * 1024 * 1024) throw new Error('PDF 或 Word 文件请控制在 16 MB 内。');
          const digest = await fingerprintFile(file);
          if (restoreOriginal) {
            if (format !== sourceFile?.format || digest !== sourceFile.digest) {
              throw new Error('选择的文件和此任务的原件不同。');
            }
            sourceFile.document = await documentBase64(file);
            prog.textContent = '原文件已恢复，可按原格式导出。';
            return;
          }
          const data = await documentBase64(file);
          const { paragraphs } = await nativeDocumentRequest({ format, document: data });
          if (!Array.isArray(paragraphs) || !paragraphs.some(p => p.trim())) {
            throw new Error('原文件中没有可直接读取的文字，请在「扫描」中识别后再送来翻译。');
          }
          next = { format, document: data, paragraphs, digest };
          content = paragraphs.join('\n\n');
        } else {
          if (restoreOriginal) throw new Error('请选择此任务原来的 PDF 或 Word 文件。');
          if (file.size > 8 * 1024 * 1024) throw new Error('文本文件请控制在 8 MB 内。');
          content = await file.text();
        }
        sourceFile = next;
        includeSource.checked = false;
        taskId = null;
        savedStatus.textContent = '';
        fileName = file.name;
        const extension = fileName.split('.').at(-1).toLowerCase();
        const defaultFormat = extension === 'markdown' ? 'md' : extension;
        if ([...exportFormat.options].some(option => option.value === defaultFormat)) {
          exportFormat.value = defaultFormat;
        }
        exportedParts = null;
        work = null;
        ta.value = content;
        clearFile.hidden = false;
        setOutput('');
        copyBtn.disabled = true;
        prog.textContent = file.name + ' 已读取' + (format === 'pdf' ? ' · 导出保留原页并附译文' : '');
      } catch (e) {
        prog.textContent = '读取失败 · ' + e.message;
      } finally {
        readingFile = false;
        updateControls();
      }
    };
    bar.appendChild(fileBtn);
    bar.appendChild(inputFile);
    bar.append(clearFile, relink, sample);

    const ocrSelect = el('select', 'vsel');
    ocrSelect.id = 'translationOCR';
    ocrSelect.setAttribute('aria-label', '截图识别引擎');
    for (const [value, title] of [['vision', '系统 OCR'], ['xiaomi', '小米 OCR']]) {
      const option = el('option', '', title);
      option.value = value;
      ocrSelect.appendChild(option);
    }
    ocrSelect.value = window.dubheOCRProvider();
    ocrSelect.onchange = () => setTranslationPrefs({ ocr: ocrSelect.value });

    // 框选后由桌面外壳使用所选识别引擎，再把识别到的原文交给当前翻译模型。
    const shotBtn = el('button', 'vbtn', '截图翻译');
    shotBtn.title = '全局快捷键 ⌘⌥⇧T';
    shotBtn.onclick = () => {
      const h = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.shell;
      if (!h) { prog.textContent = '这个能力只在桌面 App 里可用'; return; }
      shellPost({ type: 'ocr' });
    };
    const ocrTools = el('div', 'translation-ocr-tools');
    ocrTools.append(ocrSelect, shotBtn);
    bar.appendChild(ocrTools);

    // 划词翻译的权限引导。macOS 的辅助功能授权只能在系统设置里点，
    // 程序既不能代点、也不能假装能用 —— 所以给一条明确的路：
    // 检查状态 →（没开）→ 直接跳到那个面板。
    const axBar = el('div', 'tbar');
    const axBtn = el('button', 'vbtn', '检查划词权限');
    const axOpen = el('button', 'vbtn', '打开辅助功能设置');
    const screenOpen = el('button', 'vbtn', '打开屏幕录制设置');
    const readClipboard = el('button', 'vbtn', '翻译剪贴板图片');
    const axTip = el('span', 'vmeta',
      '⌃⇧T 划词翻译 · ⌘⌥⇧T 截图翻译');
    const shortcutStatus = el('span', 'vmeta');
    shortcutStatus.setAttribute('role', 'status');
    const post = (t) => {
      const h = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.shell;
      if (h) h.postMessage({ type: t });
    };
    window.dubheAxStatus = (ok) => {
      shortcutStatus.textContent = ok ? '可读取其他应用里的选中文字。'
        : '跨应用划词需要允许当前这份 Dubhe 使用辅助功能。';
    };
    window.dubheTranslationHotKeyStatus = (unavailable) => {
      if (unavailable.length) shortcutStatus.textContent = '无法注册全局快捷键 · ' + unavailable.join('、');
    };
    axBtn.onclick = () => post('axStatus');
    axOpen.onclick = () => post('openAx');
    screenOpen.onclick = () => post('openScreenCapture');
    readClipboard.onclick = () => post('translateClipboardImage');
    axBar.appendChild(axBtn);
    axBar.appendChild(axOpen);
    axBar.appendChild(screenOpen);
    axBar.appendChild(readClipboard);
    axBar.appendChild(axTip);
    axBar.appendChild(shortcutStatus);
    const permissions = el('details', 'translation-permissions');
    permissions.ontoggle = () => { if (permissions.open) post('translationHotKeyStatus'); };
    permissions.append(el('summary', '', '划词与截图权限'), axBar);

    const go = el('button', 'vbtn pri', '翻译');
    bar.appendChild(go);
    const prog = el('span', 'vmeta', '');
    prog.setAttribute('role', 'status');
    bar.appendChild(prog);
    const savedStatus = el('span', 'vmeta task-save-status', '');
    savedStatus.setAttribute('role', 'status');
    bar.appendChild(savedStatus);
    const downloadModel = el('button', 'vbtn', '下载翻译模型');
    downloadModel.onclick = () => openView('models');
    bar.appendChild(downloadModel);
    const stop = el('button', 'vbtn', '停止翻译');
    stop.hidden = true;
    stop.onclick = () => job?.abort();
    bar.appendChild(stop);
    body.appendChild(bar);
    // 结果
    const outBox = el('pre', 'translation-raw');
    const preview = el('div', 'translation-preview');
    const empty = el('p', 'translation-empty', '译文会显示在这里');
    const outCard = el('section', 'vcard translation-card');
    const resultHeader = el('div', 'translation-result-head');
    resultHeader.appendChild(el('h3', '', '译文'));
    const resultBody = el('div', 'translation-result-body');
    resultBody.append(outBox, preview, empty);
    outCard.append(resultHeader, resultBody);
    const resultActions = el('div', 'translation-result-actions');
    const copyBtn = el('button', 'vbtn', '复制译文');
    copyBtn.disabled = true;
    copyBtn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(outBox.textContent || '');
        copyBtn.textContent = '已复制';
        setTimeout(() => (copyBtn.textContent = '复制译文'), 1500);
      } catch (e) { prog.textContent = '复制失败，可以选择译文后手动复制。'; }
    };
    resultActions.appendChild(copyBtn);
    const exportGroup = el('div', 'translation-export');
    const exportFormat = el('select', 'vsel');
    exportFormat.setAttribute('aria-label', '导出格式');
    for (const [value, label] of [['md', 'Markdown'], ['txt', 'TXT'], ['pdf', 'PDF'], ['docx', 'Word']]) {
      const option = el('option', '', label);
      option.value = value;
      exportFormat.appendChild(option);
    }
    const saveDocument = el('button', 'vbtn', '导出');
    const includeSource = el('input');
    includeSource.type = 'checkbox';
    const bilingual = el('label', 'translation-bilingual');
    bilingual.append(includeSource, document.createTextNode('附原文'));
    exportGroup.append(bilingual, exportFormat, saveDocument);
    resultActions.appendChild(exportGroup);
    outCard.appendChild(resultActions);
    workspace.appendChild(outCard);
    body.append(workspace, terms, permissions);

    function setOutput(text, ready = false) {
      outBox.textContent = text;
      outputReady = ready && !!text.trim();
      outBox.hidden = outputReady || !text;
      empty.hidden = !!text;
      preview.hidden = !outputReady;
      preview.innerHTML = outputReady ? renderMarkdown(text, { codeActions: false, tables: true }) : '';
    }

    saveDocument.onclick = async () => {
      const format = exportFormat.value;
      const name = (fileName ? fileName.replace(/\.[^.]+$/, '') + '-译文' : '译文') + '.' + format;
      let text = outBox.textContent;
      let blocks;
      if (includeSource.checked) {
        const originals = sourceFile?.paragraphs ?? [ta.value];
        const results = exportedParts ?? [text];
        const sections = originals.map((original, index) => {
          const page = sourceFile?.pages?.[index];
          const heading = page ? '第 ' + page + ' 页' : originals.length > 1 ? '第 ' + (index + 1) + ' 节' : '';
          return { heading, original, translation: results[index] || '' };
        });
        text = sections.map(({ heading, original, translation }) =>
          (heading ? '## ' + heading + '\n\n' : '') + '原文\n\n' + original +
          '\n\n译文\n\n' + translation).join('\n\n');
        blocks = sections.flatMap(({ heading, original, translation }) => [
          ...(heading ? [{kind: 'h2', text: heading}] : []),
          {kind: 'h3', text: '原文'}, {kind: 'p', text: original},
          {kind: 'h3', text: '译文'}, {kind: 'p', text: translation},
        ]);
      }
      const previous = prog.textContent;
      exporting = true;
      updateControls();
      try {
        if (!window.webkit?.messageHandlers?.shell) {
          const type = format === 'md' ? 'text/markdown' : 'text/plain';
          const url = URL.createObjectURL(new Blob([text], { type: type + ';charset=utf-8' }));
          const link = el('a');
          link.href = url;
          link.download = name;
          link.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
          prog.textContent = '已导出译文文件';
          return;
        }
        prog.textContent = '正在导出…';
        let result;
        if (includeSource.checked && format === 'pdf' && sourceFile?.origin?.bookmark && exportedParts) {
          result = await nativeDocumentRequest({ action: 'scanBilingual', name,
            bookmark: sourceFile.origin.bookmark, identity: sourceFile.origin.identity,
            pages: sourceFile.pages, translations: exportedParts });
        } else if (!includeSource.checked && format === sourceFile?.format && exportedParts) {
          if (!sourceFile.document) throw new Error('请选择原文件以保留其版式，或勾选「附原文」导出双语文本。');
          const { document: translated } = await nativeDocumentRequest({
            format, document: sourceFile.document, translations: exportedParts,
          });
          if (!translated) throw new Error('没有生成译文文件。');
          result = await nativeDocumentRequest({ action: 'save', format, document: translated, name });
        } else {
          const formatted = format === 'pdf' || format === 'docx';
          result = await nativeDocumentRequest({
            action: 'export', format, name, text, ...(formatted ? { blocks: blocks ?? translationBlocks(preview) } : {}),
          });
        }
        prog.textContent = result.saved ? '已保存译文文件' : previous;
      } catch (e) {
        prog.textContent = '保存失败 · ' + e.message;
      } finally {
        exporting = false;
        updateControls();
      }
    };

    const settings = () => ({
      model: selM.value, source: selS.value, target: selL.value, glossary: glossary.value,
    });
    const canResume = () => work && !work.complete && (work.translated.length || work.segment?.finished) &&
      work.file === sourceFile && work.text === ta.value &&
      Object.entries(settings()).every(([key, value]) => value === work.prefs[key]);

    function updateControls() {
      const installed = catalog?.models.some(m => m.id === selM.value && m.installed);
      modelStatus.textContent = modelError || (!catalog ? '正在读取模型列表…'
        : !installed ? '请在模型商店下载此模型。'
        : !catalog.runtime_available ? '本机未找到翻译运行时。' : '已下载');
      go.disabled = Boolean(job || readingFile || exporting || !installed || !catalog?.runtime_available);
      go.textContent = !job && canResume() ? '继续翻译' : '翻译';
      stop.hidden = !job;
      downloadModel.hidden = !catalog || Boolean(installed);
      clearFile.hidden = !fileName && !work && !ta.value;
      relink.hidden = !sourceFile || !['pdf', 'docx'].includes(sourceFile.format) || Boolean(sourceFile.document);
      relink.disabled = sample.disabled = Boolean(job || readingFile || exporting);
      selM.disabled = Boolean(job || !catalog);
      selS.disabled = selL.disabled = glossary.disabled = Boolean(job);
      presetName.disabled = presetSelect.disabled = savePreset.disabled = Boolean(job);
      loadPreset.disabled = removePreset.disabled = Boolean(job || !presetSelect.value);
      fileBtn.disabled = clearFile.disabled = Boolean(job || readingFile || exporting);
      ta.readOnly = Boolean(job || sourceFile || readingFile);
      const native = Boolean(window.webkit?.messageHandlers?.shell);
      ocrSelect.hidden = !native;
      for (const option of exportFormat.options) {
        option.disabled = !native && (option.value === 'pdf' || option.value === 'docx');
      }
      if (exportFormat.selectedOptions[0]?.disabled) exportFormat.value = 'md';
      exportFormat.disabled = Boolean(job || exporting || readingFile);
      saveDocument.disabled = Boolean(job || exporting || readingFile || !outputReady);
    }

    function renderModels() {
      const selected = job ? selM.value : translationPrefs.model;
      selM.replaceChildren(...catalog.models.map(m => {
        const option = el('option', '', m.name + (m.installed ? '' : '（待下载）'));
        option.value = m.id;
        return option;
      }));
      selM.value = catalog.models.some(m => m.id === selected) ? selected : catalog.default_model;
    }

    async function refreshModels() {
      const request = ++catalogRequest;
      try {
        const data = await vapi('/api/translate');
        if (request !== catalogRequest) return;
        catalog = { ...data, models: data.models || [] };
        modelError = '';
        renderModels();
      } catch (e) {
        if (request !== catalogRequest) return;
        modelError = '读不到翻译模型：' + e.message;
      }
      updateControls();
    }

    selM.onchange = () => {
      setTranslationPrefs({ model: selM.value });
      updateControls();
    };

    async function checkpoint() {
      if (!work || !taskId) return;
      const { file, prefs, translated, segment, complete } = work;
      const source = file ? {
        format: file.format, paragraphs: file.paragraphs,
        digest: file.digest || '', pages: file.pages || null, origin: file.origin || null,
      } : null;
      await saveLocalTask('translation', taskId, fileName || '文本翻译', {
        fileName, source, text: file ? '' : work.text, prefs, translated, complete: !!complete,
        bilingual: includeSource.checked,
        segment: segment ? { next: segment.next, finished: segment.finished, result: segment.result } : null,
      }, savedStatus);
    }

    function paragraphSource(paragraph, pdf) {
      return pdf ? paragraph.split(/\r?\n[ \t]*\r?\n/)
        .map(block => block.replace(/[ \t]*\r?\n[ \t]*/g, ' ')).join('\n\n') : paragraph;
    }

    function restoreTask(task) {
      if (job || readingFile || exporting) {
        prog.textContent = '请先完成或停止当前操作。';
        return;
      }
      const data = task?.payload;
      if (!data || !data.prefs || !Array.isArray(data.translated)) {
        prog.textContent = '找不到这个翻译任务的记录。';
        return;
      }
      taskId = task.id;
      fileName = data.fileName || '';
      sourceFile = data.source;
      ta.value = sourceFile?.paragraphs.join('\n\n') ?? data.text;
      const restored = data.segment ? prepareTranslation(paragraphSource(
        (sourceFile?.paragraphs ?? [ta.value])[data.translated.length] ?? '',
        sourceFile?.format === 'pdf',
      )) : null;
      if (restored) Object.assign(restored, data.segment);
      work = { text: ta.value, file: sourceFile, prefs: data.prefs, translated: data.translated,
        complete: data.complete, segment: restored };
      setTranslationPrefs(data.prefs);
      selM.value = data.prefs.model;
      selS.value = data.prefs.source;
      selL.value = data.prefs.target;
      glossary.value = data.prefs.glossary;
      includeSource.checked = data.bilingual || false;
      exportedParts = work.complete ? data.translated : null;
      const partial = [...data.translated, ...(restored?.result ? [restored.result] : [])].join('\n\n');
      setOutput(partial, work.complete);
      copyBtn.disabled = !partial;
      savedStatus.textContent = '已从本机恢复';
      prog.textContent = work.complete ? '已恢复完成的译文'
        : partial ? '已恢复已完成的译文，可继续翻译。' : '已恢复待翻译的原文。';
      updateControls();
    }

    go.onclick = async () => {
      const text = ta.value;
      if (!text.trim()) { prog.textContent = '先输入要翻译的文字。'; return; }
      const resume = canResume();
      if (!resume) {
        work = { text, file: sourceFile, prefs: settings(), translated: [], segment: null, complete: false };
        taskId = crypto.randomUUID();
        setOutput('');
      }
      job = new AbortController();
      const { signal } = job;
      updateControls();
      copyBtn.disabled = true;
      exportedParts = null;
      prog.textContent = resume ? '正在继续翻译…' : '准备翻译…';
      try {
        await checkpoint();
        const { translated, prefs } = work;
        const paragraphs = sourceFile?.paragraphs ?? [text];
        const pdf = sourceFile?.format === 'pdf';
        const unit = pdf ? '页' : '段';
        let complete = translated.join('\n\n');
        let segmentCount = 0;
        for (let i = translated.length; i < paragraphs.length; i++) {
          const paragraph = paragraphs[i];
          const source = paragraphSource(paragraph, pdf);
          const position = sourceFile?.pages?.[i] ? '原第 ' + sourceFile.pages[i] + ' 页'
            : sourceFile ? (i + 1) + ' / ' + paragraphs.length + ' ' + unit : '';
          const prefix = i ? complete + '\n\n' : '';
          const segment = work.segment ??= prepareTranslation(source);
          await translateDocument(segment, prefs, signal,
            status => { prog.textContent = position || status; },
            partial => { setOutput(prefix + partial); }, checkpoint);
          translated.push(segment.result);
          complete = prefix + segment.result;
          segmentCount = segment.total;
          work.segment = null;
          await checkpoint();
        }
        if (sourceFile) {
          exportedParts = translated;
          prog.textContent = '完成 · ' + paragraphs.length + ' ' + unit;
        } else {
          prog.textContent = segmentCount ? '完成 · ' + segmentCount + ' 段' : '已原样保留代码';
        }
        setOutput(complete, true);
        work.complete = true;
        await checkpoint();
      } catch (e) {
        const partial = outBox.textContent ? '（已保留完成的译文）' : '';
        prog.textContent = signal.aborted ? '已停止' + partial : '翻译失败' + partial + ' · ' + e.message;
      } finally {
        job = null;
        if (catalog) renderModels();
        updateControls();
        copyBtn.disabled = !outBox.textContent;
      }
    };
    setOutput('');
    updateControls();
    function acceptScan(text, name, pages, origin) {
      if (job || readingFile || exporting) {
        prog.textContent = '请等当前操作完成后再送入扫描结果。';
        return;
      }
      exportedParts = work = null;
      sourceFile = pages ? { format: 'scan', paragraphs: pages.map(item => item[1].text),
        pages: pages.map(item => item[0]), origin } : null;
      taskId = null;
      includeSource.checked = Boolean(pages);
      savedStatus.textContent = '';
      fileName = name;
      ta.value = text;
      clearFile.hidden = false;
      setOutput('');
      copyBtn.disabled = true;
      prog.textContent = '已载入扫描结果';
      updateControls();
      ta.focus();
    }
    return { body, refreshModels, acceptScan, restoreTask, usesTask: id => id === taskId && !!job };
  }

  // ---------------------------------------------------------------- 扫描
  const SCAN_MODES = [
    ['document', '论文与文档', '按阅读顺序识别正文、标题、表格和公式。'],
    ['text', '文字与手写', '提取选中图片或页面上的文字，适合手写报告。'],
    ['table', '表格', '识别表格行列与合并单元格。'],
    ['formula', '公式', '将图片中的公式转成 LaTeX。'],
    ['extract', '提取字段', '按指定的字段提取信息，输出 JSON。'],
    ['ask', '图片问答', '根据当前图片或选中页面回答问题。'],
  ];
  const SCAN_FORMATS = { markdown: 'md', text: 'txt', latex: 'tex', json: 'json' };
  let scanPane;

  function viewScan() {
    setViewHead('扫描', '', [{ label: '最近任务', on: () => openView('tasks') },
      { label: '模型商店', on: () => openView('models') }]);
    scanPane ??= createScanPane();
    $('#viewBody').appendChild(scanPane.body);
    scanPane.refreshModels();
  }

  async function scanImageData(file) {
    if (file.size > 32 * 1024 * 1024) throw new Error('图片超过 32 MB，请选择更小的文件。');
    if (!/\.(png|jpe?g|webp|heic|heif)$/i.test(file.name)) {
      throw new Error('请选择 JPG、PNG、WebP 或 HEIC 图片。');
    }
    const image = el('img');
    const url = URL.createObjectURL(file);
    try {
      image.src = url;
      await image.decode().catch(() => { throw new Error('读取不到图片。可以先转换为 JPG 或 PNG。'); });
      const canvas = el('canvas');
      const scale = Math.min(1, 2400 / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const ctx = canvas.getContext('2d', { alpha: false });
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.91);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function appendScanMarkdown(host, text) {
    const appendMarkdown = value => {
      const part = el('div');
      part.innerHTML = renderMarkdown(value, { codeActions: false, tables: true });
      host.append(...part.childNodes);
    };
    let offset = 0;
    for (const match of text.matchAll(/<table\b[^>]*>[\s\S]*?<\/table>/gi)) {
      appendMarkdown(text.slice(offset, match.index));
      const inert = el('template');
      inert.innerHTML = match[0];
      const table = el('table');
      const source = inert.content.querySelector('table');
      for (const row of source?.rows || []) {
        const target = table.insertRow();
        for (const cell of row.cells) {
          const item = el(cell.localName === 'th' ? 'th' : 'td', '', cell.textContent);
          item.colSpan = Math.max(1, Math.min(100, cell.colSpan));
          item.rowSpan = Math.max(1, Math.min(100, cell.rowSpan));
          target.appendChild(item);
        }
      }
      if (table.rows.length) host.appendChild(table);
      else appendMarkdown(match[0]);
      offset = match.index + match[0].length;
    }
    appendMarkdown(text.slice(offset));
  }

  function createScanPane() {
    const body = el('div', 'scan-pane');
    const native = () => window.webkit?.messageHandlers?.shell;
    const button = (id, title, classes = '') => {
      const control = el('button', 'vbtn' + (classes ? ' ' + classes : ''), title);
      control.id = 'scan' + id;
      return control;
    };
    const status = el('p', 'scan-status');
    status.id = 'scanStatus';
    status.setAttribute('role', 'status');
    const setStatus = (value, bad = false) => {
      status.classList.toggle('bad', bad);
      status.textContent = value;
    };

    let catalog, catalogError = '', modelError = '', modelBusy = false;
    let refreshCounter = 0, poll, files = [];
    let source, page = 1, raster, lastPdfImage, drawnPage = 0, loadingImage;
    let lastPdfText = '';
    let pendingPDF, pendingSave, work, job, taskId, pdfOpened, pendingBatch, batch;
    let relinkingPDF = false, relinkingImage = false;
    const pendingPages = new Map();

    const modelBar = el('div', 'scan-model vcard');
    const modelIcon = el('span', 'scan-model-icon');
    modelIcon.appendChild(iconNode('scan'));
    const modelText = el('div', 'vmain');
    const modelStatus = el('p', 'scan-model-status');
    modelStatus.setAttribute('role', 'status');
    modelText.append(el('div', 'vname', 'Xiaomi-OCR-0'), modelStatus);
    const install = button('Install', '下载扫描模型', 'pri');
    const remove = button('Remove', '移除模型', 'ghost');
    const recheck = button('Refresh', '重新检查');
    modelBar.append(modelIcon, modelText, recheck, remove, install);

    const workspace = el('div', 'scan-workspace');
    const inputCard = el('section', 'scan-card vcard');
    const resultCard = el('section', 'scan-card vcard');
    const sourceHead = el('header', 'scan-card-head');
    const sourceTitle = el('div', 'scan-title');
    const sourceName = el('p', 'scan-source-name');
    sourceTitle.append(el('h3', '', '原件'), sourceName);
    const sourceTools = el('div', 'scan-tools');
    const openImage = button('OpenImage', '打开图片');
    const openPDF = button('OpenPDF', '打开 PDF');
    const batchOpen = button('BatchOpen', '批量 PDF');
    batchOpen.setAttribute('aria-label', '批量扫描 PDF');
    const clear = button('Clear', '移除原件', 'ghost');
    const sample = button('Sample', '试用示例', 'ghost');
    const relink = button('Relink', '重新选择原件');
    sourceTools.append(openImage, openPDF, batchOpen, sample, relink, clear);
    sourceHead.append(sourceTitle, sourceTools);
    const upload = el('input');
    upload.id = 'scanImage';
    upload.type = 'file';
    upload.accept = '.png,.jpg,.jpeg,.webp,.heic,.heif';
    upload.hidden = true;
    const imageFrame = el('div', 'scan-image-frame');
    const image = el('img');
    image.alt = '待识别页面';
    const blank = el('div', 'scan-blank');
    blank.append(iconNode('upload-image'), el('b', '', '打开或拖入图片'));
    const opening = el('div', 'scan-blank', '正在准备预览…');
    const enlarge = button('Enlarge', '放大原件');
    enlarge.classList.add('scan-enlarge');
    imageFrame.append(blank, opening, image, enlarge);
    const pagination = el('div', 'scan-paging');
    const prev = button('Previous', '上一页', 'ghost');
    const next = button('Next', '下一页', 'ghost');
    const position = el('output', 'scan-page-number');
    position.id = 'scanPage';
    pagination.append(prev, position, next);
    inputCard.append(sourceHead, upload, imageFrame, pagination);

    const options = el('div', 'scan-options');
    const modeLine = el('label', 'scan-control');
    const mode = el('select', 'vsel');
    mode.id = 'scanMode';
    for (const [value, title] of SCAN_MODES) {
      const option = el('option', '', title);
      option.value = value;
      mode.appendChild(option);
    }
    modeLine.append(el('span', '', '识别方式'), mode);
    const nativeText = el('input');
    nativeText.type = 'checkbox';
    nativeText.checked = true;
    const nativeTextLabel = el('label', 'translation-bilingual');
    nativeTextLabel.append(nativeText, document.createTextNode('优先读取 PDF 文字'));
    nativeText.onchange = () => update();
    const range = el('div', 'scan-range');
    function pageBound(title, id) {
      const label = el('label', 'scan-control');
      const field = el('input', 'vsel');
      field.type = 'number';
      field.min = '1';
      field.step = '1';
      field.id = 'scan' + id;
      label.append(el('span', '', title), field);
      range.appendChild(label);
      return field;
    }
    const from = pageBound('从页', 'From');
    const to = pageBound('到页', 'To');
    const optionRow = el('div', 'scan-option-row');
    optionRow.append(modeLine, range);
    const modeTip = el('p', 'scan-mode-tip');
    const detail = el('label', 'scan-control');
    const detailTitle = el('span');
    const detailValue = el('textarea', 'field');
    detailValue.id = 'scanDetail';
    detailValue.rows = 2;
    detail.append(detailTitle, detailValue);
    const actionRow = el('div', 'scan-action-row');
    const run = button('Run', '开始识别', 'pri');
    const stop = button('Stop', '停止识别');
    const savedStatus = el('span', 'vmeta task-save-status');
    savedStatus.setAttribute('role', 'status');
    actionRow.append(run, stop, status, savedStatus);
    options.append(optionRow, nativeTextLabel, modeTip, detail, actionRow);
    inputCard.appendChild(options);

    const resultHead = el('header', 'scan-card-head');
    const resultTitle = el('div', 'scan-title');
    const resultPage = el('p', 'scan-source-name');
    resultTitle.append(el('h3', '', '识别结果'), resultPage);
    const showRaw = button('Raw', '查看原文', 'ghost');
    resultHead.append(resultTitle, showRaw);
    const output = el('div', 'scan-output');
    output.id = 'scanResult';
    const resultTools = el('div', 'scan-tools scan-result-tools');
    const copy = button('Copy', '复制');
    const exportResult = button('Export', '导出');
    const translate = button('Translate', '送去翻译');
    const review = button('Review', '校对本页');
    const question = button('AskDocument', '问这份文档');
    resultTools.append(review, copy, exportResult, translate, question);
    resultCard.append(resultHead, output, resultTools);
    workspace.append(inputCard, resultCard);
    body.append(modelBar, workspace);

    const isPDF = () => source?.kind === 'pdf';
    const originalReady = () => Boolean(source?.image || source?.requestId);
    const selection = () => ({
      mode: mode.value, first: isPDF() ? from.valueAsNumber : 1, last: isPDF() ? to.valueAsNumber : 1,
      preferText: isPDF() && mode.value === 'document' && nativeText.checked,
      detail: /^(extract|ask)$/.test(mode.value) ? detailValue.value.trim() : '',
    });
    const validRange = value => Number.isInteger(value.first) && Number.isInteger(value.last) &&
      value.first >= 1 && value.first <= value.last && value.last <= (source?.pageCount ?? 1);
    const canContinue = value => work && work.source === source &&
      work.results.size < value.last - value.first + 1 &&
      Object.keys(value).every(key => value[key] === work[key]);
    const result = () => work?.results.get(page);
    const active = () => files.filter(activeTransfer);
    let raw = false;

    function update() {
      const activeFiles = active();
      const installed = Boolean(catalog?.installed);
      const ready = installed && Boolean(catalog.runtime_available);
      const total = files.reduce((sum, file) => sum + (file.bytes || 0), 0);
      const downloaded = files.reduce((sum, file) => sum +
        (file.installed ? file.bytes : Math.min(file.got || 0, file.bytes)), 0);
      modelStatus.textContent = modelError || catalogError || (!catalog ? '正在检查扫描模型…'
        : !catalog.runtime_available ? '本机缺少扫描运行时，请使用带模型运行时的 Mac 安装包。'
        : installed ? '已安装 · 本地识别'
        : activeFiles.length ? (activeFiles.some(file => file.state === 'running') ? '下载中 · ' +
          fmtBytes(downloaded) + ' / ' + fmtBytes(total)
          : activeFiles.some(file => file.state === 'verifying') ? '正在核对模型文件…' : '正在取消下载…')
        : files.find(file => file.state === 'error')?.error || '待安装 · ' + fmtBytes(total));
      modelStatus.classList.toggle('bad', Boolean(modelError || catalogError ||
        files.some(file => file.state === 'error')));
      recheck.hidden = !catalogError;
      install.hidden = installed || !catalog?.runtime_available;
      install.textContent = activeFiles.length ? '取消下载'
        : files.some(file => file.state === 'error' || file.state === 'cancelled') ? '继续下载' : '下载扫描模型';
      install.disabled = modelBusy || Boolean(job || batch) || !files.length ||
        activeFiles.length > 0 && activeFiles.every(file => file.state === 'cancelling');
      remove.hidden = !files.some(file => file.installed && file.path) || activeFiles.length > 0;
      remove.disabled = modelBusy || Boolean(job || batch);
      openPDF.hidden = batchOpen.hidden = !native();
      clear.hidden = !source;
      const busy = Boolean(job || pendingPDF || loadingImage || pendingSave || pendingBatch || batch);
      openImage.disabled = openPDF.disabled = batchOpen.disabled = clear.disabled = sample.disabled = relink.disabled = busy;
      batchOpen.textContent = batch ? '批量 ' + batch.current + ' / ' + batch.total : '批量 PDF';
      relink.hidden = !source || originalReady() || isPDF() && !native();
      pagination.hidden = range.hidden = !isPDF();
      mode.disabled = from.disabled = to.disabled = detailValue.disabled = nativeText.disabled = Boolean(job || batch);
      nativeTextLabel.hidden = !isPDF() || mode.value !== 'document';
      sourceName.textContent = source?.name || '';
      sourceName.title = source?.name || '';
      position.textContent = isPDF() ? '第 ' + page + ' / ' + source.pageCount + ' 页' : '';
      prev.disabled = Boolean(job || batch) || page <= 1;
      next.disabled = Boolean(job || batch) || page >= (source?.pageCount ?? 1);
      blank.hidden = Boolean(source || loadingImage || pendingPDF);
      image.hidden = enlarge.hidden = !raster;
      opening.hidden = !blank.hidden || Boolean(raster) || drawnPage === page;
      const s = selection();
      modeTip.textContent = !validRange(s) ? '请输入有效页码范围。'
        : SCAN_MODES.find(item => item[0] === s.mode)[2];
      detail.hidden = s.mode !== 'extract' && s.mode !== 'ask';
      detailTitle.textContent = s.mode === 'extract' ? '字段名称' : '问题';
      detailValue.placeholder = s.mode === 'extract' ? '例如：题目、作者、日期' : '例如：图中的纵轴表示什么？';
      stop.hidden = !job && !batch;
      stop.disabled = !!job?.signal.aborted || !!batch?.stopped;
      run.disabled = busy || modelBusy || !originalReady() || (!ready && !s.preferText) || !validRange(s) || (!detail.hidden && !s.detail);
      run.textContent = canContinue(s) ? '继续识别' : '开始识别';
      const item = result();
      showRaw.hidden = item?.kind !== 'markdown';
      showRaw.textContent = raw ? '查看预览' : '查看原文';
      review.disabled = !item || busy;
      const hasResults = Boolean(work?.results.size);
      copy.disabled = exportResult.disabled = !hasResults || Boolean(pendingSave || batch);
      translate.hidden = !hasResults || !['document', 'text'].includes(work.mode);
      translate.disabled = Boolean(job || batch);
      question.hidden = translate.hidden;
      question.disabled = Boolean(job || batch);
      exportResult.textContent = isPDF() && hasResults ? '导出已识别页' : '导出';
    }

    function renderResult() {
      const item = result();
      resultPage.textContent = isPDF() ? '第 ' + page + ' 页' : '';
      output.replaceChildren();
      if (!item) {
        output.appendChild(el('p', 'scan-result-empty', work?.results.size
          ? '这一页还没有识别结果。' : '识别结果会显示在这里。'));
      } else if (item.kind === 'markdown' && !raw) {
        const preview = el('div', 'translation-preview');
        appendScanMarkdown(preview, item.text);
        output.appendChild(preview);
      } else {
        output.appendChild(el('pre', 'scan-result-raw', item.text));
      }
      output.scrollTop = 0;
      update();
    }

    function replaceSource(nextSource) {
      if (source?.requestId) shellPost({ type: 'scanClosePDF', requestId: source.requestId });
      source = nextSource;
      page = 1;
      drawnPage = 0;
      raster = null;
      lastPdfImage = null;
      lastPdfText = '';
      work = null;
      taskId = null;
      savedStatus.textContent = '';
      raw = false;
      image.removeAttribute('src');
      from.max = to.max = String(source?.pageCount ?? 1);
      from.value = '1';
      to.value = to.max;
      setStatus('');
      showPage(1);
    }

    function pageImage(file, index) {
      if (file.kind === 'image') return Promise.resolve(file.image);
      if (lastPdfImage?.file === file && lastPdfImage.page === index) {
        return Promise.resolve(lastPdfImage.image);
      }
      const key = file.requestId + ':' + index;
      const existing = pendingPages.get(key);
      if (existing) return existing.promise;
      let finish;
      const promise = new Promise((resolve, reject) => {
        finish = (data, error, text) => {
          pendingPages.delete(key);
          clearTimeout(timeout);
          if (error) reject(error);
          else {
            if (file === source) {
              lastPdfImage = { file, page: index, image: data };
              lastPdfText = text || '';
            }
            resolve(data);
          }
        };
      });
      const timeout = setTimeout(() => finish(null, new Error('读取 PDF 页超时。')), 30000);
      pendingPages.set(key, { promise, finish });
      shellPost({ type: 'scanPDFPage', requestId: file.requestId, page: index });
      return promise;
    }

    function showRaster(file, index, data) {
      if (file !== source || index !== page) return;
      raster = data;
      image.src = raster;
      drawnPage = index;
      update();
    }

    async function showPage(index) {
      page = index;
      drawnPage = 0;
      raster = null;
      image.removeAttribute('src');
      renderResult();
      if (!source) return;
      if (!originalReady()) {
        drawnPage = page;
        setStatus('已保留结果。重新选择原件后，可继续识别或框选重扫。');
        update();
        return;
      }
      const file = source;
      try {
        showRaster(file, index, await pageImage(file, index));
      } catch (e) {
        if (file !== source || index !== page) return;
        drawnPage = index;
        setStatus('预览失败 · ' + e.message, true);
        update();
      }
    }

    function scanPageForJob(file, index, signal) {
      return new Promise((resolve, reject) => {
        const abort = () => reject(signal.reason);
        if (signal.aborted) { abort(); return; }
        signal.addEventListener('abort', abort, { once: true });
        pageImage(file, index).then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
      });
    }

    window.dubheScanPDFReady = (message) => {
      if (message.requestId !== pendingPDF) return;
      pendingPDF = null;
      const reconnect = relinkingPDF;
      relinkingPDF = false;
      let opened = false;
      if (!message.cancelled && !message.error && Number.isInteger(message.pageCount) && message.pageCount > 0) {
        const file = { kind: 'pdf', requestId: message.requestId, name: message.name, pageCount: message.pageCount,
          bookmark: message.bookmark || '', identity: message.identity || '' };
        if (reconnect) opened = reattach(file);
        else { replaceSource(file); opened = true; }
      } else if (!message.cancelled) {
        setStatus(message.error || 'PDF 没有可识别页面。', true);
      }
      pdfOpened?.(opened);
      pdfOpened = null;
      update();
    };
    window.dubheScanPDFPage = (message) => {
      const pending = pendingPages.get(message.requestId + ':' + message.page);
      if (!pending) return;
      const valid = typeof message.image === 'string' && /^data:image\/(?:jpeg|png);base64,/.test(message.image);
      pending.finish(message.image, valid && !message.error ? null : new Error(message.error || '读取 PDF 页失败。'),
        typeof message.selectableText === 'string' ? message.selectableText : '');
    };
    window.dubheScanSaved = (message) => {
      if (message.requestId !== pendingSave) return;
      pendingSave = null;
      if (message.error) setStatus(message.error, true);
      else if (message.saved) setStatus('已保存扫描结果。');
      update();
    };

    async function importImage(file, reconnect = false) {
      if (!file || job || batch || pendingBatch || pendingPDF || loadingImage || pendingSave) return;
      loadingImage = true;
      setStatus('');
      update();
      try {
        const image = await scanImageData(file);
        const nextSource = { kind: 'image', name: file.name, image, identity: await fingerprintFile(file) };
        if (reconnect) reattach(nextSource);
        else replaceSource(nextSource);
      } catch (e) {
        setStatus(e.message, true);
      } finally {
        loadingImage = false;
        update();
      }
    }

    openImage.onclick = () => { relinkingImage = false; upload.click(); };
    upload.onchange = () => {
      const file = upload.files?.[0];
      upload.value = '';
      importImage(file, relinkingImage);
      relinkingImage = false;
    };
    function openPDFPicker(reconnect = false, bookmark = '') {
      pendingPDF = crypto.randomUUID();
      relinkingPDF = reconnect;
      update();
      return new Promise(resolve => {
        pdfOpened = resolve;
        shellPost({ type: bookmark ? 'scanRestorePDF' : 'scanOpenPDF', requestId: pendingPDF, bookmark });
      });
    }
    openPDF.onclick = () => openPDFPicker();
    batchOpen.onclick = () => {
      pendingBatch = crypto.randomUUID();
      update();
      shellPost({ type: 'scanBatchOpen', requestId: pendingBatch });
    };
    window.dubheScanBatchReady = async message => {
      if (message.requestId !== pendingBatch) return;
      pendingBatch = null;
      const files = Array.isArray(message.files) ? message.files : [];
      if (message.cancelled || !files.length) {
        if (!message.cancelled) setStatus(message.error || '没有可打开的 PDF。', true);
        update();
        return;
      }
      batch = { current: 0, total: files.length, stopped: false };
      update();
      const tasks = files.map(file => ({ id: crypto.randomUUID(), kind: 'scan',
        title: file.name.slice(0, 100), payload: { source: file, mode: 'document',
          first: 1, last: file.pageCount, detail: '', results: [], page: 1, preferText: true } }));
      batch.ids = new Set(tasks.map(task => task.id));
      let done = 0;
      try {
        for (const task of tasks) await vpost('/api/tasks', { upsert: task });
        for (const task of tasks) {
          if (batch.stopped) break;
          batch.current++;
          update();
          const opened = await restoreTask(task, true);
          if (batch.stopped) break;
          if (!opened) continue;
          await run.onclick();
          if (work?.results.size === task.payload.last) done++;
        }
        const unavailable = Array.isArray(message.unavailable) ? message.unavailable.length : 0;
        setStatus('批量已完成 ' + done + ' / ' + tasks.length + ' 份' +
          (unavailable ? ' · ' + unavailable + ' 份无法打开' : '') +
          (done < tasks.length ? ' · 可在「最近任务」继续' : ' · 结果已存入「最近任务」'));
      } catch (error) {
        setStatus('批量中断 · ' + error.message + ' · 已写入的任务在「最近任务」', true);
      } finally {
        batch = null;
        update();
      }
    };
    relink.onclick = () => {
      if (isPDF()) openPDFPicker(true);
      else { relinkingImage = true; upload.click(); }
    };
    function examplePage() {
      const canvas = el('canvas');
      canvas.width = 1100;
      canvas.height = 740;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#f9f6ed';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#253b30';
      ctx.font = 'bold 48px sans-serif';
      ctx.fillText('Field notebook  ·  河流观测记录', 80, 130);
      ctx.font = '34px sans-serif';
      ['Station A / 07:30', 'Water level: 2.48 m', 'Water temperature: 19.6 °C',
        'A second reading was taken after the rain.'].forEach((line, index) => {
        ctx.fillText(line, 84, 236 + index * 86);
      });
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#92a099';
      for (let i = 0; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(80, 262 + i * 86);
        ctx.lineTo(1020, 262 + i * 86);
        ctx.stroke();
      }
      return canvas.toDataURL('image/jpeg', 0.93);
    }
    sample.onclick = () => replaceSource({
      kind: 'image', name: '试用示例 · 河流观测记录', example: true, image: examplePage(),
    });
    clear.onclick = () => replaceSource(null);
    enlarge.onclick = () => {
      const dialog = el('dialog', 'scan-image-dialog');
      const head = el('div', 'scan-card-head');
      const title = el('h3', '', source.name);
      const close = el('button', 'vbtn', '关闭');
      close.onclick = () => dialog.close();
      head.append(title, close);
      const full = el('img');
      full.src = raster;
      full.alt = isPDF() ? '原件第 ' + page + ' 页' : '原件预览';
      dialog.append(head, full);
      dialog.setAttribute('aria-label', '原件预览');
      dialog.addEventListener('close', () => dialog.remove(), { once: true });
      document.body.appendChild(dialog);
      dialog.showModal();
    };
    prev.onclick = () => showPage(page - 1);
    next.onclick = () => showPage(page + 1);
    from.oninput = to.oninput = detailValue.oninput = update;
    mode.onchange = () => { detailValue.value = ''; update(); };
    showRaw.onclick = () => { raw = !raw; renderResult(); };

    review.onclick = () => {
      const item = result();
      if (!item || job) return;
      const reviewedPage = page;
      const dialog = el('dialog', 'scan-review-dialog');
      const header = el('header', 'scan-card-head');
      const title = el('h3', '', isPDF() ? '校对第 ' + page + ' 页' : '校对识别结果');
      const close = el('button', 'vbtn', '关闭');
      close.onclick = () => dialog.close();
      header.append(title, close);
      const columns = el('div', 'scan-review-columns');
      const original = el('div', 'scan-review-original');
      const editor = el('div', 'scan-review-editor');
      const draft = el('textarea', 'field');
      draft.setAttribute('aria-label', '本页校对稿');
      draft.value = item.text;
      const actions = el('div', 'scan-tools');
      const save = el('button', 'vbtn pri', '保存本页');
      const status = el('p', 'vmeta');
      status.setAttribute('role', 'status');
      actions.append(save);
      editor.append(el('h4', '', '校对稿'), draft, actions, status);
      const controller = new AbortController();
      dialog.addEventListener('close', () => {
        controller.abort();
        dialog.remove();
        // Keep the keyboard user's place when editing finishes.
        review.focus();
      }, { once: true });

      if (raster) {
        const frame = el('div', 'scan-region-frame');
        const preview = el('img');
        preview.src = raster;
        preview.alt = '当前页原件。可拖动选择需要重扫的区域。';
        preview.draggable = false;
        const box = el('span', 'scan-region-box');
        box.hidden = true;
        const reScan = el('button', 'vbtn', '识别选区');
        reScan.disabled = true;
        let anchor, selection, activePointer;
        function point(event) {
          const bounds = preview.getBoundingClientRect();
          return {
            x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
            y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
          };
        }
        function updateBox(point) {
          const x = Math.min(anchor.x, point.x);
          const y = Math.min(anchor.y, point.y);
          const width = Math.abs(anchor.x - point.x);
          const height = Math.abs(anchor.y - point.y);
          selection = { x, y, width, height };
          Object.assign(box.style, { left: x * 100 + '%', top: y * 100 + '%',
            width: width * 100 + '%', height: height * 100 + '%' });
          box.hidden = false;
          reScan.disabled = !catalog?.installed || !catalog.runtime_available ||
            width * preview.naturalWidth < 12 || height * preview.naturalHeight < 12;
        }
        frame.onpointerdown = event => {
          if (event.button !== 0 || !preview.naturalWidth || activePointer !== undefined) return;
          activePointer = event.pointerId;
          frame.setPointerCapture(activePointer);
          anchor = point(event);
          updateBox(anchor);
        };
        frame.onpointermove = event => { if (event.pointerId === activePointer) updateBox(point(event)); };
        frame.onpointerup = event => {
          if (event.pointerId !== activePointer) return;
          updateBox(point(event));
          activePointer = undefined;
        };
        frame.onpointercancel = () => {
          activePointer = selection = undefined;
          box.hidden = true;
          reScan.disabled = true;
        };
        frame.append(preview, box);
        const regionResult = el('textarea', 'field scan-region-result');
        regionResult.placeholder = '所选区域的结果会显示在这里';
        regionResult.setAttribute('aria-label', '选区识别结果');
        const insert = el('button', 'vbtn ghost', '插入校对稿');
        insert.disabled = true;
        insert.onclick = () => {
          draft.setRangeText(regionResult.value, draft.selectionStart, draft.selectionEnd, 'end');
          draft.focus();
        };
        const regionMode = el('select', 'vsel');
        regionMode.setAttribute('aria-label', '选区识别方式');
        for (const [value, label] of [['text', '文字'], ['table', '表格'], ['formula', '公式']]) {
          const option = el('option', '', label);
          option.value = value;
          regionMode.appendChild(option);
        }
        if (['formula', 'table'].includes(work.mode)) regionMode.value = work.mode;
        reScan.onclick = async () => {
          if (!selection || reScan.disabled) return;
          const canvas = el('canvas');
          const { x, y, width, height } = selection;
          const sw = width * preview.naturalWidth;
          const sh = height * preview.naturalHeight;
          canvas.width = Math.max(1, Math.round(sw));
          canvas.height = Math.max(1, Math.round(sh));
          canvas.getContext('2d', { alpha: false }).drawImage(preview, x * preview.naturalWidth,
            y * preview.naturalHeight, sw, sh, 0, 0, canvas.width, canvas.height);
          reScan.disabled = true;
          frame.style.pointerEvents = 'none';
          insert.disabled = true;
          status.textContent = '正在识别所选区域…';
          try {
            const image = canvas.toDataURL('image/jpeg', 0.94);
            const reply = await vpost('/api/scan', { image, mode: regionMode.value }, controller.signal);
            if (!reply.text?.trim()) throw new Error('没有识别到文字。');
            regionResult.value = reply.text;
            insert.disabled = false;
            status.textContent = '选区已识别。核对后可插入或手动修改校对稿。';
          } catch (e) {
            if (!controller.signal.aborted) status.textContent = '选区识别失败 · ' + e.message;
          } finally {
            frame.style.pointerEvents = '';
            reScan.disabled = false;
          }
        };
        const regionActions = el('div', 'scan-tools');
        regionActions.append(regionMode, reScan, insert);
        original.append(el('h4', '', '原页'), frame, el('p', 'vmeta', '从原页拖出一个矩形，可识别这块区域。'),
          regionActions, regionResult);
      } else {
        original.append(vEmpty('原件尚未打开', '关闭校对后，重新选择原件可以对照原页。'));
      }
      save.onclick = async () => {
        const text = draft.value.trim();
        if (!text) { status.textContent = '校对稿不能留空。'; return; }
        if (item.kind === 'json') {
          try {
            const value = JSON.parse(text);
            if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error();
          } catch (e) {
            status.textContent = '请保留 JSON 对象格式。';
            return;
          }
        }
        save.disabled = true;
        close.disabled = true;
        work.results.set(reviewedPage, { ...item, text });
        await checkpoint();
        renderResult();
        dialog.close();
        setStatus('已更新本页校对稿。');
      };
      dialog.addEventListener('cancel', event => { if (save.disabled) event.preventDefault(); });
      columns.append(original, editor);
      dialog.append(header, columns);
      dialog.setAttribute('aria-label', title.textContent);
      document.body.appendChild(dialog);
      dialog.showModal();
    };

    async function refreshModels() {
      clearTimeout(poll);
      if (!body.isConnected || state.view !== 'scan') return;
      const request = ++refreshCounter;
      try {
        const nextCatalog = await vapi('/api/scan');
        const store = nextCatalog.runtime_available && !nextCatalog.installed ? await vapi('/api/store') : {};
        if (request !== refreshCounter) return;
        catalog = nextCatalog;
        files = (catalog.files || []).map(file => ({
          ...file, ...(store.items || []).find(item => item.id === file.id),
        }));
        catalogError = '';
      } catch (e) {
        if (request !== refreshCounter) return;
        catalogError = '检查扫描模型失败 · ' + e.message;
      }
      update();
      if (active().length) poll = setTimeout(refreshModels, 1200);
    }

    install.onclick = async () => {
      const transfers = active();
      const endpoint = transfers.length ? '/api/store/cancel' : '/api/store/download';
      const targets = transfers.length ? transfers.filter(file => file.state !== 'cancelling')
        : files.filter(file => !file.installed);
      modelError = '';
      modelBusy = true;
      update();
      const results = await Promise.allSettled(targets.map(file => vpost(endpoint, { id: file.id })));
      const failed = results.find(result => result.status === 'rejected');
      if (failed) modelError = '下载操作失败 · ' + failed.reason.message;
      await refreshModels();
      modelBusy = false;
      update();
    };
    remove.onclick = async () => {
      modelBusy = true;
      update();
      try {
        if (!await confirmModelDeletion('已下载的 Xiaomi-OCR-0 组件')) return;
        modelError = '';
        for (const file of files.filter(item => item.installed && item.path)) {
          await vpost('/api/models/delete', { path: file.path });
        }
      } catch (e) {
        modelError = '删除失败 · ' + e.message;
      } finally {
        await refreshModels();
        modelBusy = false;
        update();
      }
    };
    recheck.onclick = () => { modelError = ''; refreshModels(); };

    async function checkpoint() {
      if (!work || !taskId || !source) return;
      const { kind, name, pageCount, bookmark, identity, example } = source;
      await saveLocalTask('scan', taskId, name, {
        source: { kind, name, pageCount: pageCount ?? 1, bookmark: bookmark || '',
          identity: identity || '', example: !!example },
        mode: work.mode, first: work.first, last: work.last, detail: work.detail, preferText: work.preferText,
        results: [...work.results.entries()], page,
      }, savedStatus);
    }

    function reattach(file) {
      if (!source || file.kind !== source.kind ||
          file.identity !== source.identity || (file.pageCount ?? 1) !== (source.pageCount ?? 1)) {
        if (file.requestId) shellPost({ type: 'scanClosePDF', requestId: file.requestId });
        setStatus('这份原件与已保存的任务不同。请重新选择。', true);
        return false;
      }
      source = file;
      if (work) work.source = file;
      lastPdfImage = null;
      setStatus('已重新打开原件');
      showPage(page);
      return true;
    }

    function restoreTask(task, fromBatch = false) {
      if (job || pendingPDF || pendingSave || loadingImage || ((batch || pendingBatch) && !fromBatch)) {
        setStatus('请先完成或停止当前操作。');
        return;
      }
      const data = task?.payload;
      if (!data?.source || !Array.isArray(data.results)) {
        setStatus('找不到这个扫描任务的记录。', true);
        return;
      }
      const file = { ...data.source };
      if (file.example) file.image = examplePage();
      replaceSource(file);
      mode.value = data.mode;
      nativeText.checked = data.preferText ?? true;
      detailValue.value = data.detail;
      from.value = data.first;
      to.value = data.last;
      taskId = task.id;
      work = { source: file, mode: data.mode, first: data.first, last: data.last,
        detail: data.detail, preferText: data.preferText ?? false, results: new Map(data.results) };
      savedStatus.textContent = '已从本机恢复';
      showPage(Math.min(file.pageCount, Math.max(1, data.page || 1)));
      if (isPDF() && native() && file.bookmark) return openPDFPicker(true, file.bookmark);
      return originalReady();
    }

    stop.onclick = () => {
      if (batch) batch.stopped = true;
      job?.abort();
      update();
    };
    run.onclick = async () => {
      const s = selection();
      if (!validRange(s) || !originalReady() || (!s.preferText && !catalog?.installed)) return;
      const fields = s.mode === 'extract'
        ? [...new Set(s.detail.split(/[,，、\n]/).map(field => field.trim()).filter(Boolean))] : [];
      const bytes = value => new TextEncoder().encode(value).length;
      if (s.mode === 'extract' && (!fields.length || fields.length > 20 || fields.some(field => bytes(field) > 96))) {
        setStatus('请输入 1–20 个较短的字段名称，用逗号或换行分隔。', true);
        return;
      }
      if (s.mode === 'ask' && bytes(s.detail) > 2048) {
        setStatus('问题过长，请缩短后重试。', true);
        return;
      }
      if (!canContinue(s)) {
        work = { ...s, source, results: new Map() };
        taskId = crypto.randomUUID();
      }
      job = new AbortController();
      const signal = job.signal;
      let current = s.first;
      try {
        await checkpoint();
        for (; current <= s.last; current++) {
          if (work.results.has(current)) continue;
          page = current;
          raster = null;
          drawnPage = 0;
          image.removeAttribute('src');
          renderResult();
          setStatus(isPDF() ? '正在识别第 ' + current + ' 页 · ' +
            work.results.size + ' / ' + (s.last - s.first + 1) + ' 页完成' : '正在识别…');
          const data = await scanPageForJob(source, current, signal);
          signal.throwIfAborted();
          showRaster(source, current, data);
          const payload = { mode: s.mode, image: data };
          if (s.mode === 'extract') payload.fields = fields;
          if (s.mode === 'ask') payload.question = s.detail;
          // Ordinary text PDFs need no OCR inference. Keep scanned pages and PDFs with only
          // a few selectable marks on the image recognition path.
          const selectable = s.preferText && lastPdfImage?.page === current && lastPdfText;
          const hasText = selectable && (selectable.match(/[\p{L}\p{N}]/gu)?.length ?? 0) >= 20;
          if (!hasText && (!catalog?.installed || !catalog.runtime_available)) {
            throw new Error('本页是扫描图片。下载扫描模型后可从此页继续。');
          }
          const item = hasText ? { kind: 'markdown', text: selectable } : await vpost('/api/scan', payload, signal);
          signal.throwIfAborted();
          if (typeof item.text !== 'string' || !item.text.trim() || !Object.hasOwn(SCAN_FORMATS, item.kind)) {
            throw new Error('没有取得有效的识别结果。');
          }
          work.results.set(current, item);
          renderResult();
          await checkpoint();
        }
        setStatus(isPDF() ? '识别完成 · ' + work.results.size + ' 页' : '识别完成');
      } catch (e) {
        const kept = work.results.size ? ' · 已保留 ' + work.results.size + ' 页结果' : '';
        if (signal.aborted) setStatus('已停止识别' + kept);
        else setStatus((isPDF() ? '第 ' + current + ' 页识别失败' : '识别失败') + ' · ' + e.message + kept, true);
      } finally {
        job = null;
        update();
      }
    };

    function allResults() {
      const results = [...work.results.entries()].sort((a, b) => a[0] - b[0]);
      const firstKind = results[0][1].kind;
      const format = SCAN_FORMATS[firstKind];
      let text = results[0][1].text;
      if (isPDF() && firstKind === 'json') {
        text = JSON.stringify(results.map(([page, item]) => ({ page, fields: JSON.parse(item.text) })), null, 2);
      } else if (isPDF() && results.length > 1) {
        text = results.map(([page, item]) =>
          (firstKind === 'markdown' ? '## ' : firstKind === 'latex' ? '% ' : '') +
          '第 ' + page + ' 页\n\n' + item.text
        ).join('\n\n');
      }
      const name = source.name.replace(/\.[^.]+$/, '') + '-扫描结果';
      return { format, name, text, results };
    }

    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(allResults().text);
        setStatus('已复制扫描结果。');
      } catch (e) {
        setStatus('复制失败，可选中识别结果手动复制。', true);
      }
    };
    exportResult.onclick = () => {
      try {
        const { format, name, text } = allResults();
        const filename = name + '.' + format;
        if (native()) {
          pendingSave = crypto.randomUUID();
          update();
          shellPost({ type: 'scanSave', requestId: pendingSave, format, name: filename, text });
        } else {
          const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
          const link = el('a');
          link.href = url;
          link.download = filename;
          link.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
          setStatus('已导出扫描结果。');
        }
      } catch (e) {
        setStatus('导出失败 · ' + e.message, true);
      }
    };
    translate.onclick = () => {
      const { name, text, results } = allResults();
      openView('translate');
      const origin = isPDF() && source.bookmark ? { bookmark: source.bookmark, identity: source.identity } : null;
      translationPane.acceptScan(text, name, results, origin);
    };

    question.onclick = async () => {
      if (!state.server?.loaded) {
        setStatus('请先在模型页加载对话模型。');
        return;
      }
      const { name, results } = allResults();
      const context = state.server?.info?.n_ctx || 4096;
      const budget = Math.max(800, Math.min(12_000, Math.floor(context / 2)));
      const limit = Math.max(120, Math.floor(budget / results.length) - 48);
      let excerpt = '';
      let included = 0;
      for (const [number, item] of results) {
        const section = '[第 ' + number + ' 页]\n' + item.text.slice(0, limit)
          + (item.text.length > limit ? '\n（本页节选）' : '') + '\n\n';
        if (excerpt.length + section.length > budget) break;
        excerpt += section;
        included++;
      }
      closeView();
      newConv();
      const input = $('#input');
      input.value = '请只根据附件回答。能找到依据时标明原页码，找不到时说明附件未提供。\n我的问题：';
      input.dispatchEvent(new Event('input'));
      input.focus();
      if (included < results.length) excerpt += '（附件只包括前 ' + included + ' 个已识别页。）';
      await attachFile(new File([excerpt], name + '-页码摘录.txt', { type: 'text/plain' }), draftFor());
    };

    renderResult();
    return { body, refreshModels, importImage, restoreTask,
      usesTask: id => id === taskId && !!job || Boolean(batch?.ids?.has(id)) };
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
    $('#exportHistory').onclick = exportHistory;

    $('#input').addEventListener('input', () => {
      draftFor().text = $('#input').value;
      autoGrow();
      refreshComposer();
    });
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
        const conv = state.current;
        const cur = conv.cwd || state.server?.workspace || '~';
        pickDir(cur, (p) => {
          if (p) conv.cwd = p;
          else delete conv.cwd;
          saveConv(conv);
          renderCwd();
          log(p ? '项目文件夹 → ' + p : '项目文件夹已解绑，跟随默认目录');
        });
      };
    }

    $('#stopGeneration').onclick = () => state.abort?.abort();
    $('#send').onclick = (e) => {
      if (e.currentTarget.classList.contains('stop')) state.abort?.abort();
      else send();
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
      log(state.web ? '联网搜索已开：需要查证时会搜索' : '联网搜索已关');
    };
    // 深度思考。模型没有思考通道时按钮灰掉 —— 能点却毫无反应最让人困惑。
    // 服务器把这件事报在 /api/state 的 model.think 上（refresh_model_markers
    // 读的是 GGUF 里的 chat_template，不是猜的）。
    $('#btnThink').onclick = () => {
      const s = state.server || {};
      const info = (s.info && s.info.think !== undefined) ? s.info.think : true;
      if (info === false) {
        log('当前模型没有思考通道（chat_template 里没有 <think>）', 'err');
        return;
      }
      state.think = !state.think;
      writePreference(THINK_KEY, state.think ? '1' : '0');
      $('#btnThink').classList.toggle('on', state.think);
      $('#btnThink').setAttribute('aria-pressed', state.think ? 'true' : 'false');
      log(state.think ? '会显示模型的推理过程'
        : s.info?.direct_answer ? '模型会直接回答' : '不显示推理过程');
    };
    $('#btnThink').classList.toggle('on', state.think);

    $('#fileInput').onchange = (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      attachFile(f, filePickerDraft || draftFor());
      filePickerDraft = null;
    };

    // 拖文件进窗口。桌面 App 里这条比点按钮更常用 —— 从 Finder 直接拖过来
    // 比"点附件 → 在面板里翻目录 → 选文件"快得多。
    let dragDepth = 0;
    const hasFiles = (e) => !!(e.dataTransfer &&
      Array.from(e.dataTransfer.types || []).includes('Files'));
    const takeFile = (f) => {
      if (!f) return;
      // 模型文件不是附件：把它当图片/文本发过去只会得到一句"不支持这种类型"，
      // 而用户的意图恰恰是"换这个模型"。往模型页引，比报错有用。
      if (/\.gguf$/i.test(f.name || '')) {
        alert('这是模型文件，不是对话附件。\n\n左侧「模型」页可以加载它，'
              + '或者把 .gguf 直接拖到 Dock 里的 Dubhe 图标上。');
        return;
      }
      attachFile(f, draftFor());
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
      const file = e.dataTransfer.files?.[0];
      if (state.view === 'scan') scanPane?.importImage(file);
      else takeFile(file);
    });

    $('#tryChat').onclick = () => {
      if (!state.server?.loaded) { openView('models'); return; }
      const input = $('#input');
      input.value = '用两句话介绍一个适合初学者的观测或读书方法。';
      input.dispatchEvent(new Event('input'));
      input.focus();
    };
    $('#tryTranslation').onclick = () => openView('translate');
    $('#tryScan').onclick = () => openView('scan');

    // 左栏导航：功能视图盖在对话区上，切回来时
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
    emptyState = $('#emptyState');
    loadFallbackHistory();
    state.memory = loadMemory();
    bind();
    activateConv(state.convs[0] || null);
    renderHistoryNotice();
    syncHistory();
    const server = await refreshState();
    if (server?.share_only) { setInterval(refreshState, 8000); return; }
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
