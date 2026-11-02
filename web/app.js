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
    attach: null,   // 待发送的图片/音频/视频（一次只允许一个）
    logs: [],       // 右栏系统日志：只记真实事件，不编
    gen: null,      // 本轮生成统计 { tokens, tps, total, ttft }
    q: '',          // 左栏搜索关键词
    abort: null,    // 生成中的 AbortController，用于"停止"
    view: null,     // 当前打开的功能视图（null = 对话）
    fsPath: '',     // 文件视图当前目录
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
  function saveConvs() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state.convs)); } catch (e) {}
  }
  function loadConvs() {
    try { state.convs = JSON.parse(localStorage.getItem(STORE_KEY) || '[]'); } catch (e) { state.convs = []; }
    if (!Array.isArray(state.convs)) state.convs = [];
  }
  function saveSettings() { localStorage.setItem(SET_KEY, JSON.stringify(settings)); }

  function newConv() {
    const c = { id: 'c' + Date.now(), title: '新对话', messages: [], at: Date.now() };
    state.convs.unshift(c);
    state.current = c;
    saveConvs();
    renderConvList();
    renderMessages();
    $('#convTitle').textContent = '新对话';
    $('#input').focus();
    return c;
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
          html += `<pre><button class="copy-code" data-copy>复制</button><code>${codeBuf.join('\n')}</code></pre>`;
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
    if (inCode && codeBuf.length) html += `<pre><code>${codeBuf.join('\n')}</code></pre>`;
    closeList();
    return html;
  }

  // 隐藏模型输出的 tool_call 原文
  function stripToolCalls(s) {
    return s.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').replace(/<tool_result>[\s\S]*?<\/tool_result>/g, '').trim();
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
        $('#convTitle').textContent = c.title;
        if (window.innerWidth <= 860) closeSidebar();
      };
      list.appendChild(d);
    }
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
          const kind = m.att.kind === 'audio' ? 'audio'
            : (m.att.kind === 'video' ? 'video' : (m.att.kind === 'image' ? 'image' : 'file'));
          const f = el('div', 'mfile');
          f.appendChild(iconNode(kind, 'i mfile-i'));
          f.appendChild(el('span', '', m.att.name));
          b.appendChild(f);
        }
      }
      if (m.content) b.appendChild(document.createTextNode(m.content));
      wrap.appendChild(b);
      return wrap;
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
      st.appendChild(el('span', '', '生成中…'));
      col.appendChild(st);
    }

    const body = el('div', 'body');
    body.innerHTML = bodyHtml(m.content, m.tools && m.tools.length);
    col.appendChild(body);

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
      for (const t of m.tools) {
        const card = el('div', 'tool-card' + (t.ok === false ? ' failed' : ''));
        const head = el('div', 'tool-head');
        const ic = el('span', 'ic');
        ic.appendChild(iconNode(t.ok === false ? 'x' : 'terminal', 'i'));
        head.appendChild(ic);
        head.appendChild(el('span', '', '调用工具'));
        head.appendChild(el('span', 'tool-name', t.name));
        card.appendChild(head);
        if (t.output) {
          const tb = el('div', 'tool-body', t.output.slice(0, 1200));
          card.appendChild(tb);
        }
        col.appendChild(card);
      }
    }

    if (m.meta) {
      const meta = el('div', 'meta');
      if (m.meta.tps) meta.appendChild(el('span', '', m.meta.tps.toFixed(1) + ' tok/s'));
      if (m.meta.tokens) meta.appendChild(el('span', '', m.meta.tokens + ' tokens'));
      col.appendChild(meta);
    }
    return wrap;
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
    scrollBottom();
  }

  function scrollBottom(force) {
    const box = $('#messages');
    const near = box.scrollHeight - box.scrollTop - box.clientHeight < 220;
    if (force || near) box.scrollTop = box.scrollHeight;
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
    const qm = /(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*|BF16|F16|F32)/.exec((s && s.model_id) || '');
    if (qm) return qm[1];
    // model_id 里没有量化标记时退回 quant 串：跳过 F32/F16 这类"未量化"的边角层，
    // 取第一个真正的量化块。全都未量化才显示 F32。
    const parts = (((s && s.info && s.info.quant) || '')).split('+').filter(Boolean);
    return parts.find((x) => !/^(F32|F16|BF16)$/.test(x)) || parts[0] || '';
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
      const row = el('div', 'tool-row' + (t.ok === false ? ' fail' : ''));
      const ti = el('span', 'ti');
      ti.appendChild(iconNode(t.ok === false ? 'x' : 'terminal', 'i'));
      row.appendChild(ti);
      row.appendChild(el('span', 'tn', t.name));
      const td = el('span', 'td');
      if (t.ok === false) {
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

  function readAttachment(file) {
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
      ph.appendChild(iconNode(a.kind === 'audio' ? 'audio' : 'video', 'i'));
      chip.appendChild(ph);
    }
    chip.appendChild(el('div', 'an', a.name));
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

    const userMsg = { role: 'user', content };
    if (att) userMsg.att = { kind: att.kind, name: att.name, thumb: att.thumb, mime: att.mime };
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

    const msgs = conv.messages
      .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
      .map((m) => {
        if (m === userMsg && att) {
          const parts = [attachmentParts(att)];
          if (content) parts.push({ type: 'text', text: content });
          return { role: 'user', content: parts };
        }
        // 历史消息里的附件只留一行说明：服务端只认最后一条 user 的媒体，
        // 而且占位符没配特征会让模型看到空 token。
        const c = m.content || '';
        return { role: m.role, content: m.att ? (c ? c + '\n' : '') + `[${m.att.kind}: ${m.att.name}]` : c };
      });
    if (settings.system) msgs.unshift({ role: 'system', content: settings.system });

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
      const withTools = (opts.tools !== undefined ? opts.tools : state.tools) && !att;
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
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let raf = null;
      let genTick = 0;

      const paint = () => {
        body.innerHTML = bodyHtml(ai.content, false) || '<p></p>';
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = null; scrollBottom(); });
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
            } else if (j.dubhe.type === 'tool_result') {
              const last = ai.tools[ai.tools.length - 1];
              if (last) {
                last.output = j.dubhe.output;
                last.ok = j.dubhe.ok;
                last.ms = performance.now() - (last.t0 || performance.now());
              }
              renderTools();
              // 重绘以显示工具卡片。
              //
              // 注意这两行必须改的是外层 let：工具结果一到，整个气泡换成
              // 带卡片的新节点，后面所有增量都得写进新节点的 .body。
              const fresh = messageNode(ai);
              node.replaceWith(fresh);
              node = fresh;
              body = node.querySelector('.body');
              body.classList.add('typing');
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
      if (!String(ai.content || '').trim() && !(ai.tools && ai.tools.length)) {
        ai.content = '（这一轮没有任何输出。可能是上下文放不下，或者模型直接收尾了，换个问法再试。）';
        const nn = messageNode(ai);
        node.replaceWith(nn);
        node = nn;
        body = node.querySelector('.body');
        body.classList.remove('typing');
      }
      saveConvs();
      scrollBottom(true);
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

  function openSettings() {
    openSheet('设置', (body) => {
      body.appendChild(el('div', 'group-title', '模型'));
      if (state.server && state.server.info) {
        const i = state.server.info;
        const box = el('div');
        const kv = (k, v) => { const d = el('div', 'kv'); d.appendChild(el('span', 'k', k)); d.appendChild(el('span', 'v', v)); box.appendChild(d); };
        kv('架构', i.arch);
        kv('参数量', (i.params / 1e9).toFixed(2) + ' B');
        kv('量化', i.quant);
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
      if (state.server) {
        row(body, '服务端 Shell', state.server.shell ? '已启用' : '未启用（启动加 --allow-shell）', el('span', 'value', state.server.shell ? '开' : '关'));
      }

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
  };

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
      main.appendChild(nm);
      const bits = [];
      if (m.quant) bits.push(m.quant);
      if (m.params) bits.push((m.params / 1e9).toFixed(1) + 'B');
      bits.push(fmtBytes(m.bytes));
      bits.push(fmtAgo(m.mtime));
      main.appendChild(el('div', 'vmeta', bits.join(' · ')));
      row.appendChild(main);
      if (!m.loaded && !isMm) {
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
    $('#input').addEventListener('focus', () => setTimeout(() => scrollBottom(true), 340));
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => scrollBottom(true));
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
    $('#fileInput').onchange = async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      if (f.size > 64 * 1024 * 1024) { alert('文件太大，上限 64 MB'); return; }
      try {
        const a = await readAttachment(f);
        if (a.kind === 'image') a.thumb = await makeThumb(a.dataUrl);
        state.attach = a;
        renderAttach();
      } catch (err) {
        alert('读取文件失败：' + err.message);
      }
    };

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
      const btn = e.target.closest('[data-copy]');
      if (btn) {
        const code = btn.parentNode.querySelector('code');
        navigator.clipboard.writeText(code.textContent);
        btn.textContent = '已复制';
        setTimeout(() => (btn.textContent = '复制'), 1200);
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { closeSheet(); closeSidebar(); if (state.view) closeView(); }
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
