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

  const state = {
    convs: [],
    current: null,
    streaming: false,
    server: null,
    tools: true,
    attach: null,   // 待发送的图片/音频/视频（一次只允许一个）
    logs: [],       // 右栏系统日志：只记真实事件，不编
    gen: null,      // 本轮生成统计 { tokens, tps, total, ttft }
    q: '',          // 左栏搜索关键词
    abort: null,    // 生成中的 AbortController，用于"停止"
  };

  const settings = Object.assign(
    { temperature: 0.7, top_p: 0.95, max_tokens: 1024, system: '', tools: true },
    JSON.parse(localStorage.getItem(SET_KEY) || '{}')
  );

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
    body.innerHTML = renderMarkdown(stripToolCalls(m.content || ''));
    col.appendChild(body);

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
  async function send(text) {
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
    const node = messageNode(ai);
    box.appendChild(node);
    const body = node.querySelector('.body');
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
          dubhe_tools: settings.tools && state.tools,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let raf = null;
      let genTick = 0;

      const paint = () => {
        body.innerHTML = renderMarkdown(stripToolCalls(ai.content)) || '<p></p>';
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
              // 重绘以显示工具卡片
              const fresh = messageNode(ai);
              node.replaceWith(fresh);
              node = fresh;
              node.querySelector('.body').classList.add('typing');
              body = node.querySelector('.body');
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
      const finalNode = messageNode(ai);
      node.replaceWith(finalNode);
      body.classList.remove('typing');
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
        kv('上下文', i.n_ctx + (i.n_ctx_train > i.n_ctx ? ' / 训练 ' + i.n_ctx_train : ''));
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
      toggle(body, '启用工具调用', '允许模型读取文件、列目录、执行命令', () => settings.tools, (v) => { settings.tools = v; saveSettings(); });
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
    $('#btnTools').classList.toggle('on', true);

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

    // 左栏导航：目前只有"对话"是完整实现的。其余项不做假的空页面 ——
    // 点了给一句实话，比给一个空白界面强。
    document.querySelectorAll('.nav-item[data-view]').forEach((b) => {
      b.onclick = () => {
        document.querySelectorAll('.nav-item[data-view]').forEach((x) => x.classList.remove('on'));
        b.classList.add('on');
        if (b.dataset.view !== 'chat') {
          openSheet(b.textContent.trim(), (body) => {
            body.appendChild(el('p', 'desc',
              '这一块还在开发中。当前版本已经可用的是对话、模型切换、工具调用和 API 网关。'));
          });
        }
      };
    });

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
      if (e.key === 'Escape') { closeSheet(); closeSidebar(); }
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
