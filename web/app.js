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
      const tip = el('div', 'hw', '还没有对话');
      tip.style.padding = '14px 12px';
      list.appendChild(tip);
      return;
    }
    for (const c of state.convs) {
      const d = el('div', 'conv lit' + (state.current && c.id === state.current.id ? ' active' : ''));
      d.appendChild(el('span', 'dot'));
      d.appendChild(el('span', 'conv-title', c.title));
      d.onclick = () => {
        state.current = c;
        renderConvList();
        renderMessages();
        $('#convTitle').textContent = c.title;
        if (window.innerWidth <= 760) closeSidebar();
      };
      list.appendChild(d);
    }
  }

  function messageNode(m) {
    const wrap = el('div', 'msg ' + m.role);
    if (m.role === 'user') {
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
    const body = el('div', 'body');
    body.innerHTML = renderMarkdown(stripToolCalls(m.content || ''));
    wrap.appendChild(body);

    if (m.tools && m.tools.length) {
      for (const t of m.tools) {
        const card = el('div', 'tool-card' + (t.ok === false ? ' failed' : ''));
        const head = el('div', 'tool-head');
        const ic = el('span', 'ic');
        ic.appendChild(iconNode(t.ok === false ? 'x' : 'bolt', 'i'));
        head.appendChild(ic);
        head.appendChild(el('span', '', '调用工具'));
        head.appendChild(el('span', 'tool-name', t.name));
        card.appendChild(head);
        if (t.output) {
          const tb = el('div', 'tool-body', t.output.slice(0, 1200));
          card.appendChild(tb);
        }
        wrap.appendChild(card);
      }
    }

    if (m.meta) {
      const meta = el('div', 'meta');
      if (m.meta.tps) meta.appendChild(el('span', '', m.meta.tps.toFixed(1) + ' tok/s'));
      if (m.meta.tokens) meta.appendChild(el('span', '', m.meta.tokens + ' tokens'));
      wrap.appendChild(meta);
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
  async function refreshState() {
    try {
      const r = await fetch('/api/state');
      const s = await r.json();
      state.server = s;
      const chip = $('#modelChip');
      if (s.loaded && s.info) {
        const p = (s.info.params / 1e9).toFixed(2);
        chip.textContent = `${s.info.arch} · ${s.info.quant} · ${p}B`;
        chip.classList.add('live');
        chip.title = s.model;
      } else {
        chip.textContent = '未加载模型 · 点击设置';
        chip.classList.remove('live');
      }
      $('#hwInfo').textContent = s.hardware || '';
      // 只有加载了视觉塔/语音塔才露出附件入口
      const att = $('#btnAttach');
      if (att) att.hidden = !(s.mm_vision || s.mm_audio);
      return s;
    } catch (e) {
      $('#modelChip').textContent = '服务未连接';
      return null;
    }
  }

  // ---------------------------------------------------------------- 附件
  // 服务端一次只接受一个媒体（视觉塔和语音塔是两个独立的 mmproj，同时注入
  // 需要两套前缀），所以这里也做成单选。
  function pickAttachment() {
    const s = state.server || {};
    const acc = [];
    if (s.mm_vision) acc.push('image/*', 'video/*');
    if (s.mm_audio) acc.push('audio/*');
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

    const ai = { role: 'assistant', content: '', tools: [], meta: null };
    conv.messages.push(ai);

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
              ai.tools.push({ name: j.dubhe.name, args: j.dubhe.arguments, ok: true });
            } else if (j.dubhe.type === 'tool_result') {
              const last = ai.tools[ai.tools.length - 1];
              if (last) { last.output = j.dubhe.output; last.ok = j.dubhe.ok; }
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
          if (delta && delta.content) { ai.content += delta.content; paint(); }
          if (j.usage) {
            ai.meta = { tps: j.usage.decode_tps, tokens: j.usage.completion_tokens };
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
      setSendEnabled(true);
      const finalNode = messageNode(ai);
      node.replaceWith(finalNode);
      body.classList.remove('typing');
      saveConvs();
      scrollBottom(true);
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
    const open = $('#sidebar').classList.contains('open');
    const scrim = $('#scrim');
    if (scrim) scrim.classList.toggle('show', open);
  }

  function closeSidebar() {
    $('#sidebar').classList.remove('open');
    syncScrim();
  }

  function bindSidebar() {
    const sb = $('#sidebar');
    let sx = 0, sy = 0, drag = false;

    sb.addEventListener('touchstart', (e) => {
      if (window.innerWidth > 760) return;
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
    document.querySelectorAll('.chip, .new-btn, .icon-btn, .send-btn, .conv, .side-item')
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

  const ICON_SEND = icon('arrow-up');
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
    $('#toggleSidebar').onclick = () => {
      $('#sidebar').classList.toggle('open');
      syncScrim();
    };
    $('#scrim').onclick = closeSidebar;
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
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); newConv(); }
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
    setInterval(refreshState, 8000);
  }

  boot();
})();
