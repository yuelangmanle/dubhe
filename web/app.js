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

  const STORE_KEY = 'dubhe.conv.v1';
  const SET_KEY = 'dubhe.settings.v1';

  const state = {
    convs: [],
    current: null,
    streaming: false,
    server: null,
    tools: true,
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
      const d = el('div', 'conv' + (state.current && c.id === state.current.id ? ' active' : ''));
      d.appendChild(el('span', 'dot'));
      d.appendChild(el('span', 'conv-title', c.title));
      d.onclick = () => {
        state.current = c;
        renderConvList();
        renderMessages();
        $('#convTitle').textContent = c.title;
        if (window.innerWidth <= 760) $('#sidebar').classList.remove('open');
      };
      list.appendChild(d);
    }
  }

  function messageNode(m) {
    const wrap = el('div', 'msg ' + m.role);
    if (m.role === 'user') {
      const b = el('div', 'bubble');
      b.textContent = m.content;
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
        head.appendChild(el('span', 'ic', t.ok === false ? '✕' : '⚡'));
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
      return s;
    } catch (e) {
      $('#modelChip').textContent = '服务未连接';
      return null;
    }
  }

  // ---------------------------------------------------------------- 发送
  async function send(text) {
    if (state.streaming) return;
    const content = (text != null ? text : $('#input').value).trim();
    if (!content) return;

    if (!state.current) newConv();
    const conv = state.current;

    conv.messages.push({ role: 'user', content });
    if (conv.title === '新对话') {
      conv.title = content.slice(0, 24) + (content.length > 24 ? '…' : '');
      $('#convTitle').textContent = conv.title;
    }
    $('#input').value = '';
    autoGrow();

    const ai = { role: 'assistant', content: '', tools: [], meta: null };
    conv.messages.push(ai);

    const box = $('#messages');
    const empty = $('#emptyState');
    if (empty && empty.parentNode) empty.remove();
    box.appendChild(messageNode({ role: 'user', content }));
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
      .map((m) => ({ role: m.role, content: m.content }));
    if (settings.system) msgs.unshift({ role: 'system', content: settings.system });

    try {
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      ai.content = (ai.content || '') + `\n\n> ⚠️ 请求失败：${e.message}`;
    } finally {
      state.streaming = false;
      setSendEnabled(true);
      const finalNode = messageNode(ai);
      node.replaceWith(finalNode);
      body.classList.remove('typing');
      saveConvs();
      scrollBottom(true);
      refreshState();
    }
  }

  function setSendEnabled(on) {
    const b = $('#send');
    b.disabled = !on;
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
        kv('上下文', String(i.n_ctx));
        kv('Chat 模板', i.chatml ? 'ChatML' : '无');
        body.appendChild(box);
      } else {
        body.appendChild(el('div', 'desc', '尚未加载模型。用命令行启动：dubhed -m <model.gguf>'));
      }

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

  function bind() {
    $('#input').addEventListener('input', autoGrow);
    $('#input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
    });
    $('#send').onclick = () => send();
    $('#newChat').onclick = () => newConv();
    $('#toggleSidebar').onclick = () => $('#sidebar').classList.toggle('open');
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
      if (e.key === 'Escape') closeSheet();
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
