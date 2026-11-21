import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { access, mkdtemp, readFile, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = fileURLToPath(new URL('../web/', import.meta.url));
const profile = await mkdtemp(join(tmpdir(), 'dubhe-client-'));
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const fixtures = new Map();
const requests = [];
const translations = [];
const scans = [];
const scanDownloads = [];
const scanFiles = [
  { id: 'xiaomi-ocr-0-q4', name: '小米 OCR', bytes: 529297664 },
  { id: 'xiaomi-ocr-0-vision', name: '视觉组件', bytes: 207346624 },
  { id: 'xiaomi-ocr-0-bf16', name: '小米 OCR BF16', bytes: 1516744960 },
];
const scannerInstalled = file => scanInstalled && file.id !== 'xiaomi-ocr-0-bf16';
const history = new Map();
const tasks = new Map();
let holdChat;
let holdSummary;
let holdTranslation;
let failSecondBlock = false;
let model = '/models/a.gguf';
let historyError = false;
let historyImports = 0;
let transferState = 'verifying';
let shareOnly = false;
let removedModel = '';
let scanInstalled = false;
let pauseAtScan = 0;
const state = () => ({
  share_only: shareOnly,
  loaded: !!model, model, model_id: 'fixture', info: { params: 100, n_ctx: 4096, think: true },
  workspace: '/a-project', net: true, mm_vision: true,
});
const collect = async (req) => {
  let raw = '';
  for await (const data of req) raw += data;
  return JSON.parse(raw || '{}');
};
const handler = async (req, res) => {
  const path = new URL(req.url, 'http://127.0.0.1').pathname;
  const json = (data, status = 200) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  };
  if (path === '/api/history') {
    if (req.method === 'GET') return json({ items: [...history.values()] });
    if (historyError) return json({ error: 'fixture write failed' }, 500);
    const data = await collect(req);
    if (data.import) historyImports++;
    for (const c of (data.import || [])) if (!history.has(c.id)) history.set(c.id, c);
    for (const c of (data.upsert || [])) history.set(c.id, c);
    return json({ ok: true });
  }
  if (path === '/api/tasks') {
    if (req.method === 'GET') {
      const id = new URL(req.url, 'http://127.0.0.1').searchParams.get('id');
      if (id) return tasks.has(id) ? json({ task: tasks.get(id) }) : json({ error: '记录已移除' }, 404);
      return json({ tasks: [...tasks.values()].reverse().map(({ payload, ...summary }) => summary) });
    }
    const data = await collect(req);
    if (data.upsert) {
      const task = data.upsert;
      tasks.set(task.id, { ...task, updated: Date.now(), bytes: JSON.stringify(task.payload).length });
    }
    if (data.delete) tasks.delete(data.delete);
    return json({ ok: true });
  }
  if (path === '/api/state') return json(state());
  if (path === '/api/models') return json({ dir: '/models', current: model, items: [
    { name: 'A', path: '/models/a.gguf', loadable: true },
    { name: 'B', path: '/models/b.gguf', loadable: true },
    { name: 'Fixture MLX', path: '/models/mlx', mlx: true, loadable: true, loaded: model === '/models/mlx' },
  ].filter(item => item.path !== removedModel) });
  if (path === '/api/models/delete') {
    ({ path: removedModel } = await collect(req));
    return json({ ok: true });
  }
  if (path === '/api/translate') {
    if (req.method === 'GET') return json({
      default_model: 'hy-mt2-18b-q4', runtime_available: true,
      models: [
        { id: 'hy-mt2-18b-q4', name: 'Hy-MT2', installed: true },
        { id: 'index-translate-2b-q4', name: 'Index-Translate', installed: true },
      ],
    });
    const data = await collect(req);
    translations.push(data);
    if (data.text.includes('HTTP failure')) return json({ error: '请先下载模型' }, 404);
    if (data.text.includes('Token limit')) return json({ error: '译文达到长度上限' }, 422);
    if (data.text.includes('Cancel translation')) return;
    if (data.text === 'Background source') {
      holdTranslation = () => json({ text: '后台译文' });
      return;
    }
    if (data.text === 'Second block' && failSecondBlock) {
      failSecondBlock = false;
      return json({ error: 'fixture translation failed' }, 502);
    }
    if (data.text.includes('Drop marker')) return json({ text: '译文' });
    return json({ text: data.formatted ? data.text.replaceAll('Results', '结果').replace('Read', '阅读') : '译文' });
  }
  if (path === '/api/scan') {
    if (req.method === 'GET') return json({
      runtime_available: true,
      installed: scanInstalled,
      files: scanFiles.map(file => ({ ...file, installed: scannerInstalled(file) })),
    });
    const job = await collect(req);
    scans.push(job);
    if (scans.length === pauseAtScan) return;
    if (job.mode === 'extract') return json({ text: '{"题目":"年度报告","日期":null}', kind: 'json' });
    if (job.mode === 'ask') return json({ text: '报告提到了流量。', kind: 'text' });
    if (job.mode === 'chart') return json({ text: 'Station | Level\n--- | ---\nLake A | ~10.2', kind: 'markdown' });

    return json({
      text: '# 年度报告\n\n<table><tr><td colspan="2"><img>流量</td></tr></table>\n\n模型输出 \\(p_0 \\in [0.08, 0.20]\\)。',
      kind: 'markdown',
    });
  }
  if (path === '/api/load') {
    ({ path: model } = await collect(req));
    return json({ ok: true, model, ms: 1 });
  }
  if (path === '/api/unload') {
    model = '';
    return json({ ok: true });
  }
  if (path === '/api/store') return json({ items: [
    { id: 'test-model', name: 'Test transfer', params: '1B', quant: 'Q4', bytes: 1024,
      installed: false, note: 'fixture', state: transferState },
    { id: 'hy-mt2-18b-q4', name: 'Hy-MT2', params: '1.8B', quant: 'Q4', bytes: 1133080448,
      installed: false, translation: true, note: '离线翻译', state: 'idle' },
    { id: 'index-translate-2b-q4', name: 'Index-Translate', params: '2B', quant: 'Q4', bytes: 1312164352,
      installed: true, translation: true, note: '离线翻译', state: 'done' },
    ...scanFiles.map(file => ({ ...file, scan: true, installed: scannerInstalled(file),
      state: scannerInstalled(file) ? 'done' : scanDownloads.includes(file.id) ? 'running' : 'idle',
      got: 0, total: file.bytes })),
  ] });
  if (path === '/api/store/download') {
    const { id } = await collect(req);
    scanDownloads.push(id);
    return json({ ok: true, state: 'running' });
  }
  if (path === '/api/store/cancel') {
    transferState = 'cancelling';
    return json({ ok: true, state: transferState });
  }
  if (path === '/api/summarize') {
    await collect(req);
    holdSummary = () => json({ summary: 'compressed' });
    return;
  }
  if (path === '/v1/chat/completions') {
    const data = await collect(req);
    requests.push(data);
    const prompt = String(data.messages.at(-1).content);
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.flushHeaders();
    const finish = () => res.end('data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n');
    if (prompt === 'broken stream') return res.end('data: {"choices":[{"delta":{"content":"partial reply"}}]}\n\n');
    if (prompt === 'first chat') holdChat = finish;
    else finish();
    return;
  }
  const name = path === '/' ? 'index.html' : path.slice(1);
  if (!/^(?:vendor\/)?[\w.-]+$/.test(name)) { res.writeHead(404).end(); return; }
  const contentType = { html: 'text/html', js: 'text/javascript', css: 'text/css' };
  try {
    if (!fixtures.has(name)) fixtures.set(name, await readFile(join(web, name)));
    res.writeHead(200, { 'Content-Type': contentType[name.split('.').at(-1)] || 'application/octet-stream' });
    res.end(fixtures.get(name));
  } catch { res.writeHead(404).end(); }
};
const newServer = () => createServer((req, res) => handler(req, res).catch(error => {
  console.error(error);
  res.destroy();
}));
const server = newServer();
const secondServer = newServer();

async function chromePath() {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN;
  for (const path of [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    ...String(process.env.PATH || '').split(delimiter).flatMap(dir =>
      ['google-chrome', 'chromium', 'chromium-browser'].map(name => join(dir, name))),
  ]) {
    try { await access(path); return path; } catch {}
  }
  throw new Error('Chrome/Chromium is required. Set CHROME_BIN to its executable.');
}

async function until(check, label) {
  const end = Date.now() + 8000;
  do {
    const value = await check();
    if (value) return value;
    await wait(35);
  } while (Date.now() < end);
  assert.fail('Timed out: ' + label);
}

let chrome;
let socket;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = 'http://127.0.0.1:' + server.address().port;
  chrome = spawn(await chromePath(), [
    '--headless=new', '--user-data-dir=' + profile, '--remote-debugging-port=0',
    '--disable-background-networking', '--no-first-run', '--no-sandbox', 'about:blank',
  ], { stdio: 'ignore' });
  let chromeError;
  chrome.once('error', error => { chromeError = error; });
  const port = await until(async () => {
    if (chromeError) throw chromeError;
    if (chrome.exitCode !== null) throw new Error('Chrome exited with status ' + chrome.exitCode);
    try { return (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; }
    catch { return null; }
  }, 'Chrome startup');
  const [target] = (await (await fetch('http://127.0.0.1:' + port + '/json/list')).json())
    .filter(tab => tab.type === 'page');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let seq = 0;
  const pending = new Map();
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(data);
    if (!pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  });
  const cdp = (method, params) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const r = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  const ready = async () => {
    await until(() => evaluate('document.readyState === "complete" && !!document.querySelector("#newChat")?.onclick'), 'page loaded');
    await until(() => evaluate('document.querySelector("#liveBadge").classList.contains("on")'), 'model state loaded');
  };
  const click = (selector) => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const draft = (text) => evaluate(`(() => {
    const input = document.querySelector('#input');
    input.value = ${JSON.stringify(text)};
    input.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  const send = async (text) => { await draft(text); await click('#send'); };
  const navigate = async (url) => {
    await cdp('Page.navigate', { url });
    await until(() => evaluate('location.href === ' + JSON.stringify(new URL(url).href)), 'updated address');
  };
  const openTranslation = async () => {
    await click('[data-view="translate"]');
    await until(() => evaluate('!!document.querySelector("#viewBody select option")'), 'translation panel');
  };
  const translate = (text) => evaluate(`(() => {
    document.querySelector('#viewBody textarea').value = ${JSON.stringify(text)};
    document.querySelector('#viewBody .tbar .pri').click();
  })()`);
  const translateStatus = () => evaluate('document.querySelector("#viewBody .tbar .pri")?.parentElement.querySelector(".vmeta")?.textContent || ""');
  const translationDone = (status) => until(async () => (await translateStatus()).includes(status), status);

  await navigate(origin);
  await ready();
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1250, height: 786, deviceScaleFactor: 1, mobile: false });
  await until(() => evaluate('innerWidth === 1250'), 'desktop viewport');
  const desktopLayout = () => evaluate(`['#sidebar', 'main.chat', '#inspector'].map(selector => {
    const { x, y, width, height } = document.querySelector(selector).getBoundingClientRect();
    return { x, y, width, height };
  })`);
  const assertDesktopChat = async () => {
    const [nav, chat, inspector] = await desktopLayout();
    assert.equal(nav.width, 236, 'navigation uses the desktop grid width');
    assert.equal(inspector.width, 316, 'inspector uses the desktop grid width');
    assert.ok(nav.x < chat.x && chat.x < inspector.x, 'inspector stays to the right of chat');
    assert.equal(inspector.y, chat.y, 'inspector stays in the chat row');
    assert.equal(inspector.height, chat.height, 'the desktop panes have the same height');
    assert.ok(chat.height > 750, 'chat fills the window');
  };
  await assertDesktopChat();
  await openTranslation();
  const [, translation, hiddenInspector] = await desktopLayout();
  assert.ok(translation.width > 950, 'translation expands into the inspector column');
  assert.equal(hiddenInspector.width, 0);
  await click('#viewBack');
  await assertDesktopChat();
  await click('[data-view="scan"]');
  const [, scanning, scanInspector] = await desktopLayout();
  assert.ok(scanning.width > 950, 'scanner expands into the inspector column');
  assert.equal(scanInspector.width, 0);
  await click('#viewBack');
  await assertDesktopChat();
  await cdp('Emulation.clearDeviceMetricsOverride');
  await click('#newChat');
  await send('first chat');
  await until(() => holdChat, 'first chat streaming');
  await click('#newChat');
  await click('#btnWeb');
  await click('#btnThink');
  await draft('second chat');
  await evaluate(`document.querySelector('#input').dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }))`);
  assert.match(await evaluate('document.querySelector("#queuedMessages").textContent'), /second chat/);
  await click('#btnWeb');
  await click('#btnThink');
  await click('#newChat');
  await evaluate(`(() => {
    const t = new DataTransfer();
    t.items.add(new File(['attachment contents'], 'queued.md', { type: 'text/markdown' }));
    const f = document.querySelector('#fileInput');
    f.files = t.files;
    f.dispatchEvent(new Event('change'));
  })()`);
  await until(() => evaluate('document.querySelector("#attachBar").textContent.includes("queued.md")'), 'attachment prepared');
  await click('#send');
  assert.match(await evaluate('document.querySelector("#queuedMessages").textContent'), /queued.md/);
  await send('remove this job');
  await evaluate('Array.from(document.querySelectorAll("#queuedMessages button")).at(-1).click()');
  await draft('unsent third draft');
  await click('#newChat');
  await draft('separate active draft');
  await evaluate('document.querySelector("#input").focus()');
  holdChat();
  await until(() => requests.length === 3, 'queued chats sent');
  const second = requests[1];
  assert.deepEqual(second.messages.filter(m => m.role === 'user').map(m => m.content), ['second chat']);
  assert.equal(second.dubhe_search, true);
  assert.equal(second.dubhe_tools, false);
  assert.ok(!second.messages.some(m => m.role === 'system' && m.content.includes('web_search')));
  assert.equal(second.thinking, true);
  assert.equal(second.workspace, '/a-project');
  assert.match(requests[2].messages.at(-1).content, /attachment contents/);
  assert.equal(requests[2].dubhe_tools, false);
  assert.equal(requests[2].dubhe_search, false);
  assert.equal(await evaluate('document.querySelector("#input").value'), 'separate active draft', 'queued chat cleared a draft');
  assert.equal(await evaluate('document.querySelector("#convTitle").textContent'), '新对话', 'queued chat switched conversations');
  assert.equal(await evaluate('document.activeElement.id'), 'input', 'queued chat stole keyboard focus');
  assert.equal(await evaluate('document.querySelectorAll("#messages .msg").length'), 0, 'background reply leaked into current chat');
  await evaluate('Array.from(document.querySelectorAll("#convList .conv")).find(r => r.textContent.includes("queued.md")).click()');
  assert.equal(await evaluate('document.querySelector("#input").value'), 'unsent third draft');

  const legacy = JSON.stringify([{
    id: 'legacy', title: 'Legacy conversation',
    at: 1, messages: Array.from({ length: 18 }, (_, i) => ({
      role: i % 2 ? 'assistant' : 'user', content: 'historical turn ' + i,
    })),
  }]);
  await evaluate('localStorage.setItem("dubhe.conv.v1", ' + JSON.stringify(legacy) + ')');
  await navigate(origin + '/?legacy');
  await ready();
  await until(() => history.has('legacy'), 'legacy history imported');
  await evaluate('Array.from(document.querySelectorAll("#convList .conv")).find(r => r.textContent.includes("Legacy")).click()');
  await send('message being compressed');
  await until(() => holdSummary, 'compression started');
  await click('#newChat');
  await send('after compression');
  assert.equal(requests.length, 3, 'another job entered inference while compression was running');
  holdSummary();
  await until(() => requests.length === 5, 'both jobs ran after compression');
  assert.match(requests[3].messages[0].content, /compressed/);
  assert.equal(requests[4].messages.at(-1).content, 'after compression');
  await click('#newChat');
  await send('broken stream');
  await until(() => evaluate('document.querySelector("#messages").textContent.includes("回答中断")'), 'unfinished chat is marked interrupted');
  assert.match(await evaluate('document.querySelector("#messages").textContent'), /partial reply/);

  historyError = true;
  await evaluate(`(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key.startsWith('dubhe.history.pending')) throw new DOMException('fixture', 'QuotaExceededError');
      return original.call(this, key, value);
    };
    window.exportedHistory = null;
    window.webkit = { messageHandlers: { shell: { postMessage: m => {
      if (m.type === 'historyExport') window.exportedHistory = JSON.parse(m.text);
    } } } };
  })()`);
  await click('#newChat');
  await send('newest survives quota failure');
  await until(() => evaluate('document.querySelector("#historyNotice").textContent.includes("请导出")'), 'quota failure shown');
  assert.equal(await evaluate('localStorage.getItem("dubhe.conv.v1")'), legacy, 'legacy browser history was changed');
  await click('#exportHistory');
  const exported = await evaluate('window.exportedHistory.items.map(c => c.title)');
  assert.ok(exported.some(title => title.includes('newest')));
  assert.ok(exported.includes('Legacy conversation'));
  historyError = false;
  await until(() => [...history.values()].some(c => c.messages.some(m => m.content === 'newest survives quota failure')), 'failed history retried');

  await new Promise(resolve => secondServer.listen(0, '127.0.0.1', resolve));
  const otherOrigin = 'http://127.0.0.1:' + secondServer.address().port;
  await navigate(otherOrigin);
  await ready();
  await until(() => evaluate('document.querySelector("#convList").textContent.includes("newest")'), 'history loaded on a new port');
  assert.equal(historyImports, 1);

  await openTranslation();
  await translate('HTTP failure');
  await translationDone('失败');
  assert.match(await translateStatus(), /请先下载模型/);
  await translate('Token limit');
  await translationDone('长度上限');
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '');
  const fenced = ['~~~~txt', 'a'.repeat(1300), '~~~', 'keep exact whitespace  ', '~~~~'].join('\n');
  const previous = translations.length;
  const chatBefore = requests.length;
  const chatModel = model;
  await translate('A paragraph\n\n' + fenced + '\n\nB paragraph');
  await translationDone('完成');
  assert.equal(translations.length - previous, 2);
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '译文\n\n' + fenced + '\n\n译文');
  const displayMath = ['$$', 'x + y = 1', '', 'y = 2', '$$'].join('\n');
  const beforeMath = translations.length;
  await translate('A line\n\n' + displayMath);
  await translationDone('完成');
  assert.equal(translations.length - beforeMath, 1);
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '译文\n\n' + displayMath);
  await evaluate(`(() => {
    const sel = document.querySelector('#viewBody select');
    sel.value = 'index-translate-2b-q4';
    sel.dispatchEvent(new Event('change'));
    const glossary = document.querySelector('.translation-glossary textarea');
    glossary.value = 'Results = 结果';
    glossary.dispatchEvent(new Event('input'));
  })()`);
  const markdown = '# Results\n\nRead [Results](https://example.org/results) and `dubhe run` at 104.7 m³/s. [@smith2024] $x^2$';
  await translate(markdown);
  await translationDone('完成');
  assert.equal(translations.at(-1).model, 'index-translate-2b-q4');
  assert.deepEqual(translations.at(-1).glossary, [{ source: 'Results', target: '结果' }]);
  assert.equal(requests.length, chatBefore);
  assert.equal(model, chatModel, 'translation changed the active chat model');
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), markdown.replaceAll('Results', '结果').replace('Read', '阅读'));
  await translate('Drop marker `dubhe`');
  await translationDone('完成');
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '译文 `dubhe`');
  await translate('Cancel translation');
  await until(() => translations.at(-1).text.includes('Cancel translation'), 'translation request opened');
  await evaluate('Array.from(document.querySelectorAll("#viewBody button")).find(b => b.textContent === "停止翻译").click()');
  await translationDone('已停止');
  await translate('Background source');
  await until(() => holdTranslation, 'background translation is running');
  await click('[data-view="chat"]');
  await openTranslation();
  assert.equal(await evaluate('document.querySelector("#viewBody textarea").value'), 'Background source');
  assert.equal(await evaluate('[...document.querySelectorAll("#viewBody button")].find(b => b.textContent === "停止翻译").hidden'), false);
  await click('[data-view="chat"]');
  holdTranslation();
  await openTranslation();
  await translationDone('完成');
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '后台译文');
  await evaluate(`(() => {
    window.documentCalls = [];
    window.webkit = { messageHandlers: { shell: { postMessage: m => {
      if (m.type !== 'translationDocument') return;
      window.documentCalls.push(m);
      const response = m.action === 'save' || m.action === 'export' ? { saved: true }
        : m.translations ? { document: 'dHJhbnNsYXRlZA==' } : { paragraphs: ['First block', 'Second block'] };
      setTimeout(() => window.dubheDocumentComplete(m.id, response));
    } } } };
    const selected = new DataTransfer();
    selected.items.add(new File(['fixture'], 'research.docx'));
    const picker = document.querySelector('#viewBody input[type=file]');
    picker.files = selected.files;
    picker.dispatchEvent(new Event('change'));
  })()`);
  await translationDone('research.docx 已读取');
  assert.ok(await evaluate('document.querySelector("#viewBody textarea").readOnly'));
  const beforeDocument = translations.length;
  failSecondBlock = true;
  await click('#viewBody .tbar .pri');
  await translationDone('已保留完成的译文');
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '译文');
  await click('[data-view="models"]');
  await openTranslation();
  assert.equal(await evaluate('document.querySelector("#viewBody .tbar .pri").textContent'), '继续翻译');
  await click('#viewBody .tbar .pri');
  await translationDone('完成 · 2 段');
  assert.deepEqual(translations.slice(beforeDocument).map(t => t.text), ['First block', 'Second block', 'Second block']);
  await click('#viewBody .translation-export button');
  await translationDone('已保存译文文件');
  assert.deepEqual(await evaluate('window.documentCalls[1].translations'), ['译文', '译文']);
  assert.equal(await evaluate('window.documentCalls[2].name'), 'research-译文.docx');
  await click('#translationClear');
  const report = '# Results\n\nThe river rose.\n\n- Retain this item.\n\n| Results | Flow |\n| --- | --- |\n| A | 6 |';
  await translate(report);
  await translationDone('完成');
  assert.equal(await evaluate('document.querySelector(".translation-preview h1").textContent'), '结果');
  assert.deepEqual(await evaluate('[...document.querySelectorAll(".translation-preview tbody td")].map(td => td.textContent)'), ['A', '6']);
  for (const [width, sideBySide] of [[1100, true], [500, false]]) {
    await cdp('Emulation.setDeviceMetricsOverride', { width, height: 820, deviceScaleFactor: 1, mobile: false });
    await until(() => evaluate('innerWidth === ' + width), 'translation viewport');
    assert.equal(await evaluate(`(() => {
      const [source, result] = [...document.querySelectorAll('.translation-card')].map(card => card.getBoundingClientRect());
      return result.left > source.left && result.top === source.top;
    })()`), sideBySide);
  }
  await cdp('Emulation.clearDeviceMetricsOverride');
  for (const format of ['md', 'txt', 'pdf', 'docx']) {
    await evaluate(`(() => {
      const select = document.querySelector('.translation-export select');
      select.value = "${format}";
      select.dispatchEvent(new Event('change'));
      document.querySelector('.translation-export button').click();
    })()`);
    await translationDone('已保存译文文件');
    const exported = await evaluate('window.documentCalls.at(-1)');
    assert.equal(exported.action, 'export');
    assert.equal(exported.name, '译文.' + format);
    assert.equal(exported.text, report.replaceAll('Results', '结果'));
    if (format === 'pdf' || format === 'docx') {
      assert.deepEqual(exported.blocks.map(b => b.kind), ['h1', 'p', 'li', 'table']);
    }
  }
  await click('[data-view="chat"]');
  await evaluate(`(() => {
    window.overlays = [];
    window.webkit = { messageHandlers: { shell: { postMessage: m => window.overlays.push(m) } } };
  })()`);
  await evaluate('window.dubheOcr("OCR source")');
  await evaluate('window.dubheText("Selected source")');
  assert.deepEqual(await evaluate('window.overlays.map(m => [m.type, m.title])'), [
    ['overlay', '翻译中'], ['overlay', '译文'], ['overlay', '翻译中'], ['overlay', '译文'],
  ]);
  assert.equal(await evaluate('window.overlays[1].source'), 'OCR source');
  assert.equal(await evaluate('window.overlays[3].source'), 'Selected source');
  assert.equal(await evaluate('document.querySelector("#views").hidden'), true);
  await evaluate(`(() => {
    const input = document.querySelector('#input');
    input.value = 'before only this after';
    input.focus();
    input.setSelectionRange(7, 16);
    return window.dubheTranslateLocalSelection();
  })()`);
  await until(() => evaluate('window.overlays.at(-1).title === "译文"'), 'the local selection is translated');
  assert.equal(await evaluate('window.overlays.at(-1).source'), 'only this');
  assert.equal(translations.at(-1).text, 'only this');

  await click('[data-view="models"]');
  await until(() => evaluate('document.querySelector("#viewBody").textContent.includes("正在验证")'), 'verification shown');
  await evaluate(`(() => {
    const mirror = document.querySelector('#viewBody input[aria-label="模型下载源"]');
    mirror.focus();
    mirror.value = 'https://mirror.example';
    mirror.dispatchEvent(new Event('input'));
  })()`);
  transferState = 'running';
  await until(() => evaluate('document.querySelector("#viewBody").textContent.includes("下载中")'), 'download progress refreshed');
  assert.equal(await evaluate('document.activeElement.value'), 'https://mirror.example', 'polling preserves the mirror input and focus');
  await evaluate(`(() => {
    window.confirm = () => false;
    const row = [...document.querySelectorAll('#viewBody .vrow')].find(r => r.querySelector('.vname')?.textContent === 'B');
    [...row.querySelectorAll('button')].find(b => b.textContent === '删除').click();
  })()`);
  await until(() => evaluate('!!document.querySelector("dialog[open]")'), 'model delete confirmation appears in the page');
  assert.equal(removedModel, '');
  await click('dialog[open] button[value="cancel"]');
  await until(() => evaluate(`(() => {
    const row = [...document.querySelectorAll('#viewBody .vrow')].find(r => r.querySelector('.vname')?.textContent === 'B');
    return !document.querySelector('dialog[open]') && [...row.querySelectorAll('button')].some(b => b.textContent === '删除' && !b.disabled);
  })()`), 'cancel closes the confirmation and re-enables deletion');
  assert.equal(removedModel, '');
  await evaluate(`(() => {
    const row = [...document.querySelectorAll('#viewBody .vrow')].find(r => r.querySelector('.vname')?.textContent === 'B');
    [...row.querySelectorAll('button')].find(b => b.textContent === '删除').click();
  })()`);
  await click('dialog[open] button[value="delete"]');
  await until(() => removedModel === '/models/b.gguf', 'confirmed deletion is sent');
  await until(() => evaluate('![...document.querySelectorAll("#viewBody .vname")].some(n => n.textContent === "B")'), 'deleted model vanishes');
  assert.equal(await evaluate('document.querySelector(\'#viewBody input[aria-label="模型下载源"]\').value'), 'https://mirror.example', 'refreshing the local models preserves the mirror draft');
  assert.deepEqual(await evaluate(`(() => {
    const row = Array.from(document.querySelectorAll('#viewBody .vrow'))
      .find(r => r.textContent.includes('Fixture MLX'));
    return Array.from(row.querySelectorAll('button')).map(b => b.textContent);
  })()`), ['加载'], 'MLX models can be loaded from the model list');
  await evaluate('Array.from(document.querySelectorAll("#viewBody button")).find(b => b.textContent === "取消").click()');
  await until(() => evaluate('document.querySelector("#viewBody").textContent.includes("正在取消")'), 'cancelling shown');
  assert.ok(!(await evaluate('document.querySelector("#viewBody").textContent.includes("继续")')));
  transferState = 'cancelled';
  await until(() => evaluate('document.querySelector("#viewBody").textContent.includes("继续")'), 'download can be resumed');
  await evaluate(`Array.from(document.querySelectorAll('#viewBody .vrow'))
    .find(r => r.textContent.includes('Fixture MLX')).querySelector('button').click()`);
  await until(() => model === '/models/mlx', 'MLX model selected');
  await until(() => evaluate('document.querySelector("#viewActions").textContent.includes("卸载当前模型")'), 'unload action');
  await evaluate('[...document.querySelectorAll("#viewActions button")].find(b => b.textContent === "卸载当前模型").click()');
  await until(() => model === '', 'current model unloaded');
  await until(() => evaluate('!document.querySelector("#viewActions").textContent.includes("卸载当前模型")'), 'the model view reflects the unload');
  await until(() => evaluate(`[...document.querySelectorAll('#viewBody .vrow')].some(row =>
    row.textContent.includes('Hy-MT2') && row.querySelector('button')?.textContent === '下载')`), 'specialized weight is downloadable');
  await evaluate(`[...document.querySelectorAll('#viewBody .vrow')].find(row =>
    row.textContent.includes('Index-Translate') && row.querySelector('button')?.textContent === '去翻译').querySelector('button').click()`);
  await until(() => evaluate('document.querySelector("#viewBody select")?.value === "index-translate-2b-q4"'), 'store opens installed translator');
  assert.equal(model, '');

  await click('[data-view="scan"]');
  await until(() => evaluate('!document.querySelector("#scanInstall").disabled'), 'scanner model catalog');
  await click('#scanInstall');
  await until(() => scanDownloads.length === 2, 'both scanner components requested');
  assert.deepEqual(scanDownloads.sort(), ['xiaomi-ocr-0-q4', 'xiaomi-ocr-0-vision']);
  scanInstalled = true;
  await until(() => evaluate('document.querySelector(".scan-model-status").textContent.includes("已安装")'),
    'scanner ready');
  const chooseVariant = value => evaluate(`(() => {
    const variant = document.querySelector('#scanVariant');
    variant.value = ${JSON.stringify(value)};
    variant.dispatchEvent(new Event('change'));
  })()`);
  await chooseVariant('bf16');
  await click('#scanInstall');
  await until(() => scanDownloads.length === 3, 'high precision weight requested separately');
  assert.equal(scanDownloads.at(-1), 'xiaomi-ocr-0-bf16');
  await until(() => evaluate('!document.querySelector("#scanVariant").disabled'), 'install request registered');
  await chooseVariant('q4');
  await evaluate(`(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 80;
    canvas.height = 110;
    window.makeScannerImage = () => canvas.toDataURL('image/jpeg');
    window.scanNativeCalls = [];
    window.webkit = { messageHandlers: { shell: { postMessage: message => {
      window.scanNativeCalls.push(message);
      if (message.type === 'scanOpenPDF') window.dubheScanPDFReady({
        requestId: message.requestId, name: 'paper.docx', pageCount: 4, identity: 'paper-identity',
      });
      if (message.type === 'scanPDFPage') window.dubheScanPDFPage({
        ...message, image: window.makeScannerImage(),
      });
      if (message.type === 'scanFindRegions') window.dubheScanRegionsReady({
        requestId: message.requestId, regions: [{ mode: 'table', x: .2, y: .3, width: .6, height: .5 }],
      });
      if (message.type === 'scanSave') window.dubheScanSaved({ requestId: message.requestId, saved: true });
    } } } };
  })()`);
  await click('[data-view="translate"]');
  await until(() => evaluate('!document.querySelector("#translationOCR").hidden'), 'screenshot OCR options');
  await evaluate(`(() => {
    const chooser = document.querySelector('#translationOCR');
    chooser.value = 'xiaomi';
    chooser.dispatchEvent(new Event('change'));
  })()`);
  assert.equal(await evaluate('window.dubheOCRProvider()'), 'xiaomi');
  assert.equal(await evaluate('JSON.parse(localStorage.getItem("dubhe.translation.v1")).ocr'), 'xiaomi');
  await click('[data-view="scan"]');
  await evaluate(`(async () => {
    const data = new DataTransfer();
    const jpg = await fetch(window.makeScannerImage()).then(response => response.blob());
    data.items.add(new File([jpg], 'report.jpg', { type: 'image/jpeg' }));
    window.dispatchEvent(new DragEvent('drop', { dataTransfer: data, cancelable: true }));
  })()`);
  await until(() => evaluate('!document.querySelector("#scanRun").disabled'), 'image scanner ready');
  const chooseExport = value => evaluate(`(() => {
    const format = document.querySelector('#scanExportFormat');
    format.value = ${JSON.stringify(value)};
    format.dispatchEvent(new Event('change'));
  })()`);
  assert.equal(await evaluate('document.querySelector("#scanExportFormat").value'), 'original');
  await click('#scanExport');
  const originalImage = await evaluate('window.scanNativeCalls.at(-1)');
  assert.equal(originalImage.format, 'pdf');
  assert.equal(originalImage.original, true);
  assert.match(originalImage.image, /^data:image\/jpeg;base64,/);
  await click('#scanReview');
  await click('.scan-review-dialog button.vbtn.ghost');
  await evaluate(`(() => {
    const regions = document.querySelector('[aria-label="识别到的区域"]');
    regions.value = '0';
    regions.dispatchEvent(new Event('change'));
    document.querySelector('[aria-label="选区识别方式"]').value = 'chart';
    [...document.querySelectorAll('dialog button')].find(b => b.textContent === '识别选区').click();
  })()`);
  await until(() => evaluate('document.querySelector(".scan-region-result").value.includes("~10.2")'), 'chart crop produces rows');
  await evaluate('[...document.querySelectorAll("dialog button")].find(b => b.textContent === "用作本页结果").click()');
  await click('.scan-review-dialog .pri');
  await until(() => evaluate('document.querySelector("#scanResult table")?.textContent.includes("Lake A")'), 'save a selection before whole-page OCR');
  await chooseExport('csv');
  await click('#scanExport');
  const csv = await evaluate('window.scanNativeCalls.at(-1)');
  assert.equal(csv.format, 'csv');
  assert.match(csv.name, /图表估读\.csv$/);
  assert.ok(csv.text.includes('"Lake A","~10.2"'));
  await chooseExport('auto');
  await click('#scanRun');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent === "识别完成"'), 'image result');
  assert.match(scans.at(-1).image, /^data:image\/jpeg;base64,/);
  assert.equal(scans.at(-1).mode, 'document');
  assert.equal(await evaluate('document.querySelector("#scanResult table td").colSpan'), 2);
  assert.equal(await evaluate('!!document.querySelector("#scanResult img")'), false);
  assert.equal(await evaluate('document.querySelector("#scanExportFormat").selectedOptions[0].textContent'), 'Markdown（.md）');
  assert.equal(await evaluate('document.querySelector("#scanExport").textContent'), '导出 MD');
  await click('#scanOriginalLayout');
  assert.equal(await evaluate('document.querySelector("#scanResult img").src === document.querySelector(".scan-image-frame img").src'), true);
  assert.equal(await evaluate('!!document.querySelector("#scanResult table")'), false);
  await click('#scanOriginalLayout');
  assert.equal(await evaluate('!!document.querySelector("#scanResult table")'), true);
  await until(() => evaluate('!!document.querySelector("#scanResult math msub")'), 'inline LaTeX produces MathML subscripts');
  assert.equal(await evaluate('document.querySelector("#scanResult math mo").textContent'), '∈');
  await click('#scanEnlarge');
  await until(() => evaluate('document.querySelector(".scan-image-dialog img").style.width !== ""'), 'preview image sized');
  const fitWidth = await evaluate('document.querySelector(".scan-image-dialog img").getBoundingClientRect().width');
  await evaluate(`(() => {
    const zoom = document.querySelector('[aria-label="原件缩放"]');
    zoom.value = '173';
    zoom.dispatchEvent(new Event('input'));
  })()`);
  assert.equal(await evaluate('document.querySelector(".scan-zoom-percent").textContent'), '173%');
  assert.ok(Math.abs(await evaluate('document.querySelector(".scan-image-dialog img").getBoundingClientRect().width') / fitWidth - 1.73) < .04);
  await evaluate(`(() => {
    const viewport = document.querySelector('.scan-image-viewport');
    const bounds = viewport.getBoundingClientRect();
    viewport.dispatchEvent(new WheelEvent('wheel', {
      ctrlKey: true, deltaY: -120, clientX: bounds.x + bounds.width / 2, clientY: bounds.y + bounds.height / 2,
      cancelable: true,
    }));
  })()`);
  assert.ok(await evaluate('document.querySelector(\'[aria-label="原件缩放"]\').valueAsNumber > 173'));
  await evaluate('[...document.querySelectorAll(".scan-image-dialog button")].find(b => b.textContent === "适合窗口").click()');
  assert.equal(await evaluate('document.querySelector(".scan-zoom-percent").textContent'), '100%');
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await until(() => evaluate('!document.querySelector(".scan-image-dialog[open]")'), 'close original preview');
  assert.equal(await evaluate('document.querySelector("#views").hidden'), false);
  assert.equal(await evaluate('document.querySelector("#scanResult")?.textContent.includes("年度报告")'), true);
  await click('#scanRaw');
  assert.equal(await evaluate('document.querySelector("#scanRaw").getAttribute("aria-pressed")'), 'true');
  await click('#scanRaw');
  await click('#scanReview');
  await click('.scan-review-dialog button.vbtn.ghost');
  await evaluate(`(() => {
    const regions = document.querySelector('[aria-label="识别到的区域"]');
    regions.value = '0';
    regions.dispatchEvent(new Event('change'));
  })()`);
  assert.equal(await evaluate('document.querySelector("[aria-label=选区识别方式]").value'), 'table');
  assert.equal(await evaluate('document.querySelector(".scan-region-box").hidden'), false);
  await click('.scan-review-dialog header button');
  await evaluate('document.querySelector("#scanLayout").value = "regions"');
  await click('#scanRun');
  await until(() => evaluate('document.querySelector(".scan-card h3").textContent && document.querySelector("#scanResult table") && document.querySelector("#scanStatus").textContent === "识别完成"'),
    'document regions are reassembled');
  assert.equal(scans.at(-1).mode, 'table');
  assert.match(await evaluate('Array.from(document.querySelectorAll(".scan-source-name")).at(-1).textContent'), /分区识别/);
  await evaluate('document.querySelector("#scanLayout").value = "page"');
  await click('#scanExport');
  assert.equal(await evaluate('window.scanNativeCalls.at(-1).name'), 'report-扫描结果.md');
  await click('#scanTranslate');
  assert.match(await evaluate('document.querySelector("#viewBody textarea[aria-label=原文]").value'), /年度报告/);

  await click('[data-view="scan"]');
  await click('#scanOpenPDF');
  await until(() => evaluate('document.querySelector("#scanPage").textContent.includes("1 / 4")'), 'PDF source');
  await evaluate(`(() => {
    document.querySelector('#scanFrom').value = '2';
    document.querySelector('#scanTo').value = '3';
    document.querySelector('#scanTo').dispatchEvent(new Event('input'));
  })()`);
  const beforePDF = scans.length;
  pauseAtScan = beforePDF + 2;
  await click('#scanRun');
  await until(() => scans.length === pauseAtScan, 'second scan page is running');
  await click('#scanStop');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent.includes("已保留 1 页")'),
    'cancel keeps finished pages');
  assert.equal(await evaluate('document.querySelector("#scanRun").textContent'), '继续识别');
  await click('#scanRun');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent === "识别完成 · 2 页"'),
    'resume skips finished page');
  assert.equal(scans.length, beforePDF + 3);
  assert.deepEqual(await evaluate('window.scanNativeCalls.filter(m => m.type === "scanPDFPage").map(m => m.page)'),
    [1, 2, 3]);
  await click('#scanPrevious');
  await until(() => evaluate('document.querySelector("#scanResult").textContent.includes("年度报告")'),
    'previous PDF result remains available');
  await click('#scanOpenPDF');
  await until(() => evaluate('document.querySelector("#scanPage").textContent.includes("2 / 4")'),
    'reopen original Word document without losing position');
  assert.match(await evaluate('document.querySelector("#scanResult").textContent'), /年度报告/);
  assert.equal(await evaluate('document.querySelector("#scanExportFormat").value'), 'paired');
  await click('#scanExport');
  const pairedExport = await evaluate('window.scanNativeCalls.at(-1)');
  assert.equal(pairedExport.format, 'pdf');
  assert.deepEqual(pairedExport.pages, [2, 3]);
  assert.match(pairedExport.notes[0], /年度报告/);
  await click('#scanOriginalLayout');
  assert.equal(await evaluate('document.querySelector("#scanExportFormat").value'), 'original');
  await click('#scanExport');
  const originalPDF = await evaluate('window.scanNativeCalls.at(-1)');
  assert.equal(originalPDF.original, true);
  assert.equal(originalPDF.name, 'paper-原件.pdf');
  assert.equal(originalPDF.pages, undefined);
  await click('#scanOriginalLayout');
  await chooseExport('auto');
  await evaluate(`(() => {
    const mode = document.querySelector('#scanMode');
    mode.value = 'extract';
    mode.dispatchEvent(new Event('change'));
    const detail = document.querySelector('#scanDetail');
    detail.value = '题目、日期';
    detail.dispatchEvent(new Event('input'));
    document.querySelector('#scanLanguage').value = 'mixed';
    document.querySelector('#scanQuality').value = 'fine';
    document.querySelector('#scanRotation').value = '90';
    document.querySelector('#scanLength').value = '4096';
    document.querySelector('#scanTo').value = '2';
    document.querySelector('#scanTo').dispatchEvent(new Event('input'));
  })()`);
  await click('#scanRun');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent === "识别完成 · 1 页"'),
    'field extraction for selected page');
  assert.deepEqual(scans.at(-1).fields, ['题目', '日期']);
  assert.equal(scans.at(-1).language, 'mixed');
  assert.equal(scans.at(-1).max_tokens, 4096);
  assert.equal(await evaluate('window.scanNativeCalls.filter(m => m.type === "scanPDFPage").at(-1).quality'), 'fine');
  await click('#scanExport');
  const scanExport = await evaluate('window.scanNativeCalls.at(-1)');
  assert.equal(scanExport.format, 'json');
  assert.equal(await evaluate('document.querySelector("#scanExport").textContent'), '导出 JSON');
  assert.deepEqual(JSON.parse(scanExport.text), [{ page: 2, fields: { 题目: '年度报告', 日期: null } }]);
  await click('#scanClear');
  assert.equal(await evaluate('window.scanNativeCalls.at(-1).type'), 'scanClosePDF');

  // Restart both workspaces with only the locally saved checkpoint available.
  model = '/models/a.gguf';
  await navigate(origin + '/?restore=translation#tasks');
  await ready();
  await until(() => evaluate('[...document.querySelectorAll(".task-row .vname")].some(x => x.textContent === "research.docx")'),
    'completed translation in recent tasks after restart');
  await click('button[aria-label="打开任务 research.docx"]');
  await translationDone('已恢复完成的译文');
  assert.equal(await evaluate('document.querySelector("#viewBody pre").textContent'), '译文\n\n译文');
  assert.equal(await evaluate('[...document.querySelectorAll("#viewBody button")].find(b => b.textContent === "重新选择原文件")?.hidden'), false);
  await evaluate(`(() => {
    const glossary = document.querySelector('.translation-glossary textarea');
    glossary.value = 'Runoff = 径流';
    document.querySelector('input[aria-label="术语表名称"]').value = '水文研究';
    [...document.querySelectorAll('.glossary-presets button')].find(b => b.textContent === '保存术语表').click();
    glossary.value = '';
    [...document.querySelectorAll('.glossary-presets button')].find(b => b.textContent === '使用').click();
  })()`);
  assert.equal(await evaluate('document.querySelector(".translation-glossary textarea").value'), 'Runoff = 径流');

  const bridgeForLargePDF = () => evaluate(`(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 120;
    canvas.height = 150;
    const image = canvas.toDataURL('image/jpeg');
    window.scanNativeCalls = [];
    window.webkit = { messageHandlers: { shell: { postMessage: message => {
      window.scanNativeCalls.push(message);
      if (['scanOpenPDF', 'scanRestorePDF'].includes(message.type)) window.dubheScanPDFReady({
        requestId: message.requestId, name: 'twenty-pages.pdf', pageCount: 20,
        bookmark: 'test-bookmark', identity: 'test-identity',
      });
      if (message.type === 'scanPDFPage') window.dubheScanPDFPage({
        ...message, image: message.page === 20 && message.textSource === 'auto' ? '' : image,
        textOnly: message.page === 20 && message.textSource === 'auto',
        selectableText: message.page === 20 ? 'This last page has selectable text and uses no OCR inference.' : '',
      });
    } } } };
  })()`);
  await bridgeForLargePDF();
  await click('[data-view="scan"]');
  await click('#scanOpenPDF');
  await until(() => evaluate('!document.querySelector("#scanRun").disabled'), 'twenty-page PDF is ready');
  await evaluate('document.querySelector("#scanReadMode").value = "auto"');
  pauseAtScan = scans.length + 2;
  await click('#scanRun');
  await until(() => scans.length === pauseAtScan, 'second PDF page reached after first checkpoint');
  await click('#scanStop');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent.includes("已保留 1 页")'),
    'partial PDF checkpoint is safe');
  await navigate(origin + '/?restore=scan#tasks');
  await ready();
  await bridgeForLargePDF();
  await until(() => evaluate('!!document.querySelector(\'button[aria-label="打开任务 twenty-pages.pdf"]\')'),
    'recent task is present after reloading the workspace');
  await click('button[aria-label="打开任务 twenty-pages.pdf"]');
  await until(() => evaluate('document.querySelector("#scanRun")?.disabled === false && document.querySelector("#scanRun").textContent === "继续识别"'),
    'partial PDF opens via bookmark');
  assert.equal(await evaluate('window.scanNativeCalls.find(m => m.type === "scanRestorePDF").bookmark'), 'test-bookmark');
  const resumeCount = scans.length;
  await click('#scanRun');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent === "识别完成 · 20 页"'),
    '20-page PDF completes from saved progress');
  assert.equal(scans.length - resumeCount, 18, 'already saved page and selectable text page are not sent to OCR');
  assert.match(await evaluate('document.querySelector("#scanResult").textContent'), /selectable text/);
  await click('#scanReview');
  await until(() => evaluate('document.querySelector("dialog[open] .scan-region-frame img")?.naturalWidth > 0'),
    'review shows original PDF page');
  const bounds = await evaluate(`(() => {
    const r = document.querySelector('dialog[open] .scan-region-frame img').getBoundingClientRect();
    return [r.left + r.width * 0.2, r.top + r.height * 0.2, r.left + r.width * 0.65, r.top + r.height * 0.65];
  })()`);
  const [x1, y1, x2, y2] = bounds;
  await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', x: x1, y: y1, button: 'left', buttons: 1, clickCount: 1 });
  await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x2, y: y2, button: 'left', buttons: 1 });
  await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', x: x2, y: y2, button: 'left', buttons: 0, clickCount: 1 });
  await evaluate('[...document.querySelectorAll("dialog[open] button")].find(b => b.textContent === "识别选区").click()');
  await until(() => evaluate('document.querySelector("dialog[open] .scan-region-result")?.value.length > 0'),
    'cropped area gets a separate result');
  assert.equal(scans.at(-1).mode, 'text');
  await evaluate(`(() => {
    document.querySelector('textarea[aria-label="本页校对稿"]').value = 'Manual correction, page 20.';
    [...document.querySelectorAll('dialog[open] button')].find(b => b.textContent === '保存本页').click();
  })()`);
  await until(() => evaluate('document.querySelector("#scanResult").textContent.includes("Manual correction")'),
    'the correction appears in the results');
  const saved = [...tasks.values()].find(task => task.title === 'twenty-pages.pdf');
  assert.equal(saved.payload.results.length, 20);
  assert.match(saved.payload.results.find(([page]) => page === 1)[1].text, /年度报告/);
  assert.equal(saved.payload.results.find(([page]) => page === 20)[1].text, 'Manual correction, page 20.');
  assert.equal(JSON.stringify(saved.payload).includes('data:image'), false, 'journal stores result text and a file reference only');

  await evaluate(`(() => {
    const shell = window.webkit.messageHandlers.shell;
    const original = shell.postMessage;
    const opened = new Map();
    const image = document.createElement('canvas').toDataURL('image/jpeg');
    shell.postMessage = message => {
      if (message.type === 'scanBatchOpen') {
        window.dubheScanBatchReady({ requestId: message.requestId,
          files: ['one', 'two'].map(name => ({kind: 'pdf', name: name + '.pdf', pageCount: 1,
            bookmark: 'batch-' + name, identity: name})), unavailable: [] });
      } else if (message.type === 'scanRestorePDF' && message.bookmark.startsWith('batch-')) {
        window.scanNativeCalls.push(message);
        const name = message.bookmark.slice(6);
        opened.set(message.requestId, name);
        window.dubheScanPDFReady({requestId: message.requestId, name: name + '.pdf', pageCount: 1,
          bookmark: message.bookmark, identity: name});
      } else if (message.type === 'scanPDFPage' && opened.has(message.requestId)) {
        window.scanNativeCalls.push(message);
        window.dubheScanPDFPage({...message, image: message.textSource === 'auto' ? '' : image,
          textOnly: message.textSource === 'auto',
          selectableText: 'Selectable PDF text for ' + opened.get(message.requestId)});
      } else original(message);
    };
  })()`);
  const batchBefore = scans.length;
  await click('#scanBatchOpen');
  await until(() => evaluate('document.querySelector("#scanStatus").textContent.includes("批量已完成 2 / 2")'),
    'two PDFs complete as separate saved tasks');
  for (const name of ['one', 'two']) {
    const task = [...tasks.values()].find(item => item.title === name + '.pdf');
    assert.equal(task.payload.results[0][1].text, 'Selectable PDF text for ' + name);
  }
  assert.equal(scans.length, batchBefore);
  assert.deepEqual(await evaluate('window.scanNativeCalls.filter(m => m.type === "scanRestorePDF" && m.bookmark.startsWith("batch-")).map(restore => window.scanNativeCalls.filter(m => m.type === "scanPDFPage" && m.requestId === restore.requestId).map(m => m.textSource))'),
    [['auto'], ['auto', 'ocr']], 'text-only batch renders just the final preview');

  shareOnly = true;
  await navigate(otherOrigin + '/?share');
  await until(() => evaluate('document.querySelector("#shareNotice")?.hidden === false'), 'remote sharing shown');
  assert.equal(await evaluate('document.querySelector("#app").hidden'), true);

  console.log('Dubhe browser regressions passed.');
} finally {
  socket?.close();
  server.closeAllConnections();
  server.close();
  secondServer.closeAllConnections();
  secondServer.close();
  if (chrome && chrome.exitCode === null && chrome.pid) {
    const exited = new Promise(resolve => chrome.once('exit', resolve));
    chrome.kill();
    await exited;
  }
  await rm(profile, { force: true, recursive: true, maxRetries: 5 });
}
