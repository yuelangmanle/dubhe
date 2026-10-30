// ui_walk.mjs — 用 CDP 把界面从头到尾走一遍，截图 + 收集控制台错误
//
// 为什么需要：前端 4600 行，靠肉眼看代码判断"接没接好"不现实。这个脚本
// 做三件事，都是外面看不出来的：
//   1. 把每个视图都点开，看有没有抛异常（很多视图是懒渲染的，不点不炸）
//   2. 收集所有 console.error / uncaught / 4xx 5xx 响应
//   3. 真发一条消息，确认流式输出走通、token 计数在动
//
// 前置：Chrome 带 --remote-debugging-port=9222 跑着
// 用法：node tools/ui_walk.mjs [url] [outdir]
const url = process.argv[2] || 'http://127.0.0.1:7788/';
const outdir = process.argv[3] || '/tmp/uiwalk';
const base = 'http://127.0.0.1:9222';

import { mkdirSync, writeFileSync } from 'node:fs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });

async function firstPage() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(base + '/json')).json();
      const page = list.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return page;
    } catch {}
    await sleep(250);
  }
  throw new Error('无法连接 Chrome 调试端口 9222');
}

const t = await firstPage();
const ws = new WebSocket(t.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const problems = [];
const logs = [];

ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? rej(new Error(m.error.message)) : res(m.result);
    return;
  }
  if (m.method === 'Runtime.consoleAPICalled') {
    const txt = (m.params.args || [])
      .map((a) => a.value ?? a.description ?? a.type)
      .join(' ');
    logs.push(`[${m.params.type}] ${txt}`);
    if (m.params.type === 'error') problems.push(`console.error: ${txt}`);
  }
  if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails;
    problems.push(
      `未捕获异常: ${d.exception?.description || d.text} @${d.url || ''}:${d.lineNumber}`
    );
  }
  if (m.method === 'Log.entryAdded') {
    const en = m.params.entry;
    if (en.level === 'error') problems.push(`[log] ${en.text} ${en.url || ''}`);
  }
  if (m.method === 'Network.responseReceived') {
    const r = m.params.response;
    if (r.status >= 400) problems.push(`HTTP ${r.status} ${r.url}`);
  }
  if (m.method === 'Network.loadingFailed') {
    problems.push(`请求失败 ${m.params.errorText}`);
  }
});

await new Promise((r) => ws.addEventListener('open', r));

function send(method, params = {}) {
  const i = ++id;
  return new Promise((res, rej) => {
    pending.set(i, { res, rej });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
}

async function evalJs(expr, awaitPromise = true) {
  const r = await send('Runtime.evaluate', {
    expression: expr,
    awaitPromise,
    returnByValue: true,
    userGesture: true,
  });
  if (r.exceptionDetails) {
    throw new Error(
      'eval 失败: ' +
        (r.exceptionDetails.exception?.description || r.exceptionDetails.text)
    );
  }
  return r.result.value;
}

async function ev(expr) {
  const r = await send('Runtime.evaluate', {
    expression: expr, awaitPromise: true, returnByValue: true, userGesture: true,
  });
  if (r.exceptionDetails)
    return { __err: r.exceptionDetails.exception?.description || r.exceptionDetails.text };
  return r.result.value;
}

async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  const p = `${outdir}/${name}.png`;
  writeFileSync(p, Buffer.from(r.data, 'base64'));
  return p;
}

await send('Runtime.enable');
await send('Log.enable');
await send('Network.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440,
  height: 900,
  deviceScaleFactor: 2,
  mobile: false,
});

await send('Page.navigate', { url });
await sleep(2500);

const title = await evalJs('document.title');
console.log('标题:', title);

// 1) 初始视图
await shot('00-chat-empty');

// 2) 逐个视图点开
const views = await evalJs(`
  [...document.querySelectorAll('.nav-item[data-view]')].map(b => b.dataset.view)
`);
console.log('导航项:', views.join(', '));
for (const v of views) {
  const ok = await evalJs(`
    (() => {
      const b = document.querySelector('.nav-item[data-view="${v}"]');
      if (!b) return false;
      b.click();
      return true;
    })()
  `);
  if (!ok) {
    problems.push(`找不到导航项 ${v}`);
    continue;
  }
  await sleep(900);
  const body = await evalJs(`(document.getElementById('viewBody')||{}).innerText||''`);
  const visible = await evalJs(`
    (() => {
      const el = document.getElementById('views');
      return el ? getComputedStyle(el).display !== 'none' : false;
    })()
  `);
  await shot(`01-view-${v}`);
  if (v === 'chat') {
    // 对话不是「视图」：它用 #messages + #composer，打开时 #views 本来就该收起。
    const msgs = await ev(`(document.getElementById('messages')||{}).innerText||''`);
    console.log(`  chat     主界面，消息区 ${String(msgs).length} 字`);
    if (String(msgs).length < 5) problems.push('对话页消息区是空的');
    continue;
  }
  console.log(
    `  ${v.padEnd(8)} 视图可见=${visible}  正文 ${String(body).trim().length} 字`
  );
  if (!visible) problems.push(`视图 ${v} 打开后没有显示`);
  if (String(body).trim().length < 10)
    problems.push(`视图 ${v} 正文为空（可能渲染失败）`);
}

// 3) 回聊天页，真发一条消息
await evalJs(`
  (() => {
    const b = document.querySelector('.nav-item[data-view="chat"]');
    if (b) b.click();
    const ta = document.getElementById('input');
    ta.value = '用一句话说明什么是本地推理。';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  })()
`);
await sleep(400);
await evalJs(`document.getElementById('send').click()`);
await sleep(1200);
await shot('02-chat-streaming');

// 等流式结束（气泡里出现统计或 12 秒超时）
let waited = 0;
let text = '';
while (waited < 15000) {
  await sleep(600);
  waited += 600;
  text = await evalJs(`document.getElementById('messages').innerText`);
  if (/tok\/s|tokens/.test(text) && text.length > 40) break;
}
await shot('03-chat-done');
console.log('回复长度:', text.length, '字');
if (text.length < 30) problems.push('发送消息后没有拿到回复');

// 4) 看侧栏的实时指标有没有动
const gauge = await evalJs(`
  JSON.stringify({
    tps: (document.getElementById('gTps')||{}).textContent,
    tok: (document.getElementById('gTok')||{}).textContent,
    tpot: (document.getElementById('gTime')||{}).textContent,
    ttft: (document.getElementById('gTtft')||{}).textContent,
  })
`);
console.log('仪表:', gauge);

// 5) 打开设置面板
await evalJs(`document.getElementById('openSettings').click()`);
await sleep(600);
await shot('04-settings');
await evalJs(`document.getElementById('sheetClose').click()`);

writeFileSync(`${outdir}/console.log`, logs.join('\n'), 'utf8');

console.log('\n=== 问题 ===');
if (problems.length === 0) console.log('无');
else [...new Set(problems)].forEach((p) => console.log(' -', p));

ws.close();
process.exit(problems.length ? 1 : 0);
