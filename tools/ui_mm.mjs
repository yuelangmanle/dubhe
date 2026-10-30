// ui_mm.mjs — 通过界面走一遍多模态：选图 → 预览 → 发送 → 看回复
//
// 为什么要在界面上测而不是直接 curl：附件这条路有三段容易断
//   1. 文件读成 data URL（FileReader）
//   2. 缩略图（原图几 MB，直接进 localStorage 会撑爆配额）
//   3. attachmentParts() 拼成 OpenAI 的 image_url 结构
// 任何一段断了 curl 都发现不了。
import { mkdirSync, writeFileSync } from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:7788/';
const img = process.argv[3];
const outdir = process.argv[4] || '/tmp/uimm';
const base = 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });
if (!img) throw new Error('用法: node tools/ui_mm.mjs <url> <图片路径> [outdir]');

const list = await (await fetch(base + '/json')).json();
const page = list.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const problems = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? rej(new Error(m.error.message)) : res(m.result);
    return;
  }
  if (m.method === 'Runtime.exceptionThrown')
    problems.push('未捕获异常: ' + (m.params.exceptionDetails.exception?.description || ''));
  if (m.method === 'Network.responseReceived' && m.params.response.status >= 400)
    problems.push(`HTTP ${m.params.response.status} ${m.params.response.url}`);
});
await new Promise((r) => ws.addEventListener('open', r));
const send = (method, params = {}) =>
  new Promise((res, rej) => {
    const i = ++id;
    pending.set(i, { res, rej });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
async function ev(expr) {
  const r = await send('Runtime.evaluate', {
    expression: expr, awaitPromise: true, returnByValue: true, userGesture: true,
  });
  if (r.exceptionDetails)
    return { __err: r.exceptionDetails.exception?.description || r.exceptionDetails.text };
  return r.result.value;
}
async function shot(n) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${outdir}/${n}.png`, Buffer.from(r.data, 'base64'));
}

await send('Runtime.enable');
await send('Network.enable');
await send('Page.enable');
await send('DOM.enable');
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440, height: 900, deviceScaleFactor: 2, mobile: false,
});
await send('Page.navigate', { url });
await sleep(2500);

// 视觉塔有没有加载，看界面上的按钮状态最直接（state 在闭包里取不到）
const enabled = await ev(`
  (() => { const b = document.getElementById('btnImage'); return b ? !b.disabled : 'missing'; })()
`);
console.log('图片按钮可用:', enabled);
if (enabled !== true) problems.push('加载了视觉塔但图片按钮仍不可用');

// 把文件塞进隐藏的 input，再触发 change
const { root } = await send('DOM.getDocument', { depth: -1 });
const { nodeId } = await send('DOM.querySelector', {
  nodeId: root.nodeId, selector: '#fileInput',
});
if (!nodeId) throw new Error('找不到 #fileInput');
await send('DOM.setFileInputFiles', { nodeId, files: [img] });
await sleep(1500);
await shot('01-attached');

const att = await ev(`
  (() => {
    const b = document.getElementById('attachBar');
    return b ? b.innerText.replace(/\\s+/g, ' ').slice(0, 120) : '（无附件条）';
  })()
`);
console.log('附件条:', att);
if (/无附件条|^\s*$/.test(String(att))) problems.push('选完图后附件条没出现');

// 发送
await ev(`
  (() => {
    const ta = document.getElementById('input');
    ta.value = '这张图里是什么？用一句话说。';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('send').click();
  })()
`);
await sleep(1500);
await shot('02-streaming');

// 完成判据必须和**发送前**的快照比：历史消息里早就有「已完成」了，
// 只看当前文本会在第一轮就误判为完成（第一次写就是这么错的）。
const baseText = await ev(`document.getElementById('messages').innerText`);
let waited = 0, text = '';
while (waited < 180000) {
  await sleep(600);
  waited += 600;
  text = await ev(`document.getElementById('messages').innerText`);
  const grew = String(text).length > String(baseText).length + 10;
  const tail = String(text).slice(String(baseText).length - 20);
  if (grew && /已完成/.test(tail)) break;
}
console.log('等待', (waited / 1000).toFixed(1), '秒');
await shot('03-done');
const reply = String(text).split('这张图里是什么？用一句话说。').pop().trim();
console.log('回复:', reply.slice(0, 200));
if (reply.length < 10) problems.push('多模态回复为空');
if (/请求失败|⚠️/.test(reply)) problems.push('多模态请求失败: ' + reply.slice(0, 120));

writeFileSync(`${outdir}/problems.txt`, problems.join('\n'), 'utf8');
console.log('\n=== 问题 ===');
if (!problems.length) console.log('无');
else [...new Set(problems)].forEach((p) => console.log(' -', p));
ws.close();
