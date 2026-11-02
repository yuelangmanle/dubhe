// ui_settings_shot.mjs — 把设置面板滚到"权限"那一段再截图
//
// 设置面板是一整张长表单，默认只截到顶部（模型/外观/生成参数）。这一版加了
// "工具轮数"和"沙箱未生效"的说明，都在下面看不见的地方 —— 不滚下去截图，
// 等于没验证。
import { mkdirSync, writeFileSync } from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:7792/';
const out = process.argv[3] || '/tmp/settings-perm.png';
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync('/tmp', { recursive: true });

const list = await (await fetch(base + '/json')).json();
const page = list.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? rej(new Error(m.error.message)) : res(m.result);
  }
});
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
const send = (method, params = {}) =>
  new Promise((res, rej) => {
    const mid = ++id;
    pending.set(mid, { res, rej });
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
const js = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
  return r.result.value;
};

await send('Page.enable');
await send('Page.navigate', { url });
await sleep(1500);
await send('Page.setWebLifecycleState', { state: 'active' });
await js(`document.getElementById('openSettings').click()`);
await sleep(700);
const info = await js(`(() => {
  const body = document.querySelector('.sheet-body');
  if (!body) return { err: 'no-sheet-body' };
  // 滚到"工具轮数"那一行
  const rows = [...body.querySelectorAll('*')].filter(x => x.children.length === 0 && /工具轮数/.test(x.textContent || ''));
  const hit = rows[0];
  if (hit) hit.scrollIntoView({ block: 'center' });
  const warn = body.querySelector('.perm-warn');
  return {
    scrolled: !!hit,
    warn: warn ? warn.textContent.slice(0, 60) : null,
    rounds: (() => {
      const r = [...body.querySelectorAll('*')].find(x => /工具轮数/.test(x.textContent || '') && x.children.length === 0);
      return r ? 'found' : 'missing';
    })(),
  };
})()`);
await sleep(500);
const shot = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(out, Buffer.from(shot.data, 'base64'));
console.log(JSON.stringify(info));
ws.close();
