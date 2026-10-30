// ui_probe.mjs — 针对性探查：API 面板、附件按钮、窄屏、错误态
//
// ui_walk.mjs 走的是"每个视图能不能渲染"，这个脚本问的是另一类问题：
// 按钮点了会不会炸、窄屏会不会挤成一团、后端挂了会不会白屏。
import { mkdirSync, writeFileSync } from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:7788/';
const outdir = process.argv[3] || '/tmp/uiprobe';
// 调试端口可用 CDP_URL 覆盖。默认 9222；上一个无头 Chrome 没退干净、
// 端口还被占着的时候，换一个端口起新的就行，不用去翻进程表。
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });

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
  if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails;
    problems.push(`未捕获异常: ${d.exception?.description || d.text}`);
  }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    problems.push(
      'console.error: ' +
        (m.params.args || []).map((a) => a.value ?? a.description).join(' ')
    );
  }
  if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) {
    problems.push(`HTTP ${m.params.response.status} ${m.params.response.url}`);
  }
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
    expression: expr,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (r.exceptionDetails)
    return { __err: r.exceptionDetails.exception?.description || r.exceptionDetails.text };
  return r.result.value;
}
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${outdir}/${name}.png`, Buffer.from(r.data, 'base64'));
}

await send('Runtime.enable');
await send('Network.enable');
await send('Page.enable');

async function size(w, h) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: w, height: h, deviceScaleFactor: 2, mobile: w < 600,
  });
}

await size(1440, 900);
await send('Page.navigate', { url });
await sleep(2200);

// 1) API 服务（没有 data-view 的导航项）
const apiOpened = await ev(`
  (() => {
    const b = [...document.querySelectorAll('.nav-item')].find(x => /API/.test(x.innerText));
    if (!b) return 'no-button';
    b.click();
    return 'clicked';
  })()
`);
await sleep(900);
await shot('api-panel');
const apiText = await ev(`
  (() => {
    // 不能用 offsetParent 判断可见性：这个面板是 position: fixed，
    // 而 offsetParent 对 fixed 元素永远返回 null —— 面板明明弹着也会被判成"没打开"。
    // 直接看 computedStyle 和内容。
    const s = document.getElementById('sheet');
    if (!s) return '（没有 sheet 元素）';
    const cs = getComputedStyle(s);
    const shown = cs.display !== 'none' && cs.visibility !== 'hidden'
      && parseFloat(cs.opacity || '1') > 0;
    if (!shown) return '（面板未打开：display=' + cs.display + ' opacity=' + cs.opacity + '）';
    const t = (document.getElementById('sheetBody') || {}).innerText || '';
    return t.slice(0, 300) || '（面板打开了但正文是空的）';
  })()
`);
console.log('API 面板:', apiOpened, '→', JSON.stringify(apiText).slice(0, 200));
if (!/BASE URL|base url|v1/i.test(String(apiText))) problems.push('API 面板没打开或内容不对');

// 关掉面板
await ev(`(() => { const c = document.getElementById('sheetClose'); if (c) c.click(); })()`);
await sleep(300);

// 2) 附件入口：能不能用完全由 /api/state 的 mm_vision / mm_audio 决定。
//
// 这里只读状态、不点。点一下会拉起原生文件选择框，无头浏览器里既看不到
// 也没法关掉，只会让后面的步骤全部卡住。真正"点了能不能用"由 ui_mm.mjs
// 走完整流程验证。
const att = await ev(`
  (() => {
    const box = document.getElementById('btnAttach');
    const img = document.getElementById('btnImage');
    const mic = document.getElementById('btnMic');
    return {
      attachHidden: box ? box.hidden : null,
      imageDisabled: img ? img.disabled : null,
      micDisabled: mic ? mic.disabled : null,
    };
  })()
`);
const mmState = await ev(`fetch('/api/state').then(r=>r.json()).then(s=>({mm:s.mm, vision:s.mm_vision, audio:s.mm_audio}))`);
console.log('  附件入口:', JSON.stringify(att), '| 后端:', JSON.stringify(mmState));
// 前端显示必须和后端能力一致，否则用户会看到一个点了没反应的按钮
if (att.attachHidden === false && !mmState.mm) problems.push('后端没有多模态能力，附件按钮却露着');
if (att.attachHidden === true && mmState.mm) problems.push('后端有多模态能力，附件按钮却藏着');
if (att.imageDisabled === false && !mmState.vision) problems.push('没有视觉塔，图片按钮却是可点的');

// 3) 窄屏
await size(390, 844);
await sleep(800);
await shot('narrow-chat');
const narrow = await ev(`
  JSON.stringify({
    bodyW: document.body.clientWidth,
    overflowX: document.documentElement.scrollWidth > document.body.clientWidth + 2,
    sidebar: (() => { const s = document.getElementById('sidebar');
      return s ? Math.round(s.getBoundingClientRect().width) : -1; })(),
    composerW: (() => { const c = document.getElementById('composer');
      return c ? Math.round(c.getBoundingClientRect().width) : -1; })(),
  })
`);
console.log('窄屏:', narrow);

// 4) 窄屏下打开一个视图
await ev(`(() => { const b = document.querySelector('.nav-item[data-view="models"]'); if (b) b.click(); })()`);
await sleep(800);
await shot('narrow-models');

await size(1440, 900);
writeFileSync(`${outdir}/problems.txt`, problems.join('\n'), 'utf8');
console.log('\n=== 问题 ===');
if (!problems.length) console.log('无');
else [...new Set(problems)].forEach((p) => console.log(' -', p));
ws.close();
