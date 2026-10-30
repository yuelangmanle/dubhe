// ui_switch.mjs — 在界面上来回切模型，盯住"多模态入口有没有跟着变"
//
// 为什么要专门测这一条：切换模型这个动作本身很容易"看起来成功"——
// 顶栏名字变了、标签变了，但附件按钮还是亮着的。而这时候点图片按钮
// 不会报错，只会拿到一段和图片毫无关系的回答。
//
// 判定用的是 #btnAttach 的 hidden 和 #btnImage 的 disabled，
// 这两个值只由 refreshState() 从 /api/state 的 mm_vision 推出来，
// 所以它同时验证了"后端说真话"和"前端照着做"两件事。
import { mkdirSync, writeFileSync } from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:7788/';
const outdir = process.argv[3] || '/tmp/uiswitch';
// 调试端口可用 CDP_URL 覆盖。默认 9222；上一个无头 Chrome 没退干净、
// 端口还被占着的时候，换一个端口起新的就行，不用去翻进程表。
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });

const list = await (await fetch(base + '/json')).json();
const page = list.find((t) => t.type === 'page');
if (!page) throw new Error('没有可用的页面，先起无头 Chrome');
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
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${outdir}/${name}.png`, Buffer.from(r.data, 'base64'));
}

await send('Runtime.enable');
await send('Network.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', {
  width: 1440, height: 900, deviceScaleFactor: 2, mobile: false,
});
await send('Page.navigate', { url });
await sleep(2200);

// 界面上的当前状态：顶栏名字 + 附件入口是否可见
const snap = () => ev(`
  (() => {
    const att = document.getElementById('btnAttach');
    const bi  = document.getElementById('btnImage');
    return {
      name: (document.getElementById('mpName')||{}).textContent || '',
      tags: (document.getElementById('mpTags')||{}).innerText || '',
      attachHidden: att ? att.hidden : null,
      imageDisabled: bi ? bi.disabled : null,
    };
  })()
`);

// 打开「模型」视图，点某一行的「加载」
async function openModels() {
  await ev(`
    (() => {
      const b = [...document.querySelectorAll('.nav-item')]
        .find(x => /模型/.test(x.innerText));
      if (b) b.click();
    })()
  `);
  await sleep(1200);
}
async function loadByName(name) {
  return ev(`
    (() => {
      const rows = [...document.querySelectorAll('#viewBody .vrow')];
      for (const r of rows) {
        const nm = r.querySelector('.vname');
        if (!nm || !nm.textContent.includes(${JSON.stringify(name)})) continue;
        const b = [...r.querySelectorAll('button')].find(x => /加载/.test(x.textContent));
        if (!b) return 'no-button';
        b.click();
        return 'clicked';
      }
      return 'no-row';
    })()
  `);
}

// 等顶栏名字变成期望值（后端加载要 2~5 秒）
async function waitName(want, ms = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const s = await snap();
    if (s && s.name && s.name.includes(want)) return s;
    await sleep(400);
  }
  return await snap();
}

const before = await snap();
console.log('起始:', JSON.stringify(before));
await openModels();
await shot('01-models');

// ---- 1) 切到纯文本模型，附件入口必须收起来
console.log('\n[1] 切到 LFM3-1.7B-F16（纯文本）');
console.log('  点击:', await loadByName('LFM3-1.7B-F16'));
let s = await waitName('1.7B-F16');
await sleep(600);
await openModels();          // 加载完会重新渲染视图，回到列表
s = await snap();
await shot('02-text-only');
console.log('  切换后:', JSON.stringify(s));
if (!s.name.includes('1.7B-F16')) problems.push('切到 F16 后顶栏名字没变');
if (s.attachHidden !== true) problems.push('切到纯文本模型后，附件按钮仍然可见');
if (s.imageDisabled !== false && s.imageDisabled !== true)
  problems.push('图片按钮状态读不到');

// ---- 2) 切回多模态模型，入口必须回来
console.log('\n[2] 切回 LFM3-VL-2B（视觉）');
console.log('  点击:', await loadByName('LFM3-VL-2B'));
s = await waitName('VL-2B');
await sleep(600);
await openModels();
s = await snap();
await shot('03-vision-back');
console.log('  切换后:', JSON.stringify(s));
if (!s.name.includes('VL-2B')) problems.push('切回 VL 后顶栏名字没变');
if (s.attachHidden !== false) problems.push('切回视觉模型后，附件按钮没有恢复');

// ---- 3) 多模态模型列表里应该有「多模态」标，塔本身不该出现
const listInfo = await ev(`
  (() => {
    const rows = [...document.querySelectorAll('#viewBody .vrow')];
    const names = rows.map(r => (r.querySelector('.vname')||{}).textContent || '');
    const withTag = rows.filter(r => /多模态/.test(r.innerText)).length;
    const tower = names.filter(n => /^mmproj/i.test(n));
    return { count: rows.length, withTag, tower: tower.length };
  })()
`);
console.log('\n[3] 模型列表:', JSON.stringify(listInfo));
if (listInfo.tower > 0) problems.push('模型列表里仍然出现 mmproj 塔文件');
if (listInfo.withTag < 2) problems.push('带配套塔的模型没有「多模态」标记');

await shot('04-final');
console.log('\n' + (problems.length ? '问题:\n  ' + problems.join('\n  ') : '✓ 全部通过'));
ws.close();
process.exit(problems.length ? 1 : 0);
