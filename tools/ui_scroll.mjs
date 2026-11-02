// ui_scroll.mjs — 专测三件"看代码看不出来、用起来才发现"的事
//
// 用户的反馈原话：
//   1) 模型回答的时候界面滑不上去，并且滑动回闪烁
//   2) 光标在 html 代码块里面不能上下滑动
//   3) 写出的 html 代码不能直接运行预览
//
// 这三条都不是"代码写错了"，而是滚动位置、焦点、DOM 重建之间的相互作用，
// 只能在真浏览器里量。脚本按这个顺序验：
//
//   A. 流式期间上滑：滚到离底 400px，等 1.5 秒，看 scrollTop 有没有被拽回去
//   B. 代码块内部滚动：注入一条带长代码块的消息（写 localStorage 再刷新，
//      走的是真正的渲染路径），把 pre 滚到底，看位置能不能保住
//   C. 运行预览：点 html 代码块上的"运行"，看有没有 iframe 真的拿到 srcdoc
//
// 用法：node tools/ui_scroll.mjs [url] [outdir]     （需要 9222 上的 Chrome）
import { mkdirSync, writeFileSync } from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:7792/';
const outdir = process.argv[3] || '/tmp/uiscroll';
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });

const list = await (await fetch(base + '/json')).json();
const page = list.find((t) => t.type === 'page');
if (!page) throw new Error('没有可用的页面');
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
  }
});
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
const send = (method, params = {}) =>
  new Promise((res, rej) => {
    const mid = ++id;
    pending.set(mid, { res, rej });
    ws.send(JSON.stringify({ id: mid, method, params }));
  });

async function js(expr, awaitPromise = false) {
  const r = await send('Runtime.evaluate', {
    expression: expr,
    returnByValue: true,
    awaitPromise,
    userGesture: true,
  });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' :: ' + expr.slice(0, 80));
  return r.result.value;
}

await send('Page.enable');
await send('Runtime.enable');
await send('Emulation.setFocusEmulationEnabled', { enabled: true });

// 无头 Chrome 里页面默认是 hidden，rAF 一帧都不跑 —— 流式渲染全靠 rAF，
// 不把生命周期设成 active 就只能看到最后一帧。
await send('Page.navigate', { url });
await sleep(1200);
await send('Page.setWebLifecycleState', { state: 'active' });
await sleep(600);

// ---------------------------------------------------------------- A. 流式上滑
//
// 先把历史消息塞满，让消息区一开始就是可滚动的 —— 否则 1.7B 写三百字
// 根本撑不满一屏，"滑不上去"这个场景压根构造不出来（第一次跑就是这里
// 报"一直没长高"）。塞完再真发一条，在它流式输出期间上滑。
await js(`(() => {
  const filler = Array.from({ length: 8 }, (_, i) =>
    '这是第 ' + (i + 1) + ' 段占位内容。' + '本地推理指的是把模型权重放在本机、用本机的算力生成 token。'.repeat(6)
  );
  const msgs = [];
  for (const f of filler) {
    msgs.push({ role: 'user', content: '继续' });
    msgs.push({ role: 'assistant', content: f });
  }
  localStorage.setItem('dubhe.conv.v1', JSON.stringify([{ id: 'cfill', title: '填充', at: Date.now(), messages: msgs }]));
  return 1;
})()`);
await js(`location.reload()`);
await sleep(1600);
await send('Page.setWebLifecycleState', { state: 'active' });
await sleep(400);

const seeded = await js(`(() => {
  const box = document.querySelector('#messages');
  return { h: box.scrollHeight, ch: box.clientHeight };
})()`);
if (seeded.h <= seeded.ch + 100) {
  problems.push(`A: 填充后仍然不可滚动（${seeded.h} / ${seeded.ch}）`);
}

// 真发一条，让它流式输出
await js(`(() => {
  const ta = document.getElementById('input');
  ta.value = '请详细讲一讲 Transformer 的结构，分十段写，每段至少三行。';
  ta.dispatchEvent(new Event('input', { bubbles: true }));
  document.getElementById('send').click();
  return 1;
})()`);

// 等"正在长高"，说明流已经在出字
let grew = false;
for (let i = 0; i < 80; i++) {
  await sleep(250);
  const h = await js(`(document.querySelector('#messages')||{}).scrollHeight || 0`);
  if (h > seeded.h + 40) { grew = true; break; }
}
if (!grew) problems.push('A: 发出后消息区没有继续长高，没能进入流式状态');

// 上滑 400px，等 1.5s，看有没有被拽回底部
const a = await js(`(async () => {
  const box = document.querySelector('#messages');
  box.scrollTop = Math.max(0, box.scrollHeight - box.clientHeight - 400);
  const want = box.scrollTop;
  await new Promise(r => setTimeout(r, 1500));
  return { want, now: box.scrollTop, h2: box.scrollHeight, ch: box.clientHeight };
})()`, true);
if (Math.abs(a.now - a.want) > 4) {
  problems.push(`A: 上滑后被拽回 —— 期望停在 ${a.want}，1.5 秒后变成 ${a.now}`);
}

// 滑回底部，确认跟随会自己恢复（否则用户以为"回不到最新"）
const a2 = await js(`(async () => {
  const box = document.querySelector('#messages');
  box.scrollTop = box.scrollHeight;
  await new Promise(r => setTimeout(r, 1200));
  return { gap: box.scrollHeight - box.scrollTop - box.clientHeight };
})()`, true);
if (a2.gap > 120) problems.push(`A: 回到底部后没有恢复跟随，离底 ${Math.round(a2.gap)}px`);

// 停止这一轮，免得它继续写
await js(`(() => { const b = document.querySelector('#stop') || document.querySelector('[data-stop]'); if (b) b.click(); return 1; })()`);
await sleep(400);

// ---------------------------------------------------------------- B/C. 代码块
//
// 走真渲染路径：把一条带 html 代码块的消息塞进 localStorage，刷新。
// 这样 B（内部滚动）和 C（运行预览）测的都是 renderMarkdown 的产物。
const code = Array.from({ length: 80 }, (_, i) => `<p>line ${i + 1}</p>`).join('\\n');
await js(`(() => {
  const conv = {
    id: 'cprobe', title: '探针', at: Date.now(),
    messages: [
      { role: 'user', content: '写个 html' },
      { role: 'assistant', content: '给你一个：\\n\\n\\u0060\\u0060\\u0060html\\n${code}\\n\\u0060\\u0060\\u0060\\n\\n完。' }
    ]
  };
  localStorage.setItem('dubhe.conv.v1', JSON.stringify([conv]));
  return 1;
})()`);
await js(`location.reload()`);
await sleep(1600);
await send('Page.setWebLifecycleState', { state: 'active' });

const b = await js(`(() => {
  const pre = document.querySelector('.body .codeblock pre');
  if (!pre) return { err: 'no-pre' };
  const st = getComputedStyle(pre);
  const scrollable = pre.scrollHeight > pre.clientHeight + 2;
  pre.focus();
  pre.scrollTop = pre.scrollHeight;
  const want = pre.scrollTop;
  return { scrollable, want, overflowY: st.overflowY, tabindex: pre.getAttribute('tabindex'),
           h: pre.scrollHeight, ch: pre.clientHeight };
})()`);
if (b.err) problems.push('B: 代码块没渲染出来');
else {
  if (!b.scrollable) problems.push(`B: 代码块不可滚动（scrollHeight ${b.h} / clientHeight ${b.ch}）`);
  if (b.want < 20) problems.push(`B: pre.scrollTop 设不下去（${b.want}）`);
  if (b.tabindex == null) problems.push('B: pre 没有 tabindex，方向键滚不到它');
  if (b.overflowY !== 'auto' && b.overflowY !== 'scroll') problems.push(`B: pre overflow-y=${b.overflowY}`);
}

// 滚完之后等一会儿，确认没有别的东西把它重置回顶部（流式重建、reswap 之类）
const b2 = await js(`(async () => {
  const pre = document.querySelector('.body .codeblock pre');
  const want = pre.scrollTop;
  await new Promise(r => setTimeout(r, 900));
  return { want, now: pre.scrollTop };
})()`, true);
if (b2.want > 20 && b2.now !== b2.want) {
  problems.push(`B: 代码块滚动位置被重置 —— ${b2.want} → ${b2.now}`);
}

// C. 运行预览
const c = await js(`(async () => {
  const btn = document.querySelector('.body .codeblock .cb-btn[data-run]');
  if (!btn) return { err: 'no-run-btn' };
  btn.click();
  await new Promise(r => setTimeout(r, 700));
  const iframe = document.querySelector('.pv-panel iframe');
  if (!iframe) return { err: 'no-iframe' };
  const panel = document.querySelector('.pv-panel');
  return {
    visible: getComputedStyle(panel).display !== 'none',
    sandbox: iframe.getAttribute('sandbox') || '',
    srcdocLen: (iframe.getAttribute('srcdoc') || iframe.srcdoc || '').length,
  };
})()`, true);
if (c.err) problems.push('C: 运行预览点不开（' + c.err + '）');
else {
  if (!c.visible) problems.push('C: 预览面板没显示');
  if (!c.srcdocLen) problems.push('C: iframe 没有拿到 srcdoc');
  if (c.sandbox.includes('allow-same-origin')) {
    problems.push('C: iframe 带了 allow-same-origin，预览页和主站同源，能碰 localStorage');
  }
}

const shot = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(outdir + '/scroll.png', Buffer.from(shot.data, 'base64'));

console.log('A 流式上滑:', JSON.stringify(a), '恢复跟随 gap=', Math.round(a2.gap));
console.log('B 代码块:', JSON.stringify(b), '保持:', JSON.stringify(b2));
console.log('C 预览:', JSON.stringify(c));
console.log('\n=== 问题 ===');
if (!problems.length) console.log('无');
else for (const p of problems) console.log('- ' + p);
ws.close();
