// ui_isolation.mjs — 三件"状态到底串不串"的事，在真浏览器里量
//
//   1. 不同对话会不会串记忆（用户原话："不同的对话，对串记忆"）
//   2. 新对话会不会继承项目文件夹（"像 codex 一样，开个新对话就绑定一个文件夹"）
//   3. 历史长了会不会压缩（"上下文系统没有压缩机制"）
//
// 为什么必须在浏览器里测：这三件全在客户端（localStorage + 请求体），
// curl 直接打 /v1/chat/completions 是看不到的 —— 服务端每次都只收到
// "调用方给的那几条消息"，串不串完全取决于前端拼了什么。
//
// 判据不看模型说了什么（1.7B 说的话不可靠），只看**请求体**：
//   - 新对话的 messages 里不能出现旧对话的内容
//   - workspace 字段必须等于继承来的项目文件夹
//   - 压缩后 localStorage 里要有 summary 且 folded 前进
//
// 用法：node tools/ui_isolation.mjs [url] [outdir]
import { mkdirSync, writeFileSync } from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:7792/';
const outdir = process.argv[3] || '/tmp/uiiso';
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
  }
});
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
const send = (method, params = {}) =>
  new Promise((res, rej) => {
    const mid = ++id;
    pending.set(mid, { res, rej });
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
const js = async (expr, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', {
    expression: expr, returnByValue: true, awaitPromise, userGesture: true,
  });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' :: ' + expr.slice(0, 70));
  return r.result.value;
};

await send('Page.enable');
await send('Runtime.enable');
// 在页面任何脚本之前挂 fetch 探针，刷新后依然有效
await send('Page.addScriptToEvaluateOnNewDocument', {
  source: `
    window.__reqs = [];
    const _f = window.fetch;
    window.fetch = function(u, o) {
      try {
        if (String(u).indexOf('/v1/chat/completions') >= 0 && o && o.body) {
          window.__reqs.push(JSON.parse(o.body));
        }
      } catch (e) {}
      return _f.apply(this, arguments);
    };
  `,
});
await send('Emulation.setFocusEmulationEnabled', { enabled: true });

async function boot() {
  await send('Page.setWebLifecycleState', { state: 'active' });
  await sleep(900);
}

async function sendMsg(text) {
  await js(`(() => {
    const ta = document.getElementById('input');
    ta.value = ${JSON.stringify(text)};
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('send').click();
    return 1;
  })()`);
  // 等这一轮结束（请求体已记录 + 不再 streaming）
  for (let i = 0; i < 80; i++) {
    await sleep(300);
    const done = await js(`(() => {
      const b = document.getElementById('send');
      return b && !b.classList.contains('stop') && !document.querySelector('.msg.typing');
    })()`);
    if (done && i > 3) break;
  }
}

// ---------------------------------------------------------------- 造场景
//
// 对话 A：绑定 /tmp 作为项目文件夹，带一句暗号。
const SECRET = '暗号是紫罗兰七号';
await js(`(() => {
  localStorage.setItem('dubhe.conv.v1', JSON.stringify([
    { id: 'cA', title: 'A', cwd: '/tmp', at: Date.now(), messages: [
      { role: 'user', content: ${JSON.stringify('请记住：' + SECRET)} },
      { role: 'assistant', content: '好的，我记住了。' },
    ]},
  ]));
  localStorage.removeItem('dubhe.memory.v1');
  return 1;
})()`);
await js(`location.reload()`);
await boot();

// ---------------------------------------------------------------- 1. 项目文件夹继承
const inherit = await js(`(() => {
  const before = JSON.parse(localStorage.getItem('dubhe.conv.v1') || '[]');
  document.getElementById('newChat').click();
  const after = JSON.parse(localStorage.getItem('dubhe.conv.v1') || '[]');
  return { before: before.length, after: after.length, cwd: after[0] && after[0].cwd };
})()`);
if (inherit.after !== inherit.before + 1) problems.push('新对话没有创建出来');
if (inherit.cwd !== '/tmp') {
  problems.push(`新对话没有继承项目文件夹 —— 期望 /tmp，实际 ${JSON.stringify(inherit.cwd)}`);
}

// ---------------------------------------------------------------- 2. 记忆不串
await js(`window.__reqs = []`);
await sendMsg('你好');
const reqB = await js(`window.__reqs.length ? window.__reqs[window.__reqs.length-1] : null`);
if (!reqB) problems.push('没抓到新对话的请求体');
else {
  const blob = JSON.stringify(reqB.messages || []);
  if (blob.includes(SECRET)) problems.push('新对话的请求里带着旧对话的暗号（串记忆）');
  if (reqB.workspace !== '/tmp') {
    problems.push(`请求里的 workspace 是 ${JSON.stringify(reqB.workspace)}，期望 /tmp`);
  }
}

// 在本对话点「记住」，再去另一个新对话看它会不会被带上
const remembered = await js(`(() => {
  const b = [...document.querySelectorAll('.msg .act-btn')].find(x => x.textContent === '记住');
  if (!b) return 'no-btn';
  b.click();
  return JSON.parse(localStorage.getItem('dubhe.memory.v1') || '[]').length;
})()`);
if (remembered === 'no-btn') problems.push('助手消息上没有「记住」按钮');
else if (!remembered) problems.push('点了「记住」但记忆库还是空的');

const memScope = await js(`(() => {
  const a = JSON.parse(localStorage.getItem('dubhe.memory.v1') || '[]');
  return a.length ? a[0].scope : null;
})()`);
if (memScope && !String(memScope).startsWith('conv:')) {
  problems.push(`本对话的「记住」写成了 ${memScope}，应该是 conv:<id>`);
}

// 换一个新对话，它的请求里不能出现上一条记忆
await js(`document.getElementById('newChat').click()`);
await js(`window.__reqs = []`);
await sendMsg('再说一句');
const reqC = await js(`window.__reqs.length ? window.__reqs[window.__reqs.length-1] : null`);
if (reqC) {
  const memText = await js(`(JSON.parse(localStorage.getItem('dubhe.memory.v1')||'[]')[0]||{}).text||''`);
  if (memText && JSON.stringify(reqC.messages || []).includes(memText.slice(0, 24))) {
    problems.push('另一个对话的请求里带上了本对话的记忆（串记忆）');
  }
}

// ---------------------------------------------------------------- 3. 上下文压缩
//
// 20 条历史 > COMPRESS_AT(16)，发出下一条时应该触发 /api/summarize。
await js(`(() => {
  const msgs = [];
  for (let i = 0; i < 10; i++) {
    msgs.push({ role: 'user', content: '第 ' + i + ' 个问题：' + '说明一下本地推理的某个方面。' });
    msgs.push({ role: 'assistant', content: '第 ' + i + ' 个回答：' + '本地推理就是把权重放在本机算。'.repeat(3) });
  }
  localStorage.setItem('dubhe.conv.v1', JSON.stringify([
    { id: 'cBig', title: '长对话', at: Date.now(), messages: msgs },
  ]));
  return msgs.length;
})()`);
await js(`location.reload()`);
await boot();
await sendMsg('继续');
await sleep(1500);
const comp = await js(`(() => {
  const c = (JSON.parse(localStorage.getItem('dubhe.conv.v1')||'[]')[0]) || {};
  return { folded: c.folded || 0, summaryLen: (c.summary || '').length, msgs: (c.messages||[]).length };
})()`);
if (!comp.folded) problems.push(`20 条历史没有触发压缩（folded=${comp.folded}）`);
else if (!comp.summaryLen) problems.push('压缩后 folded 前进了但 summary 是空的');

const shot = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(outdir + '/isolation.png', Buffer.from(shot.data, 'base64'));

console.log('1 项目文件夹继承:', JSON.stringify(inherit));
console.log('2 记忆:', JSON.stringify({ scope: memScope, reqWorkspace: reqB && reqB.workspace }));
console.log('3 压缩:', JSON.stringify(comp));
console.log('\n=== 问题 ===');
if (!problems.length) console.log('无');
else for (const p of problems) console.log('- ' + p);
ws.close();
