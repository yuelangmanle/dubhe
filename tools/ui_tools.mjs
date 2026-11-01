// ui_tools.mjs — 工具开关的界面回归
//
// 为什么单独一个脚本：这里的 bug 不是"报错"，是"默认值把普通对话带坏了"。
// 2B 级模型带着工具说明问"今天天气怎么样"，回的是"我需要调用工具来获取
// 实时天气数据" —— 界面一切正常，用户看到的是一个坏掉的助手。
//
// 盯三件事：
//   1. 新用户进来，工具按钮是灭的（默认关闭）
//   2. 普通对话的回复里不出现"讲计划 / 编造工具结果"的痕迹
//   3. Agent 视图入口处写明了限制，而不是假装模型能自己决定
//
// 用法：node tools/ui_tools.mjs [url] [outdir]
import { mkdirSync, writeFileSync, writeSync } from 'node:fs';

const say = (...a) => writeSync(1, a.map(String).join(' ') + '\n');
const url = process.argv[2] || 'http://127.0.0.1:7788/';
const outdir = process.argv[3] || '/tmp/uitools';
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });

const list = await (await fetch(base + '/json')).json();
const page = list.find((t) => t.type === 'page');
if (!page?.webSocketDebuggerUrl) throw new Error('没有可用的调试页面');

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
    problems.push('未捕获异常: ' + (m.params.exceptionDetails.exception?.description || ''));
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
async function ev(expr) {
  const r = await send('Runtime.evaluate', {
    expression: expr, awaitPromise: true, returnByValue: true, userGesture: true,
  });
  if (r.exceptionDetails) throw new Error('eval 失败: ' + (r.exceptionDetails.exception?.description || ''));
  return r.result.value;
}
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${outdir}/${name}.png`, Buffer.from(r.data, 'base64'));
}

await send('Runtime.enable');
await send('Page.enable');

// ---- 1. 模拟新用户：清掉本地存储再加载
await send('Page.navigate', { url });
await sleep(1200);
await ev(`localStorage.clear()`);
await send('Page.reload', { ignoreCache: true });
await sleep(1800);

const btnOn = await ev(`(document.getElementById('btnTools')||{}).classList?.contains('on')`);
say('工具按钮默认亮着:', btnOn);
if (btnOn) problems.push('新用户进来工具按钮就是亮的，默认值没生效');

// ---- 2. 普通对话不该被工具说明污染
await ev(`
  (() => {
    const ta = document.getElementById('input');
    ta.value = '今天天气怎么样？';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('send').click();
  })()
`);

// 等流式结束：盯 .msg-status 里还有没有"生成中"
let waited = 0;
let done = false;
while (waited < 90000) {
  await sleep(500);
  waited += 500;
  const st = await ev(`
    (() => {
      const b = [...document.querySelectorAll('.msg.assistant')];
      if (!b.length) return { n: 0, busy: true };
      const last = b[b.length - 1];
      return { n: 1, busy: /生成中/.test(last.innerText || '') };
    })()
  `);
  if (st.n && !st.busy) { done = true; break; }
}
say('等待', (waited / 1000).toFixed(1), '秒', done ? '' : '(超时)');

const reply = await ev(`
  (() => {
    const b = [...document.querySelectorAll('.msg.assistant .body')];
    return b.length ? b[b.length - 1].innerText.trim() : '';
  })()
`);
say('回复:', JSON.stringify(String(reply).slice(0, 120)));
if (!String(reply).trim()) problems.push('普通对话回复为空');
if (/我需要调用|我将调用|我需要使用.*工具|我将使用.*工具/.test(String(reply)))
  problems.push('普通对话在讲工具计划，工具说明没关干净');
if (/<tool_result>|<tool_call>/.test(String(reply)))
  problems.push('普通对话出现工具标签');
await shot('01-plain-chat');

// ---- 3. Agent 视图必须写明限制
await ev(`document.querySelector('.nav-item[data-view="agent"]').click()`);
await sleep(700);
const agentText = await ev(`(document.getElementById('viewBody')||{}).innerText || ''`);
say('Agent 视图文本长度:', String(agentText).length);
say('  ', JSON.stringify(String(agentText).slice(0, 160)));
if (!/不可靠|直接试运行|不依赖模型/.test(String(agentText)))
  problems.push('Agent 视图没有说明工具调用的能力边界');
await shot('02-agent-view');

// ---- 4. 从 Agent 发一条：带工具，但不强求模型一定调得动
await ev(`
  (() => {
    const ta = document.querySelector('#viewBody textarea');
    ta.value = '读一下 README.md 的第一行，告诉我写了什么。';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('#viewBody .vbtn')].find(b => /在对话中执行/.test(b.textContent)).click();
  })()
`);
waited = 0;
done = false;
while (waited < 120000) {
  await sleep(500);
  waited += 500;
  const st = await ev(`
    (() => {
      const b = [...document.querySelectorAll('.msg.assistant')];
      if (!b.length) return { n: 0, busy: true };
      return { n: 1, busy: /生成中/.test(b[b.length - 1].innerText || '') };
    })()
  `);
  if (st.n && !st.busy) { done = true; break; }
}
say('Agent 轮等待', (waited / 1000).toFixed(1), '秒', done ? '' : '(超时)');

const agentReply = await ev(`
  (() => {
    const b = [...document.querySelectorAll('.msg.assistant')];
    if (!b.length) return { body: '', toolCards: 0 };
    const last = b[b.length - 1];
    return {
      body: (last.querySelector('.body') || {}).innerText || '',
      toolCards: last.querySelectorAll('.tool-card, .toolcall, .tool').length,
    };
  })()
`);
say('Agent 回复:', JSON.stringify(String(agentReply.body).slice(0, 140)));
say('  工具卡片数:', agentReply.toolCards);
await shot('03-agent-reply');

// 这一轮不判失败：2B 模型调不调得动工具是能力问题，不是接线问题。
// 只要没有静默的空气泡、没有未捕获异常，接线就算通。
if (!String(agentReply.body).trim() && !agentReply.toolCards)
  problems.push('Agent 轮既没有正文也没有工具卡片 —— 空气泡');

// ---- 5. 跑完 Agent 之后，普通对话不该被永久带上工具
const btnAfter = await ev(`(document.getElementById('btnTools')||{}).classList?.contains('on')`);
say('Agent 跑完后工具按钮:', btnAfter ? '亮' : '灭');
if (btnAfter) problems.push('Agent 的一次性动作污染了长期开关');

say('');
if (problems.length) {
  say('问题:');
  for (const p of problems) say('  -', p);
  process.exitCode = 1;
} else {
  say('工具开关回归通过');
}
ws.close();
