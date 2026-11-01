// ui_mm.mjs — 通过界面走一遍多模态：选图 → 预览 → 发送 → 看回复
//
// 为什么要在界面上测而不是直接 curl：附件这条路有三段容易断
//   1. 文件读成 data URL（FileReader）
//   2. 缩略图（原图几 MB，直接进 localStorage 会撑爆配额）
//   3. attachmentParts() 拼成 OpenAI 的 image_url 结构
// 任何一段断了 curl 都发现不了。
import { mkdirSync, writeFileSync, writeSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';

// 同步写日志。console.log 写管道是异步的，进程一旦被信号打断，
// 攒在缓冲里的输出会全部丢掉 —— 表现就是"一行都没有，只知道退出码"，
// 完全没法定位。调试脚本宁可慢一点也要保证说过的话能留下来。
const say = (...a) => writeSync(1, a.map(String).join(' ') + '\n');

const url = process.argv[2] || 'http://127.0.0.1:7788/';
const outdir = process.argv[4] || '/tmp/uimm';
// 调试端口可用 CDP_URL 覆盖。默认 9222；上一个无头 Chrome 没退干净、
// 端口还被占着的时候，换一个端口起新的就行，不用去翻进程表。
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(outdir, { recursive: true });
if (!process.argv[3]) throw new Error('用法: node tools/ui_mm.mjs <url> <图片路径> [outdir]');

// 必须给绝对路径。DOM.setFileInputFiles 的相对路径是按 Chrome 自己的
// 工作目录解析的，不是脚本的 cwd；给相对路径时这一步会卡住不返回，
// 整个脚本表现成"没有任何输出就死了"（踩过，查了很久）。
const img = resolvePath(process.argv[3]);

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

// 先清掉上一个测试留下的会话。不清的话历史消息会一起进 prompt ——
// ui_tools 刚问过"读一下 README.md 的第一行"，这里再发图，模型会接着
// 聊那个文件，完全不看图片，看起来像多模态坏了。
await ev(`localStorage.clear()`);
await send('Page.reload');
await sleep(2500);

// 视觉塔有没有加载，看界面上的按钮状态最直接（state 在闭包里取不到）
const enabled = await ev(`
  (() => { const b = document.getElementById('btnImage'); return b ? !b.disabled : 'missing'; })()
`);
say('图片按钮可用:', enabled);
if (enabled !== true) problems.push('加载了视觉塔但图片按钮仍不可用');

// 把文件塞进隐藏的 input，再触发 change。
//
// 先让 Chrome 别弹原生文件选择框。不拦的话，setFileInputFiles 之后渲染进程
// 会卡在那个对话框上 —— 页面看着还在，但之后所有 Runtime.evaluate 一律超时，
// 脚本表现成"没有任何输出就死了"，非常难查（这一条踩了很久）。
await send('Page.setInterceptFileChooserDialog', { enabled: true });
//
// getDocument 只要根节点（depth: 0）。用 depth: -1 会把整棵 DOM 序列化回来，
// 对话越长负载越大，没必要。querySelector 只认根节点 id。
const { root } = await send('DOM.getDocument', { depth: 0 });
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
say('附件条:', att);
if (/无附件条|^\s*$/.test(String(att))) problems.push('选完图后附件条没出现');

// 发送前先记下已有几条助手消息：之后只认"多出来的那条"。
const baseN = Number(await ev(`document.querySelectorAll('.msg.assistant').length`));

// 发送
await ev(`
  (() => {
    const ta = document.getElementById('input');
    ta.value = '这张图里是什么？用一句话说。';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('send').click();
  })()
`);
await sleep(1200);
await shot('02-streaming');

// 完成判据：多出一条助手气泡，且它的状态不再是「生成中…」。
//
// 不要靠文本里找关键词。之前用「尾部出现已完成」判断，每次都要等满
// 180 秒超时才退出 —— 完成标记是"已完成 · N tokens · X tok/s"分行渲染的，
// 按字符串切永远对不上。也不要靠消息区长度：用户那条一入列长度就涨了，
// 会在 1.8 秒就误判完成（也踩过）。状态行是应用自己维护的，最可靠。
let waited = 0;
let st = { n: 0, status: '', body: '' };
while (waited < 240000) {
  await sleep(600);
  waited += 600;
  st = await ev(`
    (() => {
      const list = [...document.querySelectorAll('.msg.assistant')];
      const last = list[list.length - 1];
      return {
        n: list.length,
        status: last ? ((last.querySelector('.msg-status') || {}).innerText || '') : '',
        body: last ? ((last.querySelector('.body') || {}).innerText || '') : '',
      };
    })()
  `);
  const fresh = st && st.n > baseN;
  const generating = st && /生成中/.test(st.status || '');
  if (fresh && !generating && (st.body || '').length > 5) break;
}
say('等待', (waited / 1000).toFixed(1), '秒 · 状态:', JSON.stringify(st.status || ''));
await shot('03-done');

const reply = String(st.body || '').trim();
say('回复:', reply.slice(0, 200));
if (reply.length < 10) problems.push('多模态回复为空');
if (/请求失败|⚠️/.test(reply)) problems.push('多模态请求失败: ' + reply.slice(0, 120));
if (/生成中/.test(String(st.status || ''))) problems.push('等了两分钟还在生成，可能卡住了');

writeFileSync(`${outdir}/problems.txt`, problems.join('\n'), 'utf8');
say('\n=== 问题 ===');
if (!problems.length) say('无');
else [...new Set(problems)].forEach((p) => say(' -', p));
ws.close();
