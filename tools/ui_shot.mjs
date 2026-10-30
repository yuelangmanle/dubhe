// ui_shot.mjs — 用 CDP 驱动无头 Chrome：发送一条消息并截图
// 用法: node tools/ui_shot.mjs <url> <out.png> [prompt]
const url = process.argv[2] || 'http://127.0.0.1:7777/';
const out = process.argv[3] || '/tmp/ui.png';
const prompt = process.argv[4] || '用三句话解释什么是本地推理引擎';

// 调试端口可用 CDP_URL 覆盖。默认 9222；上一个无头 Chrome 没退干净、
// 端口还被占着的时候，换一个端口起新的就行，不用去翻进程表。
const base = process.env.CDP_URL || 'http://127.0.0.1:9222';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(base + '/json')).json();
      const page = list.find((t) => t.type === 'page');
      if (page && page.webSocketDebuggerUrl) return page;
    } catch {}
    await sleep(250);
  }
  throw new Error('无法连接 Chrome 调试端口');
}

const t = await target();
const ws = new WebSocket(t.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();

ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});

await new Promise((r) => ws.addEventListener('open', r));

function send(method, params = {}) {
  const i = ++id;
  return new Promise((res, rej) => {
    pending.set(i, (m) => (m.error ? rej(new Error(m.error.message)) : res(m.result)));
    ws.send(JSON.stringify({ id: i, method, params }));
  });
}

await send('Page.enable');
await send('Runtime.enable');
const W = parseInt(process.env.SHOT_W || '0', 10);
const H = parseInt(process.env.SHOT_H || '0', 10);
if (W && H) {
  await send('Emulation.setDeviceMetricsOverride', {
    width: W, height: H, deviceScaleFactor: 2, mobile: W < 600,
  });
}
await send('Page.navigate', { url });
await sleep(1500);

const ev = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  return r.result && r.result.value;
};

// 关掉欢迎页，直接输入并发送
if (prompt === '--api' || prompt === '--settings') {
  await ev(`document.getElementById('${prompt === '--api' ? 'openApi' : 'openSettings'}').click()`);
  await sleep(700);
} else {
  await ev(`(() => {
    const i = document.getElementById('input');
    i.value = ${JSON.stringify(prompt)};
    i.dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('send').click();
    return true;
  })()`);

  // 等生成结束（发送按钮从 busy 状态恢复，或最多 40 秒）
  for (let i = 0; i < 80; i++) {
    await sleep(500);
    const done = await ev(`(() => {
      const m = document.querySelectorAll('.msg.assistant').length;
      const b = document.getElementById('send');
      return m > 0 && !b.classList.contains('busy') && !document.querySelector('.typing');
    })()`);
    if (done) break;
  }
}
await sleep(600);

const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
const fs = await import('node:fs');
fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
console.log('saved', out);

const errs = await ev(`window.__errs ? window.__errs.join('\\n') : ''`);
if (errs) console.log('页面错误:', errs);
ws.close();
process.exit(0);
