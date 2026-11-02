#!/usr/bin/env python3
"""concurrent_latency.py — 两个/三个客户端同时提问时，时间到底花在哪。

为什么单独一个脚本：`concurrent_check.py` 只回答"会不会串台"，它报的
"串行应≈ N ms"是把各个请求自己的耗时加起来，而那个耗时**已经包含排队等待**
—— 加出来的数字没有意义（实测 2236 ms 墙钟 vs "串行应 5135 ms"，看着像
真并发了，其实只是把等待时间重复计了两次）。

这个脚本按事件打点，把一次请求拆成四段：

    t0  发出
    t1  收到 SSE 头（服务端 accept 了）
    t2  收到 queued 事件（引擎正忙，我进队列了）
    t3  收到第一个正文 token（真正开始出字）
    t4  结束

于是能回答三个问题：
  1. 请求是不是排队的？（t2 有没有出现）
  2. 排队等了多久？（t3 - t2）
  3. 用户看到的"卡住"有多长？（t3 - t0，这段时间界面上只有"排队中…"）

用法：python3 tools/concurrent_latency.py --port 7792 --n 3
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "用一句话说明什么是 KV cache。",
    "用一句话说明什么是量化。",
    "用一句话说明什么是 tokenizer。",
    "用一句话说明什么是注意力。",
]


def one(port: int, prompt: str, idx: int, out: list) -> None:
    body = {
        "model": "dubhe",
        "stream": True,
        "max_tokens": 60,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    rec = {"idx": idx, "t0": time.time(), "t1": None, "t2": None, "t3": None, "t4": None,
           "text": "", "err": None}
    out.append(rec)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            rec["t1"] = time.time()
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                d = j.get("dubhe")
                if d:
                    if d.get("type") == "queued" and rec["t2"] is None:
                        rec["t2"] = time.time()
                    elif d.get("type") == "error":
                        rec["err"] = d.get("message")
                ch = j.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content"):
                    if rec["t3"] is None:
                        rec["t3"] = time.time()
                    rec["text"] += ch[0]["delta"]["content"]
    except Exception as e:  # noqa: BLE001
        rec["err"] = str(e)
    rec["t4"] = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7792)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    out: list = []
    t_start = time.time()
    threads = [
        threading.Thread(target=one, args=(args.port, PROMPTS[i % len(PROMPTS)], i, out))
        for i in range(args.n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.time() - t_start) * 1000

    print(f"{args.n} 个请求同时发出，墙钟 {wall:.0f} ms\n")
    print(f"{'#':>2} {'头':>8} {'排队':>8} {'首字':>8} {'总':>8}  {'空等':>8}  回答")
    work = 0.0
    for r in sorted(out, key=lambda x: x["idx"]):
        t0 = r["t0"]
        head = (r["t1"] - t0) * 1000 if r["t1"] else -1
        queued = (r["t2"] - t0) * 1000 if r["t2"] else None
        ttft = (r["t3"] - t0) * 1000 if r["t3"] else -1
        total = (r["t4"] - t0) * 1000
        gen = (r["t4"] - r["t3"]) * 1000 if r["t3"] else 0
        work += gen
        q = f"{queued:.0f}" if queued is not None else "—"
        print(f"{r['idx']:>2} {head:>8.0f} {q:>8} {ttft:>8.0f} {total:>8.0f}  {ttft:>8.0f}  "
              f"{(r['text'] or r['err'] or '')[:34]}")

    print(f"\n真正在生成的时间合计 {work:.0f} ms / 墙钟 {wall:.0f} ms"
          f" → 并发度 {work / wall:.2f}x")
    print("并发度 ≈ 1.0 表示完全串行（引擎只有一套 KV，这是预期）；"
          "> 1 说明多个请求真的在同时解码。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
