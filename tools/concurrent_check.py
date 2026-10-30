#!/usr/bin/env python3
"""并发一致性测试：同时发 N 个不同 prompt，检查回答有没有"串台"。

背景
----
http::Server 是"一个连接一个线程"，而引擎只有一套 KV cache。担心的是两个
客户端同时请求会互相踩 KV，表现成乱码或崩溃，而且**单客户端永远复现不了**。

实测结论（2026-10-30）：**不会**。engine.cpp 的 dubhe_generate 一进来就
lock_guard(e->mu)、一直持到出结果，开头还会 reset_kv()，所以生成天然是
串行的。改动前后的二进制跑这个脚本都通过。

那这个脚本还留着干什么：它是这条性质的**回归测试**。哪天有人把
dubhe_generate 里的锁挪走、或者改成"锁只包住首 token"，串台会立刻出现在
这里，而不是等用户报"模型胡言乱语"。

判据
----
不是"返回 200"，而是**回答对得上自己的问题、且没有混进别人的内容**：

  1. 回答里必须出现自己那个国家名（串台时最先丢的就是它）；
  2. 回答里**不能**出现其它 case 的首都名（串台时最先混进来的就是它）。

顺带看"墙钟 vs 串行之和"：两者接近说明请求是排队的（预期行为），
墙钟远小于串行之和说明真的并发了 —— 那时候才需要担心 KV。

用法：
  python3 tools/concurrent_check.py --port 7788 --n 3 --rounds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# (问题, 自己的关键词, 别人的首都 —— 出现任意一个就算串台)
CASES = [
    ("法国的首都是哪里？", ["法国", "France"], ["东京", "罗马", "堪培拉", "渥太华"]),
    ("日本的首都是哪里？", ["日本", "Japan"], ["巴黎", "罗马", "堪培拉", "渥太华"]),
    ("意大利的首都是哪里？", ["意大利", "Italy"], ["巴黎", "东京", "堪培拉", "渥太华"]),
    ("澳大利亚的首都是哪里？", ["澳大利亚", "Australia"], ["巴黎", "东京", "罗马", "渥太华"]),
    ("加拿大的首都是哪里？", ["加拿大", "Canada"], ["巴黎", "东京", "罗马", "堪培拉"]),
]


def ask(port: int, idx: int, max_tokens: int) -> dict:
    prompt = CASES[idx][0]
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        return {"idx": idx, "ok": True,
                "text": d["choices"][0]["message"]["content"],
                "ms": (time.time() - t0) * 1000}
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "ok": False, "text": f"<{e}>", "ms": (time.time() - t0) * 1000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()

    n = max(2, min(a.n, len(CASES)))
    fails = 0
    for rd in range(a.rounds):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(lambda i: ask(a.port, i, a.max_tokens), range(n)))
        wall = (time.time() - t0) * 1000
        print(f"--- 第 {rd+1} 轮：{n} 个并发，墙钟 {wall:.0f} ms"
              f"（串行应≈ {sum(r['ms'] for r in res):.0f} ms）---")
        for r in res:
            _, own, foreign = CASES[r["idx"]]
            has_own = any(k in r["text"] for k in own)
            crosstalk = [k for k in foreign if k in r["text"]]
            good = r["ok"] and has_own and not crosstalk
            if not good:
                fails += 1
            why = ""
            if not r["ok"]:
                why = "请求失败"
            elif not has_own:
                why = "丢了本case关键词"
            elif crosstalk:
                why = f"串台: {crosstalk}"
            print(f"  {'OK  ' if good else 'FAIL'} [{r['idx']}] {r['ms']:6.0f} ms"
                  f"  {r['text'][:64]!r}{('  ← ' + why) if why else ''}")
    print()
    print("全部通过" if fails == 0 else f"**{fails} 个失败**")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
