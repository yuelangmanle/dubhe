#!/usr/bin/env python3
"""工具调用收敛测试：一个请求里跑了几轮、最后有没有可读正文。

    python3 tools/toolconv.py "用 read_file 读一下 .gitignore，然后告诉我里面写了什么。"
    python3 tools/toolconv.py "..." 4          # 重复 4 次
    DUBHE_URL=http://127.0.0.1:7788 python3 tools/toolconv.py "..."

判据不是"回复里有没有关键词"，而是**服务端发了几次 tool_result** ——
那才是"工具真的执行了"的证据。模型自己编一段 <tool_result> 也能骗过关键词。

四种结局：
  收敛      执行了工具，且最后有正文
  重复调用  模型原地打转（服务端主动停下并给了提示）
  轮上限    到了 2 轮上限还在要工具
  没调工具  模型只是把计划说出来，一次都没调

temperature 默认 0：这个测试要的是可复现，不是多样性。同一个 prompt 在
temp=0 下每次行为都不一样，那本身就是要查的问题。
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("DUBHE_URL", "http://127.0.0.1:7788")


def once(prompt, max_tokens=180, temp=0.0):
    body = {
        "model": "x", "max_tokens": max_tokens, "stream": True,
        "temperature": temp, "dubhe_tools": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    text = ""
    events = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            if "dubhe" in j:
                events.append(j["dubhe"].get("type", ""))
                continue
            delta = j.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                text += delta["content"]
    return time.time() - t0, events, text


def classify(events, text):
    if not events.count("tool_result"):
        return "没调工具"
    if "重复调用" in text:
        return "重复调用"
    if "轮上限" in text:
        return "轮上限"
    return "收敛"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "用 read_file 读一下 .gitignore，然后告诉我里面写了什么。"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    tally = {}
    for i in range(n):
        dur, events, text = once(prompt)
        kind = classify(events, text)
        tally[kind] = tally.get(kind, 0) + 1
        print("%d) %.1fs  tool_call=%d tool_result=%d 正文 %d 字 → %s" % (
            i + 1, dur, events.count("tool_call"), events.count("tool_result"),
            len(text), kind))
        print("    ", repr(text[:120]))

    print("\n汇总:", " · ".join("%s %d" % (k, v) for k, v in tally.items()),
          "（共 %d）" % n)
    return 0 if tally.get("收敛") == n else 1


if __name__ == "__main__":
    sys.exit(main())
