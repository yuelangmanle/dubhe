#!/usr/bin/env python3
"""bench_best.py — 反复跑 dubhe bench，报**最好的一次** decode tok/s。

为什么不用均值：这台机器上有一个常驻负载（load average 稳定在 3.3），
decode 的实测值会在 ±10% 之间跳。均值会被偶发的坏样本拖低，而我们要比的是
"内核能跑到多快"——那是上界，不是平均表现。取最大值 + 多跑几次，是这种
噪声环境下唯一稳的估计方式。

用法：
  python3 tools/bench_best.py -m <model> -n 128 -r 3 -e DUBHE_ATTN_DIV=8
"""
import argparse
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-n", "--tokens", type=int, default=128)
    ap.add_argument("-r", "--rounds", type=int, default=3)
    ap.add_argument("-e", "--env", action="append", default=[],
                    help="透传给 dubhe 的环境变量，可重复")
    ap.add_argument("--bin", default=None)
    a = ap.parse_args()

    binp = a.bin or "build/dubhe"
    if not os.path.exists(binp):
        sys.exit(f"找不到 {binp}")

    env = dict(os.environ)
    for kv in a.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v

    best = None
    allv = []
    for i in range(a.rounds):
        p = subprocess.run([binp, "bench", "-m", a.model, "-n", str(a.tokens)],
                           capture_output=True, text=True, env=env)
        vals = [float(x) for x in re.findall(r"decode ([\d.]+) tok/s", p.stdout)]
        if not vals:
            print(p.stdout[-500:], p.stderr[-500:], file=sys.stderr)
            continue
        allv += vals
        best = max(vals) if best is None else max(best, max(vals))

    if best is None:
        sys.exit("没有解析到 decode 数值")
    print(f"best decode {best:.2f} tok/s   (样本 {len(allv)} 个，"
          f"范围 {min(allv):.1f}~{max(allv):.1f})")


if __name__ == "__main__":
    main()
