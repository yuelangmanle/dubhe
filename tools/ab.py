#!/usr/bin/env python3
"""ab.py — 交错 A/B(/C) 基准，专门对付"同一台机器几分钟内漂 2 倍"

为什么需要这个工具
------------------
M 系列 GPU 有 DVFS，本机 load 也不可控。我们踩过两次：

  * 跨会话对比 prefill，得出"比 MLX 慢 7.5 倍"，重测是 3 倍；
  * 裸内核微基准说新内核快 4.1 倍，塞进真实模型只快 1.18 倍。

两个结论都错在**测量方法**上，不在代码上。所以：所有对比必须是
**同一时间窗内交替执行**、每轮各跑一次、取每边的最小值。单边跑完
再跑另一边的数据一律作废。

用法
----
    # Dubhe 两个二进制对比（默认就是这种用法）
    python tools/ab.py --dubhe old=build/dubhe-old --dubhe new=build/dubhe

    # Dubhe vs MLX（MLX 需要装了 mlx-lm 的 python）
    python tools/ab.py -p 4 -n 32 \
        --dubhe dubhe=build/dubhe:$HOME/.dubhe/models/lfm3-1.7b/LFM3-1.7B-Q4_K_M.gguf \
        --mlx mlx4=~/.dubhe/models/lfm3-mlx-4bit

    # 只测 prefill 长度的影响
    python tools/ab.py --prompt long --tokens 8

输出：每轮原始值 + 每边最小值 + 相对第一边的倍数。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {
    "short": "The capital of France is",
    "long": "The history of science is a long story of careful measurement and bold conjecture. " * 40,
    # 中文：分词器对非 ASCII 的切分与英文完全不同，长度不能按字符估
    "zh": "天枢是中国古代星官名，指北斗七星的第一颗。下面谈谈本地大模型推理引擎的设计：",
}
STAT_RE = re.compile(
    r"prompt (\d+) tok / ([\d.]+) ms \(([\d.]+) tok/s\) \| gen (\d+) tok / ([\d.]+) ms \(([\d.]+) tok/s\)")


def run_dubhe(spec: str, prompt: str, n: int) -> dict:
    """spec = 标签=二进制路径[:模型路径]"""
    label, _, rest = spec.partition("=")
    binary, _, model = rest.partition(":")
    cmd = [str(Path(binary).expanduser()), "run", "-m", str(Path(model).expanduser()),
           "-p", prompt, "--raw", "-n", str(n), "--temp", "0", "--stats"]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    out = p.stdout + p.stderr
    m = STAT_RE.search(out)
    if not m:
        tail = (p.stderr or out).strip().splitlines()[-1:]
        raise SystemExit(f"[{label}] 没有统计行：{tail}")
    return {"label": label, "prefill_ms": float(m.group(2)), "decode_ms": float(m.group(5)),
            "prefill_tps": float(m.group(3)), "decode_tps": float(m.group(6))}


def run_mlx(spec: str, prompt: str, n: int, python: str) -> dict:
    """spec = 标签=模型目录"""
    label, _, rest = spec.partition("=")
    script = ROOT / "tools" / "mlx_lfm3.py"
    p = subprocess.run([python, str(script), str(Path(rest).expanduser()), prompt, str(n)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    i = p.stdout.rfind('{"tokens_in"')
    if i < 0:
        tail = (p.stderr or "").strip().splitlines()[-1:]
        raise SystemExit(f"[{label}] MLX 失败：{tail}")
    d = json.loads(p.stdout[i:].splitlines()[0])
    return {"label": label, "prefill_ms": 0.0, "decode_ms": 0.0,
            "prefill_tps": d["prefill_tps"], "decode_tps": d["decode_tps"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dubhe", action="append", default=[], metavar="LABEL=BIN:MODEL")
    ap.add_argument("--mlx", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--mlx-python", default=os.path.expanduser("~/.dubhe/venv-mlx/bin/python"))
    ap.add_argument("--prompt", default="long", choices=sorted(PROMPTS))
    ap.add_argument("-n", "--tokens", type=int, default=32)
    ap.add_argument("-p", "--rounds", type=int, default=5)
    args = ap.parse_args()

    if not args.dubhe and not args.mlx:
        ap.error("至少要给一个 --dubhe 或 --mlx")

    prompt = PROMPTS[args.prompt]
    cases = [("dubhe", s) for s in args.dubhe] + [("mlx", s) for s in args.mlx]
    hist: dict[str, list[dict]] = {}

    # 预热：第一次跑会付 page-in 权重的代价，不进统计
    for kind, spec in cases:
        label = spec.split("=")[0]
        print(f"  预热 {label} ...", flush=True)
        try:
            (run_dubhe if kind == "dubhe" else
             (lambda s, p_, n_: run_mlx(s, p_, n_, args.mlx_python)))(spec, prompt, args.tokens)
        except SystemExit as e:
            print(f"  {e}", flush=True)

    for r in range(args.rounds):
        for kind, spec in cases:
            label = spec.split("=")[0]
            try:
                if kind == "dubhe":
                    d = run_dubhe(spec, prompt, args.tokens)
                else:
                    d = run_mlx(spec, prompt, args.tokens, args.mlx_python)
            except SystemExit as e:
                print(f"  轮 {r+1} {e}", flush=True)
                continue
            hist.setdefault(label, []).append(d)
            print(f"  轮 {r+1}  {label:12s} prefill {d['prefill_tps']:7.1f} tok/s"
                  f"   decode {d['decode_tps']:6.1f} tok/s", flush=True)

    # 逐轮配对：每轮内把各目标除以基准目标，再取**比值的中位数**。
    # 为什么不能"各自取最小值再相除"：min 是对 prefill 和 decode 分别取的，
    # 会把不同轮次混在一起，一个目标可能在 prefill 上抽到好轮、在 decode 上
    # 抽到坏轮，比出来的倍数纯属拼凑。配对后机器负载在分子分母上同时出现，
    # 大部分被约掉，这是本机（load 常年 10 以上）唯一可信的口径。
    print(f"\n### 交错 A/B（prompt={args.prompt}, n={args.tokens}, {args.rounds} 轮配对）\n")
    print("| 目标 | prefill | decode | prefill 相对 | decode 相对 |")
    print("|---|---:|---:|---:|---:|")
    labels = [s.split("=")[0] for _, s in cases]
    base_label = next((l for l in labels if hist.get(l)), None)
    for kind, spec in cases:
        label = spec.split("=")[0]
        rows = hist.get(label)
        if not rows:
            print(f"| {label} | — | — | _失败_ | — |")
            continue
        bp = max(x["prefill_tps"] for x in rows)
        bd = max(x["decode_tps"] for x in rows)
        if label == base_label:
            print(f"| **{label}** | {bp:.1f} tok/s | {bd:.1f} tok/s | 1.00x | 1.00x |")
            continue
        pr, dr = [], []
        for i in range(min(len(rows), len(hist[base_label]))):
            b = hist[base_label][i]
            if b["prefill_tps"] > 0 and b["decode_tps"] > 0:
                pr.append(rows[i]["prefill_tps"] / b["prefill_tps"])
                dr.append(rows[i]["decode_tps"] / b["decode_tps"])
        pm = statistics.median(pr) if pr else float("nan")
        dm = statistics.median(dr) if dr else float("nan")
        print(f"| {label} | {bp:.1f} tok/s | {bd:.1f} tok/s | "
              f"**{pm:.2f}x** | {dm:.2f}x |")

    # 内部对照。decode 走的是另一套内核（mv_*），如果本轮改动只碰 prefill，
    # decode 的中位比值就必须落在 1.00 附近；它偏离多少，就说明这一轮还有
    # 多少漂移没被配对约掉，prefill 的倍数要按同样比例打折看待。
    print("\n内部对照：decode 中位比值应≈1.00x，偏离即残余漂移。")

    # 离散度：如果同一边的最大/最小差得太多，说明这一轮环境不稳，结论要打折
    print("\n各目标本轮内极差（min→max，反映环境噪声）：")
    for kind, spec in cases:
        label = spec.split("=")[0]
        rows = hist.get(label)
        if not rows:
            continue
        ps = [x["prefill_tps"] for x in rows]
        ds = [x["decode_tps"] for x in rows]
        print(f"- {label}: prefill {min(ps):.1f}→{max(ps):.1f}"
              f"（{max(ps)/max(min(ps),1e-9):.2f}x）  "
              f"decode {min(ds):.1f}→{max(ds):.1f}"
              f"（{max(ds)/max(min(ds),1e-9):.2f}x）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
