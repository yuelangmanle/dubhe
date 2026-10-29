#!/usr/bin/env python3
"""sweep_mm.py — 分块参数扫描器

MSL 内核是**字符串**，C++ 的 `-D` 传不进去；每个变体必须改头文件重编译。
所以扫描脚本做三件事：改 `#define`、同步 C++ 侧真源、重编译 `mm2_check`、
跑裸内核基准。改完一定还原（try/finally），否则会把半改的文件留在树上。

用法：
    python tools/sweep_mm.py                    # 默认扫描表
    python tools/sweep_mm.py --only "8,4"       # 只跑指定组合
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HDR = ROOT / "engine" / "src" / "metal_kernels.h"
BENCH = ROOT / "build" / "mm2_check"

# (MM_SG_PER_TG, MM_SG, BMAX) —— MMR 由前两者相乘得到
DEFAULT = [(8, 4, 8), (4, 4, 8), (2, 4, 8),
           (16, 2, 16), (8, 2, 16), (4, 2, 16), (8, 4, 16),
           (32, 1, 32), (16, 1, 32), (16, 2, 32), (8, 2, 32)]


def patch(sg_per_tg: int, sg: int, bmax: int) -> None:
    t = HDR.read_text()
    t2 = re.sub(r"#define MM_SG_PER_TG \d+", f"#define MM_SG_PER_TG {sg_per_tg}", t, count=1)
    t2 = re.sub(r"#define MM_SG +\d+", f"#define MM_SG        {sg}", t2, count=1)
    t2 = re.sub(r"#define BMAX \d+", f"#define BMAX {bmax}", t2, count=1)
    t2 = re.sub(r"constexpr int kBmax      = \d+", f"constexpr int kBmax      = {bmax}", t2, count=1)
    t2 = re.sub(r"constexpr int kMmSgPerTg = \d+", f"constexpr int kMmSgPerTg = {sg_per_tg}", t2, count=1)
    t2 = re.sub(r"constexpr int kMmRows    = \d+",
                f"constexpr int kMmRows    = {sg_per_tg * sg}", t2, count=1)
    # 注意：第一个组合往往就是文件里的现值，此时 t2 == t 是**正常**的，
    # 不能当成失败（第一版就是栽在这个判断上）。所以只校验"改完之后确实对"。
    for pat, want in ((r"#define MM_SG_PER_TG (\d+)", sg_per_tg),
                      (r"#define MM_SG +(\d+)", sg),
                      (r"#define BMAX (\d+)", bmax),
                      (r"constexpr int kBmax      = (\d+)", bmax),
                      (r"constexpr int kMmSgPerTg = (\d+)", sg_per_tg),
                      (r"constexpr int kMmRows    = (\d+)", sg_per_tg * sg)):
        m = re.search(pat, t2)
        if not m or int(m.group(1)) != want:
            raise SystemExit(f"补丁没有生效：{pat} 期望 {want}，实际 {m.group(1) if m else '无'}")
    HDR.write_text(t2)


def build() -> bool:
    p = subprocess.run(["make", "-C", str(ROOT), "mm2"], capture_output=True, text=True)
    if p.returncode != 0:
        print("    编译失败：" + (p.stderr or p.stdout).strip().splitlines()[-1][:160])
        return False
    return True


def bench() -> float | None:
    p = subprocess.run([str(BENCH), "--bench"], capture_output=True, text=True)
    m = re.search(r"一层\s+[\d.]+ ms\s+([\d.]+) GFLOP/s", p.stdout)
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="形如 '8,4;4,8'")
    args = ap.parse_args()
    combos = DEFAULT
    if args.only:
        combos = [tuple(int(x) for x in c.split(",")) for c in args.only.split(";")]

    orig = HDR.read_text()
    results = []
    try:
        for spt, sg, bm in combos:
            patch(spt, sg, bm)
            if not build():
                results.append((spt, sg, bm, spt * sg, None))
                continue
            g = bench()
            results.append((spt, sg, bm, spt * sg, g))
            print(f"  SG_PER_TG={spt:2d} MM_SG={sg:2d} MMR={spt*sg:3d} BMAX={bm:2d}  "
                  f"{g if g else float('nan'):7.1f} GFLOP/s", flush=True)
    finally:
        HDR.write_text(orig)
        subprocess.run(["make", "-C", str(ROOT), "mm2"], capture_output=True, text=True)
        print("\n已还原头文件并重新编译。")

    ok = [r for r in results if r[4]]
    if ok:
        best = max(ok, key=lambda r: r[4])
        print(f"\n最佳：SG_PER_TG={best[0]} MM_SG={best[1]} MMR={best[3]} BMAX={best[2]} "
              f"→ {best[4]:.1f} GFLOP/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
