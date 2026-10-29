#!/usr/bin/env python3
"""audio_diff.py — 逐层比较 C++ 语音塔和 HF oracle

用法:
    python3 audio_diff.py --cpp /tmp/aud_cpp --oracle /tmp/aud_oracle

两边的形状约定不一样，这里负责摊平比较：C++ 存的是 (token × dim) 的行主序，
oracle 的 .npy 也是 (token, dim)，直接按元素比。
mel 是 (128, T) 和 (128, T)，也一样。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

PAIRS = [("mel", "mel"), ("down", "down")] + \
        [(f"blk{i}", f"blk{i}") for i in range(17)] + \
        [("post_ln", "post_ln"), ("mm", "mm")]


def cmp(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size:
        return None
    d = a - b
    mx = np.abs(d).max()
    scale = max(np.abs(b).max(), 1e-12)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(a @ b / denom) if denom > 0 else 1.0
    return mx, mx / scale, cos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--tol", type=float, default=2e-3, help="相对误差阈值")
    a = ap.parse_args()

    bad = 0
    print(f"{'检查点':<10} {'形状':<16} {'max|Δ|':>11} {'相对':>10} {'cos':>10}")
    print("-" * 62)
    for name, oname in PAIRS:
        pf = os.path.join(a.cpp, name + ".f32")
        pn = os.path.join(a.oracle, oname + ".npy")
        if not os.path.exists(pf) or not os.path.exists(pn):
            print(f"{name:<10} {'缺失':<16}")
            continue
        x = np.fromfile(pf, dtype="<f4")
        y = np.load(pn).astype(np.float32)
        # 降采样那一步在 HF 里是 (通道, 时间)，我这里存的是 (token, 通道) ——
        # encoder 的 .transpose(1,2) 发生在 downsample 之后，所以这里转置一下
        if name == "down" and y.ndim == 2:
            y = np.ascontiguousarray(y.T)
        if x.size == y.size:
            x = x.reshape(y.shape)
        r = cmp(x, y)
        if r is None:
            print(f"{name:<10} 形状不符 {x.size} vs {y.size}")
            bad += 1
            continue
        mx, rel, cos = r
        flag = "" if rel <= a.tol else "  ← 超阈值"
        if rel > a.tol:
            bad += 1
        print(f"{name:<10} {str(y.shape):<16} {mx:11.3e} {rel:10.2e} {cos:10.6f}{flag}")
    print("-" * 62)
    print("全部一致" if bad == 0 else f"{bad} 个检查点不一致")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
