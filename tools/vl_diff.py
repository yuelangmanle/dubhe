#!/usr/bin/env python3
"""vl_diff.py — 把 C++ 的逐层 dump 和 HF 参考的 .npy 逐层对齐比较

用法：
    python tools/vl_diff.py /tmp/vl /tmp/vl        # (C++ 前缀, 参考前缀)
C++ 侧由 vlprobe --dump 产出 layerNN.f32 / out.f32，
参考侧由 vl_oracle.py --dump 产出 oracle_layerNN.npy / oracle_out.npy。

判定标准：视觉塔里除了最后的投影器，全部是 float32 累加，Q8_0 权重反量化两边
用的是同一份数据，所以除极少数接近抵消的位置，相对误差应该在 1e-5 量级。
差到 1e-2 就是实现错了，不是精度问题。
"""
from __future__ import annotations

import sys

import numpy as np


def load_f32(p):
    return np.fromfile(p, dtype=np.float32)


def stat(name, a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size:
        print(f"  {name:12s} 形状不符 C++={a.size} 参考={b.size}")
        return False
    d = np.abs(a - b)
    scale = max(np.abs(b).max(), 1e-12)
    rel = d.max() / scale
    # 余弦相似度比逐元素更能说明"是不是同一个向量"
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    ok = rel < 1e-4
    print(f"  {name:12s} max|Δ| {d.max():.3e}  rel {rel:.2e}  cos {cos:.6f}  "
          f"{'✓' if ok else '✗'}")
    return ok


def main() -> int:
    cpp, ref = sys.argv[1], sys.argv[2]
    print(f"C++ = {cpp}.layerNN.f32    参考 = {ref}.oracle_layerNN.npy")
    all_ok = True
    for i in range(0, 20):
        try:
            a = load_f32(f"{cpp}.layer{i:02d}.f32")
        except FileNotFoundError:
            break
        try:
            b = np.load(f"{ref}.oracle_layer{i:02d}.npy")
        except FileNotFoundError:
            print(f"  参考缺少 layer{i:02d}")
            break
        all_ok &= stat(f"layer{i:02d}", a, b)
    try:
        a = load_f32(f"{cpp}.out.f32")
        b = np.load(f"{ref}.oracle_out.npy")
        all_ok &= stat("projector", a, b)
    except FileNotFoundError:
        pass
    print("\n结论:", "全部一致" if all_ok else "存在分歧，看上面第一个 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
