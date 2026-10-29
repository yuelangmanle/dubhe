#!/usr/bin/env python3
"""vl_ref.py — LFM3-VL 视觉塔的 numpy 参考实现（对拍用的第二意见）

为什么不直接跑 HuggingFace 的 modeling_lfm3_vl.py：那需要 4 GB 的 safetensors
权重 + 完整的 LFM3 文本模型。**但**完全不看 HF 源码、凭理解重写一遍是危险的：
第一版这里和 C++ 犯了一模一样的两个错（漏输出门、RoPE 用 block 号当坐标），
两边错得一样 → 对拍"通过" → 假绿灯。所以现在这份代码是**逐行对照
modeling_lfm3_vl.py 转写的**，每个可疑点旁边都标了原文。

主对拍对象是 tools/vl_oracle.py（直接跑 HF 的 forward）。这份 numpy 版的价值是
不依赖 torch，能在任何机器上快速复跑。

用法：
    python tools/vl_ref.py mmproj.gguf /tmp/vl.pixels.f32 /tmp/vl.meta.txt [--dump 前缀]
"""
from __future__ import annotations

import argparse
import sys
from math import erf
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gguf_read import Gguf  # noqa: E402

# ------------------------------------------------------------------ 算子


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def layernorm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def silu(x):
    return x / (1.0 + np.exp(-x))


_ERF = np.frompyfunc(erf, 1, 1)


def gelu(x):
    # ACT2FN["gelu"] 是 erf 版；tanh 近似差 1e-3，会被最后一层放大。
    # 环境里没有 scipy，用 math.erf 逐元素（只在投影器上跑一次，慢一点无所谓）。
    return 0.5 * x * (1.0 + _ERF(x * 0.7071067811865476).astype(np.float32))


# ------------------------------------------------------------------ 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mmproj")
    ap.add_argument("pixels")
    ap.add_argument("meta")
    ap.add_argument("--dump", default=None, help="把逐层中间结果写成 .npy")
    args = ap.parse_args()

    g = Gguf(args.mmproj)

    def W(name):
        """Linear 权重：gguf_read.get() 给 (out, in)，转成 (in, out) 好写 x @ W。"""
        return g.get(name).T

    gh, gw = [int(v) for v in open(args.meta).read().split()[:2]]
    N = gh * gw
    P = int(g.kv["clip.vision.patch_size"])
    M = int(g.kv["clip.vision.spatial_merge_size"])
    D = int(g.kv["clip.vision.embedding_length"])
    NH = int(g.kv["clip.vision.attention.head_count"])
    NKV = int(g.kv["clip.vision.attention.head_count_kv"])
    NL = int(g.kv["clip.vision.block_count"])
    HD = D // NH
    OUT = int(g.kv["clip.vision.projection_dim"])
    EPS = float(g.kv["clip.vision.attention.layer_norm_epsilon"])
    THETA = 1e6

    px = np.fromfile(args.pixels, dtype=np.float32).reshape(N, 3 * 2 * P * P)
    print(f"参考实现: {N} patch（{gh}×{gw}），hidden {D}，{NL} 层，输出 {OUT}")

    def dump(name, arr):
        if args.dump:
            np.save(f"{args.dump}.{name}.npy", arr)
        print(f"  {name:14s} mean {arr.mean():+.6f} std {arr.std():.6f}")

    # ---- patchify：v.patch_embd.weight 是 4D [16,16,3,1024]，get() 还原成 (1024, 768)
    slice_n = 3 * P * P
    x = px[:, :slice_n] @ W("v.patch_embd.weight") + \
        px[:, slice_n:slice_n * 2] @ W("v.patch_embd.weight.1")
    x = rmsnorm(x, g.get("v.embedding_norm.weight"), EPS)
    dump("patchify", x)

    # ---- token 序号 → 网格坐标（merge-block 序）
    gwm = gw // M
    idx = np.arange(N)
    by = idx // (gwm * M * M)
    rem = idx % (gwm * M * M)
    bx = rem // (M * M)
    sub = rem % (M * M)
    # 坐标是 **patch 下标**（HF: hpos_ids.reshape(h//M,M,w//M,M).transpose(1,2)
    # 展平后元素值仍是 by*M+dy），不是 block 号
    ph = by * M + sub // M
    pw = bx * M + sub % M

    # ---- 轴向 2D RoPE：每 head 的 HD 维 = [h(f) w(f) h(f) w(f)]
    nf = HD // 4
    invf = 1.0 / (THETA ** (np.arange(0, nf * 2, 2) / (HD / 2)))
    fh = ph[:, None] * invf[None, :]          # (N, nf)
    fw = pw[:, None] * invf[None, :]
    half = HD // 2
    cos_half = np.concatenate([np.cos(fh), np.cos(fw)], -1).astype(np.float32)
    sin_half = np.concatenate([np.sin(fh), np.sin(fw)], -1).astype(np.float32)

    def rope(t):
        # HF apply_rotary_pos_emb_vision: t*cos + rotate_half(t)*sin
        t1, t2 = t[..., :half], t[..., half:]
        c, s = cos_half[:, None, :], sin_half[:, None, :]   # (N,1,half) 对 (N,H,HD)
        return np.concatenate([t1 * c - t2 * s, t2 * c + t1 * s], -1)

    # ---- 14 层
    for li in range(NL):
        p = f"v.blk.{li}."
        xn = rmsnorm(x, g.get(p + "ln1.weight"), EPS)
        if g.has(p + "attn_q.weight"):
            q = (xn @ W(p + "attn_q.weight")).reshape(N, NH, HD)
            k = (xn @ W(p + "attn_k.weight")).reshape(N, NKV, HD)
            v = xn @ W(p + "attn_v.weight")
            q = rope(rmsnorm(q, g.get(p + "attn_q_norm.weight"), EPS))
            k = rope(rmsnorm(k, g.get(p + "attn_k_norm.weight"), EPS))
            # 双向 GQA
            rep = NH // NKV
            kk = np.repeat(k, rep, axis=1).transpose(1, 0, 2)
            vv = np.repeat(v.reshape(N, NKV, HD), rep, axis=1).transpose(1, 0, 2)
            qh = q.transpose(1, 0, 2)
            sc = (qh @ kk.transpose(0, 2, 1)) / np.sqrt(HD)
            sc -= sc.max(-1, keepdims=True)
            pr = np.exp(sc)
            pr /= pr.sum(-1, keepdims=True)
            o = (pr @ vv).transpose(1, 0, 2).reshape(N, D)
            y = o @ W(p + "attn_out.weight")
            tag = "attn"
        else:
            # HF: B, C, x = in_proj(h).chunk(3, -1); gated = B*x; ...; out_proj(C*conv)
            #   → 第 0 段是门控、第 2 段参与门控、**第 1 段是输出门**
            B, Cg, X = np.split(xn @ W(p + "conv2d_in_proj.weight"), 3, axis=-1)
            gated = B * X
            grid = np.zeros((D, gh, gw), dtype=np.float32)
            grid[:, ph, pw] = gated.T                       # merge-block 序 → 栅格
            kern = g.get(p + "conv2d_spatial.weight")       # (D, 9)，列号 = kx + 3*ky
            pad = np.pad(grid, ((0, 0), (1, 1), (1, 1)))
            conv = np.zeros_like(grid)
            for ky in range(3):
                for kx in range(3):
                    conv += kern[:, ky * 3 + kx][:, None, None] * pad[:, ky:ky + gh, kx:kx + gw]
            y = (conv[:, ph, pw].T * Cg) @ W(p + "conv2d_out_proj.weight")
            tag = "conv"
        x = x + y

        xn = rmsnorm(x, g.get(p + "ln2.weight"), EPS)
        h = silu(xn @ W(p + "ffn_gate.weight")) * (xn @ W(p + "ffn_up.weight"))
        x = x + h @ W(p + "ffn_down.weight")
        if li % 3 == 0 or li == NL - 1:
            dump(f"layer{li:02d}_{tag}", x)

    # ---- 投影器
    x = layernorm(x, g.get("mm.input_norm.weight"), g.get("mm.input_norm.bias"), 1e-6)
    h = x.reshape(-1, D * M * M)
    h = gelu(h @ W("mm.0.weight") + g.get("mm.0.bias"))
    out = h @ W("mm.2.weight") + g.get("mm.2.bias")
    dump("out", out)

    print("\n前 8 个值:", " ".join(f"{v:.5f}" for v in out.reshape(-1)[:8]))
    if args.dump:
        np.save(f"{args.dump}.out.npy", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
