#!/usr/bin/env python3
"""gguf_read.py — 极简 GGUF 读取器，供 vl_ref.py / vl_oracle.py 共用

只做两件事：解析元数据与张量表、把张量反量化成 f32 的 (out, in) 矩阵。

**形状约定**（这里踩过坑，写清楚）：
GGUF 的 ne[0] 是最快的维度，所以一个 Linear 权重在文件里是 (in, out)。
但视觉塔的 patch_embd / conv2d_spatial 是 4D：
    v.patch_embd.weight      ne = [16, 16, 3, 1024]   → 展平后每行 768，共 1024 行
    v.blk.N.conv2d_spatial   ne = [3, 3, 1, 1024]     → 展平后每行 9，共 1024 行
所以正确的还原是 reshape(ne[-1], -1)，而不是 reshape(-1, ne[0])——后者只在
2D 时碰巧成立，4D 时会得到 (1536, 16) 这种看起来像矩阵、其实是垃圾的东西，
喂进 @ 运算还会因为维度刚好对得上而不报错。
"""
from __future__ import annotations

import struct

import numpy as np

GGML = {0: ("f32", 4, 1), 1: ("f16", 2, 1), 8: ("q8_0", 34, 32), 12: ("q4_k", 144, 256),
        14: ("q6_k", 210, 256), 30: ("bf16", 2, 1)}


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        assert self.f.read(4) == b"GGUF", "不是 GGUF"
        self.ver = self._u("I")
        nt = self._u("Q")
        nkv = self._u("Q")
        self.align = 32
        self.kv = {}
        for _ in range(nkv):
            k = self._s()
            t = self._u("I")
            if k == "general.alignment" and t == 4:
                self.align = self._u("I")
            else:
                self.kv[k] = self._v(t)
        self.tensors = {}
        for _ in range(nt):
            name = self._s()
            nd = self._u("I")
            dims = [self._u("Q") for _ in range(nd)]
            ty = self._u("I")
            off = self._u("Q")
            self.tensors[name] = (ty, dims, off)
        pos = self.f.tell()
        self.data0 = (pos + self.align - 1) // self.align * self.align

    def _u(self, fmt):
        sz = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(sz))[0]

    def _s(self):
        n = self._u("Q")
        return self.f.read(n).decode("utf-8")

    def _v(self, t):
        if t == 8:
            return self._s()
        if t == 9:
            it = self._u("I")
            n = self._u("Q")
            return [self._v(it) for _ in range(n)]
        if t in (4, 5, 6):
            return self._u("I" if t == 4 else ("i" if t == 5 else "f"))
        if t in (10, 11, 12):
            return self._u("Q" if t == 10 else ("q" if t == 11 else "d"))
        if t in (0, 1):
            return self.f.read(1)[0]
        if t in (2, 3):
            return struct.unpack("<h", self.f.read(2))[0]
        if t == 7:
            return bool(self.f.read(1)[0])
        raise ValueError(f"kv type {t}")

    def has(self, name):
        return name in self.tensors

    def get(self, name):
        """反量化成 f32。1D 张量返回 (n,)，多维返回 (最后一维, 其余展平)。"""
        ty, dims, off = self.tensors[name]
        kind, blk_b, blk_n = GGML[ty]
        total = int(np.prod(dims))
        self.f.seek(self.data0 + off)
        raw = self.f.read(total // blk_n * blk_b)
        if kind == "f32":
            a = np.frombuffer(raw, dtype="<f4")
        elif kind == "f16":
            a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif kind == "bf16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            a = u.view(np.float32)
        elif kind == "q8_0":
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
            d = b[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)
            q = b[:, 2:].copy().view(np.int8).astype(np.float32)
            a = (q * d[:, None]).reshape(-1)
        else:
            raise ValueError(f"{name}: 暂不支持 {kind}，请用 Q8_0/F16 的 mmproj")
        if len(dims) == 1:
            return a.astype(np.float32)
        return a.astype(np.float32).reshape(dims[-1], -1)
