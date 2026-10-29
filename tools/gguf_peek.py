#!/usr/bin/env python3
"""最小 GGUF 读取器 —— 只为了核对元数据和单个张量的原始数值。

用法：
  python tools/gguf_peek.py model.gguf                    # 元数据 + 张量清单
  python tools/gguf_peek.py model.gguf rope_freqs.weight  # 打印某个张量前 N 个值
"""
import struct
import sys

import numpy as np

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

SCALAR = {
    T_U8: ("<B", 1), T_I8: ("<b", 1), T_U16: ("<H", 2), T_I16: ("<h", 2),
    T_U32: ("<I", 4), T_I32: ("<i", 4), T_F32: ("<f", 4), T_BOOL: ("<?", 1),
    T_U64: ("<Q", 8), T_I64: ("<q", 8), T_F64: ("<d", 8),
}

# type -> (每 block 字节数, 每 block 元素数)
GGML_TYPE_SIZE = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32),
    8: (34, 32), 9: (36, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256),
    13: (176, 256), 14: (210, 256), 15: (292, 256), 24: (1, 1), 25: (2, 1),
    26: (4, 1), 27: (8, 1), 28: (8, 1), 30: (2, 1),
}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def scalar(self, t):
        fmt, n = SCALAR[t]
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += n
        return v

    def string(self):
        n = struct.unpack_from("<Q", self.b, self.o)[0]
        self.o += 8
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def value(self, t):
        if t in SCALAR:
            return self.scalar(t)
        if t == T_STR:
            return self.string()
        if t == T_ARR:
            et = struct.unpack_from("<I", self.b, self.o)[0]
            self.o += 4
            n = struct.unpack_from("<Q", self.b, self.o)[0]
            self.o += 8
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown gguf type {t}")


def parse(path):
    buf = open(path, "rb").read()
    r = Reader(buf)
    assert buf[:4] == b"GGUF", "不是 GGUF 文件"
    r.o = 4
    ver = r.scalar(T_U32)
    n_tensors = r.scalar(T_U64)
    n_kv = r.scalar(T_U64)

    kv = {}
    for _ in range(n_kv):
        k = r.string()
        t = struct.unpack_from("<I", buf, r.o)[0]
        r.o += 4
        kv[k] = r.value(t)

    infos = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.scalar(T_U32)
        ne = [r.scalar(T_U64) for _ in range(nd)]
        tt = r.scalar(T_U32)
        off = r.scalar(T_U64)
        infos.append({"name": name, "ne": ne, "type": tt, "off": off})

    align = kv.get("general.alignment", 32)
    data_start = (r.o + align - 1) // align * align
    return kv, infos, buf, data_start


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else None
    kv, infos, buf, ds = parse(path)
    if want is None:
        for k, v in kv.items():
            s = repr(v)
            print(f"{k:<48} {s[:100]}")
        print()
        for t in infos:
            print(f"{t['name']:<40} type={t['type']:<3} ne={t['ne']} off={t['off']}")
        return
    t = next(x for x in infos if x["name"] == want)
    bs, ne = GGML_TYPE_SIZE[t["type"]]
    n = int(np.prod(t["ne"]))
    raw = buf[ds + t["off"]: ds + t["off"] + (n // ne) * bs]
    if t["type"] == 0:
        v = np.frombuffer(raw, dtype=np.float32)
    elif t["type"] == 1:
        v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        print(f"类型 {t['type']} 暂不反量化，只显示字节数 {len(raw)}")
        return
    print(f"{want}  n={n}  shape={t['ne']}")
    for i, x in enumerate(v[:80]):
        print(f"  [{i:>3}] {x: .8e}")


if __name__ == "__main__":
    main()
