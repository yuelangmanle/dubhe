#!/usr/bin/env python3
"""gguf_sizes.py — 只读 GGUF 的张量表，算出「解码一个 token 到底要读多少字节」。

为什么需要它：decode 的 tok/s 该拿什么当分母一直说不清。模型文件里有一半
字节是 embedding 表（只在 prefill 时整张读一次），还有 F32 的 norm 权重、
Q6_K 的 output.weight —— 混在一起算「文件大小 / token 时间」会得出一个
根本没意义的上界。这个脚本按张量逐个列字节数，顺便按「每 token 必读 / 只在
prefill 读 / 只读一次」分好类。

不依赖 numpy（本机没装），纯 struct 解析。

用法：
  python3 tools/gguf_sizes.py <model.gguf> [--top 12]
"""
import struct
import sys
from collections import defaultdict

# ggml type -> (名字, 每块元素数, 每块字节数)
GGML = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    19: ("IQ3_XXS", 256, 98), 20: ("IQ1_S", 256, 50),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    28: ("BF16", 1, 2),
}

# KV 值类型 -> struct 格式（9=array 特殊处理）
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
          6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("文件被截断")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def i64(self):
        return struct.unpack("<q", self.raw(8))[0]

    def f32(self):
        return struct.unpack("<f", self.raw(4))[0]

    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            # 数组内容一律跳过，只留长度：本脚本不关心 token 列表
            if et == 8:
                for _ in range(n):
                    self.u64()  # 长度
                    # 得真读，才知道下一条从哪开始
                    raise SystemExit("字符串数组暂不支持（本脚本不需要）")
            fmt = SCALAR.get(et)
            if fmt is None:
                raise SystemExit(f"未知数组元素类型 {et}")
            self.raw(struct.calcsize("<" + fmt) * n)
            return f"<{n} 项>"
        fmt = SCALAR.get(t)
        if fmt is None:
            raise SystemExit(f"未知 KV 类型 {t}")
        return struct.unpack("<" + fmt, self.raw(struct.calcsize("<" + fmt)))[0]


def read(path):
    with open(path, "rb") as f:
        r = R(f)
        if r.raw(4) != b"GGUF":
            raise SystemExit("不是 GGUF")
        ver = r.u32()
        n_tensor = r.u64()
        n_kv = r.u64()
        meta = {}
        for _ in range(n_kv):
            k = r.string()
            t = r.u32()
            if t == 9:
                et = r.u32()
                n = r.u64()
                if et == 8:
                    # 字符串数组：逐条读长度再跳过
                    for _ in range(n):
                        r.raw(r.u64())
                    meta[k] = f"<{n} 项字符串>"
                else:
                    fmt = SCALAR[et]
                    r.raw(struct.calcsize("<" + fmt) * n)
                    meta[k] = f"<{n} 项>"
            else:
                meta[k] = r.value(t)

        tensors = []
        for _ in range(n_tensor):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            off = r.u64()
            nelem = 1
            for d in dims:
                nelem *= d
            tname, blk_elems, blk_bytes = GGML.get(ttype, (f"type{ttype}", 1, 4))
            nbytes = (nelem // blk_elems) * blk_bytes
            tensors.append((name, tname, dims, nelem, nbytes, off))

        # 张量数据区起点：对齐到 32
        pos = f.tell()
        pos = (pos + 31) // 32 * 32
        f.seek(pos)
        data_start = pos
        # 用 offset 差值补齐最后一块
        tensors.sort(key=lambda t: t[5])
        return ver, meta, tensors, data_start


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    topn = 12
    if "--top" in sys.argv:
        topn = int(sys.argv[sys.argv.index("--top") + 1])

    ver, meta, tensors, data_start = read(path)
    n_layer = meta.get("qwen2.block_count") or meta.get("llama.block_count") \
        or meta.get("general.block_count") or 0

    total = sum(t[4] for t in tensors)
    print(f"GGUF v{ver} · {len(tensors)} 张量 · 权重共 {total/1e6:.1f} MB"
          f" · 层数 {n_layer}")

    by_type = defaultdict(lambda: [0, 0])
    for name, tname, dims, nelem, nbytes, _ in tensors:
        by_type[tname][0] += nbytes
        by_type[tname][1] += 1
    print("\n按量化类型：")
    for tname, (nb, cnt) in sorted(by_type.items(), key=lambda x: -x[1][0]):
        print(f"  {tname:8s} {nb/1e6:9.1f} MB  ({100*nb/total:5.1f}%)  {cnt:4d} 个")

    # 每 token 必读 = 除 token_embd 之外的全部（output.weight 若未 tied 也要读）
    emb = sum(t[4] for t in tensors if t[0].startswith("token_embd"))
    print(f"\n  token_embd       {emb/1e6:9.1f} MB  （只在 prefill 整张读；decode 只读 1 行）")
    print(f"  → 每 token 必读  {(total-emb)/1e6:9.1f} MB")

    print(f"\n最大的 {topn} 个张量：")
    for name, tname, dims, nelem, nbytes, _ in sorted(tensors, key=lambda t: -t[4])[:topn]:
        print(f"  {nbytes/1e6:8.2f} MB  {tname:6s} {str(dims):26s} {name}")

    # 按「每层内的一类算子」汇总，直接对上 decode 循环里的 dispatch
    groups = defaultdict(lambda: [0, 0])
    for name, tname, dims, nelem, nbytes, _ in tensors:
        key = ".".join(p for p in name.split(".") if not p.isdigit())
        groups[key][0] += nbytes
        groups[key][1] += 1
    print("\n按张量角色汇总（每 token 必读的那部分）：")
    for key, (nb, cnt) in sorted(groups.items(), key=lambda x: -x[1][0]):
        if key.startswith("token_embd"):
            continue
        per_layer = nb / max(n_layer, 1)
        print(f"  {nb/1e6:8.2f} MB  {cnt:4d} 个  {key:34s} 每层 {per_layer/1e6:6.3f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
