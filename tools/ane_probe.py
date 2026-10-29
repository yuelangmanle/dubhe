#!/usr/bin/env python3
"""ANE 可行性探针 —— 用 LFM3-1.7B 的真实矩阵形状，对 CPU / GPU / ANE 三方做背靠背测量。

结论只看两件事：
  1. 单次 matvec 的等效带宽（GB/s）——decode 是带宽瓶颈，这个数直接换算成 tok/s；
  2. 延迟随 batch 的变化曲线——ANE 是脉动阵列，batch=1 时严重欠载，
     若 batch 1→32 延迟几乎不变，说明确实跑在 ANE 上。

用法：
  python tools/ane_probe.py              # 全部形状
  python tools/ane_probe.py --only ffn   # 只测 ffn 三件套
  python tools/ane_probe.py --palette    # 额外测 4bit 调色板量化权重
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb

# ---------------------------------------------------------------- 形状定义
# 名字, (n_in, n_out), 说明。全部取自 LFM3-1.7B 的 config。
SHAPES = [
    ("attn_q",    (2048, 2048), "q_proj 16 head * 128"),
    ("attn_kv",   (2048, 512),  "k_proj/v_proj 4 kv head * 128"),
    ("attn_o",    (2048, 2048), "o_proj"),
    ("ffn_up",    (2048, 7680), "ffn gate+up 单个"),
    ("ffn_down",  (7680, 2048), "ffn down"),
    ("conv_in",   (2048, 6144), "shortconv in_proj B/C/x"),
    ("conv_out",  (2048, 2048), "shortconv out_proj"),
    ("lm_head",   (2048, 128000), "输出投影，权重流量最大的一块"),
]

BATCHES = [1, 4, 16, 32, 64, 128]

CUS = [
    ("CPU", ct.ComputeUnit.CPU_ONLY),
    ("GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("ANE", ct.ComputeUnit.CPU_AND_NE),
]


def build(n_in, n_out, batch, palette_bits=0):
    """造一个 y = x @ W 的 mlprogram。W 是常量，会随模型一起打包。"""
    rng = np.random.default_rng(0)
    # 用接近真实权重的尺度，避免异常值影响调度
    W = (rng.standard_normal((n_in, n_out)) * 0.02).astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, n_in))])
    def prog(x):
        w = mb.const(val=W, name="W")
        return mb.matmul(x=x, y=w, name="y")

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=(batch, n_in))],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    if palette_bits:
        # 把常量权重压到 nbit，和 Q4_K_M 的位宽可比
        import coremltools.optimize.coreml as cto

        opcfg = cto.OpPalettizerConfig(nbits=palette_bits, mode="kmeans")
        cfg = cto.OptimizationConfig(global_config=opcfg)
        m = cto.palettize_weights(m, cfg)
    return m, W


def measure(m, batch, n_in, iters, cu):
    path = tempfile.mkdtemp(prefix="aneprobe_") + "/m.mlpackage"
    m.save(path)
    model = ct.models.MLModel(path, compute_units=cu)
    x = (np.random.default_rng(1).standard_normal((batch, n_in)) * 0.5).astype(np.float32)
    y = model.predict({"x": x})["y"]  # 预热，首次含图优化
    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict({"x": x})
    dt = (time.perf_counter() - t0) / iters
    del model
    shutil.rmtree(os.path.dirname(path), ignore_errors=True)
    return dt, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--palette", action="store_true")
    ap.add_argument("--batches", default="")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")] if args.batches else BATCHES
    shapes = [s for s in SHAPES if args.only in s[0]] if args.only else SHAPES

    print(f"coremltools {ct.__version__}  numpy {np.__version__}")
    print(f"机器: {os.uname().machine}  batches={batches}\n")

    for bits in ([0, 4] if args.palette else [0]):
        wbytes_factor = 0.5 if bits == 0 else bits / 8.0
        tag = "f16" if bits == 0 else f"int{bits}"
        print(f"=== 权重精度 {tag} ===")
        hdr = f"{'shape':<10}{'n_in':>6}{'n_out':>7}{'B':>5}"
        for nm, _ in CUS:
            hdr += f"{nm+' ms':>11}{nm+' GB/s':>10}"
        print(hdr)
        for name, (n_in, n_out), _desc in shapes:
            for batch in batches:
                # lm_head 权重太大，只测小 batch
                if n_out >= 100000 and batch > 16:
                    continue
                row = f"{name:<10}{n_in:>6}{n_out:>7}{batch:>5}"
                wb = n_in * n_out * 2 * wbytes_factor
                for _nm, cu in CUS:
                    try:
                        m, _W = build(n_in, n_out, batch, bits)
                        iters = 200 if batch <= 16 else 60
                        dt, _y = measure(m, batch, n_in, iters, cu)
                        row += f"{dt*1000:>11.3f}{wb/dt/1e9:>10.1f}"
                    except Exception as e:  # noqa: BLE001
                        row += f"{'FAIL':>11}{type(e).__name__[:9]:>10}"
                print(row, flush=True)
        print()


if __name__ == "__main__":
    sys.exit(main())
