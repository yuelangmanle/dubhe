#!/usr/bin/env python3
"""精度 × 后端 全交叉探针 —— 用 LFM3-1.7B 的真实矩阵形状做背靠背测量。

四个后端：
  CPU  = Core ML CPU_ONLY      （AMX / NEON，Accelerate 参考线）
  GPU  = Core ML CPU_AND_GPU   （苹果 Metal 路径）
  ANE  = Core ML CPU_AND_NE    （Apple Neural Engine）
  MLX  = 苹果自家为 Apple Silicon 写的框架，量化矩阵乘直接走 Metal

三档精度：
  f16  原版（无损基线）
  int8 每 64 元素一组对称量化
  int4 同上，4 bit（与 GGUF Q4_K_M 位宽可比）

看什么：
  1. 单次延迟与等效权重带宽（GB/s）—— decode 是带宽瓶颈，这个数直接换算 tok/s；
  2. 延迟随 batch 的曲线 —— ANE 是脉动阵列，batch 1→128 延迟几乎不变即为真在跑 ANE。

用法：
  python tools/probe_precision.py                  # 全量
  python tools/probe_precision.py --only ffn
  python tools/probe_precision.py --batches 1,64
"""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb

# ---------------------------------------------------------------- 形状
# 全部取自 LFM3-1.7B 的 config：hidden 2048 / inter 7680 / 16 head × 128 / 4 kv head
SHAPES = [
    ("attn_q",   (2048, 2048),   "q_proj 16*128"),
    ("attn_kv",  (2048, 512),    "k/v_proj 4*128"),
    ("attn_o",   (2048, 2048),   "o_proj"),
    ("attn_gate",(2048, 16),     "逐 head sigmoid 门"),
    ("ffn_up",   (2048, 7680),   "ffn gate/up 单个"),
    ("ffn_down", (7680, 2048),   "ffn down"),
    ("conv_in",  (2048, 6144),   "shortconv in_proj B/C/x"),
    ("conv_out", (2048, 2048),   "shortconv out_proj"),
    ("lm_head",  (2048, 128000), "输出投影，权重流量最大"),
]

BATCHES = [1, 8, 32, 128]
BITS = [0, 4, 8]          # 0 = f16 原版
PREC_NAME = {0: "f16", 4: "int4", 8: "int8"}

# 每权重字节数（int4 按 0.5 算，group scale 的开销忽略）
def wbytes(n_in, n_out, bits):
    return n_in * n_out * (2.0 if bits == 0 else bits / 8.0)


@contextlib.contextmanager
def quiet_stderr():
    """coremltools 的 MIL pass 进度条直接写 fd 2，Python 层重定向拦不住。"""
    fd = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    try:
        yield
    finally:
        os.dup2(fd, 2)
        os.close(null)
        os.close(fd)


# ---------------------------------------------------------------- Core ML
def build_coreml(n_in, n_out, batch, bits):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((n_in, n_out)) * 0.02).astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, n_in))])
    def prog(x):
        w = mb.const(val=W, name="W")
        return mb.matmul(x=x, y=w, name="y")

    with quiet_stderr():
        m = ct.convert(
            prog,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="x", shape=(batch, n_in))],
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16,
        )
        if bits:
            import coremltools.optimize.coreml as cto

            cfg = cto.OptimizationConfig(
                global_config=cto.OpPalettizerConfig(nbits=bits, mode="kmeans")
            )
            m = cto.palettize_weights(m, cfg)
    return m


def run_coreml(n_in, n_out, batch, bits, cu, iters):
    m = build_coreml(n_in, n_out, batch, bits)
    d = tempfile.mkdtemp(prefix="probe_")
    path = os.path.join(d, "m.mlpackage")
    m.save(path)
    try:
        model = ct.models.MLModel(path, compute_units=cu)
        x = (np.random.default_rng(1).standard_normal((batch, n_in)) * 0.5).astype(np.float32)
        model.predict({"x": x})  # 预热
        t0 = time.perf_counter()
        for _ in range(iters):
            model.predict({"x": x})
        return (time.perf_counter() - t0) / iters
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- MLX
def run_mlx(n_in, n_out, batch, bits, iters):
    import mlx.core as mx

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((n_out, n_in)) * 0.02).astype(np.float32)  # MLX 是 [out, in]
    x = (np.random.default_rng(1).standard_normal((batch, n_in)) * 0.5).astype(np.float32)
    xm = mx.array(x)

    if bits == 0:
        Wm = mx.array(W).astype(mx.float16)
        xm = xm.astype(mx.float16)
        f = lambda: xm @ Wm.T  # noqa: E731
    else:
        Wq, scales, biases = mx.quantize(mx.array(W), group_size=64, bits=bits)
        f = lambda: mx.quantized_matmul(  # noqa: E731
            xm, Wq, scales, biases, transpose=True, group_size=64, bits=bits
        )

    mx.eval(f())  # 预热（含 kernel 编译）
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(f())
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--batches", default="")
    ap.add_argument("--bits", default="")
    ap.add_argument("--no-mlx", action="store_true")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")] if args.batches else BATCHES
    bits_list = [int(b) for b in args.bits.split(",")] if args.bits else BITS
    shapes = [s for s in SHAPES if args.only in s[0]] if args.only else SHAPES

    cus = [
        ("CPU", ct.ComputeUnit.CPU_ONLY),
        ("GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ANE", ct.ComputeUnit.CPU_AND_NE),
    ]
    have_mlx = False
    if not args.no_mlx:
        try:
            import mlx.core as mx  # noqa: F401

            have_mlx = True
        except ImportError:
            pass

    print(f"coremltools {ct.__version__}  numpy {np.__version__}"
          f"{'  mlx ' + mx.__version__ if have_mlx else ''}")
    print(f"机器 {os.uname().machine}  batches={batches}\n")

    for bits in bits_list:
        print(f"===== 权重精度 {PREC_NAME[bits]} =====")
        hdr = f"{'shape':<10}{'n_in':>6}{'n_out':>7}{'B':>5}"
        for nm, _ in cus:
            hdr += f"{nm+' ms':>10}"
        if have_mlx:
            hdr += f"{'MLX ms':>10}"
        hdr += f"{'最快':>7}"
        print(hdr)

        for name, (n_in, n_out), _d in shapes:
            for batch in batches:
                if n_out >= 100000 and batch > 32:
                    continue
                iters = 100 if batch <= 8 else 40
                wb = wbytes(n_in, n_out, bits)
                times = []
                row = f"{name:<10}{n_in:>6}{n_out:>7}{batch:>5}"
                for _nm, cu in cus:
                    try:
                        dt = run_coreml(n_in, n_out, batch, bits, cu, iters)
                        row += f"{dt*1000:>10.3f}"
                        times.append((_nm, dt))
                    except Exception as e:  # noqa: BLE001
                        row += f"{'FAIL':>10}"
                        print(f"  [warn] {name} B={batch} {_nm} {type(e).__name__}: {e}",
                              file=sys.stderr)
                if have_mlx:
                    try:
                        dt = run_mlx(n_in, n_out, batch, bits, iters)
                        row += f"{dt*1000:>10.3f}"
                        times.append(("MLX", dt))
                    except Exception as e:  # noqa: BLE001
                        row += f"{'FAIL':>10}"
                        print(f"  [warn] {name} B={batch} MLX {type(e).__name__}: {e}",
                              file=sys.stderr)
                if times:
                    best = min(times, key=lambda t: t[1])
                    row += f"{best[0]:>7}"
                print(row, flush=True)
        print()


if __name__ == "__main__":
    sys.exit(main())
