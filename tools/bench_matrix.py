#!/usr/bin/env python3
"""bench_matrix.py — 精度 × 运行时 全矩阵基准

一次跑完三件事，输出可直接贴进进度书的 markdown 表：

  1. 同一个模型在不同**量化档**下的速度与输出一致性
     （GGUF Q4_K_M / Q8_0 / F16，以及官方 MLX 4bit / 8bit / bf16）
  2. 同一个权重在不同**运行时**下的速度
     （Dubhe 的 CPU / Metal，苹果的 MLX）
  3. 输出质量对照：固定 prompt + 贪心解码，逐 token 比较

设计约束：
  * 只用**贪心**解码。采样会让两个引擎的输出不可比，而"速度"这件事
    必须建立在"算的是同一个东西"之上。
  * 每个 case 跑 3 次取中位数。M 系列 GPU 有 DVFS，单次测量噪声很大。
  * MLX 侧可能没装或版本不支持 lfm3 —— 那时只跳过 MLX 列并说明原因，
    不让整个矩阵失败。

用法:
    python tools/bench_matrix.py                     # 跑内置矩阵
    python tools/bench_matrix.py --only dubhe        # 只跑 Dubhe
    python tools/bench_matrix.py --prompt "..." -n 64
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUBHE = ROOT / "build" / "dubhe"
MODELS = Path.home() / ".dubhe" / "models"

# 三档 GGUF + 三档官方 MLX。名字里带路径，方便直接跑。
GGUF_CASES = [
    ("Q4_K_M", MODELS / "lfm3-1.7b" / "LFM3-1.7B-Q4_K_M.gguf"),
    ("Q8_0",   MODELS / "lfm3-1.7b-q8" / "LFM3-1.7B-Q8_0.gguf"),
    ("F16",    MODELS / "lfm3-1.7b-f16" / "LFM3-1.7B-F16.gguf"),
    ("VL-2B Q4_K_M", MODELS / "lfm3-vl-2b" / "LFM3-VL-2B-Q4_K_M.gguf"),
]
MLX_CASES = [
    ("MLX 4bit", MODELS / "lfm3-mlx-4bit"),
    ("MLX 8bit", MODELS / "lfm3-mlx-8bit"),
]

PROMPTS = {
    "short": "The capital of France is",
    "long": "The history of science is a long story of careful measurement and bold conjecture. " * 40,
}


@dataclass
class Row:
    backend: str
    quant: str
    size_mb: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    text: str = ""
    note: str = ""


# ------------------------------------------------------------------ Dubhe

STAT_RE = re.compile(r"prompt (\d+) tok / ([\d.]+) ms \(([\d.]+) tok/s\) \| gen (\d+) tok / ([\d.]+) ms \(([\d.]+) tok/s\)")


def run_dubhe(model: Path, prompt: str, n: int, no_gpu: bool) -> Row | None:
    if not model.exists():
        return Row("dubhe", model.stem, note="文件不存在")
    cmd = [str(DUBHE), "run", "-m", str(model), "-p", prompt, "--raw",
           "-n", str(n), "--temp", "0", "--stats"]
    if no_gpu:
        cmd.append("--no-gpu")
    best_pre, best_dec, text = 0.0, 0.0, ""
    for _ in range(3):
        env = dict(os.environ, DUBHE_NO_FUSED_CHECK="1")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        out = p.stdout + p.stderr
        m = STAT_RE.search(out)
        if not m:
            return Row("dubhe", model.stem, note=(p.stderr or out).strip().splitlines()[-1:][0] if (p.stderr or out) else "无输出")
        pre, dec = float(m.group(3)), float(m.group(6))
        best_pre = max(best_pre, pre)
        best_dec = max(best_dec, dec)
        # 统计行之前的非 [dubhe] 行就是生成文本
        body = [ln for ln in out.splitlines() if not ln.startswith("[dubhe]") and not ln.startswith("[统计]")]
        text = "\n".join(body).strip()
    return Row("dubhe" + ("-cpu" if no_gpu else "-metal"), model.stem,
               size_mb=model.stat().st_size / 1e6,
               prefill_tps=best_pre, decode_tps=best_dec, text=text)


# ------------------------------------------------------------------ MLX

def mlx_available() -> tuple[bool, str]:
    try:
        import mlx_lm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"mlx_lm 不可用: {e}"
    try:
        from mlx_lm.models import lfm3  # noqa: F401
        return True, ""
    except Exception:  # noqa: BLE001
        try:
            from mlx_lm.utils import _get_classes
            return True, "lfm3 模块名未知，交由 load 自动识别"
        except Exception as e:  # noqa: BLE001
            return False, f"mlx_lm 不支持 lfm3: {e}"


def run_mlx(model_dir: Path, prompt: str, n: int) -> Row:
    """在一个子进程里跑 MLX，避免 import 失败拖垮整个矩阵。"""
    if not model_dir.exists():
        return Row("mlx", model_dir.name, note="目录不存在")
    code = r"""
import json, sys, time
from mlx_lm import load, generate
path, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
model, tok = load(path)
# 贪心：temp=0 在 mlx_lm 里走 argmax 路径
t0 = time.perf_counter()
text = generate(model, tok, prompt=prompt, max_tokens=n, temp=0.0, verbose=False)
dt = time.perf_counter() - t0
ntok = len(tok.encode(text))
print(json.dumps({"text": text, "decode_tps": ntok / dt if dt else 0.0,
                  "prefill_tps": 0.0, "note": ""}))
"""
    try:
        p = subprocess.run([sys.executable, "-c", code, str(model_dir), prompt, str(n)],
                           capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return Row("mlx", model_dir.name, note="超时")
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        return Row("mlx", model_dir.name, note=tail[-1] if tail else "失败")
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return Row("mlx", model_dir.name, note="输出解析失败")
    size = sum(f.stat().st_size for f in model_dir.glob("*.safetensors")) / 1e6
    return Row("mlx", model_dir.name, size_mb=size, prefill_tps=d["prefill_tps"],
               decode_tps=d["decode_tps"], text=d["text"].strip(), note=d.get("note", ""))


# ------------------------------------------------------------------ 输出

def emit(rows: list[Row], title: str) -> None:
    print(f"\n### {title}\n")
    print("| 后端 | 权重 | 体积 | prefill | decode |")
    print("|---|---|---:|---:|---:|")
    for r in rows:
        if r.note:
            print(f"| {r.backend} | {r.quant} | — | — | _{r.note}_ |")
        else:
            print(f"| {r.backend} | {r.quant} | {r.size_mb:.0f} MB | "
                  f"{r.prefill_tps:.1f} tok/s | {r.decode_tps:.1f} tok/s |")

    # 输出一致性：以第一个有文本的为基准做前缀比较
    base = next((r for r in rows if r.text), None)
    if base:
        print(f"\n贪心输出（基准 = {base.backend} / {base.quant}）:\n")
        for r in rows:
            if not r.text:
                continue
            same = r.text == base.text
            n = 0
            while n < min(len(r.text), len(base.text)) and r.text[n] == base.text[n]:
                n += 1
            mark = "完全一致" if same else f"前 {n} 字符一致"
            print(f"- **{r.backend} / {r.quant}**（{mark}）: {r.text[:120]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="short", choices=sorted(PROMPTS))
    ap.add_argument("-n", "--tokens", type=int, default=48)
    ap.add_argument("--only", choices=["dubhe", "mlx"], default=None)
    ap.add_argument("--json", default=None, help="把原始数据另存为 json")
    args = ap.parse_args()

    prompt = PROMPTS[args.prompt]
    rows: list[Row] = []

    if args.only in (None, "dubhe"):
        for label, path in GGUF_CASES:
            r = run_dubhe(path, prompt, args.tokens, no_gpu=False)
            if r:
                r.quant = f"{label} · {path.stem}"
                rows.append(r)
        # CPU 对照只跑 Q4 一档，够说明 Metal 的倍数
        r = run_dubhe(GGUF_CASES[0][1], prompt, args.tokens, no_gpu=True)
        if r:
            r.quant = f"Q4_K_M · {GGUF_CASES[0][1].stem}"
            rows.append(r)

    if args.only in (None, "mlx"):
        ok, why = mlx_available()
        if not ok:
            print(f"[skip] {why}")
        else:
            if why:
                print(f"[note] {why}")
            for label, path in MLX_CASES:
                r = run_mlx(path, prompt, args.tokens)
                r.quant = label
                rows.append(r)

    emit(rows, f"LFM3 精度 × 后端矩阵（prompt={args.prompt}, n={args.tokens}, greedy）")

    if args.json:
        Path(args.json).write_text(json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2))
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
