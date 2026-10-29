#!/usr/bin/env python3
"""vl_oracle.py — 用 HuggingFace 自己的 forward 当权威对拍对象

和 vl_ref.py 的区别：vl_ref.py 是"我照着源码转写一遍"，而这里是**原封不动
跑 HF 的 modeling_lfm3_vl.py**，只把权重来源换成我们这份 mmproj GGUF。
于是参考实现里不再含任何我的理解 —— 这是唯一能证伪"我和参考实现犯同一个
错"的手段（第一版就栽在这上面）。

它需要 torch，而且 transformers 5.21 还没有内置 lfm3_vl，所以这里把 HF 仓库
里那 5 个 .py 自动抓到 ~/.dubhe/oracle/lfm3_vl/ 当本地包导入。
（HF 的 LFM3-VL 代码是 Apache-2.0；本仓库不 vendor 它，只在本地缓存。）

用法：
    python tools/vl_oracle.py mmproj.gguf /tmp/vl.pixels.f32 /tmp/vl.meta.txt [--dump 前缀]
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gguf_read import Gguf  # noqa: E402

HF_REPO = "LiquidAI/LFM3-VL-2B"
HF_FILES = ["modeling_lfm3_vl.py", "configuration_lfm3_vl.py",
            "modeling_lfm3.py", "configuration_lfm3.py", "lfm3_conv.py"]
CACHE = Path(os.environ.get("DUBHE_ORACLE_DIR", Path.home() / ".dubhe" / "oracle")) / "lfm3_vl"


def ensure_hf_modules() -> Path:
    """把 HF 的模块文件抓到本地缓存，返回包目录。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "__init__.py").touch()
    for f in HF_FILES:
        dst = CACHE / f
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{f}"
        print(f"  下载参考实现 {f} ...", flush=True)
        urllib.request.urlretrieve(url, dst)
    return CACHE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mmproj")
    ap.add_argument("pixels")
    ap.add_argument("meta")
    ap.add_argument("--dump", default=None, help="逐层 hidden state 写成 .npy")
    args = ap.parse_args()

    pkg_root = ensure_hf_modules()
    sys.path.insert(0, str(pkg_root))

    import torch
    from lfm3_vl.configuration_lfm3_vl import Lfm3VlConfig, Lfm3VlVisionConfig
    from lfm3_vl.modeling_lfm3_vl import Lfm3VlMultiModalProjector, Lfm3VlVisionModel

    torch.set_grad_enabled(False)
    g = Gguf(args.mmproj)
    kv = g.kv

    gh, gw = [int(v) for v in open(args.meta).read().split()[:2]]
    N = gh * gw
    P = int(kv["clip.vision.patch_size"])
    M = int(kv["clip.vision.spatial_merge_size"])
    D = int(kv["clip.vision.embedding_length"])
    NL = int(kv["clip.vision.block_count"])
    T = 2 if g.has("v.patch_embd.weight.1") else 1

    layer_types = []
    for i in range(NL):
        layer_types.append("full_attention" if g.has(f"v.blk.{i}.attn_q.weight") else "conv")

    vcfg = Lfm3VlVisionConfig(
        hidden_size=D,
        intermediate_size=int(kv["clip.vision.feed_forward_length"]),
        num_hidden_layers=NL,
        num_attention_heads=int(kv["clip.vision.attention.head_count"]),
        num_key_value_heads=int(kv["clip.vision.attention.head_count_kv"]),
        layer_types=layer_types,
        patch_size=P,
        temporal_patch_size=T,
        in_channels=3,
        spatial_merge_size=M,
        conv_kernel_size=3,
        norm_eps=float(kv["clip.vision.attention.layer_norm_epsilon"]),
        rope_parameters={"rope_type": "axial", "rope_theta": 1e6},
    )
    vcfg._attn_implementation = "eager"
    vcfg.dtype = torch.float32

    vision = Lfm3VlVisionModel(vcfg).to(torch.float32).eval()

    def t(name):
        return torch.from_numpy(np.ascontiguousarray(g.get(name))).to(torch.float32)

    sd = {}
    # patchifier：GGUF 按时间片拆成两块 w[:, :, t]，每块内部是 [c][ph][pw]。
    # HF 的权重是 [o][c][t][ph][pw]，所以要在**通道维后面**插回时间维，
    # 不能简单地在输入维上首尾相接 —— 那是 t-major，和 HF 的 c-major 输入错位。
    if T == 2:
        w0 = torch.from_numpy(np.ascontiguousarray(g.get("v.patch_embd.weight"))).view(D, 3, P * P)
        w1 = torch.from_numpy(np.ascontiguousarray(g.get("v.patch_embd.weight.1"))).view(D, 3, P * P)
        cat = torch.stack([w0, w1], dim=2).reshape(D, -1)      # (D, 3*T*P*P)
    else:
        cat = torch.from_numpy(np.ascontiguousarray(g.get("v.patch_embd.weight")))
    sd["patchifier.weight"] = cat.to(torch.float32)
    sd["embedding_norm.weight"] = t("v.embedding_norm.weight")
    for i in range(NL):
        p, h = f"v.blk.{i}.", f"layers.{i}."
        sd[h + "operator_norm.weight"] = t(p + "ln1.weight")
        sd[h + "ffn_norm.weight"] = t(p + "ln2.weight")
        sd[h + "feed_forward.w1.weight"] = t(p + "ffn_gate.weight")
        sd[h + "feed_forward.w3.weight"] = t(p + "ffn_up.weight")
        sd[h + "feed_forward.w2.weight"] = t(p + "ffn_down.weight")
        if layer_types[i] == "full_attention":
            sd[h + "self_attn.q_proj.weight"] = t(p + "attn_q.weight")
            sd[h + "self_attn.k_proj.weight"] = t(p + "attn_k.weight")
            sd[h + "self_attn.v_proj.weight"] = t(p + "attn_v.weight")
            sd[h + "self_attn.out_proj.weight"] = t(p + "attn_out.weight")
            sd[h + "self_attn.q_layernorm.weight"] = t(p + "attn_q_norm.weight")
            sd[h + "self_attn.k_layernorm.weight"] = t(p + "attn_k_norm.weight")
        else:
            sd[h + "conv.in_proj.weight"] = t(p + "conv2d_in_proj.weight")
            sd[h + "conv.out_proj.weight"] = t(p + "conv2d_out_proj.weight")
            # GGUF (D, 9) 列号 = kx + 3*ky  →  torch (D, 1, ky, kx)
            sd[h + "conv.spatial_conv.weight"] = t(p + "conv2d_spatial.weight").reshape(D, 1, 3, 3)

    missing, unexpected = vision.load_state_dict(sd, strict=False)
    if unexpected:
        print("  多余的键:", unexpected[:5])
    if missing:
        print("  缺失的键:", missing[:5])

    # 像素布局：C++ 的 pack_pixels 是 [t][c][ph][pw]（与 GGUF 拆片一致），
    # HF 的 pixel_values 是 [c][t][ph][pw]。静止图数值相同、排列不同，必须换过来。
    px = np.fromfile(args.pixels, dtype=np.float32).reshape(N, T, 3, P * P)
    px = np.ascontiguousarray(px.transpose(0, 2, 1, 3).reshape(N, T * 3 * P * P))
    pv = torch.from_numpy(px).to(torch.float32)
    grid = torch.tensor([[1, gh, gw]], dtype=torch.long)

    print(f"权威参考(HF forward): {N} patch（{gh}×{gw}），{NL} 层，层类型 "
          f"{''.join('A' if x == 'full_attention' else 'C' for x in layer_types)}")
    out = vision(pixel_values=pv, grid_thw=grid, output_hidden_states=True)
    hs = out.hidden_states
    for i, h in enumerate(hs):
        a = h.numpy()
        print(f"  layer{i:02d}         mean {a.mean():+.6f} std {a.std():.6f}")
        if args.dump:
            np.save(f"{args.dump}.oracle_layer{i:02d}.npy", a)

    # ---- 投影器（用 HF 的类，eps/act/bias 全部按官方 config.json）
    pcfg = Lfm3VlConfig(
        vision_config=vcfg.to_dict(),
        text_config={"hidden_size": int(kv["clip.vision.projection_dim"])},
        downsample_factor=M,
        projector_hidden_size=int(g.get("mm.0.weight").shape[0]),
        projector_hidden_act="gelu",
        projector_bias=True,
        projector_layer_norm_eps=1e-6,
    )
    pcfg.dtype = torch.float32
    proj = Lfm3VlMultiModalProjector(pcfg).to(torch.float32).eval()
    psd = {
        "layer_norm.weight": t("mm.input_norm.weight"),
        "layer_norm.bias": t("mm.input_norm.bias"),
        "linear_1.weight": t("mm.0.weight"),
        "linear_1.bias": t("mm.0.bias"),
        "linear_2.weight": t("mm.2.weight"),
        "linear_2.bias": t("mm.2.bias"),
    }
    proj.load_state_dict(psd, strict=False)
    feats = proj(hs[-1].to(torch.float32)).numpy()
    print(f"  projector      mean {feats.mean():+.6f} std {feats.std():.6f}")
    print("\n前 8 个值:", " ".join(f"{v:.5f}" for v in feats.reshape(-1)[:8]))
    if args.dump:
        np.save(f"{args.dump}.oracle_out.npy", feats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
