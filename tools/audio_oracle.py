#!/usr/bin/env python3
"""audio_oracle.py — 用 HF 的官方实现跑语音塔参考前向，作为唯一权威判据

为什么要这个：C++ 和 numpy 参考实现如果由同一个人写，会**犯同样的错**，
对拍全绿也只是假绿灯（视觉塔那次就是这样，两个错互相掩护）。所以拿
transformers 里的 Lfm3AudioEncoder 真跑一遍，逐层存下来当标准答案。

权重不另外下载：mmproj 的 GGUF 里就是这套权重，按下面的映射装进 HF 模块。
映射本身错了怎么办 —— 那 oracle 也是错的。所以最后还有一道独立闸门：
用 `say` 合成一句人声，看引擎答得对不对。只有端到端能听懂，才说明整条链路对。

几个容易搞错的地方（都实测确认过）：
  * 语音塔内部是 **LFM2**（不是 LFM3）：`AutoModel.from_config` 走
    transformers 内置的 Lfm2Model，所以参数名是 q_layernorm/k_layernorm/out_proj，
    跟 modeling_lfm3.py 里写的 q_norm/k_norm/o_proj 不一样。
  * GGUF 的 `a.post_ln` 是 HF 的 `embedding_norm`，虽然名字叫 post 但它在
    **所有层之后**才作用（Lfm2Model.forward 的最后一行）。
  * `conv_dw/conv_pw1/conv_pw2` = `conv.conv / conv.in_proj / conv.out_proj`，
    结构是 out_proj(C * dwconv(B * x))。

用法:
    python3 audio_oracle.py --wav clip.wav --out /tmp/aud_oracle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.expanduser("~/.dubhe/oracle")
sys.path.insert(0, HERE)
sys.path.insert(0, ORACLE)

from gguf_read import Gguf  # noqa: E402

DEFAULT_CFG = os.path.join(ORACLE, "lfm3_audio", "config.json")
DEFAULT_MMPROJ = os.path.expanduser(
    "~/.dubhe/models/lfm3-audio-2b/mmproj-LFM3-Audio-2B-Q8_0.gguf")

CONV_KEYS = {
    "a.conv1d.0": "conv_in", "a.conv1d.2": "conv_mid", "a.conv1d.3": "lin_mid",
    "a.conv1d.5": "conv_end", "a.conv1d.6": "lin_end",
}
BLK_KEYS = {
    "ln1.weight": "operator_norm.weight",
    "ln2.weight": "ffn_norm.weight",
    "conv_dw.weight": "conv.conv.weight",
    "conv_pw1.weight": "conv.in_proj.weight",
    "conv_pw2.weight": "conv.out_proj.weight",
    "ffn_gate.weight": "feed_forward.w1.weight",
    "ffn_up.weight": "feed_forward.w3.weight",
    "ffn_down.weight": "feed_forward.w2.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_q_norm.weight": "self_attn.q_layernorm.weight",
    "attn_k_norm.weight": "self_attn.k_layernorm.weight",
}
PROJ_KEYS = {
    "mm.a.mlp.0.weight": "layer_norm.weight",
    "mm.a.mlp.0.bias": "layer_norm.bias",
    "mm.a.mlp.1.weight": "linear_1.weight",
    "mm.a.mlp.1.bias": "linear_1.bias",
    "mm.a.mlp.3.weight": "linear_2.weight",
    "mm.a.mlp.3.bias": "linear_2.bias",
}


def map_name(n: str, swap_ffn: bool = False):
    """GGUF 张量名 -> 组合模型里的参数名。"""
    if n.startswith("a.conv1d."):
        base = n.rsplit(".", 1)[0]
        if base in CONV_KEYS:
            return f"encoder.downsample.{CONV_KEYS[base]}.{n.rsplit('.', 1)[1]}"
    if n == "a.pre_encode.out.weight":
        return "encoder.downsample.lin_out.weight"
    if n == "a.pre_encode.out.bias":
        return "encoder.downsample.lin_out.bias"
    if n == "a.post_ln.weight":
        return "encoder.lfm.embedding_norm.weight"
    if n in PROJ_KEYS:
        return "projector." + PROJ_KEYS[n]
    if n.startswith("a.blk."):
        _, _, idx, rest = n.split(".", 3)
        if rest in BLK_KEYS:
            tgt = BLK_KEYS[rest]
            if swap_ffn:
                tgt = tgt.replace("w1.", "w3.") if "w1." in tgt else tgt.replace("w3.", "w1.")
            return f"encoder.lfm.layers.{idx}." + tgt
    return None


def load_weights(model: torch.nn.Module, gguf_path: str, swap_ffn=False) -> int:
    g = Gguf(gguf_path)
    sd = model.state_dict()
    hit = {}
    unknown = []
    for n in g.tensors:
        tgt = map_name(n, swap_ffn)
        if tgt is None:
            unknown.append(n)
            continue
        if tgt not in sd:
            raise SystemExit(f"映射到不存在的参数: {n} -> {tgt}")
        w = sd[tgt]
        a = g.get(n)
        if a.size != w.numel():
            raise SystemExit(f"{n} -> {tgt}: 元素数 {a.size} != {w.numel()}")
        hit[tgt] = torch.from_numpy(a.reshape(tuple(w.shape)).copy())
    missing = [k for k in sd if k not in hit]
    if missing:
        print(f"[warn] {len(missing)} 个参数没被填充: {missing[:6]}", file=sys.stderr)
    sd.update(hit)
    model.load_state_dict(sd, strict=False)
    # 只把编码器/投影器该有的张量报出来，gen/wav 那些 TTS 的是另一条路
    skip = [n for n in unknown if not (n.startswith("a.gen.") or n.startswith("mm."))]
    if skip:
        print(f"[warn] 未映射 {len(skip)} 个张量: {skip[:6]}", file=sys.stderr)
    return len(hit)


class AudioTower(torch.nn.Module):
    """把 encoder + projector 拼起来，只为了名字好映射。"""

    def __init__(self, cfg):
        super().__init__()
        from lfm3_audio.modeling_lfm3_audio import Lfm3AudioEncoder, Lfm3AudioProjector
        self.encoder = Lfm3AudioEncoder(cfg.audio_encoder_config)
        self.projector = Lfm3AudioProjector(cfg)

    def forward(self, mel, lengths):
        h, n = self.encoder(mel, lengths)
        return self.projector(h), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16 kHz 单声道 16bit wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=DEFAULT_CFG)
    ap.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    ap.add_argument("--swap-ffn", action="store_true",
                    help="怀疑 ffn_gate/ffn_up 映射反了时用这个")
    a = ap.parse_args()

    from lfm3_audio.configuration_lfm3_audio import Lfm3AudioConfig
    from lfm3_audio.feature_extraction_lfm3_audio import Lfm3AudioFeatureExtractor

    cfg = Lfm3AudioConfig(**json.load(open(a.cfg)))
    model = AudioTower(cfg).eval()
    n = load_weights(model, a.mmproj, a.swap_ffn)
    print(f"装了 {n} 个张量", file=sys.stderr)

    with wave.open(a.wav, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "要 16k 单声道 16bit"
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = pcm.astype(np.float32) / 32768.0

    fe = Lfm3AudioFeatureExtractor()
    feats = fe(x, sampling_rate=sr, return_tensors="pt")
    mel = feats["input_features"]
    lengths = feats["feature_attention_mask"].sum(-1)
    print(f"mel {tuple(mel.shape)}  时长 {len(x)/sr:.2f}s", file=sys.stderr)

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "mel.npy"), mel[0].numpy())

    hs = {}

    def grab(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            hs[name] = t.detach().float().numpy()[0]
        return f

    hooks = [model.encoder.lfm.layers[i].register_forward_hook(grab(f"blk{i}"))
             for i in range(len(model.encoder.lfm.layers))]
    hooks.append(model.encoder.downsample.register_forward_hook(grab("down")))
    hooks.append(model.encoder.lfm.embedding_norm.register_forward_hook(grab("post_ln")))

    with torch.no_grad():
        mm, ntok = model(mel, lengths)
    for h in hooks:
        h.remove()

    np.save(os.path.join(a.out, "mm.npy"), mm[0].numpy())
    for k, v in hs.items():
        np.save(os.path.join(a.out, f"{k}.npy"), v)
    print(f"tokens {ntok.tolist()}  mm {tuple(mm.shape)}", file=sys.stderr)
    print(f"写到 {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
