#!/usr/bin/env python3
"""mlx_lfm3.py — 在苹果 MLX 运行时上跑 LFM3，作为天枢的**参考实现**与速度基准

为什么需要这个文件：mlx-lm 0.32.2 只有 lfm2，没有 lfm3。而 Liquid 官方发布的
`LFM3-1.7B-MLX-4bit/8bit` 是 MLX 格式权重，必须有一个 mlx-lm 能识别的模型定义
才能加载。这个文件补上这一块，于是有了：

  1. 一条**完全独立**的前向实现（Python/MLX），可以用来验证 C++ 引擎算的是
     不是同一个东西 —— 比"自己和自己对拍"强得多；
  2. 苹果官方推理栈在同一个权重上的真实速度，作为"苹果的版本"的基准。

架构对齐（与 LiquidAI 的 config.json / modeling_lfm3.py 逐项核对）：

  * layer_types 逐层三分：conv / sliding_attention / full_attention
  * ShortConv：in_proj 出 [B|C|x]，hidden = B⊙x，depthwise causal conv1d
    （核宽 conv_L_cache，左补 L-1），y = C⊙conv，再 out_proj
  * 注意力：q/k 先过 per-head RMSNorm（在 RoPE **之前**），
    sigmoid(gate_proj(x)) 逐 head 缩放注意力输出（在 o_proj **之前**）
  * RoPE 是 zero-tail partial：只有前 `int(prf*head_dim)//2` 个角度真转，
    频率分母恒为 head_dim，其余角度置 0。**按层类型分别配置**：
    full_attention 的 partial_rotary_factor=0.5（32 角度，base 1e6），
    sliding_attention 缺省 1.0（64 角度 = 全量，base 1e4）
  * sandwich norm：子层输出先过 post_*_norm 再加残差

用法:
    python tools/mlx_lfm3.py <model_dir> "The capital of France is" [n_tokens]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.rope_utils import initialize_rope


# --------------------------------------------------------------------- 配置

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "lfm3"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 7680
    vocab_size: int = 128000
    max_position_embeddings: int = 262144
    norm_eps: float = 1e-5
    conv_L_cache: int = 7
    conv_bias: bool = False
    sliding_window: int = 256
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[dict] = None
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["conv"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 1e6,
                                   "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
            }


# --------------------------------------------------------------------- RoPE

class ZeroTailRoPE(nn.Module):
    """zero-tail partial RoPE。

    HF 的公式：inv_freq = 1/(base ** (arange(0, 2*angles, 2) / head_dim))，
    尾部补 0 到 head_dim/2。补 0 的那些角度 cos=1/sin=0，等于不旋转 —— 这就是
    "zero-tail" 名字的来历。注意频率分母恒为 head_dim，不是 2*angles。
    """

    def __init__(self, head_dim: int, base: float, partial_factor: float = 1.0,
                 traditional: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.traditional = traditional
        angles = int(partial_factor * head_dim) // 2
        freqs = 1.0 / (base ** (mx.arange(0, 2 * angles, 2, dtype=mx.float32) / head_dim))
        if angles < head_dim // 2:
            freqs = mx.concatenate(
                [freqs, mx.zeros((head_dim // 2 - angles,), dtype=mx.float32)])
        self._inv_freqs = freqs

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: [B, H, L, D]
        L = x.shape[-2]
        pos = mx.arange(offset, offset + L, dtype=mx.float32)
        ang = pos[:, None] * self._inv_freqs[None, :]        # [L, D/2]
        cos = mx.cos(ang)[None, None, :, :].astype(x.dtype)
        sin = mx.sin(ang)[None, None, :, :].astype(x.dtype)
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        if self.traditional:
            return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        # 非 traditional（rotate_half 风格）：相邻两维配对
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


# --------------------------------------------------------------------- 子层

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        lt = args.layer_types[layer_idx]
        self.sliding_window = args.sliding_window if lt == "sliding_attention" else None

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)

        rp = args.rope_parameters[lt]
        self.rope = ZeroTailRoPE(self.head_dim, base=rp["rope_theta"],
                                 partial_factor=rp.get("partial_rotary_factor", 1.0))

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        off = cache.offset if cache is not None else 0
        q = self.rope(q, off)
        k = self.rope(k, off)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, mask=mask, scale=self.scale)
        gate = mx.sigmoid(self.gate_proj(x))[..., None]     # [B, L, H, 1]
        out = out.transpose(0, 2, 1, 3) * gate
        return self.o_proj(out.reshape(B, L, -1))


class ShortConv(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.L_cache = args.conv_L_cache
        self.conv = nn.Conv1d(args.hidden_size, args.hidden_size,
                              kernel_size=self.L_cache, groups=args.hidden_size,
                              bias=args.conv_bias)
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=args.conv_bias)

    def __call__(self, x, mask=None, cache=None):
        B = x.shape[0]
        BCx = self.in_proj(x)
        b, c, xx = mx.split(BCx, 3, axis=-1)
        bx = b * xx

        if cache is not None:
            state = cache[0]
            if state is None:
                state = mx.zeros((B, self.L_cache - 1, x.shape[-1]), dtype=bx.dtype)
            bx = mx.concatenate([state, bx], axis=1)
            cache[0] = bx[:, -(self.L_cache - 1):, :]
            cache.advance(bx.shape[1] - (self.L_cache - 1))
        else:
            bx = mx.pad(bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])

        y = c * self.conv(bx)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)   # gate
        self.w3 = nn.Linear(dim, ff_dim, bias=False)   # up
        self.w2 = nn.Linear(ff_dim, dim, bias=False)   # down

    def __call__(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_attention = self.layer_type != "conv"
        if self.is_attention:
            self.self_attn = Attention(args, layer_idx)
        else:
            self.conv = ShortConv(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.use_sandwich = args.use_sandwich_norm
        if self.use_sandwich:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        res = x
        h = self.operator_norm(x)
        h = self.self_attn(h, mask=mask, cache=cache) if self.is_attention \
            else self.conv(h, cache=cache)
        if self.use_sandwich:
            h = self.post_operator_norm(h)
        h = h + res

        f = self.feed_forward(self.ffn_norm(h))
        if self.use_sandwich:
            f = self.post_ffn_norm(f)
        return h + f


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # 掩码按"层类型"取，conv 层用不上 mask（靠 cache 递推）
        attn_mask = create_attention_mask(h, next((c for c in cache if isinstance(c, KVCache)), None))
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=attn_mask if layer.is_attention else None, cache=c)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.model.embed_tokens.as_linear(self.model(inputs, cache, input_embeddings))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() if l.is_attention else ArraysCache(size=1) for l in self.layers]

    def sanitize(self, weights):
        """两件改名：

        1. 权重键带 `language_model.` 前缀（transformers 5.x 把文本塔包在
           多模态外壳里，即使没有视觉权重）。剥掉即可。
        2. HF 的 depthwise conv 权重是 [C, 1, L]，MLX 的 Conv1d 要 [C, L, 1]。
        """
        out = {}
        for name, p in weights.items():
            if name.startswith("language_model."):
                name = name[len("language_model."):]
            if name.endswith("conv.conv.weight") and p.ndim == 3 and p.shape[1] == 1:
                p = p.transpose(0, 2, 1)
            out[name] = p
        return out


# --------------------------------------------------------------------- 注册

def register() -> None:
    """把本模块塞进 mlx_lm.models，让 mlx_lm.load() 认得 model_type=lfm3。"""
    import mlx_lm.models as M
    sys.modules["mlx_lm.models.lfm3"] = sys.modules[__name__]
    setattr(M, "lfm3", sys.modules[__name__])


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_dir, prompt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    register()
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import greedy_sampler

    model, tok = load(model_dir)

    # 关键：传 token id 而不是字符串。mlx_lm 收到字符串会自动套 chat 模板，
    # 而天枢那边跑的是 --raw，两边 prompt 不一样的话速度和质量都不可比。
    # tok.encode 默认会加 BOS，这里显式去掉（LFM3 的 GGUF 里 add_bos=false）。
    ids = tok.encode(prompt)
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]

    # 用 stream_generate 分开 prefill 与 decode：它会在第一个 token 上带出
    # prompt_tps / generation_tps，正好对应天枢的 prefill / decode 两项。
    t_first, n_out, text = None, 0, ""
    for r in stream_generate(model, tok, prompt=ids, max_tokens=n,
                             sampler=greedy_sampler):
        if t_first is None:
            t_first = r.prompt_tps
        n_out += 1
        text += r.text
    gen_tps = r.generation_tps if n_out else 0.0

    print(json.dumps({
        "tokens_in": len(ids),
        "tokens_out": n_out,
        "prefill_tps": t_first or 0.0,
        "decode_tps": gen_tps,
        "text": text,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
