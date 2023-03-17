# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
from functools import partial
import math
import bitsandbytes as bnb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from llama.util import make_linear, LoraWrapper


def init_8bit(loading_code, **kwargs):
    def fn():
        ORIG_LINEAR = torch.nn.__dict__['Linear']

        torch.nn.__dict__['Linear'] = Linear8bitLtInference
        try:
            result = loading_code(**kwargs)
        finally:
            torch.nn.__dict__['Linear'] = ORIG_LINEAR

        return result

    return fn


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_cache=False, use_xformers=True, 
                 use_checkpoint=False,
                 use_checkpoint_activations=True,
                 linear_kwargs=None,
                 quantize_cache=False,
                 quantize_cache_after_token=0,
                 ):
        super().__init__()

        linear_kwargs = linear_kwargs or {}

        self.n_local_heads = args.n_heads // 1
        self.head_dim = args.dim // args.n_heads

        self.wq = make_linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            **linear_kwargs,
        )
        self.wv = make_linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            **linear_kwargs,
        )
        self.wk = make_linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            **linear_kwargs,
        )
        self.wo = make_linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            **linear_kwargs,
        )

        self.use_cache = use_cache
        self.use_xformers = use_xformers
        self.use_checkpoint = use_checkpoint
        self.use_checkpoint_activations = use_checkpoint_activations
        self.quantize_cache = quantize_cache
        self.quantize_cache_after_token = quantize_cache_after_token

        self.mask = None
        if self.use_xformers:
            import xformers.ops as xops
            self.xops = xops
            self.mask = xops.LowerTriangularMask()

        self.cache_k_fp16 = None
        self.cache_v_fp16 = None
        self.cache_k_int8 = None
        self.cache_v_int8 = None

        if self.use_cache:
            cache_len_fp16 = args.max_seq_len
            cache_len_int8 = args.max_seq_len
            if quantize_cache_after_token > 0:
                cache_len_fp16 = quantize_cache_after_token
                cache_len_int8 = args.max_seq_len - quantize_cache_after_token

            cache_shape_fp16 = (
                args.max_batch_size, 
                cache_len_fp16,
                self.n_local_heads,
                self.head_dim,
            )
            cache_shape_int8 = (
                self.n_local_heads,
                self.head_dim,
                cache_len_int8,
            )

            uses_fp16_cache = False
            uses_int8_cache = False

            if quantize_cache:
                uses_int8_cache = True

                assert args.max_batch_size == 1

                self.SCB_shape = (cache_shape_int8[0], cache_shape_int8[1], 1)
                self.SCB_shape_dyn = (-1, cache_shape_int8[1], 1)
                self.SCB_k = torch.zeros(self.SCB_shape, device='cuda')
                self.SCB_v = torch.zeros(self.SCB_shape, device='cuda')

                if self.quantize_cache_after_token > 0:
                    uses_fp16_cache = True
            else:      
                uses_fp16_cache = True

            if uses_fp16_cache:
                self.cache_k_fp16 = torch.zeros(
                    cache_shape_fp16,
                    dtype=self.wk.weight.dtype,
                ).cuda()
                self.cache_v_fp16 = torch.zeros(
                    cache_shape_fp16,
                    dtype=self.wk.weight.dtype,
                ).cuda()
            if uses_int8_cache:
                self.cache_k_int8 = torch.zeros(
                    cache_shape_int8,
                    dtype=torch.int8,
                ).cuda()
                self.cache_v_int8 = torch.zeros(
                    cache_shape_int8,
                    dtype=torch.int8,
                ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, start_pos, freqs_cis, mask)
        return self._forward(x, start_pos, freqs_cis, mask)

    def _forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.use_checkpoint_activations:
            xq, xk = checkpoint(apply_rotary_emb, xq, xk, freqs_cis)
        else:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        using_int8_cache = False
        transfer_to_int8_cache = False

        if self.use_cache and self.quantize_cache:
            using_int8_cache = start_pos > self.quantize_cache_after_token
            transfer_to_int8_cache = (not using_int8_cache) and start_pos + seqlen > self.quantize_cache_after_token

        if self.use_cache:
            if using_int8_cache:
                xkc = xk.view(seqlen, self.n_local_heads, self.head_dim).permute(
                    (1, 2, 0)).reshape(-1, seqlen)
                xvc = xv.view(seqlen, self.n_local_heads, self.head_dim).permute(
                    (1, 2, 0)).reshape(-1, seqlen)

                if start_pos > 0:
                    xq_k_cache = self.cache_k_int8[:, :, : start_pos]
                    xq_k_cache = xq_k_cache.reshape(-1, start_pos)
                    max1_k_cache = self.SCB_k.view(-1, 1)

                    pastk = bnb.functional.vectorwise_dequant(
                        xq_k_cache, max1_k_cache,).half()

                    keys = torch.cat([pastk, xkc], dim=-1)
                else:
                    keys = xkc

                xq_k, max1 = bnb.functional.vectorwise_quant(keys, dim=1)

                keys = keys.reshape(
                    self.n_local_heads, self.head_dim, -1).permute((2, 0, 1))[None, :].contiguous()

                cache_scatter = xq_k.reshape(self.n_local_heads, self.head_dim, -1)

                self.cache_k_int8[:, :, : start_pos + seqlen] = cache_scatter

                self.SCB_k[:] = max1.view(*self.SCB_shape_dyn)

                if start_pos > 0:
                    xq_v_cache = self.cache_v_int8[:, :, : start_pos]
                    xq_v_cache = xq_v_cache.reshape(-1, start_pos)
                    max1_v_cache = self.SCB_v.view(-1, 1)

                    pastv = bnb.functional.vectorwise_dequant(
                        xq_v_cache, max1_v_cache,).half()
                    values = torch.cat([pastv, xvc], dim=1)
                else:
                    values = xvc

                xq_v, max1 = bnb.functional.vectorwise_quant(values, dim=1)

                values = values.reshape(
                    self.n_local_heads, self.head_dim, -1).permute((2, 0, 1))[None, :].contiguous()

                cache_scatter = xq_v.reshape(self.n_local_heads, self.head_dim, -1)
                self.cache_v_int8[:, :, : start_pos + seqlen] = cache_scatter
                self.SCB_v[:] = max1.view(*self.SCB_shape_dyn)
            else:
                self.cache_k_fp16 = self.cache_k_fp16.to(xq)
                self.cache_v_fp16 = self.cache_v_fp16.to(xq)

                if transfer_to_int8_cache:
                    cache_k = self.cache_k_fp16[:bsz, : start_pos]

                    keys = torch.cat([cache_k, xk], dim=1)

                    xkc = keys.view(start_pos + seqlen, self.n_local_heads, self.head_dim).permute(
                        (1, 2, 0)
                    ).reshape(-1, start_pos + seqlen)

                    xq_k, max1 = bnb.functional.vectorwise_quant(xkc, dim=1)

                    cache_scatter = xq_k.reshape(self.n_local_heads, self.head_dim, -1)

                    self.cache_k_int8[:, :, : start_pos + seqlen] = cache_scatter

                    cache_v = self.cache_v_fp16[:bsz, : start_pos]

                    values = torch.cat([cache_v, xv], dim=1)

                    xvc = values.view(start_pos + seqlen, self.n_local_heads, self.head_dim).permute(
                        (1, 2, 0)
                    ).reshape(-1, start_pos + seqlen)

                    xq_v, max1 = bnb.functional.vectorwise_quant(xvc, dim=1)

                    cache_scatter = xq_v.reshape(
                        self.n_local_heads, self.head_dim, -1)

                    self.cache_v_int8[:, :, : start_pos + seqlen] = cache_scatter
                else:
                    self.cache_k_fp16[:bsz, start_pos: start_pos + seqlen] = xk
                    self.cache_v_fp16[:bsz, start_pos: start_pos + seqlen] = xv

                    keys = self.cache_k_fp16[:bsz, : start_pos + seqlen]
                    values = self.cache_v_fp16[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        if self.use_xformers:
            output = self.xops.memory_efficient_attention(
                xq, keys, values, attn_bias=self.mask if seqlen > 1 else None
            )
        else:
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            scores = torch.matmul(xq, keys.transpose(2, 3)) / \
                math.sqrt(self.head_dim)
            if mask is not None:
                # (bs, n_local_heads, slen, cache_len + slen)
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # (bs, n_local_heads, slen, head_dim)
            output = torch.matmul(scores, values)

            output = output.transpose(
                1, 2
            ).contiguous()

        output = output.view(bsz, seqlen, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        use_checkpoint=False,
        use_checkpoint_activations=True,
        linear_kwargs=None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        linear_kwargs = linear_kwargs or {}

        self.w1 = make_linear(
            dim, hidden_dim, bias=False,
            **linear_kwargs,
        )
        self.w2 = make_linear(
            hidden_dim, dim, bias=False, 
            **linear_kwargs,
        )
        self.w3 = make_linear(
            dim, hidden_dim, bias=False, 
            **linear_kwargs,
        )

        self.use_checkpoint = use_checkpoint
        self.use_checkpoint_activations = use_checkpoint_activations

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,)
        return self._forward(x,)
    
    def _silu_mm(self, y, x):
        return F.silu(y) * self.w3(x)

    def silu_mm(self, y, x):
        if self.use_checkpoint_activations:
            return checkpoint(self._silu_mm, y, x)
        return self._silu_mm(y, x)

    def _forward(self, x):
        return self.w2(self.silu_mm(self.w1(x), x))
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, use_xformers=True, use_checkpoint=False, use_checkpoint_activations=True,
                 use_cache=False, 
                 linear_kwargs=None,
                 quantize_cache=False,
                 quantize_cache_after_token=0,
                 ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, 
                                   use_xformers=use_xformers, 
                                   use_checkpoint=False,
                                   use_checkpoint_activations=use_checkpoint_activations,
                                   use_cache=use_cache,
                                   linear_kwargs=linear_kwargs,
                                   quantize_cache=quantize_cache,
                                   quantize_cache_after_token=quantize_cache_after_token,
                                   )
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of,
            use_checkpoint=False,
            use_checkpoint_activations=use_checkpoint_activations,
            linear_kwargs=linear_kwargs,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, start_pos, freqs_cis, mask)
        return self._forward(x, start_pos, freqs_cis, mask)

    def _forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h=x + self.attention.forward(self.attention_norm(x),
                                     start_pos, freqs_cis, mask)
        out=h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, 
                 use_xformers: bool = True,
                 use_checkpoint=True,
                 n_checkpoint_segments=4,
                 freeze_layers_below_n=0,
                 quantize_frozen=True,
                 quantize_threshold=6,
                 use_cache=False,
                 use_lora=True, 
                 use_lora_checkpoint=False,
                 lora_r=16,
                 linear_device=None,
                 fp32_logits=True,
                 allow_quantize_unembed=True,
                 quantize_cache=False,
                 quantize_cache_above=0,
                 quantize_cache_after_token=0,):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.freeze_layers_below_n = freeze_layers_below_n
        self.use_checkpoint = use_checkpoint
        self.n_checkpoint_segments = n_checkpoint_segments
        self.fp32_logits = fp32_logits

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim, 
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            base_weights_frozen = use_lora or (
                layer_id < self.freeze_layers_below_n)
            
            linear_kwargs = dict(
                use_lora=use_lora and layer_id >= self.freeze_layers_below_n,
                lora_kwargs=dict(r=lora_r, use_checkpoint=use_lora_checkpoint),
                use_8bit=quantize_frozen and base_weights_frozen,
                bnb_kwargs=dict(threshold=quantize_threshold),
                device=linear_device,
            )
            def make_layer(): return TransformerBlock(
                layer_id, params,
                use_xformers=use_xformers,
                use_checkpoint=False,
                use_checkpoint_activations=False,
                use_cache=use_cache,
                quantize_cache=quantize_cache and layer_id >= quantize_cache_above,
                quantize_cache_after_token=quantize_cache_after_token,
                linear_kwargs=linear_kwargs,
            )
            self.layers.append(make_layer())

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        linear_kwargs = dict(
            use_lora=use_lora,
            lora_kwargs=dict(r=lora_r, use_checkpoint=use_lora_checkpoint),
            use_8bit=quantize_frozen and use_lora and allow_quantize_unembed,
            bnb_kwargs=dict(threshold=quantize_threshold),
        )
        self.output = make_linear(
            params.dim, params.vocab_size, bias=False, 
            **linear_kwargs
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.use_xformers = use_xformers

        for layer in self.layers[:self.freeze_layers_below_n]:
            layer.requires_grad_(False)

        if self.freeze_layers_below_n > 0:
            self.tok_embeddings.requires_grad_(False)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if not self.use_xformers and seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        if self.use_checkpoint:
            for layer in self.layers[:self.freeze_layers_below_n]:
                h = layer(h, start_pos, freqs_cis, mask)
            h.requires_grad_(True)

            fwds = [partial(layer.forward, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
                    for layer in self.layers[self.freeze_layers_below_n:]]

            h = checkpoint_sequential(
                fwds, self.n_checkpoint_segments,
                h
            )
        else:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)

        if self.fp32_logits:
            return output.float()
        
        return output

    def merge_lora_into_base(self):
        def _merge_lora_into_base(mod):
            if isinstance(mod, LoraWrapper):
                mod.merge_lora_into_base()
        self.apply(_merge_lora_into_base)
