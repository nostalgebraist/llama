import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import bitsandbytes.nn.modules
import bitsandbytes.functional
import bitsandbytes as bnb


"""
monkeypatch Int8Params.cuda() to make it idempotent
"""


def patched_cuda(self, device):
    if self.has_fp16_weights:
        return super(bitsandbytes.nn.modules.Int8Params, self).cuda(device)
    else:
        if getattr(self, 'CB', None) is None:
            # we store the 8-bit rows-major weight
            # we convert this weight to the turning/ampere weight during the first inference pass
            B = self.data.contiguous().half().cuda(device)
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            self.data = CB
            setattr(self, "CB", CB)
            setattr(self, "SCB", SCB)
        else:
            pass
            # print(f"skipping double cuda with CB {self.CB.dtype}")

    return self


bitsandbytes.nn.modules.Int8Params.cuda = patched_cuda


def cuda4(self, device):
    if self.quant_state is not None:
        return self
    w = self.data.contiguous().half().cuda(device)
    w_4bit, quant_state = bnb.functional.quantize_4bit(
        w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
    self.data = w_4bit
    self.quant_state = quant_state

    return self


bitsandbytes.nn.modules.Params4bit.cuda = cuda4


def silent_forward(self, x: th.Tensor):
    # weights are cast automatically as Int8Params, but the bias has to be cast manually
    if self.bias is not None and self.bias.dtype != x.dtype:
        self.bias.data = self.bias.data.to(x.dtype)

    if getattr(self.weight, 'quant_state', None) is None:
        print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
    inp_dtype = x.dtype
    if self.compute_dtype is not None:
        x = x.to(self.compute_dtype)

    bias = None if self.bias is None else self.bias.to(self.compute_dtype)
    out = bnb.matmul_4bit(x, self.weight.t(), bias=bias,
                            quant_state=self.weight.quant_state)

    out = out.to(inp_dtype)
    # # don't do next line
    # print(self.weight.quant_state)

    return out


bitsandbytes.nn.modules.Linear4bit.forward = silent_forward


# avoid overhead with 1 gpu
def pre_call(device):
    return device


def post_call(prev_device):
    return


bitsandbytes.functional.pre_call = pre_call
bitsandbytes.functional.post_call = post_call


def vectorwise_quant(x, dim=1):
    C = 127.0

    max1 = th.amax(th.abs(x), dim=dim, keepdim=True)
    xq = th.round(x * (C / max1)).to(th.int8)
    return xq, max1


def vectorwise_dequant(xq, max1, dtype):
    C = 127.0
    return (xq / C * max1).to(dtype=dtype)


class HookedDict(dict):
    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if hasattr(self, 'hook'):
            self.hook(k, v)


class Wrapper(nn.Module):
    def __init__(self, child: nn.Module):
        super().__init__()
        self.__dict__['_parameters'] = HookedDict()
        self._store = {'child': child}

        def hook(k, v):
            if k in self._store['child']._parameters:
                self._store['child']._parameters[k] = v

        self.__dict__['_parameters'].hook = hook

        for n, p in child.named_parameters():
            self.register_parameter(n, p)

        # TODO: hook buffers
        for n, p in child._buffers.items():
            self.register_buffer(
                n, p, persistent=n not in child._non_persistent_buffers_set)

    def forward(self, *args, **kwargs):
        return self.child.forward(*args, **kwargs)

    def extra_repr(self):
        return f'child={self.child}'

    @property
    def child(self):
        return self._store['child']


class LoraWrapper(Wrapper):
    def __init__(self,
                 child: nn.Module,
                 r: int,
                 lora_alpha=1,
                 use_checkpoint=False,
                 autocast=False,
                 ):
        super().__init__(child)
        assert hasattr(self.child, 'in_features')
        assert hasattr(self.child, 'out_features')

        self.r = r
        self.lora_alpha = lora_alpha
        self.use_checkpoint = use_checkpoint
        self.scaling = self.lora_alpha / max(1, self.r)
        self.autocast = autocast

        if self.r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((self.child.in_features, r)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((r, self.child.out_features)))
            self.child.requires_grad_(False)

        self.reset_lora_parameters()

    def reset_parameters(self):
        self.child.reset_parameters()
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge_lora_into_base(self):
        if self.r <= 0:
            return

        assert hasattr(self.child, 'weight')
        assert self.child.weight.shape == (
            self.child.out_features, self.child.in_features)

        with th.no_grad():
            patch = self.scaling * (self.lora_A @ self.lora_B).T
            self.child.weight.data = (
                self.child.weight.data.to(dtype=patch.dtype, device=patch.device) +
                patch
            ).to(dtype=self.child.weight.data.dtype, device=self.child.weight.data.device)
            del self.lora_A
            del self.lora_B
            self.r = 0

    def _lora_down(self, x):
        if self.autocast:
            with th.cuda.amp.autocast(enabled=True, dtype=x.dtype):
                return x @ self.lora_A
        return x.to(self.lora_A.dtype) @ self.lora_A

    def _lora_up(self, y):
        if self.autocast:
            with th.cuda.amp.autocast(enabled=True, dtype=y.dtype):
                return (self.scaling * y @ self.lora_B)
        return (self.scaling * y @ self.lora_B)

    def forward(self, x):
        if self.use_checkpoint:
            out = checkpoint(self.child.forward, x)
        else:
            out = self.child(x)
        if self.r > 0:
            if self.use_checkpoint:
                y = self._lora_down(x)
                delta = checkpoint(self._lora_up, y).to(out.dtype)
            else:
                delta = self._lora_up(self._lora_down(x)).to(out.dtype)
            out = out + delta
        return out

    def extra_repr(self):
        return f'child={self.child}, r={self.r}'


def make_linear(
    in_features, out_features, bias=True,
    use_8bit=False,
    use_lora=False,
    lora_kwargs=None,
    bnb_kwargs=None,
    bnb_has_fp16_weights=False,
    device=None,
):
    lora_kwargs = lora_kwargs or {}
    bnb_kwargs = bnb_kwargs or {}
    bnb_kwargs['has_fp16_weights'] = bnb_has_fp16_weights

    linear_kwargs = dict()
    if device:
        linear_kwargs['device'] = device

    if use_lora:
        assert 'r' in lora_kwargs

    if use_8bit:
        for k in ['has_fp16_weights', 'threshold']:
            if k in bnb_kwargs:
                del bnb_kwargs[k]
        base = bnb.modules.Linear4bit(
            in_features, out_features, bias, quant_type='nf4', **bnb_kwargs)
    else:
        base = nn.Linear(in_features, out_features, bias, **linear_kwargs)

    if use_lora:
        return LoraWrapper(child=base, **lora_kwargs)

    return base
