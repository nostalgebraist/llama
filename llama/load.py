from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import gc

from pathlib import Path

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load_state_dict_meta(module, sd, device_param_copy, device_mod, lora_checkpoint=None, delete_after=True):
    outs = {'unexpected_keys': []}
    
    sd_keys = sorted(sd.keys())
    not_loaded_keys = set(sd_keys)

    has_lora_checkpoint = lora_checkpoint is not None

    lora_keys = []
    if has_lora_checkpoint:
        lora_keys = sorted(lora_checkpoint.keys())
        sd_keys = sorted(sd_keys + lora_keys)
        not_loaded_keys = set(sd_keys)

    with torch.no_grad():
        for tensor_name in sd_keys:
            # print(f"loading {tensor_name}")
            src = lora_checkpoint if 'lora_' in tensor_name else sd
            new_value = src[tensor_name]
            not_loaded_keys.remove(tensor_name)
            if delete_after:
                del src[tensor_name]
                gc.collect()

            is_buffer = tensor_name in module._buffers
            if is_buffer:
                module._buffers[tensor_name] = new_value
            else:
                try:
                    old_value = module.get_parameter(tensor_name)
                except:
                    outs['unexpected_keys'].append(tensor_name)
                    continue

                param_cls = type(old_value)
                kwargs = old_value.__dict__
                new_value = param_cls(
                    new_value, requires_grad=old_value.requires_grad, **kwargs).to(device_param_copy)

                submod_name, _, submod_tensor_name = tensor_name.rpartition(
                    '.')

                print(f"loading {tensor_name} ({new_value.dtype}) to {submod_name} ({old_value.dtype})")

                module.get_submodule(
                    submod_name)._parameters[submod_tensor_name] = new_value

                ready_to_transfer = not any(
                    k.startswith(submod_name + '.') for k in not_loaded_keys)

                if ready_to_transfer:
                    print(f"transferring {submod_name} to {device_mod}")
                    if has_lora_checkpoint and hasattr(module.get_submodule(submod_name), 'merge_lora_into_base'):
                        module.get_submodule(submod_name).merge_lora_into_base()
                    module.get_submodule(submod_name).to(device=device_mod)
                else:
                    pass
                    # print(f"not transferring {submod_name}: params still in sd {[k for k in sd if k.startswith(submod_name + '.')]}")
    return outs


toggle = {0: 0}

def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, n_ctx, ckpt_path=None, use_cache=False,
         use_xformers=True, 
         freeze_layers_below_n=0,
         use_lora=True,
         lora_r=16,
         use_lora_checkpoint=False,
         quantize_frozen=True,
         quantize_threshold=6,
         use_checkpoint=False,
         n_checkpoint_segments=4,
         max_batch_size=1,
         lowmem=False,
         lowmem_cpu_ratio=1,
         fp32_logits=True,
         checkpoint=None,
         lora_checkpoint=None,
         bf16=False,
         rope_scaling_factor=1.0,
         **kwargs,
         ) -> LLaMA:
    start_time = time.time()

    if not ckpt_path and not checkpoint:
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert (
            world_size == len(checkpoints)
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]

    def maploc(storage, loc):
        global toggle
        toggle[0] = toggle[0] + 1
        if toggle[0] % (lowmem_cpu_ratio + 1) or lowmem_cpu_ratio == 0:
            return storage.cuda(0)
        else:
            return storage.cpu()

    print("Loading")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=n_ctx, max_batch_size=max_batch_size, rope_scaling_factor=rope_scaling_factor, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    if bf16:
        torch.set_default_tensor_type(torch.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args,
                        use_cache=use_cache,
                        use_xformers=use_xformers,
                        freeze_layers_below_n=freeze_layers_below_n,
                        use_lora=use_lora,
                        use_lora_checkpoint=use_lora_checkpoint,
                        lora_r=lora_r,
                        quantize_frozen=quantize_frozen,
                        quantize_threshold=quantize_threshold,
                        use_checkpoint=use_checkpoint,
                        n_checkpoint_segments=n_checkpoint_segments,
                        linear_device='meta' if lowmem else None,
                        fp32_logits=fp32_logits,
                        **kwargs
                        )
    torch.set_default_tensor_type(torch.FloatTensor)

    if not checkpoint:
        checkpoint = torch.load(ckpt_path, map_location=maploc if (lowmem and not quantize_frozen) else "cpu")

    if isinstance(lora_checkpoint, str):
        lora_checkpoint = torch.load(lora_checkpoint, map_location=maploc if (lowmem and not quantize_frozen) else "cpu")

    if lowmem:
        outs = load_state_dict_meta(
            model, checkpoint, 'cpu', 'cuda:0', lora_checkpoint)
    else:
        outs = model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator