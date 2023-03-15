# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List
from functools import partial

import torch
from tqdm.auto import tqdm, trange

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from llama.breakruns import BreakrunsLogitsProcessor


def xformers_off(m):
    m.use_xformers=False

def xformers_on(m):
    m.use_xformers=True


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.tokens = torch.full(
            (self.model.params.max_batch_size, self.model.params.max_seq_len),
            self.tokenizer.pad_id
        ).cuda().long()

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_at_eos=True,
        allow_xformers=False,
        all_xformers=False,
        breakruns=True,
        breakruns_tau=0.035,
        print_breakruns_stats=False,
        debug=False,
        progress_bar=True,
        return_stop_reason=False,
        decode=True,
        extra_logits_processors=None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        if isinstance(prompts[0], str):
            prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        else:
            # pre encoded
            prompt_tokens = prompts

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        # tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        self.tokens[:] = self.tokenizer.pad_id

        for k, t in enumerate(prompt_tokens):
            self.tokens[k, : len(t)] = torch.tensor(t).long()
        
        input_text_mask = self.tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        lp = None
        stop_reason = None
        extra_lp = extra_logits_processors or []
        if breakruns:
            lp = BreakrunsLogitsProcessor(base_temperature=temperature, 
                                        tau=breakruns_tau, 
                                        tokenizer=self.tokenizer,
                                        debug=debug)
            temperature = 1.0

        ranger = partial(trange, mininterval=0.25) if progress_bar else range

        for cur_pos in ranger(start_pos, total_len):
            if allow_xformers and (all_xformers or (cur_pos - prev_pos > 1)):
                self.model.apply(xformers_on);
            else:
                self.model.apply(xformers_off);

            logits = self.model.forward(self.tokens[:, prev_pos:cur_pos], prev_pos)[:, -1, :]

            if lp is not None:
                logits = lp(self.tokens[:, start_pos:cur_pos], logits)

            for elp in extra_lp:
                logits = elp(self.tokens[:, start_pos:cur_pos], logits)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], self.tokens[:, cur_pos], next_token
            )
            self.tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            if stop_at_eos and bsz == 1 and (next_token == self.tokenizer.eos_id):
                stop_reason = 'eos'
                break
        
        if stop_reason is None:
            stop_reason = 'max_length'

        output = []
        if breakruns and print_breakruns_stats:
            temp_avg = lp.temp_avg.item() if len(lp.temp_avg) == 1 else lp.temp_avg.cpu().numpy()
            print(f"temp_avg: {temp_avg}")
            if lp.tau > 0:
                counter_avg = (temp_avg - lp.base_temperature) / lp.tau
                print(f"counter_avg: {counter_avg}")

            max_run_len = lp.max_run_len.item() if len(lp.max_run_len) == 1 else lp.max_run_len.cpu().numpy()
            print(f"max_run_len: {max_run_len}")

        for i, t in enumerate(self.tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            try:
                t = t[: t.index(-1)]
            except ValueError:
                pass

            if decode:
                try:
                    output.append(self.tokenizer.decode(t))
                except Exception as e:
                    print(repr(e))
                    return t
            else:
                output.append(t)
        
        if return_stop_reason:
            return output, stop_reason
        return output



def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
