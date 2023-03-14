import torch

class BreakrunsLogitsProcessor:
    """
    This is a sampling trick I use to avoid "pathological repetition" in autoregressive LM sampling.

    Description at https://nostalgebraist.tumblr.com/post/648042918390759424/breakruns
    """
    def __init__(self,
                base_temperature: float,
                tau: float,
                debug=True,
                tokenizer=None,
                first_count=0,
                disable_trigger = None,
                enable_trigger = None,
                modify_on_trigger = None,
                modify_off_trigger = None,
                temp_shift_modifier = 0.0,
                device='cuda:0',
                ):
        self.base_temperature = base_temperature
        self.tau = tau
        self.debug = debug
        self.tokenizer = tokenizer
        self.first_count = first_count
        self.disable_trigger = None if disable_trigger is None else torch.as_tensor(disable_trigger)[None, :].to(device)
        self.enable_trigger = None if enable_trigger is None else torch.as_tensor(enable_trigger)[None, :].to(device)
        self.modify_on_trigger = None if modify_on_trigger is None else torch.as_tensor(modify_on_trigger)[None, :].to(device)
        self.modify_off_trigger = None if modify_off_trigger is None else torch.as_tensor(modify_off_trigger)[None, :].to(device)
        self.temp_shift_modifier = temp_shift_modifier

        self.enabled = None
        self.modified = None
        self.breakruns_counter = None
        self.last_logits = None
        self.last_length = None
        self.first_length = None
        self.temp_avg = None
        self.max_run_len = None

    def _reset(self):
        self._dprint("BREAKRUNS: _reset")
        self.enabled = None
        self.modified = None
        self.breakruns_counter = None
        self.last_logits = None
        self.temp_avg = None
        self.max_run_len = None

    def _dprint(self, msg, fillers={}, **kwargs):
        if self.debug:
            print(msg.format(**fillers), **kwargs)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_length = input_ids.shape[1]
        if seq_length < 1:
            self._dprint("BREAKRUNS: empty sequence, no op")
            return scores

        if self.last_length is None or self.last_length > input_ids.shape[1]:
            # new sequence
            self._reset()
        self.last_length = input_ids.shape[1]
        if self.first_length is None:
            self.first_length = self.last_length

        if self.enabled is None:
            self._dprint("BREAKRUNS: init enabled")
            self.enabled = torch.ones((input_ids.shape[0],), device=input_ids.device)

        if self.modified is None:
            self._dprint("BREAKRUNS: init modified")
            self.modified = torch.zeros((input_ids.shape[0],), device=input_ids.device)

        if self.breakruns_counter is None:
            self.breakruns_counter = self.first_count * torch.ones((), device=input_ids.device)
            self.temp_avg = torch.zeros((), device=input_ids.device)
            self.max_run_len = torch.zeros((), device=input_ids.device, dtype=self.breakruns_counter.dtype)
            self._dprint(f"BREAKRUNS: init counter {self.breakruns_counter}")

        if self.last_logits is None:
            self._dprint("BREAKRUNS: init logits, no op")
            self.last_logits = scores

            return scores

        if (self.disable_trigger is not None) and (self.enable_trigger is not None):
            with torch.no_grad():
                prev_enabled = self.enabled

                disable_flip = (input_ids[:, -self.disable_trigger.shape[1]:] == self.disable_trigger.to(input_ids.device)).all(dim=1)
                enable_flip = (input_ids[:, -self.enable_trigger.shape[1]:] == self.enable_trigger.to(input_ids.device)).all(dim=1)

                self.enabled = torch.logical_or(enable_flip, self.enabled)
                self.enabled = torch.logical_and(~disable_flip, self.enabled)

                if self.debug and (self.enabled != prev_enabled).any().item():
                    self._dprint(f"BREAKRUNS: enabled {prev_enabled} -> {self.enabled}")

        # TODO: DRY
        if (self.modify_on_trigger is not None) and (self.modify_off_trigger is not None):
            with torch.no_grad():
                prev_modified = self.modified

                disable_flip = (input_ids[:, -self.modify_off_trigger.shape[1]:] == self.modify_off_trigger.to(input_ids.device)).all(dim=1)
                enable_flip = (input_ids[:, -self.modify_on_trigger.shape[1]:] == self.modify_on_trigger.to(input_ids.device)).all(dim=1)

                self._dprint(f"modify on: {input_ids[:, -self.modify_on_trigger.shape[1]:]} vs {self.modify_on_trigger}")

                # note reversed order
                self.modified = torch.logical_and(~disable_flip, self.modified)
                self.modified = torch.logical_or(enable_flip, self.modified)

                if self.debug and (self.modified != prev_modified).any().item():
                    self._dprint(f"BREAKRUNS: modified {prev_modified} -> {self.modified}")

        # check if last was top
        was_top = (input_ids[:, -1] == self.last_logits.argmax(dim=1)).to(torch.long)

        self.breakruns_counter = was_top * self.enabled * (self.breakruns_counter + 1)

        self.max_run_len = torch.max(self.max_run_len, self.breakruns_counter)

        if self.debug:
            sampled_str = repr(self.tokenizer.decode(input_ids[0, -1].item()))
            actual_top_str = repr(self.tokenizer.decode([self.last_logits.argmax(dim=1)[0].item()]))
            print(f"sampled {sampled_str} actual_top {actual_top_str} | enabled {self.enabled[0]} | modified {self.modified[0]} | was_top?: {was_top[0]} | self.breakruns_counter: {self.breakruns_counter}")

        eff_temperature = self.base_temperature + (self.breakruns_counter * self.tau) + (self.modified * self.temp_shift_modifier)
        
        avg_r = 1./(self.last_length - self.first_length)
        self.temp_avg = (1-avg_r)*self.temp_avg + avg_r*eff_temperature
        self._dprint("eff_temperature:\n{et}", fillers={"et": eff_temperature.item() if len(eff_temperature)==1 else eff_temperature})

        self.last_logits = scores

        return scores / eff_temperature[:, None].expand_as(scores)
