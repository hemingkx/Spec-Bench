import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, Union, List, Literal, Tuple, Dict
from types import MethodType
from transformers import LlamaForCausalLM, LlamaTokenizer
from .samd_config import SamdConfig, ForwardState, ForwardType, MaskState
from .utils import (
    OptionalTensor,
    CandidateType,
    SamdGenerationConfig,
    gen_candidates,
    eval_posterior,
)
from .cache import SamdCache, SamdStaticCache
from .draft import DraftModel
from .model_patch import patch_dict, attn_patch_dict

Outputs = namedtuple('Outputs', ['output_ids', 'decode_tokens', 'decode_steps', 'accepet_length_per_step'])

class SamdModel(nn.Module):
    
    def __init__(self,
        samd_config: SamdConfig,
        lm: LlamaForCausalLM,
        draft: DraftModel,
        eos_token_id: int,
        dtype: torch.dtype,
        device: str,
        stop_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.samd_config = samd_config
        self.gen_config: SamdGenerationConfig = None
        self.eos_token = eos_token_id
        self.stop_token = stop_token_id

        self.lm = lm
        self.draft = draft
        self.dtype = dtype
        self.device = device
        
        # buffers
        self.base_seq_position_ids: torch.Tensor = None
        self.base_tree_attn_mask: torch.Tensor = None
        self.base_tree_position_ids: torch.Tensor = None
        self.base_tree_retrieve_indices: torch.Tensor = None
        self.seq_position_ids: torch.Tensor = None
        self.tree_attn_mask: torch.Tensor = None
        self.tree_position_ids: torch.Tensor = None
        self.tree_retrieve_indices: torch.Tensor = None
        
        # buffers
        self.cache: Union[SamdCache, SamdStaticCache] = None
        self.forward_state = ForwardState(None)
        self.mask_state = MaskState(None)
        
        self.init_buffers()
        self.register_forward_patch()

    def register_forward_patch(self):
        for module_name, module in self.lm.named_modules():
            module_name = "root" if module_name == "" else "root.{}".format(module_name)
            if type(module) in patch_dict:
                for fn_name, fn in patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    print("setattr {} -> {}".format(module_name, fn_name))
            if type(module) in attn_patch_dict:
                for fn_name, fn in attn_patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    setattr(module, "mask_state", self.mask_state)
                    setattr(module, "forward_state", self.forward_state)
                    print("attn setattr {} -> {}".format(module_name, fn_name))

    
    def init_seq_position_ids(self):
        return torch.tensor(
            range(0, self.samd_config.n_predicts), 
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
    
    def init_buffers(self):
        self.base_seq_position_ids = self.init_seq_position_ids()
        if self.draft.tree_model is not None:
            buffers = self.draft.tree_model.gen_buffers()
            self.base_tree_attn_mask = buffers["tree_attn_mask"]
            self.base_tree_position_ids = buffers["tree_position_ids"]
            self.base_tree_retrieve_indices = buffers["tree_retrieve_indices"]
            self.mask_state.set_state(self.base_tree_attn_mask)
    
    def update_buffers(self, buffers_kwargs: Dict[str, Optional[torch.Tensor]]):
        self.seq_position_ids = buffers_kwargs.get("seq_position_ids", self.base_seq_position_ids)
        self.tree_attn_mask = buffers_kwargs.get("tree_attn_mask", self.base_tree_attn_mask)
        self.tree_position_ids = buffers_kwargs.get("tree_position_ids", self.base_tree_position_ids)
        self.tree_retrieve_indices = buffers_kwargs.get("tree_retrieve_indices", self.base_tree_retrieve_indices)
        self.mask_state.set_state(self.tree_attn_mask)
    
    # @profile_decorator("SamdModel.prefill")
    def prefill(self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        self.forward_state.forward_type = ForwardType.prefill
        outputs = self.lm(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=self.cache,
        )
        logits = outputs.logits
        last_hidden_states = outputs.last_hidden_states \
            if self.samd_config.use_last_hidden_states else None
        last_hidden_states = OptionalTensor(last_hidden_states).apply(
            lambda x: x.squeeze(0)
        ).data
        self.draft.update(
            tokens=input_ids.squeeze(0),
            last_hidden_states=last_hidden_states,
            tree_tokens=input_ids.squeeze(0),
            tree_logits=logits.squeeze(0)
        )
        self.cache.set_length()
        if self.gen_config.greedy:
            sample_p = logits[:, -1]
        else:
            sample_p = torch.softmax(logits[:, -1], dim=-1)
        return sample_p  # [1, D]
    
    # @profile_decorator("SamdModel.decode")
    def decode(self, sample_p: torch.Tensor, length: int):
        candidates = gen_candidates(
            sample_p,
            self.base_tree_retrieve_indices,
            self.draft,
            self.samd_config, 
            self.gen_config, 
            self.device
        )
        self.update_buffers(candidates.buffers_kwargs)
        if candidates.type == CandidateType.sequence:
            self.forward_state.forward_type = ForwardType.seq_decode
            position_ids = self.seq_position_ids + length
        else:
            self.forward_state.forward_type = ForwardType.tree_decode
            position_ids = self.tree_position_ids + length
        input_ids = candidates.tokens
        outputs = self.lm(
            input_ids=input_ids, 
            position_ids=position_ids,
            past_key_values=self.cache,
        )
        tree_logits = outputs.logits
        # print("tree_logits.shape:", tree_logits.shape)
        if self.samd_config.use_last_hidden_states:
            tree_last_hidden_states = OptionalTensor(outputs.last_hidden_states)
        else:
            tree_last_hidden_states = OptionalTensor(None)
        if candidates.type == CandidateType.sequence:
            candidate_logits = tree_logits
            candidate_last_hidden_states = tree_last_hidden_states
            candidate_indices = OptionalTensor(None)
        else:
            candidate_logits = tree_logits.squeeze(0)[self.tree_retrieve_indices]
            candidate_last_hidden_states = tree_last_hidden_states.apply(
                lambda x: x.squeeze(0)[self.tree_retrieve_indices]
            )
            candidate_indices = OptionalTensor(self.tree_retrieve_indices)

        best_candidate, accept_length, sample_p \
            = eval_posterior(candidate_logits, candidates.candidate_tokens, self.gen_config)
        new_tokens = self.update_state(
            input_ids.squeeze(0),
            tree_logits.squeeze(0),
            best_candidate, 
            accept_length,
            candidates.candidate_tokens,
            candidate_indices,
            candidate_last_hidden_states,
        )
        # print("new_tokens:\n{}".format(new_tokens))
        return sample_p, new_tokens

    # @profile_decorator("SamdModel.update_state")
    def update_state(self,
        tree_tokens: torch.Tensor,
        tree_logits: torch.Tensor,
        best_candidate: torch.Tensor, 
        accept_length: torch.Tensor,
        candiate_tokens: torch.Tensor,
        candidate_indices: OptionalTensor,
        candidate_last_hidden_states: OptionalTensor,
    ):
        tokens = candiate_tokens[best_candidate][:accept_length]
        
        indices: Optional[torch.Tensor] = candidate_indices.apply(
            lambda x: x[best_candidate][:accept_length]
        ).data
        last_hidden_states: Optional[torch.Tensor] = candidate_last_hidden_states.apply(
            lambda x: x[best_candidate][:accept_length]
        ).data
        
        self.draft.update(
            tokens=tokens, 
            last_hidden_states=last_hidden_states,
            tree_tokens=tree_tokens,
            tree_logits=tree_logits,
        )
        self.cache.select_indices(indices, accept_length.item())
        
        return tokens.tolist()

    def set_cache(self, generation_config: SamdGenerationConfig):
        if self.samd_config.cache_type == "dynamic":
            self.cache = SamdCache(self.lm.config.num_hidden_layers)  # use dynamic cache
        else:
            if self.cache is None:
                self.cache = SamdStaticCache(
                    self.lm.config,
                    batch_size=1,
                    max_cache_len=generation_config.max_cache_len,
                    device=self.device,
                    dtype=self.dtype,
                    hf_device_map=self.lm.hf_device_map,
                )
            else:
                self.cache.reset()
    
    @torch.inference_mode()
    def generate(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        generation_config: SamdGenerationConfig = None, 
    ) -> Outputs:
        if generation_config is None:
            generation_config = SamdGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]

        self.set_cache(generation_config)

        self.draft.reset()
        
        input_ids_list = input_ids.squeeze(0).tolist()
        sample_p = self.prefill(input_ids, attention_mask)
        
        input_length = input_ids.shape[-1]
        decode_tokens = 0
        decode_steps = 0
        accepet_length_per_step = []
        for step in range(generation_config.max_new_tokens):
            if input_length + decode_tokens + self.samd_config.max_predicts >= generation_config.max_cache_len:
                break
            sample_p, new_ids = self.decode(sample_p, input_length + decode_tokens)
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index + 1]
            elif self.stop_token is not None and self.stop_token in new_ids:
                eos_index = new_ids.index(self.stop_token)
                new_ids = new_ids[:eos_index + 1]
            input_ids_list.extend(new_ids)
            decode_steps += 1
            decode_tokens += len(new_ids)
            accepet_length_per_step.append(len(new_ids))
            # profile_accept_length("lookup", len(new_ids))
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
        input_ids_list = [input_ids_list[:input_length + generation_config.max_new_tokens]]
        return Outputs(input_ids_list, decode_tokens, decode_steps, accepet_length_per_step)

    @torch.inference_mode()
    def stream_generate(self,
        input_ids: torch.Tensor,
        tokenizer: LlamaTokenizer,
        generation_config: SamdGenerationConfig = None, 
    ):
        attention_mask = None
        if generation_config is None:
            generation_config = SamdGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]

        self.set_cache(generation_config) 

        self.draft.reset()
        
        input_ids_list = input_ids.squeeze(0).tolist()
        sample_p = self.prefill(input_ids, attention_mask)
        
        input_length = input_ids.shape[-1]
        decode_tokens = 0
        for step in range(generation_config.max_steps):
            if input_length + decode_tokens + self.samd_config.max_predicts >= generation_config.max_cache_len:
                break
            sample_p, new_ids = self.decode(sample_p, input_length + decode_tokens)
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index + 1]
            elif self.stop_token is not None and self.stop_token in new_ids:
                eos_index = new_ids.index(self.stop_token)
                new_ids = new_ids[:eos_index + 1]
            input_ids_list.extend(new_ids)
            yield {
                "text": tokenizer.decode(
                    input_ids_list[input_length:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }
            decode_tokens += len(new_ids)
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
