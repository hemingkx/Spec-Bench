import torch
import copy
from transformers import LlamaConfig, LlamaForCausalLM
from typing import List, Tuple, Dict

from ...samd_config import SamdConfig
from ..tree import TreeModel
from .eagle_config import EagleConfig
from .eagle_model import EagleModel

from .utils import gen_buffers, TOPK


class Eagle(TreeModel):
    
    def __init__(self,
        config: SamdConfig,
        lm: LlamaForCausalLM,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.tree = config.tree
        self.dtype = dtype
        self.device = device
        self.head: torch.nn.Linear = lm.lm_head
        self.model: EagleModel = EagleModel(
            config=EagleConfig(**config.tree_config),
            bias=config.tree_config.get("bias", True)
        ).to(device=device, dtype=dtype)
        self.model.gen_buffers(config.tree, device)
        self.model.load_weight(config.tree_model_path)
        
        self.accpet_tokens: torch.Tensor = None
        self.accept_hidden_states: torch.Tensor = None
        self.tree_indices: torch.Tensor = None
    
    def reset(self):
        self.model.stable_kv = None
    
    def update(self, 
        tokens: torch.Tensor,
        last_hidden_states: torch.Tensor,
        **kwargs,
    ):
        tokens = tokens.to(self.device)
        if self.accpet_tokens is None:
            self.accpet_tokens = tokens
        else:
            self.accpet_tokens = torch.cat([self.accpet_tokens, tokens], dim=-1)
        if self.accept_hidden_states is None:
            self.accept_hidden_states = last_hidden_states
        else:
            self.accept_hidden_states = torch.cat([self.accept_hidden_states, last_hidden_states], dim=-2)
    
    def gen_draft(self, start_token: int) -> List[int]:
        start_token = torch.tensor([start_token], dtype=torch.long, device=self.device)
        accpet_tokens = torch.cat((self.accpet_tokens, start_token), dim=-1)
        accept_hidden_states = self.accept_hidden_states
        self.accpet_tokens = self.accept_hidden_states = None
        pred_ids: torch.Tensor = self.model.topk_genrate(
            accept_hidden_states,
            accpet_tokens,
            self.head,
            top_k=TOPK
        )[0]
        pred_ids = torch.cat([start_token, pred_ids.view(-1)])
        pred_ids = pred_ids[self.tree_indices].tolist()
        buffers_kwargs = {}
        return pred_ids, buffers_kwargs
    
    def gen_buffers(self):
        buffers = gen_buffers(self.tree, self.device)
        self.tree_indices = buffers["tree_indices"]
        return buffers
