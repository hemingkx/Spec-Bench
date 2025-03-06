import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from ...samd_config import SamdConfig
from ..tree import TreeModel
from .utils import (
    pad_path, 
    gen_buffers
)

TOPK = 8

class TokenRecycle(TreeModel):
    
    def __init__(self,
        config: SamdConfig,
        lm: LlamaForCausalLM,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.samd_config = config
        self.dtype = dtype
        self.device = device
        self.tree = config.tree
        self.cache = {}
        
    def reset(self):
        pass  # do nothting

    def logits_to_topk(self, logits: torch.Tensor) -> List[List[int]]:
        topk_nest = logits.topk(k=TOPK).indices.tolist()
        return topk_nest
    
    def update(self, 
        tree_tokens: torch.Tensor, 
        tree_logits: torch.Tensor,
        **kwargs
    ):
        tree_tokens = tree_tokens.tolist()
        topk_nest = self.logits_to_topk(tree_logits)
        for token, topk in zip(tree_tokens, topk_nest):
            self.cache[token] = topk
    
    def gen_draft(self, start_token: int) -> List[int]:
        tree_tokens = [start_token] + [0] * (len(self.tree) - 1)
        for node_id, childs in enumerate(self.tree):
            token = tree_tokens[node_id]
            if token not in self.cache:
                continue
            topk = self.cache[token]
            for child_id, child in enumerate(childs):
                tree_tokens[child] = topk[child_id]
        buffers_kwargs = {}
        return tree_tokens, buffers_kwargs

    def gen_buffers(self) -> Dict[str, torch.Tensor]:
        return gen_buffers(self.samd_config.tree, self.device)
