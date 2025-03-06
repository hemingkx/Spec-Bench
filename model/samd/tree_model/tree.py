import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm


class TreeModel(torch.nn.Module):
    
    def __init__(self,
        samd_config=None,
        lm_config=None,
        lm=None,
        dtype: torch.dtype=None,
        device: str=None,
    ) -> None:
        super().__init__()

    def reset(self):
        raise NotImplementedError
    
    def update(self, tokens: List[int], topk_nest: List[List[int]]):
        raise NotImplementedError
    
    def gen_draft(self, start_token: int) -> Tuple[List[int], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def gen_buffers(self):
        raise NotImplementedError
    