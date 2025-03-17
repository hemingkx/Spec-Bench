import numpy as np
import torch
from transformers import PretrainedConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, Cache
from typing import Optional, Dict, Any, Tuple, List

class SamdCache(DynamicCache):
    
    def __init__(self, num_hidden_layers: int | None = None) -> None:
        super().__init__(num_hidden_layers)
        self.cache_length = 0
    
    def set_length(self):
        self.cache_length = self.get_seq_length()
    
    # @profile_decorator("SamdCache.select_indices")
    def select_indices(self,
        indices: torch.Tensor | None = None,
        accept_length: int = 1,
    ):
        start = self.cache_length
        if indices is not None:
            select_indices = start + indices
        else:
            select_indices = None
        for data in self.key_cache + self.value_cache:
            if select_indices is not None:
                select_indices = select_indices.to(data.device)
                tgt = data.index_select(-2, select_indices)
                dst = data.narrow(-2, start, accept_length)
                dst.copy_(tgt)                
        self.cache_length += accept_length
        self.crop(self.cache_length)


class SamdStaticCache(Cache):
    
    def __init__(self, 
        config, 
        batch_size = None, 
        max_cache_len = None,
        device = None, 
        dtype = torch.float32, 
        max_batch_size = None, 
        hf_device_map = None,
    ):
        super().__init__()
        if len(hf_device_map) <= 1:
            device = device
            layer_device_map = None
        else:
            device = None
            layer_device_map = {}
            for i in range(config.num_hidden_layers):
                layer_device_map[i] = hf_device_map[f"model.layers.{i}"]
        self.batch_size = batch_size or max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

        self.last_length = 0
        self.cache_length = 0

    def reset(self):
        self.cache_length = 0
        self.last_length = 0
    
    def set_length(self):
        self.cache_length = self.last_length
    
    def get_seq_length(self, layer_idx = 0):
        return self.cache_length

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.cache_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return self.max_cache_len
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs = None):
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        k_dst = k_out.narrow(2, self.cache_length, key_states.shape[2])
        v_dst = v_out.narrow(2, self.cache_length, value_states.shape[2])
        k_dst.copy_(key_states)
        v_dst.copy_(value_states)
        if layer_idx == 0:
            self.last_length = self.cache_length + key_states.shape[2]
        return (
            k_out.narrow(2, 0, self.last_length),
            v_out.narrow(2, 0, self.last_length),
        )
    
    # @profile_decorator("SamdCache.select_indices")
    def select_indices(self,
        indices: torch.Tensor | None = None,
        accept_length: int = 1,
    ):
        start = self.cache_length
        if indices is not None:
            select_indices = start + indices
        else:
            select_indices = None
        for data in self.key_cache + self.value_cache:
            if select_indices is not None:
                select_indices = select_indices.to(data.device)
                tgt = data.index_select(-2, select_indices)
                dst = data.narrow(-2, start, accept_length)
                dst.copy_(tgt, non_blocking=True)
        self.cache_length += accept_length
