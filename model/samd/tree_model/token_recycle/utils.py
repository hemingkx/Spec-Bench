import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

from ...samd_config import SamdConfig

def pad_path(path, length, pad_value=-1):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-1): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -1, -1]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def gen_buffers(
    tree: List[List[int]],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Generate buffers for the SD based on the provided bfs tree.
    
    Parameters:
    - tree (List[List[int]]): A nested list represent the SD tree structure.
    - device (torch.device): The device to save the tensors.
    
    Returns:
    - dict: A dictionary containing buffers related to the SD structure.
    """
    num_nodes = len(tree)
    
    anc_dict = {0: -1}
    for node_id, childs in enumerate(tree):
        for child in childs:
            anc_dict[child] = node_id

    level_dict = {0: 0}
    for node_id in range(1, num_nodes):
        level_dict[node_id] = level_dict[anc_dict[node_id]] + 1
    
    # Create the attention mask for Medusa
    tree_attn_mask = torch.eye(num_nodes, num_nodes)
    for node_id in range(num_nodes):
        ancs = [node_id]
        x = node_id
        while x != -1:
            ancs.append(x)
            x = anc_dict[x]
        ancs = torch.tensor(ancs, dtype=torch.long)
        tree_attn_mask[node_id, ancs] = True
    tree_attn_mask = tree_attn_mask.view(1, 1, num_nodes, num_nodes)
    
    tree_position_ids = torch.zeros((1, num_nodes), dtype=torch.long)
    for i in range(num_nodes):
        tree_position_ids[:, i] = level_dict[i]
    
    max_level = max(level_dict.values()) + 1
    retrieve_indices_nest = []
    for node_id, childs in enumerate(tree):
        if len(childs) != 0:
            continue
        retrieve_indices = [node_id]
        while retrieve_indices[-1] != 0:
            retrieve_indices.append(anc_dict[retrieve_indices[-1]])
        retrieve_indices_nest.append(list(reversed(retrieve_indices)))
    
    retrieve_indices_nest = reversed(retrieve_indices_nest)
    retrieve_indices_nest = [pad_path(x, max_level) for x in retrieve_indices_nest]
    tree_retrieve_indices = torch.tensor(retrieve_indices_nest, dtype=torch.long)

    tree_buffers = {
        "tree_attn_mask": tree_attn_mask,
        "tree_position_ids": tree_position_ids,
        "tree_retrieve_indices": tree_retrieve_indices,
    }
    
    tree_buffers = {k: (v.to(device) if v is not None else v) for k, v in tree_buffers.items()}
    return tree_buffers
