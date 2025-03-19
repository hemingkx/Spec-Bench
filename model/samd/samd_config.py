import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal, Dict, Any
from enum import Enum


@dataclass
class SamdConfig:
    n_predicts: int = field(default=40)
    max_predicts: int = field(default=70)
    len_threshold: int = field(default=5)
    len_bias: int = field(default=5)

    cache_type: Literal["dynamic", "static"] = field(
        default="static"
    )
    use_last_hidden_states: bool = field(default=False)

    tree_method: Literal["token_recycle", "eagle", "eagle2"] = field(
        default=None
    )
    tree_model_path: Optional[str] = field(default=None)
    tree_path: Optional[str] = field(default=None)
    tree: Optional[List[List[int]]] = field(default=None)
    tree_config: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if self.tree is None:
            if self.tree_method == "token_recycle":
                self.tree = load_token_recycle(self.tree_path)
            elif self.tree_method == "eagle":
                tree, tree_config = load_eagle(self.tree_model_path, self.tree_path)
                self.tree = tree
                self.tree_config = tree_config
                self.use_last_hidden_states = True
            elif self.tree_method == "eagle2":
                tree_config = load_eagle2(self.tree_model_path)
                self.tree_config = tree_config
                self.use_last_hidden_states = True


class ForwardType(str, Enum):
    prefill = "prefill"
    seq_decode = "seq_decode"
    tree_decode = "tree_decode"


class ForwardState:

    def __init__(self, forward_type: Optional[ForwardType]) -> None:
        self.forward_type = forward_type


class MaskState:

    def __init__(self, mask: Optional[torch.Tensor]) -> None:
        self.mask = mask

    def set_state(self, mask: Optional[torch.Tensor]) -> None:
        self.mask = mask


def load_token_recycle(tree_path: Optional[str] = None):
    if tree_path is None:
        tree_path = "token_recycle.json"
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", tree_path), "r") as f:
        tree_adj: dict = json.load(f)["tree_adj"]
    num_node = len(tree_adj)
    tree: List[List[int]] = []
    for i in range(num_node):
        tree.append(tree_adj[str(i)])
    print("tree_path:", tree_path)
    print("len_tree:", len(tree))
    return tree


def load_eagle(tree_model_path: str, tree_path: Optional[str] = None):
    if tree_path is None:
        tree_path = "eagle.json"
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", tree_path), "r") as f:
        tree = json.load(f)["tree_choices"]
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree, tree_config


def load_eagle2(tree_model_path: str):
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree_config
