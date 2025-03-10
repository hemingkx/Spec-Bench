import torch

# typing
from typing import List
from .utils import TOPK


class EagleNode:
    def __init__(self, parent=None, value=None, dict_key=None):
        self.parent = parent
        self.value = value
        if parent:
            self.depth = parent.depth + 1
            parent.children.append(self)
        else:
            self.depth = 0
        self.children = []
        self.dict_key = dict_key

    def is_leaf(self):
        return len(self.children) == 0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index() + [self.index]


class EagleTree:

    def __init__(self, tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root = EagleNode()
        self.node_dic = {}
        for tree_node in sorted_tree_list:
            cur_value = tree_node[-1]
            if len(tree_node) == 1:
                cur_node = EagleNode(
                    parent=self.root, value=cur_value, dict_key=tuple(tree_node)
                )
            else:
                cur_parent = self.node_dic[tuple(tree_node[:-1])]
                cur_node = EagleNode(
                    parent=cur_parent, value=cur_value, dict_key=tuple(tree_node)
                )
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c = 0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c += 1
        return num_c

    def get_node_wchild(self):
        ns = []
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index = 0
        for key in self.node_dic:
            cur_node = self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index = cur_index
                cur_index += 1


def gen_buffers_eagle(tree_choices, device="cuda"):
    tree = EagleTree(tree_choices)
    tree_len = tree.num_node_wchild()

    max_depth = tree.max_depth()
    nodes_wc = tree.get_node_wchild()

    depth_counts = [0 for _ in range(max_depth - 1)]
    for x in nodes_wc:
        depth_counts[x.depth - 1] += 1
    depth_counts_sum = [sum(depth_counts[: i + 1]) for i in range(len(depth_counts))]

    tree_attn_mask = torch.eye(tree_len, tree_len)

    for id, x in enumerate(nodes_wc):
        tree_attn_mask[id, x.all_index()] = 1
    
    tree_attn_mask_list0 = [tree_attn_mask[:ml, :ml] for ml in depth_counts_sum]
    tree_attn_mask_list = []
    for id, x in enumerate(tree_attn_mask_list0):
        x = x[-depth_counts[id] :]
        tree_attn_mask_list.append(x)

    tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
    repeat_nums = [[] for _ in depth_counts]
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        bias = 0
        repeat_j = 0
        for j in range(depth_counts[i]):
            cur_node = nodes_wc[start + j]
            cur_parent = cur_node.parent

            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    parent = cur_parent
                    repeat_nums[i].append(j - repeat_j)
                    repeat_j = j
            else:
                parent = cur_parent
            tree_indices_list[i][j] = cur_node.value + TOPK * (bias)
        repeat_nums[i].append(j - repeat_j + 1)
        start += depth_counts[i]

    position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

    tree_buffers = {
        "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
        "tree_indices": tree_indices_list,
        "position_ids": position_ids,
        "repeat_nums": repeat_nums,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: (
            [i.clone().to(device) for i in v]
            if isinstance(v[0], torch.Tensor)
            else (torch.tensor(v, device=device) if isinstance(v, torch.Tensor) else v)
        )
        for k, v in tree_buffers.items()
    }
    return tree_buffers
