import time
import pickle
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from typing import List

from .static_sam import StaticSAM
from ..samd_config import SamdConfig

def build_sam(
    batch_tokens: List[List[int]],
    eos_token: int,
):
    sam = StaticSAM.build(
        batch_tokens, 
        eos_token
    )
    return sam

def dump_sam(path: str, sam: StaticSAM):
    with open(path, "wb") as f:
        pickle.dump(sam, f)

def load_sam(path: str):
    print("load sam...")
    start = time.perf_counter()
    with open(path, "rb") as f:
        _sam = pickle.load(f)
    sam = StaticSAM()
    for key, value in vars(_sam).items():
        if hasattr(sam, key):
            setattr(sam, key, value)
            print("load [{}]".format(key))
    end = time.perf_counter()
    assert type(sam) is StaticSAM
    print("loading ended in {} seconds.".format(end - start))
    return sam
