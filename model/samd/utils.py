import torch
import torch.nn.functional as F
import random
from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import namedtuple
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .samd_config import SamdConfig
from .draft import DraftModel, Candidates, CandidateType    

class OptionalTensor:
    
    def __init__(self, data: Optional[torch.Tensor] = None):
        self.data = data

    def apply(self, fn: Callable) -> 'OptionalTensor':
        if self.data is None:
            return OptionalTensor(None)
        else:
            return OptionalTensor(fn(self.data))

@dataclass
class SamdGenerationConfig:
    max_steps: int = field(default=512)
    max_new_tokens: int = field(default=512)
    max_cache_len: int = field(default=2048)
    greedy: bool = field(default=True)
    temperature: float = field(default=0.0)
    top_p: float = field(default=0.0)
    top_k: int = field(default=0)
    logits_processor: LogitsProcessorList = field(default=None)
    
    def __post_init__(self):
        if not self.greedy:
            assert self.temperature >= 1e-5
            self.logits_processor = self.prepare_logits_processor(
                self.temperature,
                self.top_p,
                self.top_k,
            )

    @staticmethod
    def prepare_logits_processor(
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
    ) -> LogitsProcessorList:
        processor_list = LogitsProcessorList()
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


def gen_candidates(
    sample_p: torch.Tensor,
    tree_retrieve_indices: torch.Tensor,
    draft: DraftModel,
    samd_config: SamdConfig,
    gen_config: SamdGenerationConfig,
    device: torch.device,
):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - ...

    Returns:
    - tuple (torch.Tensor, List[int]): ...
    """
    # Greedy decoding: Select the most probable candidate from the original logits.
    if gen_config.greedy:
        start_token = torch.argmax(sample_p, dim=-1).item()
    else:
        start_token = torch.multinomial(sample_p, 1).item()
    candidate_type, tokens, buffers_kwargs = draft.lookup(start_token)
    tree_retrieve_indices = buffers_kwargs.get("tree_retrieve_indices", tree_retrieve_indices)
    if candidate_type == CandidateType.sequence:
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)
        candidate_tokens = tokens
    else:
        tokens_ext = torch.tensor(tokens + [0], dtype=torch.long, device=device)
        candidate_tokens = tokens_ext[tree_retrieve_indices]
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    return Candidates(
        candidate_type,
        tokens,
        candidate_tokens,
        buffers_kwargs,
    )


def eval_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    config: SamdGenerationConfig,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    if config.greedy:
        # Greedy decoding based on temperature value
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length + 1, logits[best_candidate, accept_length].view(1, -1)
    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = config.logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        sample_p = sample_p.view(1, -1)
        accept_length = torch.tensor(accept_length, dtype=torch.long, device=candidates.device)
        best_candidate = torch.tensor(best_candidate, dtype=torch.long, device=candidates.device)
        return best_candidate, accept_length, sample_p
