import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    Cache,
    StaticCache,
    AttentionMaskConverter
)
from typing import Optional, Tuple, Union, List

from ..samd_config import ForwardType

try:
    from transformers.models.llama.modeling_llama import (
        _prepare_4d_causal_attention_mask_with_cache_position
    )
except:
    _prepare_4d_causal_attention_mask_with_cache_position = LlamaModel._prepare_4d_causal_attention_mask_with_cache_position



@dataclass
class SamdCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )
    
    # assert hasattr(self, "samd_attn_mask") and hasattr(self, "forward_state")
    if self.forward_state.forward_type == ForwardType.tree_decode:
        samd_attn_mask: torch.Tensor = self.mask_state.mask.to(causal_mask.device)
        causal_mask[:, :, :, cache_position] = causal_mask.min() * (samd_attn_mask == 0)
    # if self.forward_state.forward_type == ForwardType.seq_decode:
    #     # do nothing for seq_decode
 
    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, SamdCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = last_hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return SamdCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        last_hidden_states=last_hidden_states,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


llama_patch_dict = {
    LlamaForCausalLM: [("forward", forward)]
}

llama_attn_patch_dict = {
    LlamaModel: [("_update_causal_mask", _update_causal_mask)]
}
