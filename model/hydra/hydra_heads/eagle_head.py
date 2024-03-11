import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hydra.modeling_llama_kv import _make_causal_mask as llama_make_causal_mask
from model.hydra.modeling_llama_kv import _expand_mask as expand_mask
from model.hydra.modeling_llama_kv import LlamaDecoderLayer


class EagleAttentionDecoderLayer(nn.Module):
    """
    LlamaDecoderLayer represents a single layer of the Llama decoder.

    Args:
        config (LlamaConfig): Configuration for the decoder layer.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        self_attn (LlamaAttention): Multi-headed self-attention module.
        mlp (LlamaMLP): Multi-layer perceptron module.
        input_layernorm (LlamaRMSNorm): Layer normalization for input.
        post_attention_layernorm (LlamaRMSNorm): Layer normalization after self-attention.
    """

    def __init__(
        self,
        hydra_num_layers,
        hydra_num_heads,
        grounded_heads,
        input_embed_fn,
        base_config,
        lm_head,
    ):
        super().__init__()

        self.hidden_size = base_config.hidden_size

        self.hydra_num_layers = hydra_num_layers
        assert self.hydra_num_layers == 1, "Hydra Attention must have exactly one layer."

        self.hydra_num_heads = hydra_num_heads
        assert self.hydra_num_heads == 1, "Hydra Attention must have exactly one head."

        self.grounded_heads = grounded_heads
        assert self.grounded_heads, "Hydra Attention must have grounded heads."

        self.input_embed_fn = input_embed_fn
        self.lm_head = lm_head
        if not self.lm_head.weight.requires_grad:
            "Warning: lm head isn't frozen"

        self.attn = LlamaDecoderLayer(config=base_config)
        self.fc = nn.Linear(2*base_config.hidden_size, base_config.hidden_size)

    # When training, i.e. processing seq in parallel we use
    # a standard causal attention mask even though cross attending
    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        proposal_mask = None
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, src_seq_len, tgt_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = llama_make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        else:
            mask = torch.zeros(
                1, 1, dtype=torch.float32, device=inputs_embeds.device
            )
            if past_key_values_length > 0:
                mask = torch.cat(
                    [
                        torch.zeros(
                            1, past_key_values_length, dtype=torch.float32, device=inputs_embeds.device
                        ),
                        mask,
                    ],
                    dim=-1,
                )
            combined_attention_mask =  mask[None, None, :, :].expand(
                input_shape[0], 1, 1, 1 + past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if proposal_mask is not None:
            proposal_q_len = proposal_mask.size(-2)
            proposal_kv_len = proposal_mask.size(-1)
            combined_attention_mask[:, :, -proposal_q_len:, -proposal_kv_len:][
                proposal_mask == 0
            ] = torch.finfo(combined_attention_mask.dtype).min
        
        return combined_attention_mask

    def forward(self,
                input_ids=None,
                base_hidden_states=None,
                joint_hidden_states=None,
                forward_mode="training",
                position_ids=None,
                attention_mask=None,
                proposal_mask=None,
                past_key_value= None,
                update_kv_cache=False,
                output_attentions= False,
                use_cache= False,
    ):
        """
        Forward pass for the LlamaDecoderLayer.

        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape `(batch, seq_len, embed_dim)`.
            attention_mask (torch.FloatTensor, optional): Attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (torch.LongTensor, optional): Positional IDs tensor.
            past_key_value (Tuple[torch.FloatTensor], optional): Cached past key and value projection states.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
            use_cache (bool, optional): If set to `True`, `past_key_values` key-value states are returned and can be
                used to speed up decoding.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: Tuple containing:
                - hidden_states (torch.FloatTensor): Output tensor.
                - self_attn_weights (Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]): Self-attention weights if
                  `output_attentions` is `True`.
                - present_key_value (Optional[Tuple[torch.FloatTensor]]): Cached key and value projection states if
                  `use_cache` is `True`.
        """
        if forward_mode == "training":
            assert input_ids is not None, "Input ids must be provided in training mode"
            assert base_hidden_states is not None, "Base hidden states must be provided in training mode"

            input_ids = torch.roll(input_ids, shifts=-1, dims=1)

            with torch.no_grad():
                inputs_embeds = self.input_embed_fn(input_ids)

            inputs_embeds=inputs_embeds.to(base_hidden_states.dtype)
            joint_hidden_states = self.fc(torch.cat((inputs_embeds, base_hidden_states), dim=-1))
        elif forward_mode == "decoding":
            assert joint_hidden_states is not None, "Joint hidden states must be provided in decoding mode"
        else:
            raise ValueError(f"Unknown forward mode {forward_mode}")
        
        batch_size, seq_length, _ = joint_hidden_states.shape

        # Build kv-cache info
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_value is not None:
            assert forward_mode == "decoding", "Past key values only supported in decoding mode"
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        # Building position information
        # Position ids are just for the 
        if position_ids is None:
            assert forward_mode == "training", "Position ids must be provided in decoding mode"
            position_ids = torch.arange(
                past_key_values_length,
                seq_length_with_past,
                dtype=torch.long,
                device=base_hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        # Building attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=joint_hidden_states,
            past_key_values_length=past_key_values_length,
            proposal_mask=proposal_mask
        )

        # Calling decoder layer
        layer_outputs = self.attn(
            joint_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # Todo (Zack): See if should include residual for cross attention
        hidden_states = layer_outputs[0]

        logits = self.lm_head(hidden_states)

        return logits, hidden_states
    

    def proposal(self, base_logits, base_hidden_states, hydra_buffers, past_key_values, input_ids=None):
        """
        Ok so essentially what I want to do is build the tree out myself
        The paths should be ordered by greedy probability
        So the way that it would work is for each path find the number of children that it has, then sample that many k from it
        Then we need to concat that many copies together and add it to the paths that still need to be explored

        Should probably pre-compute the number of children that need to be expanded for the path and store it in hydra buffers
        """
        children_per_head = hydra_buffers["children_per_head"]
        children_to_expand_per_head = hydra_buffers["children_to_expand_per_head"]
        retrieve_indices = hydra_buffers["retrieve_indices"]
        proposal_masks = hydra_buffers["proposal_cross_attn_masks"]

        past_key_value = past_key_values[-1]
        past_seq_len = past_key_value[0].current_length
        seq_len = past_seq_len + base_hidden_states.shape[1]

        logit_ids = torch.argmax(base_logits, dim=-1)
        if input_ids is not None:
            input_ids = torch.cat([input_ids[:, 1:], logit_ids[:, -1:]], dim=1)
        else:
            input_ids = logit_ids

        candidates_logit = input_ids[:, -1]
        candidates = torch.tensor([candidates_logit], device=candidates_logit.device)[None, ...]

        joint_hidden_states = self.fc(torch.cat((self.input_embed_fn(input_ids), base_hidden_states), dim=-1))
        
        for head_idx, (head_num_children, head_children_to_expand) in enumerate(zip(children_per_head, children_to_expand_per_head)):
            proposal_mask = proposal_masks[head_idx]
            num_choices = proposal_mask.size(-2)

            if head_idx == 0:
                position_ids = torch.arange(past_seq_len, seq_len, device=base_logits.device).unsqueeze(0).repeat(1, num_choices)
            else: 
                position_ids = torch.tensor(
                    [[seq_len + head_idx - 1]], device=base_logits.device).repeat(1, num_choices)
            hydra_preds, hydra_hidden_states = self.forward(
                joint_hidden_states=joint_hidden_states,
                forward_mode="decoding",
                proposal_mask=proposal_masks[head_idx],
                past_key_value=past_key_value,
                position_ids=position_ids,
            )

            next_joint_hidden_states = []
            for path_idx, (num_children, children_to_expand) in enumerate(zip(head_num_children, head_children_to_expand)):
                if head_idx == 0:
                    hydra_candidates = torch.topk(hydra_preds[:, -1], num_children, dim=-1).indices
                else:
                    hydra_candidates = torch.topk(hydra_preds[:, path_idx], num_children, dim=-1).indices
                candidates = torch.cat([candidates, hydra_candidates], dim=-1)
                
                if children_to_expand > 0:
                    if head_idx == 0:
                        to_expand_hidden_state = hydra_hidden_states[:, -1:].repeat(1, children_to_expand, 1)
                    else:
                        to_expand_hidden_state = hydra_hidden_states[:, path_idx:path_idx+1].repeat(1, children_to_expand, 1)
                    to_expand_input_ids = hydra_candidates[:, :children_to_expand]
                    next_joint_hidden_states.append(self.fc(torch.cat((self.input_embed_fn(to_expand_input_ids), to_expand_hidden_state), dim=-1)))
                    
            if len(next_joint_hidden_states):
                joint_hidden_states = torch.cat(next_joint_hidden_states, dim=1)
        
        cart_candidates = candidates[0, retrieve_indices]

        return cart_candidates, candidates