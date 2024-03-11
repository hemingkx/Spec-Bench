import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hydra.modeling_llama_kv import _make_causal_mask as llama_make_causal_mask
from model.hydra.modeling_llama_kv import _expand_mask as expand_mask
from model.hydra.modeling_llama_kv import (LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, LlamaRMSNorm, LlamaMLP, repeat_kv, rotate_half)

class HydraCrossAttention(nn.Module):
    """
    LlamaAttention is a multi-headed attention module based on the 'Attention Is All You Need' paper.

    Args:
        config (LlamaConfig): Configuration for the attention module.

    Attributes:
        config (LlamaConfig): Configuration for the attention module.
        hidden_size (int): The size of the hidden layer.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        num_key_value_heads (int): The number of key-value attention heads.
        num_key_value_groups (int): The number of key-value groups.
        pretraining_tp (int): The pretraining time periods.
        max_position_embeddings (int): The maximum position embeddings.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
    

    def apply_rotary_pos_emb(self, x, cos, sin, position_ids):
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            x (torch.Tensor): Input tensor.
            cos (torch.Tensor): Cosine values.
            sin (torch.Tensor): Sine values.
            position_ids (torch.Tensor): Position IDs.

        Returns:
            torch.Tensor: Query and key tensors with rotary position embeddings applied.
        """
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed


    def forward(
        self,
        input_embeds,
        base_hidden_states,
        attention_mask= None,
        inputs_position_ids= None,
        base_hidden_states_position_ids= None,
        past_key_value= None,
        update_kv_cache= False,
        output_attentions= False,
        use_cache= False,
    ):
        bsz, q_len, _ = input_embeds.size()
        bsz, k_len, _ = base_hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(input_embeds, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(base_hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(base_hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(input_embeds)
            key_states = self.k_proj(base_hidden_states)
            value_states = self.v_proj(base_hidden_states)
        
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, k_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, k_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = self.apply_rotary_pos_emb(query_states, cos, sin, inputs_position_ids)
        key_states = self.apply_rotary_pos_emb(key_states, cos, sin, base_hidden_states_position_ids)

        # [MODIFIED] Using KVCache mechanism for preallocated GPU memory optimization
        # past_key_value is utilized to leverage previously computed key and value states.
        # If past_key_value is available, reuse the states for k, v, and self_attention.
        if past_key_value is not None:
            # in place concat modifies the underlying data (see kv-cache impl.)
            if update_kv_cache:
                key_states = past_key_value[0].cat(key_states, dim=2)
                value_states = past_key_value[1].cat(value_states, dim=2)
            else:
                key_states = torch.cat((past_key_value[0].get_data(), key_states), dim=2)
                value_states = torch.cat((past_key_value[1].get_data(), value_states), dim=2)
        # Reset past_key_value to avoid return past_key_value.
        past_key_value = None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class HydraCrossAttentionDecoderLayer(nn.Module):
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

        self.cross_attn = HydraCrossAttention(config=base_config)
        self.mlp = LlamaMLP(base_config)
        self.input_layernorm = LlamaRMSNorm(base_config.hidden_size, eps=base_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            base_config.hidden_size, eps=base_config.rms_norm_eps
        )

    # When training, i.e. processing seq in parallel we use
    # a standard causal attention mask even though cross attending
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
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
        
        return combined_attention_mask

    def _build_hydra_head_attention_mask(
        self,
        attention_mask,
        proposal_mask,
        src_shape,
        tgt_shape,
        base_hidden_states,
        past_key_values_length,
        seq_length_with_past,
        forward_mode,
    ):
        batch_size, src_seq_length = src_shape
        _, tgt_seq_length = tgt_shape

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=base_hidden_states.device,
            )

        if forward_mode == "training":
            # In training we use a standard causal mask because still causal dependence even though cross-attn
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, tgt_seq_length),
                base_hidden_states,
                past_key_values_length,
            )
        else:
            # In decoding 
            assert proposal_mask is not None, "Proposal mask must be provided in decoding mode"
            attention_mask = expand_mask(
                attention_mask,
                dtype=torch.float32,
                tgt_len=src_seq_length
            )
            proposal_q_len = proposal_mask.size(-2)
            proposal_kv_len = proposal_mask.size(-1)
            attention_mask[:, :, -proposal_q_len:, -proposal_kv_len:][
                proposal_mask == 0
            ] = torch.finfo(attention_mask.dtype).min

        return attention_mask

    def forward(self,
                input_ids,
                base_hidden_states,
                forward_mode="training",
                inputs_position_ids=None,
                base_hidden_states_position_ids=None,
                attention_mask=None,
                proposal_mask=None,
                past_key_value= None,
                update_kv_cache=False,
                output_attentions= False,
                use_cache= False,
                noise=None
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
            input_ids = torch.roll(input_ids, shifts=-1, dims=1)

        inputs_embeds = self.input_embed_fn(input_ids)
        if noise is not None:
            inputs_embeds = inputs_embeds + noise
        inputs_batch_size, inputs_seq_length, _ = inputs_embeds.shape
        base_hidden_states_batch_size, base_hidden_states_seq_length, _ = base_hidden_states.shape

        assert inputs_batch_size == base_hidden_states_batch_size, "Input and hidden states batch size mismatch"

        if forward_mode == "training":
            assert base_hidden_states_seq_length == inputs_seq_length, "Input and hidden states seq length mismatch"
        elif forward_mode != "decoding":
            raise ValueError(f"Unknown forward mode {forward_mode}")

        # Build kv-cache info
        seq_length_with_past = base_hidden_states_seq_length
        past_key_values_length = 0

        if past_key_value is not None:
            assert forward_mode == "decoding", "Past key values only supported in decoding mode"
            # TODO (Zack): make sure this is right
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        # Building position information
        # Position ids are just for the 
        if base_hidden_states_position_ids is None:
            assert forward_mode == "training", "Position ids must be provided in decoding mode"
            base_hidden_states_position_ids = torch.arange(
                past_key_values_length,
                seq_length_with_past,
                dtype=torch.long,
                device=base_hidden_states.device,
            )
            base_hidden_states_position_ids = base_hidden_states_position_ids.unsqueeze(0).view(-1, base_hidden_states_seq_length)
        else:
            base_hidden_states_position_ids = base_hidden_states_position_ids.view(-1, base_hidden_states_seq_length).long()

        # TODO (Zack): Figure out if need to shift input ids
        if inputs_position_ids is None:
            inputs_position_ids = base_hidden_states_position_ids[:]
        else:
            inputs_position_ids = inputs_position_ids.view(-1, inputs_seq_length).long()
        
        # Building attention mask
        attention_mask = self._build_hydra_head_attention_mask(
            attention_mask=attention_mask,
            proposal_mask=proposal_mask,
            src_shape=(inputs_batch_size, inputs_seq_length),
            tgt_shape=(base_hidden_states_batch_size, base_hidden_states_seq_length),
            base_hidden_states=base_hidden_states,
            past_key_values_length=past_key_values_length,
            seq_length_with_past=seq_length_with_past,
            forward_mode=forward_mode,
        )


        # Calling cross-attention module
        residual = inputs_embeds

        inputs_embeds = self.input_layernorm(inputs_embeds)
        base_hidden_states = self.input_layernorm(base_hidden_states)


        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            input_embeds = inputs_embeds,
            base_hidden_states = base_hidden_states,
            attention_mask=attention_mask,
            inputs_position_ids=inputs_position_ids,
            base_hidden_states_position_ids=base_hidden_states_position_ids,
            past_key_value=past_key_value,
            update_kv_cache=update_kv_cache,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # Todo (Zack): See if should include residual for cross attention
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        logits = self.lm_head(hidden_states)

        return logits, hidden_states

        # TODO (Zack): fix when doing kv cache
        # outputs = (hidden_states,)

        # if output_attentions:
        #     outputs += (self_attn_weights,)

        # if use_cache:
        #     outputs += (present_key_value,)

        # return outputs
    
    # # # Todo (Zack): the grounded proposal framework can probably be unified
    def proposal(self,base_logits, base_hidden_states, hydra_buffers, past_key_values, input_ids=None):
        # TODO (Zack): Figure out batch size considerations
        # TODO (Zack): Other impl. is to generate all candidates up to max k at each step and then reduce using indexing. Could be more efficient than bespoke concat currently pursuing
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
        base_hidden_states_position_ids = torch.arange(
            past_seq_len, seq_len, device=base_logits.device).unsqueeze(0)

        candidates_logit = torch.argmax(base_logits[:, -1]).unsqueeze(0)
        candidates = torch.tensor([candidates_logit], device=candidates_logit.device)[None, ...]
        candidate_input_ids = candidates_logit.unsqueeze(0)

        for head_idx, (head_num_children, head_children_to_expand) in enumerate(zip(children_per_head, children_to_expand_per_head)):
            proposal_mask = proposal_masks[head_idx]
            num_choices = proposal_mask.size(-2)

            inputs_position_ids = torch.tensor(
                [[seq_len + head_idx - 1]], device=base_logits.device).repeat(1, num_choices)

            hydra_preds, hydra_hidden_states = self.forward(
                candidate_input_ids,
                base_hidden_states,
                "decoding",
                proposal_mask=proposal_masks[head_idx],
                inputs_position_ids=inputs_position_ids,
                base_hidden_states_position_ids=base_hidden_states_position_ids,
                past_key_value=past_key_value,
                update_kv_cache=head_idx == 0, # First time processing previous hidden reps so update hidden state
            )

            to_expand_hidden_states = []
            to_expand_input_ids = []
            for path_idx, (num_children, children_to_expand) in enumerate(zip(head_num_children, head_children_to_expand)):
                
                hydra_candidates = torch.topk(hydra_preds[:, path_idx], num_children, dim=-1).indices
                candidates = torch.cat([candidates, hydra_candidates], dim=-1)
                
                if children_to_expand > 0:
                    to_expand_hidden_states.append(hydra_hidden_states[:, path_idx:path_idx+1])
                    to_expand_input_ids.append(hydra_candidates[:, :children_to_expand])
            
            if len(to_expand_hidden_states):
                # TODO (Zack): Determine assertion error about next_head_embeddings being empty before finishing tree
                new_hidden_state_pos_ids = base_hidden_states_position_ids.max(dim=1, keepdim=True)[0].repeat(1, len(to_expand_hidden_states))
                if head_idx == 0:
                    base_hidden_states_position_ids = new_hidden_state_pos_ids
                else:
                    base_hidden_states_position_ids = torch.cat([base_hidden_states_position_ids, new_hidden_state_pos_ids], dim=1)

                if head_idx != 0:
                    to_expand_hidden_states = [base_hidden_states] + to_expand_hidden_states
                base_hidden_states = torch.cat(to_expand_hidden_states, dim=1)

                candidate_input_ids = torch.cat(to_expand_input_ids, dim=1)

        # TODO (Zack): Only selecting first batch element for now, change when doing bs > 1
        cart_candidates = candidates[0, retrieve_indices]

        return cart_candidates, candidates