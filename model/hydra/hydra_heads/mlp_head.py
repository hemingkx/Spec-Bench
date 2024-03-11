import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, num_condition=0):
        super().__init__()
        self.linear = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        # Handling residual connection when reducing dim
        if num_condition > 0:
            self.res_connection = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        else:
            self.res_connection = nn.Identity()
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return self.res_connection(x) + self.act(self.linear(x))

class HydraMLP(nn.Module):
    """
    A MLP module as the Hydra head.

    Args:
        hidden_size (int): The size of the hidden layers in the MLP.
        num_layers (int): The number of hidden layers in the MLP.
    """

    def __init__(
        self,
        hydra_num_layers, 
        hydra_num_heads, 
        grounded_heads, 
        input_embed_fn,
        base_config,
        lm_head_init_weight=None,
    ):
        super().__init__()

        self.hidden_size = base_config.hidden_size
        self.vocab_size = base_config.vocab_size
        
        self.hydra_num_layers = hydra_num_layers
        self.hydra_num_heads = hydra_num_heads
        self.grounded_heads = grounded_heads
        self.input_embed_fn = input_embed_fn

        assert self.hydra_num_layers > 0, "Hydra MLP must have at least one layer."

        if grounded_heads:
            self.hydra_mlp = nn.ModuleList([
                nn.Sequential(
                    ResBlock(self.hidden_size, hydra_head_idx + 1),
                    *([ResBlock(self.hidden_size)] * (self.hydra_num_layers - 1))
                ) for hydra_head_idx in range(self.hydra_num_heads)
            ])
        else:
            self.hydra_mlp = nn.ModuleList([
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * self.hydra_num_layers)
                ) for hydra_head_idx in range(self.hydra_num_heads)
            ])
        
        self.hydra_lm_head = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size) for _ in range(self.hydra_num_heads)
        ])
        if lm_head_init_weight is not None:
            print("Initializing HydraLM head with pretrained weights...")
            for i in range(hydra_num_heads):
            # Initialize the weights of each hydra_head using the base model's weights
                self.hydra_lm_head[i].weight.data[:] = lm_head_init_weight[:]

    def forward(self, base_hidden_states, input_ids=None, noise=None):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the MLP.
        """

        hydra_hidden_states = []
        if self.grounded_heads:
            assert input_ids is not None, "Input ids must be provided for grounded heads"
            with torch.inference_mode():
                input_embeds = self.input_embed_fn(input_ids)
            if noise is not None:
                input_embeds = input_embeds + noise
            hydra_inputs = [base_hidden_states]
            for i in range(self.hydra_num_heads):
                # Move input embeddings back one spot for each hydra head idx
                hydra_inputs.append(torch.roll(input_embeds, shifts=-(i+1), dims=1))
            
            for i in range(self.hydra_num_heads):
                head_input = torch.cat(hydra_inputs[:i + 2], dim=-1) 
                hydra_hidden_states.append(self.hydra_mlp[i](head_input))
        else:
            for i in range(self.hydra_num_heads):
                hydra_hidden_states.append(self.hydra_mlp[i](base_hidden_states))
        
        hydra_logits = []
        for i in range(self.hydra_num_heads):
            hydra_logits.append(self.hydra_lm_head[i](hydra_hidden_states[i]))
        
        return hydra_logits, hydra_hidden_states

    def _ungrounded_proposal(self, input_logits, base_hidden_states, hydra_buffers):
        hydra_logits = []
        for i in range(self.hydra_num_heads):
            hydra_hidden_state = self.hydra_mlp[i](base_hidden_states)
            hydra_logits.append(self.hydra_lm_head[i](hydra_hidden_state))
        hydra_logits = torch.stack(hydra_logits, dim=0)

        # Greedy decoding: Select the most probable candidate from the original logits.
        candidates_logit = torch.argmax(input_logits[:, -1]).unsqueeze(0)

        # Extract the TOPK candidates from the hydra logits.
        candidates_hydra_logits = []
        for hydra_head, beam_size in enumerate(hydra_buffers["beam_sizes"]):
            candidates_hydra_logits.append(torch.topk(hydra_logits[hydra_head, 0, -1], beam_size, dim = -1).indices)
        candidates_hydra_logits = torch.cat(candidates_hydra_logits)

        # Combine the selected candidate from the original logits with the topk hydra logits.
        candidates = torch.cat([candidates_logit, candidates_hydra_logits.view(-1)], dim=-1)

        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates = candidates[hydra_buffers["tree_indices"]]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[hydra_buffers["retrieve_indices"]]

        # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(0)
        return cart_candidates, tree_candidates
    
    def _grounded_proposal(self, input_logits, base_hidden_states, hydra_buffers):
        children_per_head = hydra_buffers["children_per_head"]
        children_to_expand_per_head = hydra_buffers["children_to_expand_per_head"]
        retrieve_indices = hydra_buffers["retrieve_indices"]

        candidate_id = torch.argmax(input_logits[:, -1]).unsqueeze(0)
        candidate_embedding = self.input_embed_fn(candidate_id).unsqueeze(0)

        candidates = torch.tensor([candidate_id], device=candidate_id.device)[None, ...]
        candidates_embeddings = torch.cat([base_hidden_states[:, -1:], candidate_embedding], dim=-1)

        for head_idx, (head_num_children, head_children_to_expand) in enumerate(zip(children_per_head, children_to_expand_per_head)):
            hydra_hidden_state = self.hydra_mlp[head_idx](candidates_embeddings)
            hydra_preds = self.hydra_lm_head[head_idx](hydra_hidden_state)
            next_head_embeddings = []

            for path_idx, (num_children, children_to_expand) in enumerate(zip(head_num_children, head_children_to_expand)):

                hydra_candidates = torch.topk(hydra_preds[:, path_idx], num_children, dim=-1).indices
                candidates = torch.cat([candidates, hydra_candidates], dim=-1)
                
                if children_to_expand > 0:
                    children_embeddings = self.input_embed_fn(hydra_candidates)[:, :children_to_expand]
                    repeat_slice = [path_idx] * children_to_expand
                    path_embeddings = candidates_embeddings[:, repeat_slice]
                    next_head_embeddings.append(torch.cat([path_embeddings, children_embeddings], dim=-1))
            
            if len(next_head_embeddings):
                # TODO (Zack): Determine assertion error about next_head_embeddings being empty before finishing tree
                candidates_embeddings = torch.cat(next_head_embeddings, dim=1)

        # TODO (Zack): Only selecting first batch element for now, change when doing bs > 1
        cart_candidates = candidates[0, retrieve_indices]

        return cart_candidates, candidates
    
    def proposal(
            self,
            input_logits,
            base_hidden_states,
            hydra_buffers,
            past_key_values=None, # Not actually used but consistent with other proposal functions,
            input_ids = None
        ):
        if self.grounded_heads:
            return self._grounded_proposal(input_logits, base_hidden_states, hydra_buffers)
        else:
            return self._ungrounded_proposal(input_logits, base_hidden_states, hydra_buffers)