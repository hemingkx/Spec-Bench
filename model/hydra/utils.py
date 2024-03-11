from collections import defaultdict
import torch

TOPK=10 # topk for sparse tree

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_hydra_buffers(hydra_choices, device="cuda"):
    """
    Generate buffers for the Hydra structure based on the provided choices.
    
    Parameters:
    - hydra_choices (list): A nested list representing tree in the Hydra structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Hydra structure.
    """

    # Sort the hydra_choices based on their lengths and then their values
    sorted_hydra_choices = sorted(hydra_choices, key=lambda x: (len(x), x))
    hydra_len = len(sorted_hydra_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_hydra_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Hydra
    hydra_attn_mask = torch.eye(hydra_len, hydra_len)
    hydra_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_hydra_choice = sorted_hydra_choices[start + j]
            # retrieve ancestor position
            if len(cur_hydra_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_hydra_choice) - 1):
                ancestor_idx.append(sorted_hydra_choices.index(cur_hydra_choice[:c+1]) + 1)
            hydra_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Hydra structure
    hydra_tree_indices = torch.zeros(hydra_len, dtype=torch.long)
    hydra_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_hydra_choice = sorted_hydra_choices[start + j]
            hydra_tree_indices[start + j + 1] = cur_hydra_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Hydra structure
    hydra_position_ids = torch.zeros(hydra_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        hydra_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Hydra structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_hydra_choices)):
        cur_hydra_choice = sorted_hydra_choices[-i-1]
        retrieve_indice = []
        if cur_hydra_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_hydra_choice)):
                retrieve_indice.append(sorted_hydra_choices.index(cur_hydra_choice[:c+1]))
                retrieve_paths.append(cur_hydra_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Compute the max number of tokens a path can accept
    max_accepts = torch.sum(retrieve_indices > 0, dim=1)

    # Generate the num children per path
    # Make the assumption that we wouldn't expand a less likely child node if share same parent
    children_per_head = []
    num_heads = max([len(path) for path in hydra_choices])
    for head in range(1, num_heads + 1):
        head_paths = sorted([path for path in hydra_choices if len(path) == head])
        prefixes = sorted(list(set([tuple(head_path[:-1]) for head_path in head_paths])))
        if head == 1:
            children_per_head.append([len(head_paths)])
        else:
            children_per_prefix = [0 for _ in prefixes]
            for head_path in head_paths:
                prefix = tuple(head_path[:-1])
                children_per_prefix[prefixes.index(prefix)] += 1
            children_per_head.append(children_per_prefix)
    
    parents = set()
    nodes = set()
    for path in hydra_choices:
        parents.add(tuple(path[:-1]))
        nodes.add(tuple(path))
    parents_per_head = []
    for head in range(num_heads):
        parents_per_head.append(sorted([parent for parent in parents if len(parent) == head]))

    def descendant_exists(ancestor, edges_away):
        suff_depth_away = [path for path in nodes if len(path) >= len(ancestor) + edges_away]
        for cand in suff_depth_away:
            if tuple(cand[:len(ancestor)]) == ancestor:
                return True
        return False

    # Computing number of children that also have children
    children_to_expand_per_head = []
    for head_idx, (parents_at_head, head_children) in enumerate(zip(parents_per_head, children_per_head)):
        head_children_to_expand = []
        for parent, parent_num_children in zip(parents_at_head, head_children):
            parent_num_expand = 0
            for child_idx in range(parent_num_children):
                child_path = parent + (child_idx,)
                if descendant_exists(child_path, 1): parent_num_expand += 1
            head_children_to_expand.append(parent_num_expand)
        children_to_expand_per_head.append(head_children_to_expand)
    
    # Build the masks for the cross attention hydra head
    # TODO (ZACK): Considerations if no nodes w/ children
    sorted_hydra_choices_with_children = [
        hydra_choice for hydra_choice in sorted_hydra_choices if descendant_exists(tuple(hydra_choice), edges_away=1)
    ]
    sorted_hydra_choices_with_children_per_head = defaultdict(list)
    for hydra_choice_with_children in sorted_hydra_choices_with_children:
        sorted_hydra_choices_with_children_per_head[
            len(hydra_choice_with_children)].append(hydra_choice_with_children)
    sorted_hydra_choices_with_grandchildren = [
        hydra_choice for hydra_choice in sorted_hydra_choices if descendant_exists(tuple(hydra_choice), edges_away=2)
    ]
    sorted_hydra_choices_with_grandchildren_per_head = defaultdict(list)
    for hydra_choice_with_grandchildren in sorted_hydra_choices_with_grandchildren:
        sorted_hydra_choices_with_grandchildren_per_head[
            len(hydra_choice_with_grandchildren)].append(hydra_choice_with_grandchildren)

    # Always have at least one element
    proposal_cross_attn_masks_per_head = [torch.tensor([[1]])]
    if sorted_hydra_choices_with_children:
        for head_idx in range(max(sorted_hydra_choices_with_children_per_head.keys())):
            head_choices_with_children = sorted_hydra_choices_with_children_per_head[head_idx + 1]
            ancestor_choices = [choice for h, choices in sorted_hydra_choices_with_grandchildren_per_head.items() for choice in choices if h < head_idx + 1]
            proposal_cross_attn_mask = torch.zeros(len(head_choices_with_children), len(ancestor_choices) + 1)
            proposal_cross_attn_mask[:, 0] = 1
            for choice_idx, head_choice_with_children in enumerate(head_choices_with_children):
                for ancestor_idx, ancestor_choice in enumerate(ancestor_choices):
                    if head_choice_with_children[:len(ancestor_choice)] == ancestor_choice:
                        proposal_cross_attn_mask[choice_idx, ancestor_idx + 1] = 1
            proposal_cross_attn_masks_per_head.append(proposal_cross_attn_mask.unsqueeze(0).unsqueeze(0))

    # Get the beam size for each head
    beam_sizes = []
    num_heads = len(max(hydra_choices, key=len))
    for head_idx in range(1, num_heads + 1):
        head_choices = [choice for choice in hydra_choices if len(choice) == head_idx]
        head_all_k = [hc[head_idx - 1] for hc in head_choices]
        beam_sizes.append(max(head_all_k) + 1)

    # Aggregate the generated buffers into a dictionary
    hydra_buffers = {
        "hydra_attn_mask": hydra_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": hydra_tree_indices,
        "hydra_position_ids": hydra_position_ids,
        "retrieve_indices": retrieve_indices,
        "max_accepts": max_accepts,
        "beam_sizes": beam_sizes
        }
    
    # Move the tensors in the dictionary to the specified device
    hydra_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in hydra_buffers.items()
    }
    hydra_buffers["proposal_cross_attn_masks"] = [
        proposal_cross_attn_mask.clone().to(device) for proposal_cross_attn_mask in proposal_cross_attn_masks_per_head
    ]
    hydra_buffers["children_per_head"] = children_per_head
    hydra_buffers["children_to_expand_per_head"] = children_to_expand_per_head

    return hydra_buffers


def initialize_hydra(input_ids, model, hydra_attn_mask, past_key_values, proposal_cross_attn_masks):
    """
    Initializes the Hydra structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Hydra logits, original model outputs, and logits.
    2. Sets the Hydra attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (HydraLMHead): The model containing the Hydra layers and base model.
    - hydra_attn_mask (torch.Tensor): The attention mask designed specifically for the Hydra structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - hydra_logits (torch.Tensor): Logits from the Hydra heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    _, outputs, logits, _ = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    if model.hidden_state_offset == 0:
        hidden_states = outputs[0].clone()
    else:
        hidden_states = outputs[1][-(model.hidden_state_offset + 1)].clone()
    model.base_model.model.hydra_mask = hydra_attn_mask
    if model.hydra_head_arch == "cross-attn":
        model.hydra_head.proposal_hydra_masks = proposal_cross_attn_masks
    return hidden_states, logits


def reset_hydra_mode(
    model,
):
    """
    Resets the Hydra settings and the past key-values to their initial state.

    This function ensures that after any operations involving Hydra,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Hydra attention mask in the base model.
    2. Resets the Hydra mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (HydraLMHead): The model containing the Hydra layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - past_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    model.base_model.model.hydra_mask = None
    model.base_model.model.hydra_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(hydra_logits, logits, beam_sizes, tree_indices, retrieve_indices):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - hydra_logits (torch.Tensor): Logits associated with the Hydra structure.
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """

    # Greedy decoding: Select the most probable candidate from the original logits.
    candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)

    # Extract the TOPK candidates from the hydra logits.
    candidates_hydra_logits = []
    for hydra_head, beam_size in enumerate(beam_sizes):
        candidates_hydra_logits.append(torch.topk(hydra_logits[hydra_head, 0, -1], beam_size, dim = -1).indices)
    candidates_hydra_logits = torch.cat(candidates_hydra_logits)

    # Combine the selected candidate from the original logits with the topk hydra logits.
    candidates = torch.cat([candidates_logit, candidates_hydra_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, tree_candidates


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    hydra_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - hydra_position_ids (torch.Tensor): Positional IDs associated with the Hydra structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns hydra logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the Hydra position IDs to the length of the input sequence.
    position_ids = hydra_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return logits for the Hydra structure, original logits, and possibly other outputs.
    _, outputs, tree_logits, _ = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    if model.hidden_state_offset == 0:
        hidden_states = outputs[0].clone()
    else:
        hidden_states = outputs[1][-(model.hidden_state_offset + 1)].clone()

    # Reorder the obtained logits and hidden states based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    hidden_states = hidden_states[0, retrieve_indices]

    return hidden_states, logits

def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold, posterior_alpha, max_accepts
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    # TODO (ZACK): Figure out lucky token intersection issue for temp > 0 case
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = (torch.minimum(candidates_accept_length, max_accepts)).max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    # Calculate posterior probabilities and thresholds for candidate selection
    posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
    candidates_prob = torch.gather(
        posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    posterior_entropy = -torch.sum(
        posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
    )  # torch.sum(torch.log(*)) is faster than torch.prod
    threshold = torch.minimum(
        torch.ones_like(posterior_entropy) * posterior_threshold,
        torch.exp(-posterior_entropy) * posterior_alpha,
    )
    posterior_mask = candidates_prob > threshold
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

    # Choose the best candidate based on the evaluated posterior probabilities
    accept_length = (torch.minimum(candidates_accept_length, max_accepts)).max()
    if accept_length == 0:
        # If no candidates are accepted, just choose the first one
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidates = torch.where(torch.minimum(candidates_accept_length, max_accepts) == accept_length)[0]
        # Accept the best one according to likelihood
        likelihood = torch.sum(
            torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
        )
        best_candidate = best_candidates[torch.argmax(likelihood)]
    return best_candidate, accept_length


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits,
    hidden_states,
    new_token,
    past_key_values_data,
    current_length_data,
    hydra_head_arch,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - logits, hydra_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - hydra_logits (torch.Tensor): Updated hydra logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # If hydra head arch w/ attn don't want to update its length
    if hydra_head_arch in ["prefix-mlp", "cross-attn", "eagle-attn"]:
        current_length_data[-2:] = prev_input_len 

    # Extract logits and hydra logits for the accepted tokens
    logits = logits[None, best_candidate,  : accept_length + 1]
    # logits = logits[None, best_candidate,  accept_length: accept_length + 1]

    # Need to select hidden state based on what we will be sampling
    hidden_states = hidden_states[None, best_candidate, : accept_length + 1]
    # hidden_states = hidden_states[None, best_candidate, accept_length: accept_length + 1]
    new_token += accept_length + 1

    return input_ids, logits, hidden_states, new_token

if __name__ == "__main__":
    from hydra.model.hydra_choices import mc_sim_7b_63, debug

    res = generate_hydra_buffers(mc_sim_7b_63, "cpu")
    # print(res["retrieve_indices"])
    # print(res["max_accepts"][23])
    # print(res["retrieve_indices"][23])
    # print(res["children_per_head"])
    # print(res["children_to_expand_per_head"])
    print(res["proposal_cross_attn_masks"])