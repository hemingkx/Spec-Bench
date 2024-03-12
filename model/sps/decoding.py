import torch
import copy
import warnings

from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput, ModelOutput
from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
    CandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
import torch.distributed as dist
import os, time
FUNC_MAP = {}
CONFIG_MAP = {}
COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decider generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


def assisted_decoding(
    self,
    input_ids: torch.LongTensor,
    assistant_model: Optional["PreTrainedModel"] = None,
    candidate_generator: Optional["CandidateGenerator"] = None,
    do_sample: bool = False,
    logits_processor: Optional[LogitsProcessorList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
    **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
    candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
    models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.candidate_decoding`] directly. Use
    generate() instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        candidate_generator (`CandidateGenerator`, *optional*):
            A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
            more information, the documentation of [`CandidateGenerator`] should be read. Only one of `assistant_model` or `candidate_generator` should be passed as input to this function.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
   """
    # handling deprecated arguments
    if (assistant_model is None) == (candidate_generator is None):
        raise ValueError("One (and only one) of `assistant_model` and `candidate_generator` should be defined.")

    if assistant_model is not None:
        candidate_generator = AssistedCandidateGenerator(
            input_ids=input_ids,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            eos_token_id=eos_token_id,
        )
        warnings.warn(
            "Passing `assistant_model` to `assisted_decoding` is deprecated and will be removed in v4.38. "
            "Pass the `candidate_generator` argument instead.",
            FutureWarning,
        )

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if eos_token_id is not None and pad_token_id is None:
        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    # other auxiliary variables
    max_len = stopping_criteria[0].max_length

    this_peer_finished = False  # used by synced_gpus only
    step = 0
    accept_length_list = []
    while True:
        step += 1
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        cur_len = input_ids.shape[-1]

        #  1. Fetch candidate sequences from a `CandidateGenerator`
        candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)
        candidate_input_ids = candidate_input_ids.to(self.device)
        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(self.device)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        last_assistant_token_is_eos = (
            ~candidate_input_ids[:, -1]
            .tile(eos_token_id_tensor.shape[0], 1)
            .ne(eos_token_id_tensor.unsqueeze(1))
            .prod(dim=0)
            .bool()
        )

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Prepare the model inputs
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = _prepare_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
        )
        candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)

        # 2.2. Run a forward pass on the candidate sequence
        outputs = self(
            **model_inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # 2.3. Process the new logits
        new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
        if len(logits_warper) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

        # 3. Select the accepted tokens. There are two possible cases:
        # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
        # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
        max_matches = max_len - cur_len - 1
        if do_sample and candidate_logits is not None:
            valid_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                last_assistant_token_is_eos,
                max_matches,
            )

        # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
        # original model logits with the candidate tokens. We can keep the candidate tokens until the first
        # mismatch, or until the max length is reached.
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

            # Ensure we don't generate beyond max_len or an EOS token
            if last_assistant_token_is_eos and n_matches == candidate_length:
                n_matches -= 1
            n_matches = min(n_matches, max_matches)
            valid_tokens = selected_tokens[:, : n_matches + 1]

        # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
        # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
        # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
        # is no match.

        # 4.1. Get the valid continuation, after the matching tokens
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]

        # 4.2. Discard past key values relative to unused assistant tokens
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        accept_length_tree = new_cur_len - cur_len
        accept_length_list.append(accept_length_tree)

        # 5. Update the candidate generation strategy if needed
        candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Store scores, attentions and hidden_states when required
        # Assistant: modified to append one tuple element per token, as in the other generation methods.
        if return_dict_in_generate:
            if output_scores:
                scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))

            if "past_key_values" not in model_kwargs:
                added_len = new_cur_len
            else:
                added_len = n_matches + 1

            if output_attentions:
                if self.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(
                        cross_attentions, outputs.cross_attentions, cur_len, added_len
                    )
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.decoder_attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
                else:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
            if output_hidden_states:
                if self.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                    )
                else:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                    )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                input_ids[:, -1]
                .tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    idx = step - 1

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids, idx, accept_length_list


def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    last_assistant_token_is_eos,
    max_matches,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if last_assistant_token_is_eos and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        n_matches = min(n_matches, max_matches)

        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = min(candidate_logits.shape[1], max_matches)
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches


def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs