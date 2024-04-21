"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

import sys

sys.path.append("../")

from model.rest.rest.model.utils import *
from model.rest.rest.model.rest_model import RestModel
from model.rest.rest.model.kv_cache import initialize_past_key_values
import draftretriever


def rest_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, top_p=0.0, datastore=None, num_draft=64, token_spans=None, max_steps=512):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    model.base_model.model.draft_mask = None
    logits = initialize_logits(
        input_ids, model, past_key_values
    )
    new_token = 0

    for idx in range(max_steps):
        candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
            logits,
            input_ids,
            datastore,
            token_spans,
            top_p,
            temperature,
            max_num_draft=num_draft,
            device=model.base_model.device
        )
        model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            draft_buffers["draft_position_ids"],
            input_ids,
            draft_buffers["retrieve_indices"],
        )
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, top_p
        )
        input_ids, logits, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            draft_buffers["retrieve_indices"],
            outputs,
            logits,
            new_token,
            past_key_values_data,
            current_length_data,
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
    return input_ids, new_token, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )
    parser.add_argument(
        "--datastore-path",
        type=str,
        required=True,
        help="The path of the datastore for retrival.",
    )
    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The maximum number of draft tokens.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="The maximum length of suffix for retrieval.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0

    args.model_id = args.model_id + "-temperature-" + str(args.temperature) + "-top_p-" + str(args.top_p)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = RestModel.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    token_spans = list(range(2, args.max_token_span + 1))[::-1]
    print("loading the datastore ...")
    datastore = draftretriever.Reader(
        index_file_path=args.datastore_path,
    )
    print("datastore loaded!")

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=rest_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        top_p=args.top_p,
        datastore=datastore,
        num_draft=args.num_draft,
        token_spans=token_spans,
    )

    reorg_answer_file(answer_file)