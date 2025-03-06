"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_eval, reorg_answer_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from model.samd import SamdConfig, SamdModel, SamdGenerationConfig, DraftModel, load_sam

def samd_forward(
    inputs, 
    model: SamdModel, 
    tokenizer: PreTrainedTokenizer, 
    max_new_tokens: int, 
    temperature: float = 0.0,
    do_sample: bool = False
):
    max_cache_len = model.lm.config.max_position_embeddings
    input_ids = inputs.input_ids
    outputs = model.generate(
        input_ids,
        generation_config=SamdGenerationConfig(
            max_new_tokens=max_new_tokens,
            max_cache_len=max_cache_len,
            greedy=not do_sample,
            temperature=temperature
        ),
    )
    output_ids = outputs.output_ids
    new_token = outputs.decode_tokens
    step = outputs.decode_steps
    accept_length_list = outputs.accepet_length_per_step
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
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
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
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
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--samd_n_predicts",
        type=int,
        default=40
    )
    parser.add_argument(
        "--static_sam_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--samd_len_threshold",
        type=int,
        default=5
    )
    parser.add_argument(
        "--samd_len_bias",
        type=int,
        default=5
    )
    parser.add_argument(
        "--samd_tree_path",
        type=str,
        default=None
    )
    parser.add_argument("--tree_method", type=str, default=None, choices=["token_recycle", "eagle2"])
    parser.add_argument("--tree_model_path", type=str, default="path/to/EAGLE-Vicuna-7B-v1.3")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    
    if args.num_gpus_total == 1:
        device_map = "cuda"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=args.attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    device = next(model.lm_head.parameters()).device
    sam = load_sam(args.static_sam_path) if args.static_sam_path is not None else None
    if sam is not None:
        sam.device = device
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
        len_threshold=args.samd_len_threshold,
        len_bias=args.samd_len_bias,
        tree_path=args.samd_tree_path,
    )
    draft = DraftModel(
        samd_config, 
        sam_static=sam,
        lm=model,
        dtype=str_to_torch_dtype(args.dtype),
        device=device,
    )
    samd_model = SamdModel(
        samd_config, 
        model, 
        draft, 
        tokenizer.eos_token_id,
        str_to_torch_dtype(args.dtype),
        device,
    )
    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=samd_model,
        tokenizer=tokenizer,
        forward_func=samd_forward,
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
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)
