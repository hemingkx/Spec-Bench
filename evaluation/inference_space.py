"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.space.modeling_llama_space import LlamaForCausalLM
import torch
from transformers.models.llama.modeling_llama import _make_causal_mask, _expand_mask
import numpy as np

def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    input_ids = inputs.input_ids
    output_ids = model.generate(
        input_ids,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    new_token = len(output_ids[0][len(input_ids[0]):])
    idx = new_token - 1
    accept_length_list = [1] * new_token
    print(tokenizer.batch_decode(output_ids[0][len(input_ids[0]):]))
    return output_ids, new_token, idx, accept_length_list


def space_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False, MASK_ID=32002, MASK_NUM=5, USE_CACHE=True, MAX_NEW_TOKENS=512):
    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    Lm = MASK_ID * torch.ones((batch_size, MASK_NUM), dtype=input_ids.dtype, device=input_ids.device)
    Lc = torch.tensor([MASK_ID for _ in range(MASK_NUM)], dtype=input_ids.dtype, device=input_ids.device).repeat(
        batch_size, 1)
    Pc = torch.tensor([torch.finfo(torch.float16).max for _ in range(MASK_NUM)], dtype=torch.float16,
                      device=input_ids.device).repeat(batch_size, 1)

    past_key_values = None
    new_decode_step = 0
    accept_length_list = []
    while True:
        new_decode_step += 1
        accept_length = 0
        input_ids_idx = input_ids.shape[-1]

        # 构建输入
        tmp = torch.hstack([torch.hstack([Lc[:, i: i + 1], Lm]) for i in range(MASK_NUM)])
        input_ids_extend = torch.hstack([input_ids, Lm, tmp])

        # 构建注意力矩阵和位置编码
        combined_attention_mask = _make_causal_mask(
            input_ids_extend.shape,
            torch.float16,
            device=input_ids_extend.device,
            past_key_values_length=0,
        )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            torch.cat([attention_mask, attention_mask.new_ones((batch_size, MASK_NUM * (MASK_NUM + 2)))], dim=-1),
            torch.float16, tgt_len=input_ids_extend.shape[-1]).to(
            input_ids_extend.device
        )
        for idx in range(input_ids_idx, expanded_attn_mask.shape[-1], MASK_NUM + 1):
            expanded_attn_mask[:, :, idx + MASK_NUM:, idx: idx + MASK_NUM] = torch.finfo(torch.float16).min
        attention_mask_extend = expanded_attn_mask + combined_attention_mask
        position_ids = (attention_mask_extend == 0).sum(axis=-1).squeeze(0) - 1

        # run LLM
        if past_key_values:
            # device = 'cpu'
            kv_cache_idx = torch.tensor(
                [input_ids_idx - new_generate_token + i * (MASK_NUM + 1) - 1 for i in range(1, new_generate_token)],
                dtype=int)
            kv_cache_idx = torch.hstack([torch.arange(0, input_ids_idx - new_generate_token, dtype=int), kv_cache_idx])

            past_key_values = [(kv_cache[0][:, :, kv_cache_idx, :], kv_cache[1][:, :, kv_cache_idx, :]) for kv_cache
                               in past_key_values]

            input_ids_extend = input_ids_extend[:, input_ids_idx - 1:]
            position_ids = position_ids[:, input_ids_idx - 1:]
            attention_mask_extend = attention_mask_extend[:, :, input_ids_idx - 1:, :]
            input_ids_idx = 1

        with torch.no_grad():
            outputs = model(input_ids_extend, attention_mask=attention_mask_extend, position_ids=position_ids,
                            past_key_values=past_key_values,
                            return_dict=True, use_cache=USE_CACHE)
        past_key_values = outputs.past_key_values

        logits = torch.softmax(outputs.logits, dim=-1)  # normalized logits

        new_generate_token = 0
        select_idx = input_ids_idx
        next_token_logit = logits[:, input_ids_idx - 1, :]
        for idx in range(MASK_NUM):
            if do_sample:
                condition = np.random.uniform() <= next_token_logit[:, Lc[:, idx]] / Pc[:, idx]
            else:
                condition = torch.argmax(next_token_logit, dim=-1) == Lc[:, idx]

            # condition = False
            if condition:
                next_tokens = Lc[:, idx]
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                accept_length += 1
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                                           dim=-1)
                new_generate_token += 1
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(
                        dim=0)
                )
                if unfinished_sequences.max() == 0:
                    break

                next_token_logit = logits[:, input_ids_idx - 1 + (idx + 1) * (MASK_NUM + 1), :]

                select_idx += MASK_NUM + 1
            else:
                break

        if do_sample:
            # torch.random.manual_seed(RANDOM_SEED)
            next_tokens = torch.multinomial(next_token_logit, num_samples=1).squeeze(1)
            Lc, Pc = [], []
            for bs in range(batch_size):
                candidate_tokens = torch.multinomial(logits[bs, select_idx: select_idx + MASK_NUM, :], num_samples=1)

                Lc.append(candidate_tokens.reshape(1, -1))
                Pc.append(
                    torch.tensor([logits[bs, select_idx + i, k] for i, k in enumerate(candidate_tokens)]).reshape(1,
                                                                                                                  -1))
            Lc = torch.cat(Lc).to(device)
            Pc = torch.cat(Pc).to(device)
        else:
            next_tokens = torch.argmax(next_token_logit, dim=-1)
            # generate new candidate tokens
            Pc, Lc = torch.max(logits[:, select_idx: select_idx + MASK_NUM, :], dim=-1)

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        accept_length += 1
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        new_generate_token += 1

        accept_length_list.append(accept_length)

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        if unfinished_sequences.max() == 0 or input_ids.shape[-1] - prompt_len >= MAX_NEW_TOKENS:
            break

    new_token = input_ids[:, inputs['input_ids'].shape[-1]:]
    assert sum(accept_length_list) == len(new_token[0])
    assert len(accept_length_list) == new_decode_step
    return input_ids, len(new_token[0]), new_decode_step, accept_length_list


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

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=space_forward,
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
