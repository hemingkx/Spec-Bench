#Path and Parameters
MODEL_PATH=/your_own_model_path/
SPEC_BENCH_PATH=/your_own_path/Spec-Bench/
GPU_DEVICES=0

# Download the models
cd $MODEL_PATH

git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
git clone https://huggingface.co/lmsys/vicuna-13b-v1.3
git clone https://huggingface.co/lmsys/vicuna-33b-v1.3
git clone https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3
git clone https://huggingface.co/FasterDecoding/medusa-vicuna-13b-v1.3
git clone https://huggingface.co/FasterDecoding/medusa-vicuna-33b-v1.3
git clone https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3
git clone https://huggingface.co/yuhuili/EAGLE-Vicuna-13B-v1.3
git clone https://huggingface.co/yuhuili/EAGLE-Vicuna-33B-v1.3
git clone https://huggingface.co/yuhuili/EAGLE3-Vicuna1.3-13B
git clone https://huggingface.co/double7/vicuna-68m
git clone https://huggingface.co/ankner/hydra-vicuna-7b-v1.3
git clone https://huggingface.co/ankner/hydra-vicuna-13b-v1.3
git clone https://huggingface.co/ankner/hydra-vicuna-33b-v1.3

cd $SPEC_BENCH_PATH

# Run the evaluation
MODEL_SIZE=7
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
Eagle_PATH=$MODEL_PATH/EAGLE-Vicuna-${MODEL_SIZE}B-v1.3
Eagle3_PATH=$MODEL_PATH/EAGLE3-Vicuna1.3-${MODEL_SIZE}B
Medusa_PATH=$MODEL_PATH/medusa-vicuna-${MODEL_SIZE}b-v1.3
Hydra_PATH=$MODEL_PATH/hydra-vicuna-${MODEL_SIZE}b-v1.3
Drafter_PATH=$MODEL_PATH/vicuna-68m
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
TEMP=0.0
GPU_DEVICES=${GPU_DEVICES}

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --tree_model_path $Eagle_PATH

# Run the evaluation
MODEL_SIZE=13
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
Eagle_PATH=$MODEL_PATH/EAGLE-Vicuna-${MODEL_SIZE}B-v1.3
Eagle3_PATH=$MODEL_PATH/EAGLE3-Vicuna1.3-${MODEL_SIZE}B
Medusa_PATH=$MODEL_PATH/medusa-vicuna-${MODEL_SIZE}b-v1.3
Hydra_PATH=$MODEL_PATH/hydra-vicuna-${MODEL_SIZE}b-v1.3
Drafter_PATH=$MODEL_PATH/vicuna-68m
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
TEMP=0.0
GPU_DEVICES=${GPU_DEVICES}

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --tree_model_path $Eagle_PATH

# Run the evaluation
MODEL_SIZE=33
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
Eagle_PATH=$MODEL_PATH/EAGLE-Vicuna-${MODEL_SIZE}B-v1.3
Eagle3_PATH=$MODEL_PATH/EAGLE3-Vicuna1.3-${MODEL_SIZE}B
Medusa_PATH=$MODEL_PATH/medusa-vicuna-${MODEL_SIZE}b-v1.3
Hydra_PATH=$MODEL_PATH/hydra-vicuna-${MODEL_SIZE}b-v1.3
Drafter_PATH=$MODEL_PATH/vicuna-68m
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
TEMP=0.0
GPU_DEVICES=${GPU_DEVICES}

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --tree_model_path $Eagle_PATH
