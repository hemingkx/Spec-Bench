Vicuna_PATH=/your_own_path/vicuna-7b-v1.3
Eagle_PATH=/your_own_path/EAGLE-Vicuna-7B-v1.3
Eagle3_PATH=/your_own_path/EAGLE3-Vicuna1.3-13B
Medusa_PATH=/your_own_path/medusa-vicuna-7b-v1.3
Hydra_PATH=/your_own_path/hydra-vicuna-7b-v1.3
Drafter_PATH=/your_own_path/vicuna-68m
MODEL_NAME=vicuna-7b-v1.3
TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-68m-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
