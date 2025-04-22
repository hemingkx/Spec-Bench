This file is the instruction of Spec-Bench evaluation

## Code Base

Download the code base

```
git clone https://github.com/hemingkx/Spec-Bench.git
```

## Installation

```
conda create -n specbench python=3.12
conda activate specbench
# install suitable pytorch version based on your own cuda env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd Spec-Bench
pip install -r requirements.txt
```

Please fill in the specific testing environment below:

|         | Device                                            | Pytorch | CUDA |
| ------- | ------------------------------------------------- | ------- | ---- |
| Example | a single NVIDIA A100 GPU (80GB) with 96 CPU cores | 2.5.1   | 11.5 |
| Yours   |                                                   |         |      |

## Evaluation

Replace `eval.sh` with `eval_script.sh` in the Spec-Bench folder, modify the following path and parameters:

- `MODEL_PATH`: the path to save all the downloaded model checkpoints
- `SPEC_BENCH_PATH`: the path of Spec-Bench code base
- `GPU_DEVICES`: GPU device id

Run the `eval_script.sh`, the results will be stored in `Spec-Bench/data/spec_bench/model_answer/`.



