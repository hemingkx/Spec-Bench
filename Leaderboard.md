# Leaderboard

We present the evaluation results on our own devices for reference. All models were evaluated uniformly on Spec-Bench using **the same device and the testing environment**. We report the mean speedup over 3 different runs.

> ❗️It is important to note that model speedup rates may differ across various devices. For more precise speedup metrics, we recommend conducting evaluations of specific models on your intended devices.

> 🤔 This is a gentle reminder that while speedup is the primary metric for assessing Speculative Decoding methods, other benefits are worth considering. For example, [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) and [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) require no extra parameters, making it simpler to integrate a wider range of models.

## Leaderboard on 3090

- Device: a single NVIDIA GeForce RTX 3090 GPU (24GB) with 12 CPU cores
- Testing environment: Pytorch 2.0.1, under CUDA 11.8
- Experimental Settings: Vicuna-7B-v1.3, greedy decoding, FP16 precision, batch size = 1

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| [EAGLE](https://sites.google.com/view/eagle-llm)🏅            |        **2.35x**        |  **1.79x**  |     2.04x      |     **1.96x**      |       **2.44x**        |         **1.80x**         |       **3.59**        | **2.08x** |
| [Hydra](https://github.com/zankner/hydra)🥈                   |          2.14x          |    1.74x    |     1.65x      |       1.91x        |         2.29x          |           1.60x           |         3.26          |   1.90x   |
| [SpS](https://huggingface.co/blog/assisted-generation)🥉      |          1.92x          |    1.33x    |     1.93x      |       1.81x        |         1.84x          |           1.76x           |         2.29          |   1.77x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.63x          |    1.11x    |   **2.41x**    |       1.27x        |         1.70x          |           1.66x           |         1.74          |   1.62x   |
| [Medusa](https://sites.google.com/view/medusa-llm)           |          1.65x          |    1.41x    |     1.33x      |       1.44x        |         1.69x          |           1.29x           |         2.32          |   1.48x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.49x          |    1.23x    |     1.26x      |       1.39x        |         1.34x          |           1.71x           |         1.41          |   1.39x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.15x          |    0.98x    |     1.07x      |       1.06x        |         1.32x          |           1.03x           |         1.65          |   1.11x   |

## Leaderboard on A100

- Device: a single NVIDIA A100 GPU (80GB) with 64 CPU cores 
- Testing environment: Pytorch 2.0.1, under CUDA 11.4
- Experimental Settings: greedy decoding, FP16 precision, batch size = 1

### Vicuna-7B-v1.3

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------: |
| [Medusa](https://sites.google.com/view/medusa-llm)🏅          |        **2.79x**        |  **2.36x**  |     2.14x      |     **2.36x**      |         2.77x          |           2.05x           | **2.42x** |
| [EAGLE](https://sites.google.com/view/eagle-llm)🥈            |          2.75x          |    2.08x    |     2.32x      |       2.23x        |       **2.79x**        |         **2.15x**         |   2.39x   |
| [Hydra](https://github.com/zankner/hydra)🥉                   |          2.51x          |    2.01x    |     1.84x      |       2.09x        |         2.58x          |           1.83x           |   2.15x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.95x          |    1.61x    |     1.63x      |       1.73x        |         2.16x          |           1.50x           |   1.77x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.67x          |    1.06x    |   **2.59x**    |       1.16x        |         1.63x          |           1.83x           |   1.66x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.72x          |    1.38x    |     1.46x      |       1.80x        |         1.31x          |           1.87x           |   1.59x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.78x          |    1.19x    |     1.78x      |       1.58x        |         1.54x          |           1.69x           |   1.59x   |

### Vicuna-13B-v1.3

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------: |
| [EAGLE](https://sites.google.com/view/eagle-llm)🏅            |        **2.88x**        |  **2.24x**  |   **2.52x**    |     **2.24x**      |       **2.90x**        |         **2.34x**         | **2.53x** |
| [Hydra](https://github.com/zankner/hydra)🥈                   |          2.51x          |    1.96x    |     1.96x      |       2.02x        |         2.55x          |           1.97x           |   2.17x   |
| [Medusa](https://sites.google.com/view/medusa-llm)🥉          |          2.39x          |    2.12x    |     1.92x      |       2.07x        |         2.49x          |           1.88x           |   2.16x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.73x          |    1.25x    |     1.76x      |       1.53x        |         1.68x          |           1.73x           |   1.61x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.68x          |    1.31x    |     1.51x      |       1.67x        |         1.29x          |           1.96x           |   1.56x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.53x          |    1.08x    |     2.25x      |       1.09x        |         1.65x          |           1.72x           |   1.54x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.57x          |    1.34x    |     1.39x      |       1.40x        |         1.82x          |           1.32x           |   1.48x   |

### Vicuna-33B-v1.3

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------: |
| [EAGLE](https://sites.google.com/view/eagle-llm)🏅            |        **2.81x**        |  **2.14x**  |   **2.53x**    |     **2.19x**      |       **3.01x**        |         **2.31x**         | **2.50x** |
| [Hydra](https://github.com/zankner/hydra)🥈                   |          2.63x          |    2.05x    |     2.08x      |       2.16x        |         2.76x          |           2.11x           |   2.31x   |
| [Medusa](https://sites.google.com/view/medusa-llm)🥉          |          2.22x          |    1.95x    |     1.85x      |       1.87x        |         2.32x          |           1.84x           |   2.01x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.79x          |    1.31x    |     1.80x      |       1.57x        |         1.73x          |           1.69x           |   1.65x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.71x          |    1.39x    |     1.57x      |       1.69x        |         1.34x          |           1.89x           |   1.59x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.45x          |    1.06x    |     1.98x      |       1.07x        |         1.54x          |           1.43x           |   1.41x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.46x          |    1.21x    |     1.32x      |       1.29x        |         1.71x          |           1.28x           |   1.38x   |

