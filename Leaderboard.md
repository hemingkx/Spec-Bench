# Leaderboard

We present the evaluation results on our own devices for reference. All models were evaluated uniformly on Spec-Bench using **the same device and the testing environment**. We report the mean speedup over 3 different runs and #mean accepted tokens per decoding step (which is `1.00` for vanilla autoregressive decoding).

> ❗️It is important to note that model speedup rates may differ across various devices. For more precise speedup metrics, we recommend conducting evaluations of specific models on your intended devices.

> 🤔 This is a gentle reminder that while speedup is the primary metric for assessing Speculative Decoding methods, other benefits are worth considering. For example, [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) and [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) require no extra parameters, making it simpler to integrate a wider range of models.

## Leaderboard on 3090

- Device: a single NVIDIA GeForce RTX 3090 GPU (24GB) with 12 CPU cores
- Testing environment: Pytorch 2.5.1, under CUDA 12.1
- Experimental Settings: Vicuna-7B-v1.3, greedy decoding, FP16 precision, batch size = 1

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| [SAM-Decoding (EAGLE-2)](https://github.com/hyx1999/SAM-Decoding)🏅 |        **2.85x**        |  **1.83x**  |   **2.64x**    |     **2.15x**      |         2.63x          |         **2.10x**         |       **4.61**        | **2.38x** |
| [EAGLE2](https://github.com/SafeAILab/EAGLE)🥈                |          2.56x          |    1.78x    |     2.09x      |       2.07x        |       **2.66x**        |           1.86x           |         4.35          |   2.19x   |
| [EAGLE](https://huggingface.co/blog/assisted-generation)🥉    |          2.31x          |    1.72x    |     2.00x      |       1.91x        |         2.38x          |           1.75x           |         3.57          |   2.03x   |
| [Hydra](https://github.com/zankner/hydra)                    |          2.18x          |    1.79x    |     1.66x      |       1.85x        |         2.28x          |           1.62x           |         3.26          |   1.91x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.94x          |    1.37x    |     1.96x      |       1.86x        |         1.81x          |           1.83x           |         2.28          |   1.79x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.64x          |    1.15x    |     2.46x      |       1.28x        |         1.72x          |           1.71x           |         1.73          |   1.64x   |
| [Medusa](https://sites.google.com/view/medusa-llm)           |          1.61x          |    1.39x    |     1.28x      |       1.40x        |         1.64x          |           1.25x           |         2.32          |   1.44x   |
| [Recycling](https://github.com/Luowaterbi/TokenRecycling)    |          1.42x          |    1.29x    |     1.43x      |       1.30x        |         1.59x          |           1.36x           |         2.73          |   1.40x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.44x          |    1.15x    |     1.17x      |       1.35x        |         1.30x          |           1.26x           |         1.63          |   1.28x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.17x          |    1.00x    |     1.11x      |       1.06x        |         1.32x          |           1.06x           |         1.64          |   1.13x   |

## Leaderboard on A100

- Device: a single NVIDIA A100 GPU (80GB) with 64 CPU cores 
- Testing environment: Pytorch 2.0.1, under CUDA 11.4
- Experimental Settings: greedy decoding, FP16 precision, batch size = 1

### Vicuna-7B-v1.3

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| [EAGLE](https://sites.google.com/view/eagle-llm)🏅            |        **2.67x**        |  **1.99x**  |     2.23x      |     **2.12x**      |       **2.67x**        |         **2.04x**         |       **3.61**        | **2.29x** |
| [Hydra](https://github.com/zankner/hydra)🥈                   |          2.45x          |    1.94x    |     1.79x      |       2.03x        |         2.49x          |           1.77x           |         3.24          |   2.09x   |
| [Medusa](https://sites.google.com/view/medusa-llm)🥉          |          2.05x          |    1.73x    |     1.57x      |       1.75x        |         2.05x          |           1.51x           |         2.32          |   1.78x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.64x          |    1.04x    |   **2.43x**    |       1.14x        |         1.61x          |           1.71x           |         1.73          |   1.59x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.66x          |    1.13x    |     1.62x      |       1.49x        |         1.47x          |           1.55x           |         2.28          |   1.49x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.63x          |    1.31x    |     1.36x      |       1.66x        |         1.21x          |           1.73x           |         1.82          |   1.48x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.40x          |    1.14x    |     1.19x      |       1.24x        |         1.55x          |           1.09x           |         1.66          |   1.27x   |

### Vicuna-13B-v1.3

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| [EAGLE](https://sites.google.com/view/eagle-llm)🏅            |        **2.68x**        |  **1.96x**  |   **2.44x**    |     **2.04x**      |       **2.70x**        |         **2.23x**         |       **3.64**        | **2.34x** |
| [Hydra](https://github.com/zankner/hydra)🥈                   |          2.46x          |    1.90x    |     1.93x      |       1.96x        |         2.48x          |           1.92x           |         3.35          |   2.12x   |
| [Medusa](https://sites.google.com/view/medusa-llm)🥉          |          1.96x          |    1.66x    |     1.63x      |       1.63x        |         2.00x          |           1.58x           |         2.39          |   1.75x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.60x          |    1.13x    |     1.68x      |       1.39x        |         1.53x          |           1.67x           |         2.18          |   1.49x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.47x          |    1.02x    |     2.19x      |       1.03x        |         1.57x          |           1.71x           |         1.68          |   1.48x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.52x          |    1.17x    |     1.37x      |       1.53x        |         1.19x          |           1.55x           |         1.82          |   1.38x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.30x          |    1.06x    |     1.20x      |       1.12x        |         1.48x          |           1.12x           |         1.63          |   1.22x   |

### Vicuna-33B-v1.3

| Models                                                       | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------ | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| [EAGLE](https://sites.google.com/view/eagle-llm)🏅            |        **2.79x**        |  **2.05x**  |   **2.51x**    |     **2.17x**      |       **2.99x**        |         **2.27x**         |       **3.39**        | **2.47x** |
| [Hydra](https://github.com/zankner/hydra)🥈                   |          2.59x          |    2.01x    |     2.04x      |       2.11x        |         2.71x          |           2.06x           |         3.24          |   2.26x   |
| [Medusa](https://sites.google.com/view/medusa-llm)🥉          |          1.98x          |    1.73x    |     1.64x      |       1.66x        |         2.07x          |           1.62x           |         2.33          |   1.79x   |
| [SpS](https://huggingface.co/blog/assisted-generation)       |          1.75x          |    1.28x    |     1.76x      |       1.53x        |         1.69x          |           1.68x           |         2.01          |   1.61x   |
| [REST](https://sites.google.com/view/rest-llm)               |          1.63x          |    1.27x    |     1.45x      |       1.61x        |         1.30x          |           1.61x           |         1.80          |   1.48x   |
| [PLD](https://github.com/apoorvumang/prompt-lookup-decoding) |          1.44x          |    1.06x    |     2.00x      |       1.07x        |         1.55x          |           1.45x           |         1.55          |   1.42x   |
| [Lookahead](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) |          1.32x          |    1.08x    |     1.20x      |       1.16x        |         1.54x          |           1.15x           |         1.61          |   1.24x   |

