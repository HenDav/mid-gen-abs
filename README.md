# Dynamic Mid-Generation Abstention for LLMs

This repository contains the code for reproducing the experiments in **"Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning"** (ICML 2026).

**Authors:** Hen Davidov, Nachshon Cohen, Oren Kalinsky, Yaron Fairstein, Guy Kushilevitz, Ram Yazdi, Patrick Rebeschini

## Overview

We propose a method for **dynamic mid-generation abstention** in large language models. Unlike prior approaches that make abstention decisions at a fixed position (before or after generation), our method learns to predict at every token whether generation should continue, achieving higher selective accuracy with fewer tokens spent.

## Installation

```bash
conda env create -f environment.yaml
conda activate abstention
```

## Data

Trajectory CSVs (~2.4 GB) are hosted on HuggingFace at [HenDav/mid-gen-abs-data](https://huggingface.co/datasets/HenDav/mid-gen-abs-data). Download them before running any figure or analysis script:

```bash
pip install huggingface_hub   # if not already installed
python download_data.py
```

This populates `traj_csvs/{main,constant_step,transfer}/` which all plotting scripts expect.

## Repository Structure

```
├── environment.yaml                  # Conda environment
├── download_data.py                  # Download traj_csvs from HuggingFace
│
├── value_head_model.py               # ValueHeadModel + TokenwiseValueHead
├── datasets.py                       # Dataset loading and preprocessing
├── early_abstention.py               # Abstention evaluation and stopping criteria
├── split_dataset.py                  # Train/test splitting
├── utils.py                          # Shared analysis utilities
│
├── train_our_method.py               # Train dynamic abstention (ours)
├── train_first_token_baseline.py     # Train constant-step probe baseline
├── train_lora_abstention.py          # Train LoRA abstention baseline
│
├── plot_abstention_rate_analysis.py  # Run evaluation → generate traj CSVs
├── main_results.py                   # Paper figures: accuracy & reward matrices
├── recalibrated_reward_plot.py       # Paper figures: reward vs r_⊥ matrices
├── calibration_comparison_figure.py  # Calibration curves (V_0 vs V_τ)
├── rebuttal_experiments.py           # Rebuttal analyses (calibration, robustness, timing)
├── poster_plots.py                   # Poster/widget panels (PDF + JSON per figure)
│
└── figures/rebuttal/                 # Pre-computed rebuttal CSVs (committed)
    ├── additive_noise.csv
    ├── cross_split_calibration.csv
    └── ...
```

## Training

### Dynamic Abstention (Our Method)

```bash
python train_our_method.py \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --data_path data/train.jsonl \
    --output_dir outputs/dynamic \
    --device cuda:0
```

Trains a `TokenwiseValueHead` that predicts correctness at every token, enabling abstention at any generation step.

### Constant-Step Probe Baseline

```bash
python train_first_token_baseline.py \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --data_path data/train.jsonl \
    --output_dir outputs/baseline \
    --device cuda:0
```

### LoRA Abstention Baseline

```bash
python train_lora_abstention.py \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --data_path data/train.jsonl \
    --output_dir outputs/lora \
    --device cuda:0
```

## Evaluation → Trajectory CSVs

Run all models on a test set to produce the trajectory CSV files consumed by all plotting scripts:

```bash
python plot_abstention_rate_analysis.py \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --data-path data/test.jsonl \
    --baseline-path outputs/baseline/trained_value_head_first_token.pth \
    --full-model-path outputs/dynamic/trained_value_head.pth \
    --lora-model-path outputs/lora/final_model \
    --output-folder traj_csvs/main/gsm8k_qwen \
    --device cuda:0
```

Repeat for each dataset/model combination (GSM8K and OlympiadBench × Phi-3 and Qwen, plus RealToxicityPrompts). Transfer experiments use `traj_csvs/transfer/` and constant-step experiments use `traj_csvs/constant_step/`.

## Reproducing Figures

After downloading data (`python download_data.py`), all figure scripts auto-discover CSVs from `traj_csvs/` and save directly to `figures/`:

```bash
# Main paper figures → figures/
python main_results.py
python recalibrated_reward_plot.py
python calibration_comparison_figure.py

# Rebuttal figures → figures/
python rebuttal_experiments.py

# Poster panels (PDF + JSON per figure) → poster/figs/
python poster_plots.py
```

## Key Components

### ValueHeadModel (`value_head_model.py`)

Wraps a frozen LLM backbone with a lightweight trainable value head:

```python
from value_head_model import ValueHeadModel, TokenwiseValueHead

value_head = TokenwiseValueHead(hidden_dim=4096)
model = ValueHeadModel(
    model_name_or_path="Qwen/Qwen2.5-Math-7B-Instruct",
    value_head=value_head,
    freeze_base_model=True,
    device="cuda",
)
```

The `TokenwiseValueHead` is a two-layer MLP: `Linear → Tanh → Dropout → Linear → scalar`.

### Dynamic Abstention at Inference

```python
generated_ids, final_value = model.generate_with_abstention(
    input_ids,
    threshold=0.5,
    max_length=512,
    tokenizer=tokenizer,
)
```

Generation stops early when the predicted value drops below the threshold.

## Supported Models

- `Qwen/Qwen2.5-Math-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`

GPU memory: ~14 GB for 7B models (frozen backbone + value head). Gradient checkpointing is enabled by default.

## Citation

```bibtex
@inproceedings{davidov2026knowing,
  title={{Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning}},
  author={Davidov, Hen and Cohen, Nachshon and Kalinsky, Oren and Fairstein, Yaron and Kushilevitz, Guy and Yazdi, Ram and Rebeschini, Patrick},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

MIT License
