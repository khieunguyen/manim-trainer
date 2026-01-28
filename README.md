# ManimTrainer: Fine-Tuning LLMs for Manim Code Generation using Visually Grounded Reinforcement Learning

[![License: CC-BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Manim Community](https://img.shields.io/badge/Manim-Community-green)](https://www.manim.community/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/SuienR/ManimBench-v1)
[![Unsloth](https://img.shields.io/badge/%F0%9F%A6%A5%20Unsloth-Accelerated-orange)](https://github.com/unslothai/unsloth)

A toolkit for fine-tuning Large Language Models (LLMs) to generate [Manim](https://www.manim.community/) animation code using Supervised Fine-Tuning (SFT) and Visually Grounded Reinforcement Learning using Group Relative Policy Optimization (GRPO/GSPO) techniques.

**📝 Research Paper: [Coming Soom]()**

**🚧 Note:** This project is still in development. Some features may not be fully implemented or tested yet.

## 📋 Table of Contents

- [Overview](#-overview) 
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Dataset](#-dataset)
- [Models](#-models)
- [Configuration](#-configuration)
- [Citation](#-citation)
- [License](#-license)

## 🔍 Overview

This repository contains the implementation for fine-tuning LLMs to generate Manim code from natural language descriptions. The approach combines:

1. **Supervised Fine-Tuning (SFT)** - Initial training on curated Manim code examples
2. **GRPO/GSPO Training** - Reinforcement learning-based optimization using text and visual rewards
3. **Wholistic Evaluation** - Automated evaluation pipeline including code execution and video quality assessment
4. **Multi-modal Reward System** - Combining code execution success, text similarity, and visual comparison metrics

## ✨ Features

- **Multiple Training Modes**: Support for SFT, GRPO, and GSPO training strategies
- **Efficient Fine-Tuning**: LoRA/QLoRA support via [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient training
- **Comprehensive Evaluation**: Automated evaluation pipeline with code execution verification
<!-- - **RAG Integration**: API-aware retrieval for enhanced code generation -->
- **Visual Reward System**: Video comparison metrics for animation quality assessment
- **Flexible Model Support**: Compatible with various LLM architectures (Llama, Qwen, Ministral, etc.)

## 📁 Project Structure

```
manim-trainer/
├── main.py                 # Main entry point
├── config.py               # Global configuration
├── src/
│   ├── evaluation/         # Evaluation engines
│   │   ├── code_evaluator.py
│   │   ├── evaluation_engine.py
│   │   ├── manim_evaluator.py
│   │   └── video_comparator.py
│   ├── grpo/               # GRPO/GSPO training utilities
│   │   ├── grpo_utils.py
│   │   └── reward_engine.py
│   ├── inference/          # Inference and generation
│   │   ├── inference_engine.py
│   │   ├── inference_utils.py
│   │   ├── model_config.py
│   │   └── planner_engine.py
│   ├── peft/               # Parameter-efficient fine-tuning
│   │   ├── peft_engine.py
│   │   └── preprocessor.py
│   ├── rag/                # Retrieval-Augmented Generation
│   │   ├── api_inspector.py
│   │   ├── call_extractor.py
│   │   └── rag_engine.py
│   └── utils/              # Utility functions
├── tools/                  # CLI tools
├── data/                   # Training/evaluation datasets
└── output/                 # Model outputs and results
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- [Manim Community Edition](https://docs.manim.community/en/stable/installation.html)

### Setup

1. **Clone the repository - T0DO**:
   ```bash
   git clone https://github.com/SuienS/manim-trainer.git
   cd manim-trainer
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f unsloth_py312_5090.yml
   conda activate unsloth-py312-5090
   ```

3. **Test Manim installation**:
   ```bash
   manim checkhealth
   ```

## 💻 Usage

### Training

Fine-tune an LLM using SFT followed by GRPO/GSPO:

```bash
python main.py grpo-trainer train \
    --train-model "unsloth/Qwen3-8B-unsloth-bnb-4bit" \
    --load-in-4bit \
    --sft-epochs 2 \
    --grpo-epochs 2 \
    --max-seq-length 2048 \
    --prompt-portion 0.2 \
    --lora-rank 8 \
    --per-device-train-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --train-data-path "data/manim_sft_dataset_train_v2.parquet" \
    --test-data-path "data/manim_sft_dataset_test_v2.parquet" \
    --learning-rate 2e-5 \
    --grpo-learning-rate 1e-6 \
    --grpo-num-generations 8 \
    --grpo-num-iterations 4 \
    --grpo-mode "grpo" \
    --grpo-start-temperature 0.9 \
    --suppress-thinking-in-grpo \
    --no-think-tag "/no_think" \
    --text-reward-n-workers 1 \
    --video-reward-n-workers 8 \
    --model-loader-type "auto" \
    --random-state 1230 \
    --output-dir "output/trained_models_v2" \
    --model-list-file "output/trained_models_v2/trained_model_list.txt" 
```

#### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train-model` | Base model identifier | Required |
| `--load-in-4bit` | Enable 4-bit quantization | `False` |
| `--sft-epochs` | Number of SFT epochs | `2` |
| `--grpo-epochs` | Number of GRPO epochs | `2` |
| `--lora-rank` | LoRA rank | `8` |
| `--grpo-mode` | Training mode (`grpo` or `gspo`) | `grpo` |
| `--max-seq-length` | Maximum sequence length | `2048` |

### Evaluation

Evaluate a trained model on Manim code generation:

```bash
python main.py manim-llm-evaluator evaluate \
    --evaluation-mode 'sft_grpo' \
    --selected-model 'unsloth/Qwen3-8B-unsloth-bnb-4bit' \
    --peft-model-path 'path/to/peft/model' \
    --dataset-path './data/manim_sft_dataset_v2.parquet' \
    --output-dir './output/eval_results' \
    --max-new-tokens 16384 \
    --load-in-4bit \
    --batch-size 1 \
    --timeout-per-eval-sample-sec 1800 \
    --manim-render-timeout-sec 300 \
    --prompt-mode 'chat' \
    --device-map 'auto' \
    --feedback-rounds 1 \
    --evaluation-list-file './output/eval_results/evaluation_list_v3.txt' \
    --watch-variable 'manim_render_success'
```

#### Evaluation Modes

| Mode | Description |
|------|-------------|
| `base` | Evaluate base model without fine-tuning |
| `sft` | Evaluate SFT-trained model |
| `sft_grpo` | Evaluate SFT + GRPO trained model |
<!-- | `sft_grpo_rag_fb` | Full pipeline with RAG and feedback | -->

## 📊 Dataset

The training and evaluation datasets are available on Hugging Face:

| Dataset | Description | Link |
|---------|-------------|------|
| **ManimBench v1 Dataset** | Curated dataset of natural language descriptions paired with Manim code | [🤗 Hugging Face](https://huggingface.co/datasets/SuienR/ManimBench-v1) |

### Dataset Format

The dataset is provided in Parquet format with the following columns:

| Column Name             | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `Generated Description` | A natural language description automatically generated by a language model |
| `Reviewed Description`  | A human-refined version of the generated description                        |
| `Code`                  | The corresponding Manim code snippet                                        |
| `Type`                  | Complexity level of the animation: `Basic`, `Intermediate`, or `Advanced`  |
| `Split`                 | Dataset split: `train` or `test`                                           |

#### Loading the dataset with HuggingFace Datasets

```python
from datasets import load_dataset
dataset = load_dataset("SuienR/ManimBench-v1", split="train")

# Top 5 samples
for sample in dataset.select(range(5)):
    print(sample["Generated Description"])
    print(sample["Code"])
```

#### Loading the dataset with Pandas

```python
import pandas as pd

splits = {'train': 'manim_sft_dataset_train.parquet', 'test': 'manim_sft_dataset_train.parquet', 'all': 'manim_sft_dataset_all.parquet'}
df = pd.read_parquet("hf://datasets/SuienR/ManimBench-v1/" + splits["train"])

# Top 5 samples
for index, row in dataset.head().iterrows():
    print(row["Generated Description"])
    print(row["Code"])
```

<!-- ## 🤖 Models - TODO

Fine-tuned models are available on Hugging Face:

| Model | Base | Training | Link |
|-------|------|----------|------|
| **ManimFineTune-Qwen3-8B** | Qwen3-8B | SFT + GSPO | [🤗 Hugging Face](todo) | -->

## ⚙️ Configuration
<!--
class Config:
    CACHE_PATH = "cache"
    EVAL_TEMP_DIR = "tmp/eval"
    MANIM_VERSION = "Manim Community v0.19.0"

class SupportedModels:
    """Supported models for the LLM."""
    QWEN3_8B_BASE = "Qwen/Qwen3-8B-Base"
    QWEN3_8B_UNSLOTH_4bit = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    QWEN3_4B_BASE = "Qwen/Qwen3-4B-Base"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    CODE_EVALUATOR_MODEL = "microsoft/codebert-base"
    VIDEO_COMPARATOR_EMBEDDING_CLIP_MODEL = "ViT-L/14" 
 -->
Global configuration parameters can be adjusted in `config.py`. Key parameters include:
- `CACHE_PATH`: Directory for caching models and datasets
- `EVAL_TEMP_DIR`: Temporary directory for evaluation artifacts
- `MANIM_VERSION`: Version of Manim Community Edition used
- `CODE_EVALUATOR_MODEL`: Model used for code evaluation
- `VIDEO_COMPARATOR_EMBEDDING_CLIP_MODEL`: Model used for video embedding comparison

<!-- ## 📖 Citation

TODO: 
``` -->

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) for the animation library
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co/) for transformer models and backbone training frameworks


## ✍️ Authors
- [Ravidu Silva](mailto:ravidus.ac@gmail.com)
- [Jordan J. Bird](mailto:jordan.bird@ntu.ac.uk)
---

**Note**: This repository is part of ongoing research.
For questions or collaboration inquiries, please contact [ravidus.ac@gmail.com](mailto:ravidus.ac@gmail.com).
