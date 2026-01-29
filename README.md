# ManimTrainer: Fine-Tuning LLMs for Manim Code Generation using Visually Grounded Reinforcement Learning

[![License: CC-BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Manim Community](https://img.shields.io/badge/Manim-Community-green)](https://www.manim.community/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/SuienR/ManimBench-v1)
[![Unsloth](https://img.shields.io/badge/%F0%9F%A6%A5%20Unsloth-Accelerated-orange)](https://github.com/unslothai/unsloth)

A toolkit for fine-tuning Large Language Models (LLMs) to generate [Manim](https://www.manim.community/) animation code using Supervised Fine-Tuning (SFT) and Visually Grounded Reinforcement Learning using Group Relative Policy Optimisation (GRPO/GSPO) techniques.

**📝 Research Paper: [Coming Soon]()**

**🚧 Note:** This project is still in development. Some features may not yet be fully implemented or tested.

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
├── manim_trainer.py                 # Main entry point
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
- CUDA-compatible GPU (recommended: 24GB+ VRAM, Tested with NVIDIA RTX 5090 32GB)
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

**🚧 Note:** Additional dependencies may be required based on the available hardware and OS type. Current setup has been tested on NVIDIA RTX 5090 with 32GB VRAM.


## 💻 Usage

### Training

Fine-tune an LLM using SFT followed by GRPO/GSPO:

```bash
python manim_trainer.py grpo-trainer train \
    --train-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --load-in-4bit \
    --sft-epochs 1 \
    --grpo-epochs 1 \
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
| `--sft-epochs` | Number of SFT epochs | `1` |
| `--grpo-epochs` | Number of GRPO epochs | `1` |
| `--lora-rank` | LoRA rank | `8` |
| `--grpo-mode` | Training mode (`grpo` or `gspo`) | `grpo` |
| `--max-seq-length` | Maximum sequence length | `2048` |

#### Training Monitoring
Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir 'output/trained_models_v2/model_folder/logs'
```

### Evaluation

Evaluate a trained model on Manim code generation:

```bash
python manim_trainer.py manim-llm-evaluator evaluate \
    --evaluation-mode 'sft_grpo' \
    --selected-model 'unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit' \
    --peft-model-path 'output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final' \
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

### Inference

Generate Manim animations using a fine-tuned model:

```bash
python manim_trainer.py inference run_inference \
    --selected-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --peft-model-path "./output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final" \
    --load-in-4bit \
    --input-prompt "Create a Manim animation that displays the  message: Welcome to the ManimTrainer repository."
```

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

## 🤖 Models
⏳ **TODO**: Models will be made available upon publication of the research paper.

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

## 📖 Citation
You can cite ManimTrainer repo as follows:

```bibtex
@software{manimtrainer2025,
  author = {Ravidu Suien Rammuni Silva and Jordan J. Bird},
  title = {ManimTrainer},
  url = {https://github.com/SuienS/manim-trainer},
  year = {2025}
}
```
**📝 Research Paper: [Coming Soon]()**

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) for the animation library
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co/) for transformer models and backbone training frameworks


## ✍️ Authors
- [Ravidu Silva](https://www.linkedin.com/in/ravidu-silva/)
- [Jordan J. Bird](https://www.linkedin.com/in/jordanbird1/)
---

**Note**: This repository is part of ongoing research.
For questions or collaboration inquiries, please contact the authors.
