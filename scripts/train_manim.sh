#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/data/hf-cache"
export TRANSFORMERS_CACHE="/data/hf-cache"

cd /data/manim-trainer

python manim_trainer.py grpo-trainer train \
  --train-model "Qwen/Qwen3.6-27B" \
  --load-in-4bit \
  --sft-epochs 3 \
  --grpo-epochs 1 \
  --max-seq-length 4096 \
  --prompt-portion 0.2 \
  --lora-rank 64 \
  --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --train-data-path "data/manim_sft_dataset_train_v2.parquet" \
  --test-data-path "data/manim_sft_dataset_test_v2.parquet" \
  --learning-rate 2e-6 \
  --grpo-learning-rate 5e-7 \
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
  --output-dir "/data/output/qwen3.6_27b_manim_sft_v2" \
  --model-list-file "/data/output/qwen3.6_27b_manim_sft_v2/trained_model_list.txt" \
 
