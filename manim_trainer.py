#!/usr/bin/env python

"""manim_trainer.py: main entry point for the Manim LLM project.

Example usage - Training:
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
    --model-list-file "output/trained_models_v2/trained_model_list.txt" \
    --sample    --sample-size 32

Example usage - Evaluation:
python manim_trainer.py manim-llm-evaluator evaluate \
    --evaluation-mode 'sft_grpo' \
    --selected-model 'unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit' \
    --peft-model-path './output/trained_models_v2/Qwen3_8B_unsloth_bnb_4bit_lora_r8_sft_gspo_rw_text_visual_20251127_110202_final' \
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
    --watch-variable 'manim_render_success' \
    --limit-samples 5

Example usage - Inference:
python manim_trainer.py inference run_inference \
    --selected-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --peft-model-path "./output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final" \
    --load-in-4bit \
    --input-prompt "Create a Manim animation that displays the  message: Welcome to the ManimTrainer repository."

    
"""

__author__      = "Ravidu Silva"
__email__       = "ravidus.ac@gmail.com"

import typer
main_app = typer.Typer(name="ManimTrainer", help="A toolkit for fine-tuning Large Language Models (LLMs) to generate Manim animation code using Supervised Fine-Tuning (SFT) and Visually Grounded Reinforcement Learning using Group Relative Policy Optimization (GRPO/GSPO) techniques.")

from tools import train_sft_grpo_unsloth, evaluate, inference

main_app.add_typer(train_sft_grpo_unsloth.grpo_trainer_app, name="grpo-trainer", help="Fine-tune an LLM with LoRA and using GRPO/GSPO via Unsloth.")
main_app.add_typer(inference.inference_app, name="inference", help="Perform inference with fine-tuned Manim LLM models.")
main_app.add_typer(evaluate.evaluator_app, name="manim-llm-evaluator", help="Evaluate an LLM model on Manim code generation.")

if __name__ == "__main__":
    main_app()