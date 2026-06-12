"""train_with_rag_grpo_unsloth.py
This script is used to train a MoE model with RAG (Retrieval-Augmented Generation) using the GRPO/GSPO (Generalized Sequence Policy Optimization) method and the Unsloth framework.

References:
- https://github.com/huggingface/trl/pull/3775/files#diff-124ec84eabfbcc029a0381c2db91ac268253f2afdbcf7103717723542bed073d
- https://huggingface.co/docs/trl/sft_trainer

Example usage:
python tools/train_sft_grpo_unsloth.py \
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

Skip SFT and continue GRPO training:
python tools/train_sft_grpo_unsloth.py \
    --skip-sft \
    --sft-model-path "output/trained_models_v2/qwen3_4b_unsloth_bnb_4bit_lora_r8_sft_2025-01-15_12-30-45_final" \
    ... (other GRPO parameters same as above)
"""
import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer

grpo_trainer_app = typer.Typer(name="grpo_trainer", help="Fine-tune an LLM with LoRA and using GRPO/GSPO via Unsloth.")

import pandas as pd
from datasets import Dataset
from functools import partial
from typing import Optional, Type, Union, Callable

from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from vllm import SamplingParams

import config
from src.utils.prompt_template import PromptTemplate, PromptChatTemplate
from src.utils import utils
from src.evaluation.code_evaluator import CodeEvaluator
from src.evaluation.manim_evaluator import ManimEvaluator
from src.evaluation.video_comparator import VideoComparator
from src.grpo.reward_engine import RewardEngine
from src.grpo.grpo_utils import (
    is_moe,
    preprocess_prompt_completion,
    format_for_sft,
)

RANDOM_STATE = 1230
LORA_RANK = 8
PROMPT_TEMPLATE = PromptTemplate.MANIM_VIDEO_GEN_CHAT_TEMPLATE
REWARD_AGGREGATION_OPTIONS = ["arithmetic", "geo"]

@grpo_trainer_app.command()
def train(
    train_model: str = typer.Option("unsloth/Qwen3-4B-unsloth-bnb-4bit", help="HF model name."),
    skip_sft: bool = typer.Option(False, help="Skip SFT training and load from sft_model_path."),
    sft_model_path: Optional[str] = typer.Option(None, help="Path to pre-trained SFT model (required if skip_sft is True)."),
    sft_epochs: int = typer.Option(1, help="Number of SFT epochs."),
    grpo_epochs: int = typer.Option(1, help="Number of GRPO/GSPO epochs."),
    max_seq_length: int = typer.Option(2048, help="Max sequence length."),
    prompt_portion: float = typer.Option(0.2, help="Portion of the prompt to use."),
    lora_rank: int = typer.Option(LORA_RANK, help="LoRA rank."),
    per_device_train_batch_size: int = typer.Option(8, help="Per device train batch size."),
    gradient_accumulation_steps: int = typer.Option(4, help="Gradient accumulation steps."),
    per_device_train_batch_size_grpo: Optional[int] = typer.Option(None, help="Per device train batch size for GRPO. If not set, uses the same as SFT."),
    gradient_accumulation_steps_grpo: Optional[int] = typer.Option(None, help="Gradient accumulation steps for GRPO. If not set, uses the same as SFT."),
    train_data_path: str = typer.Option("data/manim_sft_dataset_train.parquet", help="Path to training dataset parquet."),
    test_data_path: str = typer.Option("data/manim_sft_dataset_test.parquet", help="Path to test dataset parquet."),
    output_dir: str = typer.Option("output/trained_models", help="Output directory for the model."),
    model_list_file: str = typer.Option(None, help="File to append the trained model name."),
    load_in_4bit: bool = typer.Option(False, help="Use 4-bit quantization."),
    learning_rate: float = typer.Option(2e-6, help="Learning rate."),
    grpo_learning_rate: float = typer.Option(5e-7, help="Learning rate for GRPO."),
    grpo_num_generations: int = typer.Option(8, help="Number of generations for GRPO."),
    grpo_num_iterations: int = typer.Option(4, help="Number of GRPO iterations."),
    grpo_mode: str = typer.Option("grpo", help="GRPO mode. Options: 'gspo' or 'grpo'."),
    grpo_start_temperature: float = typer.Option(0.9, help="Starting temperature for GRPO."),
    model_loader_type: str = typer.Option("auto", help="Model loader type. Options: 'auto', 'fastmodel', 'fastlanguagemodel'."),
    suppress_thinking_in_grpo: bool = typer.Option(False, help="Suppress <think> tags in GRPO generation."),
    no_think_tag: str = typer.Option("/no_think", help="Tag to indicate no thinking."),
    no_system_role: bool = typer.Option(False, help="If set, removes the system role from the prompt."),
    text_reward_only: bool = typer.Option(False, help="Use text reward only."),
    visual_reward_only: bool = typer.Option(False, help="Use visual reward only."),
    text_reward_n_workers: int = typer.Option(1, help="Number of workers for text reward computation."),
    video_reward_n_workers: int = typer.Option(4, help="Number of workers for video reward computation."),
    reward_aggregation: str = typer.Option("arithmetic", help="Reward aggregation method. Options: " + str(REWARD_AGGREGATION_OPTIONS), show_choices=True, autocompletion=lambda: REWARD_AGGREGATION_OPTIONS, case_sensitive=False),
    token: str = typer.Option(None, help="Hugging Face token for gated models."),
    sample: bool = typer.Option(False, help="Run a test sample to test the script."),
    sample_size: int = typer.Option(3, help="Number of samples to run for testing the script."),
    random_state: int = typer.Option(RANDOM_STATE, help="Random state for reproducibility."),
):
    training_loop_start_time = utils.get_timestamp(string_output=False)

    # Set GRPO batch sizes
    if per_device_train_batch_size_grpo is None:
        per_device_train_batch_size_grpo = per_device_train_batch_size
    if gradient_accumulation_steps_grpo is None:
        gradient_accumulation_steps_grpo = gradient_accumulation_steps

    # Print all set parameters
    print("Training parameters:")
    for param, value in locals().items():
        print(f"  {param}: {value} (type: {type(value)})")
    
    train_model_name = f"{train_model.split('/')[-1].replace('-', '_')}_lora_r{str(lora_rank)}"
    os.makedirs(output_dir, exist_ok=True)

    # NOTE
    """
    1. Make sure the routing layer is frozen all the time.
    2. Fine-tune the model with RAG data. (In preparation for GRPO, since GRPO could just learn formatting and not the content.)
    3. Define a reward function based in the evaluation metrics previously used.
    4. Define the GRPO training loop to train the model with RAG data.
       - Use the Unsloth framework to implement the GRPO training loop.
       - Use the FastModel class to load the model.
    5. Further fine-tune the model with GRPO.
    """

    if text_reward_only and visual_reward_only:
        raise ValueError("Both text_reward_only and visual_reward_only cannot be True at the same time.")

    if grpo_mode not in ["gspo", "grpo"]:
        raise ValueError("Invalid grpo_mode. Choose either 'gspo' or 'grpo'.")

    if model_loader_type not in ["auto", "fastmodel", "fastlanguagemodel"]:
        raise ValueError("Invalid model_loader_type. Choose either 'auto', 'fastmodel', or 'fastlanguagemodel'.")
    
    if skip_sft and sft_model_path is None:
        raise ValueError("sft_model_path must be provided when skip_sft is True.")

    # Check if model_list_file exists
    if model_list_file is not None:
        model_list_path = Path(model_list_file)
        if not model_list_path.parent.exists():
            model_list_path.parent.mkdir(parents=True, exist_ok=True)

    is_moe_model = is_moe(train_model)
    print(f"Detected MoE model: {is_moe_model}")

    ModelLoader: Union[Type[FastModel], Type[FastLanguageModel]] = FastModel  # Default
    if model_loader_type == "fastmodel":
        if not is_moe_model:
            print(f"Warning: Model {train_model} is not detected as a MoE model. Proceeding with FastModel may cause issues.")
        print(f"Using FastModel to load the model {train_model}.")
        ModelLoader = FastModel
    elif model_loader_type == "fastlanguagemodel":
        if is_moe_model:
            print(f"Warning: Model {train_model} is detected as a MoE model. Proceeding with FastLanguageModel may cause issues.")
        print(f"Using FastLanguageModel to load the model {train_model}.")
        ModelLoader = FastLanguageModel
    else:  # auto
        if is_moe_model:
            print(f"Model {train_model} is detected as a Mixture of Experts (MoE) model. Using FastModel for loading.")
            ModelLoader = FastModel
        else:
            print(f"Model {train_model} is detected as a standard model. Using FastLanguageModel for loading.")
            ModelLoader = FastLanguageModel

    # Set no_system_role if model is codegemma
    if "codegemma" in train_model.lower():
        no_system_role = True
        print("Detected CodeGemma model. Setting no_system_role to True.")

    print("Loading the Dataset...")
    train_dataset_df = pd.read_parquet(train_data_path)
    test_dataset_df = pd.read_parquet(test_data_path)
    if sample:
        train_dataset_df = train_dataset_df.sample(
            n=sample_size, random_state=random_state)
        test_dataset_df = test_dataset_df.sample(
            n=sample_size, random_state=random_state)
    train_dataset = Dataset.from_pandas(train_dataset_df)
    test_dataset = Dataset.from_pandas(test_dataset_df)

    # Rename columns to handle case sensitivity issues
    column_renames = {
        col: col.strip().replace(" ", "_").lower() for col in train_dataset.column_names
    }
    train_dataset = train_dataset.rename_columns(column_renames)
    test_dataset = test_dataset.rename_columns(column_renames)

    print("Dataset loaded successfully.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test columns: {test_dataset.column_names}")

    print("Preprocessing the Dataset...")
    # Use the updated preprocess_function to preprocess the dataset
    preprocess_prompt_completion_fn = partial(
        preprocess_prompt_completion,
        prompt_template=PROMPT_TEMPLATE,
        x_column_name="reviewed_description",
        y_column_name="code",
        suppress_thinking=suppress_thinking_in_grpo,
        no_think_tag=no_think_tag,
        no_system_role=no_system_role,
    )

    train_dataset = train_dataset.map(
        preprocess_prompt_completion_fn,
        remove_columns=train_dataset.column_names,
        # num_proc=4,
        batched=False,
    )

    test_dataset = test_dataset.map(
        preprocess_prompt_completion_fn,
        remove_columns=test_dataset.column_names,
        # num_proc=4,
        batched=False,
    )

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    if not skip_sft:
        print("Loading model and tokenizer...")

        # USE the FastModel class to load the model since the model is an MoE model
        model, tokenizer = ModelLoader.from_pretrained(
            model_name = train_model,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            full_finetuning = False,
            max_lora_rank = lora_rank,
            # fast_inference=True,
            # gpu_memory_utilization = 0.85,
            token = token,
        )

        # Get PEFT model
        from peft import PeftModel
        model: PeftModel = ModelLoader.get_peft_model(
            model,
            r = lora_rank,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = lora_rank*2,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            max_seq_length = max_seq_length,
            random_state = random_state,
        )
        print("Model and tokenizer loaded successfully.")

        # Format the dataset for SFT training
        format_for_sft_fn = partial(
            format_for_sft,
            tokenizer=tokenizer,
        )

        train_dataset_sft = train_dataset.map(
            format_for_sft_fn,
            remove_columns=train_dataset.column_names,
            # num_proc=4,
            batched=False,
        )

        test_dataset_sft = test_dataset.map(
            format_for_sft_fn,
            remove_columns=test_dataset.column_names,
            # num_proc=4,
            batched=False,
        )

        print("Example preprocessed sample:")
        print(train_dataset_sft[0])

        print("Dataset Columns", train_dataset_sft.column_names)

        print("Dataset preprocessed successfully.")

        print("Starting model training (SFT)...")
        
        sft_timestamp = utils.get_timestamp()

        sft_model_unique_name = f"{train_model_name}_sft_{sft_timestamp}"

        sft_output_dir = os.path.join(output_dir, sft_model_unique_name)
        os.makedirs(sft_output_dir, exist_ok=True)
        sft_logging_dir = os.path.join(sft_output_dir, "logs")
        os.makedirs(sft_logging_dir, exist_ok=True)

        print("SFT Output Directory:", sft_output_dir)
        print("SFT Tensorboard Logging Directory:", sft_logging_dir)

        sft_config = SFTConfig(
            completion_only_loss=True,
            dataset_text_field="messages",
            num_train_epochs=sft_epochs,
            warmup_ratio=0.1,
            learning_rate=learning_rate,
            lr_scheduler_type = "linear",
            per_device_train_batch_size=per_device_train_batch_size,
            # per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            logging_steps=gradient_accumulation_steps,

            eval_strategy="steps",
            eval_steps=gradient_accumulation_steps,

            save_strategy="steps",
            save_steps=gradient_accumulation_steps,
            save_total_limit=1,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            optim = "adamw_8bit",
            weight_decay = 0.01,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),

            output_dir=sft_output_dir,
            logging_dir=sft_logging_dir,
            # TODO: ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            seed=random_state,
            report_to="tensorboard",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset_sft,
            eval_dataset=test_dataset_sft,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=sft_config,
        )
        trainer.train()

        print("Model training (SFT) completed.")
        model_save_name = f"{sft_model_unique_name}_final"
        model_save_path = os.path.join(output_dir, model_save_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path) # Ensure tokenizer is saved
        with open(model_list_path, 'a') as model_list_f:
            model_list_f.write(f"{model_save_path}\n")
        print("SFT Model saved:", model_save_path)

        # ------------------------------------------------------------------
        # FIX: Reload model to clear SFT state and ensure clean GRPO start
        # ------------------------------------------------------------------
        print("Cleaning up SFT resources...")
        del trainer, model, tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    else:
        model_save_path = sft_model_path
        sft_timestamp = utils.get_timestamp()

        print(f"Skipping SFT training. Using pre-trained SFT model from {model_save_path}...")


    print(f"Reloading model from {model_save_path} for {grpo_mode.upper()} training...")
    # Reload the model with the trained SFT adapters
    model, tokenizer = ModelLoader.from_pretrained(
        model_name = model_save_path, # Load the just-trained adapters
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        max_lora_rank = lora_rank,
        token = token,
    )
    
    # Enable Unsloth training optimizations (IMP for TRL compatibility)
    ModelLoader.for_training(model)

    print("========== Starting GRPO/GSPO model training ============")
    reward_engine = RewardEngine(
        text_reward_n_workers=text_reward_n_workers if not visual_reward_only else 0,
        video_reward_n_workers=video_reward_n_workers if not text_reward_only else 0,
        video_fps=5,
        visual_reward_weight=0.8,
        text_reward_weight=0.2,
        clip_model_device="cuda" if torch.cuda.is_available() else "cpu",
        manim_eval_timeout_sec=300,
    )

    max_prompt_length = int(max_seq_length * prompt_portion)
    max_completion_length = int(max_seq_length - max_prompt_length)

    vllm_sampling_params = SamplingParams(
        top_k = -1,
        min_p = 0.1,
        top_p = 0.9,
        temperature = grpo_start_temperature,
        max_tokens=max_completion_length,
        include_stop_str_in_output = True,
        stop = [tokenizer.eos_token],
        seed = random_state
    )

    grpo_generation_kwargs = {
        "do_sample": True,
        "min_p": 0.1,
        "top_p": 0.9,  # Nucleus sampling
        "temperature": grpo_start_temperature,
        "max_length": max_seq_length,
        "max_new_tokens": max_completion_length,
        "pad_token_id": getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
        "eos_token_id": tokenizer.eos_token_id,
    }

    #TODO: Adjust evals_steps and save_steps based on dataset size and batch size
    
    reward_label = ""

    reward_func_list: Union[list, Callable] = []

    if visual_reward_only:
        print("Using Visual Reward only for GRPO/GSPO training.")
        reward_func_list.append(reward_engine.get_video_similarity_reward_parallel)
        reward_label += "_visual"
    elif text_reward_only:
        print("Using Text Reward only for GRPO/GSPO training.")
        reward_func_list.append(reward_engine.get_codebertbleu_score_reward_parallel)
        reward_label += "_text"
    else:
        print(f"Using unified Text and Visual Rewards with {reward_aggregation} aggregation for GRPO/GSPO training.")
        if reward_aggregation == "arithmetic":
            reward_func_list.append(reward_engine.get_arithmetic_weighted_mean_unified_reward)
            reward_label += "_mean_text_visual"
        elif reward_aggregation == "geo":
            reward_func_list.append(reward_engine.get_geo_mean_unified_reward)
            reward_label += "_geomean_text_visual"
        else:
            raise ValueError(f"Invalid reward aggregation method: {reward_aggregation}")

    grpo_model_unique_name = f"{train_model_name}_sft_{grpo_mode}_rw{reward_label}_{sft_timestamp}"

    grpo_output_dir = os.path.join(output_dir, grpo_model_unique_name)
    os.makedirs(grpo_output_dir, exist_ok=True)
    grpo_logging_dir = os.path.join(grpo_output_dir, "logs")
    os.makedirs(grpo_logging_dir, exist_ok=True)

    print("GRPO/GSPO Output Directory:", grpo_output_dir)
    print("GRPO/GSPO Tensorboard Logging Directory:", grpo_logging_dir)

    num_steps_per_epoch = len(train_dataset) // (per_device_train_batch_size_grpo * gpu_count * gradient_accumulation_steps_grpo)

    if grpo_mode == "gspo":
        print("Using GSPO mode for GRPO training.")
        # grpo_trainer_config.steps_per_generation=gradient_accumulation_steps*4  # partition rollout batch into 4 mini-batches. GSPO paper (v2), section 5.1. Must be 4 times gradient_accumulation_steps
    else:
        print("Using GRPO mode for GRPO training.")


    grpo_trainer_config = GRPOConfig(
        num_iterations=grpo_num_iterations, # Number of GRPO iterations

        num_train_epochs = grpo_epochs,
        per_device_train_batch_size = per_device_train_batch_size_grpo,
        per_device_eval_batch_size = per_device_train_batch_size_grpo,
        gradient_accumulation_steps = gradient_accumulation_steps_grpo,

        # GRPO/GSPO specific parameters
        # GSPO Trainer configuration: 
        # - https://github.com/huggingface/trl/pull/3775/files#diff-124ec84eabfbcc029a0381c2db91ac268253f2afdbcf7103717723542bed073d
        # - https://huggingface.co/docs/trl/en/paper_index
        importance_sampling_level = "sequence" if grpo_mode == "gspo" else "token",

        # GSPO specific. Only when used with importance_sampling_level="sequence" and loss_type="dr_grpo"
        # Because we are using DR-GSPO, we do not need to mask truncated completions.
        # It explicitly wants to keep the reward signal over all generated tokens, even if they’re truncated,
        # to stabilize the importance-sampling reweighting process. 
        # Treats truncation as part of the trajectory and not something to mask out.
        mask_truncated_completions = False,
        loss_type = "dr_grpo",

        beta=0.0 if grpo_mode == "gspo" else 0.005,  # GSPO set KL regularization to zero: https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306 
        epsilon=3e-4 if grpo_mode == "gspo" else 0.2,  # GSPO paper (v2), section 5.1
        epsilon_high=4e-4 if grpo_mode == "gspo" else None,  # GSPO paper (v2), section 5.1

        num_generations=grpo_num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,

        temperature =  grpo_start_temperature,
        learning_rate = grpo_learning_rate, # Lower learning rate for GRPO
        lr_scheduler_type = "linear",
        warmup_ratio = 0.1,

        logging_steps = gradient_accumulation_steps,

        eval_strategy = "no", # No evaluation during GRPO/GSPO training

        save_strategy = "epoch",

        save_total_limit = grpo_epochs,  # Keep last n epoch models

        # load_best_model_at_end = True,
        # metric_for_best_model = "rewards/manim_pass_reward_func/mean",
        # greater_is_better = True,

        # metric_for_best_model = "eval_loss",
        # greater_is_better = False,

        # TODO: use bf16 if posible
        fp16 = True #not torch.cuda.is_bf16_supported(),
        bf16 = False #torch.cuda.is_bf16_supported(),

        optim = "adamw_8bit",
        weight_decay = 0.01,

        output_dir = grpo_output_dir,
        logging_dir = grpo_logging_dir,

        seed = random_state,
        report_to = "tensorboard",

        # vLLM specific parameters
        # vllm_sampling_params = vllm_sampling_params,
        # use_vllm = True,

        # non vLLM mode
        use_vllm = False,
        generation_kwargs = grpo_generation_kwargs,
    )

    print(f"Sampling Level: {grpo_trainer_config.importance_sampling_level}")

    grpo_trainer = GRPOTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        processing_class = tokenizer,
        reward_funcs = reward_func_list,
        args = grpo_trainer_config,
    )

    grpo_trainer.train()

    # Save the final model (after GRPO/GSPO training)
    model_save_name = f"{grpo_model_unique_name}_final"
    model_save_path = os.path.join(output_dir, model_save_name)
    model.save_pretrained(model_save_path)

    with open(model_list_path, 'a') as model_list_f:
        model_list_f.write(f"{model_save_path}\n")

    print(f"SFT {grpo_mode.upper()} Model saved: {model_save_path}")

    print("========== GRPO/GSPO model training completed ============")

    training_loop_end_time = utils.get_timestamp(string_output=False)
    print(f"Total training time: {training_loop_end_time - training_loop_start_time}")


if __name__ == "__main__":
    grpo_trainer_app()