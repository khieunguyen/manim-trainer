"""inference.py: Tool for performing inference with fine-tuned Manim LLM models.

Example usage:
python tools/inference.py \
    --selected-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --peft-model-path "./output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final" \
    --load-in-4bit \
    --video-output-path "." \
    --input-prompt "Create a Manim animation that displays the  message: Welcome to the ManimTrainer repository."

Example advanced usage:
python tools/inference.py \
    --selected-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --peft-model-path "./output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final" \
    --load-in-4bit \
    --prompt-mode "chat" \
    --device-map "auto" \
    --max-new-tokens 2048 \
    --video-output-path "./output/inference_results/generated_video.mp4" \
    --timeout 600 \
    --input-prompt "Create a Manim animation that displays the  message: Welcome to the ManimTrainer repository."
"""

import os
from pathlib import Path
import shutil
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
# Initialize Typer app
inference_app = typer.Typer(name="inference", help="Tool for performing inference with fine-tuned Manim LLM models.")

from unsloth import FastLanguageModel, FastModel

from src.inference.inference_engine import InferenceEngine
from src.evaluation.manim_evaluator import ManimEvaluator
from src.grpo.grpo_utils import is_moe
from src.utils import utils


@inference_app.command("run_inference")
def run_inference(
    selected_model: str = typer.Option(..., help="Path to the fine-tuned Manim LLM model."),
    peft_model_path: str = typer.Option(None, help="Path to the PEFT model if applicable."),
    load_in_4bit: bool = typer.Option(False, help="Whether to load the model in 4-bit precision."),
    prompt_mode: str = typer.Option("chat", help="Prompt mode: 'chat' or 'direct'."),
    device_map: str = typer.Option("auto", help="Device map for model loading."),
    input_prompt: str = typer.Option(..., help="Input prompt for the model."),
    max_new_tokens: int = typer.Option(16384, help="Maximum number of new tokens to generate."),
    video_output_path: str = typer.Option(".", help="Path to save the generated video."),
    timeout: int = typer.Option(300, help="Timeout for Manim rendering in seconds.")
):
    """
    Perform inference with a fine-tuned Manim LLM model and generate a Manim script video.
    """

    remove_token_type_ids = False
    no_system_role_support = False
    
    if "codegemma" in selected_model.lower():
        max_new_tokens = 8192
        device_map = "cuda:0" if device_map == "auto" else device_map
        no_system_role_support = True
        print(f"CodeGemma model detected: setting max_new_tokens to {max_new_tokens}, device_map to {device_map}, no_system_role_support to {no_system_role_support}")

    if "seed-coder" in selected_model.lower():
        remove_token_type_ids = True
        print(f"SeedCoder model detected: setting remove_token_type_ids to {remove_token_type_ids}")
    
    is_moe_model = is_moe(selected_model)
    ModelLoader = FastLanguageModel if not is_moe_model else FastModel
    inference_engine = None


    print("Loading model for inference...")
    if peft_model_path:
        print("Loading PEFT Model...")

        saved_peft_model, tokenizer = ModelLoader.from_pretrained(
            model_name=peft_model_path,
            max_seq_length=max_new_tokens,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        ModelLoader.for_inference(saved_peft_model)

        inference_engine = InferenceEngine(
            hf_model_name=selected_model,
            backend='preloaded_peft',
            loaded_base_model=None,
            loaded_peft_model=saved_peft_model,
            loaded_tokenizer=tokenizer,
            device_map=device_map,
            remove_token_type_ids=remove_token_type_ids,
            no_system_role_support=no_system_role_support,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            prompt_mode=prompt_mode,
            # rag_engine=rag_engine TODO
        )

    else:
        print("Using base model without PEFT.")

        loaded_base_model, tokenizer = ModelLoader.from_pretrained(
            model_name=selected_model,
            max_seq_length=max_new_tokens,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        ModelLoader.for_inference(loaded_base_model)

        inference_engine = InferenceEngine(
            hf_model_name=None,
            loaded_base_model=loaded_base_model,
            loaded_tokenizer=tokenizer,
            backend='preloaded',
            device_map=device_map,
            remove_token_type_ids=remove_token_type_ids,
            no_system_role_support=no_system_role_support,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            prompt_mode=prompt_mode,
            # rag_engine=rag_engine TODO
        )

    print("Model loaded. Generating Manim script...")
    model_response = inference_engine.generate_manim_code(input_prompt)

    print("Model response received. Generating video...")
    manim_evaluator = ManimEvaluator(
        timeout=timeout
    )

    generated_manim_code = utils.extract_manim_code_from_llm_response(model_response.generated_code)

    if not generated_manim_code.strip():
        print("No Manim code generated by the model.")
        return

    render_result = manim_evaluator.evaluate_code(generated_manim_code, clear_output=False)

    if render_result.success:
        # Move the generated video to the specified output path
        generated_video_path = render_result.video_path
        final_video_path = Path(video_output_path)
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(generated_video_path, final_video_path)
        print(f"Video successfully generated at: {final_video_path}")
    else:
        print("Video generation failed.")
        print(f"Info: {render_result.info}")
        print(f"Errors: {render_result.errors}")


    # TODO Test this script with various models and prompts

if __name__ == "__main__":
    inference_app()


