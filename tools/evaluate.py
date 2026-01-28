"""evaluate.py: Main evaluation script for Manim code generation models.
Runs evaluation on a specified dataset using a given model, with options for RAG and feedback.
Saves results to CSV and logs evaluation paths.

Usage:
python tools/evaluate.py \
    --evaluation-mode 'sft_grpo' \
    --selected-model 'unsloth/Qwen3-8B-unsloth-bnb-4bit' \
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
"""

import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
# Initialize Typer app
evaluator_app = typer.Typer(name="evaluate", help="Evaluate a fine-tuned Manim LLM model.")

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from unsloth import FastLanguageModel, FastModel

from src.inference.inference_engine import InferenceEngine
from src.evaluation.code_evaluator import CodeEvaluator
from src.evaluation.manim_evaluator import ManimEvaluator
from src.evaluation.evaluation_engine import EvaluationEngine
from src.evaluation.video_comparator import VideoComparator
from src.rag.rag_engine import RAGEngine
from src.rag.api_inspector import ApiInspector
from src.rag.call_extractor import CallExtractor
from src.grpo.grpo_utils import is_moe
from src.utils import utils
import config


EVAL_MODES = ['base', 'base_rag', 'base_fb', 'base_rag_fb',
              'sft', 'sft_rag', 'sft_fb', 'sft_rag_fb',
              'sft_grpo', 'sft_grpo_rag', 'sft_grpo_fb', 'sft_grpo_rag_fb',
              'sft_gspo', 'sft_gspo_rag', 'sft_gspo_fb', 'sft_gspo_rag_fb']

SUPPORTED_PROMPT_MODES = ['chat']

TEST_TEXT_SCRIPT = """Set the background color to a light beige. Display a large black double-struck "M" shifted to the left and up. Show a green circle shifted to the left, a blue square shifted up, and a red triangle shifted to the right. Group these shapes and the "M" together, centering the group on the screen."""
TEST_EXPECTED_CODE = """from manim import *

class ManimCELogo(Scene):
    def construct(self):
        self.camera.background_color = "#ece6e2"
        logo_green = "#87c2a5"
        logo_blue = "#525893"
        logo_red = "#e07a5f"
        logo_black = "#343434"
        ds_m = MathTex(r"\\mathbb\\{M\\}", fill_color=logo_black).scale(7)
        ds_m.shift(2.25 * LEFT + 1.5 * UP)
        circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
        square = Square(color=logo_blue, fill_opacity=1).shift(UP)
        triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
        logo = VGroup(triangle, square, circle, ds_m)
        logo.move_to(ORIGIN)
        self.add(logo)
"""

def setup_environment(cache_path_override: str = None):
    """Sets up environment variables and paths."""
    # Load .env file
    load_dotenv()
    
    # Determine cache path
    env_cache_path = os.getenv('CACHE_PATH')
    target_cache_path = cache_path_override if cache_path_override else env_cache_path

    if target_cache_path:
        print(f"Setting cache path to: {target_cache_path}")
        os.environ["HF_HOME"] = f"{target_cache_path}/transformers"
        os.environ["HF_DATASETS_CACHE"] = f"{target_cache_path}/datasets"
        os.environ["TORCH_HOME"] = f"{target_cache_path}/torch"
        os.environ["TFHUB_CACHE_DIR"] = f"{target_cache_path}/tfhub"

@evaluator_app.command()
def evaluate(
    selected_model: str = typer.Option(
        'unsloth/Qwen3-8B-unsloth-bnb-4bit', 
        help="Base model identifier (e.g., unsloth/Qwen3-8B...)"
    ),
    peft_model_path: str = typer.Option(
        None,
        help="Path to the fine-tuned PEFT adapter"
    ),
    evaluation_mode: str = typer.Option(
        'base', 
        help="Evaluation mode. Available modes: " + ", ".join(EVAL_MODES),
        case_sensitive=False, 
        show_choices=True, 
        autocompletion=lambda: EVAL_MODES
    ),
    dataset_path: str = typer.Option(
        './data/manim_sft_dataset_v2.parquet',
        help="Path to the evaluation dataset"
    ),
    output_dir: Path = typer.Option(
        Path("./output/eval_results"),
        help="Directory to save evaluation results"
    ),
    max_new_tokens: int = typer.Option(16384, help="Maximum new tokens for generation"),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4-bit quantization"),
    batch_size: int = typer.Option(1, help="Batch size for evaluation"),
    remove_token_type_ids: bool = typer.Option(False, help="Remove token type IDs for certain models"),
    no_system_role_support: bool = typer.Option(False, help="Disable system role support in prompts"),
    timeout_per_eval_sample_sec: int = typer.Option(60 * 30, help="Timeout per evaluation sample in seconds"),
    manim_render_timeout_sec: int = typer.Option(60 * 5, help="Timeout for Manim rendering in seconds"),
    prompt_mode: str = typer.Option('chat', help="Prompt mode (chat, text, etc.)"),
    device_map: str = typer.Option("auto", help="Device mapping (auto, cuda:0, etc.)"),
    feedback_rounds: int = typer.Option(0, help="Number of feedback rounds"),
    limit_samples: int = typer.Option(-1, help="Limit the number of samples to evaluate (for testing)"),
    evaluation_list_file: str = typer.Option("./output/eval_results/evaluation_list_v3.txt", help="File to log evaluation result paths"),
    watch_variable: str = typer.Option("manim_render_success", help="Variable to watch and display in progress")
):
    """
    Runs the Manim evaluation pipeline using a specified model and dataset.
    """
    print("Starting evaluation script...")
    eval_start_time = utils.get_timestamp(string_output=False)
    # Print all set parameters
    print("Evaluation parameters:")
    for param, value in locals().items():
        print(f"  {param}: {value} (type: {type(value)})")

    setup_environment()

    # Configuration Logic
    selected_model_name_str = selected_model.replace('/', '_').replace('-', '_').replace('.', '_')

    evaluation_mode = evaluation_mode.lower()
    if evaluation_mode not in EVAL_MODES:
        raise ValueError(f"Invalid evaluation mode: {evaluation_mode}. Must be one of {EVAL_MODES}.")
    
    rag_enabled = "_rag" in evaluation_mode
    fb_enabled = "_fb" in evaluation_mode

    if "sft_gspo" in evaluation_mode:
        if not peft_model_path:
            raise ValueError("PEFT model path must be provided for SFT GSPO evaluation modes.")
        else:
            if not "sft_gspo" in peft_model_path.lower():
                raise ValueError("PEFT model path does not appear to be an SFT GSPO model.")
    elif "sft_grpo" in evaluation_mode:
        if not peft_model_path:
            raise ValueError("PEFT model path must be provided for SFT GRPO evaluation modes.")
        else:
            if not "sft_grpo" in peft_model_path.lower():
                raise ValueError("PEFT model path does not appear to be an SFT GRPO model.")
    elif "sft" in evaluation_mode:
        if not peft_model_path:
            raise ValueError("PEFT model path must be provided for SFT evaluation modes.")
        else:
            if not "sft" in peft_model_path.lower():
                raise ValueError("PEFT model path does not appear to be an SFT model.")
            if "grpo" in peft_model_path.lower() or "gspo" in peft_model_path.lower():
                raise ValueError("PEFT model path appears to be a GRPO/GSPO model, but evaluation mode is SFT.")
    elif "base" in evaluation_mode:
        if peft_model_path:
            print("WARNING: PEFT model path provided but evaluation mode is 'base'. The PEFT model will be ignored.")
            peft_model_path = None
    else:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")
    
    if fb_enabled:
        if feedback_rounds < 1 or feedback_rounds > 10:
            raise ValueError("Feedback rounds must be between 1 and 10.")
    else:
        print("Feedback loop disabled. Ignoring feedback rounds setting.")
        feedback_rounds = 0

    print(f"RAG Enabled: {rag_enabled}, Feedback Enabled: {fb_enabled}")
        
    if prompt_mode not in SUPPORTED_PROMPT_MODES:
        raise ValueError(f"Invalid prompt mode: {prompt_mode}. Must be one of {SUPPORTED_PROMPT_MODES}.")
    
    # Check if evaluation list file exists, create if not
    eval_list_path = Path(evaluation_list_file)
    if not eval_list_path.exists():
        eval_list_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Adjust max tokens and device map based on model capabilities
    if "codegemma" in selected_model.lower():
        max_new_tokens = 8192
        device_map = "cuda:0" if device_map == "auto" else device_map
        no_system_role_support = True
        print(f"CodeGemma model detected: setting max_new_tokens to {max_new_tokens}, device_map to {device_map}, no_system_role_support to {no_system_role_support}")

    if "seed-coder" in selected_model.lower():
        remove_token_type_ids = True
        print(f"SeedCoder model detected: setting remove_token_type_ids to {remove_token_type_ids}")

    # Initialize Engines
    rag_engine = None
    if rag_enabled:
        print("Initializing RAG Engine...")
        import manim
        api_inspector = ApiInspector(manim)
        call_extractor = CallExtractor()
        rag_engine = RAGEngine(
            api_inspector=api_inspector,
            call_extractor=call_extractor
        )

    print("Initializing Inference Engine...")
    is_moe_model = is_moe(selected_model)
    print(f"Is MOE Model: {is_moe_model}")
    print("Using Model Loader:", "FastModel" if is_moe_model else "FastLanguageModel")
    ModelLoader = FastLanguageModel if not is_moe_model else FastModel
    inference_engine = None

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
            rag_engine=rag_engine
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
            rag_engine=rag_engine
        )

    # Initialize Evaluators
    print("Initializing Evaluators...")
    embedding_clip_model = "ViT-L/14"
    video_comp_fps = 5
    video_comp_device = "cuda:0"

    code_evaluator = CodeEvaluator(config.SupportedModels.CODE_EVALUATOR_MODEL)
    manim_evaluator = ManimEvaluator(Path(config.Config.EVAL_TEMP_DIR), timeout=manim_render_timeout_sec)
    video_comparator = VideoComparator(
        embedding_clip_model=embedding_clip_model,
        fps=video_comp_fps,
        device=video_comp_device
    )

    evaluation_engine = EvaluationEngine(
        inference_engine=inference_engine,
        code_evaluator=code_evaluator,
        manim_evaluator=manim_evaluator,
        video_comparator=video_comparator,
    )
    print("Evaluators initialized.")

    print(f"Inference Engine Model Type: {inference_engine.get_inference_model_type()}")

    # Run Test Evaluation
    result_sample = evaluation_engine.evaluate_sample(
        textual_script=TEST_TEXT_SCRIPT,
        expected_code=TEST_EXPECTED_CODE,
        feedback_round=fb_enabled,
        feedback_round_count=feedback_rounds,
        forced_feedback_round=False,
        generation_timeout_sec=timeout_per_eval_sample_sec
    )
    print("\n--- Test Evaluation Result ---")
    print(result_sample)
    print("--- End of Test Evaluation ---\n")

    # Load Data
    print(f"Loading dataset from {dataset_path}...")
    test_set = pd.read_parquet(dataset_path)
    test_set = test_set[test_set['Split'] == 'test']
    
    if limit_samples > 0:
        print(f"Limiting evaluation to first {limit_samples} samples.")
        test_set = test_set.head(limit_samples)

    test_set_x_y = test_set[['Reviewed Description', 'Code']]
    descriptions = list(test_set_x_y['Reviewed Description'])
    expected_codes = list(test_set_x_y['Code'])
    
    print(f"Starting evaluation on {len(descriptions)} samples...")

    # Run Evaluation
    result_batch = evaluation_engine.run_evaluation(
        batch_size=batch_size,
        textual_scripts=descriptions,
        expected_codes=expected_codes,
        feedback_round=fb_enabled,
        feedback_round_count=feedback_rounds,
        forced_feedback_round=False,
        generation_timeout_sec=timeout_per_eval_sample_sec,
        watch_variable=watch_variable
    )

    # Process and Save Results
    df_result_sample = pd.DataFrame(result_batch)
    
    # Calculate stats
    numeric_columns = [
        'visual_similarity', 'codebertbleu', 'manim_render_success', 'syntax_error',
        'visual_semantic_score', 'ssim_score', 'codebert_similarity', 'codebleu',
        'ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score',
        'ast_distance_norm', 'ast_distance_raw', 'ast_distance_max'
    ]
    
    print("\n--- Results Summary ---")
    try:
        print(df_result_sample[numeric_columns].mean())
    except KeyError:
        print("Could not calculate means (columns missing).")

    # Save
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = ""
    if not peft_model_path:
        file_name = f"eval_result_{evaluation_mode}_{selected_model_name_str}_{timestamp}.csv"
    else:
        peft_model_name_str = Path(peft_model_path).name
        # Example: Qwen3_8B_unsloth_bnb_4bit_lora_r8_sft_gspo_rw_text_visual_20251127_110202_final
        if 'checkpoint' in peft_model_name_str:
            print("PEFT model path appears to be a checkpoint. Using parent folder name for PEFT model identifier.")
            # Example: Qwen3_14B_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_135811/checkpoint-632
            # Get previous part folder name
            peft_model_name_str = Path(peft_model_path).parent.name + f"_{peft_model_name_str}"
        peft_model_name_str = peft_model_name_str.split('_lora_')[-1]
        file_name = f"eval_result_{evaluation_mode}_{selected_model_name_str}_peft_{peft_model_name_str}_{timestamp}.csv"
    output_path = output_dir / file_name
    
    df_result_sample.to_csv(output_path)
    print(f"\nEvaluation results saved to {output_path}")
    
    with open(eval_list_path, 'a') as eval_list_f:
        eval_list_f.write(f"{output_path}\n")
    
    eval_end_time = utils.get_timestamp(string_output=False)
    total_eval_duration = eval_end_time - eval_start_time
    print(f"Total evaluation time: {total_eval_duration}")

if __name__ == "__main__":
    evaluator_app()