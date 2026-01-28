"""evaluation_engine.py
Evaluation engine for the Manim LLM project.
"""
__author__      = "Ravidu Silva"

from tqdm import tqdm
from functools import partial
import concurrent.futures

from src.inference.inference_engine import InferenceEngine, ModelResponse
from src.evaluation.code_evaluator import CodeEvaluator
from src.evaluation.manim_evaluator import ManimEvaluator
from src.evaluation.video_comparator import VideoComparator
from src.utils import utils

class EvaluationEngine:
    """Evaluation engine for the Manim LLM project."""


    def __init__(self, inference_engine: InferenceEngine, code_evaluator: CodeEvaluator, manim_evaluator: ManimEvaluator, video_comparator: VideoComparator):
        """
        Initialize the evaluation engine with the specified inference engine.

        Args:
            inference_engine (InferenceEngine): The inference engine to use for evaluation.
            code_evaluator (CodeEvaluator): The code evaluator to use for evaluation.
            manim_evaluator (ManimEvaluator): The Manim evaluator to use for evaluation.
        """

        self.inference_engine = inference_engine
        self.code_evaluator = code_evaluator
        self.manim_evaluator = manim_evaluator
        self.video_comparator = video_comparator

        # Set the model to evaluation mode
        self.inference_engine.set_eval_mode()

        print(f"Evaluation engine initialized with model: {self.inference_engine.hf_model_name}")


    def evaluate_sample(
            self, 
            textual_script: str, 
            expected_code: str, 
            prepared_code_plan: str = None, 
            feedback_round:bool = False, 
            feedback_round_count:int = 1, 
            forced_feedback_round:bool = False,
            generation_timeout_sec: int = 1800
        ) -> dict:
        """
        Evaluate a single sample of textual script against expected code.

        Args:
            textual_script (str): The textual script to evaluate.
            expected_code (str): The expected ManimCE code.
            prepared_code_plan (str): Pre-generated thinking content to assist in code generation.
            feedback_round (bool): Whether to enable feedback loop for evaluation.
            feedback_round_count (int): The number of feedback loop iterations to perform.
            generation_timeout_sec (int): Timeout in seconds for code generation.

        Returns:
            dict: A dictionary containing the evaluation results.
        """

        # Generate ManimCE code from the textual script
        raw_generated_content = None
        generation_timed_out = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if feedback_round:
                generate_manim_code_exec = executor.submit(
                    self.inference_engine.generate_manim_code_with_feedback,
                    textual_script=textual_script,
                    manim_evaluator=self.manim_evaluator,
                    forced_feedback_round=forced_feedback_round,
                    prepared_code_plan=prepared_code_plan,
                    feedback_round_count=feedback_round_count
                )
            else:
                generate_manim_code_exec = executor.submit(
                    self.inference_engine.generate_manim_code,
                    textual_script=textual_script,
                    prepared_code_plan=prepared_code_plan
                )
            try:
                raw_generated_content = generate_manim_code_exec.result(timeout=generation_timeout_sec)
            except concurrent.futures.TimeoutError:
                generation_timed_out = True
                print(f"WARN: Code generation timed out after {generation_timeout_sec} seconds.")
                # Return a default/error result
                raw_generated_content = ModelResponse(
                    generated_code="ERROR: Generation Timed Out",
                    thinking_content="",
                    code_plan="",
                    rag_info="",
                    initial_code=""
                )

        raw_generated_code = raw_generated_content.generated_code
        raw_generated_thinking = raw_generated_content.thinking_content
        raw_generated_code_plan = raw_generated_content.code_plan
        raw_rag_info = raw_generated_content.rag_info
        raw_initial_code = raw_generated_content.initial_code

        # Extract the generated ManimCE code from the response
        generated_code = utils.extract_manim_code_from_llm_response(raw_generated_code) \
            if not generation_timed_out else ""

        # Evaluate the generated code against the expected code
        evaluation_results = self.code_evaluator.evaluate_code(generated_code, expected_code)

        # Evaluate the generated code using Manim
        manim_result = self.manim_evaluator.evaluate_code(generated_code, clear_output=False)
        generate_video_path = manim_result.video_path

        video_similarity_results = {
            "ssim_score": 0.0,
            "visual_semantic_score": 0.0
        }
        if generated_code.strip() != "" and generate_video_path is not None and generate_video_path.exists():
            # Generate expected video path
            expected_video_result = self.manim_evaluator.evaluate_code(expected_code, clear_output=False)

            if expected_video_result.success is False:
                raise RuntimeError(f"Failed to generate expected video for video comparison. Code: \n{expected_code}\n\nErrors: \n{expected_video_result.errors}")
            
            if expected_video_result.video_path is None:
                raise RuntimeError("Failed to generate expected video for video comparison.")

            expected_video_path = expected_video_result.video_path

            # Compare the generated video with the expected video
            video_similarity_results = self.video_comparator.calculate_video_similarity(
                video_path_ref=expected_video_path.as_posix(),
                video_path_test=generate_video_path.as_posix()
            )
            # Clean up expected video file
            expected_video_path.unlink(missing_ok=False)
        
        # Clean up generated video file
        if generate_video_path is not None:
            generate_video_path.unlink(missing_ok=True)

        evaluation_results.update(video_similarity_results)

        visual_similarity = (video_similarity_results["ssim_score"] * video_similarity_results["visual_semantic_score"]) ** 0.5
        evaluation_results["visual_similarity"] = visual_similarity

        # CodeBERTBLEU score
        codebertbleu_score = (evaluation_results["codebert_similarity"] * evaluation_results["codebleu"]) ** 0.5
        evaluation_results["codebertbleu"] = codebertbleu_score

        # Add Manim evaluation result to the evaluation results
        evaluation_results["manim_render_success"] = manim_result.success
        evaluation_results["manim_render_info"] = manim_result.info
        evaluation_results["manim_render_error"] = manim_result.errors

        # Add additional information to the evaluation results
        evaluation_results["raw_generated_code"] = raw_generated_code
        evaluation_results["raw_generated_thinking"] = raw_generated_thinking
        evaluation_results["raw_generated_code_plan"] = raw_generated_code_plan
        evaluation_results["raw_rag_info"] = raw_rag_info
        evaluation_results["raw_initial_code"] = raw_initial_code

        evaluation_results["text_script"] = textual_script
        evaluation_results["expected_code"] = expected_code
        evaluation_results["generated_code"] = generated_code

        return evaluation_results

    
    def evaluate_batch(
            self, 
            textual_scripts: list, 
            expected_codes: list, 
            thinking_contents: list = None, 
            feedback_round:bool = False, 
            feedback_round_count:int = 1, 
            forced_feedback_round:bool = False,
            generation_timeout_sec: int = 1800
        ) -> list:
        """
        Evaluate a batch of textual scripts against expected codes.

        Args:
            textual_scripts (list): A list of textual scripts to evaluate.
            expected_codes (list): A list of expected ManimCE codes.
            thinking_contents (list, optional): A list of pre-generated thinking content to assist in code generation.
            feedback_round (bool, optional): Whether to enable feedback loop for evaluation. Defaults to False.
            feedback_round_count (int, optional): The number of feedback loop iterations to perform. Defaults to 1.
            generation_timeout_sec (int): Timeout in seconds for code generation.

        Returns:
            list: A list of dictionaries containing the evaluation results for each sample.
        """

        # ToDo: Parallelize this function - Use num_poc=4 in map
        results = list(map(
            partial(
                self.evaluate_sample,
                prepared_code_plan=thinking_contents,
                feedback_round=feedback_round,
                forced_feedback_round=forced_feedback_round,
                feedback_round_count=feedback_round_count,
                generation_timeout_sec=generation_timeout_sec
            ), 
            textual_scripts, 
            expected_codes
        ))

        return results
    
    def run_evaluation(
            self, 
            batch_size: int, 
            textual_scripts: list, 
            expected_codes: list, 
            thinking_contents: list = None, 
            feedback_round:bool = False, 
            feedback_round_count:int = 1, 
            forced_feedback_round:bool = False,
            generation_timeout_sec: int = 3600,
            watch_variable:str = 'manim_render_success'
        ) -> list:
        """
        Run the evaluation process for a batch of textual scripts and expected codes.

        Args:
            batch_size (int): The size of the batch to evaluate.
            textual_scripts (list): A list of textual scripts to evaluate.
            expected_codes (list): A list of expected ManimCE codes.
            thinking_contents (list, optional): A list of pre-generated thinking content to assist in code generation.
            feedback_round (bool, optional): Whether to enable feedback loop for evaluation. Defaults to False.
            feedback_round_count (int, optional): The number of feedback loop iterations to perform. Defaults to 1.
            generation_timeout_sec (int): Timeout in seconds for code generation (per sample). Default is 3600 seconds.
            watch_variable (str): The variable to monitor and display average during evaluation.

        Returns:
            list: A list of dictionaries containing the evaluation results for each sample.
        """

        # Split the textual scripts and expected codes into batches
        textual_scripts_batches = [textual_scripts[i:i + batch_size] for i in range(0, len(textual_scripts), batch_size)]
        expected_code_batches = [expected_codes[i:i + batch_size] for i in range(0, len(expected_codes), batch_size)]
        if thinking_contents:
            thinking_contents_batches = [thinking_contents[i:i + batch_size] for i in range(0, len(thinking_contents), batch_size)]

        results = []
        # Evaluate each batch with tqdm progress bar
        tqdm_pbar = tqdm(zip(textual_scripts_batches, expected_code_batches), total=len(textual_scripts_batches))
        for batch_textual_scripts, batch_expected_codes in tqdm_pbar:
            batch_results = self.evaluate_batch(
                batch_textual_scripts,
                batch_expected_codes, 
                thinking_contents=thinking_contents_batches if thinking_contents else None,
                feedback_round=feedback_round,
                forced_feedback_round=forced_feedback_round,
                feedback_round_count=feedback_round_count,
                generation_timeout_sec=generation_timeout_sec
            )
            results.extend(batch_results)

            # Update tqdm description with average of watch_variable
            avg_watch_value = sum(result.get(watch_variable, 0.0) for result in results) / len(results)
            tqdm_pbar.set_description(f"Evaluating batches | Avg {watch_variable}: {avg_watch_value:.4f}")

        return results
