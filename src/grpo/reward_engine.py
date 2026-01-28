"""reward_engine.py
Reward engine for the Manim LLM project.
"""
__author__      = "Ravidu Silva"

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from src.evaluation.manim_evaluator import ManimEvaluator
from src.evaluation.video_comparator import VideoComparator
from src.evaluation.code_evaluator import CodeEvaluator
from src.utils import utils
import config

class RewardEngine:
    def __init__(
        self,
        text_reward_n_workers: int = 4,
        video_reward_n_workers: int = 4,
        visual_reward_weight: float = 0.8,
        text_reward_weight: float = 0.2,
        code_eval_model_name: str = config.SupportedModels.CODE_EVALUATOR_MODEL,
        embeding_clip_model_name: str = config.SupportedModels.VIDEO_COMPARATOR_EMBEDDING_CLIP_MODEL,
        clip_model_device: str = "cuda",
        video_fps: int = 5,
        manim_eval_timeout_sec: int = 300,
    ):
        """
        Initialize the RewardEngine with specified parameters.

        Args:
            text_reward_n_workers (int): Number of workers for text reward computation.
            video_reward_n_workers (int): Number of workers for video reward computation.
            visual_reward_weight (float): Weight for visual reward in unified reward computation. Default is 0.8.
            text_reward_weight (float): Weight for text reward in unified reward computation. Default is 0.2.
            code_eval_model_name (str): Model name for code evaluation.
            embeding_clip_model_name (str): Model name for video embedding and comparison.
            clip_model_device (str): Device to run the clip model on.
            video_fps (int): Frames per second for video processing.
            manim_eval_timeout_sec (int): Timeout in seconds for Manim code evaluation.
        """

        if not (0.0 <= visual_reward_weight <= 1.0):
            raise ValueError("visual_reward_weight must be between 0.0 and 1.0")
        if not (0.0 <= text_reward_weight <= 1.0):
            raise ValueError("text_reward_weight must be between 0.0 and 1.0")
        self.visual_reward_weight = visual_reward_weight
        self.text_reward_weight = text_reward_weight

        print(f"Initializing RewardEngine with {text_reward_n_workers} text reward workers and {video_reward_n_workers} video reward workers.")
        self.text_reward_n_workers = text_reward_n_workers
        self.video_reward_n_workers = video_reward_n_workers

        self.code_eval_workers = [CodeEvaluator(
            model_name=code_eval_model_name
        ) for _ in range(text_reward_n_workers)]

        self.comparator_workers = [VideoComparator(
            embedding_clip_model=embeding_clip_model_name,
            device=clip_model_device,
            fps=video_fps,
        ) for _ in range(video_reward_n_workers)]
        self.manim_eval_workers = [ManimEvaluator(
            timeout=manim_eval_timeout_sec
        ) for _ in range(video_reward_n_workers)]

        self.text_reward_worker_queue = queue.Queue()
        for text_reward_worker in self.code_eval_workers:
            self.text_reward_worker_queue.put(text_reward_worker)

        self.video_reward_worker_queue = queue.Queue()
        for comp_worker, manim_worker in zip(self.comparator_workers, self.manim_eval_workers):
            self.video_reward_worker_queue.put( (comp_worker, manim_worker) )

        print("RewardEngine initialized.")

    

    def get_geo_mean_unified_reward(self, completions: list, expected_code: list, **kwargs) -> list[float]:
        """
        Compute the unified reward as the geometric mean of CodeBERT-CodeBLEU score and Video Similarity score.

        Args:
            completions (list): A list of dictionaries containing the generated completions.
            expected_code (list): A list of dictionaries containing the expected code.

        Returns:
            list[float]: A list of unified scores for each completion.
        """
        code_rewards = self.get_codebertbleu_score_reward_parallel(
            completions=completions,
            expected_code=expected_code,
        )
        video_rewards = self.get_video_similarity_reward_parallel(
            completions=completions,
            expected_code=expected_code,
        )

        unified_rewards = [(code_reward * video_reward) ** 0.5 for code_reward, video_reward in zip(code_rewards, video_rewards)]
        return unified_rewards

    def get_arithmetic_weighted_mean_unified_reward(self, completions: list, expected_code: list, **kwargs) -> list[float]:
        """
        Compute the unified reward as the arithmetic weighted mean of CodeBERT-CodeBLEU score and Video Similarity score.

        Args:
            completions (list): A list of dictionaries containing the generated completions.
            expected_code (list): A list of dictionaries containing the expected code.

        Returns:
            list[float]: A list of unified scores for each completion.
        """
        code_rewards = self.get_codebertbleu_score_reward_parallel(
            completions=completions,
            expected_code=expected_code,
        )
        video_rewards = self.get_video_similarity_reward_parallel(
            completions=completions,
            expected_code=expected_code,
        )

        unified_rewards = [code_reward*self.text_reward_weight + video_reward*self.visual_reward_weight for code_reward, video_reward in zip(code_rewards, video_rewards)]
        return unified_rewards


    def _get_codebertbleu_score_reward_single(self, completion: dict, expected_code: str, **kwargs) -> float:
        """
        Compute the geometric mean of CodeBERT and CodeBLEU scores for a single generated completion against the expected code.

        Args:
            completion (dict): A dictionary containing the generated completion.
            expected_code (str): The expected code as a string.

        Returns:
            float: The geometric mean score.
        """
        # Block until a worker is available
        evaluator = self.text_reward_worker_queue.get()
        try:
            gen = completion
            exp = expected_code.strip()

            gen_code = gen['content']
            # print("Generated Code:", gen_code)
            gen_code = utils.extract_manim_code_from_llm_response(gen_code)
            # print("Extracted Code:", gen_code)
            # print("Expected Code:", exp)
            if gen_code is None or gen_code.strip() == "":
                # print("WARNING: Generated code is empty or None. Assigning reward of 0.0")
                return 0.0
            exp_code = exp
            codebleu_score = evaluator.calculate_codebleu(gen_code, exp_code)["codebleu"]
            codebert_score = evaluator.codebert_similarity(gen_code, exp_code)
            # Geometric mean of CodeBERT and CodeBLEU scores
            geo_mean_score = (codebert_score * codebleu_score) ** 0.5
            return geo_mean_score
        finally:
            # Return the worker to the queue
            self.text_reward_worker_queue.put(evaluator)

    
    def get_codebertbleu_score_reward_parallel(self, completions: list, expected_code: list, **kwargs) -> list[float]:
        """
        Compute the geometric mean of CodeBERT and CodeBLEU scores for a list of generated completions against the expected code using parallel workers.

        Args:
            completions (list): A list of dictionaries containing the generated completions.
            expected_code (list): A list of dictionaries containing the expected code.

        Returns:
            list[float]: A list of geometric mean scores for each completion.
        """
        assert len(completions) == len(expected_code), "Length of completions and expected_code must be the same."

        rewards = [0.0] * len(completions)
        num_samples = len(completions)

        assert self.text_reward_n_workers <= num_samples, "Number of workers must be less than or equal to number of samples."

        with ThreadPoolExecutor(max_workers=self.text_reward_n_workers) as executor:
            future_to_index_eval_proc = {}
            for i in range(num_samples):
                future_proc = executor.submit(
                    self._get_codebertbleu_score_reward_single,
                    completion=completions[i][0],
                    expected_code=expected_code[i],
                )
                future_to_index_eval_proc[future_proc] = i 
            for future_proc in as_completed(future_to_index_eval_proc):
                index = future_to_index_eval_proc[future_proc]
                try:
                    score = future_proc.result()
                except Exception as e:
                    print(f"ERROR: Exception occurred while processing sample index {index}: {e}")
                    score = 0.0
                rewards[index] = score
        return rewards
    
    
    def _get_video_similarity_reward_single(self, completion: dict, expected_code: str, **kwargs) -> float:
        """
        Compute the video similarity reward for a single generated completion against the expected code.

        Args:
            completion (dict): A dictionary containing the generated completion.
            expected_code (str): The expected code as a string.
        Returns:
            float: The video similarity score.
        """

        # Block until a worker is available
        comparator, manim_evaluator = self.video_reward_worker_queue.get()
        try:
            gen_code = completion['content']
            expected_code = expected_code.strip()
            gen_code = utils.extract_manim_code_from_llm_response(gen_code)
            if gen_code is None or gen_code.strip() == "":
                if False:
                    print("WARNING: Generated code is empty or None. Assigning reward of 0.0")
                    truncated_completion = ""
                    max_log_length = 100
                    if len(completion['content']) > max_log_length:
                        truncated_completion = completion['content'][:max_log_length//2] + "..." + completion['content'][-max_log_length//2:]
                    else:
                        truncated_completion = completion['content']

                    print("Generated Completion (Truncated):\n", truncated_completion)
                return 0.0

            gen_video_path = manim_evaluator.evaluate_code(gen_code, clear_output=False).video_path
            if gen_video_path is None:
                # print("WARNING: One of the videos could not be generated. Assigning reward of 0.0")
                return 0.0
            
            exp_video_path = manim_evaluator.evaluate_code(expected_code, clear_output=False).video_path
            if exp_video_path is None:
                raise ValueError("Expected code did not generate a video. Cannot compute similarity reward.")

            score = comparator.calculate_video_similarity(
                video_path_ref=exp_video_path.as_posix(),
                video_path_test=gen_video_path.as_posix()
            )

            # Clean up generated video files
            gen_video_path.unlink(missing_ok=False)
            exp_video_path.unlink(missing_ok=False)
            # Get geometric mean of SSIM and Semantic scores
            score = (score["ssim_score"] * score["visual_semantic_score"]) ** 0.5
            return score
        finally:
            # Return the worker to the queue
            self.video_reward_worker_queue.put((comparator, manim_evaluator))


    def get_video_similarity_reward_parallel(self, completions: list, expected_code: list, **kwargs) -> list[float]:
        """
        Compute the video similarity reward for a list of generated completions against the expected code using parallel workers.

        Args:
            completions (list): A list of dictionaries containing the generated completions.
            expected_code (list): A list of dictionaries containing the expected code.

        Returns:
            list[float]: A list of video similarity scores for each completion.
        """
        assert len(completions) == len(expected_code), "Length of completions and expected_code must be the same."

        rewards = [0.0] * len(completions)
        num_samples = len(completions)

        with ThreadPoolExecutor(max_workers=self.video_reward_n_workers) as executor:
            future_to_index_eval_proc = {}
            for i in range(num_samples):
                future_proc = executor.submit(
                    self._get_video_similarity_reward_single,
                    completion=completions[i][0],
                    expected_code=expected_code[i],
                )
                future_to_index_eval_proc[future_proc] = i 
            for future_proc in as_completed(future_to_index_eval_proc):
                index = future_to_index_eval_proc[future_proc]
                try:
                    score = future_proc.result()
                except Exception as e:
                    print(f"ERROR: Exception occurred while processing sample index {index}: {e}")
                    score = 0.0
                rewards[index] = score
        return rewards
    


    # def get_video_similarity_reward(self, completions: list, expected_code: list, comparator: VideoComparator, manim_evaluator: ManimEvaluator, **kwargs) -> list[float]:
    #     """
    #     Compute the video similarity reward for a list of generated completions against the expected code.

    #     Args:
    #         completions (list): A list of dictionaries containing the generated completions.
    #         expected_code (list): A list of dictionaries containing the expected code.
    #         comparator (VideoComparator): An instance of the VideoComparator class to compute the similarity.
    #         manim_evaluator (ManimEvaluator): An instance of the ManimEvaluator class to evaluate the code.

    #     Returns:
    #         list[float]: A list of video similarity scores for each completion.
    #     """
    #     assert len(completions) == len(expected_code), "Length of completions and expected_code must be the same."
    #     assert all(len(gen) == 1 for gen in completions), "Each completion should be a list with a single dict."

    #     rewards = []
    #     for gen, exp in zip(completions, expected_code):
    #         gen = gen[0]
    #         exp = exp
    #         # print("Generated Code:\n", gen['content'])
    #         # print("\n\nExpected Code:\n", exp)

    #         score = self.get_video_similarity_reward_single(
    #             completion=gen,
    #             expected_code=exp,
    #             comparator=comparator,
    #             manim_evaluator=manim_evaluator,
    #         )
    #         rewards.append(score)
    #     return rewards

