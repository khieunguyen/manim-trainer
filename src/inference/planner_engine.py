"""planner_engine.py
This module contains the PlannerEngine class, which is responsible for planning tasks."""

__author__      = "Ravidu Silva"

import typing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.inference.inference_utils import generate_completions_with_api_v1, generate_completions_with_huggingface, APIConfig, SUPPORTED_BACKENDS, DEFAULT_BACKEND
from src.inference.model_config import ModelConfig
from src.utils.constants import Constants
from src.utils.prompt_template import PromptTemplate, PromptChatTemplate

class PlannerEngine:
    """PlannerEngine class for planning tasks using an inference engine."""

    def __init__(
            self,
            backend: typing.Literal["huggingface", "unsloth", "api", "preloaded"],
            prompt_mode: typing.Literal['text', 'chat'],
            api_config: APIConfig = None,
            hf_model_name: str =None, 
            load_in_4bit: bool = False, 
            device_map: str = "auto", 
            remove_token_type_ids: bool = True,
            max_new_tokens: int = ModelConfig.MAX_NEW_TOKENS,
            thinking_token_id: int = 151668,  # </think>
            thinking_token: str = "</think>",
            loaded_base_model= None,
            loaded_tokenizer = None,
            generation_prompt: PromptChatTemplate = PromptTemplate.MANIM_PLAN_GEN_CHAT_PROMPT,
            enable_chat_thinking: bool = True
        ):
        """
        Initialize the PlannerEngine with the specified parameters.

        Args:
            backend (str): The backend to use for planning tasks.
            prompt_mode (str): The mode of the prompt, either 'text' or 'chat'.
            api_config (APIConfig): Configuration for the API if using API backend.
            hf_model_name (str): The Hugging Face model name if using Hugging Face backend.
            load_in_4bit (bool): Whether to load the model in 4-bit quantization.
            device_map (str): Device map for model loading.
            remove_token_type_ids (bool): Whether to remove token type IDs from the model inputs.
            max_new_tokens (int): The maximum number of new tokens to generate.
            thinking_token_id (int): The token ID used to separate thinking content from the generated code.
            thinking_token (str): The token used to separate thinking content from the generated code.
            loaded_base_model: Preloaded base model if available.
            loaded_tokenizer: Preloaded tokenizer if available.
            generation_prompt (str): The prompt template for generation.
            enable_chat_thinking (bool): Whether to enable chat thinking mode. If True, the model will generate its own thinking content.
        """
        if prompt_mode != "chat":
            raise NotImplementedError("Only 'chat' prompt mode is currently supported for the PlannerEngine.")

        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Supported backends are: {SUPPORTED_BACKENDS}")

        self.backend = backend
        self.prompt_mode = prompt_mode
        self.api_config = api_config
        self.hf_model_name = hf_model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.remove_token_type_ids = remove_token_type_ids
        self.max_new_tokens = max_new_tokens
        self.thinking_token_id = thinking_token_id
        self.thinking_token = thinking_token
        self.loaded_base_model = loaded_base_model
        self.loaded_tokenizer = loaded_tokenizer
        self.generation_prompt = generation_prompt
        self.enable_chat_thinking = enable_chat_thinking

        self.tokenizer = None

        self.model_in_use = None

        # Initialize the model and tokenizer based on the backend
        if self.backend == "api":
            if not self.api_config:
                raise ValueError("APIConfig must be provided when using 'api' backend.")
        elif self.backend == "huggingface" or self.backend == "unsloth":
            print("Loading model from Hugging Face or Unsloth backend...")
            print("Warning: This mode is not fully supported yet, please use 'api' backend for now.")
            if api_config is not None:
                print("Warning: `api_config` is provided but will be ignored for Hugging Face or Unsloth backends.")
                print(f"Loading model: '{hf_model_name}'" + (" with 4bit..." if load_in_4bit else "..."))
            self.hf_model_name = hf_model_name
            self.load_in_4bit = load_in_4bit
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            # Setting the pad token to eos token since the model is causal
            # and the model does not have a pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if load_in_4bit:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                ).eval()
            else:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=device_map
                ).eval()
            self.model_in_use = self.base_model
        elif backend == 'preloaded':
            if api_config is not None:
                print("Warning: `api_config` is provided but will be ignored for preloaded models.")
            if hf_model_name is not None:
                print("Warning: `hf_model_name` is provided but will be ignored for preloaded models.")
            if loaded_base_model and loaded_tokenizer:
                print("Using pre-loaded model and tokenizer. `hf_model_name` will be ignored.")
                self.tokenizer = loaded_tokenizer
                self.base_model = loaded_base_model
            else:
                raise ValueError("For 'preloaded' backend, both `loaded_base_model` and `loaded_tokenizer` must be provided.")
            self.model_in_use = self.base_model
        else:
            raise ValueError(f"Unsupported backend type: {backend}. Supported backends are 'huggingface', 'unsloth', 'api', and 'preloaded'.")

        print(f"Prompt mode: {prompt_mode}")

    def generate_manim_code_plan(
            self,
            textual_script: str,
            enable_chat_thinking: bool = None,
    ) -> typing.Tuple[str, str]:
        """
        Generate Manim code from a textual script using the configured model.

        Args:
            textual_script (str): The textual script to generate code from.
            enable_chat_thinking (bool): Whether to enable chat thinking mode. If True, the model will generate its own thinking content.

        Returns:
            tuple[str, str]: A tuple containing the thinking content and the generated Manim code plan. [thinking_content, generated_plan]
        """
        # Prepare the model input prompt
        system_prompt = self.generation_prompt.system_prompt_template
        user_prompt = self.generation_prompt.user_prompt_template.format(
            reviewed_description=textual_script
        )

        # Generate code using the appropriate backend
        if self.backend == "api":
            # Use API to generate code
            model_input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            thinking_content, generated_plan = generate_completions_with_api_v1(
                model_input_messages=model_input_messages,
                api_config=self.api_config,
                max_new_tokens=self.max_new_tokens,
                thinking_token=self.thinking_token,
                prompt_mode=self.prompt_mode,
                tokenizer=self.tokenizer
            )
        elif self.backend in ["huggingface", "unsloth", "preloaded"]:
            enable_chat_thinking = enable_chat_thinking if enable_chat_thinking is not None else self.enable_chat_thinking
            # Use Hugging Face model to generate code
            if self.prompt_mode == "text":
                raise NotImplementedError("Text prompt mode is not supported yet. Please use 'chat' mode.")
            elif self.prompt_mode == "chat":
                model_input_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                thinking_content, generated_plan = generate_completions_with_huggingface(
                    model=self.model_in_use,
                    tokenizer=self.tokenizer,
                    prompt_mode=self.prompt_mode,
                    thinking_token_id=self.thinking_token_id,
                    max_new_tokens=self.max_new_tokens,
                    remove_token_type_ids=self.remove_token_type_ids,
                    model_input_messages=model_input_messages,
                    enable_chat_thinking=enable_chat_thinking
                )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Supported backends are: {SUPPORTED_BACKENDS}")
        
        # Return the generated plan and thinking content
        return thinking_content, generated_plan

