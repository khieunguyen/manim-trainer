"""inference_engine.py
Inference engine for the Manim LLM project.
"""

__author__      = "Ravidu Silva"

import typing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass
import requests

from src.utils import utils
from src.inference.inference_utils import generate_completions_with_api_v1, generate_completions_with_huggingface, APIConfig, SUPPORTED_BACKENDS, DEFAULT_BACKEND
from src.inference.model_config import ModelConfig
from src.inference.planner_engine import PlannerEngine
from src.utils.prompt_template import PromptTemplate, PromptChatTemplate
from src.utils.constants import Constants
from src.rag.rag_engine import RAGEngine
from src.evaluation.manim_evaluator import ManimEvaluator

@dataclass
class ModelResponse:
    """Model response class to handle the output from the model."""

    generated_code: str
    thinking_content: str
    initial_code: str = ""
    code_plan: str = ""
    rag_info: str = ""


class InferenceEngine:
    """Inference engine for the Manim LLM project."""

    def __init__(
            self, 
            hf_model_name: str, 
            load_in_4bit: bool = False, 
            device_map: str = "auto", 
            max_new_tokens: int = ModelConfig.MAX_NEW_TOKENS,
            generation_prompt_template: str = PromptTemplate.MANIM_VIDEO_GEN_PROMPT,
            rag_generation_prompt_template: str = PromptTemplate.MANIM_VIDEO_GEN_W_RAG_PROMPT,
            chat_gen_prompt_template: PromptChatTemplate = PromptTemplate.MANIM_VIDEO_GEN_CHAT_TEMPLATE,
            chat_rag_fb_gen_prompt_template: PromptChatTemplate = PromptTemplate.MANIM_VIDEO_GEN_CHAT_RAG_FB_TEMPLATE,
            chat_rag_only_gen_prompt_template: PromptChatTemplate = PromptTemplate.MANIM_VIDEO_GEN_CHAT_RAG_ONLY_TEMPLATE,
            chat_fb_only_gen_prompt_template: PromptChatTemplate = PromptTemplate.MANIM_VIDEO_GEN_CHAT_FB_ONLY_TEMPLATE,
            prompt_mode: typing.Literal["text", "chat"] = "text",
            enable_chat_thinking: bool = False,
            remove_token_type_ids: bool = False,
            no_system_role_support: bool = False,
            rag_engine: RAGEngine = None,
            planner_engine: 'PlannerEngine' = None,
            thinking_token_id: int = 151668,  # </think>
            thinking_token: str = "</think>",
            loaded_base_model = None,
            loaded_peft_model = None,
            loaded_tokenizer = None,
            backend: typing.Literal["huggingface", "unsloth", "api", "preloaded", "preloaded_peft"] = DEFAULT_BACKEND,
            api_config: APIConfig = None,
        ):
        """
        Initialize the inference engine with the specified model.

        Args:
            hf_model_name (str): The name of the Hugging Face model to use. Set to None if the model is loaded via `loaded_base_model`.
            load_in_4bit (bool): Whether to load the model in 4-bit quantization. Ignored if `loaded_base_model` is provided.
            device_map (str): The device map to use for the model. Ignored if `loaded_base_model` is provided.
            max_new_tokens (int): The maximum number of new tokens to generate.
            generation_prompt_template (str): The default prompt template for generating ManimCE code. This can be overridden by the `generation_template` parameter in the `generate_manim_code` method.
            rag_generation_prompt_template (str): The default prompt template for generating ManimCE code with RAG. This can be overridden by the `rag_generation_template` parameter in the `generate_manim_code` method.
            chat_gen_prompt_template (PromptChatTemplate): The default chat prompt template for generating ManimCE code.
            chat_rag_gen_prompt_template (PromptChatTemplate): The default chat prompt template for generating ManimCE code with RAG.
            chat_rag_only_gen_prompt_template (PromptChatTemplate): The default chat prompt template for generating ManimCE code with RAG only.
            chat_fb_only_gen_prompt_template (PromptChatTemplate): The default chat prompt template for generating ManimCE code with feedback only.
            enable_chat_thinking (bool): Whether to enable chat thinking mode. If True, the model will generate its own thinking content in chat mode.
            prompt_mode (str): The mode of the prompt. Can be "text" or "chat". If "chat", the `chat_prompt_template` will be used.
            remove_token_type_ids (bool): Whether to remove token type IDs from the model inputs.
            no_system_role_support (bool): Whether the model backend supports system role in chat mode. If False and `prompt_mode` is "chat", content in system role will be added to the user role.
            rag_engine (RAGEngine): The RAG engine to use for API call extraction and filtering.
            thinking_token_id (int): The token ID used to indicate the end of the model's thinking process. For 'api' backend, this will be ignored and the `thinking_token` will be used.
            thinking_token (str): The token used to indicate the end of the model's thinking process. Default is '</think>'. Only used for 'api' backend. For other backends, this use `thinking_token_id`.
            loaded_base_model: The pre-loaded base model. If provided, `hf_model_name` will be ignored.
            loaded_tokenizer: The pre-loaded tokenizer. If provided, `hf_model_name` will be ignored.
            backend (str): The backend to use for inference. Supported values are 'huggingface', 'unsloth', 'api', 'preloaded' and 'preloaded_peft'.
                - 'huggingface': Use Hugging Face Transformers library for inference.
                - 'unsloth': Use Unsloth for inference (only supports Unsloth models).
                - 'api': Use an external API for inference. Requires `api_config`.
                - 'preloaded': Use a pre-loaded model and tokenizer.
                - 'preloaded_peft': Use a pre-loaded PEFT model and tokenizer.
            api_config (APIConfig): The configuration for the API backend. Required if `backend` is 'api'.
        """
        # Validate backend type
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Invalid backend type: {backend}. Supported backends are 'huggingface', 'unsloth', 'api', and 'preloaded'.")
        
        self.backend = backend
        self.tokenizer = None
        self.hf_model_name = None
        self.load_in_4bit = None
        self.api_config = None
        self.base_model = None
        self.peft_model = None
        self.model_in_use = None
        self.max_new_tokens = max_new_tokens
        self.generation_prompt_template = generation_prompt_template
        self.rag_generation_prompt_template = rag_generation_prompt_template
        self.chat_gen_prompt_template = chat_gen_prompt_template
        self.chat_rag_fb_gen_prompt_template = chat_rag_fb_gen_prompt_template
        self.chat_rag_only_gen_prompt_template = chat_rag_only_gen_prompt_template
        self.chat_fb_only_gen_prompt_template = chat_fb_only_gen_prompt_template
        self.prompt_mode = prompt_mode
        self.enable_chat_thinking = enable_chat_thinking
        self.remove_token_type_ids = remove_token_type_ids
        self.no_system_role_support = no_system_role_support
        self.thinking_token_id = thinking_token_id
        self.thinking_token = thinking_token
        self.device_map = device_map

        if backend == 'api':
            if api_config is None:
                raise ValueError("API backend requires an APIConfig object. Please provide a valid APIConfig.")
            self.api_config = api_config
        elif backend == 'unsloth' or backend == 'huggingface':
            if api_config is not None:
                print("Warning: `api_config` is provided but will be ignored for Hugging Face or Unsloth backends.")
                print(f"Loading model: '{hf_model_name}'" + (" with 4bit..." if load_in_4bit else "..."))

            if hf_model_name is None:
                raise ValueError(f"`hf_model_name` must be provided for '{backend}' backend. Please provide a valid Hugging Face model name.")

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
                )
            else:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=device_map
                )
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
        elif backend == 'preloaded_peft':
            if api_config is not None:
                print("Warning: `api_config` is provided but will be ignored for preloaded PEFT models.")
            if loaded_base_model is not None:
                print("Warning: `loaded_base_model` is provided but will be ignored for preloaded PEFT models.")
            if loaded_peft_model and loaded_tokenizer:
                print("Using pre-loaded PEFT model and tokenizer. `hf_model_name` will have no effect for the base model.")
                self.hf_model_name = hf_model_name
                self.tokenizer = loaded_tokenizer
                self.set_peft_model(loaded_peft_model)
                self.base_model = None
            else:
                raise ValueError("For 'preloaded_peft' backend, `loaded_peft_model` and `loaded_tokenizer` must be provided.")
        else:
            raise ValueError(f"Unsupported backend type: {backend}. Supported backends are 'huggingface', 'unsloth', 'api', and 'preloaded'.")


        print(f"Model in use: {self.get_inference_model_type()}")
        print(f"Prompt mode: {prompt_mode}")

        self.rag_engine = rag_engine
        if rag_engine is not None:
            print("RAG engine is set up and enabled.")
        else:
            print("RAG engine is not enabled.")

        self.planner_engine = planner_engine
        if planner_engine is not None:
            print("Planner engine is set up and enabled.")
        else:
            print("Planner engine is not enabled.")
        

    def __generate_code(
            self,
            model_input_prompt: str = None,
            model_input_messages: typing.List[dict] = None,
            enable_chat_thinking: bool = False
    ) -> ModelResponse:
        """Generate code based on the model input prompt.
        Args:
            model_input_prompt (str): The prompt to generate code from. Used if 'prompt_mode' is 'text'.
            model_input_messages (list): A list of messages to use for chat-based generation. Used if 'prompt_mode' is 'chat'.
            enable_chat_thinking (bool): Whether to enable chat thinking mode. If True, the model will generate its own thinking content.
        Returns:
            ModelResponse: An object containing the generated code and the model's internal thought process.
        """
        if model_input_prompt is None and model_input_messages is None:
            raise ValueError("Either `model_input_prompt` or `model_input_messages` must be provided.")
        
        if self.prompt_mode == "text" and model_input_prompt is None:
            raise ValueError("`model_input_prompt` must be provided when `prompt_mode` is 'text'.")
        
        if self.prompt_mode == "chat" and model_input_messages is None:
            raise ValueError("`model_input_messages` must be provided when `prompt_mode` is 'chat'.")
        
        
        if self.no_system_role_support:
            # Move system content to user role
            combined_user_content = f"{model_input_messages[0]['content']}\n\n{model_input_messages[1]['content']}"
            model_input_messages = [
                {"role": "user", "content": combined_user_content}
            ]

        thinking_content, generated_code = "", ""

        if self.backend == 'api':
            # Use API to generate code
            if self.prompt_mode == "text":
                thinking_content, generated_code = generate_completions_with_api_v1(
                    model_input_prompt=model_input_prompt,
                    api_config=self.api_config,
                    max_new_tokens=self.max_new_tokens,
                    thinking_token=self.thinking_token,
                    prompt_mode=self.prompt_mode,
                    tokenizer=self.tokenizer,
                )
            elif self.prompt_mode == "chat":
                thinking_content, generated_code = generate_completions_with_api_v1(
                    model_input_messages=model_input_messages,
                    api_config=self.api_config,
                    max_new_tokens=self.max_new_tokens,
                    thinking_token=self.thinking_token,
                    prompt_mode=self.prompt_mode,
                    tokenizer=self.tokenizer,
                )
        # ToDo: Implement Unsloth backend if needed
        # elif self.backend == 'unsloth':
        #     # Use Unsloth to generate code
        #     pass
        else:
            # Use Hugging Face model to generate code
            if self.prompt_mode == "text":
                thinking_content, generated_code = generate_completions_with_huggingface(
                    model=self.model_in_use,
                    tokenizer=self.tokenizer,
                    prompt_mode=self.prompt_mode,
                    thinking_token_id=self.thinking_token_id,
                    max_new_tokens=self.max_new_tokens,
                    remove_token_type_ids=self.remove_token_type_ids,
                    model_input_prompt=model_input_prompt,
                    enable_chat_thinking=enable_chat_thinking
                )
            elif self.prompt_mode == "chat":
                thinking_content, generated_code = generate_completions_with_huggingface(
                    model=self.model_in_use,
                    tokenizer=self.tokenizer,
                    prompt_mode=self.prompt_mode,
                    thinking_token_id=self.thinking_token_id,
                    max_new_tokens=self.max_new_tokens,
                    remove_token_type_ids=self.remove_token_type_ids,
                    model_input_messages=model_input_messages,
                    enable_chat_thinking=enable_chat_thinking
                )

        return ModelResponse(
            generated_code=generated_code,
            thinking_content=thinking_content
        )


    def generate_manim_code(
            self, 
            textual_script: str,
            feedback_round: bool = False,
            provided_initial_code: str = None,
            render_errors: str = None,
            prepared_code_plan: str = None,
            generation_template: str = None,
            rag_fb_generation_template: str = None,
            rag_only_generation_template: str = None,
            fb_only_generation_template: str = None,
            enable_chat_thinking: bool = None
        ) -> ModelResponse:
        """
        Generate ManimCE code based on the textual script.
        
        Args:
            textual_script (str): The textual script describing the Manim animation.
            feedback_round (bool): Whether this is a feedback round. If True, `provided_initial_code` must be provided.
            provided_initial_code (str, optional): The initial code provided for feedback rounds. Required if `feedback_round` is True.
            render_errors (str, optional): Render errors from the previous
            prepared_code_plan (str, optional): Pre-generated code plan to assist in code generation. Only provide either this or `planner_engine`.
            generation_template (str, optional): The template for generating the ManimCE code. If not provided, the default template will be used.
            rag_fb_generation_template (str, optional): The template for generating the ManimCE code with RAG in feedback rounds. If not provided, the default RAG template will be used.
            rag_only_generation_template (str, optional): The template for generating the ManimCE code with RAG only. If not provided, the default RAG only template will be used.
            fb_only_generation_template (str, optional): The template for generating the ManimCE code with feedback only. If not provided, the default feedback only template will be used.
            enable_chat_thinking (bool, optional): Whether to enable chat thinking mode. If None, the default value from the inference engine will be used.
        
        Returns:
            ModelResponse: An object containing the generated ManimCE code and the model's internal thought process.
        """
        generation_template = generation_template or self.generation_prompt_template
        enable_chat_thinking = enable_chat_thinking if enable_chat_thinking is not None else self.enable_chat_thinking
        generated_plan = None
        model_response = None
        initial_code = None

        if not feedback_round:
            if self.planner_engine is not None:
                if prepared_code_plan is not None:
                    raise ValueError("`prepared_code_plan` is provided but `planner_engine` is not None. Please choose one of them.")
                
                # Use the planner engine to generate a plan
                _, generated_plan = self.planner_engine.generate_manim_code_plan(
                    textual_script=textual_script
                )

                prepared_code_plan = utils.extract_manim_code_plan_from_llm_response(
                    response=generated_plan,
                    return_multiple_plan_blocks=False,
                    select_index=-1
                )

            if self.prompt_mode == "text":
                model_input_prompt = None
                if prepared_code_plan is None:
                    model_input_prompt = generation_template.format(
                        reviewed_description=textual_script
                    )
                else:
                    model_input_prompt = generation_template.format(
                        reviewed_description=textual_script,
                        code_plan=prepared_code_plan
                    )
                model_response = self.__generate_code(
                    model_input_prompt=model_input_prompt
                )

            elif self.prompt_mode == "chat":
                system_content = None
                user_content = None
                if prepared_code_plan is None:
                    system_content = self.chat_gen_prompt_template.system_prompt_template
                    user_content = self.chat_gen_prompt_template.user_prompt_template.format(
                        reviewed_description=textual_script
                    )
                else:
                    system_content = self.chat_gen_prompt_template.system_prompt_template
                    user_content = self.chat_gen_prompt_template.user_prompt_template.format(
                        reviewed_description=textual_script,
                        code_plan=prepared_code_plan
                    )

            # Create messages for chat-based generation
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            model_response = self.__generate_code(
                model_input_messages=messages,
                enable_chat_thinking=enable_chat_thinking
            )

            if self.rag_engine is not None:
                # If RAG is enabled, extract API calls and add to the model response
                rag_only_generation_template = rag_only_generation_template or self.rag_generation_prompt_template
                initial_code = utils.extract_manim_code_from_llm_response(
                    model_response.generated_code,
                    return_multiple_code_blocks=False,
                    select_index=-1
                )
                api_info = self.rag_engine.get_formatted_api_info(
                    code=initial_code
                )

                if self.prompt_mode == "text":
                    raise NotImplementedError("RAG only generation with text prompt mode is not implemented yet.")
                elif self.prompt_mode == "chat":
                    messages = [
                        {"role": "system", "content": self.chat_rag_only_gen_prompt_template.system_prompt_template},
                        {"role": "user", "content": self.chat_rag_only_gen_prompt_template.user_prompt_template.format(
                            reviewed_description=textual_script,
                            initial_code=initial_code,
                            api_info=api_info
                        )}
                    ]
                    model_response = self.__generate_code(
                        model_input_messages=messages,
                        enable_chat_thinking=enable_chat_thinking
                    )

                model_response.rag_info = api_info
                model_response.initial_code = initial_code
        
        else:
            # Feedback round
            if provided_initial_code is None:
                raise ValueError("`provided_initial_code` must be provided for feedback rounds.")

            # If RAG is enabled, extract API calls and regenerate code
            if self.rag_engine is not None:
                rag_fb_generation_template = rag_fb_generation_template or self.chat_rag_fb_gen_prompt_template
                initial_code = provided_initial_code
                api_info = self.rag_engine.get_formatted_api_info(
                    code=initial_code
                )
                render_errors = render_errors or "NONE"

                if self.prompt_mode == "text":
                    raise NotImplementedError("RAG with feedback generation with text prompt mode is not implemented yet.")
                elif self.prompt_mode == "chat":
                    messages = [
                        {"role": "system", "content": self.chat_rag_fb_gen_prompt_template.system_prompt_template},
                        {"role": "user", "content": self.chat_rag_fb_gen_prompt_template.user_prompt_template.format(
                            reviewed_description=textual_script,
                            initial_code=initial_code,
                            api_info=api_info,
                            render_errors=render_errors
                        )}
                    ]
                    model_response = self.__generate_code(
                        model_input_messages=messages,
                        enable_chat_thinking=enable_chat_thinking
                    )
                
                # Add RAG info and initial code to the model response
                model_response.rag_info = api_info
                model_response.initial_code = initial_code

            else:
                # No RAG, just feedback based on render errors
                fb_only_generation_template = fb_only_generation_template or self.chat_fb_only_gen_prompt_template

                initial_code = provided_initial_code
                render_errors = render_errors or "NONE"

                if self.prompt_mode == "text":
                    raise NotImplementedError("Feedback only generation with text prompt mode is not implemented yet.")
                elif self.prompt_mode == "chat":
                    messages = [
                        {"role": "system", "content": self.chat_fb_only_gen_prompt_template.system_prompt_template},
                        {"role": "user", "content": self.chat_fb_only_gen_prompt_template.user_prompt_template.format(
                            reviewed_description=textual_script,
                            initial_code=initial_code,
                            render_errors=render_errors
                        )}
                    ]
                    model_response = self.__generate_code(
                        model_input_messages=messages,
                        enable_chat_thinking=enable_chat_thinking
                    )
                
                # Add initial code to the model response
                model_response.initial_code = initial_code
        
        # If there is a generated plan, add it to the model response
        if generated_plan is not None and model_response is not None:
            model_response.code_plan = generated_plan
        return model_response
    

    def generate_manim_code_with_feedback(
            self,
            textual_script: str,
            manim_evaluator: ManimEvaluator,
            forced_feedback_round: bool = False,
            feedback_round_count: int = 1,
            max_error_lines: int = 10,
            prepared_code_plan: str = None,
            generation_template: str = None,
            rag_fb_generation_template: str = None,
            rag_only_generation_template: str = None,
            fb_only_generation_template: str = None,
            enable_chat_thinking: bool = None,
        ) -> ModelResponse:
        """
        Generate ManimCE code with feedback rounds.

        Args:
            textual_script (str): The textual script describing the Manim animation.
            manim_evaluator (ManimEvaluator): An instance of the ManimEvaluator class to evaluate Manim code snippets.
            forced_feedback_round (bool): Whether to force feedback rounds even if the previous round was successful.
            feedback_round_count (int): The number of feedback rounds to perform. Default is 1.
            prepared_code_plan (str, optional): Pre-generated code plan to assist in code generation. Only provide either this or `planner_engine`.
            generation_template (str, optional): The template for generating the ManimCE code. If not provided, the default template will be used.
            rag_fb_generation_template (str, optional): The template for generating the ManimCE code with RAG in feedback rounds. If not provided, the default RAG template will be used.
            rag_only_generation_template (str, optional): The template for generating the ManimCE code with RAG only. If not provided, the default RAG only template will be used.
            fb_only_generation_template (str, optional): The template for generating the ManimCE code with feedback only. If not provided, the default feedback only template will be used.
            enable_chat_thinking (bool, optional): Whether to enable chat thinking mode. If None, the default value from the inference engine will be used.
        Returns:
            ModelResponse: An object containing the generated ManimCE code and the model's internal thought process.
        """
        if feedback_round_count < 1:
            raise ValueError("`feedback_round_count` must be at least 1.")
        if manim_evaluator is None:
            raise ValueError("Manim evaluator is not set. Please set the Manim evaluator before generating code with feedback rounds.")

        if enable_chat_thinking is None:
            enable_chat_thinking = self.enable_chat_thinking
        generation_template = generation_template or self.generation_prompt_template
        rag_fb_generation_template = rag_fb_generation_template or self.chat_rag_fb_gen_prompt_template
        rag_only_generation_template = rag_only_generation_template or self.chat_rag_only_gen_prompt_template
        fb_only_generation_template = fb_only_generation_template or self.chat_fb_only_gen_prompt_template

        model_response = None
        initial_code = None
        render_errors = None
        render_result = None
        feedback_round_count+= 1  # Adjust for the initial generation round
        for round_num in range(feedback_round_count):
            if round_num == 0:
                # First round, generate code based on the initial script
                model_response = self.generate_manim_code(
                    textual_script=textual_script,
                    feedback_round=False,
                    provided_initial_code=None,
                    render_errors=None,
                    prepared_code_plan=prepared_code_plan,
                    generation_template=generation_template,
                    rag_fb_generation_template=rag_fb_generation_template,
                    rag_only_generation_template=rag_only_generation_template,
                    fb_only_generation_template=fb_only_generation_template,
                    enable_chat_thinking=enable_chat_thinking
                )

            else:
                # Subsequent rounds, use the previous generated code as the initial code
                model_response = self.generate_manim_code(
                    textual_script=textual_script,
                    feedback_round=True,
                    provided_initial_code=initial_code,
                    render_errors=render_errors,
                    prepared_code_plan=prepared_code_plan,
                    generation_template=generation_template,
                    rag_fb_generation_template=rag_fb_generation_template,
                    rag_only_generation_template=rag_only_generation_template,
                    fb_only_generation_template=fb_only_generation_template,
                    enable_chat_thinking=enable_chat_thinking
                )
            
            # Update the initial code for the next round
            initial_code = utils.extract_manim_code_from_llm_response(
                model_response.generated_code, 
                return_multiple_code_blocks=False, 
                select_index=-1
            )

            # Evaluate the generated code using the Manim evaluator if available
            render_result = manim_evaluator.evaluate_code(
                manim_code=initial_code,
                clear_output=True
            )

            if render_result.success:
                render_errors = "NONE"  # Reset render errors if the rendering is successful
                # If the rendering is successful, break the loop
                if not forced_feedback_round:
                    break
            else:
                # Get last 10 lines of the error message
                render_errors = "\n".join(render_result.errors.splitlines()[-max_error_lines:])
                # If there are no render errors, set it to "NONE"
                if not render_errors:
                    render_errors = "NONE"

        return model_response


    def set_peft_model(self, peft_model: PeftModel):
        """
        Set the PEFT model for the inference engine.

        Args:
            peft_model (PeftModel): The PEFT model to use.
        """

        if not isinstance(peft_model, PeftModel):
            raise ValueError("The provided model is not a PEFT model. Please provide a valid PEFT model.")
        if self.peft_model is not None:
            print(f"Warning: Overwriting existing PEFT model: {self.peft_model}.")

        # Set the PEFT model
        self.peft_model = peft_model
        self.select_model(use_peft_model=True)
        print(f"`peft_model` of the inference engine set to {self.peft_model}.")


    def select_model(self, use_peft_model: bool = False):
        """
        Select the model to use for inference.

        Args:
            use_peft_model (bool): Whether to use the PEFT model or the base model.
        """
        if self.backend == 'api':
            print("Warning: `select_model` is not applicable for API backend. The model in use will always be the API model.")
            self.model_in_use = self.base_model
            return
        if use_peft_model:
            if self.peft_model is None:
                raise ValueError("PEFT model is not set. Please set the PEFT model before selecting it.")
            self.model_in_use = self.peft_model
        else:
            self.model_in_use = self.base_model

        print(f"Model in use: {self.get_inference_model_type()}")

    def set_eval_mode(self):
        """
        Set the model to evaluation mode.
        """
        if self.backend == 'api':
            print("Warning: `set_eval_mode` is not applicable for API backend. The model in use will always be in evaluation mode.")
            return
        if self.base_model is not None:
            self.base_model.eval()
        else:
            print("Base model is not set. Skipping evaluation mode setting for base model.")
        if self.peft_model is not None:
            self.peft_model.eval()
        else:
            print("Warning: PEFT model is not set. Skipping evaluation mode setting for PEFT model.")


        print(f"Model set to evaluation mode.")


    def set_train_mode(self):
        """
        Set the model to training mode.
        """
        if self.backend == 'api':
            print("Warning: `set_train_mode` is not applicable for API backend. The model in use will always be in evaluation mode.")
            return

        self.base_model.train()
        if self.peft_model is not None:
            self.peft_model.train()
        else:
            print("Warning: PEFT model is not set. Skipping training mode setting for PEFT model.")
        print(f"Model set to training mode.")


    def get_inference_model_type(self) -> typing.Literal["BASE", "PEFT"]:
        """
        Get the type of inference model in use.

        Returns:
            str: The type of inference model in use.
                - "BASE" for the base model
                - "PEFT" for the PEFT model
        """
        return "API" if self.backend == 'api' else "PEFT" if isinstance(self.model_in_use, PeftModel) else "BASE"