import typing
import requests
from dataclasses import dataclass

from src.inference.model_config import ModelConfig
from src.utils.constants import Constants


SUPPORTED_BACKENDS = ['huggingface', 'unsloth', 'api', 'preloaded', 'preloaded_peft']
DEFAULT_BACKEND = 'huggingface'

@dataclass
class APIConfig:
    """Configuration for API-based models.

    This class holds the configuration parameters for API-based inference, including the base URL,
    API key, model name, timeout, and generation parameters like temperature and top_p.
    It provides a method to create an instance from a dictionary.

    Attributes:
        base_url (str): The base URL of the API.
        api_key (str, optional): The API key for authentication.
        model_name (str): The name of the model to use for inference.
        timeout_sec (int): The timeout for API requests in seconds.
        temperature (float, optional): The temperature for sampling. If None, the API will use
    """
    base_url: str = "http://localhost:8000/v1"
    api_key: typing.Optional[str] = None
    model_name: str = None
    timeout_sec: int = 300
    #max_tokens: int = ModelConfig.MAX_NEW_TOKENS  # Default is set to ModelConfig's max new tokens
    temperature: float = None  # Default is None, which means the API will use its default temperature
    top_p: float = None  # Default is None, which means the API will use its default top_p
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'APIConfig':
        return cls(**config_dict)


def generate_completions_with_api_v1(
        prompt_mode: str,
        api_config: APIConfig,
        thinking_token: str,
        tokenizer = None,
        model_input_prompt: str = None,
        model_input_messages: typing.List[dict] = None,
        max_new_tokens: int = ModelConfig.MAX_NEW_TOKENS
) -> tuple[str, str]:
    """Generate code using an API based on the model input prompt.
    
    Args:
        prompt_mode (str): The mode of the prompt, either 'text' or 'chat'.
        api_config (APIConfig): Configuration for the API including base URL, model name, and other parameters.
        thinking_token (str): The token used to separate thinking content from the generated code.
        tokenizer: The tokenizer associated with the model, required if `prompt_mode` is 'text'.
        model_input_prompt (str): The prompt to generate code from. Used if 'prompt_mode' is 'text'.
        model_input_messages (list): A list of messages to use for chat-based generation. Used if 'prompt_mode' is 'chat'.
        max_new_tokens (int): The maximum number of new tokens to generate.
    
    Returns:
        tuple[str, str]: A tuple containing the model's internal thought process and generated completion [thinking_content, completion]

    """
    headers = {
        "Content-Type": "application/json",
    }
    if api_config.api_key:
        headers["Authorization"] = f"Bearer {api_config.api_key}"

    payload = {"max_tokens": max_new_tokens}
    
    if prompt_mode == "text":
        if model_input_prompt is None:
            raise ValueError("`model_input_prompt` must be provided when `prompt_mode` is 'text'.")
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided when `prompt_mode` is 'text'.")
        # Append the thinking token to the prompt
        model_input_prompt = model_input_prompt + tokenizer.eos_token if tokenizer else model_input_prompt
        payload["messages"] = [
            {"role": "user", "content": model_input_prompt}
        ]
    elif prompt_mode == "chat":
        if model_input_messages is None:
            raise ValueError("`model_input_messages` must be provided when `prompt_mode` is 'chat'.")
        payload["messages"] = model_input_messages

    if api_config.model_name:
        payload["model"] = api_config.model_name
    if api_config.temperature is not None:
        payload["temperature"] = api_config.temperature
    if api_config.top_p is not None:
        payload["top_p"] = api_config.top_p
    
    try:
        response = requests.post(
            f"{api_config.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=api_config.timeout_sec
        )
        response.raise_for_status()
    # Handle timeout and other request exceptions
    except requests.exceptions.Timeout:
        print(f"Warning: API request timed out after {api_config.timeout_sec} seconds.")
        response = None
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {e}")
    
    if response is None:
        return "", ""
    
    response_data = response.json()
    if "choices" not in response_data or len(response_data["choices"]) == 0:
        raise RuntimeError("API response does not contain 'choices' or is empty.")
    generated_text: str = response_data["choices"][0]["message"]["content"]
    if not generated_text or generated_text == "":
        print("Warning: Generated text is empty. Returning empty ModelResponse.")
        generated_text = ""
    
    # Extract thinking content and completion based on the thinking token
    try:
        index = generated_text.rindex(thinking_token)
    except ValueError:
        index = 0 # If thinking token is not found, set index to 0

    thinking_content = generated_text[:index]
    completion = generated_text[index:].removeprefix(thinking_token).strip("\n") if index > 0 else generated_text.strip("\n")
    
    return thinking_content, completion

def generate_completions_with_huggingface(
        model,
        tokenizer,
        prompt_mode: str,
        thinking_token_id: int,
        max_new_tokens: int = ModelConfig.MAX_NEW_TOKENS,
        remove_token_type_ids: bool = False,
        model_input_prompt: str = None,
        model_input_messages: typing.List[dict] = None,
        enable_chat_thinking: bool = False
) -> tuple[str, str]:
    """Generate code using Hugging Face model based on the model input prompt.
    
    Args:
        model: The Hugging Face model to use for generation.
        tokenizer: The tokenizer associated with the model.
        prompt_mode (str): The mode of the prompt, either 'text' or 'chat'.
        thinking_token_id (int): The token ID used to separate thinking content from the generated code.
        max_new_tokens (int): The maximum number of new tokens to generate.
        remove_token_type_ids (bool): Whether to remove token type IDs from the model inputs.
        model_input_prompt (str): The prompt to generate code from. Used if 'prompt_mode' is 'text'.
        model_input_messages (list): A list of messages to use for chat-based generation. Used if 'prompt_mode' is 'chat'.
        enable_chat_thinking (bool): Whether to enable chat thinking in the generation process. This cannot be guaranteed to work with all models.

    Returns:
        tuple[str, str]: A tuple containing the model's internal thought process and generated completion [thinking_content, completion]
    """
    
    # Tokenize the prompt
    if prompt_mode == "text":
        if model_input_prompt is None:
            raise ValueError("`model_input_prompt` must be provided when `prompt_mode` is 'text'.")
        model_input_prompt = model_input_prompt + tokenizer.eos_token if tokenizer else model_input_prompt

    elif prompt_mode == "chat":
        if model_input_messages is None:
            raise ValueError("`model_input_messages` must be provided when `prompt_mode` is 'chat'.")
        model_input_prompt = tokenizer.apply_chat_template(
            model_input_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_chat_thinking
        )


    model_inputs = tokenizer(
        text=[model_input_prompt],
        return_tensors="pt"
    ).to(Constants.DEVICE)
    if remove_token_type_ids and 'token_type_ids' in model_inputs:
        del model_inputs['token_type_ids']

    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,  # Ensure generation stops at EOS
        use_cache = True
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(thinking_token_id)
    except ValueError:
        index = 0

    thinking_content =  tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    completion =  tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return thinking_content, completion