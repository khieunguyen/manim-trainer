from transformers import AutoConfig

from src.utils.prompt_template import PromptChatTemplate


def is_moe(model_id: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        txt = str(cfg).lower()
    except Exception:
        txt = model_id.lower()
    return any(k in txt for k in ["moe", "mixtral", "experts", "num_experts", "router"])


def preprocess_prompt_completion(
        sample, 
        prompt_template: PromptChatTemplate, 
        x_column_name: str = "reviewed_description", 
        y_column_name: str = "code", 
        suppress_thinking: bool = True,
        no_system_role: bool = False,
        no_think_tag: str = "/no_think"
    ) -> dict:
    """
    Preprocess the dataset sample by formatting it according to the specified prompt template.

    Format from docs below: https://huggingface.co/docs/trl/sft_trainer
    {
        "prompt": [{"role": "user", "content": example["Question"]}],
        "completion": [
            {"role": "assistant", "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}"}
        ],
    }
    
    """
    system_prompt = prompt_template.system_prompt_template
    user_prompt = prompt_template.user_prompt_template.format(**{x_column_name: sample[x_column_name]})
    completion_prompt = prompt_template.completion_prompt_template.format(**{y_column_name: sample[y_column_name]})

    if suppress_thinking:
        user_prompt += "\n" + no_think_tag

    formatted_dict = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "completion": [
            {"role": "assistant", "content": completion_prompt}
        ],
        "expected_code": sample[y_column_name], # For evaluation purposes
    }

    if no_system_role:
        formatted_dict["prompt"] = [
            {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
        ]

    return formatted_dict

def format_for_sft(example: dict, tokenizer) -> dict:
    messages = example["prompt"] + example["completion"]
    return {
        "messages": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ),
    }


