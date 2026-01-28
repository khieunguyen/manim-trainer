import re, os
import typing

def extract_manim_code_from_llm_response(response: str, return_multiple_code_blocks: bool = False, select_index: int = -1) -> typing.Union[str, list[str]]:
    """
    Extracts the Manim code from the response string.

    Args:
        response (str): The response string containing the Manim code.
        mutliple_code_blocks (bool): If True, returns a list of code blocks. If False, returns a single code block.
        select_index (int): The index of the code block to return if mutliple_code_blocks is False. Preferred to be 0 or -1.

    Returns:
        str or list[str]: The extracted Manim code. If mutliple_code_blocks is True, returns a list of code blocks.
    """
    # Remove  '`<CODE>`' and '`</CODE>`' from the response
    response = re.sub(r"`<CODE>`|`</CODE>`", "", response)

    # Fix '<CODE>' and '</CODE>' tags
    response = response.replace("<code>", "<CODE>").replace("</code>", "</CODE>").replace("CODE>", "<CODE>").replace("</<CODE>", "</CODE>")

    # Regular expression to match the code block
    code_block_pattern = r"<CODE>(.*?)</CODE>"

    # Python code block pattern
    python_code_block_pattern = r"```python(.*?)```"
    
    # Find all matches of the pattern in the response
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    # If no matches found, try to find Python code block pattern
    if not matches:
        matches = re.findall(python_code_block_pattern, response, re.DOTALL)
    
    # If mutliple_code_blocks is True, return them as a list
    if return_multiple_code_blocks:
        return [re.sub(r'<CODE>|</CODE>|```python|```', '', match).strip() for match in matches]
    
    # If mutliple_code_blocks is False, return the select_index match
    return re.sub(r'<CODE>|</CODE>|```python|```', '', matches[select_index]).strip() if matches else ""

def extract_manim_code_plan_from_llm_response(response: str, return_multiple_plan_blocks: bool = False, select_index: int = -1) -> str:
    """
    Extracts the Manim code plan from the response string.

    Args:
        response (str): The response string containing the Manim code plan.
        return_multiple_plan_blocks (bool): If True, returns a list of plan blocks. If False, returns a single plan block.
        select_index (int): The index of the plan block to return if return_multiple_plan_blocks is False. Preferred to be 0 or -1.

    Returns:
        str: The extracted Manim code plan.
    """

    # Remove  '`<PLAN>`' and '`</PLAN>`' from the response
    response = re.sub(r"`<PLAN>`|`</PLAN>`", "", response)

    # Fix '<PLAN>' and '</PLAN>' tags
    response = response.replace("<plan>", "<PLAN>").replace("</plan>", "</PLAN>").replace("PLAN>", "<PLAN>").replace("</<PLAN>", "</PLAN>")

    # Regular expression to match the plan block
    plan_block_pattern = r"<PLAN>(.*?)</PLAN>"

    # Find all matches of the pattern in the response
    matches = re.findall(plan_block_pattern, response, re.DOTALL)

    # If return_multiple_plan_blocks is True, return them as a list
    if return_multiple_plan_blocks:
        return [re.sub(r'<PLAN>|</PLAN>|```text|```', '', match).strip() for match in matches]
    
    return re.sub(r'<PLAN>|</PLAN>|```text|```', '', matches[select_index]).strip() if matches else ""


def calculate_mean_from_dict_list(dict_list: list, key: str) -> float:
    """
    Calculate the mean of a specific key from a list of dictionaries.
    
    Args:
        dict_list (list): A list of dictionaries.
        key (str): The key to calculate the mean for.
    
    Returns:
        float: The mean value.
    """
    return sum(d[key] for d in dict_list) / len(dict_list)

def set_custom_cache_path():
    """
    Set the custom cache path for Hugging Face models.
    """
    from dotenv import load_dotenv 
    # loading variables from .env file
    load_dotenv()
    if os.getenv('CACHE_PATH') is not None:
        # Set the cache path of librarys to the one in .env file
        print("Setting cache path to: " + os.getenv('CACHE_PATH'))
        os.environ["HF_HOME"] = os.getenv('CACHE_PATH') + '/transformers'
        os.environ["HF_DATASETS_CACHE"] = os.getenv('CACHE_PATH') + '/datasets'
        os.environ["TORCH_HOME"] = os.getenv('CACHE_PATH') + '/torch'
        os.environ["TFHUB_CACHE_DIR"] = os.getenv('CACHE_PATH') + '/tfhub'

def get_timestamp(string_output: bool = True) -> typing.Union[str, 'datetime']:
    """
    Get the current timestamp in the format YYYYMMDD_HHMMSS.
    
    Returns:
        str: The current timestamp.
    """
    from datetime import datetime
    now = datetime.now()
    if string_output:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return timestamp
    return now

def strip_ansi_codes(text):
    """Remove ANSI escape codes from text"""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def get_manim_version() -> str:
    """
    Get the installed Manim version.

    Returns:
        str: The Manim version.
    """
    import subprocess

    try:
        manim_version_output = subprocess.run(
            ["manim", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        manim_version = manim_version_output.stdout.strip()
        manim_version = strip_ansi_codes(manim_version).strip()
        return manim_version
    except subprocess.CalledProcessError as e:
        print("Error occurred while fetching Manim version:", e)
        return "Unknown"
