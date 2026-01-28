"""preprocessor.py: Preprocessor for PEFT models."""

import pandas as pd
from typing import List, Dict, Literal
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

from src.rag.api_inspector import ApiInspector
from src.inference.inference_engine import InferenceEngine
from src.utils.prompt_template import PromptTemplate
from src.utils.constants import Constants
from src.rag.rag_engine import RAGEngine
from src.utils import utils


# # Constants for special tokens to be added when more data is available
# TOK_START_INST = "<|INST|>"
# TOK_END_INST = "<|ENDINST|>"
# TOK_START_RESPONSE = "<|START_RESPONSE|>"
# TOK_END_RESPONSE = "<|END_RESPONSE|>"

class Preprocessor:

    """Preprocessor for PEFT models."""

    def __init__(self, tokenizer, is_sft: bool = True):
        """
        Initialize the preprocessor with the specified tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for preprocessing.
            is_sft (bool): Whether to use SFT (Supervised Fine-Tuning) or not (Pretraining type Fine-Tuning).
                Default is True.
        """
        self.tokenizer = tokenizer
        self.is_sft = is_sft

        self.data_collator = None
        if is_sft:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                return_tensors="pt"
            )
            # ToDO: Add special tokens to the tokenizer when more data is available
            # self.tokenizer.add_special_tokens({
            #     "additional_special_tokens": [
            #         TOK_START_INST,
            #         TOK_END_INST,
            #         TOK_START_RESPONSE,
            #         TOK_END_RESPONSE
            #     ]
            # })

            # # Resize the tokenizer to accommodate the new special tokens
            # inference_engine.base_model.resize_token_embeddings(len(self.tokenizer))
        else:
            raise ValueError("Preprocessor currently only supports SFT (Supervised Fine-Tuning) mode.")
        print(f"Preprocessor initialized with SFT mode: {self.is_sft}")
        print(f"Data collator initialized...")


    def preprocess_sample_for_sft(
            self,
            prompt_data: dict,
            response_data: dict, 
            max_prompt_length: int,
            prompt_template: PromptTemplate, 
            response_template: PromptTemplate
    ):
        """
        Preprocess a single sample for SFT (Supervised Fine-Tuning).

        Args:
            prompt_data (dict): The data for the prompt.
            response_data (dict): The data for the response.
            max_prompt_length (int): The maximum length of the prompt.
            prompt_template (PromptTemplate): The template for the prompt.
            response_template (PromptTemplate): The template for the response.

        Returns:
            dict: A dictionary containing the input IDs, attention mask, and labels for the sample.
        """

        sft_prompt, sft_full_prompt = self._format_prompt(
            prompt_data=prompt_data,
            response_data=response_data,
            prompt_template=prompt_template,
            response_template=response_template,
            eos_token=self.tokenizer.eos_token
        )
        # print(f"Formatted SFT prompt: {sft_prompt}")
        # print(f"Formatted SFT full prompt: {sft_full_prompt}")
        toks = self.tokenizer(
            sft_full_prompt,
            truncation=True,
            max_length=max_prompt_length,
            padding="max_length",
            return_attention_mask=True
        )
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        prompt_len = len(
            self.tokenizer(sft_prompt, add_special_tokens=False)["input_ids"])
        labels = [
            -100 if i < prompt_len else token_id
            for i, token_id in enumerate(input_ids)
        ]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    
    def preprocess_dataset(
            self,
            dataset_df: pd.DataFrame,
            x_names: List[str], y_names: List[str],
            prompt_template: str,
            response_template: str,
            max_prompt_length: int,
            prepared_hf_dataset: Dataset = None,
        ) -> Dataset:
        """
        Preprocess a dataset for SFT (Supervised Fine-Tuning).

        Args:
            dataset_df (pd.DataFrame): The dataset to preprocess. If `prepared_hf_dataset` is provided, this will be ignored.
            x_names (List[str]): The names of the input columns.
            y_names (List[str]): The names of the output columns.
            prompt_template (str): The template for the prompt.
            response_template (str): The template for the response.
            max_prompt_length (int): The maximum length of the prompt.
            prepared_hf_dataset (Dataset, optional): A pre-prepared Hugging Face dataset. Defaults to None. If provided, `dataset_df` will be ignored.
        
        Returns:
            Dataset: The preprocessed Hugging Face dataset.
        """
        print("Preprocessing dataset for SFT...")
        
        x_names = [x_name.lower().replace(" ", "_") for x_name in x_names]
        y_names = [y_name.lower().replace(" ", "_") for y_name in y_names]
        
        if prepared_hf_dataset is None:
            if dataset_df is None:
                raise ValueError("Either `dataset_df` or `prepared_hf_dataset` must be provided.")
            prepared_hf_dataset = self._prepare_hf_dataset(
                pandas_dataset_df=dataset_df,
                x_names=x_names,
                y_names=y_names
            )
        else:
            if dataset_df is not None:
                print("Warning: `dataset_df` is provided but `prepared_hf_dataset` is already set. Ignoring `dataset_df`.")
            else:
                print("Using pre-prepared Hugging Face dataset for preprocessing.")
            prepared_hf_dataset = prepared_hf_dataset.rename_columns(
                {col: col.lower().replace(" ", "_") for col in prepared_hf_dataset.column_names}
            )

        hf_dataset_preprocessed = prepared_hf_dataset.map(
            lambda x: self.preprocess_sample_for_sft(
                prompt_data={x_name: x[x_name] for x_name in x_names},
                response_data={y_name: x[y_name] for y_name in y_names},
                max_prompt_length=max_prompt_length,
                prompt_template=prompt_template,
                response_template=response_template
            ),
            batched=False,
            remove_columns=prepared_hf_dataset.column_names
        )
        print("Dataset preprocessing completed.")
        return hf_dataset_preprocessed


    def get_data_collator(self):
        """
        Get the data collator for the preprocessor.
        
        Returns:
            DataCollatorForLanguageModeling: The data collator.
        """
        return self.data_collator
    
    def _format_prompt(
            self,
            prompt_data: dict,
            response_data: dict,
            prompt_template: str, 
            response_template: str, 
            eos_token
    ):
        sft_prompt = prompt_template.format(**prompt_data) + eos_token
        sft_full_prompt = sft_prompt + \
            response_template.format(**response_data) + eos_token
        return sft_prompt, sft_full_prompt
    
    def _prepare_hf_dataset(
            self,
            pandas_dataset_df: pd.DataFrame,
            x_names: List[str], 
            y_names: List[str]
        ) -> Dataset:
        """
        Prepare a Hugging Face dataset from a pandas DataFrame.

        Args:
            pandas_dataset_df (pd.DataFrame): The dataset to prepare.
            x_names (list[str]): The names of the input columns.
            y_names (list[str]): The names of the output columns.

        Returns:
            Dataset: The prepared Hugging Face dataset.
        """
        hf_dataset = Dataset.from_pandas(pandas_dataset_df)

        # Rename columns of the hf_dataset to have lowercase names with underscores
        hf_dataset = hf_dataset.rename_columns(
            {col: col.lower().replace(" ", "_") for col in hf_dataset.column_names}
        )

        x_names = [x_name.lower().replace(" ", "_") for x_name in x_names]
        y_names = [y_name.lower().replace(" ", "_") for y_name in y_names]

        # Check if all x_names and y_names are present in the dataset
        missing_x_names = [x_name for x_name in x_names if x_name not in hf_dataset.column_names]
        missing_y_names = [y_name for y_name in y_names if y_name not in hf_dataset.column_names]
        if missing_x_names or missing_y_names:
            raise ValueError(f"Missing columns in dataset: {missing_x_names + missing_y_names}")
        
        return hf_dataset
    
    def create_rag_dataset(
            self,
            dataset_df: pd.DataFrame,
            textual_script_col: str,
            initial_code_col: str,
            api_info_col: str,
            x_names: List[str],
            y_names: List[str],
            inference_engine: InferenceEngine,
            rag_engine: RAGEngine,
            api_info_extraction_source: Literal['initial_code', 'output_code'] = 'initial_code'
        ) -> Dataset:
        """
        Create a RAG dataset from a pandas DataFrame. Or if a Hugging Face dataset is already prepared, use that.

        Args:
            dataset_df (pd.DataFrame): The dataset to prepare.
            x_names (List[str]): The names of the input columns.
            inference_engine (InferenceEngine): The inference engine to use for RAG.
            rag_engine (RAGEngine): The RAG engine to use for extracting API calls.

        Returns:
            Dataset: The prepared RAG dataset.
        """
        textual_script_col = textual_script_col.lower().replace(" ", "_")
        initial_code_col = initial_code_col.lower().replace(" ", "_")
        api_info_col = api_info_col.lower().replace(" ", "_")

        print("Creating RAG dataset...")
        hf_dataset = self._prepare_hf_dataset(
            pandas_dataset_df=dataset_df,
            x_names=x_names,
            y_names=y_names
        )
        
        print("Generating initial ManimCE code for RAG dataset...")
        rag_dataset = hf_dataset.map(
            lambda x: {
                initial_code_col: utils.extract_manim_code_from_llm_response(
                    inference_engine.generate_manim_code(
                        textual_script=x[textual_script_col]
                    ).generated_code, 
                    return_multiple_code_blocks=False, 
                    select_index=-1
                )
            },
            batched=False
        )

        print("Extracting API calls and information for RAG dataset...")
        api_extraction_code_col: str = initial_code_col
        if api_info_extraction_source == "output_code":
            initial_code_col = y_names[0]
        elif api_info_extraction_source != "initial_code":
            raise ValueError("api_info_extraction_source must be either 'initial_code' or 'output_code'.")

        rag_dataset = rag_dataset.map(
            lambda x: {
                api_info_col: rag_engine.get_formatted_api_info(
                    code=x[api_extraction_code_col]
                )
            },
            batched=False
        )

        print("Created RAG dataset with the following columns: ", rag_dataset.column_names)

        return rag_dataset

