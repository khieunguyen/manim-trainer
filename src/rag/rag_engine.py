"""rag_engine.py"""
from typing import Dict, List, Tuple
from functools import lru_cache
from src.rag.api_inspector import ApiInspector
from src.rag.call_extractor import CallExtractor

class RAGEngine:
    """
    RAGEngine is responsible for extracting API calls from code and filtering out common built-in functions.
    It uses an ApiInspector to get live API signatures and docstrings, and a CallExtractor to parse the code.
    """

    def __init__(
            self,
            api_inspector: ApiInspector,
            call_extractor: CallExtractor,
            excluded_builtins_funcs: set = {
                "self",
                "print", 
                "len",
                "range", 
                "int", 
                "str", 
                "float", 
                "list", 
                "dict", 
                "set", 
                "tuple", 
                "bool", 
                "type", 
                "object",
                "np",
                "math",
                "random",
                # Manim specific built-ins
                "add",
                "remove",
                "clear",
                "play",
                "wait",
                "show",
                "hide",
            }
    ):
        """
        Initialize the RAGEngine with an ApiInspector and CallExtractor.
        This engine will use the ApiInspector to extract live API signatures and docstrings,
        and the CallExtractor to parse the code for function calls.

        Args:
            api_inspector (ApiInspector): The API inspector to use for extracting live API signatures and docstrings.
            call_extractor (CallExtractor): The call extractor to use for extracting function calls from code.
        """
        self.api_inspector = api_inspector
        self.call_extractor = call_extractor
        self.excluded_builtins_funcs = excluded_builtins_funcs

    def extract_api_calls_and_assignments(self, code: str, new_code: bool = True) -> Tuple[set, set]:
        """
        Parse the given source and return a set of all function call names.
        """
        if new_code:
            # Reset the call extractor to its initial state
            self.call_extractor.reset()
        self.call_extractor.visit_code(code)

        # Filter out common builtins
        sorted_calls = sorted(self._filter_builtins(self.call_extractor.calls, self.excluded_builtins_funcs))
        sorted_assignments = sorted(self._filter_builtins(self.call_extractor.variables, self.excluded_builtins_funcs))

        # Remove intersection between calls and assignments from the assignments
        sorted_assignments = sorted(set(sorted_assignments) - set(sorted_calls))

        return sorted_calls, sorted_assignments

    @staticmethod
    def _filter_builtins(calls: set, builtins_exclude: set) -> set:
        """
        Filter out common built-in functions from the extracted calls.
        Args:
            calls (set): A set of function call names extracted from the code.
            builtins_exclude (set): A set of built-in function names to exclude.

        Returns:
            set: A filtered set of function call names excluding built-ins.
        """

        return set(
            c for c in calls
            if c.split(".", 1)[0] not in builtins_exclude and c.split(".", 1)[-1] not in builtins_exclude
        )
    
    def extract_api_info(self, code: str, new_code: bool = True) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
        """
        Extract API calls from the code and get their signatures and docstrings.

        Args:
            code (str): The source code to parse.

        Returns:
            Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]: A tuple containing two dictionaries:
                - The first dictionary maps function call names to their signatures and docstrings.
                - The second dictionary maps variable assignments to their signatures and docstrings.
        """
        api_calls, api_assignments = self.extract_api_calls_and_assignments(code, new_code)

        api_calls_info = {call: self.api_inspector.get_api_info(call) for call in api_calls}
        api_assignments_info = {assignment: self.api_inspector.get_api_info(assignment) for assignment in api_assignments}

        return api_calls_info, api_assignments_info
    
    def get_formatted_api_info(self, code: str, new_code: bool = True) -> str:
        """
        Get formatted API information for the given code.

        Args:
            code (str): The source code to parse.

        Returns:
            str: A formatted string containing API information.
        """
        api_calls_info, api_assignments_info = self.extract_api_info(code, new_code)
        formatted_api_info = self._format_api_assignments_doc(api_assignments_info) + "\n" + self._format_api_calls_doc(api_calls_info)
        return formatted_api_info
    
    @staticmethod
    def _format_api_calls_doc(api_calls_info: Dict[str, Tuple[str, str]]) -> str:
        """
        Format the API information into a string for use in prompts.

        Args:
            api_calls_info (Dict[str, Tuple[str, str]]): A dictionary mapping function call names to their signatures and docstrings.

        Returns:
            str: A formatted string containing API information.
        """
        return "\n".join(
            f"- **{name}**: signature `{sig}` - {desc.split('Examples')[0].strip() if 'Examples' in desc else desc.strip()}\n"
            for name, (sig, desc) in api_calls_info.items()
        )
    
    @staticmethod
    def _format_api_assignments_doc(api_assignment_info: Dict[str, Tuple[str, str]]) -> str:
        """
        Format the API assignments information into a string for use in prompts.

        Args:
            api_assignment_info (Dict[str, Tuple[str, str]]): A dictionary mapping variable names to their signatures and docstrings.

        Returns:
            str: A formatted string containing API assignments information.
        """
        if not api_assignment_info:
            return ""
        
        formatted_lines = []
        for name, (sig, desc) in api_assignment_info.items():
            # Clean up description by removing examples section
            clean_desc = desc.split('Examples')[0].strip() if 'Examples' in desc else desc.strip()
            
            # Format differently based on signature type
            if sig.startswith("@property"):
                formatted_lines.append(f"- **{name}** {sig} - {clean_desc}")
            elif sig.startswith(":"):
                # For attributes with type info
                formatted_lines.append(f"- **{name}** (attribute{sig}) - {clean_desc}")
            else:
                # For other cases
                formatted_lines.append(f"- **{name}** (property/variable) - {clean_desc}")
        
        return "\n".join(formatted_lines) + "\n"
    
    