"""api_inspector.py
Extracts live API signatures and docstrings from a given package.
"""
import ast
import inspect
from typing import List, Tuple, Any
from functools import lru_cache

from src.utils.prompt_template import PromptTemplate


class ApiInspector:
    """
    Introspects installed Manim (or any) package to extract
    live API signatures and docstrings without external files.
    """
    def __init__(self, package):
        self.package = package

    #@lru_cache(maxsize=128)
    def get_api_info(self, name: str) -> Tuple[str, str]:
        """
        Given a dot-delimited attribute path (e.g., "Circle.shift" or "Scene.camera.background_color"),
        return its signature and docstring description.

        Args:
            name (str): The dot-delimited attribute path.
        Returns:
            Tuple[str, str]: A tuple containing the signature and docstring.

        """
        try:
            parts = name.split(".")
            obj = self.package
            parent_obj = None
            
            # Navigate through the attribute path
            for i, part in enumerate(parts):
                parent_obj = obj
                obj = getattr(obj, part)

                # If we hit a property object, get the class of that object
                if self._is_property(obj) and i < len(parts) - 1:
                    obj = self._get_property_type(parent_obj, part)
                            
            # Handle the final object
            return self._extract_info_from_object(obj, name)
            
        except (AttributeError, ValueError, TypeError) as e:
            return "()", PromptTemplate.NO_DOCUMENTATION_PROMPT_PART.format(invalid_param=name)

    def _is_property(self, obj: Any) -> bool:
        """Check if an attribute is a property object."""
        try:
            return isinstance(obj, property) or any(
                isinstance(getattr(obj.__class__, attr, None), property)
                for attr in dir(obj.__class__)
            )
        except Exception:
            # If we can't determine if it's a property, assume it's not 
            return False

    def _get_property_type(self, parent_obj: Any, property_name: str) -> Any:
        """Get the type of a property from its parent object."""
        prop = getattr(parent_obj, property_name, None)
        if isinstance(prop, property):
            # If it's a property, return its origin type
            return prop.fget.__annotations__.get('return', None)
        return None

    def _extract_info_from_object(self, obj: Any, name: str) -> Tuple[str, str]:
        """Extract signature and documentation from an object."""
        sig = ""
        doc = ""
        
        try:
            # Try to get signature for callable objects
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (ValueError, TypeError):
                    # For built-in functions or methods without signatures
                    sig = "()"
            else:
                # For non-callable objects (attributes, constants, etc.)
                obj_type = type(obj).__name__
                if hasattr(obj, '__class__'):
                    sig = f": {obj_type}"
                else:
                    sig = f": {obj_type}"
            
            # Get documentation
            doc = inspect.getdoc(obj) or ""
            
            # If no documentation, try to get it from the class
            if not doc and hasattr(obj, '__class__'):
                doc = inspect.getdoc(obj.__class__) or ""
            
            # If still no documentation, provide type information
            if not doc:
                if callable(obj):
                    doc = f"Callable object of type {type(obj).__name__}"
                else:
                    doc = f"Attribute of type {type(obj).__name__}"
                    if hasattr(obj, '__module__'):
                        doc += f" from module {obj.__module__}"
                        
        except Exception:
            sig = "()"
            doc = f"Object of type {type(obj).__name__}"
        
        # Return formatted results
        return (
            sig if sig else "()",
            doc if doc else PromptTemplate.NO_DOCUMENTATION_PROMPT_PART.format(invalid_param=name)
        )
