"""call_extractor.py
Extracts function calls from Python source code, including class methods and variable assignments to classes.
"""
import ast



class CallExtractor(ast.NodeVisitor):
    """
    Extracts function calls from Python source code, including class methods and variable assignments to classes.
    This visitor traverses the AST and collects function calls, handling class definitions,
    variable assignments to class instances, and method calls on those instances.
    """
    
    def __init__(self):
        self.calls = set()
        self.variables = set()
        self.var_types = {}  # e.g., b1 -> Brace
        self.current_class = None
        self.class_bases = {}  # e.g., Brace -> [Line, Dot]

    def visit_code(self, code):
        """
        Parse the given source code and return a set of all function call names.
        
        Args:
            code (str): The source code to parse.
        
        Returns:
            tuple: A tuple containing (function_calls_set, variables_set)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Warning: Syntax error in code: {e}. Could not parse the code to extract function calls.")
            return set(), set()
        self.visit(tree)
        return self.calls, self.variables
    
    def reset(self):
        """
        Reset the extractor to its initial state.
        This is useful if you want to reuse the same instance for multiple code snippets.
        """
        self.calls.clear()
        self.variables.clear()
        self.var_types.clear()
        self.current_class = None
        self.class_bases.clear()

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

        prev_class = self.current_class
        self.current_class = node.name
        self.class_bases[self.current_class] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.class_bases[self.current_class].append(base.id)
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_Assign(self, node):
        # Map variables to class types if they are class constructor calls
        if isinstance(node.value, ast.Call):
            call_func = node.value.func
            class_name = None
            if isinstance(call_func, ast.Name):
                class_name = call_func.id
            elif isinstance(call_func, ast.Attribute):
                class_name = self.get_full_name(call_func)
                if class_name:
                    class_name = class_name.split('.')[0]

            if class_name:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.var_types[target.id] = class_name
        
        # Handle attribute assignments like self.camera.background_color = RED
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                full_name = self.get_full_name(target)
                if full_name:
                    self.variables.add(full_name)  # Add to variables instead of calls

        # Also check the right-hand side for variable access
        self._extract_variable_access(node.value)
        self.generic_visit(node)

    def _extract_variable_access(self, node):
        """Extract variable access from expressions (right-hand side of assignments, function arguments, etc.)"""
        if isinstance(node, ast.Attribute):
            full_name = self.get_full_name(node)
            if full_name:
                self.variables.add(full_name)
        elif isinstance(node, ast.Name):
            # Simple variable access
            if node.id in self.var_types:
                self.variables.add(f"{self.var_types[node.id]}")
        elif hasattr(node, '__iter__') and not isinstance(node, str):
            # Recursively check nested structures
            for child in ast.iter_child_nodes(node):
                self._extract_variable_access(child)

    def visit_Call(self, node):
        full_name = self.get_full_name(node.func)
        if full_name:
            self.calls.add(full_name)
        
        # Check arguments for variable access
        for arg in node.args + [kw.value for kw in node.keywords]:
            self._extract_variable_access(arg)
            
        self.generic_visit(node)

    def get_full_name(self, node):
        if isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Attribute):
            value = node.value
            attr = node.attr

            # Handle self.method()
            if isinstance(value, ast.Name) and value.id == "self" and self.current_class:
                bases = self.class_bases.get(self.current_class, [])
                if bases:
                    return f"{bases[0]}.{attr}"
                else:
                    # If no bases are defined, set self to the current class
                    return f"self.{attr}"

            # Handle var.method(), where var was previously assigned a class
            elif isinstance(value, ast.Name):
                var_name = value.id
                if var_name in self.var_types:
                    return f"{self.var_types[var_name]}.{attr}"
                else:
                    return f"{var_name}.{attr}"

            # Handle chained calls like Line(...).set_color()
            elif isinstance(value, ast.Call):
                base = self.get_full_name(value.func)
                if base:
                    base_class = base.split('.')[0]
                    return f"{base_class}.{attr}"
                
            # Handle constant attributes like self.background_color
            elif isinstance(value, ast.Constant):
                return f"{value.value}.{attr}"
            
            # Handle other attributes
            elif isinstance(value, ast.Attribute):
                base = self.get_full_name(value)
                if base:
                    return f"{base}.{attr}"

        return None