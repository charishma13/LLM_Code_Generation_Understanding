import json
import re
import ast
import pandas as pd
import subprocess
import textwrap

def replace_functions(source_code, df_functions, global_variables):
    """
    Replaces function definitions in `source_code` with those in `df_functions`,
    ensuring that dependent (child) functions are also included immediately before their parent functions.
    The global variables are inserted once at the very top before all functions, but only those not already present.

    Args:
        source_code (str): The original script as a string.
        df_functions (pd.DataFrame): DataFrame containing function names, code, and dependencies.
        global_variables (dict): Dictionary of global variables to be included before functions.

    Returns:
        str: The modified source code with updated functions and global variables.
    """
    # List of function names to be skipped from replacement
    fixed_functions = {"extracting_positions", "create_dataframes", "compare_coulomb_energy"}

    tree = ast.parse(source_code)
    modified_code = source_code  # Start with the original code

    # A set of function names already in the source code (for adding missing functions)
    existing_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    # Extract the existing global variables from the source code
    existing_globals = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):  # Look for assignments (global variables are usually assigned outside functions)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    existing_globals.add(target.id)

    # First, process child functions
    added_functions = {}  # Keep track of where each function's code should be added

    for _, row in df_functions.iterrows():
        called_functions_dict = row["called_functions_dict"]  # Dictionary of dependencies
        function_name = row["function_name"]

        # Skip fixed functions
        if function_name in fixed_functions:
            continue

        if called_functions_dict:
            for child_name, child_code in called_functions_dict.items():
                if child_name not in existing_functions and child_name not in added_functions:
                    # Add child function immediately before its parent function
                    added_functions[child_name] = child_code

    # Now, replace the functions with their updated versions
    for _, row in df_functions.iterrows():
        function_name = row["function_name"]
        updated_func_code = row["func_code"]
        called_functions_dict = row["called_functions_dict"]  # Dictionary of dependencies

        # Skip fixed functions
        if function_name in fixed_functions:
            continue

        # Replace the function definition in the source code
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                old_func_code = ast.get_source_segment(source_code, node)

                # Insert the function code in place of the old one
                modified_code = modified_code.replace(old_func_code, updated_func_code)

                # Also insert child functions immediately before the parent function
                if called_functions_dict:
                    for child_name, child_code in called_functions_dict.items():
                        if child_name in added_functions:
                            # Insert the child function code right before the parent function
                            modified_code = modified_code.replace(updated_func_code, added_functions[child_name] + "\n\n" + updated_func_code)

    # Insert global variables once before all function definitions, but only if they are not already present
    if isinstance(global_variables, dict):
        # Format global variables as code, only adding those not already in the source code
        global_code = "\n".join(f"{var} = {repr(global_variables[var])}" for var in global_variables if var not in existing_globals)
    else:
        raise ValueError("global_variables must be a dictionary.")

    # Find the first function definition and insert the global variables before it
    function_start_index = None
    for idx, line in enumerate(modified_code.splitlines()):
        if line.strip().startswith("def "):  # Check for the first function definition
            function_start_index = idx
            break

    if function_start_index is not None:
        # Insert the global variables just before the first function definition
        modified_code_lines = modified_code.splitlines()
        modified_code_lines.insert(function_start_index, global_code)
        modified_code = "\n".join(modified_code_lines)

    return modified_code

def extract_functions_with_dependencies_from_text(text):
    # Parse JSON
    data = json.loads(text)

    # Extract code
    python_code = data.get("Code", "").strip()

    python_code = re.sub(r'^\n+|\n+$', '', python_code)

    # Remove any unwanted leading/trailing newlines or whitespace
    python_code = re.sub(r'^\n+|\n+$', '', python_code)

    python_code = textwrap.dedent(python_code)

    try:
        tree = ast.parse(python_code)
        print("Parsed AST:", tree)
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        
    functions_data = {}

    # First, extract all function definitions
    function_codes = {}
    function_docstrings = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_codes[node.name] = ast.get_source_segment(python_code, node)
            
            # Extract docstring using AST's method
            docstring = ast.get_docstring(node)
            if docstring:
                function_docstrings[node.name] = docstring
            else:
                function_docstrings[node.name] = None

    # Extract global variables (variables in the global scope)
    global_variables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):  # Global variable assignment
            for target in node.targets:
                if isinstance(target, ast.Name):  # Only name targets are considered variables
                    # Extract the value of the global variable
                    value = ast.literal_eval(node.value) if isinstance(node.value, ast.Constant) else None
                    global_variables[target.id] = value

    # Now, extract function calls and variables within each function
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            function_code = function_codes[function_name]
            function_docstring = function_docstrings[function_name]

            # Find all function calls inside this function
            called_functions_dict = {}
            for child in ast.walk(node):  # Traverse the function body
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    called_func_name = child.func.id  # Get called function name
                    if called_func_name in function_codes:  # Only store if it's a defined function
                        called_functions_dict[called_func_name] = function_codes[called_func_name]

            # Store function details
            functions_data[function_name] = {
                "func_code": function_code,
                "docstring": function_docstring,
                "called_functions_dict": called_functions_dict if called_functions_dict else None
            }

    # Convert to DataFrame for functions
    df = pd.DataFrame([
        {
            "function_name": key,
            "func_code": value["func_code"],
            "docstring": value["docstring"],
            "called_functions_dict": value["called_functions_dict"]
        }
        for key, value in functions_data.items()
    ])

    return df, global_variables

def execute_code(code: str) -> dict:
    """
    Executes a Python code string and captures the output, errors, and return code.

    Args:
        code (str): The Python code to execute.

    Returns:
        dict: Dictionary with 'stdout', 'stderr', and 'return_code'.
    """
    try:
        process = subprocess.Popen(
            ["/Users/pulicharishma/anaconda3/bin/python", "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output, errors = process.communicate()
        
        return {
            "stdout": output.strip().split("\n") if output else [],
            "stderr": errors.strip().split("\n") if errors else [],
            "return_code": process.returncode
        }
    except Exception as e:
        return {"error": str(e)}


