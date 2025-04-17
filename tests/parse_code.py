import ast

import ast
import pandas as pd

def extract_functions_with_children(file_path):
    """
    Extracts functions from a Python script, distinguishing parent and child functions.

    Args:
        file_path (str): Path to the Python script.

    Returns:
        pd.DataFrame: DataFrame with 'function_name', 'func_code', and 'child_func_dict'.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    functions_data = []

    for node in tree.body:  # Iterate over top-level elements
        if isinstance(node, ast.FunctionDef):  # Only process functions
            function_name = node.name
            function_code = ast.get_source_segment(source_code, node)

            # Extract child functions inside the parent function
            child_func_dict = {}
            for child in node.body:
                if isinstance(child, ast.FunctionDef):  # Check if it's a nested function
                    child_name = child.name
                    child_code = ast.get_source_segment(source_code, child)
                    child_func_dict[child_name] = child_code

            # Append to list
            functions_data.append({
                "function_name": function_name,
                "func_code": function_code,
                "child_func_dict": child_func_dict if child_func_dict else None  # Store None if no child functions
            })

    # Convert to DataFrame
    df = pd.DataFrame(functions_data, columns=["function_name", "func_code", "child_func_dict"])
    return df

def extract_functions_from_file(file_path):
    """
    Extracts all function names and their corresponding code as a dictionary from a given Python file.
    
    Args:
        file_path (str): Path to the Python file.

    Returns:
        dict: A dictionary where keys are function names and values are the corresponding function code.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):  # Extract function definitions
            function_name = node.name
            function_code = ast.get_source_segment(source_code, node)
            functions[function_name] = function_code

    return functions

# Example Usage:
file_path = "Coulomb_input.py"  # Replace with the actual file path
function_dict = extract_functions_from_file(file_path)

# Print extracted functions
for func_name, func_code in function_dict.items():
    print(f"Function: {func_name}\nCode:\n{func_code}\n{'-'*50}")

df_functions = extract_functions_with_children(file_path)

# Print DataFrame
print(df_functions)
