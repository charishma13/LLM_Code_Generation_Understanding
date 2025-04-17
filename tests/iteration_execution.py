import subprocess
import re
import sys

from concurrent.futures import ThreadPoolExecutor

def execute_parallel(codes: list):
    with ThreadPoolExecutor() as executor:
        results = executor.map(execute_code, codes)
    return list(results)

def execute_code(code: str) -> dict:
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


# Function to extract the parent function and its children
def extract_parent_with_children(code, function_name):
    parent_function_pattern = r"(def " + re.escape(function_name) + r".*?return.*?)(?=\n|$)"
    parent_match = re.search(parent_function_pattern, code, re.DOTALL)

    if parent_match:
        parent_function_body = parent_match.group(0)
        function_call_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\s?\(.*?\))"
        function_calls = re.findall(function_call_pattern, parent_function_body)

        related_functions = set()
        for call in function_calls:
            func_name = call.split('(')[0].strip()
            related_functions.add(func_name)

        related_functions_body = ''
        for func_name in related_functions:
            child_function_pattern = r"(def " + re.escape(func_name) + r".*?return.*?)(?=\n|$)"
            child_function_match = re.search(child_function_pattern, code, re.DOTALL)

            if child_function_match:
                related_functions_body += child_function_match.group(0) + "\n"

        return related_functions_body
    else:
        return None

def replace_function_by_index(Coulomb_input, function_body, function_name):
    pattern = r"(def " + re.escape(function_name) + r".*?return.*?)(?=\n|$)"
    updated_code = re.sub(pattern, function_body, Coulomb_input, flags=re.DOTALL)
    return updated_code

