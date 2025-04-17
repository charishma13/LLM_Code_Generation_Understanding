# user_prompt.py
from LLM_config import llm_config  # Import the LLMTestConfig class

def read_context_file():
    """Reads the content of the context file if provided."""
    if llm_config.context_file:
        try:
            with open(llm_config.context_file, "r") as file:
                return file.read()
        except FileNotFoundError:
            print(f"Warning: Context file '{llm_config.context_file}' not found.")
            return "Context file not found."
        except Exception as e:
            print(f"Error reading context file: {e}")
            return "Error reading context file."
    return ""

# Read context file content
context_content = read_context_file()

def generate_user_prompt(Coulomb_input, feedback):

    input_message = (
        f"{Coulomb_input}"
        "\n\n"
        f"""Please review the provided code and implement the missing functions needed to compute Coulombic contributions using the traditional Ewald Summation Method.

Produce a JSON output in the following format:
  "Reasoning": "#---Reasoning Text---#",
  "Code": "#--Modified Code--#"

Ensure that:
1. The JSON output is **properly formatted** with **double quotes** around keys.
2. The **Reasoning** section provides a concise and clear explanation of the approach taken.
3. The **Code** section contains well-structured, indented, and syntactically correct Python (or the required language).
4. The JSON output does **not** contain extraneous symbols such as triple backticks (` ``` `) or markdown indicators.
5. The output should be **fully valid JSON** and parseable using `json.loads()`.

Context:
{context_content}

Description:

The terms on the right-hand side of the equality to be computed are:

1) The real-space term Ereal,

2) The Fourier-space term, Efourier,

3) The self-correction term Eself,

4) The intramolecular term Eintra.

Note:

The output should always contain a python code to modify these functions not other functions

If multiple functions are provided for solving, complete them one by one in each iteration. Ensure that the print statements remain unchanged, even as you incorporate the necessary modifications.

Write your code in the section marked #--- Complete this code ---# and include any related functions as needed. However, do not modify other parts of the code.

You may access the required columns from the given dataframes as parameters to the function to calculate the energies.

Strictly remember: Do not modify the function names, parameters, and `compare_coulomb_energy` function, as it manages the iteration loop.

The automated iteration loop allows you to run iterations until the desired result is obtained.

Additionally, if feedback is provided, please refer to the user's feedback for the next steps.
"""
        "\n\n"
        f"{feedback}"
    )
    return input_message
