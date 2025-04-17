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

def generate_user_prompt(Coulomb_input, n, iteration, feedback):

    if iteration == 1:
        first_line = f"Please review the raw code, which currently works perfectly for cubic box configurations, and modify the functions to adjust the method so that it accommodates triclinic box configurations instead.\n"
    else:
        first_line = f"Please find the code generated in iteration {iteration}. You have {n-iteration} iterations remaining to complete the code for triclinic box configurations by changing the cubic box configurations.\n"

    input_message = (
        f"{first_line}"
        "\n\n"
        f"{Coulomb_input}"
        "\n\n"
        f"""
Context:
{context_content}

Primary task:

Modify the functions in the code above, considering the input format.

Incorporate any additional functions that might be required.

Make the necessary changes to the dataframes to accommodate the updated input format.

You may access the required columns from the given dataframes as parameters to the function to calculate the energies.

You may modify other parts of the code, except the compare_LJ_coulomb_energy, creating_dataframes function.

Focus more on Real, Fourier, Self, Intra Energies.

Strict Requirements:

Do not modify the function names or parameters.

The output should always include Python code for modifying the respective functions.

The creating_dataframes function should remain unchanged, as it defines the input dataframes.

The compare_LJ_coulomb_energy function should remain unchanged, as it controls the iteration loop.

Ensure that the print statements in the compare_LJ_coulomb_energy function remain exactly as they are, even as you make modifications to the other functions.

Iteration Notes: 

If multiple functions are provided to solve the task, work through them one by one in each iteration.

The iteration loop will continue until the desired outcome is achieved.

Output Format:

Produce a JSON output in the following format:
  "Reasoning": "",
  "Code": ""

Ensure that:
1. The JSON output is **properly formatted** with **double quotes** around keys.
2. The **Reasoning** section provides a concise and clear explanation of the approach taken.
3. The **Code** section contains well-structured, indented, and syntactically correct Python (or the required language).
4. The JSON output does **not** contain extraneous symbols such as triple backticks (` ``` `) or markdown indicators.
5. The output should be **fully valid JSON** and parseable using `json.loads()`.

Feedback Integration: 

When feedback is provided, ensure you proceed with the next steps based on the iteration results and the correct answers generated. Focus on the L2 score and relative errors to reduce errors.

"""
        "\n"
        f"{feedback}"
    )

    return input_message
