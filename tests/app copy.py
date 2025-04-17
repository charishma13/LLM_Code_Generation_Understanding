import config
import os
import csv

from user_prompt import generate_user_prompt
from feedback import handle_feedback
from LLM_config import llm_config  # Import the LLMTestConfig class
from rag_pipeline import graph
from rag_pipeline import config  # assuming `graph` is properly imported from wherever you define it
from iteration_execution import execute_code  # assuming `execute_code` is defined elsewhere
from iteration_execution import extract_parent_with_children, replace_function_by_index  # assuming defined elsewhere

# File to store results
csv_file = 'feedback_results.csv'
header = ['Run_ID', 'Iteration', 'llm_config', 'Feedback', 'AI_Message', 'Updated_Coulomb_input']

# Get the next run_id from CSV
def get_next_run_id():
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            last_run_id = 0
            for row in reader:
                run_id = int(row[0])  # Extract the numeric Run_ID
                last_run_id = max(last_run_id, run_id)
            return last_run_id + 1
    return 1

# Initialize variables
run_id = get_next_run_id()  # Get next run_id
Iteration = {}
feedback = ""
n = llm_config.max_iterations  # Set iterations based on LLMTestConfig
i = 1
last_output = ""
patience = llm_config.patience  # Initialize patience from LLM config

# Check if the CSV file exists, if not, create it and write the header
try:
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        pass  # If file exists, do nothing
except FileNotFoundError:
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header if the file does not exist

# Read Coulomb_input for the first time
with open("Coulomb_input.py", "r") as file:
    Coulomb_input = file.read()

while i <= n:  # Ensure it runs for max_iterations
    print(f"================================ Iteration {i} ================================\n")
    
    # Generate the user input message only for the first iteration
    if i == 1:
        input_message = generate_user_prompt(Coulomb_input, feedback)
        print(input_message)
    
    # Process LLM response with the graph pipeline
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()
    text = step["messages"][-1].content

    functions_filled = {}

    function_names = [
        "compute_real_energies",
        "compute_fourier_energies",
        "compute_intra_energies",
        "compute_self_energies"
    ]

    for function_name in function_names:
        extracted_function = extract_parent_with_children(text, function_name)
        if extracted_function:
            functions_filled[function_name] = extracted_function

    if not functions_filled:
        feedback = "No Python code found. Please review the code more clearly and provide a Python code after changes."
    else:
        for function_name, function_code in functions_filled.items():
            Coulomb_input = replace_function_by_index(Coulomb_input, function_code, function_name)

    result = execute_code(Coulomb_input)

    # Check if the output has repeated
    if last_output and last_output == result['stdout'][-1]:
        patience -= 1  # Reduce patience
        print(f"Output repeated. Remaining patience: {patience}")

        if patience <= 0:
            feedback = (
                "The output has repeated for several iterations. The loop has stopped due to patience reaching 0."
            )
            print(feedback)
            break  # Exit the loop
    else:
        patience = llm_config.patience  # Reset patience if output changes

    # Handle feedback and determine next steps
    feedback, i, n, last_output = handle_feedback(result, feedback, i, n, Iteration, last_output, patience)
    
    # Append the feedback, text, and updated Coulomb_input for this iteration to the CSV file
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([run_id, i, llm_config, feedback, text, Coulomb_input])
    
    if feedback == "LLM did a great job! Exiting from the loop":
        break
