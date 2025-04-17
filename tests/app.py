import config
import os
import csv
import pandas as pd

from user_prompt import generate_user_prompt
from feedback import handle_feedback
from datetime import datetime
from LLM_config import llm_config  # Import the LLMTestConfig class
from rag_pipeline import graph
from rag_pipeline import config  # assuming `graph` is properly imported from wherever you define it
from iteration_execution import execute_code  # assuming `execute_code` is defined elsewhere
from iteration_execution import extract_parent_with_children, replace_function_by_index  # assuming defined elsewhere

# Dynamically create the file name to store results
if llm_config.is_reasoning:
    csv_file = f'../outputs/results_logs_{llm_config.model_name}_{llm_config.reasoning_factor}.csv'
else:
    csv_file = f'../outputs/results_logs_{llm_config.model_name}_{llm_config.temperature}.csv'
## Load existing CSV or create a new DataFrame
if os.path.exists(csv_file):
    results_logs = pd.read_csv(csv_file)
else:
    header = ['Run_ID', 'Iteration', 'Timestamp', 'model_name', 'reasoning_factor','context_type','Feedback', 'AI_Message', 'Updated_Coulomb_input']
    results_logs = pd.DataFrame(columns=header)

# Function to get the next run_id
def get_next_run_id(result_logs):
    if not results_logs.empty and "Run_ID" in results_logs.columns:
        return int(results_logs["Run_ID"].max()) + 1 if pd.notna(results_logs["Run_ID"].max()) else 1
    return 1

# Initialize variables
run_id = get_next_run_id(results_logs)  # Get next run_id
Iteration = {}
feedback = ""
iteration = 1
n = llm_config.max_iterations
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

while iteration <= n:  # Ensure it runs for max_iterations
    print(f"================================ Iteration {iteration} ================================\n")
    
    # Generate the user input message only for the first iteration
    input_message = generate_user_prompt(Coulomb_input, feedback)

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
    
        print("executing the code")
        result = execute_code(Coulomb_input)

        print("patience updation")
        # Check if the output has repeated
        if llm_config.stop_on_repeat and last_output and result['stdout'] and last_output == result['stdout'][-1]:
            patience -= 1  # Reduce patience
            print(f"Output repeated. Remaining patience: {patience}")

            if patience <= 0:
                print("The output has repeated for several iterations. The loop has stopped due to patience reaching 0.")
                break  # Exit the loop
        else:
            patience = llm_config.patience  # Reset patience if output changes""

        print("feedback updation")
        # Handle feedback and determine next steps
        feedback, iteration, n, last_output = handle_feedback(result, iteration, n, Iteration, last_output)
    
    # Append data to DataFrame
    new_row = {
        'Run_ID': run_id,
        'Iteration': iteration,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': llm_config.model_name,
        'reasoning_factor': llm_config.reasoning_factor,
        'context_type': llm_config.context_type,
        'Feedback': feedback,
        'AI_Message': text,
        'Updated_Coulomb_input': Coulomb_input
    }
    
    print("storing results")  
    results_logs = pd.concat([results_logs, pd.DataFrame([new_row])], ignore_index=True)

    if feedback == "LLM did a great job! Exiting from the loop":
        break

results_logs.to_csv(csv_file, index=False)
print(f"Results saved to {csv_file}")
