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
from iteration_execution import extract_functions_with_dependencies_from_text, replace_functions  # assuming defined elsewhere

from datetime import datetime

# Get the current time in the format you want
current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# Dynamically create the file name to store results
if llm_config.is_reasoning:
    csv_file = f'../triclinic_outputs/results_triclinic_logs_{llm_config.model_name}_{llm_config.reasoning_factor}_{current_time}.csv'
else:
    csv_file = f'../triclinic_outputs/results_triclinic__logs_{llm_config.model_name}_{llm_config.temperature}_{current_time}.csv'

print(csv_file)

## Load existing CSV or create a new DataFrame
if os.path.exists(csv_file):
    results_logs = pd.read_csv(csv_file)
else:
    header = ['Run_ID', 'Iteration', 'Timestamp', 'model_name', 'reasoning_factor','context_type','Feedback', 'Human_Message', 'AI_Message', 'Updated_Coulomb_input']
    results_logs = pd.DataFrame(columns=header)

# Function to get the next run_id
def get_next_run_id(result_logs):
    if not results_logs.empty and "Run_ID" in results_logs.columns:
        return int(results_logs["Run_ID"].max()) + 1 if pd.notna(results_logs["Run_ID"].max()) else 1
    return 1

# Initialize variables
run_id = get_next_run_id(results_logs)  # Get next run_id
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
with open("Trappe_zeo_spc_e_code.py", "r") as file:
    Coulomb_input = file.read()

while iteration <= n:  # Ensure it runs for max_iterations
    print(f"================================ Iteration {iteration} ================================\n")
      
    # Generate the user input message only for the first iteration
    input_message = generate_user_prompt(Coulomb_input, n, iteration, feedback)

    # Process LLM response with the graph pipeline
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()
    # Assuming 'step' contains the necessary context and text
    text = step["messages"][-1].content

    error_cnt = 0

    # Extract functions and their dependencies from the text
    print("Extracting the functions from the code")
    df_functions, global_variables  = extract_functions_with_dependencies_from_text(text)

    # Check if there are any function names extracted in the DataFrame
    if df_functions is not None and (df_functions.empty or df_functions['function_name'].isnull().all()):
        feedback = "No Python code found. Please review the code more clearly and provide a Python code after changes."
    else:
        # Replace functions in the Coulomb_input (or the input script)
        Coulomb_input = replace_functions(Coulomb_input, df_functions, global_variables)
        print("Functions successfully replaced.")

        print("executing the code")
        result = execute_code(Coulomb_input)

        print("feedback updation",iteration,n)
        # Handle feedback and determine next steps
        feedback, iteration, n, error_cnt = handle_feedback(result, iteration, n)
        result_note = "Result from the code generated:\n"
        feedback = result_note + feedback

        print("patience updation",patience)
        if llm_config.stop_on_repeat and last_output and result['stdout'] and last_output == result['stdout']:
            print(f"Output repeated. Remaining patience: {patience-1}")

            if patience == 1:
                Warning = (
                    "Warning: You have 1 iteration left before the loop will stop due to repeated output. "
                    "This is your last chance to refine your solution. Take a moment to understand the problem thoroughly and "
                    "ensure you're producing the best possible output. We believe you can do it!"
                )
                feedback += Warning

            if patience == 0:
                user_input = input("The output has repeated several times. Do you want to stop the loop? (yes to stop, no to continue): ").strip().lower()
                
                if user_input == "yes":
                    print("Exiting the loop as per user request.")
                    break
                else:
                    Warning = f"""Warning: Although you've exceeded the patience, we are continuing the loop for {llm_config.max_iterations} iterations to complete the task as we trust you to give it your best shot!"""
                    feedback += Warning
                    patience = llm_config.patience
                    
            patience -= 1
        else:
            patience = llm_config.patience

        last_output = result['stdout']
    
    # Append data to DataFrame
    new_row = {
        'Run_ID': run_id,
        'Iteration': iteration-1,
        'Timestamp': current_time,
        'model_name': llm_config.model_name,
        'reasoning_factor': llm_config.reasoning_factor if llm_config.is_reasoning else llm_config.temperature,
        'context_type': llm_config.context_type,
        'Feedback': feedback,
        'Human_Message': input_message,
        'AI_Message': text,
        'Updated_Coulomb_input': Coulomb_input
    }
    
    print("storing results")  
    # Append data directly to the CSV file using csv.writer
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(new_row.values())  # Append the new row data

    if error_cnt == 3:
        print("Encountering error from past 3 runs")
        break

    if feedback == "LLM did a great job! Exiting from the loop":
        break

print(f"Results saved to {csv_file}")