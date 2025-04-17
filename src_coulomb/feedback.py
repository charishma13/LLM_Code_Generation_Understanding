import re

def handle_feedback(result, i, n):
    # Check if stdout exists and has enough entries
    print("STDOUT Output:", result['stdout'])  # Debugging print statement

    if result['stdout'] and len(result['stdout']) >= 2:
        try:
            incorrect_match = re.search(r"\d+", result['stdout'][-1])
            correct_match = re.search(r"\d+", result['stdout'][-2])

            incorrect = int(incorrect_match.group(0)) if incorrect_match else 0
            correct = int(correct_match.group(0)) if correct_match else 0
        except (IndexError, AttributeError, ValueError) as e:
            raise ValueError("Error extracting test results from output. Please check the format of the stdout.")
    else:
        # Masking the file paths or private information from the stderr message
        feedback = (
            "Your code resulted in the following error:\n"
            "Error\n"
            + '\n'.join([re.sub(r'/\S+', '[MASKED_PATH]', line) for line in result['stderr']])  # Masking file paths
            + "\nReview the specific part or function causing the error in the next iteration and correct it.\n"
        )
        
        # Handle additional iteration request if needed
        if i + 1 == n - 1:
            try:
                additional_iterations = int(input("How many iterations do you want to add? "))
                n += additional_iterations  # Extend the loop
            except ValueError:
                print("Invalid input! Please enter an integer.")
        
        return feedback, i + 1, n  # Move to next iteration safely

    # Check if all tests passed
    if correct == 6:
        print(f"In Iteration {i}, LLM did a great job! Exiting from the loop")
        feedback = "LLM did a great job! Exiting from the loop"
        return feedback, i, n
    
    feedback = (
    "Your code hasn't yet passed all of the NIST benchmark tests. "
    "Here's the current progress:\n\n"
    + '\n'.join(result['stdout']) + "\n\n"
    + "Out of the tests, " + str(correct) + " answers have been correct, with " + str(incorrect) + " remaining. \n"
    + ("Focus on getting the code to align with the benchmark for at least one part! Continue refining the code step by step until it passes all the tests. "
       if incorrect == '4' 
       else "Keep going – you're getting closer! Continue refining the code step by step until it passes all the tests.") 
    + "\n\n"
    + "Complete each function sequentially, incorporating feedback to optimize efficiency and align with NIST benchmarks. "
    + "Aim to match the benchmark values as closely as possible with each attempt, ensuring visible numerical improvements. "
    + "Refer to the context and revise accordingly.")

    # Handle additional iteration request if needed
    if i + 1 == n - 1:
        try:
            additional_iterations = int(input("How many iterations do you want to add? "))
            n += additional_iterations  # Extend the loop
        except ValueError:
            print("Invalid input! Please enter an integer.")
    
    return feedback, i + 1, n  # Increment iteration safely
