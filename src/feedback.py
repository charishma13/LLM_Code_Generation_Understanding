import re

def handle_feedback(result, i, n):
    # Check if stdout exists and has enough entries
    print("STDOUT Output:", result['stdout'])  # Debugging print statement

    error_cnt = 0
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
        feedback = """
            Your code resulted in the following error from {} iterations. The threshold is 3. Resolve the error before the 3rd attempt:
            Error
            {} 
            Review the specific part or function causing the error in the next iteration and correct it.
            """.format(error_cnt, '\n'.join([re.sub(r'/\S+', '[MASKED_PATH]', line) for line in result['stderr']]))

        error_cnt += 1
        
        # Handle additional iteration request if needed
        if i == n:
            try:
                additional_iterations = int(input("How many iterations do you want to add? "))
                n += additional_iterations  # Extend the loop
            except ValueError:
                print("Invalid input! Please enter an integer.")
        
        return feedback, i + 1, n, error_cnt  # Move to next iteration safely

    # Check if all tests passed
    if correct == 4:
        print(f"In Iteration {i}, LLM did a great job! Exiting from the loop.")
        print('\n'.join(result['stdout']))
        feedback = "LLM did a great job! Exiting from the loop"
        return feedback, i, n
    # Focus on getting the code to align with the benchmark for at least one part! Continue refining the code step by step until it passes all the tests. 
    feedback = (
    "Your code hasn't yet passed all of the NIST benchmark tests. "
    "Here's the current progress:\n"
    + '\n'.join(result['stdout']) + "\n\n"
    + "Out of the tests, " + str(correct) + " answers have been correct, with " + str(incorrect) + " remaining. \n"
    + ("Start on getting the code to align with the benchmark for at least one part, such as Dispersion! Continue refining the code step by step until it passes all the tests. For example, after Dispersion, proceed to LRC, then move on to Real, Fourier, Self, and Intra. "
       if correct != 6 
       else "Keep going – you're getting closer! Continue refining the code step by step until it passes all the tests.") 
    + "\n\nTo improve further:\n"
    #+ "Work through each function systematically, incorporating feedback to optimize efficiency and align with NIST benchmarks. \n"
    + "If a function's output aligns with the NIST benchmark values, move on to the next function.\n"
    #+ "Aim to match the benchmark values as closely as possible with each attempt, ensuring visible numerical improvements. \n"
    + "Compare your results with the benchmark values and aim for visible numerical improvements.\n"
    #+ "Refer to the context and revise accordingly.\n")
    + "Refer to the context to make targeted refinements.")

    # Handle additional iteration request if needed
    if i == n:
        try:
            additional_iterations = int(input("How many iterations do you want to add? "))
            n += additional_iterations  # Extend the loop
        except ValueError:
            print("Invalid input! Please enter an integer.")
    
    return feedback, i + 1, n, error_cnt  # Increment iteration safely
