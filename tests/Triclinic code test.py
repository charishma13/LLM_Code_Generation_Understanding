s = int(input('enter the value:'))
print(s)

i = 1
n = 2
if i == n - 1:
    print('yes')

import subprocess

# Execute a simple command with a newline
result = subprocess.run(['echo', 'print(Hello)\nprint(World)'], capture_output=True, text=True)

# Get the output
print(result.stdout)

import subprocess
import tempfile

# Create a temporary Python script with the code
with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_script:
    temp_script.write(b'print("Hello")\nprint("World")\n')
    temp_script_path = temp_script.name

# Execute the Python script using subprocess
result = subprocess.run(['python3', temp_script_path], capture_output=True, text=True)

# Get the output
print(result.stdout)

# Optionally, remove the temporary file
import os
os.remove(temp_script_path)


# Import required library
import pandas as pd

# Copy and paste the following data into the clipboard:
# Date;Event;Cost
# 10/2/2011;Music;10000
# 11/2/2011;Poetry;12000
# 12/2/2011;Theatre;5000
# 13/2/2011;Comedy;8000

# Read data from clipboard
df = pd.read_clipboard(sep=';')

# Print the DataFrame
print(df)


