import os
import pandas as pd
from codebleu import calc_codebleu
from tabulate import tabulate
import numpy as np


def read_file(fp):
    with open(fp, 'r') as f:
        return f.read()


# Path to the directories containing the files
set_a_folder = '../set_A/'
set_b_folder = '../set_B/'
strong_llms_folder = '../strong_llms/'

# List all files in the set_B directory (you can adjust this to any set)
files = [f for f in os.listdir(set_b_folder) if os.path.isfile(os.path.join(set_b_folder, f))]

# Prepare a dictionary to collect data for Excel export
excel_data = {}

# Prepare a list to collect data for text file
text_data = []

# Loop through each file in the directory
for name_file in files:
    # Construct the full paths for each file
    prediction_a = read_file(os.path.join(set_a_folder, name_file))
    prediction_b = read_file(os.path.join(set_b_folder, name_file))
    reference = read_file(os.path.join(strong_llms_folder, name_file))

    # Calculate CodeBLEU for prediction_a vs. reference
    result_a = calc_codebleu([reference], [prediction_a], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                             tokenizer=None)
    print(f"CodeBLEU score for {name_file} (prediction A): {result_a}")

    # Calculate CodeBLEU for prediction_b vs. reference
    result_b = calc_codebleu([reference], [prediction_b], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                             tokenizer=None)
    print(f"CodeBLEU score for {name_file} (prediction B): {result_b}")

    # Assuming result_a and result_b are dictionaries or tuples with various fields
    result_a_dict = result_a[0] if isinstance(result_a, tuple) else result_a
    result_b_dict = result_b[0] if isinstance(result_b, tuple) else result_b

    # Prepare the data for all fields in CodeBLEU results
    comparison_result = {
        "Field": list(result_a_dict.keys()),  # Extract the fields (keys) from the dictionary
        "A - codebleu": list(result_a_dict.values()),  # Get the values for A - codebleu
        "B - codebleu": list(result_b_dict.values())  # Get the values for B - codebleu
    }

    # Add the results to the dictionary for Excel export
    excel_data[name_file] = pd.DataFrame(comparison_result)

    # Calculate the average for each row (field)
    a_values = list(result_a_dict.values())
    b_values = list(result_b_dict.values())

    a_avg = np.mean(a_values)  # Average for A
    b_avg = np.mean(b_values)  # Average for B

    # Add the averages to the rows
    comparison_result["A - codebleu avg"] = [a_avg] * len(comparison_result["Field"])
    comparison_result["B - codebleu avg"] = [b_avg] * len(comparison_result["Field"])

    # Prepare text data for the table format (this assumes keys are the fields and values are the corresponding values)
    text_data.append([name_file] + a_values + b_values + [a_avg, b_avg])

    # Print separator for clarity
    print('-' * 50)

# Export to Excel
excel_path = 'codebleu_comparison_with_avg.xlsx'
with pd.ExcelWriter(excel_path) as writer:
    for file_name, df in excel_data.items():
        df.to_excel(writer, sheet_name=file_name)

print(f"Excel file saved at: {excel_path}")

# Prepare text data header for the table
fields = list(result_a_dict.keys())
text_header = ["File"] + ["A - " + field for field in fields] + ["B - " + field for field in fields] + [
    "A - codebleu avg", "B - codebleu avg"]

# Write to a text file in table format
text_path = 'codebleu_comparison_with_avg.txt'
with open(text_path, 'w') as f:
    f.write(tabulate([text_header] + text_data, headers="firstrow", tablefmt="grid"))

print(f"Text file saved at: {text_path}")
