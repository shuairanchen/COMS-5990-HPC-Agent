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

# List all files in the set_B directory
files = [f for f in os.listdir(set_b_folder) if os.path.isfile(os.path.join(set_b_folder, f))]

# Prepare lists for Excel and text export
text_data = []
result_fields = None  # we'll set this after first file

# Loop through each file in the directory
for name_file in files:
    prediction_a = read_file(os.path.join(set_a_folder, name_file))
    prediction_b = read_file(os.path.join(set_b_folder, name_file))
    reference = read_file(os.path.join(strong_llms_folder, name_file))

    result_a = calc_codebleu([reference], [prediction_a], lang="python", weights=(0.25, 0.25, 0.25, 0.25))
    result_b = calc_codebleu([reference], [prediction_b], lang="python", weights=(0.25, 0.25, 0.25, 0.25))

    result_a_dict = result_a[0] if isinstance(result_a, tuple) else result_a
    result_b_dict = result_b[0] if isinstance(result_b, tuple) else result_b

    if result_fields is None:
        result_fields = list(result_a_dict.keys())

    a_values = list(result_a_dict.values())
    b_values = list(result_b_dict.values())
    a_avg = np.mean(a_values)
    b_avg = np.mean(b_values)

    row = [name_file] + a_values + b_values + [a_avg, b_avg]
    text_data.append(row)

    print(f"Processed {name_file}")
    print('-' * 50)

# Compute averages across all rows
avg_row = ['Average']
for i in range(1, len(text_data[0])):  # Skip filename
    avg_value = np.mean([row[i] for row in text_data])
    avg_row.append(avg_value)

text_data.append(avg_row)

# Headers (A and B side-by-side per metric)
text_header = (
    ["File"]
    + [f"A - {field}" for field in result_fields]
    + [f"B - {field}" for field in result_fields]
    + ["A - codebleu avg", "B - codebleu avg"]
)

# Output to TXT
text_path = 'codebleu_comparison_with_avg.txt'
with open(text_path, 'w') as f:
    f.write(tabulate([text_header] + text_data, headers="firstrow", tablefmt="grid"))

print(f"Text file saved at: {text_path}")

# Output to Excel (same structure)
df_excel = pd.DataFrame(text_data, columns=text_header)
excel_path = 'codebleu_comparison_with_avg.xlsx'
df_excel.to_excel(excel_path, index=False, sheet_name="CodeBLEU Comparison")

print(f"Excel file saved at: {excel_path}")
