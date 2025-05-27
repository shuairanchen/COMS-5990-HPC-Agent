import os
import time
import traceback

import pandas as pd
from codebleu import calc_codebleu
from tabulate import tabulate
import numpy as np
from llm_code_eval import evaluate_translation

def read_file(fp):
    with open(fp, 'r') as f:
        return f.read()

from llm_code_eval import *
import os
import openai

fp_key='/home/hungphd/git/OPENAI_API_KEY.txt'
f1=open(fp_key,'r')
str_key=f1.read().strip()
f1.close()

os.environ["OPENAI_API_KEY"]=str_key
model_openai = openai.chat.completions
openai.api_key=str_key


# Path to the directories containing the files
set_input = '../large_set_exp/torch/samples_100/'
set_a_folder = '../large_set_exp/4o-mini_org/samples_100/'
set_b_folder = '../large_set_exp/4o-mini_jsonl/samples_100/'
strong_llms_folder = '../large_set_exp/o1-pro-manually/samples_100/'

# List all files in the set_B directory
# files = [f for f in os.listdir(set_b_folder) if os.path.isfile(os.path.join(set_b_folder, f))]
files = ['{}.py'.format(i) for i in range(1,101)]

selected_model='gpt-4o'
aspect='usefulness'
task="code-translation-torch2jax"
# Prepare lists for Excel and text export
text_data = []
result_fields = None  # we'll set this after first file
max_attempt=3
# Loop through each file in the directory
for name_file in files:
    input_source = read_file(os.path.join(set_input, name_file))
    prediction_a = read_file(os.path.join(set_a_folder, name_file))
    prediction_b = read_file(os.path.join(set_b_folder, name_file))
    reference = read_file(os.path.join(strong_llms_folder, name_file))

    index_attempt=1
    is_success=False
    result_a=0
    result_b=0
    while index_attempt<=max_attempt and not is_success:
        try:
            result_a = score = evaluate_translation(input_source_code="'''\n" + input_source + "\n'''",
                                                    output_translated_code="'''\n" + prediction_a + "\n'''",
                                                    # reference="'''\n" + reference + "\n'''",
                                                    task=task, aspect=aspect, model=selected_model)
            result_b = evaluate_translation(input_source_code="'''\n" + input_source + "\n'''",
                                            output_translated_code="'''\n" + prediction_b + "\n'''",
                                            # reference="'''\n" + reference + "\n'''",
                                            task=task, aspect=aspect, model=selected_model)
            int(result_a)
            int(result_b)
            is_success=True
        except Exception as e:
            traceback.print_exc()
        index_attempt+=1
        if (not is_success):
            time.sleep(30)
        print('{} {} {} {}'.format(name_file,max_attempt,result_a,result_b))


    result_a_dict = {'ice-score_gpt-4o_no-cot':result_a}
    result_b_dict = {'ice-score_gpt-4o_no-cot':result_b}

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
    + ["A - avg", "B - avg"]
)

# Output to TXT
text_path = 'large_ice-score_comparison_with_avg.txt'
with open(text_path, 'w') as f:
    f.write(tabulate([text_header] + text_data, headers="firstrow", tablefmt="grid"))

print(f"Text file saved at: {text_path}")

# Output to Excel (same structure)
df_excel = pd.DataFrame(text_data, columns=text_header)
excel_path = 'large_ice-score_comparison_with_avg.xlsx'
df_excel.to_excel(excel_path, index=False, sheet_name="CodeBLEU Comparison")

print(f"Excel file saved at: {excel_path}")
