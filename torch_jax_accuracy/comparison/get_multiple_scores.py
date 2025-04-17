import os
from codebleu import calc_codebleu


def read_file(fp):
    with open(fp, 'r') as f:
        return f.read()


# Path to the directories containing the files
set_a_folder = '../set_A/'
set_b_folder = '../set_B/'
strong_llms_folder = '../strong_llms/'

# List all files in the set_A directory
files = [f for f in os.listdir(set_b_folder) if os.path.isfile(os.path.join(set_b_folder, f))]

# Loop through each file in the directory
for name_file in files:
    # Construct the full paths for each file
    prediction_a = read_file(os.path.join(set_a_folder, name_file))
    prediction_b = read_file(os.path.join(set_b_folder, name_file))
    reference = read_file(os.path.join(strong_llms_folder, name_file))

    # Print the name of the file being processed
    print(f"Processing {name_file}")

    # Calculate CodeBLEU for prediction_a vs. reference
    result_a = calc_codebleu([reference], [prediction_a], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                             tokenizer=None)
    print(f"CodeBLEU score for {name_file} (prediction A): {result_a}")

    # Calculate CodeBLEU for prediction_b vs. reference
    result_b = calc_codebleu([reference], [prediction_b], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                             tokenizer=None)
    print(f"CodeBLEU score for {name_file} (prediction B): {result_b}")

    print('-' * 50)  # Separator between file results
