import os
import json
import traceback

from openai import OpenAI
import logging
# import environ
from dotenv import load_dotenv
import shutil

load_dotenv()

# Set up logging
logging.basicConfig(filename="translation.log", level=logging.INFO)


fp_hf_token='../../HF_TOKEN.txt'
fp_openai_api_key='../../OPENAI_API_KEY.txt'
fp_openrouter_api_key='../../OPENROUTER_API_KEY.txt'

f1=open(fp_hf_token,'r')
str_hf_token=f1.read()
f1.close()

f1=open(fp_openai_api_key,'r')
str_openai_api_key=f1.read().strip()
f1.close()

f1=open(fp_openrouter_api_key,'r')
str_openrouter_api_key=f1.read().strip()
f1.close()


os.environ['HF_TOKEN']=str_hf_token
os.environ["OPENAI_API_KEY"]=str_openai_api_key
os.environ["OPENROUTER_API_KEY"]=str_openrouter_api_key


# OpenAI API setup
api_key = os.environ.get("OPENAI_API_KEY")  # Or replace with "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key=api_key)

# Directories

input_dir = "large_test_datasets_codeparrot_v1/samples_100/"  # Path to the directory containing PyTorch files
output_dir = "prompt_codeparrot_4omini/set_100/"
fp_csv='data.csv'
fp_jsonl='pytorch_to_jax_code_rules_20_cases.json'
fp_input_template='prompt_4omini_template.txt'
fp_system_prompt='prompts/prompt_optimize.txt'
f1=open(fp_input_template,'r')
str_p_t=f1.read().strip()
f1.close()
f1=open(fp_system_prompt,'r')
str_s_p=f1.read().strip()
f1.close()
os.makedirs(output_dir, exist_ok=True)


# Process each file
for i in range(1, 101):  # example_1.py to example_100.py
    input_file = os.path.join(input_dir, f"{i}.py")
    output_folder_item = os.path.join(output_dir, f"{i}/")
    os.makedirs(output_folder_item, exist_ok=True)
    f1=open(input_file,'r')
    str_code_input=f1.read()
    f1.close()
    str_prompt_p2=str_p_t.replace('{code}',str_code_input)
    f1=open(output_folder_item+'1_prompt_part1.txt','w')
    f1.write(str_s_p)
    f1.close()
    f1 = open(output_folder_item + '1_prompt_part2.txt', 'w')
    f1.write(str_prompt_p2)
    f1.close()

    shutil.copy(fp_jsonl, output_folder_item+'2_errors_and_fixes.json')
    from datasets import load_dataset
    dataset = json.load(open(output_folder_item+'2_errors_and_fixes.json','r'))
    print('len {}'.format(len(dataset["Pytorch_to_JAX_Examples"])))
    shutil.copy(fp_csv, output_folder_item + '3_data.csv')

    f1 = open(output_folder_item + '4_output.txt', 'w')
    f1.write('')
    f1.close()

print("Translation complete. Check jax_files and translation.log for details.")