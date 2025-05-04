import os
import json
import traceback

from openai import OpenAI
import logging
# import environ
from dotenv import load_dotenv
import tiktoken
load_dotenv()

def count_tokens_o3(prompt: str, model: str = "openai/o3-mini") -> int:
    """
    Count the number of tokens in a prompt for a given OpenAI model (default: o3-mini).
    """
    try:
        # Load tokenizer for the given model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to a general tokenizer if specific one isn't found
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode and count tokens
    tokens = encoding.encode(prompt)
    return len(tokens)
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
num_samples=2000
input_dir = "large_test_datasets_codeparrot_v1/samples_{}/".format(num_samples)  # Path to the directory containing PyTorch files
output_dir = "large_test_datasets_codeparrot_translation_org/samples_{}/".format(num_samples)
os.makedirs(output_dir, exist_ok=True)

# Translation prompt
prompt_template = """
Translate the following PyTorch code to equivalent JAX code. Make sure the when we run the translated code its output should be similar to the output when running input code. Return only the translated code, no explanations.

1) Input as PyTorch Code:
```python
{code}
```

2) Return the output as JAX Code:
```python
```
"""
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

test_mode=False

# Process each file
list_lengths=[]
for i in range(1, num_samples):  # example_1.py to example_100.py
    input_file = os.path.join(input_dir, f"{i}.py")
    output_file = os.path.join(output_dir, f"{i}.py")
    
    try:
        # Read PyTorch code
        with open(input_file, "r") as f:
            pytorch_code = f.read()
        
        # Skip empty or invalid files
        if not pytorch_code.strip():
            logging.warning(f"Skipping empty file: {input_file}")
            continue
        
        # Prepare prompt
        prompt = prompt_template.format(code=pytorch_code)
        len_prompt=count_tokens_o3(prompt)
        list_lengths.append(len_prompt)
        # Call OpenAI API
        # response = client.chat.completions.create(
        #     model="openai/o3-mini",  # Or "gpt-4" if available
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=8000  # Adjust based on code length
        # )
        # print('{} {}'.format(type(response),response))
        # # jax_code = response.choices[0].message.content.strip()
        # jax_code = response.choices[0].message.content.strip()
        #
        # # Extract code from response
        # if jax_code.startswith("```python"):
        #     jax_code = jax_code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        #
        # # Save JAX code
        # with open(output_file, "w") as f:
        #     f.write(jax_code)
        
        logging.info(f"Successfully translated {input_file} to {output_file}")

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Failed to translate {input_file}: {str(e)}")
        # continue
    if test_mode:
        break
print("Translation complete. Check jax_files and translation.log for details.")

import matplotlib.pyplot as plt
import numpy as np
import statistics

def analyze_integer_list(int_list, bins=10, title="Integer Distribution"):
    if not int_list:
        raise ValueError("Input list is empty.")

    # Calculate stats
    minimum = min(int_list)
    maximum = max(int_list)
    median = statistics.median(int_list)
    mean = statistics.mean(int_list)

    print(f"Min: {minimum}")
    print(f"Max: {maximum}")
    print(f"Median: {median}")
    print(f"Mean: {mean:.2f}")

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(int_list, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {"min": minimum, "max": maximum, "median": median, "mean": mean}
analyze_integer_list(list_lengths)