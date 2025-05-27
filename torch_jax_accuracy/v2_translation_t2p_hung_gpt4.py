import os
import json
import time
import traceback

from openai import OpenAI
import logging
# import environ
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(filename="translation.log", level=logging.INFO)
max_run=10


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
output_dir = "large_test_datasets_codeparrot_translation_gpt-4o/samples_100/"
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
for i in range(1, 101):  # example_1.py to example_100.py
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

        index_run = 0
        is_run_ok = False
        while (index_run <= max_run):
            index_run += 1
            # Prepare prompt
            try:
                if os.path.exists(output_file):
                    is_run_ok=True
                    break
                else:
                    print('file {} not exists'.format(i))
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="openai/gpt-4o",  # Or "gpt-4" if available
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048  # Adjust based on code length
                )
                # print('{} {}'.format(type(response),response))
                # jax_code = response.choices[0].message.content.strip()
                jax_code = response.choices[0].message.content.strip()
                if jax_code!='':
                    # Extract code from response
                    if jax_code.startswith("```python"):
                        jax_code = jax_code.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                    # Save JAX code
                    with open(output_file, "w") as f:
                        f.write(jax_code)
                    is_run_ok = True
                    # break
                    logging.info(f"Successfully translated {input_file} to {output_file}")
            except Exception as e:
                traceback.print_exc()
            print('handle {} {}'.format(i, is_run_ok))
            # time.sleep(0.5)
            if is_run_ok:
                break

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Failed to translate {input_file}: {str(e)}")
        # continue
    if test_mode:
        break

print("Translation complete. Check jax_files and translation.log for details.")