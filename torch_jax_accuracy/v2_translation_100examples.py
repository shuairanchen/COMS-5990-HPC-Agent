import os
import json
from openai import OpenAI
import logging
# import environ
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(filename="translation.log", level=logging.INFO)

# OpenAI API setup
api_key = os.environ.get("OPENAI_API_KEY")  # Or replace with "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key=api_key)

# Directories

input_dir = "/Users/sonvu/Documents/Son Vu/Research/Fall2025/LLM-Agent/CC6-HPC-Agent-main/pytorch_files"  # Path to the directory containing PyTorch files
output_dir = "/Users/sonvu/Documents/Son Vu/Research/Fall2025/LLM-Agent/CC6-HPC-Agent-main/pytorch_to_jax_files"
os.makedirs(output_dir, exist_ok=True)

# Translation prompt
prompt_template = """
Translate the following PyTorch code to equivalent JAX code. Make sure the output of the translated code is similar to the input code. Return only the translated code, no explanations.

PyTorch Code:
```python
{code}
```

JAX Code:
```python
```
"""
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)
# Process each file
for i in range(1, 101):  # example_1.py to example_100.py
    input_file = os.path.join(input_dir, f"example_{i}.py")
    output_file = os.path.join(output_dir, f"jax_example_{i}.py")
    
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
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="openai/o3-mini",  # Or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000  # Adjust based on code length
        )
        jax_code = response.choices[0].message.content.strip()
        
        # Extract code from response
        if jax_code.startswith("```python"):
            jax_code = jax_code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        # Save JAX code
        with open(output_file, "w") as f:
            f.write(jax_code)
        
        logging.info(f"Successfully translated {input_file} to {output_file}")
    
    except Exception as e:
        logging.error(f"Failed to translate {input_file}: {str(e)}")
        continue

print("Translation complete. Check jax_files and translation.log for details.")