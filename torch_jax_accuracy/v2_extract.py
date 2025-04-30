from dotenv import load_dotenv
load_dotenv()
import os
from huggingface_hub import login
login(token= os.getenv('HF_TOKEN'))  # Get token from Hugging Face settings

# Then load dataset
from datasets import load_dataset
# ds = load_dataset("bigcode/the-stack-v2", split="train")
# # full dataset (file IDs only)
# ds = load_dataset("bigcode/the-stack-v2", split="train")

# specific language (e.g. Dockerfiles)
ds = load_dataset("bigcode/the-stack-v2", "Python", split="train")
pytorch_ds = ds.filter(lambda x: "import torch" in x["content"])

pytorch_examples = []
for sample in iter(pytorch_ds):
    if "import torch" in sample["content"]:
        pytorch_examples.append(sample["content"])
        if len(pytorch_examples) >= 100:
            break

import json
with open("pytorch_100_examples.jsonl", "w") as f:
    for code in pytorch_examples:
        f.write(json.dumps({"content": code}) + "\n")


# dataset streaming (will only download the data as needed)
# ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
# # print(ds.cache_files)
# for sample in iter(pytorch_ds):
#     if "import torch" in sample["content"]:
#         print(sample["content"])