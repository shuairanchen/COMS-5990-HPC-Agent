import json
import os
from huggingface_hub import login
from datasets import load_dataset

login(token=os.getenv("HF_TOKEN"))
ds = load_dataset("codeparrot/github-code", "Python-all", streaming=True, split="train")

# Collect 100 PyTorch examples
pytorch_examples = []
max_file_size = 100_000  # 100KB limit
for sample in iter(ds):
    code = sample["code"]  # Field for code
    if ("import torch" in code and 
        any(pattern in code for pattern in ["torch.nn", "torch.tensor", "torch.optim", "torch.utils.data"])):
        if len(code.encode("utf-8")) <= max_file_size:
            pytorch_examples.append(code)
            if len(pytorch_examples) >= 100:
                break

# Save to JSONL
with open("pytorch_100_examples.jsonl", "w") as f:
    for code in pytorch_examples:
        f.write(json.dumps({"code": code}) + "\n")

# Extract to .py files
output_dir = "pytorch_files"
os.makedirs(output_dir, exist_ok=True)
for i, code in enumerate(pytorch_examples):
    with open(f"{output_dir}/example_{i+1}.py", "w") as py_file:
        py_file.write(code)

# import json
# import os
# from huggingface_hub import login
# from datasets import load_dataset

# login(token=os.getenv('HF_TOKEN'))
# # ds = load_dataset("bigcode/the-stack-v2", "Python", streaming=True, split="train")
# ds = load_dataset("codeparrot/github-code", "Python", streaming=True, split="train")
# pytorch_ds = ds.filter(lambda x: "import torch" in x["path"])
# for sample in ds: print(sample.keys()); break
# # Collect 100 PyTorch examples
# pytorch_examples = []
# for sample in iter(pytorch_ds):
#     # if "import torch" in sample["text"]:  # Replace "code" with "text"
#     pytorch_examples.append(sample["path"])
#     if len(pytorch_examples) >= 100:
#         break

# # Save to JSONL
# with open("pytorch_100_examples.jsonl", "w") as f:
#     for code in pytorch_examples:
#         f.write(json.dumps({"text": code}) + "\n")

# # Extract to .py files
# output_dir = "pytorch_files"
# os.makedirs(output_dir, exist_ok=True)
# for i, code in enumerate(pytorch_examples):
#     with open(f"{output_dir}/example_{i+1}.py", "w") as py_file:
#         py_file.write(code)
# from dotenv import load_dotenv
# load_dotenv()
# import os
# from huggingface_hub import login
# login(token= os.getenv('HF_TOKEN'))  # Get token from Hugging Face settings

# # Then load dataset
# from datasets import load_dataset
# # ds = load_dataset("bigcode/the-stack-v2", split="train")
# # # full dataset (file IDs only)
# # ds = load_dataset("bigcode/the-stack-v2", split="train")

# # specific language (e.g. Dockerfiles)
# ds = load_dataset("bigcode/the-stack-v2", "Python", split="train")
# # pytorch_ds = ds.filter(lambda x: "import torch" in x["content"])
# print(ds.cache_files)

# pytorch_examples = []
# for sample in iter(pytorch_ds):
#     if "import torch" in sample["content"]:
#         pytorch_examples.append(sample["content"])
#         if len(pytorch_examples) >= 100:
#             break

# import json
# with open("pytorch_100_examples.jsonl", "w") as f:
#     for code in pytorch_examples:
#         f.write(json.dumps({"content": code}) + "\n")


# dataset streaming (will only download the data as needed)
# ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
# # print(ds.cache_files)
# for sample in iter(pytorch_ds):
#     if "import torch" in sample["content"]:
#         print(sample["content"])
