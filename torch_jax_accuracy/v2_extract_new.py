import json
import os
from huggingface_hub import login
from datasets import load_dataset

fp_hf_token='../../HF_TOKEN.txt'
fp_aws_token='../../aws_info.txt'

f1=open(fp_hf_token,'r')
str_hf_token=f1.read()
f1.close()

f1=open(fp_aws_token,'r')
arr_content=f1.read().strip().split('\n')
f1.close()


os.environ['HF_TOKEN']=str_hf_token


login(token=os.environ["HF_TOKEN"])
# print(os.environ["HF_TOKEN"])

fop_sample='large_test_datasets_codeparrot/'

max_file_size = 100_000  # 100KB limit
list_sizes=[100,200,500,1000,2000]

for ind_size in range(0,len(list_sizes)):
    # Collect 100 PyTorch examples
    ds = load_dataset("codeparrot/github-code", "Python-all", streaming=True, split="train",
                      trust_remote_code=True).shuffle()

    num_sample=list_sizes[ind_size]
    pytorch_examples = []
    print('begin {}'.format(num_sample))
    for sample in iter(ds):
        code = sample["code"]  # Field for code
        if ("import torch" in code and
            any(pattern in code for pattern in ["torch.nn", "torch.tensor", "torch.optim", "torch.utils.data"])):
            if len(code.encode("utf-8")) <= max_file_size:
                pytorch_examples.append(code)
                if len(pytorch_examples) >= num_sample:
                    break

    # Save to JSONL
    with open(fop_sample+"samples_{}.jsonl".format(num_sample), "w") as f:
        for code in pytorch_examples:
            f.write(json.dumps({"code": code}) + "\n")

    # Extract to .py files
    output_dir = fop_sample+ "samples_{}/".format(num_sample)
    os.makedirs(output_dir, exist_ok=True)
    for i, code in enumerate(pytorch_examples):
        with open(f"{output_dir}/{i+1}.py", "w") as py_file:
            py_file.write(code)
    print('end {}'.format(num_sample))

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
