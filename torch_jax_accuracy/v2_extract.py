from dotenv import load_dotenv
load_dotenv()
import os
from huggingface_hub import login
from pathlib import Path
import os
import boto3
from smart_open import open



def download_contents(blob_id, src_encoding):
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        content = fin.read().decode(src_encoding)

    return {"content": content}

fp_hf_token='../../HF_TOKEN.txt'
fp_aws_token='../../aws_info.txt'

f1=open(fp_aws_token,'r')
arr_content=f1.read().strip().split('\n')
f1.close()
os.environ["AWS_ACCESS_KEY_ID"]=arr_content[0]
os.environ["AWS_SECRET_ACCESS_KEY"]=arr_content[1]

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
s3 = session.client("s3")


f1=open(fp_hf_token,'r')
str_hf_token=f1.read()
f1.close()
os.environ['HF_TOKEN']=str_hf_token
login(token= os.getenv('HF_TOKEN'))  # Get token from Hugging Face settings
fop_output='../../torch_jax_data'
Path(fop_output).mkdir(exist_ok=True)
num_example=100


# Then load dataset
from datasets import load_dataset
# ds = load_dataset("bigcode/the-stack-v2", split="train")
# # full dataset (file IDs only)
# ds = load_dataset("bigcode/the-stack-v2", split="train")

# specific language (e.g. Dockerfiles)
# Load streamed dataset
ds = load_dataset("bigcode/the-stack-v2", "Python", streaming=True, split="train")
ds = ds.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))

# Filter and sample
filtered_ds = ds.filter(lambda x: "import torch" in x["content"])
pytorch_ds_sample = filtered_ds.shuffle(seed=42).take(num_example)  # `take()` for streaming datasets

pytorch_examples = []
for sample in iter(pytorch_ds_sample):
    if "import torch" in sample["content"]:
        pytorch_examples.append(sample["content"])
        # if len(pytorch_examples) >= 100:
        #     break

import json
with open(fop_output+"pytorch_100_examples.jsonl", "w") as f:
    for code in pytorch_examples:
        f.write(json.dumps({"content": code}) + "\n")


# dataset streaming (will only download the data as needed)
# ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
# # print(ds.cache_files)
# for sample in iter(pytorch_ds):
#     if "import torch" in sample["content"]:
#         print(sample["content"])