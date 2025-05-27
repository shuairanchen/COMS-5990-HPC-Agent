import os

fop_input='/home/hungphd/Downloads/result_4omini_jsonl/result_4omini_jsonl/set_100/'
fop_output='/home/hungphd/git/COMS-5990-HPC-Agent/torch_jax_accuracy/large_set_exp/4o-mini_jsonl/sample_100/'
from pathlib import Path
import shutil
Path(fop_output).mkdir(exist_ok=True)
for i in range(1,101):
    fp_out=fop_input+'{}/4_output.txt'.format(i)
    if os.path.exists(fp_out):
        shutil.copy2(fp_out, fop_output+'{}.py'.format(i))
    else:
        print('miss file {}'.format(i))