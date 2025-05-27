fop='/home/hungphd/git/COMS-5990-HPC-Agent/torch_jax_accuracy/large_set_exp/'
out='/home/hungphd/git/COMS-5990-HPC-Agent/torch_jax_accuracy/data_Tuan_compare/input/'

import os


def create_nested_folder(path):
    """
    Creates a folder and all intermediate-level directories if they don't exist.

    Args:
        path (str): A string representing the full path to the folder you want to create.

    Returns:
        str: Absolute path to the created folder.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)
    except Exception as e:
        raise RuntimeError(f"Failed to create folder '{path}': {e}")


from pathlib import Path

create_nested_folder(out)
fop_source=fop+'torch/samples_100/'
fop_cand_A=fop+'4o-mini_org/samples_100/'
fop_cand_B=fop+'4o-mini_jsonl/samples_100/'


compare_msg_template=f'''You are an expert in Pytorch to JAX translation.
I provide 3 inputs: 1 . Pytorch input code; 2. Translated Code Candidate A; 3. Translated Code Candidate B. Which candidate is a better translation result for this Pytorch code.

Input Pytorch code:
‘’’
{{input}}
‘’’
2. Translated Code A:
‘’’
{{translated code A}}
‘’’
3. Translated Code B:
‘’’
{{translated code B}}
‘’’

Please also provide the reason why you consider a candidate better than the other translated code candidate.
'''


def get_content(fp):
    f1=open(fp,'r')
    str_c=f1.read().strip()
    f1.close()
    return str_c

for i in range(1,101):
    str_source=get_content(fop_source+'{}.py'.format(i))
    str_cand_A = get_content(fop_cand_A + '{}.py'.format(i))
    str_cand_B = get_content(fop_cand_B + '{}.py'.format(i))
    str_prompt=compare_msg_template.replace('{input}',str_source).replace('{translated code A}',str_cand_A).replace('{translated code B}',str_cand_B)
    f1=open(out+'{}.txt'.format(i),'w')
    f1.write(str_prompt)
    f1.close()

