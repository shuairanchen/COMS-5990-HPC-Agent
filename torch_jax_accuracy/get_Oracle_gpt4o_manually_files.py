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


fop_input='data_Tuan_trans_3_results/'
fop_output='/home/hungphd/git/COMS-5990-HPC-Agent/torch_jax_accuracy/large_set_exp/gpt-4o-manually/samples_100/'
create_nested_folder(fop_output)
from pathlib import Path
import shutil
Path(fop_output).mkdir(exist_ok=True)
for i in range(1,101):
    fp_out=fop_input+'{}/2_output.txt'.format(i)
    if os.path.exists(fp_out):
        shutil.copy2(fp_out, fop_output+'{}.py'.format(i))
    else:
        print('miss file {}'.format(i))