import os.path
from pathlib import Path
import nbformat
from glob import glob

fop='fix_progress/'
fop_out='strong_llms/'
Path(fop_out).mkdir(parents=True, exist_ok=True)
list_files=sorted(glob(fop+'*.ipynb'))


# Load the notebook file
for fp in list_files:
    name=os.path.basename(fp).replace('.ipynb','.py')
    with open(fp, 'r') as f:
        notebook = nbformat.read(f, as_version=4)

    # Get the second cell (index 1)
    second_cell = notebook['cells'][1]
    str_code='EMPTy'

    # Check if it's a code cell or markdown
    if second_cell['cell_type'] == 'code':
        print('Code cell:')
        print('Source code:', ''.join(second_cell['source']))
        str_code=second_cell['source']
    # elif second_cell['cell_type'] == 'markdown':
    #     print('Markdown cell:')
    #     print('Source:', ''.join(second_cell['source']))
    # else:
    #     print('Other cell type:', second_cell['cell_type'])
    f1=open(fop_out+name,'w')
    f1.write(str_code)
    f1.close()

