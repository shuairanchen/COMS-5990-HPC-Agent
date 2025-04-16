import os.path
from glob import glob

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

def read_file(fp):
    f1=open(fp,'r')
    s1=f1.read()
    f1.close()
    return s1

fop_input_A='../set_A/'
fop_input_B='../set_B/'
fop_input_C='../set_C_fixed_code/'
fop_input_O='../set_O/'
dir_path='..prompts/prompt_compare/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created.")
else:
    print(f"Directory '{dir_path}' already exists.")

list_files=sorted(glob(fop_input_C+'*.py'))
for file in list_files:
    name=os.path.basename(file)

    str_content= compare_msg_template.replace('{{input}}',read_file(fop_input_O+name)).replace('{{translated code A}}',read_file(fop_input_A+name)).replace('{{translated code B}}',read_file(fop_input_B+name))
    f1=open(dir_path+name.replace('.py','.txt'),'w')
    f1.write(str_content)
    f1.close()


