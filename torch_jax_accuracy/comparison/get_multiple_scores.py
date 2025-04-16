from codebleu import calc_codebleu

def read_file(fp):
    f1=open(fp,'r')
    s1=f1.read()
    f1.close()
    return s1
name_file='h4.py'
prediction_a = read_file('../set_A/'+name_file)
prediction_b=read_file('../set_B/'+name_file)
reference = read_file('../set_C_fixed_code/'+name_file)

print(name_file)
result_a = calc_codebleu([reference], [prediction_a], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result_a)
result_b = calc_codebleu([reference], [prediction_b], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result_b)