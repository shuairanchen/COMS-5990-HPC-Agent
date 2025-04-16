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
