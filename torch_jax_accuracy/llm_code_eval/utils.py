TASK_PROMPTS = {
    "code-gen": {
        "functional correctness": 
            {
                "reference-free":
"""\
You will be given the code snippet for a problem. 
Your task is to rate the code snippet only on one metric.
Please make sure you read and understand these instructions carefully.
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Functional Correctness (0-4) - Execution-based quality of the code snippet combined with the problem. The correctness is measured by the all possible unit tests, and the comparison of the reference code. The combination of the code snippet and the problem should pass all the possible tests based on your understanding of the reference code. The length of the code snippet can not determine the correctness. You need to assess the logics line by line.
- A score of 0  (failing all possible test) means that the code snippet is totally incorrect and meaningless.
- A score of 4  (passing all possible test) means that the code snippet is totally correct and can handle all cases.


Evaluation Steps:
1. Read the problem carefully and identify required functionalities of the implementation.
2. Read the code snippet and compare it to the problem. Check if the code snippet covers all required functionalities of the problem. 
3. Assign a score for functional correctness on a scale of 0 to 4, where 0 is the lowest and 4 is the highest based on the Evaluation Criteria.

Problem:

{{PROBLEM}}

Code Snippet:

{{OUTPUT}}

Evaluation Form:
Functional Correctness (scores ONLY):
""",
                "reference-enhanced":
"""\
You will be given the code snippet for a problem. 
Your task is to rate the code snippet only on one metric.
Please make sure you read and understand these instructions carefully.
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Functional Correctness (0-4) - Execution-based quality of the code snippet combined with the problem. The correctness is measured by the all possible unit tests, and the comparison of the reference code. The combination of the code snippet and the problem should pass all the possible tests based on your understanding of the reference code. The length of the code snippet can not determine the correctness. You need to assess the logics line by line.
- A score of 0  (failing all possible test) means that the code snippet is totally incorrect and meaningless.
- A score of 4  (passing all possible test) means that the code snippet is totally correct and can handle all cases.


Evaluation Steps:
1. Read the problem carefully and identify required functionalities of the implementation.
2. Read the code snippet and compare it to the reference code. Check if the code snippet covers all required functionalities of the problem, and if it is as good as the reference code. 
3. Assign a score for functional correctness on a scale of 0 to 4, where 0 is the lowest and 4 is the highest based on the Evaluation Criteria.

Problem:

{{PROBLEM}}

Reference Code:

{{REFERENCE}}

Code Snippet:

{{OUTPUT}}

Evaluation Form:
Functional Correctness (scores ONLY):
"""
            },
        "usefulness":
            {
                "reference-free":
"""\
You will be given the code snippet for a problem.
Your task is to rate the code snippet only on one metric.
Please make sure you read and understand these instructions carefully.
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Usefulness (0-4) Usefulness of the code snippet based on the problem description.

- A score of 0: Snippet is not at all helpful, it is irrelevant to the problem.
- A score of 1: Snippet is slightly helpful, it contains information relevant to the problem, but it is easier to write the solution from scratch.
- A score of 2: Snippet is somewhat helpful, it requires significant changes (compared to the size of the snippet), but is still useful.
- A score of 3: Snippet is helpful, but needs to be slightly changed to solve the problem.
- A score of 4: Snippet is very helpful, it solves the problem.

Evaluation Steps:
1. Read the problem carefully and identify required functionalities of the implementation.
2. Read the code snippet and compare it to the problem. Check if the code snippet covers all required functionalities of the problem, and if it presents them in a clear and logical order. 
3. Assign a score for usefulness on a scale of 0 to 4, where 0 is the lowest and 4 is the highest based on the Evaluation Criteria.

Problem:

{{PROBLEM}}

Code Snippet:

{{OUTPUT}}

Evaluation Form:
Usefulness (scores ONLY):
""",
                "reference-enhanced":
"""\
You will be given the code snippet for a problem.
Your task is to rate the code snippet only on one metric.
Please make sure you read and understand these instructions carefully.
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Usefulness (0-4) Usefulness of the code snippet based on the problem description and the comparison of reference code.

- A score of 0: Snippet is not at all helpful, it is irrelevant to the problem.
- A score of 1: Snippet is slightly helpful, it contains information relevant to the problem, but it is easier to write the solution from scratch.
- A score of 2: Snippet is somewhat helpful, it requires significant changes (compared to the size of the snippet), but is still useful.
- A score of 3: Snippet is helpful, but needs to be slightly changed to solve the problem.
- A score of 4: Snippet is very helpful, it solves the problem.

Evaluation Steps:
1. Read the problem carefully and identify required functionalities of the implementation.
2. Read the code snippet and compare it to the problem and reference code. Check if the code snippet covers all required functionalities of the problem, and if it presents them in a clear and logical order. 
3. Assign a score for usefulness on a scale of 0 to 4, where 0 is the lowest and 4 is the highest based on the Evaluation Criteria.

Problem:

{{PROBLEM}}

Reference Code:

{{REFERENCE}}

Code Snippet:

{{OUTPUT}}

Evaluation Form:
Usefulness (scores ONLY):
"""                  
            }
    },
"code-translation-torch2jax": {
    "functional correctness": {
        "reference-free":
"""\
You will be given a JAX code snippet that was translated from PyTorcH source code.  
Your task is to rate the snippet on **one metric only**: its **functional correctness**.

Please ensure you read and understand these instructions carefully before reviewing.
Refer to this guide as needed during the evaluation process.

Evaluation Criteria:  
Functional Correctness (0–4) — How well the JAX code preserves the behavior of the original PyTorch code.  
You are to assess whether the JAX code would produce equivalent outputs to the original PyTorch code across possible inputs, even though the PyTorch code is not shown. Consider unit-test-style logic and general expectations of equivalence.

- A score of 0: The translation is completely incorrect and meaningless.
- A score of 4: The translation is fully correct and handles all core functionalities as expected.

Evaluation Steps:
1. Assume the code was translated from PyTorch and should preserve its logic.
2. Evaluate whether the JAX code appears complete, meaningful, and implementationally correct based on general expectations for such translations.
3. Assign a score for functional correctness on a scale from 0 to 4.

Input Source Code in PyTorch:  
{{SOURCE_CODE}}

Translated JAX Code Snippet:

{{TRANSLATED_CODE}}

Evaluation Form:  
Functional Correctness (scores ONLY):
""",
        "reference-enhanced":
"""\
You will be given a JAX code snippet and its corresponding PyTorch source code.  
Your task is to rate the JAX code on **one metric only**: its **functional correctness**.

Please ensure you read and understand these instructions carefully before reviewing.
Refer to this guide as needed during the evaluation process.

Evaluation Criteria:  
Functional Correctness (0–4) — How accurately the JAX code preserves the behavior and logic of the PyTorch source code.  
You should assess whether the translated code is functionally equivalent, even if written differently.

- A score of 0: The translation is completely incorrect and breaks all functionality.
- A score of 4: The translation is functionally identical and handles all edge cases.

Evaluation Steps:
1. Read the PyTorch input code carefully and understand its function and structure.
2. Evaluate whether the JAX translated code replicates the same functionality.
3. Assign a score for functional correctness from 0 to 4.

Input Source Code in PyTorch:

{{SOURCE_CODE}}

Translated JAX Code Snippet:

{{TRANSLATED_CODE}}

Evaluation Form:  
Functional Correctness (scores ONLY):
"""
    },
    "usefulness": {
        "reference-free":
"""\
You will be given a JAX code snippet that was translated from PyTorch.  
Your task is to rate the snippet on **one metric only**: its **usefulness** for understanding and reusing the logic of a typical PyTorch implementation.

Please ensure you read and understand these instructions carefully before reviewing.
Refer to this guide as needed during the evaluation process.

Evaluation Criteria:  
Usefulness (0–4) — How useful the JAX code is for replicating or adapting the functionality of a typical PyTorch source code implementation.

- A score of 0: The JAX translated snippet is irrelevant or confusing and does not help at all.
- A score of 1: The JAX translated snippet includes some related elements but is mostly unhelpful.
- A score of 2: The JAX translated snippet is somewhat useful but needs substantial modification.
- A score of 3: The JAX translated snippet is helpful with minor revisions needed.
- A score of 4: The JAX translated snippet is very helpful and covers the intended functionality clearly.

Evaluation Steps:
1. Assume the PyTorch source code performs a well-defined functionality.
2. Determine whether the JAX translated code snippet enables meaningful reuse or guidance toward equivalent implementation.
3. Assign a score for usefulness from 0 to 4.

Input Source Code in PyTorch:  
{{SOURCE_CODE}}

Translated JAX Code Snippet:

{{TRANSLATED_CODE}}

Evaluation Form:  
Usefulness (scores ONLY):
""",
        "reference-enhanced":
"""\
You will be given a JAX code snippet and its corresponding PyTorch source code.  
Your task is to rate the JAX code on **one metric only**: its **usefulness** for understanding or adapting the PyTorch implementation.

Please ensure you read and understand these instructions carefully before reviewing.
Refer to this guide as needed during the evaluation process.

Evaluation Criteria:  
Usefulness (0–4) — How useful the JAX code is in replicating the functionality and intent of the PyTorch code, either directly or with minor adjustments.

- A score of 0: The JAX translated snippet is irrelevant or significantly misleading.
- A score of 1: The JAX translated snippet is only marginally helpful and poorly aligned.
- A score of 2: The JAX translated snippet conveys some intent but needs major revision.
- A score of 3: The JAX translated snippet is mostly useful with few needed fixes.
- A score of 4: The JAX translated snippet is highly useful and clearly matches the original implementation’s intent.

Evaluation Steps:
1. Read and understand the PyTorch input source code.
2. Review the JAX translation to determine if it helps replicate or generalize the PyTorch functionality.
3. Assign a score from 0 to 4 based on usefulness.

Input Source Code in PyTorch:

{{SOURCE_CODE}}

Translated JAX Code Snippet:

{{TRANSLATED_CODE}}

Evaluation Form:  
Usefulness (scores ONLY):
"""
    }
}

}