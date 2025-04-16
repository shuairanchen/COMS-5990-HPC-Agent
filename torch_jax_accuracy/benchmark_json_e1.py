import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from codebleu import calc_codebleu
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import ast
import sys
from io import StringIO
import contextlib
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
except:
    nltk.download('punkt', quiet=True)

# Your JAX code snippets
CODE_SNIPPET_1 = """
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax

class LinearRegressionModel:
    def __init__(self, key):
        key, subkey = random.split(key)
        self.w = random.uniform(subkey, (1, 1), minval=-1.0, maxval=1.0)
        key, subkey = random.split(key)
        self.b = random.uniform(subkey, (1,), minval=-1.0, maxval=1.0)
        self.params = {'w': self.w, 'b': self.b}

    def __call__(self, x):
        return jnp.dot(x, self.params['w']) + self.params['b']

def loss_fn(params, x, y):
    preds = LinearRegressionModel(params)(x)
    return jnp.mean((preds - y) ** 2)

@jit
def update(params, x, y, learning_rate):
    grads = grad(loss_fn)(params, x, y)
    updated_params = {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }
    return updated_params

key = random.PRNGKey(42)
key, subkey = random.split(key)
X = random.uniform(subkey, (100, 1), minval=0.0, maxval=10.0)
noise = random.normal(subkey, (100, 1))
y = 2 * X + 3 + noise  

model = LinearRegressionModel(key)

epochs = 1000
learning_rate = 0.01
for epoch in range(epochs):
    model.params = update(model.params, X, y, learning_rate)
    if (epoch + 1) % 100 == 0:
        current_loss = loss_fn(model.params, X, y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

learned_w = model.params['w']
learned_b = model.params['b']
print(f"Learned weight: {learned_w[0, 0]:.4f}, Learned bias: {learned_b[0]:.4f}")

X_test = jnp.array([[4.0], [7.0]])
predictions = model(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
"""

CODE_SNIPPET_2 = """
import jax
import jax.numpy as jnp
from jax import grad, jit, random

def generate_data(num_samples=100):
    key = random.PRNGKey(0)
    X = jnp.linspace(0, 10, num_samples).reshape(-1, 1)
    noise = random.normal(key, shape=X.shape)
    y = 2 * X + 3 + noise  
    return X, y

def model(params, x):
    return jnp.dot(x, params["w"]) + params["b"]

def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

@jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)

@jit
def train_step(params, x, y):
    grads = compute_gradient(params, x, y)
    return {
        "w": params["w"] - 0.01 * grads["w"],
        "b": params["b"] - 0.01 * grads["b"]
    }

def train_model(X, y, num_epochs=1000):
    bound = 1.0  
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
    params = {"w": w, "b": b}
    
    for epoch in range(num_epochs):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        params = {
            "w": params["w"] - 0.01 * grads["w"],
            "b": params["b"] - 0.01 * grads["b"]
        }

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
    return params

def main():
    X, y = generate_data(100)
    learned_params = train_model(X, y)
    learned_w = learned_params["w"][0, 0]
    learned_b = learned_params["b"][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")
    
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()
"""

def compute_bleu(code1, code2):
    """Compute BLEU score between two code snippets."""
    tokens1 = nltk.word_tokenize(code1)
    tokens2 = nltk.word_tokenize(code2)
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([tokens1], tokens2, smoothing_function=smoothie)
    return bleu_score

def compute_codebleu(code1, code2, lang="python"):
    """Compute CodeBLEU score with improved error handling."""
    try:
        result = calc_codebleu([code1], [code2], lang=lang, weights=(0.25, 0.25, 0.25, 0.25))
        return result['codebleu'] if 'codebleu' in result else 0.0
    except Exception as e:
        print(f"CodeBLEU computation failed: {e}")
        tokens1 = set(nltk.word_tokenize(code1))
        tokens2 = set(nltk.word_tokenize(code2))
        overlap = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        return overlap * 0.5

def compute_codebertscore(code1, code2):
    """Compute CodeBERTScore using CodeBERT model."""
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer([code1, code2], return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, 0, :]
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
    return similarity

def check_compilation(code):
    """Check if the code compiles without syntax errors."""
    try:
        ast.parse(code)
        return True, "No syntax errors"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def check_functional_correctness(code, test_cases, snippet_id):
    """Check functional correctness for JAX Linear Regression model."""
    results = []
    try:
        local_namespace = {
            'jax': jax, 'jnp': jnp, 'random': random,
            'grad': jax.grad, 'jit': jax.jit, 'vmap': jax.vmap
        }
        try:
            local_namespace['optax'] = __import__('optax')
        except ImportError:
            pass
        
        exec(code, local_namespace)
        
        generate_data = local_namespace.get('generate_data')
        train_model = local_namespace.get('train_model')
        model = local_namespace.get('model')
        
        if not (generate_data and train_model and model):
            print(f"Snippet {snippet_id}: Missing required functions")
            return 0.0, [False] * len(test_cases)
        
        X, y = generate_data(num_samples=100)
        learned_params = train_model(X, y, num_epochs=1000)
        
        for input_tensor, expected_dict in test_cases:
            try:
                preds = model(learned_params, input_tensor)
                expected = expected_dict[f"snippet{snippet_id}"]
                is_correct = jnp.allclose(preds, expected, atol=1.5)  # Tolerance for noise
                results.append(is_correct)
            except Exception as e:
                print(f"Snippet {snippet_id} Test failed: {e}")
                results.append(False)
    except Exception as e:
        print(f"Snippet {snippet_id} Execution failed: {e}")
        results = [False] * len(test_cases)
    
    success_rate = sum(results) / len(results) if results else 0.0
    return success_rate, results

def main():
    code1 = CODE_SNIPPET_1.strip()
    code2 = CODE_SNIPPET_2.strip()
    
    bleu_score = compute_bleu(code1, code2)
    codebleu_score = compute_codebleu(code1, code2)
    codebert_score = compute_codebertscore(code1, code2)
    
    compiles1, error1 = check_compilation(code1)
    compiles2, error2 = check_compilation(code2)
    
    test_cases = [
        (
            jnp.array([[4.0], [7.0]]),
            {
                "snippet1": jnp.array([9.0, 15.0]),
                "snippet2": jnp.array([11.0, 17.0])
            }
        ),
        (
            jnp.array([[0.0]]),
            {
                "snippet1": jnp.array([1.0]),
                "snippet2": jnp.array([3.0])
            }
        ),
        (
            jnp.array([[10.0]]),
            {
                "snippet1": jnp.array([21.0]),
                "snippet2": jnp.array([23.0])
            }
        ),
        (
            jnp.array([[5.0]]),
            {
                "snippet1": jnp.array([11.0]),
                "snippet2": jnp.array([13.0])
            }
        ),
    ]
    
    func_score1, func_results1 = check_functional_correctness(code1, test_cases, snippet_id=1)
    func_score2, func_results2 = check_functional_correctness(code2, test_cases, snippet_id=2)
    
    print("\n=== Textual Similarity ===")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"CodeBLEU Score: {codebleu_score:.4f}")
    print(f"CodeBERTScore: {codebert_score:.4f}")
    
    print("\n=== Compilation Accuracy ===")
    print(f"Code 1: {'Compiles' if compiles1 else 'Does not compile'} ({error1})")
    print(f"Code 2: {'Compiles' if compiles2 else 'Does not compile'} ({error2})")
    
    print("\n=== Functional Correctness ===")
    print(f"Code 1 Success Rate: {func_score1:.4f}")
    print(f"Code 2 Success Rate: {func_score2:.4f}")
    print("\nTest Case Results:")
    for i, (input_tensor, expected_dict) in enumerate(test_cases, 1):
        print(f"Test {i} (Input: {input_tensor.tolist()})")
        print(f"  Code 1 Expected: {expected_dict['snippet1'].tolist()}, {'Pass' if func_results1[i-1] else 'Fail'}")
        print(f"  Code 2 Expected: {expected_dict['snippet2'].tolist()}, {'Pass' if func_results2[i-1] else 'Fail'}")

if __name__ == "__main__":
    main()