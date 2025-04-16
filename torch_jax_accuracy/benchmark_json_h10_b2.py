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
import flax.linen as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import difflib
import time
# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
except:
    nltk.download('punkt', quiet=True)

# Code snippets
CODE_SNIPPET_1 = """
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ResNet18(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))
        x = nn.Conv(128, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(256, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        return x

model = ResNet18()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)))

def model_fn(params, x):
    return model.apply(params, x)

def compute_gradients_and_activations(model_fn, params, x, class_idx):
    grads = grad(model_fn)(params, x)
    return grads

image = jnp.ones((1, 224, 224, 3))

output = model_fn(params, image)
predicted_class = output.argmax(axis=-1).item()

grads = compute_gradients_and_activations(model_fn, params, image, predicted_class)

weights = grads.mean(axis=(1, 2), keepdims=True)
activations = output[0]
heatmap = (weights * activations).sum(axis=-1).squeeze()

heatmap = heatmap - heatmap.min()
heatmap = heatmap / heatmap.max()
heatmap = np.uint8(255 * heatmap)

heatmap_image = Image.fromarray(heatmap)
heatmap_image = heatmap_image.resize((224, 224), Image.BILINEAR)

plt.imshow(image[0])
plt.imshow(heatmap_image, alpha=0.5, cmap='jet')
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()
"""

CODE_SNIPPET_2 = """
import jax
import jax.numpy as jnp
import jax.nn as nn
import flax.linen as nn
import numpy as np
import torchvision.transforms as transforms
from flax import serialization
import matplotlib.pyplot as plt
from PIL import Image

class SomeLayer(nn.Module):
    features: int
    key: jax.random.PRNGKey
    @nn.compact
    def __call__(self, x):
        subkey = self.make_rng('rng')
        return nn.relu(nn.Dense(self.features)(x))

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = jax.random.split(key)
    return jax.random.normal(subkey, shape, dtype=dtype)

def main():
    key = jax.random.PRNGKey(0)
    input_tensor_shape = (10, 10)
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer(features=5, key=key)
    params = layer.init({'params': key, 'rng': key}, input_tensor)
    output = layer.apply(params, input_tensor, rngs={'rng': key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)
    output_img = Image.fromarray(output_np.astype(np.uint8))
    heatmap = transforms.Resize(image.size)(output_img)
    
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

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

def compute_ast_similarity(code1, code2):
    """Compute AST similarity between two code snippets."""
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        ast1_str = ast.dump(tree1, indent=2)
        ast2_str = ast.dump(tree2, indent=2)
        sm = difflib.SequenceMatcher(None, ast1_str, ast2_str)
        return sm.ratio()
    except Exception as e:
        print(f"AST comparison failed: {e}")
        return 0.0
    
def measure_execution_time(code, snippet_id):
    """Measure execution time of the code."""
    try:
        local_namespace = {
            'jax': jax, 'jnp': jnp, 'jnn': nn, 'nn': nn, 'random': random,
            'np': np, 'transforms': transforms, 'Image': Image, 'plt': plt,
            'torch': torch
        }
        try:
            local_namespace['optax'] = __import__('optax')
            local_namespace['flax'] = __import__('flax')
            local_namespace['torchvision'] = __import__('torchvision')
        except ImportError:
            pass
        
        start_time = time.time()
        exec(code, local_namespace)
        end_time = time.time()
        return end_time - start_time, "Execution successful"
    except Exception as e:
        return float('inf'), f"Execution failed: {e}"
    
def check_functional_correctness(code, test_cases, snippet_id):
    """Check functional correctness for neural network snippets."""
    results = []
    try:
        local_namespace = {
            'jax': jax, 'jnp': jnp, 'jnn': nn, 'nn': nn, 'random': random,
            'np': np, 'transforms': transforms, 'Image': Image, 'plt': plt,
            'torch': torch
        }
        try:
            local_namespace['optax'] = __import__('optax')
            local_namespace['flax'] = __import__('flax')
            local_namespace['torchvision'] = __import__('torchvision')
        except ImportError:
            pass
        
        exec(code, local_namespace)
        
        SomeLayer = local_namespace.get('SomeLayer')
        generate_random_tensor = local_namespace.get('generate_random_tensor')
        if not (SomeLayer and generate_random_tensor):
            print(f"Snippet {snippet_id}: Missing required components")
            return 0.0, [False] * len(test_cases)
        
        for test_input, expected in test_cases:
            try:
                key = random.PRNGKey(0)
                input_tensor = generate_random_tensor(
                    test_input['shape'], jnp.float32, key
                )
                layer = SomeLayer(features=expected['output_shape'][1], key=key)
                
                # Initialize and apply layer
                if snippet_id == 1:
                    output = layer(input_tensor)
                else:
                    params = layer.init({'params': key, 'rng': key}, input_tensor)
                    output = layer.apply(params, input_tensor, rngs={'rng': key})
                
                # Generate heatmap
                output_np = np.array(output)
                # Reshape output to a square 2D array for heatmap (e.g., pad or reshape)
                size = int(np.ceil(np.sqrt(output_np.size)))
                output_2d = np.pad(
                    output_np.ravel(),
                    (0, size * size - output_np.size),
                    mode='constant'
                ).reshape(size, size)
                # Normalize to 0-255 for valid image
                output_2d = (output_2d - output_2d.min()) / (output_2d.max() - output_2d.min() + 1e-8) * 255
                output_2d = output_2d.astype(np.uint8)
                output_img = Image.fromarray(output_2d, mode='L')  # Grayscale image
                dummy_image = Image.fromarray(
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                )
                heatmap = transforms.Resize(dummy_image.size)(output_img)
                heatmap_np = np.array(heatmap)
                
                # Check expectations
                is_correct = (
                    output.shape == expected['output_shape'] and
                    jnp.all(output >= 0) == expected['non_negative_output'] and
                    not jnp.isnan(output).any() and
                    heatmap_np.shape == expected['heatmap_size']
                )
                results.append(is_correct)
            except Exception as e:
                print(f"Snippet {snippet_id} Test {len(results)+1} failed: {e}")
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
    
    ast_similarity = compute_ast_similarity(code1, code2)
    print(f"AST Similarity: {ast_similarity:.4f}")

    exec_time1, exec_msg1 = measure_execution_time(code1, snippet_id=1)
    exec_time2, exec_msg2 = measure_execution_time(code2, snippet_id=2)

    test_cases = [
        (
            {'shape': (10, 10)},
            {
                'output_shape': (10, 5),
                'heatmap_size': (100, 100),
                'non_negative_output': True
            }
        ),
        (
            {'shape': (5, 10)},
            {
                'output_shape': (5, 5),
                'heatmap_size': (100, 100),
                'non_negative_output': True
            }
        ),
        (
            {'shape': (20, 10)},
            {
                'output_shape': (20, 5),
                'heatmap_size': (100, 100),
                'non_negative_output': True
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

    print("\n=== Execution Time ===")
    print(f"Code 1: {exec_time1:.4f} seconds ({exec_msg1})")
    print(f"Code 2: {exec_time2:.4f} seconds ({exec_msg2})")
    
    print("\n=== Functional Correctness ===")
    print(f"Code 1 Success Rate: {func_score1:.4f}")
    print(f"Code 2 Success Rate: {func_score2:.4f}")
    print("\nTest Case Results:")
    for i, (test_input, expected) in enumerate(test_cases, 1):
        print(f"Test {i} (Input shape: {test_input['shape']})")
        print(f"  Code 1 Expected: heatmap_size={expected['heatmap_size']}, {'Pass' if func_results1[i-1] else 'Fail'}")
        print(f"  Code 2 Expected: output_shape={expected['output_shape']}, heatmap_size={expected['heatmap_size']}, {'Pass' if func_results2[i-1] else 'Fail'}")

if __name__ == "__main__":
    main()