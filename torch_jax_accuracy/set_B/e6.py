import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
from jax import value_and_grad
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples=100):
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    X = random.uniform(subkey, shape=(num_samples, 1)) * 10
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(num_samples, 1))
    y = 2 * X + 3 + noise
    return X, y

def model(params, x):
    return jnp.dot(x, params['w']) + params['b']

def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

@jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)

@jit
def update(params, x, y, learning_rate=0.01):
    grads = compute_gradient(params, x, y)
    new_params = {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }
    return new_params

def init_params(key):
    bound = 1.0
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
    params = {'w': w, 'b': b}
    return params

def train_model(X, y, num_epochs=1000):
    key = random.PRNGKey(0)
    params = init_params(key)

    for epoch in range(num_epochs):
        params = update(params, X, y)

        if (epoch + 1) % 100 == 0:
            loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

    return params

def main():
    X, y = generate_data(100)

    learned_params = train_model(X, y, num_epochs=1000)

    learned_w = learned_params['w'][0, 0]
    learned_b = learned_params['b'][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

    plt.figure(figsize=(4, 4))
    plt.scatter(X, y, label='Training Data')
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)
    plt.plot(X_line, learned_w * X_line + learned_b, 'r', label='Model Fit')
    plt.legend()
    plt.show()

    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()
