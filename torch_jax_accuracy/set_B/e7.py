import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import pickle
import matplotlib.pyplot as plt

# Define a simple model with a linear layer
def model(params, x):
    return jnp.dot(x, params['w']) + params['b']

# Loss function (MSE)
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Gradient computation
@jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)

# Training step
@jit
def train_step(params, x, y, learning_rate=0.01):
    grads = compute_gradient(params, x, y)
    new_params = {k: params[k] - learning_rate * grads[k] for k in params}
    return new_params

# Main function to perform training and model saving/loading
def main():
    # Generate synthetic data (with added noise)
    key = random.PRNGKey(42)
    key, subkey_x, subkey_y = random.split(key, 3)
    X = random.uniform(subkey_x, (100, 1)) * 10  # 100 samples
    noise = random.normal(subkey_y, (100, 1)) * 0.1
    y = 3 * X + 2 + noise

    # Initialize parameters
    params = {
        'w': random.normal(key, (1,)),  # Initialize weights
        'b': random.normal(key, (1,))   # Initialize bias
    }

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        params = train_step(params, X, y)  # Update parameters
        if epoch % 10 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch}/{epochs}], Loss: {current_loss:.4f}")

    # Save model to file
    with open("model.pth", "wb") as f:
        pickle.dump(params, f)

    # Load the model back
    with open("model.pth", "rb") as f:
        loaded_params = pickle.load(f)

    # Testing on new data
    X_test = jnp.array([[0.5], [1.0], [1.5]])
    predictions = model(loaded_params, X_test)
    print(f"Predictions after loading: {predictions}")

    # Visualization (optional)
    plt.scatter(X, y, label='Data')
    plt.plot(X, model(loaded_params, X), 'r', label='Model Fit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
