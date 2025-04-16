import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from jax import random, grad, jit


# Generate synthetic data
def generate_data(num_samples=100):
    key = random.PRNGKey(0)
    X = jax.random.uniform(key, shape=(num_samples, 1)) * 10
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(num_samples, 1))
    y = 2 * X + 3 + noise
    return X, y


# Define the Linear Regression Model
def model(params, X):
    return jnp.dot(X, params["w"]) + params["b"]


# Loss function
def loss_fn(params, X, y):
    preds = model(params, X)
    return jnp.mean((preds - y) ** 2)


# Gradient computation
@jit
def compute_gradient(params, X, y):
    return grad(loss_fn)(params, X, y)


# Training step
@jit
def train_step(params, X, y, lr=0.01):
    grads = compute_gradient(params, X, y)
    new_params = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"]
    }
    return new_params


# Initialize model parameters
def init_params(key):
    bound = 1.0
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
    return {"w": w, "b": b}


# Main function
def main():
    key = random.PRNGKey(42)
    X, y = generate_data(100)

    # Initialize parameters
    params = init_params(key)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        params = train_step(params, X, y, lr=0.01)
        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

    # Display the learned parameters
    learned_w = params["w"][0, 0]
    learned_b = params["b"][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

    # Plot the model fit to the train data
    plt.figure(figsize=(4, 4))
    plt.scatter(X, y, label='Training Data')
    plt.plot(X, learned_w * X + learned_b, 'r', label='Model Fit')
    plt.legend()
    plt.show()

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()
