import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
import numpy as np

# Initialize PRNG key
key = jax.random.PRNGKey(0)  # // MODIFIED: Initialize PRNG key explicitly

# Define the model function
def model(X, key):  # // MODIFIED: Pass PRNG key as a parameter
    w_key, b_key = jax.random.split(key)  # Split key for weights and bias
    w = jax.random.normal(w_key, (1,))  # // MODIFIED: Use PRNG key for randomness
    b = jax.random.normal(b_key, (1,))  # // MODIFIED: Use PRNG key for randomness
    return jnp.dot(X, w) + b

# Jitted function to compute the loss
@jit  # // MODIFIED: Decorate with jit for compilation
def loss_fn(X, y, key):  # // MODIFIED: Pass PRNG key as a parameter
    pred = model(X, key)  # Use key here
    return jnp.mean((pred - y) ** 2)

# Function to perform optimization step
@jit  # // MODIFIED: Ensure this function is stateless
def update(params, X, y, key):
    grads = grad(loss_fn)(X, y, key)  # Compute gradients
    return params - 0.01 * grads  # Simple SGD update

def main():
    # Data preparation
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([[2.0], [4.0], [6.0]])

    # Model fitting
    params = None  # Initialize parameters (could be weights and bias)

    for epoch in range(100):  # Training loop
        params = update(params, X, y, key)  # // MODIFIED: Key passed in updates

    # Visualization
    plt.scatter(X, y, label='Data')
    plt.plot(X, model(X, key), 'r', label='Model Fit')  # // MODIFIED: Key used
    plt.legend()
    plt.show()

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(X_test, key)  # // MODIFIED: Pass key during prediction
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()