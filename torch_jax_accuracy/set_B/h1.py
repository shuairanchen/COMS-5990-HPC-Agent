import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random

# Generate synthetic data with noise
key = random.PRNGKey(42)
key, subkey_X = random.split(key)
X = random.uniform(subkey_X, shape=(100, 1)) * 10
key, subkey_noise = random.split(key)
noise = random.normal(subkey_noise, shape=(100, 1))
y = 2 * X + 3 + noise  # Linear relationship with noise

# Define a custom activation function similar to SiLU (sigmoid-weighted linear unit)
def custom_activation(x):
    return jnp.tanh(x) + x

# Initialize model parameters
def init_params(key):
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-1.0, maxval=1.0)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
    return {'w': w, 'b': b}

# Model function using the learned parameters
def model(params, X):
    linear_output = jnp.dot(X, params['w']) + params['b']
    return custom_activation(linear_output)

# Loss function - using MSE for simplicity, could also apply Huber loss
def loss_fn(params, X, y):
    preds = model(params, X)
    return jnp.mean((preds - y) ** 2)

# Update function using gradient descent
def update(params, X, y, learning_rate=0.01):
    grads = jax.grad(loss_fn)(params, X, y)
    new_params = {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }
    return new_params

# Training loop
def train_model(params, X, y, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        params = update(params, X, y, learning_rate)
        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")
    return params

# Main function to simulate training and testing
def main():
    key = random.PRNGKey(0)  # Initialize PRNG key
    params = init_params(key)

    # Train the model
    params = train_model(params, X, y, epochs=1000, learning_rate=0.01)

    # Display learned parameters
    learned_w = params['w'][0, 0]
    learned_b = params['b'][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

    # Plotting the model fit
    plt.scatter(X, y, label='Training Data')
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)
    plt.plot(X_line, learned_w * X_line + learned_b, 'r', label='Model Fit')
    plt.legend()
    plt.show()

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()
