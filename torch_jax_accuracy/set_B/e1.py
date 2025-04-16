import jax
import jax.numpy as jnp
import optax
from jax import random, grad, jit
from flax import linen as nn
import numpy as np

# Generate synthetic data (similar to the PyTorch version)
key = random.PRNGKey(42)
key, subkey_X = random.split(key)
X = random.uniform(subkey_X, shape=(100, 1)) * 10
key, subkey_noise = random.split(key)
noise = random.normal(subkey_noise, shape=(100, 1))
y = 2 * X + 3 + noise

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def setup(self):
        self.linear = self.param('linear', nn.initializers.xavier_uniform(), (1, 1))

    def __call__(self, x):
        return jnp.dot(x, self.linear)  # Output prediction

# Loss function: Mean Squared Error
def loss_fn(params, inputs, targets, model):
    predictions = model.apply(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Gradient computation using JAX
def compute_gradients(params, inputs, targets, model):
    return grad(loss_fn)(params, inputs, targets, model)

# Training loop
def update(params, inputs, targets, learning_rate=0.01):
    grads = compute_gradients(params, inputs, targets, model)
    new_params = {k: params[k] - learning_rate * grads[k] for k in params}
    return new_params

# Main function
def main():
    key = random.PRNGKey(42)
    model = LinearRegressionModel()

    # Initialize model parameters
    params = model.init(key, X)

    epochs = 1000
    learning_rate = 0.01

    # Training loop
    for epoch in range(epochs):
        params = update(params, X, y, learning_rate)
        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y, model)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}")

    # Output learned parameters
    learned_weight = params['linear']
    print(f"Learned weight: {learned_weight[0, 0]:.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model.apply(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()
