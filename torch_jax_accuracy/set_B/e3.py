import jax
import jax.numpy as jnp
from jax import random, grad
from flax import linen as nn
import optax
from tensorboardX import SummaryWriter
import numpy as np

# Linear regression model definition
class LinearRegressionModel(nn.Module):
    input_dim: int

    def setup(self):
        self.w = self.param('w', nn.initializers.xavier_uniform(), (self.input_dim, 1))
        self.b = self.param('b', nn.initializers.zeros, (1,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

# Loss function
def loss_fn(params, inputs, targets, model):
    predictions = model.apply(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Jitted gradient computation using vectorization
def compute_gradients(params, inputs, targets, model):
    return grad(loss_fn)(params, inputs, targets, model)

# Training function
def train_model(model, inputs, targets, epochs=1000, learning_rate=0.01):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)

    for epoch in range(epochs):
        grads = compute_gradients(model.params, inputs, targets, model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model.params = optax.apply_updates(model.params, updates)

        if epoch % 100 == 0:
            current_loss = loss_fn(model.params, inputs, targets, model)
            print(f"Epoch {epoch}, Loss: {current_loss}")

    return model

# Main function
def main():
    # Generate synthetic data
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    X = random.uniform(subkey, shape=(100, 2), minval=0.0, maxval=1.0) * 10
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(100, 1))
    y = (X[:, 0:1] + X[:, 1:2] * 2) + noise

    # Initialize model
    model = LinearRegressionModel(input_dim=2)

    # Train the model
    trained_model = train_model(model, X, y)

    # Testing on new data
    X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
    predictions = trained_model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()  # Entry point of the program
