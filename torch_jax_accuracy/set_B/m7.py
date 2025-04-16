import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

# Load MNIST dataset in JAX (manually handle data loading)
def load_data():
    # Replace with actual data loading code if needed, here I simulate the data
    X = np.random.randn(60000, 28*28)  # 60000 samples, 28*28 pixels flattened
    y = np.random.randint(0, 10, size=(60000,))
    return X, y

# Define the simple neural network model in JAX
class SimpleNN:
    def __init__(self, key):
        self.params = self.init_params(key)

    def init_params(self, key):
        keys = random.split(key, 2)
        w1 = random.normal(keys[0], (28*28, 128))  # Weights for the first layer
        b1 = jnp.zeros((128,))
        w2 = random.normal(keys[1], (128, 10))  # Weights for the second layer (output)
        b2 = jnp.zeros((10,))
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def __call__(self, x):
        x = jnp.dot(x, self.params["w1"]) + self.params["b1"]
        x = jax.nn.relu(x)
        return jnp.dot(x, self.params["w2"]) + self.params["b2"]

# Define the loss function (CrossEntropy Loss)
def loss_fn(params, X, y):
    logits = model(X)
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=1))

# Initialize model, loss function, and optimizer
key = random.PRNGKey(0)
model = SimpleNN(key)
learning_rate = 0.01
optimizer = optax.sgd(learning_rate)
opt_state = optimizer.init(model.params)

# Training loop
epochs = 5
X_train, y_train = load_data()  # Load data

for epoch in range(epochs):
    # Simulate a batch of data for training
    batch_size = 64
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(model.params, X_batch, y_batch)

        # Update parameters using gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        model.params = optax.apply_updates(model.params, updates)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Testing loop
X_test = np.random.randn(100, 28*28)  # 100 test samples
y_test = np.random.randint(0, 10, size=(100,))
logits = model(X_test)
predictions = jnp.argmax(logits, axis=1)
accuracy = jnp.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
