import jax
import jax.numpy as jnp
import optax
from jax import random
from flax import linen as nn

# Define the CNN model using flax.linen
class VanillaCNNModel(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv2 = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.pool = nn.max_pool
        self.fc1 = nn.Dense(128)
        self.fc2 = nn.Dense(10)
        self.relu = nn.relu

    def __call__(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)), window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Loss function (Huber Loss)
def huber_loss(params, X, y, delta=1.0):
    preds = model(params, X)
    error = jnp.abs(preds - y)
    loss = jnp.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return jnp.mean(loss)

# Training step
def train_step(params, X, y, optimizer):
    loss, grads = jax.value_and_grad(huber_loss)(params, X, y)
    new_params = optimizer.apply_updates(params, grads)
    return new_params, loss

# Model training loop
def train_model(model, train_loader, epochs=10):
    optimizer = optax.adam(learning_rate=0.001)
    params = model.init(rng, X)  # Initialize parameters
    for epoch in range(epochs):
        for batch in train_loader:
            params, loss = train_step(params, batch['x'], batch['y'], optimizer)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")
