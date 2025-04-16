import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
import optax
import torchvision.transforms as transforms
from torchvision import datasets

# Define your model using flax.linen
class SimpleModel(nn.Module):
    features: int

    def setup(self):
        self.dense = nn.Dense(self.features)

    def __call__(self, x):
        return self.dense(x)

# Initialization of weights and bias instead of global variables
def initialize_params(key, input_shape):
    model = SimpleModel(features=10)  # Specify the number of features
    params = model.init(key, jnp.ones(input_shape))
    return params

@jit
def train_step(params, x_batch, y_batch, key):
    model = SimpleModel(features=10)
    # Forward pass
    predictions = model.apply(params, x_batch)
    loss = jnp.mean((predictions - y_batch) ** 2)  # Example loss function
    # Compute gradients
    grads = grad(lambda p: jnp.mean((model.apply(p, x_batch) - y_batch) ** 2))(params)
    # Update parameters using an optimizer
    return params - 0.01 * grads  # Example learning rate of 0.01

def train(data, targets, num_epochs, key):
    input_shape = (data.shape[0], data.shape[1])  # Assuming data has shape (batch_size, features)
    params = initialize_params(key, input_shape)
    
    for epoch in range(num_epochs):
        for x_batch, y_batch in zip(data, targets):
            key, subkey = random.split(key)  # Split the key for randomness
            params = train_step(params, x_batch, y_batch, subkey)
    return params

def main():
    # Random PRNG key initialization
    key = random.PRNGKey(0)  # MODIFIED: Initialize random key

    # Example dataset initialization
    # Here you should load your dataset
    data = jnp.array(...)  # Replace with actual data loading logic
    targets = jnp.array(...)  # Replace with actual target loading logic
    num_epochs = 10  # Set the number of epochs

    train(data, targets, num_epochs, key)  # Call the train function

if __name__ == "__main__":
    main()  # Entry point