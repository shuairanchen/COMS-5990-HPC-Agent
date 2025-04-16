import jax
import jax.numpy as jnp
import optax
from jax import random
import matplotlib.pyplot as plt

# Initialize PRNG key
key = random.PRNGKey(0)

# Define the Generator model
class Generator:
    def __init__(self, input_dim, output_dim, key):
        # Initialize weights and biases using the PRNG key
        keys = random.split(key, 2)
        self.fc1_w = random.uniform(keys[0], shape=(input_dim, 128), minval=-1.0, maxval=1.0)
        self.fc1_b = random.uniform(keys[0], shape=(128,), minval=-1.0, maxval=1.0)
        self.fc2_w = random.uniform(keys[1], shape=(128, 256), minval=-1.0, maxval=1.0)
        self.fc2_b = random.uniform(keys[1], shape=(256,), minval=-1.0, maxval=1.0)
        self.fc3_w = random.uniform(keys[1], shape=(256, output_dim), minval=-1.0, maxval=1.0)
        self.fc3_b = random.uniform(keys[1], shape=(output_dim,), minval=-1.0, maxval=1.0)

    def forward(self, x):
        # Forward pass: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh
        x = jnp.dot(x, self.fc1_w) + self.fc1_b
        x = jax.nn.relu(x)
        x = jnp.dot(x, self.fc2_w) + self.fc2_b
        x = jax.nn.relu(x)
        x = jnp.dot(x, self.fc3_w) + self.fc3_b
        return jnp.tanh(x)

# Define the Discriminator model
class Discriminator:
    def __init__(self, input_dim, key):
        # Initialize weights and biases using the PRNG key
        keys = random.split(key, 2)
        self.fc1_w = random.uniform(keys[0], shape=(input_dim, 256), minval=-1.0, maxval=1.0)
        self.fc1_b = random.uniform(keys[0], shape=(256,), minval=-1.0, maxval=1.0)
        self.fc2_w = random.uniform(keys[1], shape=(256, 128), minval=-1.0, maxval=1.0)
        self.fc2_b = random.uniform(keys[1], shape=(128,), minval=-1.0, maxval=1.0)
        self.fc3_w = random.uniform(keys[1], shape=(128, 1), minval=-1.0, maxval=1.0)
        self.fc3_b = random.uniform(keys[1], shape=(1,), minval=-1.0, maxval=1.0)

    def forward(self, x):
        # Forward pass: Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear -> Sigmoid
        x = jnp.dot(x, self.fc1_w) + self.fc1_b
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = jnp.dot(x, self.fc2_w) + self.fc2_b
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = jnp.dot(x, self.fc3_w) + self.fc3_b
        return jax.nn.sigmoid(x)

# Generator and Discriminator forward pass
latent_dim = 10
data_dim = 1
G = Generator(latent_dim, data_dim, key)
D = Discriminator(data_dim, key)

# Generate synthetic data
real_data = jnp.array(random.uniform(key, (100, 1)) * 2 - 1)

# Training loop (simplified for illustration)
epochs = 1000
for epoch in range(epochs):
    # Train Discriminator
    latent_samples = random.normal(key, (real_data.shape[0], latent_dim))
    fake_data = G.forward(latent_samples)

    real_labels = jnp.ones(real_data.shape[0])
    fake_labels = jnp.zeros(real_data.shape[0])

    real_loss = jnp.mean((D.forward(real_data) - real_labels) ** 2)
    fake_loss = jnp.mean((D.forward(fake_data) - fake_labels) ** 2)
    loss_D = real_loss + fake_loss

    # Train Generator
    latent_samples = random.normal(key, (real_data.shape[0], latent_dim))
    fake_data = G.forward(latent_samples)
    loss_G = jnp.mean((D.forward(fake_data) - real_labels) ** 2)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

# Generate new samples with the trained Generator
latent_samples = random.normal(key, (5, latent_dim))
generated_data = G.forward(latent_samples)
print(f"Generated data: {generated_data.tolist()}")
