import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import random
import matplotlib.pyplot as plt

# Initialize PRNG key
key = random.PRNGKey(42)


# Define the Autoencoder model in JAX using flax.linen
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Decoder
        x = nn.ConvTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', output_padding=(1, 1))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, kernel_size=(3, 3), strides=(2, 2), padding='SAME', output_padding=(1, 1))(x)
        return nn.sigmoid(x)  # To keep pixel values between 0 and 1


# Loss function (Mean Squared Error)
def loss_fn(params, model, images, targets):
    reconstructed = model.apply({'params': params}, images)
    return jnp.mean((reconstructed - targets) ** 2)


# Optimizer initialization
def create_optimizer(params):
    tx = optax.adam(learning_rate=0.001)
    return tx.init(params)


# Training step
def train_step(params, images, targets, model, optimizer):
    grads = jax.grad(loss_fn)(params, model, images, targets)
    updates, optimizer = optax.adam(learning_rate=0.001).update(grads, optimizer)
    params = optax.apply_updates(params, updates)
    return params, optimizer


# Main function to simulate the training process
def main():
    # Generate synthetic MNIST-like data for the demo (use real MNIST in practice)
    X = random.normal(key, (64, 28, 28, 1))  # Batch of 64 images of size 28x28
    y = X  # Autoencoder target is the input image

    # Initialize the model
    model = Autoencoder()

    # Initialize parameters using PRNG
    params = model.init(key, X)  # Use a dummy input to initialize the parameters

    # Initialize optimizer
    optimizer = create_optimizer(params)

    # Training loop (simplified)
    epochs = 10
    for epoch in range(epochs):
        params, optimizer = train_step(params, X, y, model, optimizer)
        if (epoch + 1) % 1 == 0:  # Print loss every epoch
            current_loss = loss_fn(params, model, X, y)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

    # Plotting (using matplotlib)
    plt.imshow(X[0, :, :, 0], cmap='gray')
    plt.title("Original Image")
    plt.show()


if __name__ == "__main__":
    main()
