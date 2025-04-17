## Strong LLM
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import flax.linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
# The original PyTorch code applies ToTensor and Normalize((0.5,), (0.5,)).
# When MNIST images (originally uint8 in [0, 255]) are converted to float and normalized,
# the transformation is: x -> (x/255 - 0.5)/0.5 = 2*x/255 - 1.
# We'll do the same here.

def preprocess(example):
    image = example['image']  # shape (28, 28, 1) or (28, 28)
    # Convert to float32 in [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    # Normalize to [-1, 1] as in PyTorch
    image = (image - 0.5) / 0.5
    # Ensure image has channel dimension (H, W, C)
    if image.shape.ndims == 2:
        image = tf.expand_dims(image, axis=-1)
    return {'image': image}

batch_size = 64
# Load and preprocess the training set
train_ds = tfds.load('mnist', split='train', shuffle_files=True)
train_ds = train_ds.map(preprocess)
train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(1)
# (For testing, a similar pipeline can be built from the 'test' split)

# ---------------------------
# Model Definition using Flax
# ---------------------------
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')  # 28->14
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')  # 14->7

        # Decoder
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.sigmoid(x)  # Keep output values between 0 and 1
        return x

# ---------------------------
# Initialize Model and Optimizer
# ---------------------------
rng = random.PRNGKey(0)
# Dummy input with shape [batch, height, width, channels]; note that Flax defaults to NHWC.
dummy_input = jnp.ones([1, 28, 28, 1])
model = Autoencoder()
params = model.init(rng, dummy_input)

# Set up the optimizer (Adam with learning rate 0.001)
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# ---------------------------
# Loss and Training Step
# ---------------------------
def mse_loss(params, batch):
    """Compute mean squared error between the reconstruction and input."""
    recon = model.apply(params, batch)
    return jnp.mean((recon - batch) ** 2)

@jit
def train_step(params, opt_state, images):
    loss, grads = value_and_grad(mse_loss)(params, images)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ---------------------------
# Training Loop
# ---------------------------
epochs = 10
for epoch in range(epochs):
    # Iterate over the TFDS training dataset (convert batches to numpy arrays)
    for batch in tfds.as_numpy(train_ds):
        images = batch['image']  # shape: [batch, 28, 28, 1]
        params, opt_state, loss = train_step(params, opt_state, images)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")
