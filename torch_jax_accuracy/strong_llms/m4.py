## Strong LLM
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state

# -------------------------------
# Synthetic Data Generation
# -------------------------------
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

# Create PRNG keys
key = jax.random.PRNGKey(42)
key, subkey1, subkey2 = jax.random.split(key, 3)

# Generate synthetic CT images and binary segmentation masks
ct_images = jax.random.normal(subkey1, (batch, num_slices, channels, width, height))
segmentation_masks = (jax.random.normal(subkey2, (batch, num_slices, 1, width, height)) > 0).astype(jnp.float32)

print("CT images (train examples) shape:", ct_images.shape)
print("Segmentation binary masks (labels) shape:", segmentation_masks.shape)

# -------------------------------
# Dummy ResNet18 Backbone Definition
# -------------------------------
class DummyResNet18(nn.Module):
    """
    Mimics a truncated ResNet18:
    Given an input of shape [B*D, 3, 256, 256],
    returns a feature map of shape [B*D, 512, new_w, new_h],
    where new_w and new_h are ~8 (downsampling via stride=32).
    """
    @nn.compact
    def __call__(self, x):
        # Using a single convolution with stride 32 to simulate the downsampling:
        x = nn.Conv(features=512,
                    kernel_size=(7, 7),
                    strides=(32, 32),
                    padding='SAME',
                    dimension_numbers=('NCHW', 'OIHW', 'NCHW'))(x)
        x = nn.relu(x)
        return x

# -------------------------------
# MedCNN Model Definition in Flax
# -------------------------------
class MedCNN(nn.Module):
    backbone: nn.Module
    out_channel: int = 1

    @nn.compact
    def __call__(self, x):
        # x shape: [B, D, C, W, H]
        b, d, c, w, h = x.shape
        print("Input shape [B, D, C, W, H]:", (b, d, c, w, h))

        # Reshape for backbone (2D convolutions): [B*D, C, W, H]
        x = x.reshape((b * d, c, w, h))
        features = self.backbone(x)
        print("Backbone (ResNet) output shape [B*D, C, W, H]:", features.shape)

        # Get new dimensions and reshape back: [B, D, new_c, new_w, new_h]
        _, new_c, new_w, new_h = features.shape
        x = features.reshape((b, d, new_c, new_w, new_h))
        # Permute to [B, new_c, D, new_w, new_h] for 3D convolutions
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        print("Reshaped for 3D conv [B, C, D, W, H]:", x.shape)

        # Define dimension numbers for 3D convolutions (NCDHW ordering)
        dim3d = ("NCDHW", "OIDHW", "NCDHW")

        # Downsampling 3D convolutions:
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding='SAME', dimension_numbers=dim3d)(x)
        x = nn.relu(x)
        print("After 3D Conv #1:", x.shape)

        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding='SAME', dimension_numbers=dim3d)(x)
        x = nn.relu(x)
        print("After 3D Conv #2:", x.shape)

        # Upsampling 3D transposed convolutions:
        x = nn.ConvTranspose(features=32, kernel_size=(1, 4, 4), strides=(1, 4, 4),
                             padding='SAME', dimension_numbers=dim3d)(x)
        x = nn.relu(x)
        print("After 3D Transposed Conv #1:", x.shape)

        x = nn.ConvTranspose(features=16, kernel_size=(1, 8, 8), strides=(1, 8, 8),
                             padding='SAME', dimension_numbers=dim3d)(x)
        x = nn.relu(x)
        print("After 3D Transposed Conv #2:", x.shape)

        # Final segmentation layer (from 16 to 1 channel)
        x = nn.Conv(features=self.out_channel, kernel_size=(1, 1, 1), padding='SAME', dimension_numbers=dim3d)(x)
        x = jax.nn.sigmoid(x)
        print("Final output shape:", x.shape)
        return x

# -------------------------------
# Dice Loss Function
# -------------------------------
def compute_dice_loss(pred, labels, eps=1e-8):
    """
    Args:
      pred: [B, D, 1, W, H]
      labels: [B, D, 1, W, H]

    Returns:
      Dice coefficient (scalar)
    """
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator

# -------------------------------
# Set Up Model and Training State
# -------------------------------
# Instantiate the backbone and MedCNN model.
backbone = DummyResNet18()
model = MedCNN(backbone=backbone)

# Create a training state using Flax's TrainState and Optax's Adam optimizer.
class TrainState(train_state.TrainState):
    pass

# Initialize model parameters using the synthetic data.
rng = jax.random.PRNGKey(0)
params = model.init(rng, ct_images)
tx = optax.adam(learning_rate=0.01)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# -------------------------------
# Training Loop
# -------------------------------
epochs = 5
for epoch in range(epochs):
    def loss_fn(params):
        pred = model.apply(params, ct_images)
        loss = compute_dice_loss(pred, segmentation_masks)
        return loss, pred

    # Compute loss and gradients.
    (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Update the model parameters.
    state = state.apply_gradients(grads=grads)
    print(f"Loss at epoch {epoch}: {loss}")
