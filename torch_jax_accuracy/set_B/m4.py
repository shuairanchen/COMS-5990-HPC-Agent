import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn

# Initialize PRNG key
key = random.PRNGKey(42)

# Generate synthetic CT-scan data (batches, slices, RGB)
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

key, subkey = random.split(key)
ct_images = random.normal(subkey, shape=(batch, num_slices, channels, width, height))
segmentation_masks = (random.normal(subkey, shape=(batch, num_slices, 1, width, height)) > 0).astype(jnp.float32)

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

class MedCNN(nn.Module):
    def __init__(self, backbone, out_channel=1):
        super().__init__()
        self.backbone = backbone
        self.conv1 = nn.Conv(64, (3, 3, 3), padding='SAME')
        self.conv2 = nn.Conv(64, (3, 3, 3), padding='SAME')
        self.conv_transpose1 = nn.ConvTranspose(32, (1, 4, 4), strides=(1, 4, 4))
        self.conv_transpose2 = nn.ConvTranspose(16, (1, 8, 8), strides=(1, 8, 8))
        self.final_conv = nn.Conv(out_channel, (1, 1, 1))

    def __call__(self, x):
        b, d, c, w, h = x.shape
        x = x.reshape((b * d, c, w, h))
        features = self.backbone(x)
        _, new_c, new_w, new_h = features.shape
        x = features.reshape((b, d, new_c, new_w, new_h))
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv_transpose1(x))
        x = nn.relu(self.conv_transpose2(x))
        x = jax.nn.sigmoid(self.final_conv(x))
        return x

def compute_dice_loss(pred, labels, eps=1e-8):
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator

# Define the ResNet backbone model (as before)
# Replace this with an actual JAX-based backbone for the model
# Just a placeholder here as this part isn't directly translatable from PyTorch.
resnet_model = nn.Sequential(*[nn.Dense(512), nn.Dense(256)])

model = MedCNN(backbone=resnet_model)
params = model.init(key, ct_images)

# Optimizer and training loop
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

def update(params, ct_images, segmentation_masks, opt_state):
    loss, grads = jax.value_and_grad(compute_dice_loss)(params, ct_images, segmentation_masks)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
epochs = 5
for epoch in range(epochs):
    params, opt_state, loss = update(params, ct_images, segmentation_masks, opt_state)
    print(f"Loss at epoch {epoch}: {loss}")
