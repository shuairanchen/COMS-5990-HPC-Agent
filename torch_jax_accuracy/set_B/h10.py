import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import grad, jit, random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the CNN Model in Flax (Simplified version)
class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (3, 3), padding="SAME")(x)  # Simplified convolution layer
        x = nn.relu(x)
        x = nn.Dense(10)(x)  # Final dense layer (for class prediction)
        return x

# Function to compute the loss
def loss_fn(params, model, X, y):
    preds = model.apply({'params': params}, X)  # Get predictions
    return jnp.mean((preds - y) ** 2)  # Mean Squared Error Loss

# Grad-CAM implementation in JAX
def grad_cam(model, params, X, target_class):
    # Get the output of the model and the gradients w.r.t the last convolution layer
    def compute_loss(params, X, y):
        preds = model.apply({'params': params}, X)
        return jnp.mean((preds - y) ** 2)

    grads = grad(compute_loss)(params, X, target_class)
    return grads

# Generate synthetic data (for testing)
key = random.PRNGKey(0)
X = random.uniform(key, shape=(1, 224, 224, 3))  # Example input data (224x224 RGB image)
y = jnp.array([[1]])  # Example target (class label)

# Initialize the model and parameters
model = CNNModel()
params = model.init(key, X)

# Perform a forward pass
output = model.apply({'params': params}, X)
predicted_class = output.argmax()

# Compute Grad-CAM for the predicted class
grads = grad_cam(model, params, X, y)

# Visualize the Grad-CAM output (simplified)
heatmap = grads.mean(axis=(1, 2))  # Averaging over spatial dimensions for simplicity
heatmap = jnp.maximum(heatmap, 0)  # ReLU to keep positive values
heatmap = heatmap / jnp.max(heatmap)  # Normalize the heatmap

# Overlay heatmap on the image
plt.imshow(X[0, :, :, :], alpha=0.7)  # Original image
plt.imshow(heatmap, alpha=0.5, cmap='jet')  # Grad-CAM heatmap
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()
