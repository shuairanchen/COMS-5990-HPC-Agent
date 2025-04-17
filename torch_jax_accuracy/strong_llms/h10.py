## Strong LLM
!pip install flax
!pip install flaxmodels
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import flaxmodels
import matplotlib.pyplot as plt
from PIL import Image

# --- Helper: Image Preprocessing ---
def preprocess_image(image):
    """
    Resize image to 224x224, convert to a float32 array in [0,1],
    normalize using ImageNet means and stds, and add a batch dimension.
    (Note: JAX/Flax models typically expect NHWC layout.)
    """
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0  # shape (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return jnp.expand_dims(img, axis=0)  # shape (1, H, W, C)

# --- Load Pre-trained ResNet-18 in Flax ---
# Here we use flaxmodels which provides pre-trained weights similar to torchvision.
model = flaxmodels.ResNet18(pretrained=True)
# Initialize model parameters using a dummy input.
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
variables = model.init(key, dummy_input)
params = variables['params']

# --- Define Functions for Grad-CAM ---
# For Grad-CAM we want to “split” the network.
# We assume that our model can be decomposed into a feature_extractor and classifier_head.
# (If not, you can modify the model to output the intermediate features you need.)

def feature_extractor(params, x):
    """
    Applies the model up to the target layer. In this example we assume
    that flaxmodels.ResNet18 exposes a method `extract_features` that returns
    the activation maps from the target layer (here analogous to PyTorch’s
    model.layer4[1].conv2).
    """
    # The following call is library‑dependent. In your case, you may need to modify it.
    features = model.extract_features({'params': params}, x)
    return features  # expected shape: (1, H, W, C)

def classifier_head(params, features):
    """
    Applies the rest of the model (global pooling, fully-connected layer, etc.)
    to convert the feature maps into logits.
    """
    logits = model.classify_features({'params': params}, features)
    return logits

# --- Get a Sample Image ---
# In PyTorch the sample came from FakeData. Here we simulate with a random image.
image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

# Preprocess the image
input_tensor = preprocess_image(image)

# --- Forward Pass: Get Prediction ---
# We run the full model to obtain the logits.
logits = model.apply({'params': params}, input_tensor)
predicted_class = int(jnp.argmax(logits, axis=-1)[0])
print("Predicted class:", predicted_class)

# Also run the feature extractor to obtain the target activation maps.
target_activation = feature_extractor(params, input_tensor)  # e.g. shape (1, H, W, C)

# --- Backward Pass: Compute Gradients for Grad-CAM ---
# Define a function that gives the score (logit) for the predicted class,
# given the intermediate activations.
def score_from_activation(act):
    # Pass the feature maps through the classifier head.
    logits = classifier_head(params, act)
    # Return the logit for the previously predicted class.
    return logits[0, predicted_class]

# Compute the gradient of the score with respect to the target activation.
grad_fn = jax.grad(score_from_activation)
gradients = grad_fn(target_activation)  # same shape as target_activation

# --- Compute Grad-CAM Heatmap ---
# In PyTorch the weights are computed as the average over spatial dims (channels in C,H,W).
# Here we assume target_activation is in NHWC so we average over H and W (axes 1 and 2).
weights = jnp.mean(gradients, axis=(1, 2), keepdims=True)  # shape (1, 1, 1, C)

# Compute the weighted combination of the feature maps.
cam = jnp.sum(weights * target_activation, axis=-1)  # shape (1, H, W)
cam = jnp.maximum(cam, 0.0)  # Apply ReLU
cam = cam[0]  # Remove batch dimension
cam = cam / (jnp.max(cam) + 1e-8)  # Normalize between 0 and 1

# --- Convert Heatmap for Visualization ---
cam_np = np.array(cam)  # convert to NumPy array
# Convert the normalized heatmap to an image and resize to the original image size.
heatmap = Image.fromarray(np.uint8(255 * cam_np))
heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)

# --- Display the Image with Grad-CAM Overlay ---
plt.imshow(image)
plt.imshow(heatmap, alpha=0.5, cmap='jet')
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()
