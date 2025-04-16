import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from jax import grad, jit, value_and_grad
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from torchvision.datasets import FakeData

class ResNet18(nn.Module):
    def setup(self):
        self.model = flax.models.ResNet18()

    def __call__(self, x):
        return self.model(x)

gradients = None
activations = None

def save_activations(model, x):
    global activations
    activations = model(x)

def save_gradients(grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def grad_cam(model, image):
    def forward_hook_fn(model, x):
        save_activations(model, x)
        return x

    def backward_hook_fn(grad_in, grad_out):
        save_gradients(grad_in, grad_out)

    output = model(image)
    predicted_class = jnp.argmax(output, axis=1)
    loss = output[0, predicted_class]
    grads = grad(loss)(output)
    weights = gradients.mean(axis=(2, 3), keepdims=True)
    heatmap = (weights * activations).sum(axis=1).squeeze().relu()

    heatmap = heatmap / heatmap.max()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)
    return heatmap, predicted_class

dataset = FakeData(transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))
image, _ = dataset[0]
image = transforms.ToPILImage()(image)

input_tensor = preprocess_image(image)

model = ResNet18()
params = model.init(jax.random.PRNGKey(0), input_tensor)

heatmap, predicted_class = grad_cam(model, input_tensor)

plt.imshow(image)
plt.imshow(heatmap, alpha=0.5, cmap='jet')
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()