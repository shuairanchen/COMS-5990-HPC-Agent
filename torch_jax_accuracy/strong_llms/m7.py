# Strong LLM
import time
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import optax
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Flatten

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def preprocess(example):
    # Convert image to float32, scale to [0,1] then normalize to [-1,1] (like (x-0.5)/0.5)
    image = np.array(example['image'], dtype=np.float32) / 255.0
    image = (image - 0.5) / 0.5
    label = example['label']
    return image, label

def dataset_to_batches(ds, batch_size):
    # Convert the TFDS dataset to numpy arrays and create batches.
    ds = tfds.as_numpy(ds)
    images, labels = [], []
    for example in ds:
        img, lab = preprocess(example)
        images.append(img)
        labels.append(lab)
    images = np.stack(images)
    labels = np.array(labels)
    num_batches = images.shape[0] // batch_size
    batches = []
    for i in range(num_batches):
        batch_images = images[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        batches.append((batch_images, batch_labels))
    return batches

batch_size = 64
train_ds = tfds.load('mnist', split='train', shuffle_files=True)
test_ds  = tfds.load('mnist', split='test',  shuffle_files=False)

train_batches = dataset_to_batches(train_ds, batch_size)
test_batches  = dataset_to_batches(test_ds, batch_size)

# ---------------------------
# Model Definition using stax
# ---------------------------
# Define a simple neural network that mirrors the PyTorch model:
# - Flatten the input (28x28 or 28x28x1)
# - Dense layer with 128 units and ReLU activation
# - Dense layer with 10 outputs (for the 10 classes)
init_random_params, predict = stax.serial(
    Flatten,
    Dense(128),
    Relu,
    Dense(10)
)

# Initialize model parameters. The expected input shape is (batch, 28, 28, 1).
rng = jax.random.PRNGKey(0)
_, params = init_random_params(rng, (-1, 28, 28, 1))

# ---------------------------
# Loss Function and Optimizer
# ---------------------------
# Define the cross-entropy loss function. Note that optax.softmax_cross_entropy
# expects logits and one-hot encoded labels.
def loss_fn(params, batch):
    images, labels = batch
    logits = predict(params, images)
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

# Use SGD optimizer with learning rate 0.01
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params)

# Define a single training step with JIT compilation.
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ---------------------------
# Training Loop with Benchmarking
# ---------------------------
epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    for batch in train_batches:
        params, opt_state, loss = train_step(params, opt_state, batch)
    end_time = time.time()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}, Time: {end_time - start_time:.4f}s")

# ---------------------------
# Evaluation on Test Set with Benchmarking
# ---------------------------
correct = 0
total = 0
start_time = time.time()
for batch in test_batches:
    images, labels = batch
    logits = predict(params, images)
    predictions = jnp.argmax(logits, axis=1)
    correct += int(jnp.sum(predictions == labels))
    total += images.shape[0]
end_time = time.time()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {end_time - start_time:.4f}s")
