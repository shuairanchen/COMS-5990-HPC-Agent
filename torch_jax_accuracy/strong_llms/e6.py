#Strong LLM
import jax
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter  # pip install tensorboardX
import numpy as np


key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10  # 100 data points between 0 and 10
key, subkey = jax.random.split(key)
y = 3 * X + 5 + jax.random.normal(subkey, shape=(100, 1))  # Linear relationship with noise

def predict(params, x):
    # Linear model: y = x*w + b
    return jnp.dot(x, params["w"]) + params["b"]

def loss_fn(params, X, y):
    preds = predict(params, X)
    return jnp.mean((preds - y) ** 2)

# Initialize weight and bias (mimicking default behavior in nn.Linear)
key, subkey = jax.random.split(key)
params = {
    "w": 0.1 * jax.random.normal(subkey, shape=(1, 1)),
    "b": jnp.zeros((1,))
}

optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params)

writer = SummaryWriter(log_dir="runs/linear_regression")

epochs = 100
for epoch in range(epochs):
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # Log loss to TensorBoard
    writer.add_scalar("Loss/train", float(loss), epoch)
    
    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

writer.close()

# To run TensorBoard using the generated logs, execute the following command in your terminal:
# tensorboard --logdir=runs
