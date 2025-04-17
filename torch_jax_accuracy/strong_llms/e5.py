#Strong LLM
import jax
import jax.numpy as jnp
import optax
import numpy as np


# Create a PRNG key
key = jax.random.PRNGKey(42)

# Generate synthetic data: 100 data points with 2 features in [0, 10)
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 2)) * 10

# Compute target: y = (X[:, 0] + 2 * X[:, 1]) + noise, with shape (100, 1)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
# Compute (X[:,0] + 2*X[:,1]) and expand dims to (100, 1)
y = (X[:, 0] + 2 * X[:, 1]).reshape(-1, 1) + noise


# We define a two-layer network:
#  - fc1: linear transformation from 2 -> 10
#  - ReLU activation
#  - fc2: linear transformation from 10 -> 1
def predict(params, x):
    hidden = jnp.dot(x, params['W1']) + params['b1']  # (batch, 10)
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params['W2']) + params['b2']  # (batch, 1)
    return output


# For simplicity, we initialize weights using a uniform distribution.
# You could also use other initializers (e.g., Xavier/Glorot uniform).
def init_params(key):
    keys = jax.random.split(key, 4)
    # fc1: shape (2, 10)
    W1 = jax.random.uniform(keys[0], shape=(2, 10), minval=-1.0, maxval=1.0)
    b1 = jax.random.uniform(keys[1], shape=(10,), minval=-1.0, maxval=1.0)
    # fc2: shape (10, 1)
    W2 = jax.random.uniform(keys[2], shape=(10, 1), minval=-1.0, maxval=1.0)
    b2 = jax.random.uniform(keys[3], shape=(1,), minval=-1.0, maxval=1.0)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

params = init_params(key)


def loss_fn(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)


# Use the Adam optimizer from optax
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

epochs = 1000

# Define a function to compute loss and gradients
loss_and_grad = jax.value_and_grad(loss_fn)

for epoch in range(epochs):
    loss, grads = loss_and_grad(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")


X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
predictions = predict(params, X_test)
print(f"Predictions for {np.array(X_test).tolist()}: {np.array(predictions).tolist()}")