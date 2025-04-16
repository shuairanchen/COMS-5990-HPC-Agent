import jax
import jax.numpy as jnp
from jax import grad, jit, random
import matplotlib.pyplot as plt

# Generate synthetic data
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = 2 * X + 3 + noise

# Define the Linear Regression Model
def model(params, X):
    return jnp.dot(X, params['w']) + params['b']

# Loss function (Huber Loss)
def huber_loss(params, X, y, delta=1.0):
    preds = model(params, X)
    error = jnp.abs(preds - y)
    loss = jnp.where(error <= delta,
                     0.5 * error**2,
                     delta * (error - 0.5 * delta))
    return jnp.mean(loss)

# Update function
def update(params, X, y, learning_rate=0.01):
    loss_value, grads = jax.value_and_grad(huber_loss)(params, X, y, 1.0)
    new_params = {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }
    return new_params

# Initialize Parameters
bound = 1.0
key, subkey = jax.random.split(key)
w = jax.random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
key, subkey = jax.random.split(key)
b = jax.random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
params = {'w': w, 'b': b}

# Training loop
epochs = 1000
for epoch in range(epochs):
    params = update(params, X, y, learning_rate=0.01)
    if (epoch + 1) % 100 == 0:
        current_loss = huber_loss(params, X, y, 1.0)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

# Display the learned parameters
learned_w = params['w'][0, 0]
learned_b = params['b'][0]
print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

# Plot the model fit to the training data
plt.figure(figsize=(4, 4))
plt.scatter(X, y, label='Training Data')
X_line = jnp.linspace(0, 10, 100).reshape(-1, 1)
plt.plot(X_line, learned_w * X_line + learned_b, 'r', label='Model Fit')
plt.legend()
plt.show()

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
