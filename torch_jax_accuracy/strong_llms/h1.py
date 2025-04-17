#Strong LLM
import jax
import jax.numpy as jnp
import pickle


key = jax.random.PRNGKey(42)
# Generate 100 data points with one feature
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1))
key, subkey = jax.random.split(key)
# Create targets: y = 3*x + 2 + noise (noise scaled by 0.1)
y = 3 * X + 2 + 0.1 * jax.random.normal(subkey, shape=(100, 1))

def predict(params, x):
    """Simple linear model: y = x * w + b"""
    return jnp.dot(x, params["w"]) + params["b"]

def loss_fn(params, X, y):
    preds = predict(params, X)
    return jnp.mean((preds - y) ** 2)

key, subkey = jax.random.split(key)
params = {
    "w": 0.1 * jax.random.normal(subkey, shape=(1, 1)),
    "b": jnp.zeros((1,))
}

lr = 0.01
epochs = 100

for epoch in range(epochs):
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    params["w"] = params["w"] - lr * grads["w"]
    params["b"] = params["b"] - lr * grads["b"]
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

with open("model.pth", "wb") as f:
    pickle.dump(params, f)

with open("model.pth", "rb") as f:
    loaded_params = pickle.load(f)

X_test = jnp.array([[0.5], [1.0], [1.5]])
predictions = predict(loaded_params, X_test)
print("Predictions after loading:", predictions)