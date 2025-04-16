import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import matplotlib.pyplot as plt

def generate_data(num_samples=100):
    key = random.PRNGKey(42)
    X = random.uniform(key, shape=(num_samples, 1)) * 10
    noise = random.normal(key, shape=X.shape)
    y = 2 * X + 3 + noise
    return X, y

def custom_activation(x):
    return jax.nn.tanh(x) + x

def model(params, x):
    w, b = params['w'], params['b']
    return custom_activation(jnp.dot(x, w) + b)

def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

@jit
def compute_gradients(params, x, y):
    return grad(loss_fn)(params, x, y)

@jit
def update(params, x, y, learning_rate=0.01):
    grads = compute_gradients(params, x, y)
    new_params = {
        "w": params["w"] - learning_rate * grads["w"],
        "b": params["b"] - learning_rate * grads["b"]
    }
    return new_params

def train_model(X, y, num_epochs=1000, learning_rate=0.01):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-1.0, maxval=1.0)
    
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)

    params = {"w": w, "b": b}

    for epoch in range(num_epochs):
        params = update(params, X, y, learning_rate)

        if (epoch + 1) % 100 == 0:
            loss_value = loss_fn(params, X, y)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_value:.4f}")

    return params

def main():
    X, y = generate_data(100)
    learned_params = train_model(X, y)
    w, b = learned_params["w"], learned_params["b"]
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    plt.figure(figsize=(4, 4))
    plt.scatter(X, y, label='Training Data')
    plt.plot(X, w.item() * X + b.item(), 'r', label='Model Fit')
    plt.legend()
    plt.show()

    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()