import jax
import jax.numpy as jnp
from jax import grad, jit, random


# Generate synthetic data
def generate_data(num_samples=100):
    key = random.PRNGKey(0)  # Initialize PRNG key
    X = jnp.linspace(0, 10, num_samples).reshape(-1, 1)  # 100 data points between 0 and 10
    noise = random.normal(key, shape=X.shape)  # Adding noise
    y = 2 * X + 3 + noise  # Linear relationship with noise
    return X, y


# Linear regression model
def model(params, x):
    return jnp.dot(x, params["w"]) + params["b"]  # Use matrix multiplication


# Loss function
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)


# Gradient computation
@jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)


# Training step
@jit
def train_step(params, x, y):
    grads = compute_gradient(params, x, y)
    return {
        "w": params["w"] - 0.01 * grads["w"],
        "b": params["b"] - 0.01 * grads["b"]
    }


# Training loop
def train_model(X, y, num_epochs=1000):
    bound = 1.0  # Range for initializing weights and bias
    key = random.PRNGKey(0)

    # Initialize parameters using random values
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)

    params = {"w": w, "b": b}

    for epoch in range(num_epochs):
        # Perform training step
        params = train_step(params, X, y)

        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss:.4f}")

    return params


# Main function
def main():
    # Generate data
    X, y = generate_data(100)

    # Train the model
    learned_params = train_model(X, y)

    learned_w = learned_params["w"][0, 0]
    learned_b = learned_params["b"][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()
