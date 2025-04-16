import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import optax

# Generate synthetic data
def generate_data(num_samples=100):
    X = jnp.linspace(0, 10, num_samples).reshape(-1, 1)
    y = 2 * X + 1 + jnp.random.normal(0, 1, X.shape)  # Adjusted for noise
    return X, y

# Linear regression model
def model(params, x):
    w, b = params
    return w * x + b

# Loss function
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Gradient computation
@jax.jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)

# Training step
@jax.jit
def train_step(params, x, y, rng_key):
    grads = compute_gradient(params, x, y)
    return params - 0.01 * grads  # Update rate

# Batch training step using vectorization
def train_model(X, y, num_epochs=1000):
    key = random.PRNGKey(0)  # Initialize PRNG key
    params = jnp.array([0.0, 0.0])  # Initial parameters (w, b)
    
    # Vectorized training over epochs
    for epoch in range(num_epochs):
        # Create a new PRNG key for each batch (if batching)
        key, subkey = random.split(key)
        params = train_step(params, X, y, subkey)  # // MODIFIED: Removed loop for vectorization

    return params

# Main function
def main():
    # Generate data
    X, y = generate_data(100)
    
    # Train the model
    learned_params = train_model(X, y)
    
    # Display the learned parameters
    w, b = learned_params
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()