import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import pandas as pd

# Generate synthetic data
key = random.PRNGKey(42)
key, subkey = random.split(key)
X = random.uniform(subkey, shape=(100, 1)) * 10  # 100 data points between 0 and 10
key, subkey = random.split(key)
noise = random.normal(subkey, shape=(100, 1))  # Noise
y = 2 * X + 3 + noise  # Linear relationship with noise

# Save the generated data to data.csv
data = jnp.concatenate((X, y), axis=1)
df = pd.DataFrame(data.numpy(), columns=['X', 'y'])
df.to_csv('data.csv', index=False)

# Define the Linear Regression Model using a simple JAX function
def model(params, x):
    return jnp.dot(x, params['w']) + params['b']

# Loss function (Mean Squared Error)
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Gradient computation (Using JAX's grad)
@jit
def compute_gradient(params, x, y):
    return grad(loss_fn)(params, x, y)

# Training step with manual gradient update
@jit
def train_step(params, x, y, learning_rate=0.01):
    grads = compute_gradient(params, x, y)
    params['w'] -= learning_rate * grads['w']
    params['b'] -= learning_rate * grads['b']
    return params

# Training loop
def train_model(X, y, num_epochs=1000, learning_rate=0.01):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    w = random.uniform(subkey, shape=(1, 1), minval=-1.0, maxval=1.0)
    key, subkey = random.split(key)
    b = random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
    params = {'w': w, 'b': b}

    for epoch in range(num_epochs):
        params = train_step(params, X, y, learning_rate)

        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, X, y)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4f}")

    return params

# Main function
def main():
    # Load data
    df = pd.read_csv('data.csv')
    X_data = jnp.array(df['X'].values, dtype=jnp.float32).reshape(-1, 1)
    y_data = jnp.array(df['y'].values, dtype=jnp.float32).reshape(-1, 1)

    # Train the model
    learned_params = train_model(X_data, y_data)

    # Display the learned parameters
    learned_w = learned_params['w'][0, 0]
    learned_b = learned_params['b'][0]
    print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(learned_params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()
