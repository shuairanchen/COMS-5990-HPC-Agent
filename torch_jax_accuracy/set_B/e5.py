import jax
import jax.numpy as jnp
from jax import grad, jit, random
from flax import linen as nn
import optax


# Generate synthetic data
def generate_data(num_samples=100):
    key = random.PRNGKey(0)
    X = jax.random.uniform(key, (num_samples, 2)) * 10  # 100 data points with 2 features
    noise = jax.random.normal(key, (num_samples, 1))
    y = (X[:, 0] + X[:, 1] * 2).reshape(-1, 1) + noise  # Non-linear relationship with noise
    return X, y


# Define the Deep Neural Network Model using Flax
class DNNModel(nn.Module):
    def setup(self):
        self.fc1 = nn.Dense(10)  # Input layer to hidden layer
        self.fc2 = nn.Dense(1)  # Hidden layer to output layer

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.relu(x)  # Activation function
        x = self.fc2(x)
        return x


# Loss function (Mean Squared Error)
def loss_fn(params, model, x, y):
    predictions = model.apply(params, x)
    return jnp.mean((predictions - y) ** 2)


# Training step
@jit
def train_step(params, model, x, y, learning_rate=0.01):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, x, y)
    params = optax.apply_updates(params, optax.sgd(learning_rate).update(grads, params))
    return params, loss


# Training loop
def train_model(model, num_epochs=1000):
    key = random.PRNGKey(0)
    X, y = generate_data()  # Get synthetic data
    params = model.init(key, X)  # Initialize model parameters

    for epoch in range(num_epochs):
        params, loss = train_step(params, model, X, y)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

    return params


# Testing the model on new data
def test_model(model, params):
    X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
    predictions = model.apply(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


# Main function
def main():
    model = DNNModel()  # Initialize model
    params = train_model(model)  # Train the model
    test_model(model, params)  # Test the model on new data


if __name__ == "__main__":
    main()
