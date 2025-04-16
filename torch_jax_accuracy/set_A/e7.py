import jax.numpy as jnp  # MODIFIED: Consistently import jax.numpy as jnp
from jax import grad, jit, random, vmap
import flax.linen as nn
import optax
import pickle

class SimpleModel(nn.Module):
    """A simple neural network model using Flax."""
    
    def setup(self):
        """Define the layers of the model."""
        self.dense = nn.Dense(features=1)  # A layer with one output feature

    def __call__(self, x):
        """Forward pass of the model."""
        return self.dense(x)

def train_model(X, y):
    """Train the model with the given data."""
    model = SimpleModel()
    params = model.init(random.PRNGKey(0), X)
    # Loss function and optimization setup
    loss_fn = lambda params: jnp.mean((model.apply(params, X) - y) ** 2)
    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(params)
    
    for epoch in range(100):  # Simple training loop
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    
    return params

def main():
    """Main function to execute the training and evaluation of the model."""
    X_train = jnp.array([[0.0], [1.0], [2.0], [3.0]])  # Training data
    y_train = jnp.array([[0.0], [2.0], [4.0], [6.0]])  # Expected outputs

    # Train the model
    trained_params = train_model(X_train, y_train)

    # Verify the model works after loading
    X_test = jnp.array([[0.5], [1.0], [1.5]])  # Test data
    model = SimpleModel()  # Initialize model
    predictions = model.apply(trained_params, X_test)  # Get predictions
    print(f"Predictions after training: {predictions}")

if __name__ == "__main__":  # Entry point for the program
    main()  # Execute the main function