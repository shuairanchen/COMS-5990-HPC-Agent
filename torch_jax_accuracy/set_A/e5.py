import jax.numpy as jnp  # MODIFIED: Ensured consistent import for jax.numpy as jnp
from jax import random  # MODIFIED: Added necessary import for random functionality
from typing import Any, Tuple

def init_params(key: Any, input_shape: Tuple[int, ...]) -> Any:
    """Initialize parameters for the model."""
    param_shape = (input_shape[0], 1)  # Example shape for parameters
    return random.normal(key, param_shape)  # Use explicit PRNG key

def loss_fn(params: Any, inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Calculate the loss."""
    predictions = jnp.dot(inputs, params)  # Simulate predictions
    return jnp.mean((predictions - targets) ** 2)  # Mean Squared Error

def main() -> None:
    """Main entry point for the program."""
    key = random.PRNGKey(0)  # Create an explicit PRNG key
    input_shape = (5, 10)  # Define input shape
    params = init_params(key, input_shape)  # Initialize parameters
    inputs = jnp.ones((5, 10))  # Example input data
    targets = jnp.ones((5,))  # Example target data

    # Calculate loss
    loss_value = loss_fn(params, inputs, targets)  # Using loss function
    print(f"Loss: {loss_value}")  # Displaying loss

if __name__ == "__main__":
    main()  # Entry point for the program