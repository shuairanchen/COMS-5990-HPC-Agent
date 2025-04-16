import jax.numpy as jnp  # MODIFIED: Consistent import of jax.numpy as jnp
from jax import random

def generate_random_numbers(shape):
    """
    Generate random numbers following a normal distribution.

    Args:
        shape (tuple): The shape of the output array.

    Returns:
        jnp.ndarray: An array of random numbers of the specified shape.
    """
    return random.normal_random(key=random.PRNGKey(0), shape=shape)  # Example method to generate random numbers

# Example usage of the generate_random_numbers function
def main():
    # Generate a 3x3 array of random numbers
    random_numbers = generate_random_numbers((3, 3))
    print("Generated Random Numbers:\n", random_numbers)

if __name__ == "__main__":
    main()  # Entry point of the program

# Additional code can go here, e.g., model definition, training loops, etc.
# Training loop
# epochs = 1000
# for epoch in range(epochs):
#     model_params, optimizer_state, loss = train_step(model, X, y, optimizer_state)
#     model = model.replace(slope=model_params)
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}, Loss: {loss:.4f}')