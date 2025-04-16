import jax  # Import JAX library for numerical computations
import jax.numpy as jnp  # Import JAX's NumPy replacement
import jax.nn as jnn  # Import JAX's neural network functions
from flax import linen as nn  # Import Flax's neural network modules
from flax.training import train_state  # Import training state module from Flax
import optax  # Import Optax for optimization
import numpy as np  # Import NumPy for general numerical operations

class SimpleNN(nn.Module):
    """A simple neural network module."""

    @nn.compact
    def __call__(self, x):
        """Forward pass of the neural network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the neural network.
        """
        x = nn.Dense(128)(x)  # Apply first dense layer
        x = jnn.relu(x)  # Apply ReLU activation
        x = nn.Dense(10)(x)  # Apply second dense layer
        return x

def create_train_state(rng, learning_rate):
    """Create a train state for the model.

    Args:
        rng: Random number generator key.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A TrainState object.
    """
    model = SimpleNN()  # Initialize the model
    params = model.init(rng, jnp.ones([1, 784]))  # Initialize parameters
    tx = optax.adam(learning_rate)  # Create an Adam optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def evaluate_model(state, images, labels):
    """Evaluate the model on the provided dataset.

    Args:
        state: The current training state of the model.
        images: Input images for evaluation.
        labels: True labels corresponding to the input images.

    Returns:
        The accuracy of the model on the dataset.
    """
    labels = jnp.array(labels)  # Convert labels to JAX array

    logits = state.apply_fn({'params': state.params}, images)  # Get predictions
    predicted = jnp.argmax(logits, axis=1)  # Get predicted class indices
    total = labels.shape[0]  # Number of samples
    correct = jnp.sum(predicted == labels)  # Count correct predictions

    accuracy = 100 * correct / total  # Calculate accuracy
    print(f"Test Accuracy: {accuracy:.2f}%")  # Print the accuracy
    return accuracy  # Return the accuracy

def main():
    """Main function to run the training and evaluation."""
    rng = jax.random.PRNGKey(0)  # Initialize random number generator
    learning_rate = 0.001  # Set learning rate
    state = create_train_state(rng, learning_rate)  # Create training state

    # Dummy data for demonstration purposes
    images = np.random.rand(100, 784)  # Generate random images
    labels = np.random.randint(0, 10, size=(100,))  # Generate random labels

    evaluate_model(state, images, labels)  # Evaluate the model

if __name__ == "__main__":  # Entry point of the program
    main()  # Run the main function