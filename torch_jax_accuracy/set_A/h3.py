import jax
import jax.numpy as jnp  # Ensured consistent import statement
from flax import linen as nn
from jax import random, grad, jit, vmap
import optax


class SimpleModel(nn.Module):
    """A simple feedforward neural network model."""
    @nn.compact
    def __call__(self, x):
        """Forward pass of the model."""
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def create_model() -> SimpleModel:
    """Create an instance of the SimpleModel."""
    return SimpleModel()


def compute_loss(logits, labels):
    """Compute the binary cross-entropy loss."""
    return jnp.mean(jax.nn.sigmoid_cross_entropy(logits=logits, labels=labels))


def accuracy(logits, labels):
    """Calculate the accuracy of the model predictions."""
    preds = jnp.round(jax.nn.sigmoid(logits))
    return jnp.mean(preds == labels)


@jit
def train_step(optimizer, model, batch):
    """Perform a single training step."""
    def loss_fn(params):
        logits = model.apply({'params': params}, batch['X'])
        return compute_loss(logits, batch['y'])
    
    grads = grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grads)
    return optimizer


def train_model(X, y, num_epochs, key):
    """Train the model on the provided data."""
    model = create_model()
    params = model.init(key, jnp.ones((1, X.shape[1])))
    optimizer = optax.adam(learning_rate=0.001).init(params)

    dataset_size = X.shape[0]
    
    for epoch in range(num_epochs):
        # Shuffle dataset
        perm = random.permutation(key, dataset_size)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        for i in range(0, dataset_size, 32):
            batch = {
                'X': X_shuffled[i:i + 32],
                'y': y_shuffled[i:i + 32]
            }
            optimizer = train_step(optimizer, model, batch)
        
        # Example log after each epoch
        logits = model.apply({'params': optimizer.target}, X)
        train_acc = accuracy(logits, y)
        print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}")


def main():
    """Main entry point for the script."""
    # Example data generation with explicit PRNG key
    key = random.PRNGKey(0)  # Initialize PRNG key
    X = random.uniform(key, (1000, 10))  # MODIFIED: Added explicit PRNG key
    y = jnp.array([0, 1] * 500)  # Sample labels

    num_epochs = 10
    train_model(X, y, num_epochs, key)  # MODIFIED: pass key to train_model


if __name__ == "__main__":
    main()