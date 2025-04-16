import jax.numpy as jnp  # MODIFIED: Consistently import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.experimental import optimizers
import numpy as np

def initialize_params(input_size, output_size):
    """Initialize model parameters."""
    key = random.PRNGKey(0)
    W = random.normal(key, (input_size, output_size))
    b = random.normal(key, (output_size,))
    return W, b

def predict(params, inputs):
    """Predict outputs based on inputs and parameters."""
    W, b = params
    return jnp.dot(inputs, W) + b

def loss(params, inputs, targets):
    """Compute the loss as the mean squared error."""
    preds = predict(params, inputs)
    return jnp.mean((preds - targets) ** 2)

def update(params, opt_state, inputs, targets):
    """Perform a single update of the model parameters."""
    params_grads = grad(loss)(params, inputs, targets)
    opt_state = optimizers.apply_updates(opt_state, params_grads)
    return opt_state, loss(params, inputs, targets)

def main():
    """Main function to run training."""
    input_size, output_size, epochs, batch_size = 10, 1, 100, 5
    params = initialize_params(input_size, output_size)
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate=0.01)
    opt_state = opt_init(params)

    # Dummy data for demonstration
    X_train = jnp.array(np.random.randn(100, input_size), dtype=jnp.float32)
    y_train = jnp.array(np.random.randn(100, output_size), dtype=jnp.float32)

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            params, opt_state, loss_value = update(get_params(opt_state), opt_state, x_batch, y_batch)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.4f}")

    # Test the model on new data
    X_test = jnp.array(np.random.randn(5, input_size), dtype=jnp.float32)
    predictions = predict(params, X_test)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()  # Entry point of the program