import jax
import jax.numpy as jnp  # MODIFIED: Ensure consistent import
from jax import grad, jit, random  # MODIFIED: PRNG keys usage
from flax import linen as nn
from flax.training import train_state
import optax

class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        return x

def create_train_state(rng, learning_rate):
    model = SimpleNN()
    params = model.init(rng, jnp.ones([1, 1]))  # Initialize with dummy input
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = state.apply_fn(params, batch['x'])
        return jnp.mean((predictions - batch['y']) ** 2)

    grads = grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

def main():
    rng = random.PRNGKey(0)  # Initialize PRNG key
    learning_rate = 0.001
    state = create_train_state(rng, learning_rate)
    
    # Example training loop (with dummy data)
    for epoch in range(10):
        batch = {'x': jnp.array([[1.0], [2.0]]), 'y': jnp.array([[2.0], [4.0]])}  # Dummy input and output
        state = train_step(state, batch)

    # Output learned parameters
    w = state.params['params']['Dense_0']['kernel'].flatten()[0]
    b = state.params['params']['Dense_0']['bias'].flatten()[0]
    print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = state.apply_fn(state.params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":  # MODIFIED: Ensure entry point
    main()