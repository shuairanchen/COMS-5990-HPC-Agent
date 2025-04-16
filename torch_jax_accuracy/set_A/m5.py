import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np

# RNN Cell Definition
class RNNCell(nn.Module):
    hidden_size: int

    def setup(self):
        # MODIFIED: Initialize weights for the RNN cell
        self.W_ih = self.param('W_ih', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size))
        self.W_hh = self.param('W_hh', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size))

    def __call__(self, x, hidden_state):
        # MODIFIED: Ensure hidden state is properly utilized and returned
        new_hidden_state = jnp.tanh(jnp.dot(x, self.W_ih) + jnp.dot(hidden_state, self.W_hh))
        return new_hidden_state

# RNN Module Definition
class RNN(nn.Module):
    hidden_size: int
    output_size: int

    def setup(self):
        self.rnn_cell = RNNCell(self.hidden_size)
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        # MODIFIED: Initialized hidden state explicitly
        hidden_state = jnp.zeros((x.shape[0], self.hidden_size))
        
        def rnn_step(hidden_state, x_t):
            return self.rnn_cell(x_t, hidden_state)  # MODIFIED: Pass hidden state explicitly

        # Using jax.lax.scan for efficient state propagation
        hidden_states = jax.lax.scan(rnn_step, hidden_state, x)[0]  # MODIFIED: Capture hidden states
        output = self.fc(hidden_states)
        return output

# Loss Function
def compute_loss(logits, targets):
    return jnp.mean(jax.nn.softmax_cross_entropy(logits=logits, labels=targets))

# Main Function
def main():
    # Sample data for training (Dummy data)
    x_train = jnp.array(np.random.rand(100, 10, 1))  # 100 samples, 10 timesteps, 1 feature
    y_train = jnp.array(np.random.randint(0, 2, (100, 10, 2)))  # 2 classes

    model = RNN(hidden_size=16, output_size=2)  # Instantiate the RNN model
    params = model.init(jax.random.PRNGKey(0), x_train)  # Initialize parameters

    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # Training Loop
    epochs = 5
    for epoch in range(epochs):
        # Forward pass
        logits = model.apply(params, x_train)
        loss = compute_loss(logits, y_train)

        # Compute gradients and update parameters
        grads = jax.grad(compute_loss)(params, y_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Testing on new data
    X_test = np.linspace(4 * np.pi, 5 * np.pi, 10).reshape(-1, 1)
    X_test = jnp.expand_dims(X_test, axis=0)  # Add batch dimension

    predictions = model.apply(params, X_test)
    print(f"Predictions for new sequence: {predictions.tolist()}")

if __name__ == "__main__":
    main()