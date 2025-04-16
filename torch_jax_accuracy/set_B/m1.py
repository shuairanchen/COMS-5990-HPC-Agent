import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt


# Generate synthetic data
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)


def generate_data(num_samples=100):
    key = random.PRNGKey(0)
    X = jnp.linspace(0, 4 * 3.14159, num_samples).reshape(-1, 1)
    y = jnp.sin(X)
    return X, y


X_seq, y_seq = create_in_out_sequences(generate_data()[1], 10)


# Define a simple model using JAX
class LSTMModel:
    def __init__(self, input_dim, hidden_units):
        self.hidden_units = hidden_units
        self.Wxi = jax.random.normal(random.PRNGKey(0), (input_dim, hidden_units))
        self.Whi = jax.random.normal(random.PRNGKey(1), (hidden_units, hidden_units))
        self.bi = jnp.zeros(hidden_units)
        # Initialize other weights similarly

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        H = jnp.zeros((batch_size, self.hidden_units))
        C = jnp.zeros((batch_size, self.hidden_units))

        all_hidden_states = []
        for t in range(seq_len):
            X_t = inputs[:, t, :]
            I_t = jax.nn.sigmoid(jnp.dot(X_t, self.Wxi) + jnp.dot(H, self.Whi) + self.bi)
            # Add other gate computations (F_t, O_t, C_tilde)
            C = F_t * C + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            all_hidden_states.append(H)

        return jnp.stack(all_hidden_states, axis=1)


# Define the LSTM model and other components in JAX
def loss_fn(params, inputs, targets):
    predictions = model.forward(inputs)
    return jnp.mean((predictions - targets) ** 2)


def train_step(params, X, y):
    grads = jax.grad(loss_fn)(params, X, y)
    new_params = {k: params[k] - 0.01 * grads[k] for k in params}
    return new_params


# Initialize and train model
model = LSTMModel(1, 50)
params = model.init_params()
