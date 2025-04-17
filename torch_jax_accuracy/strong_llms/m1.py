## Strong LLM - o3-mini-high

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from flax import linen as nn
from flax.training import train_state

# ----------------------------
# Data Preparation
# ----------------------------
# Set random seed for numpy (JAX’s PRNG is used in the models)
np.random.seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X_vals = np.linspace(0, 4 * np.pi, num_samples).reshape(-1, 1)
y_vals = np.sin(X_vals)

def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    # data is assumed to be of shape (N, 1)
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.array(in_seq), jnp.array(out_seq)

X_seq, y_seq = create_in_out_sequences(y_vals, sequence_length)  # shapes: (num_samples-seq_length, seq_length, 1) and (num_samples-seq_length, 1)

# ----------------------------
# Custom LSTM Model (Hand-Coded)
# ----------------------------
class CustomLSTMModel(nn.Module):
    input_dim: int
    hidden_units: int

    @nn.compact
    def __call__(self, inputs, H_C=None):
        """
        inputs: shape (batch, seq_len, input_dim)
        H_C: tuple of (H, C); if None, initialize with random normal values using the "lstm" RNG.
        Returns:
          pred: predictions of shape (batch, seq_len, 1)
          (H, C): the final hidden and cell states.
        """
        batch_size, seq_len, _ = inputs.shape
        if H_C is None:
            # Use the "lstm" rng to generate initial states.
            key = self.make_rng("lstm")
            key_H, key_C = jax.random.split(key)
            H = jax.random.normal(key_H, (batch_size, self.hidden_units))
            C = jax.random.normal(key_C, (batch_size, self.hidden_units))
        else:
            H, C = H_C

        # Initialize gate parameters. (Each gate uses a weight matrix for input and hidden state, plus a bias.)
        Wxi = self.param("Wxi", nn.initializers.normal(), (self.input_dim, self.hidden_units))
        Whi = self.param("Whi", nn.initializers.normal(), (self.hidden_units, self.hidden_units))
        bi  = self.param("bi", nn.initializers.zeros, (self.hidden_units,))

        Wxf = self.param("Wxf", nn.initializers.normal(), (self.input_dim, self.hidden_units))
        Whf = self.param("Whf", nn.initializers.normal(), (self.hidden_units, self.hidden_units))
        bf  = self.param("bf", nn.initializers.zeros, (self.hidden_units,))

        Wxo = self.param("Wxo", nn.initializers.normal(), (self.input_dim, self.hidden_units))
        Who = self.param("Who", nn.initializers.normal(), (self.hidden_units, self.hidden_units))
        bo  = self.param("bo", nn.initializers.zeros, (self.hidden_units,))

        Wxc = self.param("Wxc", nn.initializers.normal(), (self.input_dim, self.hidden_units))
        Whc = self.param("Whc", nn.initializers.normal(), (self.hidden_units, self.hidden_units))
        bc  = self.param("bc", nn.initializers.zeros, (self.hidden_units,))

        dense = nn.Dense(features=1)

        outputs = []
        for t in range(seq_len):
            X_t = inputs[:, t, :]  # (batch, input_dim)
            I_t = jax.nn.sigmoid(jnp.dot(X_t, Wxi) + jnp.dot(H, Whi) + bi)
            F_t = jax.nn.sigmoid(jnp.dot(X_t, Wxf) + jnp.dot(H, Whf) + bf)
            O_t = jax.nn.sigmoid(jnp.dot(X_t, Wxo) + jnp.dot(H, Who) + bo)
            C_tilde = jnp.tanh(jnp.dot(X_t, Wxc) + jnp.dot(H, Whc) + bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            outputs.append(H)
        outputs = jnp.stack(outputs, axis=1)  # shape (batch, seq_len, hidden_units)
        pred = dense(outputs)  # shape (batch, seq_len, 1)
        return pred, (H, C)

# ----------------------------
# Built-In LSTM Model using Flax's LSTMCell
# ----------------------------
class LSTMModel(nn.Module):
    hidden_size: int = 50

    @nn.compact
    def __call__(self, inputs):
        """
        inputs: shape (batch, seq_len, input_dim) where input_dim is assumed to be 1.
        The LSTM cell is applied over the time axis; the final hidden state is passed through a Dense layer.
        Returns:
          out: predictions of shape (batch, 1)
        """
        batch_size, seq_len, _ = inputs.shape
        # Initialize carry for the LSTM cell.
        carry = nn.LSTMCell.initialize_carry(self.make_rng("lstm"), (batch_size,), self.hidden_size)
        lstm_cell = nn.LSTMCell()
        outputs = []
        for t in range(seq_len):
            carry, y = lstm_cell(carry, inputs[:, t, :])
            outputs.append(y)
        outputs = jnp.stack(outputs, axis=1)  # shape (batch, seq_len, hidden_size)
        dense = nn.Dense(features=1)
        out = dense(outputs[:, -1, :])  # use the last output for prediction
        return out

# ----------------------------
# Training Utilities
# ----------------------------
# A simple TrainState to hold parameters and the optimizer state.
class TrainState(train_state.TrainState):
    pass

# Loss functions
def loss_fn_custom(params, rng, batch_x, batch_y):
    # Model returns a sequence of predictions; we use the last time-step.
    pred, _ = custom_model.apply({"params": params}, batch_x, rngs={"lstm": rng})
    pred_last = pred[:, -1, 0]  # shape (batch,)
    loss = jnp.mean((pred_last - batch_y.squeeze()) ** 2)
    return loss

def loss_fn_builtin(params, rng, batch_x, batch_y):
    pred = lstm_model.apply({"params": params}, batch_x, rngs={"lstm": rng})
    pred = pred.squeeze()  # shape (batch,)
    loss = jnp.mean((pred - batch_y.squeeze()) ** 2)
    return loss

# Training steps (jitted)
@jax.jit
def train_step_custom(state, batch_x, batch_y, rng):
    loss, grads = jax.value_and_grad(loss_fn_custom)(state.params, rng, batch_x, batch_y)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def train_step_builtin(state, batch_x, batch_y, rng):
    loss, grads = jax.value_and_grad(loss_fn_builtin)(state.params, rng, batch_x, batch_y)
    state = state.apply_gradients(grads=grads)
    return state, loss

# ----------------------------
# Initialize Models and Optimizers
# ----------------------------
# Create instances of both models.
custom_model = CustomLSTMModel(input_dim=1, hidden_units=50)
lstm_model = LSTMModel(hidden_size=50)

# Initialize parameters by “calling” the model once with a sample input.
dummy_input_custom = jnp.ones((X_seq.shape[0], sequence_length, 1))
dummy_input_builtin = jnp.ones((X_seq.shape[0], sequence_length, 1))

rng_custom = jax.random.PRNGKey(0)
rng_builtin = jax.random.PRNGKey(1)

params_custom = custom_model.init({"params": rng_custom, "lstm": rng_custom}, dummy_input_custom)["params"]
params_builtin = lstm_model.init({"params": rng_builtin, "lstm": rng_builtin}, dummy_input_builtin)["params"]

# Create training states with Adam optimizer (learning rate = 0.01).
optimizer = optax.adam(0.01)
state_custom = TrainState.create(apply_fn=custom_model.apply, params=params_custom, tx=optimizer)
state_builtin = TrainState.create(apply_fn=lstm_model.apply, params=params_builtin, tx=optimizer)

# ----------------------------
# Training Loop for Custom LSTM Model
# ----------------------------
epochs = 500
rng = jax.random.PRNGKey(42)
for epoch in range(epochs):
    rng, step_rng = jax.random.split(rng)
    state_custom, loss_value = train_step_custom(state_custom, X_seq, y_seq, step_rng)
    if (epoch + 1) % 50 == 0:
        print(f"[Custom LSTM] Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}")

# ----------------------------
# Training Loop for Built-In LSTM Model
# ----------------------------
rng = jax.random.PRNGKey(100)
for epoch in range(epochs):
    rng, step_rng = jax.random.split(rng)
    state_builtin, loss_value = train_step_builtin(state_builtin, X_seq, y_seq, step_rng)
    if (epoch + 1) % 50 == 0:
        print(f"[Built-In LSTM] Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}")

# ----------------------------
# Testing on New Data
# ----------------------------
test_steps = 100  # should be greater than sequence_length
X_test = np.linspace(0, 5 * np.pi, test_steps).reshape(-1, 1)
y_test = np.sin(X_test)
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)  # (test_steps-seq_length, seq_length, 1)

# Get predictions from both models.
rng, custom_test_rng = jax.random.split(rng)
pred_custom, _ = custom_model.apply({"params": state_custom.params}, X_test_seq, rngs={"lstm": custom_test_rng})
# Use the last time-step predictions (flattened)
pred_custom = pred_custom[:, -1, 0]

rng, builtin_test_rng = jax.random.split(rng)
pred_builtin = lstm_model.apply({"params": state_builtin.params}, X_test_seq, rngs={"lstm": builtin_test_rng})
pred_builtin = pred_builtin.squeeze()

print("Predictions with Custom Model for new sequence:")
print(np.array(pred_custom))
print("\nPredictions with Built-In Model:")
print(np.array(pred_builtin))

# ----------------------------
# Plot the predictions
# ----------------------------
plt.figure(figsize=(8, 4))
plt.plot(pred_custom, label="Custom LSTM Model")
plt.plot(pred_builtin, label="Built-In LSTM Model")
plt.legend()
plt.title("Predictions on New Sine Wave Sequence")
plt.xlabel("Time step")
plt.ylabel("Predicted value")
plt.show()
