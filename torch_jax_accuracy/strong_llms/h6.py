## Strong LLM
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import pickle

# ---------------------------------------------------------------------
# Define an LSTM stack that processes the sequence step by step.
# We use Flax’s LSTMCell and a Python loop to build a multi-layer LSTM.
# ---------------------------------------------------------------------
class LSTMStack(nn.Module):
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_length, embed_size)
        batch_size = x.shape[0]
        # Initialize LSTM states (carry, hidden) for each layer.
        states = [
            nn.LSTMCell.initialize_carry(self.make_rng('lstm'), (batch_size,), self.hidden_size)
            for _ in range(self.num_layers)
        ]
        out = None
        seq_length = x.shape[1]
        # Process each time step sequentially.
        for t in range(seq_length):
            inp = x[:, t, :]
            new_states = []
            for i in range(self.num_layers):
                # Create an LSTM cell for layer i (parameters are registered by name).
                lstm_cell = nn.LSTMCell(name=f'lstm_cell_{i}', hidden_size=self.hidden_size)
                # Update state and get output.
                states[i], out = lstm_cell(states[i], inp)
                # For next layer, use the output from the current layer.
                inp = out
                new_states.append(states[i])
            states = new_states
        # Return the output of the last time step (from the last layer).
        return out

# ---------------------------------------------------------------------
# Define the LanguageModel using Flax modules.
# It embeds the input tokens, processes them through the LSTM stack,
# applies a Dense layer, and returns softmax probabilities.
# ---------------------------------------------------------------------
class LanguageModel(nn.Module):
    vocab_size: int
    embed_size: int
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        # x has shape (batch, seq_length) containing token indices.
        x_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)(x)
        lstm_out = LSTMStack(hidden_size=self.hidden_size, num_layers=self.num_layers)(x_embed)
        logits = nn.Dense(features=self.vocab_size)(lstm_out)
        probabilities = nn.softmax(logits)
        return probabilities

# ---------------------------------------------------------------------
# Create synthetic training data (similar to torch.randint).
# ---------------------------------------------------------------------
key = jax.random.PRNGKey(42)
vocab_size = 50
seq_length = 10
batch_size = 32
X_train = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
key, subkey = jax.random.split(key)
y_train = jax.random.randint(subkey, (batch_size,), 0, vocab_size)

# ---------------------------------------------------------------------
# Initialize the model, loss, and optimizer.
# ---------------------------------------------------------------------
embed_size = 64
hidden_size = 128
num_layers = 2
model = LanguageModel(
    vocab_size=vocab_size,
    embed_size=embed_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
)

# Initialize model parameters.
# Note: We pass two PRNG keys – one for parameters and one for the LSTM initialization.
variables = model.init({'params': key, 'lstm': key}, X_train)
params = variables['params']

# Define a simple cross-entropy loss.
def loss_fn(params, x, y):
    preds = model.apply({'params': params}, x)
    # Compute the negative log likelihood for the true classes.
    loss = -jnp.mean(jnp.log(preds[jnp.arange(preds.shape[0]), y] + 1e-7))
    return loss

# Set up the Adam optimizer using Optax.
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# ---------------------------------------------------------------------
# Training loop (5 epochs).
# ---------------------------------------------------------------------
epochs = 5
for epoch in range(epochs):
    params, opt_state, loss_val = train_step(params, opt_state, X_train, y_train)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss_val:.4f}")

# ---------------------------------------------------------------------
# Quantization: Simulate dynamic quantization by converting parameters to int8 and back.
# This is only a simple simulation; JAX does not offer PyTorch-like dynamic quantization.
# ---------------------------------------------------------------------
def quantize_param(param):
    scale = jnp.max(jnp.abs(param))
    # Avoid division by zero.
    scale = jnp.where(scale == 0, 1.0, scale)
    # Scale, round, and cast to int8.
    param_int8 = jnp.round(param / scale * 127).astype(jnp.int8)
    # Dequantize back to float32.
    return param_int8.astype(jnp.float32) * scale / 127

def quantize_params(params):
    if isinstance(params, dict):
        return {k: quantize_params(v) for k, v in params.items()}
    else:
        return quantize_param(params)

quantized_params = quantize_params(params)

# Save the quantized parameters (similarly to torch.save).
with open("quantized_language_model.pkl", "wb") as f:
    pickle.dump(quantized_params, f)

# Load the quantized parameters.
with open("quantized_language_model.pkl", "rb") as f:
    loaded_params = pickle.load(f)

# ---------------------------------------------------------------------
# Evaluate the quantized model.
# ---------------------------------------------------------------------
test_input = jax.random.randint(key, (1, seq_length), 0, vocab_size)
predictions = model.apply({'params': loaded_params}, test_input)
predicted_class = jnp.argmax(predictions, axis=1)
print(f"Prediction for input {test_input.tolist()}: {int(predicted_class[0])}")
