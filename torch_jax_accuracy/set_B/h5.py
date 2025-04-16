import jax
import jax.numpy as jnp
from jax import grad, jit, random
from flax import linen as nn
import optax

# Define the Encoder
class Encoder(nn.Module):
    def setup(self):
        self.embedding = nn.Embed(20, 32)
        self.lstm = nn.LSTMCell()

    def __call__(self, x):
        embedded = self.embedding(x)
        hidden, cell = self.lstm(embedded)
        return embedded, (hidden, cell)

# Define the Decoder with Attention
class Decoder(nn.Module):
    def setup(self):
        self.embedding = nn.Embed(20, 32)
        self.attention = nn.Dense(32)
        self.attention_combine = nn.Dense(32)
        self.lstm = nn.LSTMCell()
        self.fc_out = nn.Dense(20)

    def __call__(self, x, encoder_outputs, hidden, cell):
        x = x[:, None]  # Add sequence dimension
        embedded = self.embedding(x)

        # Attention mechanism
        attention_weights = self.attention(jnp.concatenate([embedded.squeeze(1), hidden[-1]]))
        context_vector = jnp.dot(attention_weights, encoder_outputs)

        # Combine context and embedded input
        combined = jnp.concatenate([embedded.squeeze(1), context_vector.squeeze(1)], axis=-1)
        combined = self.attention_combine(combined).tanh()[:, None]

        # LSTM and output
        lstm_out, (hidden, cell) = self.lstm(combined, (hidden, cell))
        output = self.fc_out(lstm_out.squeeze(1))
        return output, hidden, cell

# Synthetic data initialization
key = random.PRNGKey(42)
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16
src_data = random.randint(key, (batch_size, src_seq_length), 0, 20)
tgt_data = random.randint(key, (batch_size, tgt_seq_length), 0, 20)

# Initialize models
encoder = Encoder()
decoder = Decoder()

# Training loop
epochs = 100
for epoch in range(epochs):
    encoder_outputs, (hidden, cell) = encoder(src_data)
    loss = 0
    decoder_input = jnp.zeros(batch_size, dtype=jnp.int32)  # Start token

    for t in range(tgt_seq_length):
        output, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)
        loss += jnp.mean((output - tgt_data[:, t])**2)  # MSE loss
        decoder_input = tgt_data[:, t]  # Teacher forcing

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Test the sequence-to-sequence model with new input
test_input = random.randint(key, (1, src_seq_length), 0, 20)
encoder_outputs, (hidden, cell) = encoder(test_input)
decoder_input = jnp.zeros(1, dtype=jnp.int32)  # Start token
output_sequence = []

for _ in range(tgt_seq_length):
    output, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)
    predicted = output.argmax(axis=1)
    output_sequence.append(predicted.item())
    decoder_input = predicted

print(f"Input: {test_input.tolist()}, Output: {output_sequence}")
