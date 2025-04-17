#Strong LLM
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np


class LSTMLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, initial_state):
        # x shape: (batch, seq_length, features)
        lstm_cell = nn.LSTMCell()
        def step_fn(carry, x_t):
            new_carry, y = lstm_cell(carry, x_t)
            return new_carry, y
        final_state, outputs = nn.scan(
            step_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(initial_state, x)
        return outputs, final_state

class Encoder(nn.Module):
    input_dim: int        # vocabulary size of source
    embed_dim: int
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, src_seq_length) of token IDs
        emb = nn.Embed(num_embeddings=self.input_dim, features=self.embed_dim)(x)
        # emb shape: (batch, src_seq_length, embed_dim)
        batch_size = emb.shape[0]
        outputs = emb
        states = []
        # Process through a stack of LSTM layers
        for i in range(self.num_layers):
            # Initialize carry (cell, hidden) for this layer
            initial_state = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), self.hidden_dim)
            outputs, final_state = LSTMLayer(self.hidden_dim, name=f"lstm_layer_{i}")(outputs, initial_state)
            states.append(final_state)
        # Collect final states from each layer.
        hidden = jnp.stack([s[1] for s in states], axis=0)  # shape: (num_layers, batch, hidden_dim)
        cell   = jnp.stack([s[0] for s in states], axis=0)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    output_dim: int       # vocabulary size of target
    embed_dim: int
    hidden_dim: int
    num_layers: int
    src_seq_length: int

    @nn.compact
    def __call__(self, x, encoder_outputs, hidden, cell):
        # x shape: (batch,) token IDs; add a time dimension.
        x = x[:, None]  # shape becomes (batch, 1)
        embedded = nn.Embed(num_embeddings=self.output_dim, features=self.embed_dim)(x)
        # embedded shape: (batch, 1, embed_dim)
        embedded_squeezed = jnp.squeeze(embedded, axis=1)  # (batch, embed_dim)
        # Attention: combine embedded input with last-layer hidden state.
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        attn_input = jnp.concatenate([embedded_squeezed, last_hidden], axis=1)  # (batch, embed_dim+hidden_dim)
        # Map to raw attention scores (one score per encoder time step).
        attn_scores = nn.Dense(self.src_seq_length)(attn_input)  # (batch, src_seq_length)
        attention_weights = jax.nn.softmax(attn_scores, axis=1)   # (batch, src_seq_length)
        # Compute context vector as weighted sum over encoder outputs.
        # encoder_outputs shape: (batch, src_seq_length, hidden_dim)
        attention_weights_exp = attention_weights[:, None, :]  # (batch, 1, src_seq_length)
        context_vector = jnp.matmul(attention_weights_exp, encoder_outputs)  # (batch, 1, hidden_dim)
        context_vector = jnp.squeeze(context_vector, axis=1)  # (batch, hidden_dim)
        # Combine context vector and embedded input.
        combined = jnp.concatenate([embedded_squeezed, context_vector], axis=1)  # (batch, embed_dim+hidden_dim)
        combined = nn.Dense(self.embed_dim)(combined)
        combined = jnp.tanh(combined)
        combined = combined[:, None, :]  # (batch, 1, embed_dim)
        # Pass through a one-step multi-layer LSTM.
        new_hidden = []
        new_cell = []
        x_t = jnp.squeeze(combined, axis=1)  # (batch, embed_dim)
        for i in range(self.num_layers):
            lstm_cell = nn.LSTMCell(name=f"decoder_lstm_cell_{i}")
            state = (cell[i], hidden[i])
            new_state, y = lstm_cell(state, x_t)
            new_cell.append(new_state[0])
            new_hidden.append(new_state[1])
            x_t = y  # output becomes input for the next layer
        new_hidden = jnp.stack(new_hidden, axis=0)
        new_cell = jnp.stack(new_cell, axis=0)
        # Map final LSTM output to target vocabulary logits.
        output = nn.Dense(self.output_dim)(y)  # (batch, output_dim)
        return output, new_hidden, new_cell

def get_data(key):
    src_vocab_size = 20
    tgt_vocab_size = 20
    src_seq_length = 10
    tgt_seq_length = 12
    batch_size = 16
    key, subkey = jax.random.split(key)
    src_data = jax.random.randint(subkey, shape=(batch_size, src_seq_length), minval=0, maxval=src_vocab_size)
    key, subkey = jax.random.split(key)
    tgt_data = jax.random.randint(subkey, shape=(batch_size, tgt_seq_length), minval=0, maxval=tgt_vocab_size)
    return src_data, tgt_data, src_vocab_size, tgt_vocab_size, src_seq_length, tgt_seq_length, batch_size

def cross_entropy_loss(logits, targets):
    # logits shape: (batch, num_classes); targets shape: (batch,)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)

def loss_fn(encoder_params, decoder_params, src_data, tgt_data, encoder, decoder, tgt_seq_length):
    # Run encoder.
    enc_vars = {'params': encoder_params}
    encoder_outputs, (hidden, cell) = encoder.apply(enc_vars, src_data)
    loss = 0.0
    # Start token (assumed 0) for the decoder.
    decoder_input = jnp.zeros((src_data.shape[0],), dtype=jnp.int32)
    dec_vars = {'params': decoder_params}
    for t in range(tgt_seq_length):
        logits, hidden, cell = decoder.apply(dec_vars, decoder_input, encoder_outputs, hidden, cell)
        loss += cross_entropy_loss(logits, tgt_data[:, t])
        # Teacher forcing: next input is current target.
        decoder_input = tgt_data[:, t]
    return loss / tgt_seq_length

def main():
    key = jax.random.PRNGKey(42)
    src_data, tgt_data, src_vocab_size, tgt_vocab_size, src_seq_length, tgt_seq_length, batch_size = get_data(key)

    input_dim = src_vocab_size      # Source vocabulary size
    output_dim = tgt_vocab_size     # Target vocabulary size
    embed_dim = 32
    hidden_dim = 64
    num_layers = 2

    encoder = Encoder(input_dim=input_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    decoder = Decoder(output_dim=output_dim, embed_dim=embed_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, src_seq_length=src_seq_length)

    # Initialize model parameters.
    encoder_vars = encoder.init(key, src_data)
    encoder_params = encoder_vars['params']
    encoder_outputs, (hidden, cell) = encoder.apply({'params': encoder_params}, src_data)
    dummy_decoder_input = jnp.zeros((batch_size,), dtype=jnp.int32)
    decoder_vars = decoder.init(key, dummy_decoder_input, encoder_outputs, hidden, cell)
    decoder_params = decoder_vars['params']

    # Combine parameters and set up the optimizer.
    params = {'encoder': encoder_params, 'decoder': decoder_params}
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, src_data, tgt_data):
        def loss_wrapper(params):
            return loss_fn(params['encoder'], params['decoder'], src_data, tgt_data, encoder, decoder, tgt_seq_length)
        loss_val, grads = jax.value_and_grad(loss_wrapper)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    epochs = 100
    for epoch in range(epochs):
        params, opt_state, loss_val = train_step(params, opt_state, src_data, tgt_data)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss_val:.4f}")

    key, subkey = jax.random.split(key)
    test_input = jax.random.randint(subkey, shape=(1, src_seq_length), minval=0, maxval=src_vocab_size)
    enc_vars = {'params': params['encoder']}
    encoder_outputs, (hidden, cell) = encoder.apply(enc_vars, test_input)
    decoder_input = jnp.zeros((1,), dtype=jnp.int32)  # Start token
    output_sequence = []
    dec_vars = {'params': params['decoder']}
    for _ in range(tgt_seq_length):
        logits, hidden, cell = decoder.apply(dec_vars, decoder_input, encoder_outputs, hidden, cell)
        predicted = jnp.argmax(logits, axis=-1)
        output_sequence.append(int(predicted[0]))
        decoder_input = predicted
    print(f"Input: {np.array(test_input).tolist()}, Output: {output_sequence}")

if __name__ == "__main__":
    main()
