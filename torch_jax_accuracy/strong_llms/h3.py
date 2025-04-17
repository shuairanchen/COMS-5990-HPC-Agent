#Strong LLM
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np


class TransformerEncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train):
        # Multi-head self-attention; note: dropout is disabled when deterministic=True.
        attn_output = nn.SelfAttention(num_heads=self.num_heads,
                                       qkv_features=self.embed_dim,
                                       dropout_rate=self.dropout_rate,
                                       deterministic=not train)(x)
        # Add & Norm
        x = nn.LayerNorm()(x + attn_output)
        # Feedforward network
        ff_output = nn.Dense(self.ff_dim)(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dense(self.embed_dim)(ff_output)
        # Add & Norm
        x = nn.LayerNorm()(x + ff_output)
        return x

class TransformerModel(nn.Module):
    input_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train=True):
        # x shape: (batch, seq_length, input_dim)
        # Map input to embedding space.
        x = nn.Dense(self.embed_dim)(x)
        # Pass through a stack of Transformer encoder blocks.
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(embed_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        ff_dim=self.ff_dim,
                                        dropout_rate=self.dropout_rate)(x, train=train)
        # Pool across the sequence dimension (mean pooling).
        x = jnp.mean(x, axis=1)
        # Map to output.
        x = nn.Dense(self.output_dim)(x)
        return x

key = jax.random.PRNGKey(42)
seq_length = 10
num_samples = 100
input_dim = 1

# Generate random sequences with shape (num_samples, seq_length, input_dim)
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(num_samples, seq_length, input_dim))
# Target: sum of each sequence, resulting in shape (num_samples, 1)
y = jnp.sum(X, axis=1)

# Hyperparameters
embed_dim = 16
num_heads = 2
num_layers = 2
ff_dim = 64
output_dim = 1

# Create the model instance.
model = TransformerModel(input_dim=input_dim,
                         embed_dim=embed_dim,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         ff_dim=ff_dim,
                         output_dim=output_dim)

# Initialize model parameters using a dummy input.
dummy_input = jnp.ones((num_samples, seq_length, input_dim))
rng = jax.random.PRNGKey(0)
params = model.init(rng, dummy_input, train=True)

# Define Mean Squared Error loss.
def mse_loss(params, x, y, train):
    preds = model.apply(params, x, train=train)
    return jnp.mean((preds - y) ** 2)

# Setup the Adam optimizer.
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

epochs = 1000

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y, True)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for epoch in range(epochs):
    params, opt_state, loss = train_step(params, opt_state, X, y)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Generate new random test data with shape (2, seq_length, input_dim)
key, subkey = jax.random.split(key)
X_test = jax.random.uniform(subkey, shape=(2, seq_length, input_dim))
predictions = model.apply(params, X_test, train=False)
print("Predictions for", np.array(X_test).tolist(), ":", np.array(predictions).tolist())