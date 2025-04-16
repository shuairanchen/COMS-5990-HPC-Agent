import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


# Define the Transformer Model in Flax
class TransformerModel(nn.Module):
    input_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int

    def setup(self):
        self.embedding = nn.Dense(self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_layer = nn.Dense(self.output_dim)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(axis=1)  # Pooling across the sequence
        return self.output_layer(x)


# Loss function
def mse_loss(params, model, X, y):
    predictions = model.apply({'params': params}, X)
    return jnp.mean((predictions - y) ** 2)


# Create and initialize the model
def create_model(input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim, rng):
    model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim)
    params = model.init(rng, jnp.ones((1, input_dim)))  # Initialize parameters with dummy data
    return model, params


# Update function for training
@jax.jit
def update(params, model, X, y, optimizer, opt_state):
    loss, grads = jax.value_and_grad(mse_loss)(params, model, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# Training loop
def train(model, X, y, num_epochs=1000, learning_rate=0.001):
    key = jax.random.PRNGKey(0)  # PRNG key for random initialization
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model.params)

    for epoch in range(num_epochs):
        model.params, opt_state, loss = update(model.params, model, X, y, optimizer, opt_state)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

    return model


# Generate synthetic data
key = jax.random.PRNGKey(42)
num_samples = 100
seq_length = 10
input_dim = 1
X = jax.random.uniform(key, (num_samples, seq_length, input_dim))  # Random sequences
y = jnp.sum(X, axis=1)  # Target is the sum of each sequence

# Initialize the model
embed_dim = 16
num_heads = 2
num_layers = 2
ff_dim = 64
output_dim = 1
model, params = create_model(input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim, key)

# Train the model
trained_model = train(model, X, y)

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = trained_model.apply({'params': trained_model.params}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
