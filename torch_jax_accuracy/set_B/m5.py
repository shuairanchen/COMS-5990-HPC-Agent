import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax


# Generate synthetic sequential data
def generate_data(num_samples=100):
    X = jnp.linspace(0, 4 * 3.14159, num_samples).reshape(-1, 1)
    y = jnp.sin(X)
    return X, y


# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)


X, y = generate_data()
sequence_length = 10
X_seq, y_seq = create_in_out_sequences(y, sequence_length)


# Define the RNN Model
def rnn_model(params, x):
    hidden = params['rnn_hidden']
    output = jnp.dot(x, params['rnn_weights']) + hidden
    return output


# Loss function
def loss_fn(params, X, y):
    preds = rnn_model(params, X)
    return jnp.mean((preds - y) ** 2)


# Gradient computation
@jit
def compute_gradient(params, X, y):
    return grad(loss_fn)(params, X, y)


# Training step
@jit
def train_step(params, X, y, learning_rate=0.001):
    grads = compute_gradient(params, X, y)
    new_params = {
        'rnn_weights': params['rnn_weights'] - learning_rate * grads['rnn_weights'],
        'rnn_hidden': params['rnn_hidden'] - learning_rate * grads['rnn_hidden']
    }
    return new_params


# Model initialization
def init_model(key, input_dim=1, hidden_dim=50):
    params = {
        'rnn_weights': random.normal(key, (input_dim, hidden_dim)),
        'rnn_hidden': random.normal(key, (hidden_dim,))
    }
    return params


# Training loop
def train_model(X, y, epochs=500, learning_rate=0.001):
    key = random.PRNGKey(42)
    params = init_model(key)

    for epoch in range(epochs):
        for sequences, labels in zip(X_seq, y_seq):
            sequences = sequences.reshape(1, -1)  # Reshape to (1, seq_length)
            labels = labels.reshape(1, -1)  # Reshape to (1, 1)

            # Train step
            params = train_step(params, sequences, labels, learning_rate)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_fn(params, sequences, labels)}")

    return params


# Testing on new data
X_test = jnp.linspace(4 * 3.14159, 5 * 3.14159, 10).reshape(-1, 1)
X_test = X_test.reshape(1, -1)  # Add batch dimension

params = train_model(X, y)

# Predictions
predictions = rnn_model(params, X_test)
print(f"Predictions for new sequence: {predictions.tolist()}")
