import jax
import jax.numpy as jnp
import pandas as pd
import optax

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10 
noise = jax.random.normal(key, shape=X.shape)
y = 2 * X + 3 + noise  

data = jnp.concatenate((X, y), axis=1)
df = pd.DataFrame(data.numpy(), columns=['X', 'y'])
df.to_csv('data.csv', index=False)

class LinearRegressionDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = jnp.array(self.data['X'].values, dtype=jnp.float32).reshape(-1, 1)
        self.y = jnp.array(self.data['y'].values, dtype=jnp.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, batch_size):
        idx = jax.random.permutation(key, self.X.shape[0])
        for start in range(0, len(self.data), batch_size):
            end = min(start + batch_size, len(self.data))
            batch_idx = idx[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

def model(params, x):
    w, b = params
    return jnp.dot(x, w) + b

def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

def init_params():
    key, subkey = jax.random.split(key)
    w = jax.random.uniform(subkey, shape=(1, 1), minval=-1.0, maxval=1.0)
    key, subkey = jax.random.split(key)
    b = jax.random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
    return w, b

@jax.jit
def update(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)
    w, b = params
    new_params = (w - lr * grads[0], b - lr * grads[1])
    return new_params

def train_model(dataset, epochs=1000, batch_size=32):
    params = init_params()
    for epoch in range(epochs):
        for batch_X, batch_y in dataset.get_batch(batch_size):
            params = update(params, batch_X, batch_y)

        if (epoch + 1) % 100 == 0:
            current_loss = loss_fn(params, batch_X, batch_y)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

    return params

def main():
    dataset = LinearRegressionDataset('data.csv')
    params = train_model(dataset)

    w, b = params
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    X_test = jnp.array([[4.0], [7.0]])
    predictions = model(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()
