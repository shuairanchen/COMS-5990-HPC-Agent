import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax

class LinearRegressionModel:
    def __init__(self, key):
        key, subkey = random.split(key)
        self.w = random.uniform(subkey, (1, 1), minval=-1.0, maxval=1.0)
        key, subkey = random.split(key)
        self.b = random.uniform(subkey, (1,), minval=-1.0, maxval=1.0)
        self.params = {'w': self.w, 'b': self.b}

    def __call__(self, x):
        return jnp.dot(x, self.params['w']) + self.params['b']

def loss_fn(params, x, y):
    preds = LinearRegressionModel(params)(x)
    return jnp.mean((preds - y) ** 2)

@jit
def update(params, x, y, learning_rate):
    grads = grad(loss_fn)(params, x, y)
    updated_params = {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }
    return updated_params

key = random.PRNGKey(42)
key, subkey = random.split(key)
X = random.uniform(subkey, (100, 1), minval=0.0, maxval=10.0)
noise = random.normal(subkey, (100, 1))
y = 2 * X + 3 + noise  

model = LinearRegressionModel(key)

epochs = 1000
learning_rate = 0.01
for epoch in range(epochs):
    model.params = update(model.params, X, y, learning_rate)
    if (epoch + 1) % 100 == 0:
        current_loss = loss_fn(model.params, X, y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.4f}")

learned_w = model.params['w']
learned_b = model.params['b']
print(f"Learned weight: {learned_w[0, 0]:.4f}, Learned bias: {learned_b[0]:.4f}")

X_test = jnp.array([[4.0], [7.0]])
predictions = model(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
