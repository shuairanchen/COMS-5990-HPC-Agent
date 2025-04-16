import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import optax

class LinearModel:
    def __init__(self, key):
        self.w = random.normal(key, (1,))
        self.b = random.normal(key, ())

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

def loss_fn(model, x, y):
    preds = model(x)
    return jnp.mean((preds - y) ** 2)

def update(params, x, y, learning_rate=0.1):
    w, b = params
    loss_value, grads = jax.value_and_grad(loss_fn)(lambda x: model(x), x, y)
    w -= learning_rate * grads[0]
    b -= learning_rate * grads[1]
    return w, b

def train_model(key, model, x, y, epochs=100):
    for epoch in range(epochs): 
        model.w, model.b = update((model.w, model.b), x, y) 
    return model

def main():
    key = random.PRNGKey(0)  
    model = LinearModel(key)

    x = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([[2.0], [4.0], [6.0]])

    model = train_model(key, model, x, y, epochs=100)

    predictions = model(x)
    print(f"Predictions for {x.tolist()}: {predictions.tolist()}")
    print(f"Trained weights: {model.w}, bias: {model.b}")

if __name__ == "__main__":
    main()