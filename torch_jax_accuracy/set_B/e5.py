import jax.numpy as jnp
from jax import random
from typing import Any, Tuple

def init_params(key: Any, input_shape: Tuple[int, ...]) -> Any:
    param_shape = (input_shape[0], 1)
    return random.normal(key, param_shape)

def loss_fn(params: Any, inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
    predictions = jnp.dot(inputs, params)
    return jnp.mean((predictions - targets) ** 2)

def main() -> None:
    key = random.PRNGKey(0)
    input_shape = (5, 10)
    params = init_params(key, input_shape)
    inputs = jnp.ones((5, 10))
    targets = jnp.ones((5,))
    loss_value = loss_fn(params, inputs, targets)
    print(f"Loss: {loss_value}")

if __name__ == "__main__":
    main()