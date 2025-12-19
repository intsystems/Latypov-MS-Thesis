import jax.numpy as jnp

def square_loss(x, y):
    return 1/2 * (x - y)**2

def abs_loss(x, y):
    return 1/2 * jnp.abs(x-y)