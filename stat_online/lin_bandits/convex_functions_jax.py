"""Convex functions using JAX for automatic differentiation."""

import jax
import jax.numpy as jnp
from typing import Callable

# Enable 64-bit precision for more accurate derivatives
jax.config.update("jax_enable_x64", True)

def linear_scale(x: float) -> float:
    """Linear scaling: (x + 1)/2"""
    return (x + 1) / 2

def squared(x: float) -> float:
    """Quadratic function: x^2"""
    return x**2

def exp_scale(x: float) -> float:
    """Exponential scaling: (exp(x) - exp(-1))/(exp(1) - exp(-1))"""
    return (jnp.exp(x) - jnp.exp(-1)) / (jnp.exp(1) - jnp.exp(-1))

def logistic(x: float) -> float:
    """Logistic function: 1/(1 + exp(-2x))"""
    return 1 / (1 + jnp.exp(-2*x))

def abs_power(x: float, p: float = 1.5) -> float:
    """Absolute value with power: |x|^p / 2"""
    return (jnp.abs(x)**p) / 2

def smooth_max(x: float) -> float:
    """Smooth maximum: log(1 + exp(x)) / log(1 + exp(1))"""
    return jnp.log(jnp.exp(-x) + jnp.exp(x)) / jnp.log(1 + jnp.exp(1))

def softplus_scale(x: float) -> float:
    """Softplus scaling: log(1 + exp(x)) / log(1 + exp(1))"""
    return jnp.log(1 + jnp.exp(x)) / jnp.log(1 + jnp.exp(1))

def shifted_cos(x: float) -> float:
    """Shifted cosine: (1 - cos(πx))/2"""
    return (1 - jnp.cos(jnp.pi * x)) / 2

def cubic(x: float) -> float:
    """Cubic function: (x^3 + 1)/2"""
    return (x**3 + 1) / 2

def arctan_scale(x: float) -> float:
    """Arctangent scaling: (arctan(πx/2) + arctan(π/2))/(2*arctan(π/2))"""
    return (jnp.arctan(jnp.pi*x/2) + jnp.arctan(jnp.pi/2)) / (2*jnp.arctan(jnp.pi/2))


def check_convexity(f: Callable, x_range=(-1, 1), n_points=100) -> bool:
    """Check if function is convex using second derivatives."""
    x = jnp.linspace(x_range[0], x_range[1], n_points)
    second_deriv = jax.vmap(jax.grad(jax.grad(f)))(x)
    return jnp.all(second_deriv >= 0)

def get_functions(x_scale = 1) -> list[Callable]:
    """Return a list of convex functions."""
    funcs = [
        linear_scale, squared, exp_scale, logistic,
        cubic, arctan_scale, softplus_scale,
        abs_power, shifted_cos, smooth_max,
    ]
    return funcs
    return [lambda x,f=f: f(x * x_scale) for f in funcs]


if __name__ == "__main__":
    # Test all functions
    x_test = jnp.linspace(-1, 1, 5)
    functions = [
        linear_scale, squared, exp_scale, logistic,
        cubic, arctan_scale, softplus_scale,
        abs_power, shifted_cos, smooth_max, 
    ]
    
    print("Function values at test points:")
    for func in functions:
        print(f"{func.__name__}: {func(x_test)}")
    
    print("\nConvexity check:")
    for func in functions:
        is_convex = check_convexity(func)
        print(f"{func.__name__}: {'Convex' if is_convex else 'Not convex'}")
