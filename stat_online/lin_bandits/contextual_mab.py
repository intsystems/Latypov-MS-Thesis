
from abc import abstractmethod
from typing import Any, Callable
import numpy as np
from jax import grad as jax_grad
import jax.numpy as jnp
import jax


class BaseOptimizer:
    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bounds(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class BaseFunction:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, w, x) -> float:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def set_params(self, params: Any) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    def get_params(self) -> Any:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class LinearFunction:
    def __init__(self, n_params) -> None:
        self.n_params = n_params
        self.params = np.random.rand(n_params)
        self.params = self.params / np.linalg.norm(self.params)


    def __call__(self, w: np.ndarray, x: np.ndarray) -> float:
        return (jnp.inner(w, x))

    def set_params(self, params: Any) -> None:
        self.params = params

    def get_params(self) -> Any:
        return self.params
    
class GeneralizedLinearFunction(BaseFunction):
    def __init__(self, n_params, nonlinearity: Callable, random_state: int = 123) -> None:
        rng = np.random.default_rng(random_state)
        self.n_params = n_params
        self.params = rng.normal(size=(n_params,))
        self.params = self.params / np.linalg.norm(self.params)
        self.params *= 0.5
        self.nonlinearity = nonlinearity

    def __call__(self, w: np.ndarray, x: np.ndarray) -> float:
        """
        
        """
        return (self.nonlinearity(jnp.inner(w, x[:self.n_params])))

    def get_params(self) -> Any:
        return self.params

    def set_params(self, params: Any) -> None:
        self.params = params


def loss_template(w, x, y, func, loss_fn, reg_term):
    return loss_fn(func(w, x), y) + reg_term * jnp.linalg.norm(w)

class OnlineGradArmOptimizer:
    def __init__(self,
            function: BaseFunction,
            lr_scaler: float,
            loss_fn: Callable, D, G, reg_term: float = 0.
            ) -> None:
        self.function = function
        self.lr_scaler = lr_scaler
        self.loss_fn = loss_fn
        self.reg_term = reg_term
        # self.loss_and_grad = jax.value_and_grad(loss_template, argnums=1)
        # self.loss_and_grad = jax.value_and_grad(lambda w, x, y: \
        #             self.loss_fn(self.function(w, x), y) + self.reg_term * jnp.linalg.norm(w), argnums=0
        #             )

        self.num_steps = 1
        self.D = D  # Diameter of the parameter space
        self.G = G  # Lipschitz constant of the loss function
        self.lr_scaler = min(lr_scaler, self.D / self.G)
    
    def call(self, x):
        w = self.function.get_params()
        res = self.function(w, x)
        assert not np.any(np.isnan(res)), f"function returns None, {w=}, {x=}"
        return res
    
    def get_loss(self, x, y):
        loss = self.loss_fn(self.call(x), y)
        return loss

    def step(self, x, y):
        w = self.function.get_params()
        loss_and_grad = jax.value_and_grad(loss_template, argnums=0)
        
        loss, grad = loss_and_grad(w, x, y,
                            self.function,
                            self.loss_fn,
                            self.reg_term
                            )

        loss, grad = np.array(loss), np.array(grad)

        lr_t = self.lr_scaler / np.sqrt(self.num_steps)

        # print(np.linalg.norm(grad))
        # updating params
        w_new = w - lr_t * grad
        w_new = w_new * (1/max(1, np.linalg.norm(w_new)))
        assert not np.any(np.isnan(w_new)), f"parameter becomes None, {w_new=}, {w=}, {grad=} {x=} {y=}, {self.call(x)=}"
        self.function.set_params(w_new)
        self.num_steps += 1
        
        return loss

    def regret_bounds(self, num_steps):
        return 3 * self.D * self.G * np.sqrt(num_steps)


class FullOptimizer:
    def __init__(self,
            function: BaseFunction,
            lr_scaler: float,
            loss_fn: Callable,
            D,
            G,
            reg_term: float = 0.,
            n_steps: int = 10
            ) -> None:
        self.function = function
        self.lr_scaler = lr_scaler
        self.loss_fn = loss_fn
        self.reg_term = reg_term
        self.n_steps = n_steps
        # self.loss_and_grad = jax.value_and_grad(loss_template, argnums=1)
        # self.loss_and_grad = jax.value_and_grad(lambda w, x, y: \
        #             self.loss_fn(self.function(w, x), y) + self.reg_term * jnp.linalg.norm(w), argnums=0
        #             )

        self.num_steps = 1
        self.D = D  # Diameter of the parameter space
        self.G = G  # Lipschitz constant of the loss function
        self.lr_scaler = min(lr_scaler, self.D / self.G)

        self.datapoints = []
        self.C = np.eye(self.function.n_params)
    
    def call(self, x):
        w = self.function.get_params()
        res = self.function(w, x)
        assert not np.any(np.isnan(res)), f"function returns None, {w=}, {x=}"
        return res
    
    def get_loss(self, x, y):
        loss = self.loss_fn(self.call(x), y)
        return loss

    def full_optimize(self, w):
        """
        finds argmin_w sum_{i=1}^n loss(f(w, x_i), y_i) + reg_term * ||w||_2

        do n_steps of gradient descent on the full dataset
        """

        lr_full = 0.1 * self.lr_scaler
        for _ in range(self.n_steps):
            total_loss, total_grad = 0., 0.
            for x_i, y_i in self.datapoints:
                loss_and_grad = jax.value_and_grad(loss_template, argnums=0)
                loss_i, grad_i = loss_and_grad(w, x_i, y_i,
                                    self.function,
                                    self.loss_fn,
                                    self.reg_term
                                    )
                total_loss += loss_i
                total_grad += grad_i
            # regularization
            total_grad += self.reg_term * w # / (np.linalg.norm(w) + 1e-8)

            total_loss, total_grad = np.array(total_loss), np.array(total_grad)
            w_new = w - lr_full * total_grad / len(self.datapoints)
            w_new = w_new * (1/max(1, np.linalg.norm(w_new)))
            w = w_new
        return w

    def step(self, x, y):
        self.datapoints.append((x, y))

        w = self.function.get_params()
        loss = self.get_loss(x, y)
        self.full_optimize(w)

        w = self.full_optimize(w)
        self.function.set_params(w)
        return loss

    def regret_bounds(self, num_steps):
        return 3 * self.D * self.G * np.sqrt(num_steps)

