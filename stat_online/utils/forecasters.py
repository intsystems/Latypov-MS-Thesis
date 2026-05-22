from typing import Any
from numpy import dtype, ndarray


import numpy as np
from typing import Callable


class normalizerForecaster:
    """
    this is a class to handle a function psi for INF algorithm
    it contains a method to find normalizing constant C such that sum_i psi(x_i - C) = 1
    and contains function psi itself. Psi function should be monotone.
    
    [1] https://www.jmlr.org/papers/volume11/audibert10a/audibert10a.pdf
    """

    def __init__(self, psi: Callable[[np.ndarray], np.ndarray],
                 psi_inv: Callable[[np.ndarray], np.ndarray],) -> None:
        """
        
        Parameters
        __________
        psi : Callable[[np.ndarray], np.ndarray]
            Monotone function applied elementwise, e.g., psi(z) = np.exp(-eta*z).
        
        psi_inv: Callable[[np.ndarray], np.ndarray]
            Inverted function for psi. Needed to compute bounds for C.
            see Lemma 1 in [1]
        """
        self.psi = psi
        self.psi_inv = psi_inv

    @staticmethod
    def bin_search(f, lo, hi, tol=1e-9, max_iter=1000):
        f_lo, f_hi = f(lo), f(hi)

        if f_lo * f_hi > 0:
            raise RuntimeError("No root found in search interval. Adjust range.")

        # binary search
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = f(mid)
            if abs(val) < tol:
                return mid
            if f_lo * val < 0:
                hi = mid
                f_hi = val
            else:
                lo = mid
                f_lo = val
        return 0.5 * (lo + hi)

    def find_normalizer(self, x: np.ndarray, tol: float = 1e-9, max_iter: int = 10_00) -> float:
        """
        Find C such that sum_i psi(x_i - C) = 1.

        Parameters
        ----------
        x : np.ndarray
            Input vector (size M).
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations for binary search.

        Returns
        -------
        C : float
            Normalizing constant.
        """
        # define target function
        def f(C):
            return np.sum(self.psi(x - C)) - 1.0

        # We need an interval where root exists
        lo = np.max(x) - 0.5
        hi = np.max(x) - self.psi_inv(1.0 / len(x)) + 0.5

        res =  self.bin_search(f, lo, hi, tol, max_iter)

        assert np.isclose(f(res), 0, atol=1e-3), "Root finding did not converge"
        return res

    def normalize_distribution(self, x: np.ndarray) -> np.ndarray:
        """
        Compute normalized distribution: p_i = psi(x_i - C), sum_i p_i = 1
        """
        C = self.find_normalizer(x)
        p = self.psi(x - C)
        return p / np.sum(p)  # numerical guard


class expertForecaster(normalizerForecaster):
    def __init__(self, K: int, eta: float = 1, gamma: float = 0.1) -> None:
        self.K = K
        self.eta = eta
        self.gamma = gamma
        super().__init__(self._psi, self._psi_inv)

    def _psi(self, x):
        return np.exp(self.eta * x) + self.gamma / self.K

    def _psi_inv(self, y):
        return np.log(y - self.gamma / self.K) / self.eta


class banditForecaster(normalizerForecaster):
    def __init__(self, K: int, eta: float = 1, gamma: float = 0.1, q=1.) -> None:
        self.K = K
        self.eta = eta
        self.gamma = gamma
        self.q = q
        super().__init__(self._psi, self._psi_inv)

    def _psi(self, x):
        return np.power(self.eta / (-x), self.q) + self.gamma / self.K

    def _psi_inv(self, y):
        return self.eta / (-(y - self.gamma / self.K)**(1/self.q) )


class LogBarrierOMD(normalizerForecaster):
    @staticmethod
    def log_barrier_omd(pt, ell, eta_t):
        """
        Implements Algorithm 2: Find p_{t+1} such that
        1 / p_{t+1,i} = 1 / p_{t,i} + eta_t[i] * (ell[i] - lambda)
        with lambda chosen so that sum_i p_{t+1,i} = 1.
        """
        pt = np.asarray(pt, dtype=float)
        ell = np.asarray(ell, dtype=float)
        eta_t = np.asarray(eta_t, dtype=float)
        M = len(pt)
        inv_pt = 1.0 / pt
        eta_safe = np.maximum(eta_t, 1e-16)

        # define function S(lambda) = sum_i 1/(inv_pt[i] + eta_i*(ell_i - lambda)) - 1
        def f(lmbda):
            denom = inv_pt + eta_safe * (ell - lmbda)
            # avoid division by zero or negative denom: if denom <= 0, treat resulting p as large (clamp)
            # but theoretically denom should stay positive for feasible lambda; clamp for safety
            denom = np.maximum(denom, 1e-16)
            p_next = 1.0 / denom
            return p_next.sum() - 1.0

        # The root always exists below the smallest singularity of the denominators:
        #   inv_pt[i] + eta_i * (ell[i] - lambda) = 0
        # so lambda < ell[i] + inv_pt[i] / eta_i for every i.
        # A very low lower bound makes all denominators large and positive, which
        # guarantees f(lo) < 0, while the upper bound approaches the singularity
        # where f(hi) > 0.
        singularity = ell + inv_pt / eta_safe
        hi = float(np.min(singularity) - 1e-12)
        lo = float(np.min(ell) - np.max(inv_pt / eta_safe) - 1.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(np.min(ell) - 1.0)
            hi = float(np.max(ell) + 1.0)
        lmbda = LogBarrierOMD.bin_search(f, lo, hi, tol=1e-9, max_iter=200)

        # build distribution
        denom = inv_pt + eta_safe * (ell - lmbda)
        denom = np.maximum(denom, 1e-16)
        p_next = 1.0 / denom

        # normalize (should be already normalized but numerical issues)
        p_next = p_next / p_next.sum()
        return p_next
