"""Implementation of various bandit algorithms including UCB, Exp3, EpsilonGreedy and INF."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Any, List
import numpy as np
import scipy.stats
from stat_online.lin_bandits.contextual_mab import OnlineGradArmOptimizer
from stat_online.utils.forecasters import LogBarrierOMD

@dataclass
class HistoryElem:
    upper_bound: float
    lower_bound: float
    func_value: float
    mean_val: float
    optimized_count: int
    pull_step: int


class Arm:
    def __init__(self, optimizer: OnlineGradArmOptimizer, delta: float = 1e-2) -> None:
        self.optimizer = optimizer
        self.used_count = 0
        self.optimized_count = 0
        self.cumulative_loss = 0.0
        self.delta = delta

        self.loss_hist = []

    def call(self, x: np.ndarray) -> float:
        self.used_count += 1
        pred_t = self.optimizer.call(x)
        return pred_t
    
    def get_loss(self, x: np.ndarray, y: Any) -> float:
        self.used_count += 1
        return self.optimizer.get_loss(x, y)

    def inference_call(self, x: np.ndarray) -> float:
        """Inference step without updating the optimizer."""
        return self.optimizer.call(x)

    def step(self, x: np.ndarray, y: Any, pull_step: int) -> float:
        self.optimized_count += 1
        loss_t = self.optimizer.step(x, y)

        # # update element of history
        self.loss_hist.append(HistoryElem(
            upper_bound=float(self.upper_bound()),
            lower_bound=float(self.lower_bound(x)),
            mean_val=float(self.cumulative_loss/self.optimized_count),
            func_value=float(loss_t),
            optimized_count=self.optimized_count,
            pull_step=pull_step
        ))

        self.cumulative_loss += loss_t
        return loss_t

    def lower_bound(self, *args) -> float:
        if self.used_count == 0:
            return -np.inf
        n_t = self.optimized_count

        u_b = min(1., self.upper_bound())
        return ((self.cumulative_loss / n_t) -
                np.sqrt(2/3 * u_b * np.log(1/self.delta) / n_t) -
                (self.optimizer.regret_bounds(n_t) / n_t))

    def upper_bound(self) -> float:
        if self.used_count == 0:
            return np.inf
        n_t = self.optimized_count
        return ((self.cumulative_loss / n_t) +
                np.sqrt(1/2 * np.log(n_t/self.delta) / n_t))
    
    def loss_estimation(self) -> float:
        if self.used_count == 0:
            return float("inf")
        return self.cumulative_loss/self.optimized_count


class ContextualArm(Arm):
    """Contextual arm with additional context vector."""
    
    def __init__(self, optimizer: OnlineGradArmOptimizer, dim: int,delta: float = 1e-2, alpha = 1. ) -> None:
        super().__init__(optimizer, delta)
        self.context_matrix = np.eye(dim, dtype=float)
        self.mean_vect = np.zeros(dim, dtype=float)
        self.dim = dim
        self.alpha = 1


    def step(self, x: np.ndarray[tuple[Any, ...], np.dtype[Any]], y: Any, pull_step: int) -> None:
        super().step(x, y, pull_step)

        # Update context matrix as inverse matrix using Sherman-Morrison
        
        x_centered = x - self.mean_vect
        self.mean_vect = (self.optimized_count - 1)/self.optimized_count * self.mean_vect + x / self.optimized_count
        
        Cx = self.context_matrix @ x_centered
        denom = 1 + np.inner(x_centered, Cx)
        update_matr = np.outer(Cx, Cx / denom)
        self.context_matrix -= update_matr

    def context_adjustment(self, x: np.ndarray) -> float:
        """Calculate context adjustment based on context matrix."""
        if self.used_count == 0:
            return 0.0
        # Calculate context adjustment as sqrt(x^T C x)
        x_centered = x - self.mean_vect
        return np.sqrt(np.inner(x_centered, self.context_matrix @ x_centered))

    def lower_bound(self, x) -> float:
        if self.used_count == 0:
            return -np.inf

        n_t = self.optimized_count
        lower_bound_base = super().lower_bound()

        # Adjust lower bound based on context matrix
        context_adjustment = self.context_adjustment(x)
        l_b = lower_bound_base + self.alpha * context_adjustment
        return l_b

class dummyContextualArm(ContextualArm):
    def lower_bound(self, x) -> float:
        context_adjustment = np.sqrt(np.inner(x, self.context_matrix @ x))
        return context_adjustment

class Strategy:
    def __init__(self, arms: list[Arm], num_optimize: int = 1, ) -> None:
        self.arms = arms
        self.num_optimize = num_optimize
        self.selection_history = []
        self.loss_history = []
        # self.optimization_selection = optimization_selection

    # def get_arms2train(self) -> list[Arm]:

    def step(self, x: np.ndarray, y: Any) -> None:
        raise NotImplementedError

    def run(self, X: np.ndarray, Y: Any) -> None:
        x, y = X[0], Y[0]
        for arm in self.arms:
            arm.step(x, y, 0)
        for x_t, y_t in zip(X[1:], Y[1:]):
            self.step(x_t, y_t)
        return
    

# class GroupedUCB(Strategy):
#     def __init__(self, arms: list[Arm], num_optimize: int = 1) -> None:
#         super().__init__(arms, num_optimize)
#         self.num_groups =

class UCBAlgorithm(Strategy):
    def __init__(self, arms: list[Arm], num_optimize: int = 1, **kwargs) -> None:
        """
        :param arms: list of arms to optimize
        :param num_optimize: number of arms to optimize at each step. 
        default is 1, which means only best selected arm will be optimized.
        """
        super().__init__(arms, num_optimize)
        self.num_steps = 0

    def step(self, x: np.ndarray, y: Any) -> None:
        # select arms to make decision
        lower_bounds = np.array([arm.lower_bound(x) for arm in self.arms])
        arms_to_optimize = list(np.argsort(lower_bounds)[:self.num_optimize])
        assert len(arms_to_optimize) == self.num_optimize
        # select arm with lowes upper bound
        # arm_t = min(arms_to_optimize, key=lambda i: self.arms[i].upper_bound())
        arm_t = np.argmin(lower_bounds)
        arm = self.arms[arm_t]
        # call selected arm
        loss_t = arm.get_loss(x, y)
        self.selection_history.append(arm_t)
        self.loss_history.append(loss_t)
        # optimize arms

        # arms_to_optimize = np.argsort(lower_bounds)
        # if hasattr(self.arms[0], 'context_adjustment'):
        #     lower_bounds = np.array([arm.context_adjustment(x) for arm in self.arms])
        for arm_idx in arms_to_optimize:
            self.arms[arm_idx].step(x, y, self.num_steps)
        self.num_steps += 1

        # removing step
        upper_bound = np.argmin([arm.upper_bound() for arm in self.arms])
        self.arms = list(filter(lambda arm: arm.lower_bound(x) < upper_bound, self.arms))        

    
    def run(self, X: np.ndarray, Y: Any) -> None:
        """
        Run the UCB algorithm for one step.
        :param x: list of context vectors
        :param y: target value
        """

        # initialization step
        x, y = X[0], Y[0]
        for arm in self.arms:
            arm.step(x, y, 0)
        for x_t, y_t in zip(X[1:], Y[1:]):
            self.step(x_t, y_t)
        return
        


class Exp3Algorithm(Strategy):
    """Exponential-weight algorithm for Exploration and Exploitation."""
    
    def __init__(
        self,
        arms: list[Arm],
        T: float = -1,
        delta: float = 0,
        eta: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.1,
        num_optimize: int = 1
    ) -> None:
        """
        Algorithm from Prediction Learning and Games, Theorem 6.10 
        
        Args:
            arms: List of arms to optimize
            loss_func: Function to calculate loss (prediction, target)
            eta: Learning rate parameter
            num_optimize: Number of arms to optimize at each step
        """
        super().__init__(arms, num_optimize)
        self.K = len(arms)
        if T > 0:
            beta = (1 / (self.K * T) * np.log(self.K / delta)) ** 0.5 / 5
            gamma = 4 * self.K * beta / (3 + beta) / 10
            eta = gamma / (2 * self.K) * 50

        self.eta = eta
        self.beta = beta
        self.gamma = gamma
        self.num_steps = 0
        self.weights = np.ones(self.K, dtype=float)
        self.probs = self.weights / np.sum(self.weights)

    def step(self, x: np.ndarray, y: Any) -> None:

        # Select arm probabilistically
        arm_idx = np.random.choice(len(self.arms), p=self.probs)
        arm = self.arms[arm_idx]

        # Get loss and update history
        loss_t = arm.get_loss(x, y)

        self.selection_history.append(arm_idx)
        self.loss_history.append(loss_t)
        
        # Update weights
        # assert np.abs(loss_t) <= 1, f"Exp3 algorithm consider only [-1,1] losses. \n get {loss_t=} "

        delta_weights = self.beta / self.probs
        estimated_loss = (0. - loss_t) / self.probs[arm_idx]
        delta_weights[arm_idx] += estimated_loss
        self.weights *= np.exp(self.eta * delta_weights)
        self.weights /= np.max(self.weights)
        
        self.probs = (1 - self.gamma) * self.weights/np.sum(self.weights) + self.gamma/self.K
        
        # Optimize selected arms
        # for arm_index in np.argsort(self.probs)[-self.num_optimize:]:
        #     arm = self.arms[arm_index]
        #     arm.step(x, y, self.num_steps)

        for arm_idx in np.argsort(self.weights)[-self.num_optimize:]:
            self.arms[arm_idx].step(x, y, self.num_steps)

        self.num_steps += 1


class EpsilonGreedyAlgorithm(Strategy):
    """Epsilon-Greedy bandit algorithm implementation."""
    
    def __init__(
        self,
        arms: list[Arm],
        epsilon: float = 0.1,
        num_optimize: int = 1,
        **kwargs
    ) -> None:
        """Initialize Epsilon-Greedy algorithm.
        
        Args:
            arms: List of arms to optimize
            loss_func: Function to calculate loss (prediction, target)
            epsilon: Exploration probability
            num_optimize: Number of arms to optimize at each step
        """
        super().__init__(arms, num_optimize)
        self.epsilon = epsilon
        self.num_steps = 0

    def step(self, x: np.ndarray, y: Any) -> None:
        # Exploration vs exploitation
        if np.random.random() < self.epsilon:
            arm_idx = np.random.randint(len(self.arms))
        else:
            arm_idx = np.argmin([arm.loss_estimation() for arm in self.arms])
        
        arm = self.arms[arm_idx]

        loss_t = arm.get_loss(x, y)
        self.selection_history.append(arm_idx)
        self.loss_history.append(loss_t)

        # Optimize selected arms
        for arm in sorted(self.arms, key=lambda a: a.loss_estimation())[:self.num_optimize]:
            arm.step(x, y, self.num_steps)
            
        self.num_steps += 1

class ThompsonSamplingAlgorithm(Strategy):
    """Thompson Sampling algorithm implementation.
    
    Reference: Thompson, W.R. (1933). "On the likelihood that one unknown probability
    exceeds another in view of the evidence of two samples". Biometrika.
    """
    
    def __init__(self, arms: list[Arm], num_optimize: int = 1) -> None:
        super().__init__(arms, num_optimize)
        self.alpha = np.ones(len(arms))
        self.beta = np.ones(len(arms))
        self.num_steps = 0

    def step(self, x: np.ndarray, y: Any) -> None:
        # Sample from beta distributions
        samples = [scipy.stats.beta(a, b).rvs() 
                  for a, b in zip(self.alpha, self.beta)]
        
        # Select arm with highest sample
        arm_idx = np.argmax(samples)
        arm = self.arms[arm_idx]
        
        # Get loss and update history
        loss_t = arm.get_loss(x, y)
        self.selection_history.append(arm_idx)
        self.loss_history.append(loss_t)
        
        # Update beta distribution parameters
        if loss_t < 0.5:  # Considered a "success"
            self.alpha[arm_idx] += 1
        else:
            self.beta[arm_idx] += 1
            
        # Optimize selected arms
        for arm in sorted(self.arms, key=lambda a: a.loss_estimation())[:self.num_optimize]:
            arm.step(x, y, self.num_steps)
            
        self.num_steps += 1


class SuccessiveEliminationAlgorithm(Strategy):
    """Successive Elimination algorithm implementation.
    https://d1wqtxts1xzle7.cloudfront.net/86788406/COLT02-libre.pdf?1654023603
    Reference: Even-Dar, E., Mannor, S., & Mansour, Y. (2002). "PAC bounds for
    multi-armed bandit and Markov decision processes". COLT.
    """
    
    def __init__(self, arms: list[Arm], delta: float = 0.1, num_optimize: int = 1, **kwargs) -> None:
        super().__init__(arms, num_optimize)
        self.delta = delta
        self.num_steps = 0
        self.active_arms = list(range(len(arms)))
        self.empirical_losses = np.zeros(len(arms))
        self.pull_counts = np.zeros(len(arms))

    def step(self, x: np.ndarray, y: Any) -> None:
        # Select arm with lowest empirical loss among active arms
        active_losses = [self.empirical_losses[i] for i in self.active_arms]
        best_active_idx = np.argmin(active_losses)
        arm_idx = self.active_arms[best_active_idx]
        arm = self.arms[arm_idx]
        
        # Get loss and update history
        loss_t = arm.get_loss(x, y)
        self.selection_history.append(arm_idx)
        self.loss_history.append(loss_t)
        
        # Update empirical estimates
        self.empirical_losses[arm_idx] = (
            (self.empirical_losses[arm_idx] * self.pull_counts[arm_idx] + loss_t) / 
            (self.pull_counts[arm_idx] + 1)
        )
        self.pull_counts[arm_idx] += 1
        
        # Eliminate suboptimal arms
        min_loss = min(self.empirical_losses[i] for i in self.active_arms)
        confidence = np.sqrt(np.log(2 * len(self.arms) * self.num_steps**2 / self.delta) / 
                          (2 * self.num_steps))
        
        self.active_arms = [
            i for i in self.active_arms
            if self.empirical_losses[i] - min_loss <= 2 * confidence
        ]
        
        # Optimize selected arms
        for arm in sorted(self.arms, key=lambda a: a.loss_estimation())[:self.num_optimize]:
            arm.step(x, y, self.num_steps)
            
        self.num_steps += 1


class GroupedUCB(Strategy):
    """Grouped UCB algorithm with fixed groups per epoch."""
    
    def __init__(self, arms: list[Arm], num_optimize: int = 1, epoch_length: int = 50, **kwargs) -> None:
        """
        :param arms: list of arms to optimize
        :param num_optimize: number of arms to optimize at each step
        :param epoch_length: number of steps before regrouping
        :param kwargs: additional arguments
        """
        super().__init__(arms, num_optimize)
        self.epoch_length = epoch_length
        self.current_step = 0
        self.current_epoch = 0
        self.groups = self._create_groups(0)
        self.group_usages = defaultdict(int)
        
    def _create_groups(self, x) -> list[list[Arm]]:
        """Create groups of arms sorted by upper bound."""
        sorted_arms = sorted(self.arms, key=lambda a: a.lower_bound(x))
        # sorted_arms = sorted(self.arms, key=lambda a: a.upper_bound())
        num_groups = len(self.arms) // self.num_optimize
        return [
            sorted_arms[i*self.num_optimize:(i+1)*self.num_optimize]
            for i in range(num_groups)
        ]
    
    def _get_group_lower_bound(self, group: list[Arm],x) -> float:
        """Get group lower bound as max lower bound in group."""
        return max(arm.lower_bound(x) for arm in group)
    
    def step(self, x: np.ndarray, y: Any) -> None:
        # Select arm with minimal lower bound for decision
        decision_arm = min(self.arms, key=lambda a: a.lower_bound(x))
        loss_t = decision_arm.get_loss(x, y)
        self.selection_history.append(self.arms.index(decision_arm))
        self.loss_history.append(loss_t)
        
        # Select group with minimal group lower bound for training
        if self.current_step % self.epoch_length == 0:
            self.groups = self._create_groups(x)
            self.current_epoch += 1
            
        group_bounds = [self._get_group_lower_bound(g, x) for g in self.groups]
        group_idx = np.argmin(group_bounds)
        self.group_usages[group_idx] += 1
        for arm in self.groups[group_idx]:
            arm.step(x, y, self.current_step)
            
        self.current_step += 1



# def log_barrier_omd(pt, ell, eta_t):
#     """
#     Implements Algorithm 2: Find p_{t+1} such that
#     1 / p_{t+1,i} = 1 / p_{t,i} + eta_t[i] * (ell[i] - lambda)
#     with lambda chosen so that sum_i p_{t+1,i} = 1.
#     """
#     pt = np.asarray(pt, dtype=float)
#     ell = np.asarray(ell, dtype=float)
#     eta_t = np.asarray(eta_t, dtype=float)
#     M = len(pt)
#     inv_pt = 1.0 / pt
    
#     # define function S(lambda) = sum_i 1/(inv_pt[i] + eta_i*(ell_i - lambda)) - 1
#     def S(lmbda):
#         denom = inv_pt + eta_t * (ell - lmbda)
#         # avoid division by zero or negative denom: if denom <= 0, treat resulting p as large (clamp)
#         # but theoretically denom should stay positive for feasible lambda; clamp for safety
#         denom = np.maximum(denom, 1e-16)
#         p_next = 1.0 / denom
#         return p_next.sum() - 1.0
    
#     # search interval for lambda: algorithm suggests lambda in [min ell, max ell]
#     lo, hi = ell.min(), ell.max()
#     slo, shi = S(lo), S(hi)
    
#     # If no sign change (can happen numerically), expand interval
#     if slo * shi > 0:
#         lo, hi = ell.min() - 1.0, ell.max() + 1.0
#         slo, shi = S(lo), S(hi)
#         # if still no sign change, fallback: return normalized exp(-eta*ell) (mirror of mirror descent)
#         if slo * shi > 0:
#             denom = inv_pt + eta_t * (ell - ell.mean())
#             denom = np.maximum(denom, 1e-16)
#             p = 1.0 / denom
#             return p / p.sum()
    
#     # bisection
#     for _ in range(60):
#         mid = 0.5 * (lo + hi)
#         sm = S(mid)
#         if sm == 0:
#             lo = hi = mid
#             break
#         if slo * sm <= 0:
#             hi = mid
#             shi = sm
#         else:
#             lo = mid
#             slo = sm
#     lmbda = 0.5*(lo+hi)
#     denom = inv_pt + eta_t * (ell - lmbda)
#     denom = np.maximum(denom, 1e-16)
#     p_next = 1.0 / denom

#     # normalize (should be already normalized but numerical issues)
#     p_next = p_next / p_next.sum()
#     return p_next


class CORRAL(Strategy):
    """CORRAL algorithm adapted to Strategy base class.
    https://proceedings.mlr.press/v65/agarwal17b/agarwal17b.pdf#page=4.77
    """

    def __init__(self, arms: List[Arm], T: int = 1000, eta: float = 0.1,
                 num_optimize: int = 1, **kwargs) -> None:
        """
        :param arms: List of arms to optimize
        :param T: Time horizon
        :param eta: Learning rate parameter
        :param num_optimize: Number of arms to optimize at each step
        """
        super().__init__(arms, num_optimize)
        self.M = len(arms)
        self.T = T
        self.gamma = 1.0 / T
        self.beta = np.exp(1.0 / np.log(T + 1.0))
        self.eta = eta

        # Initialize distributions
        self.p = np.ones(self.M) / self.M
        self.pbar = self.p.copy()
        self.rho = np.full(self.M, 2 * self.M)
        self.eta_vec = np.full(self.M, eta)
        self.num_steps = 0

    def step(self, x: np.ndarray, y: Any) -> None:
        """Execute one step of CORRAL algorithm."""
        # Sample arm according to pbar distribution
        i = np.random.choice(self.M, p=self.pbar)
        arm = self.arms[i]
        
        # Get loss from selected arm
        loss_t = arm.get_loss(x, y)
        
        # Update history
        self.selection_history.append(i)
        self.loss_history.append(loss_t)
        
        # Optimize selected arm
        arm.step(x, y, self.num_steps)
        
        # Importance-weighted loss estimate for meta-update
        ell = np.zeros(self.M)
        ell[i] = loss_t / max(self.pbar[i], 1e-16)
        
        # Update meta-distribution using log-barrier OMD
        p_next = LogBarrierOMD.log_barrier_omd(self.p, ell, self.eta_vec)
        pbar_next = (1 - self.gamma) * p_next + self.gamma * (np.ones(self.M) / self.M)
        
        # Update rho and eta_vec
        for j in range(self.M):
            if 1.0 / pbar_next[j] > self.rho[j]:
                self.rho[j] = 2.0 / pbar_next[j]
                self.eta_vec[j] = self.beta * self.eta_vec[j]
        
        self.p = p_next
        self.pbar = pbar_next
        self.num_steps += 1


class DynamicBalancing(Strategy):
    """Dynamic Balancing algorithm adapted to Strategy base class.
    https://proceedings.mlr.press/v139/cutkosky21a/cutkosky21a.pdf
    """
    
    def __init__(self, arms: List[Arm], R_funcs: List[Callable], v: List[float], 
                 bias_funcs: List[Callable], delta: float = 0.05, c: float = 1.0,
                 num_optimize: int = 1, **kwargs) -> None:
        """
        :param arms: List of arms to optimize
        :param R_funcs: List of regret bound functions R_i(n)
        :param v: List of scaling coefficients v_i
        :param bias_funcs: List of bias functions b_i(n)
        :param delta: Confidence parameter
        :param c: Confidence bound multiplier
        :param num_optimize: Number of arms to optimize at each step
        """
        super().__init__(arms, num_optimize)
        self.M = len(arms)
        self.R_funcs = R_funcs
        self.v = np.array(v, dtype=float)
        self.bias_funcs = bias_funcs
        self.delta = delta
        self.c = c
        
        # Statistics
        self.n = np.zeros(self.M, dtype=int)  # Number of times each arm selected
        self.U = np.zeros(self.M, dtype=float)  # Cumulative reward per arm
        self.active_set = set(range(self.M))
        self.num_steps = 0

    def _confidence_bound(self, i: int) -> float:
        """Compute confidence band for arm i."""
        if self.n[i] == 0:
            return np.inf
        return self.c * np.sqrt(np.log(self.M * np.log(max(2, self.n[i])) / self.delta) / self.n[i])

    def _adjusted_reward(self, i: int) -> float:
        """Compute adjusted reward for arm i."""
        if self.n[i] == 0:
            return 0.0
        return self.U[i] / self.n[i] - self.bias_funcs[i](self.n[i])

    def step(self, x: np.ndarray, y: Any) -> None:
        """Execute one step of Dynamic Balancing algorithm."""
        # Select arm from active set
        candidates = [(self.v[i] * self.R_funcs[i](self.n[i]), i) for i in self.active_set]
        _, i = min(candidates, key=lambda x: x[0])
        arm = self.arms[i]
        
        # Get loss from selected arm
        loss_t = arm.get_loss(x, y)
        reward = 1.0 - loss_t  # Convert loss to reward
        
        # Update history
        self.selection_history.append(i)
        self.loss_history.append(loss_t)
        
        # Optimize selected arm
        arm.step(x, y, self.num_steps)
        
        # Update statistics
        self.n[i] += 1
        self.U[i] += reward
        
        # Recompute active set
        eta = np.zeros(self.M)
        gamma = np.zeros(self.M)
        scores = np.zeros(self.M)
        
        for j in range(self.M):
            if self.n[j] > 0:
                eta[j] = self._adjusted_reward(j)
                gamma[j] = self._confidence_bound(j)
                scores[j] = eta[j] + gamma[j] + self.R_funcs[j](self.n[j]) / self.n[j]
        
        max_score = np.max(eta + gamma)
        self.active_set = {j for j in range(self.M) if scores[j] >= max_score}
        
        self.num_steps += 1


class DDRB(Strategy):
    """Data-Driven Regret Balancer (D3RB/ED2RB) adapted to Strategy base class.
    https://arxiv.org/pdf/2306.02869
    """
    
    def __init__(self, arms: List[Arm], d_min: float = 1.0, delta: float = 0.05, 
                 c: float = 1.0, mode: str = "ED2RB", num_optimize: int = 1, **kwargs) -> None:
        """
        :param arms: List of arms to optimize
        :param d_min: Minimum regret coefficient
        :param delta: Confidence parameter
        :param c: Confidence bound multiplier
        :param mode: Algorithm mode ("D3RB" or "ED2RB")
        :param num_optimize: Number of arms to optimize at each step
        """
        super().__init__(arms, num_optimize)
        self.M = len(arms)
        self.d_min = d_min
        self.delta = delta
        self.c = c
        self.mode = mode
        
        # Statistics
        self.n = np.zeros(self.M, dtype=int)  # Number of times each arm selected
        self.u = np.zeros(self.M, dtype=float)  # Total value (reward)
        self.dhat = np.full(self.M, d_min)  # Regret coefficients
        self.phi = np.full(self.M, d_min)  # Balancing potentials
        self.num_steps = 0

    def _confidence_bound(self, n: int) -> float:
        """Compute confidence bound."""
        if n <= 0:
            return 1e6
        return np.sqrt(np.log(self.M * max(1.0, np.log(n)) / self.delta) / n)

    def step(self, x: np.ndarray, y: Any) -> None:
        """Execute one step of Data-Driven Regret Balancer algorithm."""
        # Select arm with minimal phi
        i = int(np.argmin(self.phi))
        arm = self.arms[i]
        
        # Get loss from selected arm
        loss_t = arm.get_loss(x, y)
        reward = 1.0 - loss_t  # Convert loss to reward
        
        # Update history
        self.selection_history.append(i)
        self.loss_history.append(loss_t)
        
        # Optimize selected arm
        arm.step(x, y, self.num_steps)
        
        # Update statistics
        self.n[i] += 1
        self.u[i] += reward
        
        # Update regret coefficients and potentials based on mode
        if self.mode == "D3RB":
            # Misspecification test
            if self.n[i] > 0:
                lhs = (self.u[i] / self.n[i]) + (self.dhat[i] * np.sqrt(self.n[i]) / self.n[i]) + self.c * self._confidence_bound(self.n[i])
                rhs = max([(self.u[j] / self.n[j] - self.c * self._confidence_bound(self.n[j])) if self.n[j] > 0 else -1e9 for j in range(self.M)])
                if lhs < rhs:
                    self.dhat[i] *= 2
                self.phi[i] = self.dhat[i] * np.sqrt(self.n[i])
        else:  # ED2RB
            if self.n[i] > 0:
                max_adj = max([(self.u[j] / self.n[j] - self.c * self._confidence_bound(self.n[j])) if self.n[j] > 0 else -1e9 for j in range(self.M)])
                val = np.sqrt(self.n[i]) * (max_adj - (self.u[i] / self.n[i]) - self.c * self._confidence_bound(self.n[i]))
                self.dhat[i] = max(self.d_min, val)
                new_phi = self.dhat[i] * np.sqrt(self.n[i])
                self.phi[i] = np.clip(new_phi, self.phi[i], 2 * self.phi[i])
        
        self.num_steps += 1


class LimitedAdvice(Strategy):
    """Limited Advice algorithm adapted to Strategy base class.
    https://proceedings.mlr.press/v32/seldin14.pdf
    """

    def __init__(self, arms: List[Arm], eta: float = 0.1, num_optimize: int = 1, **kwargs) -> None:
        """
        :param arms: List of arms to optimize
        :param eta: Learning rate parameter
        :param num_optimize: Number of arms to optimize at each step
        """
        super().__init__(arms, num_optimize)
        self.K = len(arms)
        self.eta = eta

        # Initialize probabilities uniformly
        self.p = np.ones(self.K) / self.K
        self.cumulative_losses = np.zeros(self.K)
        self.num_steps = 0

    def step(self, x: np.ndarray, y: Any) -> None:
        """Execute one step of Limited Advice algorithm."""
        # Sample expert based on current probabilities
        i = np.random.choice(self.K, p=self.p)
        arm = self.arms[i]

        # Get loss from selected arm
        loss_t = arm.get_loss(x, y)

        # Update history
        self.selection_history.append(i)
        self.loss_history.append(loss_t)

        # Optimize selected arm
        arm.step(x, y, self.num_steps)

        # Select additional arms to optimize (if needed)
        if self.num_optimize > 1:
            indices = set(range(self.K)) - {i}
            other_indices = np.random.choice(list(indices), size=min(self.num_optimize - 1, len(indices)), replace=False)
            update_indices = np.concatenate(([i], other_indices))
        else:
            update_indices = [i]
        
        assert len(update_indices) == self.num_optimize

        # Update probabilities for all arms that were optimized
        for j in update_indices:
            # Calculate adjusted loss using importance weighting
            l_t = self.arms[j].step(x, y, self.num_steps)
            adjusted_loss = l_t / (self.p[j] + (1 - self.p[j]) * (self.num_optimize - 1) / (self.K - 1))

            self.cumulative_losses[j] += adjusted_loss

            # Update probability using exponential weights
            self.p[j] = self.p[j] * np.exp(-self.eta * adjusted_loss)

        # Normalize probabilities
        self.p = self.p / np.sum(self.p)
        self.num_steps += 1


