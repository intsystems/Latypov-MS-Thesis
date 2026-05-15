
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from stat_online.classical_bandits.environment import BaseEnvironment
from stat_online.utils.forecasters import LogBarrierOMD


@dataclass
class ChoosenArm:
    arm: int

@dataclass
class MChoosenArm(ChoosenArm):
    """
    Parameters:
        arm: int - the choosen arm to make decision
        exploration_arms: List[int] - the arms chosen aming M algorithms
        exploration_chooses: List[int] - the arms chosen by each M algorithm
    """
    decision_arm: int
    exploration_arms: List[int]
    exploration_chooses: List[int]

@dataclass
class Reward:
    reward: float

@dataclass
class MReward(Reward):
    """
    Parameters:
        exploration_rewards: List[float] - rewards obtained by exploration arms
    """
    exploration_rewards: List[float]


class BaseBandit:
    def __init__(self, K) -> None:
        self.K = K
        self.counts = np.zeros(K) + 1e-10
        self.rewards = np.zeros(K, dtype=float)
        self.num_steps = 0

    @abstractmethod
    def get_arm(self) -> ChoosenArm:
        """
        should choose and return some arm
        """
        pass

    @abstractmethod
    def update(self, arm: ChoosenArm, reward: Reward):
        self.counts[arm.arm] += 1
        self.rewards[arm.arm] += reward.reward
        self.num_steps += 1

    @abstractmethod
    def regret_bounds(self) -> float:
        """
        returns theoretical regret bounds for that moment
        """
        raise NotImplementedError


class EpsilonGreedy(BaseBandit):
    def __init__(self, K, epsion=0.1, c = 0.1) -> None:
        super().__init__(K)
        self.eps = epsion
        self.num_steps = 0
        self.c = c

    @property
    def n_s(self):
        return self.num_steps - self.K + 1
    @property
    def eps_t(self):
        return self.eps * (self.K * np.log(self.n_s)/ self.n_s) ** (1/3)

    def get_arm(self) -> ChoosenArm:
        """
        choose and return arm to pull
        """

        if self.num_steps < self.K:
            return ChoosenArm(arm=self.num_steps)


        if np.random.rand() < self.eps_t:
            # randomly choose
            return ChoosenArm(arm=np.random.choice(self.K))

        return ChoosenArm(arm=int(np.argmax(self.rewards / self.counts)))

    def regret_bounds(self) -> float:
        if self.num_steps < self.K:
            return float('inf')
        return self.c * (self.n_s) ** (2/3) * (self.K * np.log(self.n_s)) ** (1/3)
        # return self.n_s * self.eps_t + \
        #     2 * (2 * self.K * self.n_s * np.log(self.n_s)/ self.eps_t) ** 0.5

class UCB(BaseBandit):
    def __init__(self, K, c) -> None:
        super().__init__(K)
        self.c = c

    @property
    def exploration_bonus(self):
        return self.c * np.sqrt(np.log(self.num_steps) / (self.counts + 1e-8))

    def get_arm(self) -> ChoosenArm:
        if self.num_steps < self.K:
            return ChoosenArm(arm=self.num_steps)

        # else choose by ucb indices
        mean_rewards = self.rewards / self.counts
        return ChoosenArm(arm=int(np.argmax(mean_rewards + self.exploration_bonus)))

    def regret_bounds(self):
        if self.num_steps < self.K:
            return float('inf')

        return np.sum(self.c * np.sum(np.sqrt(self.counts)) * \
                      np.sqrt(np.log(self.num_steps)))


class BaseModelSelection(BaseBandit):
    """
    base class for algorithms with model selectio
    """
    def __init__(self, bandit_algorithms, M) -> None:
        super().__init__(len(bandit_algorithms))
        self.bandit_algorithms = bandit_algorithms
        self.M = M

        # how much times each algorithm was chosen for decision making
        self.selection_for_decisions = np.zeros(len(bandit_algorithms))

        self.total_pulls = 0  # grows to M at every step


    def smoothed_decision(self, arm: int):
        """
        safe advice as in paper
        here we define empirical distribution for each arm
        and sample from it. It will give us some good properties
        """
        posterior = self.bandit_algorithms[arm].counts + 1e-8 # to prevent division to zero

        posterior = (posterior) / np.sum(posterior)
        sample = np.random.choice(len(posterior), p=posterior)


        return sample

    def get_confidence_bounds(self):
        """
        returns confidence bounds for each arm using its regret_bounds
        """
        mean_rewards = self.rewards / self.counts
        exploration_bonus = [a.regret_bounds()/self.counts[i] for i, a in enumerate(self.bandit_algorithms)]

        log_scaler = np.log(self.num_steps/self.delta + 1)/(self.counts + 1e-8)
        upper_bonus = self.c * np.sqrt(log_scaler)
        lower_bonus = self.c * np.sqrt(log_scaler ) + \
                        0.66 * log_scaler

        lower_bounds = mean_rewards - lower_bonus
        upper_bonus = mean_rewards + exploration_bonus + lower_bonus

        return lower_bounds, upper_bonus


class MLCB(BaseModelSelection):
    name = "M-LCB"
    """
    This is our algorithm.
    It holds M algorithms of MAB as its own arms, and chooses among them by UCB strategy.
    Then choosen arm chooses environment arm to pull

    Params:
        bandit_algorithms: list of bandit algorithms to choose from
        M: number of bandit algorithms. This number shows the number of arms,
            which will do exploration at each step
    """
    def __init__(self,
                 bandit_algorithms: list,
                 M: int = 1,
                 c: float = 0.1,
                 delta = 0.1) -> None:
        BaseModelSelection.__init__(self, bandit_algorithms, M)
        self.c = c
        self.delta = delta

    def get_arm(self) -> MChoosenArm:
        decision_arm = None  # arm to make decision
        exploration_arms = []  # chosen M algorithms to do exploration
        # if self.total_pulls < self.K:
        #     decision_arm = self.num_steps
        #     exploration_arms = [i % self.K for i in range(self.num_steps,self.num_steps + self.M)]

        # else choose by ucb indices
        # first choose M algorithms by UCB
        lower_bounds, upper_bounds = self.get_confidence_bounds()

        exploration_arms = np.argsort(upper_bounds)[-self.M:].tolist()
        decision_arm = max(exploration_arms, key=lambda i: lower_bounds[i])

        exploration_arms_chooses = []
        for arm in exploration_arms:
            chosen = self.bandit_algorithms[arm].get_arm()
            exploration_arms_chooses.append(chosen.arm)

        # now choose decisions for arms
        decision = self.smoothed_decision(decision_arm)

        return MChoosenArm(arm=decision,
                           decision_arm=decision_arm,
                           exploration_arms=exploration_arms,
                           exploration_chooses=exploration_arms_chooses)

    def update(self, arm: MChoosenArm, reward: MReward):
        self.total_pulls += len(arm.exploration_arms)
        self.num_steps += 1
        self.selection_for_decisions[arm.decision_arm] += 1

        for exp_arm, exp_chosen, exp_rew in zip(arm.exploration_arms,
                                                arm.exploration_chooses,
                                                reward.exploration_rewards
                                                ):
            # for each that arm do update and update this algorithm counters
            self.counts[exp_arm] += 1
            self.rewards[exp_arm] += exp_rew
            # print(f"Updating arm {exp_arm} with reward {exp_rew} gt arm {exp_chosen}")
            self.bandit_algorithms[exp_arm].update(
                    arm=ChoosenArm(exp_chosen),
                    reward=Reward(exp_rew)
                )

    def regret_bounds(self):
        """
        for now skipped
        """
        raise NotImplementedError

def gumbel_sample(logits):
    # Генерируем шум Gumbel из равномерного шума
    u = np.random.uniform(size=len(logits))
    gumbel_noise = -np.log(-np.log(u))
    # Просто берем аргмакс от суммы
    return np.argmax(logits + gumbel_noise)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs

class LimitedAdvice(BaseModelSelection):
    def __init__(self,
                 bandit_algorithms: list[BaseBandit],
                 M: int = 1,
                 eta_scaler: float = 1,
                 ) -> None:
        BaseModelSelection.__init__(self, bandit_algorithms, M)
        self.eta_scaler = eta_scaler

        self.cumulative_losses = np.zeros(len(bandit_algorithms))
        self.p = np.zeros(self.K)

    @property
    def eta_t(self,):
        # +1 так как чисо шагов на один запаздывает
        return self.eta_scaler * (self.M * np.log(self.K)/ ( self.K * (self.num_steps + 1) ) ) ** (1/2)

    def get_arm(self) -> ChoosenArm:

        # sample arm to make decision
        logits = - self.eta_t * self.cumulative_losses
        self.p = softmax(logits)
        decision_algo_idx = np.random.choice(self.K, p=self.p)

        # select arms to optimize

        exploration_arms = [decision_algo_idx]

        if self.M > 1:
            indices = list(set(range(self.K)) - {decision_algo_idx})
            if indices:
                other_indices = np.random.choice(
                    indices,
                    size=min(self.M - 1, len(indices)),
                    replace=False)
                exploration_arms.extend(other_indices)

        # get arms from indices
        exploration_arms_choses = []

        # [TODO] не используется сглаживание!!
        for algo_idx in exploration_arms:
            chosen = self.bandit_algorithms[algo_idx].get_arm()
            exploration_arms_choses.append(chosen.arm)

        decision_action = exploration_arms_choses[0]

        return MChoosenArm(
            arm=decision_action,
            decision_arm=decision_algo_idx,
            exploration_arms=exploration_arms,
            exploration_chooses=exploration_arms_choses
        )

    def update(self, arm: MChoosenArm, reward: MReward):
        self.total_pulls += len(arm.exploration_arms)
        self.num_steps += 1
        self.selection_for_decisions[arm.decision_arm] += 1

        for exp_arm, exp_chosen, exp_rew in zip(arm.exploration_arms,
                                                arm.exploration_chooses,
                                                reward.exploration_rewards
                                                ):
            # for each that arm do update and update this algorithm counters
            self.counts[exp_arm] += 1
            self.rewards[exp_arm] += exp_rew
            self.bandit_algorithms[exp_arm].update(
                    arm=ChoosenArm(exp_chosen),
                    reward=Reward(exp_rew)
                )
            # limited advice Weight update

            loss_t = 1.0 - exp_rew

            prob_inclision = self.p[exp_arm] + (1 - self.p[exp_arm]) * ((self.M - 1)/(max(1, self.K - 1)))
            adjusted_loss = loss_t / prob_inclision

            self.cumulative_losses[exp_arm] += adjusted_loss


class DDRB(BaseModelSelection):
    """
    Data-Driven Regret Balancer (D3RB/ED2RB) adapted for selecting bandit algorithms.
    It treats other bandit algorithms as arms.
    """
    def __init__(self,
                 bandit_algorithms: list[BaseBandit],
                 M: int = 1,
                 d_min: float = 1.0,
                 delta: float = 1.0,
                 c: float = 1.0,
                 mode: Literal["ED2RB", "D3RB"] = "ED2RB",
                 ) -> None:
        """
        :param bandit_algorithms: List of bandit algorithms to choose from
        :param M: Number of algorithms to run (optimize) at each step (M)
        :param d_min: Minimum regret coefficient
        :param delta: Confidence parameter
        :param c: Confidence bound multiplier
        :param mode: Algorithm mode ("D3RB" or "ED2RB")
        """

        BaseModelSelection.__init__(self, bandit_algorithms, M)

        self.delta = delta
        self.d_min = d_min
        self.c = c

        self.mode = mode

        self.dhat = np.full(self.M, d_min)
        self.phi = np.full(self.M, d_min)

    def get_arm(self) -> ChoosenArm:
        # 1. Select candidates to optimize (exploration arms) based on minimal phi
        # If we need to select multiple, we select top-k with smallest phi
        exploration_arms = np.argsort(self.phi)[:self.M].tolist()

        # 2. Select decision arm. In DDRB usually the one with smallest phi is "best" currently
        decision_algo_idx = exploration_arms[0]

        # 3. Get actual actions from selected algorithms
        exploration_arms_chooses = []
        for algo_idx in exploration_arms:
            chosen = self.bandit_algorithms[algo_idx].get_arm()
            exploration_arms_chooses.append(chosen.arm)  # Assuming get_arm returns object with .arm attribute

        # 4. Get decision from the decision algorithm
        # Using the same logic as MLCB: smoothed decision or direct
        # Here we just ask the chosen algo what it would do
        decision_action = exploration_arms_chooses[0]

        return MChoosenArm(arm=decision_action,
                           decision_arm=decision_algo_idx,
                           exploration_arms=exploration_arms,
                           exploration_chooses=exploration_arms_chooses)

    def _confidence_bound(self, n: int) -> float:
        """Compute confidence bound."""
        if n <= 0:
            return 1e6
        return np.sqrt(np.log(self.M * max(1.0, np.log(n)) / self.delta) / n)

    def update(self, arm: MChoosenArm, reward: MReward):
        self.num_steps += 1
        self.total_pulls += len(arm.exploration_arms)
        self.selection_for_decisions[arm.decision_arm] += 1

        # Precompute mean rewards and bounds for all arms (for the RHS of inequality)
        # Handle division by zero for unpulled arms
        means = np.divide(self.rewards, self.counts, out=np.zeros_like(self.rewards), where=self.counts>0)
        bounds = np.array([self.c * self._confidence_bound(c) for c in self.counts])
        adj_values = means - bounds
        # For unpulled arms, set value to -large
        adj_values[self.counts == 0] = -1e9
        max_adj = np.max(adj_values)


        # Iterate over all algorithms that were selected for optimization/exploration
        for exp_algo_idx, exp_action, exp_rew in zip(arm.exploration_arms,
                                                     arm.exploration_chooses,
                                                     reward.exploration_rewards):

            # 1. Update internal BaseBandit stats
            self.counts[exp_algo_idx] += 1
            self.rewards[exp_algo_idx] += exp_rew

            # 2. Update the sub-algorithm itself
            self.bandit_algorithms[exp_algo_idx].update(
                arm=ChoosenArm(exp_action),
                reward=Reward(exp_rew)
            )

            # 3. DDRB Specific Logic
            i = exp_algo_idx
            n_i = self.counts[i]
            u_i = self.rewards[i] # This assumes reward is accumulated. If u is average, divide by n_i

            # Note: The original code used u as sum of rewards.
            # We need mean reward for the formulas below.
            mean_reward_i = u_i / n_i

            if self.mode == "D3RB":
                # Misspecification test
                conf_i = self.c * self._confidence_bound(n_i)
                lhs = mean_reward_i + (self.dhat[i] * np.sqrt(n_i) / n_i) + conf_i

                if lhs < max_adj:
                    self.dhat[i] *= 2
                self.phi[i] = self.dhat[i] * np.sqrt(n_i)

            else:  # ED2RB
                conf_i = self.c * self._confidence_bound(n_i)
                # Term: sqrt(n) * (max_rhs - mean - conf)
                val = np.sqrt(n_i) * (max_adj - mean_reward_i - conf_i)

                self.dhat[i] = max(self.d_min, val)
                new_phi = self.dhat[i] * np.sqrt(n_i)
                self.phi[i] = np.clip(new_phi, self.phi[i], 2 * self.phi[i])


class SmoothCORRAL(BaseModelSelection):
    """CORRAL algorithm adapted to Strategy base class.
    https://proceedings.mlr.press/v65/agarwal17b/agarwal17b.pdf#page=4.77
    """
    def __init__(self,
                 bandit_algorithms: list,
                 M: int,
                 T: int,
                 eta: float = 0.1,
                 ) -> None:
        BaseModelSelection.__init__(self, bandit_algorithms, M)

        self.T = T
        self.gamma = 1.0 / T
        self.beta = np.exp(1.0 / np.log(T + 1.0))
        self.eta = eta

        # Initialize distributions
        self.p = np.ones(self.K) / self.K
        self.pbar = self.p.copy()
        self.rho = np.full(self.K, 2 * self.K)
        self.eta_vec = np.full(self.K, eta)
        self.num_steps = 0

    def get_arm(self) -> ChoosenArm:
        decision_arm = np.random.choice(self.K, p = self.pbar)

        decision = self.smoothed_decision(decision_arm)

        exploration_arms = [decision_arm]
        exoloration_arms_chooses = [self.bandit_algorithms[decision_arm].get_arm().arm]
        return MChoosenArm(arm=decision,
                           decision_arm=decision_arm,
                           exploration_arms=exploration_arms,
                           exploration_chooses=exoloration_arms_chooses
                    )

    def update(self, arm: MChoosenArm, reward: MReward):
        self.total_pulls += len(arm.exploration_arms)
        self.num_steps += 1
        self.selection_for_decisions[arm.decision_arm] += 1

        # 1. update arm parameters
        selected_arm = arm.exploration_arms[0]
        selected_arm_inn = arm.exploration_chooses[0]
        selected_reward = reward.exploration_rewards[0]

        self.bandit_algorithms[selected_arm].update(
            arm=ChoosenArm(selected_arm_inn),
            reward=Reward(selected_reward)
        )

        # 2. Specific updates
        ell = np.zeros(self.K)

        i = selected_arm
        loss_t = 1 - selected_reward
        ell[i] = loss_t / max(self.pbar[i], 1e-8)

        p_next = LogBarrierOMD.log_barrier_omd(self.p, ell, self.eta_vec)

        pbar_next = (1 - self.gamma) * p_next + self.gamma * (np.full(self.K, 1/self.K))

        for j in range(self.K):
            if 1.0 / pbar_next[j] > self.rho[j]:
                self.rho[j] = 2.0 / pbar_next[j]
                self.eta_vec[j] = self.beta * self.eta_vec[j]
        self.p = p_next
        self.pbar = pbar_next
