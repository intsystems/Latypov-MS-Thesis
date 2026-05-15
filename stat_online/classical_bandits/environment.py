
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union
import numpy as np
from typing import Literal

from stat_online.utils.forecasters import LogBarrierOMD


class BaseEnvironment:
    """
    Base class for environment in bandit problems.
    """

    @abstractmethod
    def pull(self, arm: int) -> float:
        """
        samples reward for chosen arm and returns it
        """
        pass

    @property
    @abstractmethod
    def num_arms(self):
        """
        returns number of arms in environment
        """
        pass

    @abstractmethod
    def get_true_rewards(self,) -> list[float]:
        """
        returns true rewards of all arms in environment
        """
        pass

    @abstractmethod
    def get_best_reward(self) -> float:
        """
        returns the best reward in environment
        """
        pass

    @abstractmethod
    def get_best_arm(self):
        """
        returns the best arm in environment
        """
        pass


class BernoulliBanditEnvironment(BaseEnvironment):
    """
    this is an environment class for classic bandit problem
    With K arms, which rewards have some bernoulli distributions
    with parameters p_i for all i \in [K]

    Args:
        K: number of arms of bandit
        probs: probabilities of success of different arms. If not given,
               then take uniformly

    """

    def __init__(self, K, probs: Union[list, None] = None) -> None:
        self.K = K

        if probs is None:
            probs = np.cumsum(np.ones((K,), float) / (K + 1))

        assert probs is not None
        # to prevent memorization
        np.random.shuffle(probs)
        self.probs = probs


    def pull(self, arm: int) -> float:
        """
        samples reward for chosen arm and returns it
        """
        assert 0 <= arm <= self.K

        return np.random.binomial(n=1, p=self.probs[arm])

    @property
    def num_arms(self):
        """
        returns number of arms in environment
        """
        return self.K
    def get_true_rewards(self,) -> list[float]:
        """
        returns true rewards of all arms in environment
        """
        return self.probs
    def get_best_reward(self):
        return max(self.probs)

    def get_best_arm(self):
        """
        returns the best arm in environment
        """
        return int(np.argmax(self.probs))
