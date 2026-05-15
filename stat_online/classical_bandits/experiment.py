

# Now implement class to do experiment, given environment and bandit algorithm

from collections import defaultdict
from typing import List

import numpy as np

from stat_online.classical_bandits.algorithm import BaseBandit, BaseModelSelection, MChoosenArm, MReward, Reward
from stat_online.classical_bandits.environment import BaseEnvironment


class Experiment:
    def __init__(self,
            environment: BaseEnvironment,
            algorithm: BaseBandit,
            ) -> None:
        self.environment = environment
        self.algorithm = algorithm

        self.reward_history = []
        self.arm_selection_history = []
        self.best_reward_history = []

    def run(self, n_steps):


        # do initial pulls
        # for i in range(self.environment.num_arms):
        #     reward = self.environment.pull(i)
        #     self
            # update arm as
        for _ in range(n_steps):
            # algorithm reward viewing
            choosen_arms = self.algorithm.get_arm()
            arm_t = choosen_arms.arm
            reward_t = self.environment.pull(arm_t)

            # update
            if isinstance(choosen_arms, MChoosenArm):
                rewards = []
                per_arm_reward = defaultdict()
                for arm in choosen_arms.exploration_chooses:
                    rew = per_arm_reward.get(arm, self.environment.pull(arm))
                    per_arm_reward[arm] = rew
                    rewards.append(rew)
                # print(rewards)
                reward = MReward(
                    reward=reward_t,
                    exploration_rewards=rewards
                )
            else:
                reward = Reward(reward=reward_t)

            # print("reward", reward)
            self.algorithm.update(choosen_arms, reward)

            self.reward_history.append(reward_t)
            self.arm_selection_history.append(arm_t)
            self.best_reward_history.append(self.environment.get_best_reward())  # made to make it dependable on time

    def get_cumulative_regret(self):
        best_rewards = np.array(self.best_reward_history)
        obtained_rewards = np.array(self.reward_history)
        regret = np.cumsum(best_rewards - obtained_rewards)
        return regret

    def get_expected_regret(self):
        best_reward = self.environment.get_best_reward()
        ground_truth = self.environment.get_true_rewards()
        # print(ground_truth)
        obtained_rewards = np.array([ground_truth[arm] for arm in self.arm_selection_history])
        regret = np.cumsum(best_reward - obtained_rewards)
        return regret


class ListExperiment:
    """
    in this experiment we consider, that each bandit have its own environment
    """
    def __init__(self,
            environment_list: List[BaseEnvironment],
            algorithm: BaseModelSelection,
            ) -> None:

        self.environment_list = environment_list
        self.algorithm = algorithm

        assert len(environment_list) == len(self.algorithm.bandit_algorithms)

        self.reward_history = []
        self.arm_selection_history = []
        self.best_reward_history = []

    def get_best_reward(self):
        return max([env.get_best_reward() for env in self.environment_list])

    def run(self, n_steps):


        # do initial pulls
        # for i in range(self.environment.num_arms):
        #     reward = self.environment.pull(i)
        #     self
            # update arm as
        for _ in range(n_steps):
            # algorithm reward viewing
            choosen_arms: MChoosenArm = self.algorithm.get_arm()

            arm_t = choosen_arms.arm
            expert_t = choosen_arms.decision_arm
            reward_t = self.environment_list[expert_t].pull(arm_t)

            # update
            if isinstance(choosen_arms, MChoosenArm):
                rewards = []

                for arm, expert in zip(
                                    choosen_arms.exploration_chooses,
                                    choosen_arms.exploration_arms):
                    rew = self.environment_list[expert].pull(arm)
                    rewards.append(rew)
                # print(rewards)
                reward = MReward(
                    reward=reward_t,
                    exploration_rewards=rewards
                )
            else:
                reward = Reward(reward=reward_t)

            # print("reward", reward)
            self.algorithm.update(choosen_arms, reward)

            self.reward_history.append(reward_t)
            self.arm_selection_history.append((expert_t, arm_t))
            self.best_reward_history.append(self.get_best_reward())  # made to make it dependable on time

    def get_cumulative_regret(self):
        best_rewards = np.array(self.best_reward_history)
        obtained_rewards = np.array(self.reward_history)
        regret = np.cumsum(best_rewards - obtained_rewards)
        return regret

    def get_expected_regret(self):
        best_reward = self.get_best_reward()
        ground_truth = [env.get_true_rewards() for env in self.environment_list]
        # print(ground_truth)
        obtained_rewards = np.array([ground_truth[expert][arm] for expert, arm in self.arm_selection_history])
        regret = np.cumsum(best_reward - obtained_rewards)
        return regret
