from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from dl.rl.agents.base import BaseAgent
from dl.rl.buffer import Transition
from dl.utils.statistics import RLEntry


@dataclass
class Policy:
    pass


@dataclass
class EpsGreedyPolicy(Policy):
    epsilon: float = 0.3
    epsilon_final: float = 0.01
    epsilon_decay: float = 40000.0

    def current_epsilon(self, step: int) -> float:
        current_epsilon = self.epsilon + (self.epsilon_final - self.epsilon) * step / self.epsilon_decay
        current_epsilon = max(current_epsilon, self.epsilon_final)
        return current_epsilon

    def noise(self, shape: np.shape, current_step: int) -> np.ndarray:
        return self.current_epsilon(current_step) * np.random.randn(*shape)

    def clipped_action(self, action: np.ndarray, current_step: int, min_: float = -1.0,
                       max_: float = 1.0) -> np.ndarray:
        return np.clip(action + self.noise(action.shape, current_step), min_, max_)


class AgentTrainer(ABC):
    def __init__(self, agent: BaseAgent, env: gym.Env, policy: Policy):
        self.agent = agent
        self.env = env
        self.test_env = deepcopy(self.env)
        self.policy = policy

    @abstractmethod
    def train(self, n_episodes: int = 10000) -> None:
        raise NotImplementedError

    def evaluate(self, current_step: int, n_episodes: int) -> float:
        reward_mean = -np.inf
        if (current_step + 1) % (n_episodes // self.agent.config.evaluate_times) == 0:
            rewards = self.agent.evaluate(self.test_env)
            reward_mean = np.mean(rewards)
            self.agent.statistics.update(
                RLEntry(step=current_step, reward_mean=reward_mean, reward_std=np.std(rewards)),
                send=True
            )
        return reward_mean


class GreedyTrainer(AgentTrainer):
    def __init__(self, agent: BaseAgent, env: gym.Env, policy: EpsGreedyPolicy):
        super().__init__(agent, env, policy)
        self.policy = policy

    def train(self, n_episodes=10000) -> None:
        state = self.env.reset()
        best_reward = -np.inf
        for episode in range(n_episodes):
            action = self.agent.act(state)
            perturbed_action = self.policy.clipped_action(action, episode)
            next_state, reward, done, _, _ = self.env.step(perturbed_action)
            self.agent.update(Transition(state, perturbed_action, next_state, reward, done))
            state = next_state if not done else self.env.reset()
            current_reward = self.evaluate(episode, n_episodes)
            if current_reward > best_reward:
                best_reward = current_reward
                self.agent.save()
