import os
from logging import Logger
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
import gymnasium as gym

from dl.rl.utils.config import Config
from dl.rl.utils.networks import BaseAlgorithm
from dl.utils.logger import get_logger
from dl.rl.buffer import Transition, ReplayBuffer, BufferFactory
from dl.utils.statistics import RLStatistics, RLEntry


class BaseAgent(ABC):
    """
    Base class for RL agent that has the following methods:
    - save
    - load
    - act
    """
    def __init__(self, config: Config):
        self.config: Config = config
        self.logger: Logger = get_logger(config.log_level)
        self.networks: BaseAlgorithm = self.config.nets
        self.statistics: RLStatistics = RLStatistics(config.stats_maxlen, send_each=config.stats_period)
        self.replay_buffer: ReplayBuffer = BufferFactory.create(config.replay_buffer, config.replay_maxlen)

    def save(self) -> None:
        os.makedirs(self.config.savepath_full, exist_ok=True)
        for name, network in self.config.networks.items():
            torch.save(network.state_dict(), os.path.join(self.config.savepath_full, f"{name}.pt"))
        self.logger.info(f"Agent saved to {self.config.savepath_full}")

    @classmethod
    def load(cls, config: Config):
        instance = cls(config)
        for model_name in os.listdir(config.savepath_full):
            network_name = model_name.split(".")[0]
            model = config.networks[network_name]
            model.load_state_dict(torch.load(os.path.join(config.savepath_full, model_name)))
        return instance

    @abstractmethod
    def act(self, state) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update(self, transition: Transition) -> None:
        raise NotImplementedError

    def evaluate(self, env: gym.Env, episodes: int = 10) -> List[float]:
        episode_rewards = []
        for episode in range(episodes):
            done = False
            state = env.reset()
            total_reward = 0

            while not done:
                state, reward, done, _, _ = env.step(self.act(state)) #TODO check what the type of state
                total_reward += reward
            episode_rewards.append(total_reward)
        return episode_rewards
