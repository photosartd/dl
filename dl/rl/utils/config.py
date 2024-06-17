import os
import logging
from dataclasses import dataclass
from typing import Dict, Union, Callable

import torch

from dl.rl.buffer import BufferFactory
from dl.rl.utils.networks import BaseAlgorithm, AlgorithmFactory


@dataclass
class Config:
    name: str
    networks: Dict[str, torch.nn.Module]
    savepath: str = "./agents"
    log_level: int = logging.INFO
    stats_maxlen: int = 2000
    stats_period: int = 10
    replay_buffer: str = BufferFactory.UNIFORM
    replay_maxlen: int = 2000
    device: Union[str, torch.device] = "cpu"
    batch_size: int = 256
    num_batches_to_sample: int = 16
    evaluate_times: int = 100

    @property
    def savepath_full(self):
        return os.path.join(self.savepath, self.name)

    @property
    def nets(self) -> BaseAlgorithm:
        return AlgorithmFactory.load(self.name, self.networks)

    @property
    def min_batch_sampling(self) -> int:
        return self.batch_size * self.num_batches_to_sample


@dataclass
class TD3Config(Config):
    gamma: float = 0.99
    tau: float = 0.005
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    noise: float = 0.2
    noise_clip: float = 0.5
    critic_loss: Callable = torch.nn.MSELoss()
