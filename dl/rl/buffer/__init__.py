from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Iterable, Union, Dict, Tuple
from dataclasses import dataclass

import torch
import numpy as np

from .deque_buffer import UniformDequeBuffer

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer(ABC):
    storage = None

    def __init__(self, maxlen: int = 2000):
        self.maxlen = maxlen

    @abstractmethod
    def add(self, transition: Union[Transition, Iterable[Transition]]):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, size: int):
        raise NotImplementedError()

    @abstractmethod
    def get(self, indices: Iterable[int]):
        raise NotImplementedError()

    @staticmethod
    def to_torch(transitions: Iterable[Transition], device: Union[torch.device, str] = "cpu") \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.tensor(np.array(state), device=device, dtype=torch.float)
        action = torch.tensor(np.array(action), device=device, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), device=device, dtype=torch.float)
        reward = torch.tensor(np.array(reward), device=device, dtype=torch.float)
        done = torch.tensor(np.array(done), device=device, dtype=torch.float)
        return state, action, reward, next_state, done


class BufferFactory:
    UNIFORM: str = "uniform"
    MAPPING = {
        "uniform": UniformDequeBuffer
    }

    @staticmethod
    def create(type_: str, maxlen: int = 2000) -> ReplayBuffer:
        return BufferFactory.MAPPING[type_](maxlen)
