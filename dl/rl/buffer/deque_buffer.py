import random
from typing import Iterable, Union
from collections import deque

from . import ReplayBuffer, Transition


class UniformDequeBuffer(ReplayBuffer):
    def __init__(self, maxlen: int):
        super().__init__(maxlen)
        self.storage = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.storage)

    def add(self, transition: Union[Transition, Iterable[Transition]]):
        if isinstance(transition, Transition):
            self.storage.append(transition)
        else:
            self.storage.extend(transition)

    def sample(self, size: int) -> Iterable[Transition]:
        if size > len(self):
            raise IndexError(f'Size was more that buffer size')
        return random.sample(self.storage, size)

    def get(self, indices: Iterable[int]):
        return [self.storage[idx] for idx in indices]
