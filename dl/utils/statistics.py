import logging
from abc import ABC, abstractmethod
from typing import Dict, Deque
from dataclasses import dataclass, field
from collections import deque
from dl.utils.logger import get_logger


logger = get_logger(logging.INFO)


@dataclass
class Entry(object):
    step: int = 0
    loss: float = 0.0
    others: Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        return f"""
        Step: {self.step}
        Loss: {self.loss}
        Others: {self.others}
        """


@dataclass
class RLEntry(Entry):
    reward_mean: float = 0.0
    reward_std: float = 0.0

    def __str__(self) -> str:
        parent_str = super().__str__()
        return f"""
        {parent_str}
        Reward mean: {self.reward_mean}
        Reward std: {self.reward_std}
        """


class Statistics(ABC):
    def __init__(self, maxlen: int, send_each: int = 10):
        self.step: int = 0
        self.maxlen = maxlen
        self.send_each = send_each

    @abstractmethod
    def update(self, entry: Entry, send: bool = False):
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError


class RLStatistics(Statistics):
    def update(self, entry: RLEntry, send: bool = False):
        self.entries.append(entry)
        if send:
            logger.info(str(entry))

    def to_dict(self, step: int = -1) -> Dict:
        entry: RLEntry = self.entries[step]
        return {
            "step": self.step,
        }

    def __init__(self, maxlen: int = 2000, send_each: int = 10):
        super().__init__(maxlen, send_each)
        self.entries: Deque[RLEntry] = deque(maxlen=maxlen)
