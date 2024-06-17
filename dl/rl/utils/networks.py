from typing import Dict, TypeVar, Type
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class BaseAlgorithm:
    pass


@dataclass
class DQN(BaseAlgorithm):
    policy_net: nn.Module
    target_net: nn.Module


@dataclass
class TD3(BaseAlgorithm):
    actor: nn.Module
    critic1: nn.Module
    critic2: nn.Module

    target_actor: nn.Module
    target_critic1: nn.Module
    target_critic2: nn.Module


class AlgorithmFactory(object):
    DQN_alg = "DQN"
    DDPG_alg = "DDPG"
    TD3_alg = "TD3"
    MAP = {
        DQN_alg: DQN,
        TD3_alg: TD3,
    }

    @staticmethod
    def load(algorithm: str, mapping: Dict[str, nn.Module]) -> BaseAlgorithm:
        return AlgorithmFactory.MAP[algorithm](**mapping)
