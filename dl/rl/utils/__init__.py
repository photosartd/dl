import torch
import torch.nn as nn


TAU = 0.005


def soft_update(target: nn.Module, source: nn.Module, tau: float = TAU):
    """Soft update of target network parameters"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)
