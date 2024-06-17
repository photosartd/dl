import numpy as np
import torch
from torch.optim import Adam

from dl.rl.agents.base import BaseAgent
from dl.rl.buffer import Transition, ReplayBuffer
from dl.rl.utils.config import Config, TD3Config
from dl.rl.utils.networks import TD3 as TD3Algorithm, AlgorithmFactory
from dl.rl.utils import soft_update


class TD3(BaseAgent):
    def __init__(self, config: Config):
        super().__init__(config)
        # TODO stupid, find a way to do it better
        if isinstance(self.networks, TD3Algorithm):
            self.networks: TD3Algorithm = self.networks
        if isinstance(self.config, TD3Config):
            self.config: TD3Config = self.config

        self.optim_actor = Adam(self.networks.actor.parameters(), lr=self.config.actor_lr)
        self.optim_critic1 = Adam(self.networks.critic1.parameters(), lr=self.config.critic_lr)
        self.optim_critic2 = Adam(self.networks.critic2.parameters(), lr=self.config.critic_lr)

    def act(self, state) -> np.ndarray:
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.config.device)
            return self.networks.actor(state).cpu().numpy()[0]

    def update(self, transition: Transition) -> None:
        self.replay_buffer.add(transition)
        if len(self.replay_buffer) < self.config.min_batch_sampling:
            return
        # Sampling
        states, actions, rewards, next_states, dones = ReplayBuffer.to_torch(
            self.replay_buffer.sample(
                self.config.batch_size
            ),
            self.config.device
        )
        # Update Critic
        with torch.no_grad():
            next_actions = (self.networks.target_actor(next_states) + self.noise(actions.shape)).clamp(-1.0, 1.0)
            q_target_1 = self.networks.target_critic1(next_states, next_actions)
            q_target_2 = self.networks.target_critic2(next_states, next_actions)
            q_target = torch.min(q_target_1, q_target_2)
            q_target = rewards + (1 - dones) * self.config.gamma * q_target

        # Get current Q-Value estimates
        q_1 = self.networks.critic1(states, actions)
        q_2 = self.networks.critic2(states, actions)

        # Compute critic loss
        critic_loss = self.config.critic_loss(q_1, q_target) + self.config.critic_loss(q_2, q_target)
        self.optim_critic1.zero_grad()
        self.optim_critic2.zero_grad()
        critic_loss.backward()
        self.optim_critic1.step()
        self.optim_critic2.step()

        if self.statistics.step % 2 == 0:
            # Update actor
            actor_loss = -self.networks.critic1(states, self.networks.actor(states)).mean()
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

            soft_update(self.networks.target_critic1, self.networks.critic1, tau=self.config.tau)
            soft_update(self.networks.target_critic2, self.networks.critic2, tau=self.config.tau)
            soft_update(self.networks.target_actor, self.networks.actor, tau=self.config.tau)

    def noise(self, shape: torch.Size) -> torch.Tensor:
        return ((torch.randn(shape) * self.config.noise)
                .clamp(-self.config.noise_clip, self.config.noise_clip)
                .to(self.config.device))
