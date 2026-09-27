"""Native T2 actor for post-transfer fine-tune (31-DoF)."""

from __future__ import annotations

import torch
import torch.nn as nn

from g1_to_t2.loader import _mlp


class T2StabilityActor(nn.Module):
    """Simple MLP: proprio obs -> 31 joint actions."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 31,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 256]
        self.actor = _mlp(obs_dim, hidden_dims, num_actions, activation)
        self.critic = _mlp(obs_dim, hidden_dims, 1, activation)
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        self.num_actions = num_actions
        self.obs_dim = obs_dim

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def act(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
