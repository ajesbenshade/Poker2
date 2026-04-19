"""Networks for Deep CFR.

Both the per-player advantage network V_p and the shared average-policy
network \u03a0 use the same trunk: an MLP with residual blocks taking the dense
observation from :mod:`engine.encoder`. The legality mask is concatenated to
the observation so the network can learn to ignore illegal actions even
though the loss explicitly masks them.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class _Trunk(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int,
                 num_blocks: int, dropout: float):
        super().__init__()
        in_dim = obs_dim + num_actions   # observation + legality mask
        self.input = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden, dropout)
                                     for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, obs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, legal], dim=-1)
        x = self.input(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        return self.head(x)


class AdvantageNet(nn.Module):
    """Predicts per-action regret (counterfactual advantage)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256,
                 num_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.trunk = _Trunk(obs_dim, num_actions, hidden, num_blocks, dropout)

    def forward(self, obs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs, legal)


class PolicyNet(nn.Module):
    """Predicts the average strategy. Output is logits; apply softmax with mask."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256,
                 num_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.trunk = _Trunk(obs_dim, num_actions, hidden, num_blocks, dropout)

    def forward(self, obs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs, legal)

    def strategy(self, obs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs, legal)
        # Mask illegal actions to -inf before softmax.
        masked = logits.masked_fill(legal < 0.5, float("-inf"))
        probs = F.softmax(masked, dim=-1)
        # Guard against rows with zero legal actions (shouldn't happen, but safe).
        probs = torch.nan_to_num(probs, nan=0.0)
        return probs
