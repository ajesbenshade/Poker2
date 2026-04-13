import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        outputs = self.norm(inputs)
        outputs = self.act(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return residual + outputs


class CustomBeta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self._dist = torch.distributions.Beta(alpha, beta)

    def sample(self):
        return self._dist.rsample()

    def mode(self):
        denom = torch.clamp(self.alpha + self.beta - 2.0, min=1e-6)
        return torch.clamp((self.alpha - 1.0) / denom, min=0.0, max=1.0)

    def log_prob(self, value):
        value = torch.clamp(value, 1e-6, 1.0 - 1e-6)
        return self._dist.log_prob(value)


class ActorCriticModel(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, depth, dropout=0.0, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(depth)])

        self.actor_norm = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, num_actions)

        self.raise_norm = nn.LayerNorm(hidden_dim)
        self.raise_alpha = nn.Linear(hidden_dim, 1)
        self.raise_beta = nn.Linear(hidden_dim, 1)

        self.critic_norm = nn.LayerNorm(hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _backbone(self, states):
        x = self.input_proj(states)
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, states):
        x = self._backbone(states)
        logits = self.actor_head(self.actor_norm(x))
        values = self.critic_head(self.critic_norm(x)).squeeze(-1)

        raise_hidden = self.raise_norm(x)
        alpha = torch.nn.functional.softplus(self.raise_alpha(raise_hidden)).squeeze(-1) + 1.1
        beta = torch.nn.functional.softplus(self.raise_beta(raise_hidden)).squeeze(-1) + 1.1
        return logits, values, alpha, beta
