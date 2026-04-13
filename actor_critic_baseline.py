"""
Minimal actor-critic + PPO baseline for poker on AMD ROCm (7900 XT safe).
Validates single-agent RL on your hardware before hybrid integration with Deep CFR.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
from enum import Enum

os.environ.setdefault('OMP_NUM_THREADS', '1')

from environment import setup_rocmo
setup_rocmo()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class Suit(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2


class Config:
    # Hardware-specific (AMD ROCm safe)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    AMP_ENABLED = False

    # Training hyperparams
    LR = 1e-4
    GAMMA = 0.99
    BATCH_SIZE = 4096
    NUM_TRAINING_STEPS = 8
    T_MAX = 1000

    # Game & simulation
    INITIAL_STACK = 1000
    SMALL_BLIND = 10
    BIG_BLIND = 20
    NUM_HANDS = 200
    NUM_SIMULATIONS = 64
    VALIDATION_GAMES = 20
    VALIDATION_INTERVAL = 50

    MAX_OPPONENTS = 5
    SEED = 42
    MCTS_SIM_DEPTH = 20

    REPLAY_BUFFER_CAPACITY = 100_000

    STATE_SIZE = 169
    ACTION_SIZE = 3
    ACTOR_HIDDEN_SIZE = 1024
    CRITIC_HIDDEN_SIZE = 1024
    NUM_RES_BLOCKS = 2
    RESIDUAL_DROPOUT = 0.1

    OPPONENT_DECAY = 0.95
    EXPLORATION_FACTOR = 0.2
    EXPLORATION_DECAY = 0.995
    MIN_EXPLORATION = 0.05


class EquityNet(nn.Module):
    """Lightweight equity evaluator."""
    def __init__(self, input_dim=106, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        outputs = self.norm(inputs)
        outputs = self.activation(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return residual + outputs


class ActorCritic(nn.Module):
    """Actor-Critic network for poker."""
    def __init__(self, state_size, action_size, hidden_size, num_blocks, dropout):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared input projection
        self.input_proj = nn.Linear(state_size, hidden_size)

        # Shared residual blocks
        self.shared_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)]
        )

        # Actor head
        self.actor_norm = nn.LayerNorm(hidden_size)
        self.actor_head = nn.Linear(hidden_size, action_size)

        # Critic head
        self.critic_norm = nn.LayerNorm(hidden_size)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Return action logits and state value."""
        x = self.input_proj(state)
        for block in self.shared_blocks:
            x = block(x)

        action_logits = self.actor_head(self.actor_norm(x))
        value = self.critic_head(self.critic_norm(x))

        return action_logits, value

    def get_action_and_value(self, state):
        """Sample action and compute log prob + value."""
        action_logits, value = self(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)


class SimplePokerEnv:
    """Minimal poker environment for testing."""
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        """Reset environment to new hand."""
        self.state = np.zeros(self.config.STATE_SIZE, dtype=np.float32)
        self.pot = self.config.SMALL_BLIND + self.config.BIG_BLIND
        self.stack = self.config.INITIAL_STACK - self.config.BIG_BLIND
        self.done = False
        self.reward = 0.0
        return self.state

    def step(self, action):
        """Take one action and return next state, reward, done."""
        if self.done:
            return self.state, 0.0, True

        # Minimal game logic
        if action == Action.FOLD.value:
            self.reward = -self.config.BIG_BLIND
            self.done = True
        elif action == Action.CALL.value:
            self.stack -= self.config.BIG_BLIND
            self.pot += self.config.BIG_BLIND
            if self.stack <= 0:
                self.done = True
        elif action == Action.RAISE.value:
            raise_amount = min(self.stack, self.config.BIG_BLIND * 3)
            self.stack -= raise_amount
            self.pot += raise_amount
            if self.stack <= 0:
                self.done = True

        self.state = np.random.randn(self.config.STATE_SIZE).astype(np.float32)
        return self.state, float(self.reward), self.done


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer (without log_prob/value which need recomputation)."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class ActorCriticTrainer:
    """Trainer for actor-critic baseline."""
    def __init__(self, config):
        self.config = config
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)

        self.network = ActorCritic(
            config.STATE_SIZE,
            config.ACTION_SIZE,
            config.ACTOR_HIDDEN_SIZE,
            config.NUM_RES_BLOCKS,
            config.RESIDUAL_DROPOUT,
        ).to(config.DEVICE)

        self.optimizer = optim.AdamW(self.network.parameters(), lr=config.LR, weight_decay=1e-4)
        self.env = SimplePokerEnv(config)
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY, config.DEVICE)

        self.total_hands = 0
        self.total_reward = 0.0
        self.validation_reward = 0.0

    def train_step(self, batch):
        """Train on a batch of experiences (recompute actions and values)."""
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.config.DEVICE)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.config.DEVICE)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.config.DEVICE)

        # Recompute log probs and values for gradient flow
        action_logits, values = self.network(states_tensor)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions_tensor)
        values = values.squeeze(-1)

        # Compute returns (simple 1-step for now)
        returns = rewards_tensor + self.config.GAMMA * (1 - torch.as_tensor(dones, dtype=torch.float32, device=self.config.DEVICE))

        # Actor loss
        advantages = returns.detach() - values.detach()
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = ((returns - values) ** 2).mean()

        # Total loss
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
        self.optimizer.step()

        return float(loss.item()), float(actor_loss.item()), float(critic_loss.item())

    def collect_experience(self):
        """Collect one hand of experience."""
        state = self.env.reset()
        total_reward = 0.0

        for _ in range(self.config.NUM_TRAINING_STEPS):
            next_state, reward, done = self.env.step(np.random.randint(0, self.config.ACTION_SIZE))
            self.replay_buffer.add(state, np.random.randint(0, self.config.ACTION_SIZE), reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        self.total_hands += 1
        self.total_reward += total_reward
        return total_reward

    def train(self):
        """Main training loop."""
        logger.info(
            'Starting Actor-Critic baseline | hands=%s | batch=%s | device=%s | dtype=%s',
            self.config.NUM_HANDS,
            self.config.BATCH_SIZE,
            self.config.DEVICE,
            self.config.DTYPE,
        )

        for hand in range(self.config.NUM_HANDS):
            self.collect_experience()

            # Train on batch
            batch = self.replay_buffer.sample(min(self.config.BATCH_SIZE, len(self.replay_buffer.buffer)))
            if batch:
                loss, actor_loss, critic_loss = self.train_step(batch)

                if (hand + 1) % self.config.VALIDATION_INTERVAL == 0:
                    avg_reward = self.total_reward / max(1, self.total_hands)
                    logger.info(
                        'Hand %s | avg_reward %.4f | loss %.4f | actor_loss %.4f | critic_loss %.4f | buffer %s',
                        hand + 1,
                        avg_reward,
                        loss,
                        actor_loss,
                        critic_loss,
                        len(self.replay_buffer.buffer),
                    )

        logger.info('Training complete | total_hands=%s | final_avg_reward=%.4f', self.total_hands, self.total_reward / max(1, self.total_hands))


def main():
    config = Config()
    trainer = ActorCriticTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
