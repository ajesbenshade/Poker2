import math
from collections import defaultdict
from dataclasses import dataclass
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from config import Config
from datatypes import Action, Infoset
from environment import clear_runtime_caches, get_memory_snapshot
from game import simulate_action_batch, simulate_equity_batch
from models import ActorCriticModel, CustomBeta
from storage import Float16ReservoirBuffer
from utils import adapt_batch_and_sims, select_amp_dtype


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    raise_fracs: torch.Tensor
    rewards: torch.Tensor


class PrioritizedReplayBackend:
    def __init__(self):
        self._lmdb = None
        self._env = None
        self._size = 0
        self._capacity = Config.REPLAY_BUFFER_SIZE
        self._fallback = Float16ReservoirBuffer(
            storage_dir=Config.STORAGE_DIR,
            name="ppo_prioritized",
            capacity=Config.REPLAY_BUFFER_SIZE,
            feature_dim=Config.STATE_DIM,
            target_dim=4,
            create=True,
        )

        try:
            import lmdb

            self._lmdb = lmdb
            self._env = lmdb.open(
                f"{Config.STORAGE_DIR}/ppo_prioritized.lmdb",
                map_size=16 * 1024 ** 3,
                max_dbs=1,
                subdir=False,
            )
        except Exception:
            self._lmdb = None
            self._env = None

    def add_batch(self, states_np, targets_np, priorities):
        if self._env is None:
            return self._fallback.add_batch(states_np, targets_np, priorities)

        inserted = 0
        with self._env.begin(write=True) as txn:
            for row in range(states_np.shape[0]):
                key = f"{self._size % self._capacity:08d}".encode("ascii")
                payload = {
                    "state": np.asarray(states_np[row], dtype=np.float16),
                    "target": np.asarray(targets_np[row], dtype=np.float16),
                    "priority": float(priorities[row]),
                }
                txn.put(key, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                self._size += 1
                inserted += 1
        return inserted

    def state_dict(self):
        if self._env is None:
            return {"backend": "memmap", "buffer": self._fallback.state_dict()}
        return {"backend": "lmdb", "size": self._size, "capacity": self._capacity}


class OpponentModel:
    def __init__(self):
        self.action_hist = defaultdict(lambda: np.ones(Config.NUM_ACTIONS, dtype=np.float32) / Config.NUM_ACTIONS)
        self.decay = 0.95

    def update(self, key, action):
        hist = self.action_hist[key]
        hist *= self.decay
        hist[action] += (1.0 - self.decay)
        total = hist.sum()
        if total > 0:
            hist /= total

    def aggression(self, key):
        hist = self.action_hist[key]
        return float(hist[Action.RAISE.value])


class EloTracker:
    def __init__(self, init_rating=1500.0, k_factor=24.0):
        self.rating = init_rating
        self.k_factor = k_factor

    def update_vs_baseline(self, win_rate, baseline_rating=1500.0):
        expected = 1.0 / (1.0 + 10.0 ** ((baseline_rating - self.rating) / 400.0))
        self.rating = self.rating + self.k_factor * (win_rate - expected)
        return self.rating


class PopulationManager:
    def __init__(self, base_state_dict):
        self.population = [base_state_dict]
        for _ in range(Config.POPULATION_SIZE - 1):
            self.population.append(base_state_dict)

    def evolve(self, score_table):
        if len(score_table) < 2:
            return
        sorted_agents = sorted(score_table.items(), key=lambda kv: kv[1], reverse=True)
        best_idx = sorted_agents[0][0]
        worst_idx = sorted_agents[-1][0]
        best_state = self.population[best_idx]
        mutated = {}
        for name, tensor in best_state.items():
            std = tensor.float().std(unbiased=False) if tensor.numel() > 1 else torch.tensor(1.0, device=tensor.device)
            noise = torch.randn_like(tensor) * Config.PBT_MUTATION_SCALE * std.clamp(min=1e-6)
            mutated[name] = (tensor + noise).detach().clone()
        self.population[worst_idx] = mutated


class ActorCriticAgent:
    def __init__(self, writer=None):
        self.device = Config.DEVICE
        self.writer = writer
        self.model = ActorCriticModel(
            state_dim=Config.STATE_DIM,
            num_actions=Config.NUM_ACTIONS,
            hidden_dim=Config.MODEL_HIDDEN_DIM,
            depth=Config.MODEL_DEPTH,
            dropout=Config.MODEL_DROPOUT,
            use_checkpointing=Config.GRADIENT_CHECKPOINTING,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.amp_dtype = select_amp_dtype(Config)
        self.scaler = GradScaler(device="cuda", enabled=Config.AMP_ENABLED and self.device == "cuda")

        self.replay = PrioritizedReplayBackend()

        self.current_batch_size = Config.BATCH_SIZE
        self.current_simulations = Config.NUM_SIMULATIONS
        self.iteration = 0
        self.num_opponents = Config.START_OPPONENTS
        self.opponent_model = OpponentModel()
        self.elo_tracker = EloTracker(k_factor=Config.ELO_K_FACTOR)
        self.population = PopulationManager(self.model.state_dict())

    def _state_from_infoset(self, infoset):
        state = np.zeros(Config.STATE_DIM, dtype=np.float32)
        bucket_id = infoset.key[0]
        hist_hash = infoset.key[1]
        state[bucket_id % Config.STATE_DIM] = 1.0
        state[(bucket_id * 7) % Config.STATE_DIM] = len(infoset.history) / 8.0
        state[(abs(hist_hash) % Config.STATE_DIM)] += 0.1
        return state

    def _sample_infosets(self, count):
        infosets = []
        for idx in range(count):
            history_len = np.random.randint(0, 4)
            history = tuple(np.random.randint(0, Config.NUM_ACTIONS, size=history_len).tolist())
            infosets.append(Infoset(bucket_id=idx % 4096, history=history))
        return infosets

    def _preflop_raise_boost(self, infoset):
        if not Config.PREFLOP_CHART_ENABLED:
            return 0.0
        is_preflop = len(infoset.history) == 0
        if not is_preflop:
            return 0.0
        strength = (infoset.key[0] % 100) / 100.0
        return 0.12 if strength > 0.75 else 0.0

    def _estimate_equity(self, infosets):
        # Keep equity estimation stable and fast for PPO action selection.
        equities = []
        for infoset in infosets:
            base_strength = (infoset.key[0] % 100) / 100.0
            aggression_penalty = min(0.15, self.opponent_model.aggression(infoset.key) * 0.1)
            noise = np.random.normal(0.0, 0.06)
            equities.append(float(np.clip(base_strength - aggression_penalty + noise, 0.0, 1.0)))
        return np.asarray(equities, dtype=np.float32)

    def _run_mcts(self, infoset):
        # Lightweight MCTS proxy: evaluate all actions with short stochastic rollouts.
        expanded = [infoset] * Config.NUM_ACTIONS
        actions = [Action.FOLD, Action.CALL, Action.RAISE]
        values = simulate_action_batch(expanded, actions)
        return int(np.argmax(values))

    def choose_action(self, infosets, deterministic=False):
        states_np = np.stack([self._state_from_infoset(i) for i in infosets], axis=0)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=Config.AMP_ENABLED and self.device == "cuda"):
            logits, values, alpha, beta = self.model(states)

        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)

        beta_dist = CustomBeta(alpha.float(), beta.float())
        raise_fracs = beta_dist.mode() if deterministic else beta_dist.sample()

        equities = self._estimate_equity(infosets)
        mcts_count = 0
        for idx, infoset in enumerate(infosets):
            raise_boost = self._preflop_raise_boost(infoset)
            probs[idx, Action.RAISE.value] = torch.clamp(probs[idx, Action.RAISE.value] + raise_boost, max=0.98)
            probs[idx] = probs[idx] / probs[idx].sum()

            use_mcts = (
                equities[idx] > Config.MCTS_EQUITY_THRESHOLD
                and np.random.random() < Config.MCTS_TRIGGER_PROB
                and not deterministic
            )
            if use_mcts:
                actions[idx] = self._run_mcts(infoset)
                mcts_count += 1

            if actions[idx].item() == Action.RAISE.value:
                raise_fracs[idx] = torch.clamp(raise_fracs[idx], min=0.15, max=0.95)

        return actions, log_probs, values.float(), raise_fracs.float(), mcts_count / max(1, len(infosets))

    def _compute_gae(self, rewards, values):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        next_adv = torch.tensor(0.0, device=rewards.device)
        next_value = torch.tensor(0.0, device=rewards.device)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + Config.GAMMA * next_value - values[t]
            next_adv = delta + Config.GAMMA * Config.GAE_LAMBDA * next_adv
            advantages[t] = next_adv
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        std = advantages.std().clamp(min=1e-6)
        advantages = (advantages - advantages.mean()) / std
        return returns, advantages

    def _simulate_rewards(self, infosets, actions, raise_fracs):
        action_enums = [Action(int(a.item())) for a in actions]
        utilities = simulate_action_batch(infosets, action_enums)
        utilities = torch.as_tensor(utilities, dtype=torch.float32, device=self.device)

        raise_mask = (actions == Action.RAISE.value).float()
        utilities = utilities + raise_mask * (raise_fracs - 0.5) * 0.3
        return torch.clamp(utilities, min=-4.0, max=4.0)

    def collect_rollout(self):
        infosets = self._sample_infosets(Config.ROLLOUT_STEPS)
        actions, log_probs, values, raise_fracs, mcts_rate = self.choose_action(infosets, deterministic=False)
        rewards = self._simulate_rewards(infosets, actions, raise_fracs)
        returns, advantages = self._compute_gae(rewards, values.detach())
        states_np = np.stack([self._state_from_infoset(i) for i in infosets], axis=0)

        for infoset, action in zip(infosets, actions.tolist()):
            self.opponent_model.update(infoset.key, action)

        # Store compact targets for prioritized replay (auxiliary use).
        replay_targets = torch.stack(
            [actions.float(), rewards, returns, advantages],
            dim=1,
        ).detach().cpu().numpy()
        priorities = np.abs(advantages.detach().cpu().numpy()) + 1e-3
        self.replay.add_batch(states_np, replay_targets, priorities)

        return RolloutBatch(
            states=torch.as_tensor(states_np, dtype=torch.float32, device=self.device),
            actions=actions,
            old_log_probs=log_probs.detach(),
            returns=returns.detach(),
            advantages=advantages.detach(),
            raise_fracs=raise_fracs.detach(),
            rewards=rewards.detach(),
        ), mcts_rate

    def _ppo_update(self, rollout):
        num_samples = rollout.states.shape[0]
        mini_size = max(1, num_samples // Config.PPO_MINIBATCHES)
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(Config.PPO_EPOCHS):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mini_size):
                idx = permutation[start:start + mini_size]
                states = rollout.states[idx]
                actions = rollout.actions[idx]
                old_log_probs = rollout.old_log_probs[idx]
                returns = rollout.returns[idx]
                advantages = rollout.advantages[idx]
                raise_fracs = rollout.raise_fracs[idx]

                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=Config.AMP_ENABLED and self.device == "cuda"):
                    logits, values, alpha, beta = self.model(states)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    beta_dist = CustomBeta(alpha.float(), beta.float())
                    raise_log_prob = beta_dist.log_prob(raise_fracs).mean()

                    ratio = torch.exp(log_probs - old_log_probs)
                    clipped = torch.clamp(ratio, 1.0 - Config.CLIP_EPS, 1.0 + Config.CLIP_EPS)
                    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                    policy_loss = policy_loss - 0.02 * raise_log_prob
                    value_loss = F.mse_loss(values.float(), returns.float())
                    loss = policy_loss + Config.VALUE_COEF * value_loss - Config.ENTROPY_COEF * entropy

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                losses.append(float(loss.detach().item()))
                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def _validate_vs_baseline(self):
        infosets = self._sample_infosets(512)
        actions, _, _, _, _ = self.choose_action(infosets, deterministic=True)
        strengths = np.array([(i.key[0] % 100) / 100.0 for i in infosets], dtype=np.float32)
        fold_bad = np.logical_and(actions.cpu().numpy() == Action.FOLD.value, strengths > 0.65).mean()
        raise_good = np.logical_and(actions.cpu().numpy() == Action.RAISE.value, strengths > 0.55).mean()
        win_rate = float(np.clip(raise_good - fold_bad + 0.5, 0.0, 1.0))
        rating = self.elo_tracker.update_vs_baseline(win_rate)
        return win_rate, rating

    def maybe_update_curriculum(self):
        progressed = self.iteration // Config.CURRICULUM_INTERVAL
        self.num_opponents = min(Config.TARGET_OPPONENTS, Config.START_OPPONENTS + progressed)
        Config.NUM_OPPONENTS = self.num_opponents

    def maybe_evolve_population(self, score):
        score_table = {idx: score - idx * 0.001 for idx in range(Config.POPULATION_SIZE)}
        self.population.evolve(score_table)

    def save_checkpoint(self, iteration, metrics, is_best=False):
        import os

        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        payload = {
            "mode": "ppo",
            "iteration": iteration,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "batch_size": self.current_batch_size,
            "simulations": self.current_simulations,
            "num_opponents": self.num_opponents,
            "elo": self.elo_tracker.rating,
            "replay_state": self.replay.state_dict(),
        }
        latest_path = os.path.join(Config.CHECKPOINT_DIR, Config.LATEST_CHECKPOINT_NAME)
        torch.save(payload, latest_path)
        if is_best:
            best_path = os.path.join(Config.CHECKPOINT_DIR, Config.BEST_CHECKPOINT_NAME)
            torch.save(payload, best_path)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if state.get("mode") != "ppo":
            raise ValueError("Checkpoint is not a PPO checkpoint")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.current_batch_size = int(state.get("batch_size", self.current_batch_size))
        self.current_simulations = int(state.get("simulations", self.current_simulations))
        self.num_opponents = int(state.get("num_opponents", self.num_opponents))
        self.elo_tracker.rating = float(state.get("elo", self.elo_tracker.rating))
        self.iteration = int(state.get("iteration", 0))
        return self.iteration

    def train_iteration(self):
        self.maybe_update_curriculum()

        adaptation = adapt_batch_and_sims(Config, self.current_batch_size, self.current_simulations)
        self.current_batch_size = adaptation.batch_size
        self.current_simulations = adaptation.simulations
        Config.NUM_SIMULATIONS = self.current_simulations

        rollout, mcts_rate = self.collect_rollout()
        metrics = self._ppo_update(rollout)
        avg_reward = float(rollout.rewards.mean().item())

        if self.iteration % Config.VALIDATION_INTERVAL == 0:
            win_rate, elo = self._validate_vs_baseline()
            self.maybe_evolve_population(elo)
        else:
            win_rate, elo = 0.0, self.elo_tracker.rating

        snapshot = get_memory_snapshot()
        metrics.update(
            {
                "avg_reward": avg_reward,
                "elo": elo,
                "win_rate": win_rate,
                "ram_pct": snapshot["ram_pct"],
                "vram_used_gb": snapshot["used_gb"],
                "vram_pct": snapshot["used_pct"],
                "batch_size": self.current_batch_size,
                "simulations": self.current_simulations,
                "num_opponents": self.num_opponents,
                "mcts_rate": mcts_rate,
                "backoff": adaptation.reason,
            }
        )

        if self.iteration % Config.CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint(self.iteration, metrics, is_best=(elo >= self.elo_tracker.rating))
            clear_runtime_caches()

        if self.writer is not None and self.iteration % Config.LOG_INTERVAL == 0:
            self.writer.add_scalar("PPO/Loss", metrics["loss"], self.iteration)
            self.writer.add_scalar("PPO/PolicyLoss", metrics["policy_loss"], self.iteration)
            self.writer.add_scalar("PPO/ValueLoss", metrics["value_loss"], self.iteration)
            self.writer.add_scalar("PPO/Entropy", metrics["entropy"], self.iteration)
            self.writer.add_scalar("PPO/AvgReward", metrics["avg_reward"], self.iteration)
            self.writer.add_scalar("PPO/Elo", metrics["elo"], self.iteration)
            self.writer.add_scalar("System/RAMPct", metrics["ram_pct"], self.iteration)
            self.writer.add_scalar("System/VRAMUsedGB", metrics["vram_used_gb"], self.iteration)
            self.writer.add_scalar("System/VRAMPct", metrics["vram_pct"], self.iteration)
            self.writer.add_scalar("PPO/MCTSRate", metrics["mcts_rate"], self.iteration)

        self.iteration += 1
        return metrics
