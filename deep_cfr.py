import logging
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

from environment import clear_runtime_caches, get_memory_snapshot, setup_rocmo

setup_rocmo()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')

from config import Config
from datatypes import Action, Infoset
from game import simulate_action_batch, terminal
from storage import Float16ReservoirBuffer


logger = logging.getLogger(__name__)


def resolve_checkpoint_path(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    candidate = os.path.join(Config.CHECKPOINT_DIR, checkpoint_path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')


def _sanitize_numpy(array, clamp_value=None, replacement=0.0):
    array = np.nan_to_num(array, nan=replacement, posinf=replacement, neginf=replacement)
    if clamp_value is not None:
        array = np.clip(array, -clamp_value, clamp_value)
    return array.astype(np.float32, copy=False)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, inputs):
        residual = inputs
        outputs = self.norm(inputs)
        outputs = self.activation(self.fc1(outputs))
        outputs = self.fc2(outputs)
        return residual + outputs


class DeepCFRNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(Config.MODEL_DEPTH)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        outputs = self.input_proj(inputs)
        for block in self.blocks:
            if Config.GRADIENT_CHECKPOINTING and self.training:
                outputs = checkpoint(block, outputs, use_reentrant=False)
            else:
                outputs = block(outputs)
        outputs = self.output_norm(outputs)
        return self.output_head(outputs)


class DeepCFRAgent:
    def __init__(self, infosets, bucket_centroids, writer=None, resume=False):
        self.device = Config.DEVICE
        self.writer = writer
        self.infosets = list(infosets)
        self.bucket_centroids = np.asarray(bucket_centroids, dtype=np.float32)
        self.autocast_device = 'cuda' if self.device == 'cuda' else 'cpu'
        self.amp_enabled = self.device == 'cuda'
        self.current_batch_size = Config.BATCH_SIZE
        self.current_num_traversals = Config.NUM_TRAVERSALS
        self.current_rollouts = Config.EQUITY_ROLLOUTS
        self.backoff_events = 0
        self.recovery_events = 0
        self.skipped_batches = 0
        self.sanitization_events = 0
        self.last_backoff_iteration = -Config.BACKOFF_COOLDOWN
        self.last_backoff_reason = 'none'
        self.best_score = float('-inf')
        self.iteration = 0
        self._rng = np.random.default_rng()

        self.advantage_net = DeepCFRNetwork(
            input_dim=Config.DEEP_CFR_FEATURE_DIM,
            hidden_dim=Config.MODEL_HIDDEN_DIM,
            output_dim=Config.NUM_ACTIONS,
        ).to(self.device)
        self.strategy_net = DeepCFRNetwork(
            input_dim=Config.DEEP_CFR_FEATURE_DIM,
            hidden_dim=Config.MODEL_HIDDEN_DIM,
            output_dim=Config.NUM_ACTIONS,
        ).to(self.device)
        self.advantage_optimizer = torch.optim.AdamW(self.advantage_net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.strategy_optimizer = torch.optim.AdamW(self.strategy_net.parameters(), lr=1e-4, weight_decay=1e-4)

        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.STORAGE_DIR, exist_ok=True)
        self.advantage_buffer = Float16ReservoirBuffer(
            storage_dir=os.path.join(Config.STORAGE_DIR, 'advantage_buffer'),
            name='advantage',
            capacity=Config.REPLAY_BUFFER_SIZE,
            feature_dim=Config.DEEP_CFR_FEATURE_DIM,
            target_dim=Config.NUM_ACTIONS,
            create=not resume,
        )
        self.strategy_buffer = Float16ReservoirBuffer(
            storage_dir=os.path.join(Config.STORAGE_DIR, 'strategy_buffer'),
            name='strategy',
            capacity=Config.REPLAY_BUFFER_SIZE,
            feature_dim=Config.DEEP_CFR_FEATURE_DIM,
            target_dim=Config.NUM_ACTIONS,
            create=not resume,
        )

    def _sanitize_tensor(self, tensor, name, clamp_value=Config.LOSS_CLAMP, replacement=0.0):
        invalid_mask = ~torch.isfinite(tensor)
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum().item())
            self.sanitization_events += invalid_count
            logger.warning('Sanitizing %s invalid values in %s at iteration %s', invalid_count, name, self.iteration)
        tensor = torch.nan_to_num(tensor, nan=replacement, posinf=replacement, neginf=replacement)
        return torch.clamp(tensor, min=-clamp_value, max=clamp_value)

    def _encode_infosets(self, infosets):
        features = np.zeros((len(infosets), Config.DEEP_CFR_FEATURE_DIM), dtype=np.float32)
        num_centroids = max(1, len(self.bucket_centroids) - 1)
        for index, infoset in enumerate(infosets):
            bucket_id = int(infoset.key[0])
            history = infoset.history
            history_len = len(history)
            if 0 <= bucket_id < len(self.bucket_centroids):
                features[index, :5] = self.bucket_centroids[bucket_id]

            counts = np.bincount(history, minlength=Config.NUM_ACTIONS) if history else np.zeros(Config.NUM_ACTIONS, dtype=np.float32)
            features[index, 5] = bucket_id / num_centroids
            features[index, 6] = history_len / max(1, Config.MAX_CFR_DEPTH)
            features[index, 7:10] = counts / max(1, history_len)
            if history:
                last_action = int(history[-1])
                if 0 <= last_action < Config.NUM_ACTIONS:
                    features[index, 10 + last_action] = 1.0
                history_array = np.asarray(history, dtype=np.float32)
                features[index, 14] = history_array.mean() / max(1, Config.NUM_ACTIONS - 1)
                features[index, 15] = history_array.std() / max(1, Config.NUM_ACTIONS - 1)
            features[index, 13] = history_len % 2

        return _sanitize_numpy(features, clamp_value=4.0)

    def _normalize_strategy(self, strategy):
        strategy = self._sanitize_tensor(strategy, 'strategy', clamp_value=1.0, replacement=0.0)
        strategy = torch.clamp(strategy, min=0.0, max=1.0)
        denom = strategy.sum(dim=-1, keepdim=True)
        valid = torch.isfinite(denom) & (denom > 0)
        uniform = torch.full_like(strategy, 1.0 / Config.NUM_ACTIONS)
        normalized = torch.where(valid, strategy / torch.clamp(denom, min=1e-6), uniform)
        return self._sanitize_tensor(normalized, 'normalized_strategy', clamp_value=1.0, replacement=1.0 / Config.NUM_ACTIONS)

    def _regret_matching(self, advantages):
        positive_advantages = torch.clamp(self._sanitize_tensor(advantages, 'advantages', clamp_value=Config.LOSS_CLAMP), min=0.0)
        return self._normalize_strategy(positive_advantages)

    def _strategy_from_networks(self, features_tensor):
        with torch.no_grad():
            with torch.autocast(device_type=self.autocast_device, dtype=Config.AMP_DTYPE, enabled=self.amp_enabled):
                advantage_outputs = self.advantage_net(features_tensor)
                strategy_logits = self.strategy_net(features_tensor)

        advantage_outputs = self._sanitize_tensor(advantage_outputs, 'advantage_outputs')
        strategy_logits = self._sanitize_tensor(strategy_logits, 'strategy_logits')
        regret_strategy = self._regret_matching(advantage_outputs)
        learned_strategy = self._normalize_strategy(F.softmax(strategy_logits, dim=-1))
        mix_weight = 0.25 if len(self.strategy_buffer) < Config.REPLAY_WARMUP_SAMPLES else 0.5
        return self._normalize_strategy((1.0 - mix_weight) * regret_strategy + mix_weight * learned_strategy)

    def _sample_actions(self, strategy):
        strategy = self._normalize_strategy(strategy)
        return torch.multinomial(strategy, num_samples=1).squeeze(-1).cpu().numpy()

    def _compute_action_utilities(self, infosets):
        repeated_infosets = []
        repeated_actions = []
        actions = [Action(action_idx) for action_idx in range(Config.NUM_ACTIONS)]
        for infoset in infosets:
            repeated_infosets.extend([infoset] * Config.NUM_ACTIONS)
            repeated_actions.extend(actions)

        utilities = simulate_action_batch(repeated_infosets, repeated_actions)
        utilities = utilities.reshape(len(infosets), Config.NUM_ACTIONS)
        utilities = _sanitize_numpy(utilities, clamp_value=Config.UTILITY_CLAMP)
        return torch.as_tensor(utilities, dtype=torch.float32, device=self.device)

    def _maybe_backoff(self, snapshot, reason=None):
        if reason is None:
            if snapshot['used_gb'] <= Config.VRAM_SOFT_LIMIT_GB and snapshot['ram_pct'] <= Config.RAM_SOFT_LIMIT_PCT:
                if self.backoff_events <= self.recovery_events:
                    return None
                if self.iteration - self.last_backoff_iteration < Config.BACKOFF_COOLDOWN:
                    return None
                low_pressure = snapshot['used_gb'] < (Config.VRAM_SOFT_LIMIT_GB * 0.75) and snapshot['ram_pct'] < (Config.RAM_SOFT_LIMIT_PCT - 8.0)
                if not low_pressure:
                    return None
                new_batch_size = min(Config.MAX_BATCH_SIZE, self.current_batch_size * 2)
                new_traversals = min(Config.MAX_NUM_TRAVERSALS, self.current_num_traversals * 2)
                new_rollouts = min(64, self.current_rollouts * 2)
                if new_batch_size == self.current_batch_size and new_traversals == self.current_num_traversals and new_rollouts == self.current_rollouts:
                    return None
                self.current_batch_size = new_batch_size
                self.current_num_traversals = new_traversals
                self.current_rollouts = new_rollouts
                Config.BATCH_SIZE = self.current_batch_size
                Config.EQUITY_ROLLOUTS = self.current_rollouts
                self.recovery_events += 1
                self.last_backoff_reason = 'recovered'
                return 'recovered'
            reason = 'memory-pressure'

        previous_state = (self.current_batch_size, self.current_num_traversals, self.current_rollouts)
        self.current_batch_size = max(Config.MIN_BATCH_SIZE, self.current_batch_size // 2)
        self.current_num_traversals = max(Config.MIN_NUM_TRAVERSALS, self.current_num_traversals // 2)
        self.current_rollouts = max(Config.MIN_EQUITY_ROLLOUTS, self.current_rollouts // 2)
        Config.BATCH_SIZE = self.current_batch_size
        Config.EQUITY_ROLLOUTS = self.current_rollouts
        self.backoff_events += 1
        self.last_backoff_iteration = self.iteration
        self.last_backoff_reason = reason
        clear_runtime_caches()

        new_state = (self.current_batch_size, self.current_num_traversals, self.current_rollouts)
        if new_state == previous_state:
            return None
        return reason

    def _is_oom(self, runtime_error):
        message = str(runtime_error).lower()
        return 'out of memory' in message or 'hip error' in message or 'cuda error' in message

    def _train_network(self, network, optimizer, replay_buffer, steps, objective_name):
        if len(replay_buffer) < max(32, Config.REPLAY_WARMUP_SAMPLES):
            return None, None

        losses = []
        grad_norms = []
        for _ in range(steps):
            batch = replay_buffer.sample(self.current_batch_size, self.device)
            if batch is None:
                break

            features = self._sanitize_tensor(batch['features'], f'{objective_name}_features', clamp_value=4.0)
            targets = self._sanitize_tensor(batch['targets'], f'{objective_name}_targets')
            weights = torch.clamp(self._sanitize_tensor(batch['weights'], f'{objective_name}_weights'), min=0.0, max=8.0)
            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.autocast(device_type=self.autocast_device, dtype=Config.AMP_DTYPE, enabled=self.amp_enabled):
                    outputs = self._sanitize_tensor(network(features), f'{objective_name}_outputs')
                    if objective_name == 'advantage':
                        per_sample_loss = F.smooth_l1_loss(outputs, targets, reduction='none').mean(dim=-1)
                    else:
                        target_probs = self._normalize_strategy(targets)
                        log_probs = F.log_softmax(outputs, dim=-1)
                        per_sample_loss = -(target_probs * log_probs).sum(dim=-1)
                    loss = (per_sample_loss * weights).mean()

                if not torch.isfinite(loss):
                    self.skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), Config.MAX_GRAD_NORM)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    self.skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                losses.append(float(loss.detach().float().item()))
                grad_norms.append(float(torch.as_tensor(grad_norm).detach().float().item()))
            except RuntimeError as runtime_error:
                optimizer.zero_grad(set_to_none=True)
                if self._is_oom(runtime_error):
                    self._maybe_backoff(get_memory_snapshot(), reason=f'{objective_name}-oom')
                    continue
                raise

        if not losses:
            return None, None
        return float(np.mean(losses)), float(np.mean(grad_norms))

    def _sample_root_infosets(self):
        if not self.infosets:
            return []
        sample_indices = self._rng.integers(0, len(self.infosets), size=self.current_num_traversals)
        return [self.infosets[index] for index in sample_indices]

    def _run_traversals(self, max_depth):
        active_infosets = self._sample_root_infosets()
        if not active_infosets:
            return {
                'avg_utility': 0.0,
                'exploitability_proxy': 0.0,
                'advantage_examples': 0,
                'strategy_examples': 0,
            }

        utility_values = []
        exploitability_values = []
        advantage_examples = 0
        strategy_examples = 0

        for _ in range(max_depth):
            if not active_infosets:
                break

            encoded_features = self._encode_infosets(active_infosets)
            feature_tensor = torch.as_tensor(encoded_features, dtype=torch.float32, device=self.device)
            strategy = self._strategy_from_networks(feature_tensor)
            action_utilities = self._compute_action_utilities(active_infosets)
            expected_utility = (strategy * action_utilities).sum(dim=-1, keepdim=True)
            expected_utility = self._sanitize_tensor(expected_utility, 'expected_utility', clamp_value=Config.UTILITY_CLAMP)
            regrets = self._sanitize_tensor(action_utilities - expected_utility, 'regrets', clamp_value=Config.UTILITY_CLAMP)
            exploitability = torch.relu(action_utilities.max(dim=-1).values - expected_utility.squeeze(-1))
            exploitability = self._sanitize_tensor(exploitability, 'exploitability', clamp_value=Config.UTILITY_CLAMP)

            importance_weights = np.full(len(active_infosets), 1.0, dtype=np.float32)
            advantage_examples += self.advantage_buffer.add_batch(encoded_features, regrets.detach().cpu().numpy(), importance_weights)
            strategy_examples += self.strategy_buffer.add_batch(encoded_features, strategy.detach().cpu().numpy(), importance_weights)

            utility_values.extend(expected_utility.squeeze(-1).detach().cpu().numpy().tolist())
            exploitability_values.extend(exploitability.detach().cpu().numpy().tolist())

            chosen_actions = self._sample_actions(strategy)
            next_infosets = []
            for infoset, action_index in zip(active_infosets, chosen_actions):
                next_infoset = Infoset(int(infoset.key[0]), infoset.history + (int(action_index),))
                if terminal(next_infoset) or len(next_infoset.history) >= max_depth:
                    continue
                next_infosets.append(next_infoset)
            active_infosets = next_infosets

        avg_utility = float(np.mean(utility_values)) if utility_values else 0.0
        exploitability_proxy = float(np.mean(exploitability_values)) if exploitability_values else 0.0
        return {
            'avg_utility': avg_utility,
            'exploitability_proxy': exploitability_proxy,
            'advantage_examples': advantage_examples,
            'strategy_examples': strategy_examples,
        }

    def _log_metrics(self, iteration, metrics, snapshot):
        backoff_label = self.last_backoff_reason if self.iteration == self.last_backoff_iteration else 'none'
        log_message = (
            f'Iter {iteration} | avg_utility {metrics["avg_utility"]:.4f} | exploitability {metrics["exploitability_proxy"]:.4f} | '
            f'vram {snapshot["used_pct"]:.1f}% ({snapshot["used_gb"]:.2f}/{snapshot["total_gb"]:.2f} GB) | '
            f'ram {snapshot["ram_pct"]:.1f}% | batch {self.current_batch_size} | traversals {self.current_num_traversals} | '
            f'rollouts {self.current_rollouts} | adv_buffer {len(self.advantage_buffer):,} | strat_buffer {len(self.strategy_buffer):,} | '
            f'backoff {backoff_label}'
        )
        logger.info(log_message)

        if self.writer is None:
            return

        self.writer.add_scalar('DeepCFR/AvgUtility', metrics['avg_utility'], iteration)
        self.writer.add_scalar('DeepCFR/ExploitabilityProxy', metrics['exploitability_proxy'], iteration)
        if metrics.get('advantage_loss') is not None:
            self.writer.add_scalar('DeepCFR/AdvantageLoss', metrics['advantage_loss'], iteration)
        if metrics.get('strategy_loss') is not None:
            self.writer.add_scalar('DeepCFR/StrategyLoss', metrics['strategy_loss'], iteration)
        if metrics.get('advantage_grad_norm') is not None:
            self.writer.add_scalar('DeepCFR/AdvantageGradNorm', metrics['advantage_grad_norm'], iteration)
        if metrics.get('strategy_grad_norm') is not None:
            self.writer.add_scalar('DeepCFR/StrategyGradNorm', metrics['strategy_grad_norm'], iteration)
        self.writer.add_scalar('DeepCFR/BatchSize', self.current_batch_size, iteration)
        self.writer.add_scalar('DeepCFR/Traversals', self.current_num_traversals, iteration)
        self.writer.add_scalar('DeepCFR/BackoffEvents', self.backoff_events, iteration)
        self.writer.add_scalar('DeepCFR/SanitizationEvents', self.sanitization_events, iteration)
        self.writer.add_scalar('System/VRAMPct', snapshot['used_pct'], iteration)
        self.writer.add_scalar('System/RAMPct', snapshot['ram_pct'], iteration)

    def save_checkpoint(self, iteration, metrics, is_best=False):
        checkpoint = {
            'mode': 'deep',
            'iteration': iteration,
            'metrics': metrics,
            'best_score': self.best_score,
            'advantage_state_dict': self.advantage_net.state_dict(),
            'strategy_state_dict': self.strategy_net.state_dict(),
            'advantage_optimizer_state_dict': self.advantage_optimizer.state_dict(),
            'strategy_optimizer_state_dict': self.strategy_optimizer.state_dict(),
            'advantage_buffer': self.advantage_buffer.state_dict(),
            'strategy_buffer': self.strategy_buffer.state_dict(),
            'bucket_centroids': self.bucket_centroids,
            'current_batch_size': self.current_batch_size,
            'current_num_traversals': self.current_num_traversals,
            'current_rollouts': self.current_rollouts,
            'backoff_events': self.backoff_events,
            'recovery_events': self.recovery_events,
            'sanitization_events': self.sanitization_events,
            'skipped_batches': self.skipped_batches,
        }
        latest_path = os.path.join(Config.CHECKPOINT_DIR, Config.LATEST_CHECKPOINT_NAME)
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, Config.LATEST_CHECKPOINT_NAME)
        if is_best:
            best_path = os.path.join(Config.CHECKPOINT_DIR, Config.BEST_CHECKPOINT_NAME)
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, Config.BEST_CHECKPOINT_NAME)

        self.advantage_buffer.flush()
        self.strategy_buffer.flush()
        clear_runtime_caches()

    def load_checkpoint_state(self, checkpoint_state, resolved_path='<in-memory>'):
        self.advantage_net.load_state_dict(checkpoint_state['advantage_state_dict'])
        self.strategy_net.load_state_dict(checkpoint_state['strategy_state_dict'])
        self.advantage_optimizer.load_state_dict(checkpoint_state['advantage_optimizer_state_dict'])
        self.strategy_optimizer.load_state_dict(checkpoint_state['strategy_optimizer_state_dict'])
        self.advantage_buffer = Float16ReservoirBuffer.from_state_dict(checkpoint_state['advantage_buffer'])
        self.strategy_buffer = Float16ReservoirBuffer.from_state_dict(checkpoint_state['strategy_buffer'])
        self.bucket_centroids = np.asarray(checkpoint_state.get('bucket_centroids', self.bucket_centroids), dtype=np.float32)
        self.current_batch_size = int(checkpoint_state.get('current_batch_size', self.current_batch_size))
        self.current_num_traversals = int(checkpoint_state.get('current_num_traversals', self.current_num_traversals))
        self.current_rollouts = int(checkpoint_state.get('current_rollouts', self.current_rollouts))
        self.backoff_events = int(checkpoint_state.get('backoff_events', self.backoff_events))
        self.recovery_events = int(checkpoint_state.get('recovery_events', self.recovery_events))
        self.sanitization_events = int(checkpoint_state.get('sanitization_events', self.sanitization_events))
        self.skipped_batches = int(checkpoint_state.get('skipped_batches', self.skipped_batches))
        self.best_score = float(checkpoint_state.get('best_score', self.best_score))
        Config.BATCH_SIZE = self.current_batch_size
        Config.EQUITY_ROLLOUTS = self.current_rollouts
        logger.info('Resumed Deep CFR from %s at iteration %s', resolved_path, checkpoint_state['iteration'])
        return int(checkpoint_state['iteration']) + 1

    def load_checkpoint(self, checkpoint_path):
        resolved_path = resolve_checkpoint_path(checkpoint_path)
        checkpoint_state = torch.load(resolved_path, map_location='cpu', weights_only=False)
        return self.load_checkpoint_state(checkpoint_state, resolved_path)

    def train(self, iterations, start_iteration=0):
        start_time = time.time()
        final_metrics = {
            'avg_utility': 0.0,
            'exploitability_proxy': 0.0,
        }

        for iteration in range(start_iteration, iterations):
            self.iteration = iteration
            max_depth = min(Config.MAX_CFR_DEPTH, Config.START_MAX_DEPTH + (iteration // Config.CURRICULUM_INTERVAL))
            iteration_metrics = self._run_traversals(max_depth=max_depth)
            advantage_loss, advantage_grad_norm = self._train_network(
                self.advantage_net,
                self.advantage_optimizer,
                self.advantage_buffer,
                Config.ADVANTAGE_TRAIN_STEPS,
                'advantage',
            )
            strategy_loss, strategy_grad_norm = self._train_network(
                self.strategy_net,
                self.strategy_optimizer,
                self.strategy_buffer,
                Config.STRATEGY_TRAIN_STEPS,
                'strategy',
            )

            iteration_metrics['advantage_loss'] = advantage_loss
            iteration_metrics['strategy_loss'] = strategy_loss
            iteration_metrics['advantage_grad_norm'] = advantage_grad_norm
            iteration_metrics['strategy_grad_norm'] = strategy_grad_norm
            snapshot = get_memory_snapshot()
            memory_event = self._maybe_backoff(snapshot)
            if memory_event is not None:
                logger.warning('Deep CFR runtime adjustment at iteration %s: %s', iteration, memory_event)

            if iteration % Config.LOG_INTERVAL == 0:
                self._log_metrics(iteration, iteration_metrics, snapshot)

            score = iteration_metrics['avg_utility'] - iteration_metrics['exploitability_proxy']
            is_best = score > self.best_score
            if is_best:
                self.best_score = score

            if iteration % Config.CHECKPOINT_INTERVAL == 0 or is_best:
                self.save_checkpoint(iteration, iteration_metrics, is_best=is_best)

            if iteration % Config.CACHE_CLEAR_INTERVAL == 0:
                clear_runtime_caches()

            final_metrics = iteration_metrics

        elapsed = time.time() - start_time
        final_metrics['elapsed_seconds'] = elapsed
        final_metrics['backoff_events'] = self.backoff_events
        final_metrics['recovery_events'] = self.recovery_events
        final_metrics['skipped_batches'] = self.skipped_batches
        final_metrics['sanitization_events'] = self.sanitization_events
        return final_metrics