import torch
import numpy as np
import ray  # Added import to fix NameError
from collections import defaultdict
from game import terminal, simulate_action_batch, simulate_action  # Add simulate_action here
from config import Config
from datatypes import Action, Infoset

# Local nodes as fallback
nodes = defaultdict(lambda: {
    'regret_sum': torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE),
    'strategy_sum': torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)
})

def _regret_matching_core(regret_sum):
    # Compile-friendly regret matching keeps this hot tensor path fast on ROCm.
    positive_regret = torch.clamp(regret_sum, min=0)
    normalizer = positive_regret.sum()
    uniform_strategy = torch.full_like(positive_regret, 1.0 / Config.NUM_ACTIONS)
    safe_normalizer = torch.clamp(normalizer, min=torch.finfo(positive_regret.dtype).eps)
    normalized_strategy = positive_regret / safe_normalizer
    return torch.where(normalizer > 0, normalized_strategy, uniform_strategy)


try:
    # Compile the frequent regret-matching kernel for the 7900XT tensor path.
    _REGRET_MATCHING_IMPL = torch.compile(_regret_matching_core, mode='max-autotune', fullgraph=False) if torch.cuda.is_available() else _regret_matching_core
except Exception:
    _REGRET_MATCHING_IMPL = _regret_matching_core


def get_strategy(regret_sum):
    # Route regret matching through the compiled helper to reduce per-node overhead.
    return _REGRET_MATCHING_IMPL(regret_sum)


@torch.no_grad()
def apply_regret_matching_boost(regret_sum, strategy_sum, weight):
    # Periodically blend average strategy toward regret matching for lightweight stabilization.
    boosted_strategy = get_strategy(regret_sum)
    return strategy_sum.lerp(boosted_strategy, weight)


@torch.no_grad()
def mccfr(infoset: Infoset, iteration, prob=1.0, actor=None, depth=0, max_depth=3, player=0):
    if terminal(infoset) or depth >= max_depth:
        return simulate_action(infoset, None) + np.random.normal(0, 0.1)
    
    key = infoset.key
    if actor is not None:
        regret_sum = ray.get(actor.get_regret_sum.remote(key)).to(Config.DEVICE)
    else:
        regret_sum = nodes[key]['regret_sum']
    
    strategy = get_strategy(regret_sum)
    
    actions = [Action(i) for i in range(Config.NUM_ACTIONS)]
    sub_utils = torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)
    sampled_mask = torch.zeros(Config.NUM_ACTIONS, dtype=torch.bool, device=Config.DEVICE)
    
    # Sample and compute sub-utilities
    greedy_idx = torch.argmax(strategy)
    for a_idx in range(Config.NUM_ACTIONS):
        if np.random.rand() < Config.SAMPLING_RATE or a_idx == greedy_idx:
            next_infoset = Infoset(infoset.key[0], infoset.history + (actions[a_idx].value,))
            next_player = 1 - player  # Alternate player
            sub_util = mccfr(next_infoset, iteration, prob * strategy[a_idx], actor, depth+1, max_depth, next_player)
            sub_utils[a_idx] = -sub_util if next_player == 0 else sub_util  # Negate if opponent's turn
            sampled_mask[a_idx] = True
    
    # Fill unsampled with mean of sampled (avoid bias/NaN)
    if sampled_mask.sum() > 0:
        sampled_mean = sub_utils[sampled_mask].mean()
        sub_utils[~sampled_mask] = sampled_mean
    else:
        # Fallback if no samples (rare): use uniform estimate
        sub_utils.fill_(0.0)  # Or simulate quick rollout
    
    util = torch.sum(strategy * sub_utils).item()
    regrets = sub_utils - util
    
    # Linear discount for sustained updates
    discount = iteration / (iteration + 10.0) if iteration > 0 else 1.0
    regret_delta = regrets * prob * discount
    
    # CFR+ clamping: Floor negative regrets in accumulation
    if actor is not None:
        current_regret = ray.get(actor.get_regret_sum.remote(key)).to(Config.DEVICE)
        new_regret = torch.maximum(current_regret - regret_delta * (1 if player == 0 else -1), torch.zeros_like(regret_delta))  # Adjust sign for player
        actor.update_regret_sum.remote(key, new_regret - current_regret)
        actor.update_strategy_sum.remote(key, strategy * prob)
    else:
        current_regret = nodes[key]['regret_sum']
        nodes[key]['regret_sum'] = torch.maximum(current_regret - regret_delta * (1 if player == 0 else -1), torch.zeros_like(regret_delta))
        nodes[key]['strategy_sum'] += strategy * prob
    
    return util


@torch.no_grad()
def average_strategy(infoset: Infoset, actor=None):
    key = infoset.key
    if actor is not None:
        strategy_sum = ray.get(actor.get_strategy_sum.remote(key)).to(Config.DEVICE)
    else:
        strategy_sum = nodes[key]['strategy_sum']
    total = strategy_sum.sum()
    return strategy_sum / total if total > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)