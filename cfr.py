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

def get_strategy(regret_sum):
    regret_sum = torch.clamp(regret_sum, min=0)
    norm = regret_sum.sum()
    return regret_sum / norm if norm > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)

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

def average_strategy(infoset: Infoset, actor=None):
    key = infoset.key
    if actor is not None:
        strategy_sum = ray.get(actor.get_strategy_sum.remote(key)).to(Config.DEVICE)
    else:
        strategy_sum = nodes[key]['strategy_sum']
    total = strategy_sum.sum()
    return strategy_sum / total if total > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)