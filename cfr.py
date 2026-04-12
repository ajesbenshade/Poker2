import torch
import numpy as np

from config import Config
from datatypes import Infoset
from environment import SimplifiedPokerEnvironment
from storage import LocalNodeStore, coerce_node_store

default_node_store = LocalNodeStore()


def _sanitize_tensor(tensor, *, dtype=None):
    cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned.to(dtype=dtype) if dtype is not None else cleaned


def get_strategy(regret_sum):
    regret_sum = _sanitize_tensor(torch.clamp(regret_sum, min=0.0), dtype=Config.DTYPE)
    norm = regret_sum.sum()
    return regret_sum / norm if norm > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)


def mask_strategy(strategy, actions):
    masked = torch.zeros(Config.NUM_ACTIONS, device=strategy.device, dtype=strategy.dtype)
    legal_indices = [action.value for action in actions]
    if not legal_indices:
        return torch.full(
            (Config.NUM_ACTIONS,),
            1.0 / Config.NUM_ACTIONS,
            device=strategy.device,
            dtype=strategy.dtype,
        )

    legal_probs = _sanitize_tensor(strategy[legal_indices], dtype=strategy.dtype)
    legal_probs = torch.clamp(legal_probs, min=0.0)
    total = legal_probs.sum()
    if float(total.item()) <= 0.0:
        legal_probs = torch.full(
            (len(legal_indices),),
            1.0 / len(legal_indices),
            device=strategy.device,
            dtype=strategy.dtype,
        )
    else:
        legal_probs = legal_probs / total
    masked[legal_indices] = legal_probs
    return masked


def mccfr(infoset: Infoset, iteration, prob=1.0, actor=None, node_store=None, environment=None, depth=0, max_depth=None, player=None):
    del player
    node_store = coerce_node_store(node_store=node_store, actor=actor, default_store=default_node_store)
    environment = environment if environment is not None else SimplifiedPokerEnvironment()
    max_depth = Config.MAX_DEPTH if max_depth is None else max_depth

    if environment.is_terminal(infoset, depth=depth, max_depth=max_depth):
        with torch.no_grad():
            return float(environment.evaluate_state(infoset))

    actions = environment.legal_actions(infoset)
    key = infoset.key
    regret_sum = node_store.get_regret_sum(key)
    strategy = mask_strategy(get_strategy(regret_sum), actions)

    sub_utils = torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)
    sampled_mask = torch.zeros(Config.NUM_ACTIONS, dtype=torch.bool, device=Config.DEVICE)
    legal_mask = torch.zeros(Config.NUM_ACTIONS, dtype=torch.bool, device=Config.DEVICE)
    legal_mask[[action.value for action in actions]] = True

    greedy_idx = max((action.value for action in actions), key=lambda idx: float(strategy[idx].item()))
    for action in actions:
        a_idx = action.value
        if np.random.rand() < Config.SAMPLING_RATE or a_idx == greedy_idx:
            next_infoset = environment.next_infoset(infoset, action)
            next_player = next_infoset.acting_player
            sub_util = mccfr(
                next_infoset,
                iteration,
                prob * float(strategy[a_idx].item()),
                node_store=node_store,
                environment=environment,
                depth=depth + 1,
                max_depth=max_depth,
                player=next_player,
            )
            sub_utils[a_idx] = -sub_util if next_player == 0 else sub_util
            sampled_mask[a_idx] = True

    sampled_legal_mask = sampled_mask & legal_mask
    if bool(sampled_legal_mask.any().item()):
        sampled_mean = sub_utils[sampled_legal_mask].mean()
        sub_utils[legal_mask & ~sampled_mask] = sampled_mean
    else:
        sub_utils[legal_mask] = 0.0

    sub_utils[~legal_mask] = 0.0
    sub_utils = _sanitize_tensor(sub_utils, dtype=Config.DTYPE)
    util = torch.sum(strategy * sub_utils).item()
    regrets = _sanitize_tensor(sub_utils - util, dtype=Config.DTYPE)
    regrets[~legal_mask] = 0.0

    discount = iteration / (iteration + 10.0) if iteration > 0 else 1.0
    regret_delta = regrets * prob * discount

    current_regret = node_store.get_regret_sum(key)
    new_regret = torch.maximum(
        _sanitize_tensor(current_regret + regret_delta, dtype=Config.DTYPE),
        torch.zeros_like(regret_delta),
    )
    node_store.update_regret_sum(key, new_regret - current_regret)
    node_store.update_strategy_sum(key, strategy * prob)

    return util


def deep_cfr_traverse(infoset: Infoset, iteration, deep_agent, environment=None, depth=0, max_depth=None):
    environment = environment if environment is not None else SimplifiedPokerEnvironment()
    max_depth = Config.MAX_DEPTH if max_depth is None else max_depth
    deep_agent.register_infoset(infoset)

    if environment.is_terminal(infoset, depth=depth, max_depth=max_depth):
        return float(environment.evaluate_state(infoset))

    actions = environment.legal_actions(infoset)
    legal_mask = torch.zeros(Config.NUM_ACTIONS, dtype=torch.bool, device=Config.DEVICE)
    legal_mask[[action.value for action in actions]] = True

    strategy = mask_strategy(
        deep_agent.advantage_strategy(infoset, infoset.acting_player).to(dtype=Config.NN_DTYPE),
        actions,
    )
    deep_agent.record_strategy(infoset, strategy, weight=iteration + 1)

    action_values = torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.NN_DTYPE)
    for action in actions:
        action_idx = action.value
        next_infoset = environment.next_infoset(infoset, action)
        next_value = deep_cfr_traverse(
            next_infoset,
            iteration,
            deep_agent=deep_agent,
            environment=environment,
            depth=depth + 1,
            max_depth=max_depth,
        )
        action_values[action_idx] = -float(next_value)

    node_value = torch.sum(strategy.to(dtype=Config.NN_DTYPE) * action_values)
    node_value = torch.nan_to_num(node_value, nan=0.0, posinf=0.0, neginf=0.0)
    advantage_targets = torch.nan_to_num(action_values - node_value, nan=0.0, posinf=0.0, neginf=0.0)
    advantage_targets[~legal_mask] = 0.0
    deep_agent.record_advantage(
        infoset,
        infoset.acting_player,
        advantage_targets,
        weight=1.0,
    )

    return float(node_value.item())


def average_strategy(infoset: Infoset, actor=None, node_store=None):
    node_store = coerce_node_store(node_store=node_store, actor=actor, default_store=default_node_store)
    key = infoset.key
    strategy_sum = node_store.get_strategy_sum(key)
    total = strategy_sum.sum()
    return strategy_sum / total if total > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)
