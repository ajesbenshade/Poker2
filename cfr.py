import torch
import numpy as np

from config import Config
from datatypes import Infoset
from environment import SimplifiedPokerEnvironment
from storage import LocalNodeStore, coerce_node_store

default_node_store = LocalNodeStore()


def get_strategy(regret_sum):
    regret_sum = torch.clamp(regret_sum, min=0)
    norm = regret_sum.sum()
    return regret_sum / norm if norm > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)


def mccfr(infoset: Infoset, iteration, prob=1.0, actor=None, node_store=None, environment=None, depth=0, max_depth=None, player=None):
    node_store = coerce_node_store(node_store=node_store, actor=actor, default_store=default_node_store)
    environment = environment if environment is not None else SimplifiedPokerEnvironment()
    max_depth = Config.MAX_DEPTH if max_depth is None else max_depth
    player = infoset.acting_player if player is None else player

    if environment.is_terminal(infoset, depth=depth, max_depth=max_depth):
        with torch.no_grad():
            return environment.evaluate_state(infoset)

    key = infoset.key
    regret_sum = node_store.get_regret_sum(key)
    strategy = get_strategy(regret_sum)

    actions = environment.legal_actions(infoset)
    sub_utils = torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)
    sampled_mask = torch.zeros(Config.NUM_ACTIONS, dtype=torch.bool, device=Config.DEVICE)

    greedy_idx = torch.argmax(strategy)
    for a_idx in range(Config.NUM_ACTIONS):
        if np.random.rand() < Config.SAMPLING_RATE or a_idx == greedy_idx:
            next_infoset = environment.next_infoset(infoset, actions[a_idx])
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

    if sampled_mask.sum() > 0:
        sampled_mean = sub_utils[sampled_mask].mean()
        sub_utils[~sampled_mask] = sampled_mean
    else:
        sub_utils.fill_(0.0)

    util = torch.sum(strategy * sub_utils).item()
    regrets = sub_utils - util

    discount = iteration / (iteration + 10.0) if iteration > 0 else 1.0
    regret_delta = regrets * prob * discount

    current_regret = node_store.get_regret_sum(key)
    new_regret = torch.maximum(current_regret - regret_delta * (1 if player == 0 else -1), torch.zeros_like(regret_delta))
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
    if len(actions) != Config.NUM_ACTIONS:
        raise ValueError(f"Expected {Config.NUM_ACTIONS} legal actions, got {len(actions)}")

    strategy = deep_agent.advantage_strategy(infoset, infoset.acting_player)
    deep_agent.record_strategy(infoset, strategy, weight=iteration + 1)

    action_values = torch.zeros(Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.NN_DTYPE)
    for action_idx, action in enumerate(actions):
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
    deep_agent.record_advantage(
        infoset,
        infoset.acting_player,
        action_values - node_value,
        weight=1.0,
    )

    return float(node_value.item())


def average_strategy(infoset: Infoset, actor=None, node_store=None):
    node_store = coerce_node_store(node_store=node_store, actor=actor, default_store=default_node_store)
    key = infoset.key
    strategy_sum = node_store.get_strategy_sum(key)
    total = strategy_sum.sum()
    return strategy_sum / total if total > 0 else torch.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, device=Config.DEVICE, dtype=Config.DTYPE)
