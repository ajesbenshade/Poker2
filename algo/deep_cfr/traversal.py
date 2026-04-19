"""External-sampling MCCFR traversal.

Refactored to be **device- and buffer-agnostic**: ``traverse_one`` takes a
``strategy_fn(obs, legal, seat) -> probs`` callable and accumulates samples
into Python lists, which the caller turns into numpy arrays for bulk
insertion into the appropriate :class:`ReservoirBuffer`. That decouples the
recursive game-tree walk from torch / device / locking concerns and lets us
run many traversals in parallel worker processes (no GPU access required).

Backward-compatible :func:`external_sampling` is preserved as a serial
wrapper that constructs a strategy_fn from the passed advantage nets.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from engine import (
    GameState,
    encode_observation,
    is_terminal,
    legal_action_mask,
    apply_action,
    payoffs,
)
from engine.actions import ActionSpace

from .buffer import ReservoirBuffer
from .network import AdvantageNet


# ---------------------------------------------------------------------------
# Regret matching
# ---------------------------------------------------------------------------

def regret_matching(advantages: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Return a strategy distribution from raw per-action advantages.

    Standard CFR+ rule: positive part of regrets, normalized; uniform over
    legal actions if all regrets are non-positive.
    """
    pos = np.maximum(advantages, 0.0) * legal
    total = pos.sum()
    if total > 0:
        return pos / total
    legal_count = legal.sum()
    if legal_count <= 0:
        return np.zeros_like(legal)
    return legal / legal_count


# ---------------------------------------------------------------------------
# Strategy callables
# ---------------------------------------------------------------------------

# Signature: (obs, legal_mask, seat) -> probability distribution
StrategyFn = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def make_uniform_strategy_fn() -> StrategyFn:
    def _fn(obs: np.ndarray, legal: np.ndarray, seat: int) -> np.ndarray:
        c = legal.sum()
        if c <= 0:
            return np.zeros_like(legal)
        return legal / c
    return _fn


def make_net_strategy_fn(
    nets: List[Optional[AdvantageNet]],
    device: torch.device,
) -> StrategyFn:
    """Build a strategy_fn that queries one of the per-seat advantage nets.

    Nets are expected to be in eval mode and on ``device``. Returns the
    uniform distribution if a seat's net is ``None`` (iter 0).
    """
    @torch.no_grad()
    def _fn(obs: np.ndarray, legal: np.ndarray, seat: int) -> np.ndarray:
        net = nets[seat]
        if net is None:
            c = legal.sum()
            if c <= 0:
                return np.zeros_like(legal)
            return legal / c
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        legal_t = torch.from_numpy(legal).to(device).unsqueeze(0)
        adv = net(obs_t, legal_t).squeeze(0).float().cpu().numpy()
        return regret_matching(adv, legal)
    return _fn


# Sample tuple: (obs, legal_mask, target_vec, weight)
Sample = Tuple[np.ndarray, np.ndarray, np.ndarray, float]


# ---------------------------------------------------------------------------
# Recursive traversal (sample-collecting)
# ---------------------------------------------------------------------------

def _traverse(
    state: GameState,
    traverser: int,
    strategy_fn: StrategyFn,
    adv_samples: List[Sample],
    strat_samples: List[Sample],
    iter_t: int,
    rng: np.random.Generator,
    linear_weight: bool,
    big_blind: int,
) -> float:
    """Recursive traversal returning the counterfactual value at ``state``
    for the traverser (in chips). Pushes regret/strategy samples into the
    passed-in lists rather than directly into a buffer."""
    if is_terminal(state):
        return float(payoffs(state)[traverser])

    seat = state.to_act
    legal_list = legal_action_mask(state)
    legal = np.asarray(legal_list, dtype=np.float32)
    obs = encode_observation(state, perspective_seat=seat)

    sigma = strategy_fn(obs, legal, seat)
    weight = float(iter_t) if linear_weight else 1.0

    if seat == traverser:
        action_values = np.zeros_like(legal, dtype=np.float32)
        for a in range(len(legal_list)):
            if not legal_list[a]:
                continue
            action_values[a] = _traverse(
                apply_action(state, a),
                traverser,
                strategy_fn,
                adv_samples,
                strat_samples,
                iter_t,
                rng,
                linear_weight,
                big_blind,
            )
        node_value = float((sigma * action_values).sum())
        # Normalize regrets by big_blind to keep targets ~O(1).
        instantaneous_regret = ((action_values - node_value) / max(1, big_blind)) * legal
        adv_samples.append((obs, legal, instantaneous_regret.astype(np.float32), weight))
        return node_value

    # External sampling: opponent decision \u2014 sample one action from sigma.
    strat_samples.append((obs, legal, sigma.astype(np.float32), weight))
    legal_idx = np.flatnonzero(legal)
    probs = sigma[legal_idx]
    s = probs.sum()
    if s <= 0:
        chosen = int(rng.choice(legal_idx))
    else:
        probs = probs / s
        chosen = int(rng.choice(legal_idx, p=probs))
    return _traverse(
        apply_action(state, chosen),
        traverser,
        strategy_fn,
        adv_samples,
        strat_samples,
        iter_t,
        rng,
        linear_weight,
        big_blind,
    )


def traverse_one(
    *,
    traverser: int,
    strategy_fn: StrategyFn,
    iter_t: int,
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    button: int,
    action_space: ActionSpace,
    deal_rng: random.Random,
    sample_rng: np.random.Generator,
    linear_weight: bool = True,
) -> Tuple[List[Sample], List[Sample]]:
    """Run a single external-sampling traversal and return collected samples.

    ``deal_rng`` is a stdlib ``random.Random`` used by the engine to deal
    cards (engine API requires Random). ``sample_rng`` is a numpy
    ``Generator`` used for in-traversal opponent action sampling.
    """
    from engine import new_hand
    state = new_hand(
        num_players=num_players,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        button=button,
        rng=deal_rng,
        action_space=action_space,
    )
    adv: List[Sample] = []
    strat: List[Sample] = []
    _traverse(
        state, traverser, strategy_fn, adv, strat,
        iter_t, sample_rng, linear_weight, big_blind,
    )
    return adv, strat


def samples_to_arrays(
    samples: List[Sample],
    obs_dim: int,
    num_actions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of sample tuples into numpy arrays.

    Returns empty (0-row) arrays of the correct shape if ``samples`` is empty.
    """
    if not samples:
        return (
            np.zeros((0, obs_dim), dtype=np.float32),
            np.zeros((0, num_actions), dtype=np.float32),
            np.zeros((0, num_actions), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    obs = np.stack([s[0] for s in samples]).astype(np.float32, copy=False)
    legal = np.stack([s[1] for s in samples]).astype(np.float32, copy=False)
    target = np.stack([s[2] for s in samples]).astype(np.float32, copy=False)
    weight = np.array([s[3] for s in samples], dtype=np.float32)
    return obs, legal, target, weight


# ---------------------------------------------------------------------------
# Serial wrapper (back-compat with prior call sites and tests)
# ---------------------------------------------------------------------------

def external_sampling(
    traverser: int,
    advantage_nets: List[Optional[AdvantageNet]],
    advantage_buffer: ReservoirBuffer,
    strategy_buffer: ReservoirBuffer,
    iter_t: int,
    *,
    num_traversals: int,
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    action_space: ActionSpace,
    rng,
    device: torch.device,
    linear_weight: bool = True,
) -> None:
    """Serial external-sampling MCCFR. Used by tests and the no-pool path.

    ``rng`` may be a ``random.Random`` (legacy) or a ``np.random.Generator``;
    it is used to seed both card dealing and opponent action sampling.
    """
    if isinstance(rng, np.random.Generator):
        deal_rng = random.Random(int(rng.integers(0, 2**63 - 1)))
        sample_rng = rng
    else:
        deal_rng = rng if isinstance(rng, random.Random) else random.Random()
        sample_rng = np.random.default_rng(deal_rng.randint(0, 2**63 - 1))

    for net in advantage_nets:
        if net is not None:
            net.eval()
    strategy_fn = make_net_strategy_fn(advantage_nets, device)

    adv_all: List[Sample] = []
    strat_all: List[Sample] = []
    for k in range(num_traversals):
        button = k % num_players
        adv, strat = traverse_one(
            traverser=traverser,
            strategy_fn=strategy_fn,
            iter_t=iter_t,
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            button=button,
            action_space=action_space,
            deal_rng=deal_rng,
            sample_rng=sample_rng,
            linear_weight=linear_weight,
        )
        adv_all.extend(adv)
        strat_all.extend(strat)

    obs_dim = advantage_buffer.obs_dim
    num_actions = advantage_buffer.num_actions
    a_obs, a_legal, a_target, a_weight = samples_to_arrays(adv_all, obs_dim, num_actions)
    s_obs, s_legal, s_target, s_weight = samples_to_arrays(strat_all, obs_dim, num_actions)
    if a_obs.shape[0] > 0:
        advantage_buffer.add_arrays(a_obs, a_legal, a_target, a_weight)
    if s_obs.shape[0] > 0:
        strategy_buffer.add_arrays(s_obs, s_legal, s_target, s_weight)
