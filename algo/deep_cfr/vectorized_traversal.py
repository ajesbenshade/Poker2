"""Opt-in vectorized Deep CFR traversal helpers.

This module implements a batched external-sampling traversal. It keeps the
engine's scalar, correctness-oriented :class:`engine.state.GameState` and
transition APIs, but batches strategy inference at both traverser and opponent
decision nodes.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from engine import (
    GameState,
    apply_action,
    encode_observation,
    is_terminal,
    legal_action_mask,
    new_hand,
    payoffs,
)
from engine.actions import ActionSpace
from engine.fast_state import (
    FastGameState,
    apply_action_in_place,
    new_fast_hand,
    restore_state,
    snapshot_state,
)

from .traversal import BatchStrategyFn, Sample, StrategyFn


def _sample_action(legal: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> int:
    legal_idx = np.flatnonzero(legal)
    if legal_idx.size == 0:
        raise RuntimeError("cannot sample an action with no legal actions")
    probs = sigma[legal_idx].astype(np.float64, copy=False)
    total = probs.sum()
    if total <= 0:
        return int(rng.choice(legal_idx))
    probs = probs / total
    return int(rng.choice(legal_idx, p=probs))


def _encode_states_for_seat(
    states: List[GameState],
    seat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    obs_batch = np.stack([
        encode_observation(state, perspective_seat=seat)
        for state in states
    ]).astype(np.float32, copy=False)
    legal_batch = np.stack([
        np.asarray(legal_action_mask(state), dtype=np.float32)
        for state in states
    ]).astype(np.float32, copy=False)
    return obs_batch, legal_batch


def _traverse_batch(
    states: List[GameState],
    sample_rngs: List[np.random.Generator],
    *,
    traverser: int,
    batch_strategy_fn: BatchStrategyFn,
    adv_samples: List[Sample],
    strat_samples: List[Sample],
    iter_t: int,
    linear_weight: bool,
    big_blind: int,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
) -> np.ndarray:
    """Original slow (immutable) implementation - kept for correctness testing and fallback."""
    if len(states) != len(sample_rngs):
        raise ValueError("states and sample_rngs must have the same length")
    if not states:
        return np.zeros(0, dtype=np.float32)

    values = np.zeros(len(states), dtype=np.float32)
    grouped: dict[int, List[int]] = defaultdict(list)
    for idx, state in enumerate(states):
        if is_terminal(state):
            values[idx] = float(payoffs(state)[traverser])
            continue
        grouped[state.to_act].append(idx)

    weight = float(iter_t) if linear_weight else 1.0
    if linear_weight:
        adv_w = float(iter_t) ** adv_weight_power
        strat_w = float(iter_t) ** strat_weight_power
    else:
        adv_w = 1.0
        strat_w = 1.0
    del weight  # legacy var name kept for diff readability
    for seat, frame_indices in grouped.items():
        seat_states = [states[idx] for idx in frame_indices]
        obs_batch, legal_batch = _encode_states_for_seat(seat_states, seat)
        sigma_batch = batch_strategy_fn(obs_batch, legal_batch, seat).astype(
            np.float32, copy=False
        )

        if seat == traverser:
            action_values = np.zeros_like(legal_batch, dtype=np.float32)
            for action_id in range(legal_batch.shape[1]):
                local_rows = np.nonzero(legal_batch[:, action_id] > 0.5)[0]
                if local_rows.size == 0:
                    continue
                child_states = [apply_action(seat_states[row], action_id) for row in local_rows]
                child_rngs = [sample_rngs[frame_indices[row]] for row in local_rows]
                child_values = _traverse_batch(
                    child_states,
                    child_rngs,
                    traverser=traverser,
                    batch_strategy_fn=batch_strategy_fn,
                    adv_samples=adv_samples,
                    strat_samples=strat_samples,
                    iter_t=iter_t,
                    linear_weight=linear_weight,
                    big_blind=big_blind,
                    adv_weight_power=adv_weight_power,
                    strat_weight_power=strat_weight_power,
                )
                action_values[local_rows, action_id] = child_values

            node_values = (sigma_batch * action_values).sum(axis=1).astype(np.float32)
            regrets = ((action_values - node_values[:, None]) / max(1, big_blind)) * legal_batch
            for local_row, frame_idx in enumerate(frame_indices):
                adv_samples.append((
                    obs_batch[local_row],
                    legal_batch[local_row],
                    regrets[local_row].astype(np.float32, copy=False),
                    adv_w,
                ))
                values[frame_idx] = node_values[local_row]
            continue

        child_states: List[GameState] = []
        child_rngs: List[np.random.Generator] = []
        child_frame_indices: List[int] = []
        for local_row, frame_idx in enumerate(frame_indices):
            legal = legal_batch[local_row]
            sigma = sigma_batch[local_row]
            strat_samples.append((
                obs_batch[local_row],
                legal,
                sigma.astype(np.float32, copy=False),
                strat_w,
            ))
            chosen = _sample_action(legal, sigma, sample_rngs[frame_idx])
            child_states.append(apply_action(states[frame_idx], chosen))
            child_rngs.append(sample_rngs[frame_idx])
            child_frame_indices.append(frame_idx)

        child_values = _traverse_batch(
            child_states,
            child_rngs,
            traverser=traverser,
            batch_strategy_fn=batch_strategy_fn,
            adv_samples=adv_samples,
            strat_samples=strat_samples,
            iter_t=iter_t,
            linear_weight=linear_weight,
            big_blind=big_blind,
            adv_weight_power=adv_weight_power,
            strat_weight_power=strat_weight_power,
        )
        for frame_idx, child_value in zip(child_frame_indices, child_values):
            values[frame_idx] = child_value

    return values


# =============================================================================
# FAST PATH (in-place mutation + snapshot/restore) - Priority 1 optimization
# =============================================================================

def _encode_states_for_seat_fast(
    states: List[FastGameState],
    seat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as _encode_states_for_seat but accepts FastGameState (attribute-compatible)."""
    obs_batch = np.stack([
        encode_observation(state, perspective_seat=seat)
        for state in states
    ]).astype(np.float32, copy=False)
    legal_batch = np.stack([
        np.asarray(legal_action_mask(state), dtype=np.float32)
        for state in states
    ]).astype(np.float32, copy=False)
    return obs_batch, legal_batch


def _traverse_batch_fast(
    states: List[FastGameState],
    sample_rngs: List[np.random.Generator],
    *,
    traverser: int,
    batch_strategy_fn: BatchStrategyFn,
    adv_samples: List[Sample],
    strat_samples: List[Sample],
    iter_t: int,
    linear_weight: bool,
    big_blind: int,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
) -> np.ndarray:
    """Fast version of _traverse_batch using in-place mutation + snapshot/restore.

    This is the high-impact implementation that eliminates the vast majority of
    Python object allocations in the Deep CFR traversal hot loop.
    """
    if len(states) != len(sample_rngs):
        raise ValueError("states and sample_rngs must have the same length")
    if not states:
        return np.zeros(0, dtype=np.float32)

    values = np.zeros(len(states), dtype=np.float32)
    grouped: dict[int, List[int]] = defaultdict(list)
    for idx, state in enumerate(states):
        if is_terminal(state):
            values[idx] = float(payoffs(state)[traverser])
            continue
        grouped[state.to_act].append(idx)

    if linear_weight:
        adv_w = float(iter_t) ** adv_weight_power
        strat_w = float(iter_t) ** strat_weight_power
    else:
        adv_w = 1.0
        strat_w = 1.0

    for seat, frame_indices in grouped.items():
        seat_states = [states[idx] for idx in frame_indices]
        obs_batch, legal_batch = _encode_states_for_seat_fast(seat_states, seat)
        sigma_batch = batch_strategy_fn(obs_batch, legal_batch, seat).astype(
            np.float32, copy=False
        )

        if seat == traverser:
            action_values = np.zeros_like(legal_batch, dtype=np.float32)
            for action_id in range(legal_batch.shape[1]):
                local_rows = np.nonzero(legal_batch[:, action_id] > 0.5)[0]
                if local_rows.size == 0:
                    continue

                # FAST PATH: mutate in place, recurse, restore
                child_states: List[FastGameState] = []
                child_snaps: List[dict] = []
                child_rngs: List[np.random.Generator] = []

                for row in local_rows:
                    snap = apply_action_in_place(seat_states[row], action_id)
                    child_states.append(seat_states[row])
                    child_snaps.append(snap)
                    child_rngs.append(sample_rngs[frame_indices[row]])

                child_values = _traverse_batch_fast(
                    child_states,
                    child_rngs,
                    traverser=traverser,
                    batch_strategy_fn=batch_strategy_fn,
                    adv_samples=adv_samples,
                    strat_samples=strat_samples,
                    iter_t=iter_t,
                    linear_weight=linear_weight,
                    big_blind=big_blind,
                    adv_weight_power=adv_weight_power,
                    strat_weight_power=strat_weight_power,
                )

                # Restore all children before continuing
                for st, sn in zip(child_states, child_snaps):
                    restore_state(st, sn)

                action_values[local_rows, action_id] = child_values

            node_values = (sigma_batch * action_values).sum(axis=1).astype(np.float32)
            regrets = ((action_values - node_values[:, None]) / max(1, big_blind)) * legal_batch
            for local_row, frame_idx in enumerate(frame_indices):
                adv_samples.append((
                    obs_batch[local_row],
                    legal_batch[local_row],
                    regrets[local_row].astype(np.float32, copy=False),
                    adv_w,
                ))
                values[frame_idx] = node_values[local_row]
            continue

        # Opponent seats (external sampling): sample one action per state
        child_states: List[FastGameState] = []
        child_snaps: List[dict] = []
        child_rngs: List[np.random.Generator] = []
        child_frame_indices: List[int] = []

        for local_row, frame_idx in enumerate(frame_indices):
            legal = legal_batch[local_row]
            sigma = sigma_batch[local_row]
            strat_samples.append((
                obs_batch[local_row],
                legal,
                sigma.astype(np.float32, copy=False),
                strat_w,
            ))
            chosen = _sample_action(legal, sigma, sample_rngs[frame_idx])

            snap = apply_action_in_place(states[frame_idx], chosen)
            child_states.append(states[frame_idx])
            child_snaps.append(snap)
            child_rngs.append(sample_rngs[frame_idx])
            child_frame_indices.append(frame_idx)

        child_values = _traverse_batch_fast(
            child_states,
            child_rngs,
            traverser=traverser,
            batch_strategy_fn=batch_strategy_fn,
            adv_samples=adv_samples,
            strat_samples=strat_samples,
            iter_t=iter_t,
            linear_weight=linear_weight,
            big_blind=big_blind,
            adv_weight_power=adv_weight_power,
            strat_weight_power=strat_weight_power,
        )

        for st, sn in zip(child_states, child_snaps):
            restore_state(st, sn)

        for frame_idx, child_value in zip(child_frame_indices, child_values):
            values[frame_idx] = child_value

    return values


def traverse_many_vectorized(
    *,
    traverser: int,
    strategy_fn: StrategyFn,
    batch_strategy_fn: BatchStrategyFn,
    iter_t: int,
    num_traversals: int,
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    button_offset: int,
    action_space: ActionSpace,
    deal_rng: random.Random,
    sample_seed_rng: np.random.Generator,
    linear_weight: bool = True,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
    use_fast_path: bool = True,
) -> Tuple[List[Sample], List[Sample]]:
    """Run a batch of external-sampling traversals.

    The returned samples use the same tuple format as ``traverse_one``. The
    ``strategy_fn`` argument is kept for API symmetry with the recursive path;
    the batched backend uses ``batch_strategy_fn`` for all strategy calls.

    When use_fast_path=True (default), the high-performance in-place mutable
    FastGameState implementation is used. This is the major training optimization.
    """
    adv_samples: List[Sample] = []
    strat_samples: List[Sample] = []
    sample_rngs: List[np.random.Generator] = []

    if use_fast_path:
        states: List[FastGameState] = []
        for k in range(num_traversals):
            button = (button_offset + k) % num_players
            state = new_fast_hand(
                num_players=num_players,
                starting_stack=starting_stack,
                small_blind=small_blind,
                big_blind=big_blind,
                button=button,
                rng=deal_rng,
                action_space=action_space,
            )
            seed = int(sample_seed_rng.integers(0, 2**63 - 1))
            states.append(state)
            sample_rngs.append(np.random.default_rng(seed))

        _ = strategy_fn
        _traverse_batch_fast(
            states,
            sample_rngs,
            traverser=traverser,
            batch_strategy_fn=batch_strategy_fn,
            adv_samples=adv_samples,
            strat_samples=strat_samples,
            iter_t=iter_t,
            linear_weight=linear_weight,
            big_blind=big_blind,
            adv_weight_power=adv_weight_power,
            strat_weight_power=strat_weight_power,
        )
    else:
        states_slow: List[GameState] = []
        for k in range(num_traversals):
            button = (button_offset + k) % num_players
            state = new_hand(
                num_players=num_players,
                starting_stack=starting_stack,
                small_blind=small_blind,
                big_blind=big_blind,
                button=button,
                rng=deal_rng,
                action_space=action_space,
            )
            seed = int(sample_seed_rng.integers(0, 2**63 - 1))
            states_slow.append(state)
            sample_rngs.append(np.random.default_rng(seed))

        _ = strategy_fn
        _traverse_batch(
            states_slow,
            sample_rngs,
            traverser=traverser,
            batch_strategy_fn=batch_strategy_fn,
            adv_samples=adv_samples,
            strat_samples=strat_samples,
            iter_t=iter_t,
            linear_weight=linear_weight,
            big_blind=big_blind,
            adv_weight_power=adv_weight_power,
            strat_weight_power=strat_weight_power,
        )

    return adv_samples, strat_samples
