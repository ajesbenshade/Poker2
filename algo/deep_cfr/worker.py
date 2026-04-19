"""Multiprocessing worker for parallel Deep CFR external-sampling traversals.

Each worker process holds CPU snapshots of the current per-seat advantage
nets and serves traversal chunks dispatched by the trainer. Workers return
numpy arrays (advantage + strategy samples) which the main process bulk-
inserts into the shared :class:`ReservoirBuffer` instances via
:meth:`ReservoirBuffer.add_arrays`.

The pool uses ``forkserver`` start (configured in train.py) so we avoid
re-importing torch on every chunk and we don't fork a fully-loaded GPU
process. The eval7 hand-evaluator cache (engine.cards) is lazy-init at
module import in each worker — a one-time per-process cost.
"""
from __future__ import annotations

import io
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from engine.actions import ActionSpace

from .network import AdvantageNet
from .traversal import (
    make_net_strategy_fn,
    make_uniform_strategy_fn,
    samples_to_arrays,
    traverse_one,
)


# Module-level state, populated by ``_init_worker``.
_NETS: List[Optional[AdvantageNet]] = []
_DEVICE: torch.device = torch.device("cpu")
_OBS_DIM: int = 0
_NUM_ACTIONS: int = 0
_ACTION_SPACE: Optional[ActionSpace] = None
_NUM_PLAYERS: int = 2
_STARTING_STACK: int = 200
_SMALL_BLIND: int = 1
_BIG_BLIND: int = 2
_HIDDEN: int = 256
_BLOCKS: int = 4


def _init_worker(
    state_dict_blobs: List[Optional[bytes]],
    *,
    obs_dim: int,
    num_actions: int,
    hidden: int,
    blocks: int,
    action_space: ActionSpace,
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
) -> None:
    """Initializer run once per worker process.

    Builds CPU AdvantageNets and loads the provided state_dicts. ``None`` in
    ``state_dict_blobs`` means "use uniform strategy for that seat" (iter 0).
    """
    global _NETS, _DEVICE, _OBS_DIM, _NUM_ACTIONS, _ACTION_SPACE
    global _NUM_PLAYERS, _STARTING_STACK, _SMALL_BLIND, _BIG_BLIND, _HIDDEN, _BLOCKS

    # Limit per-worker thread pools — workers are inherently parallel.
    torch.set_num_threads(1)

    _DEVICE = torch.device("cpu")
    _OBS_DIM = obs_dim
    _NUM_ACTIONS = num_actions
    _ACTION_SPACE = action_space
    _NUM_PLAYERS = num_players
    _STARTING_STACK = starting_stack
    _SMALL_BLIND = small_blind
    _BIG_BLIND = big_blind
    _HIDDEN = hidden
    _BLOCKS = blocks

    nets: List[Optional[AdvantageNet]] = []
    for blob in state_dict_blobs:
        if blob is None:
            nets.append(None)
            continue
        net = AdvantageNet(obs_dim, num_actions, hidden=hidden, num_blocks=blocks)
        sd = torch.load(io.BytesIO(blob), map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    _NETS = nets


def update_nets(state_dict_blobs: List[Optional[bytes]]) -> None:
    """Replace per-seat CPU nets between iterations without re-init."""
    global _NETS
    nets: List[Optional[AdvantageNet]] = []
    for blob in state_dict_blobs:
        if blob is None:
            nets.append(None)
            continue
        net = AdvantageNet(_OBS_DIM, _NUM_ACTIONS, hidden=_HIDDEN, num_blocks=_BLOCKS)
        sd = torch.load(io.BytesIO(blob), map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    _NETS = nets


def run_chunk(
    traverser: int,
    chunk_size: int,
    iter_t: int,
    base_seed: int,
    button_offset: int,
    linear_weight: bool,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Run ``chunk_size`` traversals as ``traverser`` and return samples.

    Returns ``(adv_obs, adv_legal, adv_target, adv_weight,
    strat_obs, strat_legal, strat_target, strat_weight)``.
    """
    if _ACTION_SPACE is None:
        raise RuntimeError("worker not initialized — _init_worker must be called first")

    deal_rng = random.Random(base_seed)
    sample_rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)

    if any(n is not None for n in _NETS):
        strategy_fn = make_net_strategy_fn(_NETS, _DEVICE)
    else:
        strategy_fn = make_uniform_strategy_fn()

    adv_all = []
    strat_all = []
    for k in range(chunk_size):
        button = (button_offset + k) % _NUM_PLAYERS
        adv, strat = traverse_one(
            traverser=traverser,
            strategy_fn=strategy_fn,
            iter_t=iter_t,
            num_players=_NUM_PLAYERS,
            starting_stack=_STARTING_STACK,
            small_blind=_SMALL_BLIND,
            big_blind=_BIG_BLIND,
            button=button,
            action_space=_ACTION_SPACE,
            deal_rng=deal_rng,
            sample_rng=sample_rng,
            linear_weight=linear_weight,
        )
        adv_all.extend(adv)
        strat_all.extend(strat)

    a_obs, a_legal, a_target, a_weight = samples_to_arrays(
        adv_all, _OBS_DIM, _NUM_ACTIONS
    )
    s_obs, s_legal, s_target, s_weight = samples_to_arrays(
        strat_all, _OBS_DIM, _NUM_ACTIONS
    )
    return (
        a_obs, a_legal, a_target, a_weight,
        s_obs, s_legal, s_target, s_weight,
    )


def serialize_state_dict(net: Optional[AdvantageNet]) -> Optional[bytes]:
    """Pickle a net's CPU state_dict to bytes for IPC to workers."""
    if net is None:
        return None
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()
