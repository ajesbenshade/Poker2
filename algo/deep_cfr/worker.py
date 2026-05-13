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
import os
import random
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch

from engine.actions import ActionSpace

from .inference_server import QueueInferenceClient, make_queue_strategy_fn
from .network import AdvantageNet
from .traversal import (
    BatchStrategyFn,
    StrategyFn,
    make_batched_net_strategy_fn,
    make_batched_strategy_fn_from_single,
    make_batched_uniform_strategy_fn,
    make_net_strategy_fn,
    make_uniform_strategy_fn,
    samples_to_arrays,
    traverse_one,
)
from .vectorized_traversal import traverse_many_vectorized


# Module-level state, populated by ``_init_worker``.
_NETS: List[Optional[torch.nn.Module]] = []
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
_WORKER_TORCH_THREADS: int = 1
_SCRIPT_WORKER_NETS: bool = False
_INFERENCE_CLIENT: Optional[QueueInferenceClient] = None


def _strip_compile_prefix(state_dict):
    """Return a state_dict compatible with plain worker networks.

    torch.compile wraps modules and prefixes state_dict keys with
    ``_orig_mod.``. Workers always rebuild uncompiled CPU AdvantageNets, so
    normalize incoming snapshots defensively at every IPC load boundary.
    """
    return {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in state_dict.items()
    }


def _load_state_dict_blob(blob: bytes):
    return _strip_compile_prefix(torch.load(io.BytesIO(blob), map_location="cpu"))


def _finalize_worker_net(net: AdvantageNet) -> torch.nn.Module:
    net.eval()
    if _SCRIPT_WORKER_NETS:
        scripted = torch.jit.script(net)
        scripted.eval()
        return scripted
    return net


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
    worker_torch_threads: int = 1,
    script_worker_nets: bool = False,
    inference_request_queue=None,
    inference_response_queues=None,
    inference_worker_counter=None,
    inference_worker_counter_lock=None,
) -> None:
    """Initializer run once per worker process.

    Builds CPU AdvantageNets and loads the provided state_dicts. ``None`` in
    ``state_dict_blobs`` means "use uniform strategy for that seat" (iter 0).
    """
    global _NETS, _DEVICE, _OBS_DIM, _NUM_ACTIONS, _ACTION_SPACE
    global _NUM_PLAYERS, _STARTING_STACK, _SMALL_BLIND, _BIG_BLIND, _HIDDEN, _BLOCKS
    global _WORKER_TORCH_THREADS, _SCRIPT_WORKER_NETS
    global _INFERENCE_CLIENT

    # Limit per-worker thread pools — workers are inherently parallel.
    _WORKER_TORCH_THREADS = max(1, int(worker_torch_threads))
    _SCRIPT_WORKER_NETS = bool(script_worker_nets)
    torch.set_num_threads(_WORKER_TORCH_THREADS)

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

    _INFERENCE_CLIENT = None
    if inference_request_queue is not None and inference_response_queues is not None:
        if inference_worker_counter is None or inference_worker_counter_lock is None:
            raise RuntimeError("inference server worker assignment primitives missing")
        with inference_worker_counter_lock:
            worker_id = int(inference_worker_counter.value)
            inference_worker_counter.value += 1
        worker_id %= len(inference_response_queues)
        _INFERENCE_CLIENT = QueueInferenceClient(
            inference_request_queue,
            inference_response_queues[worker_id],
            worker_id,
        )
        _NETS = [None] * num_players
        return

    nets: List[Optional[torch.nn.Module]] = []
    for blob in state_dict_blobs:
        if blob is None:
            nets.append(None)
            continue
        net = AdvantageNet(obs_dim, num_actions, hidden=hidden, num_blocks=blocks)
        net.load_state_dict(_load_state_dict_blob(blob))
        nets.append(_finalize_worker_net(net))
    _NETS = nets


def update_nets(state_dict_blobs: List[Optional[bytes]]) -> None:
    """Replace per-seat CPU nets between iterations without re-init."""
    global _NETS
    if _INFERENCE_CLIENT is not None:
        return
    nets: List[Optional[torch.nn.Module]] = []
    for blob in state_dict_blobs:
        if blob is None:
            nets.append(None)
            continue
        net = AdvantageNet(_OBS_DIM, _NUM_ACTIONS, hidden=_HIDDEN, num_blocks=_BLOCKS)
        net.load_state_dict(_load_state_dict_blob(blob))
        nets.append(_finalize_worker_net(net))
    _NETS = nets


def _make_strategy_fns() -> Tuple[StrategyFn, BatchStrategyFn]:
    if _INFERENCE_CLIENT is not None:
        single_fn = make_queue_strategy_fn(_INFERENCE_CLIENT)
        return single_fn, make_batched_strategy_fn_from_single(single_fn)
    if any(n is not None for n in _NETS):
        return make_net_strategy_fn(_NETS, _DEVICE), make_batched_net_strategy_fn(_NETS, _DEVICE)
    return make_uniform_strategy_fn(), make_batched_uniform_strategy_fn()


def run_chunk(
    traverser: int,
    chunk_size: int,
    iter_t: int,
    base_seed: int,
    button_offset: int,
    linear_weight: bool,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
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

    strategy_fn, _ = _make_strategy_fns()

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
            adv_weight_power=adv_weight_power,
            strat_weight_power=strat_weight_power,
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


def run_chunk_vectorized(
    traverser: int,
    chunk_size: int,
    iter_t: int,
    base_seed: int,
    button_offset: int,
    linear_weight: bool,
    vectorized_batch_size: int,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Run a traversal chunk with the opt-in vectorized backend."""
    if _ACTION_SPACE is None:
        raise RuntimeError("worker not initialized — _init_worker must be called first")

    deal_rng = random.Random(base_seed)
    sample_seed_rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    strategy_fn, batch_strategy_fn = _make_strategy_fns()

    adv_all = []
    strat_all = []
    processed = 0
    batch_size = max(1, int(vectorized_batch_size))
    while processed < chunk_size:
        current = min(batch_size, chunk_size - processed)
        adv, strat = traverse_many_vectorized(
            traverser=traverser,
            strategy_fn=strategy_fn,
            batch_strategy_fn=batch_strategy_fn,
            iter_t=iter_t,
            num_traversals=current,
            num_players=_NUM_PLAYERS,
            starting_stack=_STARTING_STACK,
            small_blind=_SMALL_BLIND,
            big_blind=_BIG_BLIND,
            button_offset=button_offset + processed,
            action_space=_ACTION_SPACE,
            deal_rng=deal_rng,
            sample_seed_rng=sample_seed_rng,
            linear_weight=linear_weight,
            adv_weight_power=adv_weight_power,
            strat_weight_power=strat_weight_power,
        )
        adv_all.extend(adv)
        strat_all.extend(strat)
        processed += current

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


def _save_chunk_result(
    result: Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ],
    result_dir: str,
) -> str:
    os.makedirs(result_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(
        prefix="deep_cfr_chunk_",
        suffix=".npz",
        dir=result_dir,
    )
    os.close(fd)
    (
        a_obs, a_legal, a_target, a_weight,
        s_obs, s_legal, s_target, s_weight,
    ) = result
    np.savez(
        path,
        a_obs=a_obs,
        a_legal=a_legal,
        a_target=a_target,
        a_weight=a_weight,
        s_obs=s_obs,
        s_legal=s_legal,
        s_target=s_target,
        s_weight=s_weight,
    )
    return path


def run_chunk_to_file(
    traverser: int,
    chunk_size: int,
    iter_t: int,
    base_seed: int,
    button_offset: int,
    linear_weight: bool,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
    result_dir: str = ".",
) -> str:
    return _save_chunk_result(
        run_chunk(
            traverser,
            chunk_size,
            iter_t,
            base_seed,
            button_offset,
            linear_weight,
            adv_weight_power,
            strat_weight_power,
        ),
        result_dir,
    )


def run_chunk_vectorized_to_file(
    traverser: int,
    chunk_size: int,
    iter_t: int,
    base_seed: int,
    button_offset: int,
    linear_weight: bool,
    vectorized_batch_size: int,
    adv_weight_power: float = 1.0,
    strat_weight_power: float = 1.0,
    result_dir: str = ".",
) -> str:
    return _save_chunk_result(
        run_chunk_vectorized(
            traverser,
            chunk_size,
            iter_t,
            base_seed,
            button_offset,
            linear_weight,
            vectorized_batch_size,
            adv_weight_power,
            strat_weight_power,
        ),
        result_dir,
    )


def serialize_state_dict(net: Optional[AdvantageNet]) -> Optional[bytes]:
    """Pickle a net's CPU state_dict to bytes for IPC to workers."""
    if net is None:
        return None
    # torch.compile wraps modules so state_dict keys gain an "_orig_mod."
    # prefix. Workers rebuild plain (uncompiled) nets, so strip it for IPC.
    sd = {}
    for k, v in net.state_dict().items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        sd[k] = v.detach().cpu()
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()
