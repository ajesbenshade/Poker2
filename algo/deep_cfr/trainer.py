"""Deep CFR training loop.

For each iteration ``t = 1..T``:

  1. For each traverser player ``p``, run ``traversals_per_iter`` external
     sampling MCCFR traversals using the current advantage networks. Each
     traversal pushes regret samples into ``advantage_buffers[p]`` and
     opponent-strategy samples into the shared ``strategy_buffer``.
  2. Train the per-player advantage network from its buffer (optionally
     resetting weights first \u2014 standard Deep CFR practice).
  3. Periodically train the average-policy network from the strategy
     buffer.
  4. Periodically evaluate the policy against scripted baselines and
     checkpoint.
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from engine.actions import ActionSpace, DEFAULT_BET_FRACTIONS
from engine.encoder import OBS_DIM

from .buffer import ReservoirBuffer
from .config import DeepCFRConfig
from .eval import evaluate_vs_baselines
from .inference_server import InferenceServerHandle
from .lbr import evaluate_lbr
from .network import AdvantageNet, PolicyNet
from .proxy_net import distill_proxy_net, make_proxy_advantage_net
from .traversal import (
    external_sampling,
    make_batched_net_strategy_fn,
    make_batched_uniform_strategy_fn,
    make_net_strategy_fn,
    make_uniform_strategy_fn,
    samples_to_arrays,
)
from .vectorized_traversal import traverse_many_vectorized
from . import worker as _worker_mod
from .worker import (
    _init_worker as _worker_init,
    update_nets as _worker_update_nets,
    run_chunk as _worker_run_chunk,
    run_chunk_to_file as _worker_run_chunk_to_file,
    run_chunk_vectorized as _worker_run_chunk_vectorized,
    run_chunk_vectorized_to_file as _worker_run_chunk_vectorized_to_file,
    serialize_state_dict as _worker_serialize,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingTraversalBatch:
    iter_t: int
    dispatched_at: float
    by_player: List[tuple]


def _traversal_chunk_size(total: int, num_workers: int, configured_chunk: int) -> int:
    """Choose traversal task size for multiprocessing dispatch.

    ``configured_chunk`` is intentionally a target task size, not a minimum:
    smaller tasks give the pool enough work units to smooth out variable game
    tree depth across workers. ``0`` keeps the old auto behavior of roughly one
    task per worker.
    """
    total = max(1, int(total))
    num_workers = max(1, int(num_workers))
    configured_chunk = int(configured_chunk)
    if configured_chunk > 0:
        return min(total, configured_chunk)
    return max(1, (total + num_workers - 1) // num_workers)


def _select_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name not in ("cuda", "cpu"):
        try:
            return torch.device(name)
        except Exception:
            pass
    return torch.device("cpu")


def _select_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if name in ("float32", "fp32"):
        return None
    if name in ("bfloat16", "bf16"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if name in ("float16", "fp16"):
        return torch.float16
    return None


class DeepCFRTrainer:
    def __init__(self, cfg: DeepCFRConfig):
        self.cfg = cfg
        if cfg.traversal_backend not in ("recursive", "vectorized"):
            raise ValueError(
                "traversal_backend must be either 'recursive' or 'vectorized'"
            )
        if cfg.worker_result_transport not in ("ipc", "file"):
            raise ValueError("worker_result_transport must be either 'ipc' or 'file'")
        if cfg.use_proxy_nets and cfg.proxy_training_steps <= 0:
            raise ValueError("proxy_training_steps must be positive when proxy nets are enabled")
        self.device = _select_device(cfg.device)
        self.amp_dtype = _select_amp_dtype(cfg.amp_dtype, self.device)
        self.num_actions = ActionSpace(cfg.bet_fractions).num_actions
        self.action_space = ActionSpace(cfg.bet_fractions)
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # One advantage net per player.
        self.advantage_nets: List[Optional[AdvantageNet]] = [None] * cfg.num_players
        self.proxy_advantage_nets: List[Optional[AdvantageNet]] = [None] * cfg.num_players
        # Policy net is shared (it learns the average opponent strategy too).
        self.policy_net = PolicyNet(
            obs_dim=OBS_DIM,
            num_actions=self.num_actions,
            hidden=cfg.hidden_size,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
        ).to(self.device)
        if cfg.use_torch_compile and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(
                    self.policy_net, mode="reduce-overhead", dynamic=False
                )
            except Exception as e:
                logger.warning("torch.compile(policy_net) failed: %s", e)

        # Buffers
        self.advantage_buffers = [
            ReservoirBuffer(cfg.advantage_buffer_size, OBS_DIM, self.num_actions,
                            seed=cfg.seed + 1 + p)
            for p in range(cfg.num_players)
        ]
        self.strategy_buffer = ReservoirBuffer(
            cfg.strategy_buffer_size, OBS_DIM, self.num_actions,
            seed=cfg.seed + 100,
        )

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        self._worker_result_dir = os.path.join(cfg.checkpoint_dir, "_worker_results")
        if cfg.worker_result_transport == "file":
            os.makedirs(self._worker_result_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.log_dir)
        self.iter = 0
        self._best_score = -float("inf")
        self._best_lbr = float("inf")
        self._pin_training_batches = bool(cfg.pin_training_batches)
        self._pin_memory_warning_emitted = False
        self._adv_streams = []
        if cfg.concurrent_advantage_training and self.device.type == "cuda":
            self._adv_streams = [torch.cuda.Stream() for _ in range(cfg.num_players)]

        # Worker pool for parallel traversals (None until first use).
        self._pool = None
        self._pool_workers = 0
        self._inference_server: Optional[InferenceServerHandle] = None

    # -------------------------------------------------------------------
    # Network training helpers
    # -------------------------------------------------------------------

    def _make_advantage_net(self) -> AdvantageNet:
        net = AdvantageNet(
            obs_dim=OBS_DIM,
            num_actions=self.num_actions,
            hidden=self.cfg.hidden_size,
            num_blocks=self.cfg.num_blocks,
            dropout=self.cfg.dropout,
        ).to(self.device)
        if self.cfg.use_torch_compile and self.device.type == "cuda":
            try:
                net = torch.compile(net, mode="reduce-overhead", dynamic=False)
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)
        return net

    def _make_proxy_net(self) -> AdvantageNet:
        return make_proxy_advantage_net(
            num_actions=self.num_actions,
            hidden=self.cfg.proxy_hidden_size,
            num_blocks=self.cfg.proxy_num_blocks,
            dropout=self.cfg.dropout,
            device=self.device,
        )

    def _traversal_net_shape(self) -> tuple[int, int]:
        if self.cfg.use_proxy_nets:
            return self.cfg.proxy_hidden_size, self.cfg.proxy_num_blocks
        return self.cfg.hidden_size, self.cfg.num_blocks

    def _traversal_nets(self) -> List[Optional[torch.nn.Module]]:
        if self.cfg.use_proxy_nets:
            return self.proxy_advantage_nets
        return self.advantage_nets

    def _array_to_device(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        non_blocking = False
        if self.device.type == "cuda" and self._pin_training_batches:
            try:
                tensor = tensor.pin_memory()
                non_blocking = True
            except RuntimeError as exc:
                if not self._pin_memory_warning_emitted:
                    logger.warning("pin_memory failed; using blocking transfers: %s", exc)
                    self._pin_memory_warning_emitted = True
                self._pin_training_batches = False
        return tensor.to(self.device, non_blocking=non_blocking)

    def _train_net(
        self,
        net: torch.nn.Module,
        buffer: ReservoirBuffer,
        steps: int,
        loss_kind: str,   # "regression" | "ce_soft"
    ) -> Dict[str, float]:
        if len(buffer) == 0 or steps <= 0:
            return {"loss": float("nan"), "steps": 0}
        net.train()
        opt = torch.optim.AdamW(
            net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        bs = min(self.cfg.train_batch_size, len(buffer))
        total_loss = 0.0
        loss_samples = 0
        last_steps = 0
        loss_log_interval = max(1, int(self.cfg.loss_log_interval))

        # Optional early-stop on held-out validation loss (advantage nets only).
        es_patience = max(0, int(self.cfg.adv_early_stop_patience))
        es_enabled = es_patience > 0 and loss_kind == "regression"
        es_eval_every = max(1, int(self.cfg.adv_early_stop_eval_every))
        es_min_steps = max(0, int(self.cfg.adv_early_stop_min_steps))
        es_val_frac = float(self.cfg.adv_early_stop_val_fraction)
        es_val_size = max(64, int(bs * es_val_frac)) if es_enabled else 0
        es_best = float("inf")
        es_no_improve = 0
        cfr_plus_clamp = bool(self.cfg.cfr_plus) and loss_kind == "regression"
        v_obs = None
        v_legal = None
        v_target = None
        v_weight = None
        if es_enabled:
            v_obs_np, v_legal_np, v_target_np, v_weight_np = buffer.sample(es_val_size)
            v_obs = self._array_to_device(v_obs_np)
            v_legal = self._array_to_device(v_legal_np)
            v_target = self._array_to_device(v_target_np)
            v_weight = self._array_to_device(v_weight_np)
            if cfr_plus_clamp:
                v_target = torch.clamp(v_target, min=0.0)
        for step in range(steps):
            obs_np, legal_np, target_np, weight_np = buffer.sample(bs)
            obs = self._array_to_device(obs_np)
            legal = self._array_to_device(legal_np)
            target = self._array_to_device(target_np)
            weight = self._array_to_device(weight_np)
            if cfr_plus_clamp:
                target = torch.clamp(target, min=0.0)
            opt.zero_grad(set_to_none=True)

            ctx = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if self.amp_dtype is not None
                else _NullCtx()
            )
            with ctx:
                pred = net(obs, legal)
                # Use weighted average (sum(w*loss) / sum(w)) so reported loss
                # is per-sample and effective LR is independent of iter_t.
                w_sum = weight.sum().clamp_min(1e-8)
                if loss_kind == "regression":
                    # Masked MSE on legal actions only, weighted by sample weight.
                    diff = (pred - target) ** 2 * legal
                    per_sample = diff.sum(dim=-1)
                    loss = (per_sample * weight).sum() / w_sum
                elif loss_kind == "ce_soft":
                    # Cross-entropy with soft targets, masked.
                    masked_logits = pred.masked_fill(legal < 0.5, float("-inf"))
                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    log_probs = torch.nan_to_num(log_probs, nan=0.0,
                                                neginf=0.0, posinf=0.0)
                    target_norm = target * legal
                    s = target_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    target_norm = target_norm / s
                    per_sample = -(target_norm * log_probs).sum(dim=-1)
                    loss = (per_sample * weight).sum() / w_sum
                else:
                    raise ValueError(loss_kind)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()
            if (step + 1) % loss_log_interval == 0 or step + 1 == steps:
                total_loss += float(loss.detach())
                loss_samples += 1
            last_steps = step + 1
            if es_enabled and (step + 1) >= es_min_steps and (step + 1) % es_eval_every == 0:
                with torch.no_grad():
                    v_pred = net(v_obs, v_legal)
                    v_diff = (v_pred - v_target) ** 2 * v_legal
                    v_per = v_diff.sum(dim=-1)
                    v_w_sum = v_weight.sum().clamp_min(1e-8)
                    val_loss = float((v_per * v_weight).sum() / v_w_sum)
                if val_loss + 1e-6 < es_best:
                    es_best = val_loss
                    es_no_improve = 0
                else:
                    es_no_improve += 1
                    if es_no_improve >= es_patience:
                        break
                net.train()
        net.eval()
        return {"loss": total_loss / max(1, loss_samples), "steps": last_steps}

    def _train_net_on_stream(
        self,
        stream: torch.cuda.Stream,
        net: torch.nn.Module,
        buffer: ReservoirBuffer,
        steps: int,
        loss_kind: str,
    ) -> Dict[str, float]:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            return self._train_net(net, buffer, steps, loss_kind)

    def _train_advantage_nets(self) -> List[Dict[str, float]]:
        for p in range(self.cfg.num_players):
            if self.cfg.reset_advantage_net_each_iter or self.advantage_nets[p] is None:
                self.advantage_nets[p] = self._make_advantage_net()

        if (
            self.cfg.concurrent_advantage_training
            and self.device.type == "cuda"
            and len(self._adv_streams) == self.cfg.num_players
            and self.cfg.num_players > 1
        ):
            stats_by_player: List[Optional[Dict[str, float]]] = [None] * self.cfg.num_players
            with ThreadPoolExecutor(max_workers=self.cfg.num_players) as executor:
                futures = []
                for p in range(self.cfg.num_players):
                    futures.append(executor.submit(
                        self._train_net_on_stream,
                        self._adv_streams[p],
                        self.advantage_nets[p],
                        self.advantage_buffers[p],
                        self.cfg.advantage_train_steps,
                        "regression",
                    ))
                for p, future in enumerate(futures):
                    stats_by_player[p] = future.result()
            torch.cuda.synchronize()
            return [stats for stats in stats_by_player if stats is not None]

        return [
            self._train_net(
                self.advantage_nets[p],
                self.advantage_buffers[p],
                self.cfg.advantage_train_steps,
                "regression",
            )
            for p in range(self.cfg.num_players)
        ]

    def _refresh_proxy_nets(self, t: int) -> List[Dict[str, float]]:
        if not self.cfg.use_proxy_nets:
            return []
        interval = max(1, int(self.cfg.proxy_refresh_interval))
        needs_initial = any(
            self.proxy_advantage_nets[p] is None and self.advantage_nets[p] is not None
            for p in range(self.cfg.num_players)
        )
        if not needs_initial and t % interval != 0:
            return []

        stats_by_player: List[Dict[str, float]] = []
        for p in range(self.cfg.num_players):
            teacher = self.advantage_nets[p]
            if teacher is None:
                stats_by_player.append({"loss": float("nan"), "strategy_l1": float("nan"), "steps": 0})
                continue
            if self.proxy_advantage_nets[p] is None:
                self.proxy_advantage_nets[p] = self._make_proxy_net()
            stats = distill_proxy_net(
                proxy_net=self.proxy_advantage_nets[p],
                teacher_net=teacher,
                buffer=self.advantage_buffers[p],
                steps=self.cfg.proxy_training_steps,
                batch_size=self.cfg.train_batch_size,
                learning_rate=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
                grad_clip=self.cfg.grad_clip,
                device=self.device,
                amp_dtype=self.amp_dtype,
                loss_log_interval=self.cfg.loss_log_interval,
                array_to_device=self._array_to_device,
            )
            self.writer.add_scalar(f"loss/proxy_p{p}", stats["loss"], t)
            self.writer.add_scalar(f"proxy/strategy_l1_p{p}", stats["strategy_l1"], t)
            stats_by_player.append(stats)
        return stats_by_player

    # -------------------------------------------------------------------
    # Worker pool
    # -------------------------------------------------------------------

    def _snapshot_state_dicts(self):
        return [_worker_serialize(net) for net in self._traversal_nets()]

    def _use_inference_server(self) -> bool:
        return self.cfg.traversal_inference_mode == "server" and self.cfg.num_workers > 0

    def _use_vectorized_traversal(self) -> bool:
        return self.cfg.traversal_backend == "vectorized"

    def _ensure_pool(self):
        if self.cfg.num_workers <= 0:
            return None
        if self._pool is not None:
            return self._pool
        import multiprocessing as mp
        from functools import partial
        # ROCm/CUDA cannot be initialized safely in a forked child. Server mode
        # uses spawn for both the GPU server and workers so queues/locks share a
        # single multiprocessing context and the server gets a clean runtime.
        ctx = mp.get_context("spawn" if self._use_inference_server() else "forkserver")
        blobs = self._snapshot_state_dicts()
        traversal_hidden, traversal_blocks = self._traversal_net_shape()
        if self._use_inference_server() and self._inference_server is None:
            self._inference_server = InferenceServerHandle(
                ctx,
                num_workers=self.cfg.num_workers,
                state_dict_blobs=blobs,
                obs_dim=OBS_DIM,
                num_actions=self.num_actions,
                hidden=traversal_hidden,
                blocks=traversal_blocks,
                dropout=self.cfg.dropout,
                num_players=self.cfg.num_players,
                device_name=str(self.device),
                amp_dtype_name=self.cfg.amp_dtype,
                batch_size=self.cfg.inference_server_batch_size,
                timeout_ms=self.cfg.inference_server_timeout_ms,
                queue_size=self.cfg.inference_server_queue_size,
            )
            logger.info(
                "started traversal inference server: device=%s batch=%d timeout=%.2fms",
                self.device,
                self.cfg.inference_server_batch_size,
                self.cfg.inference_server_timeout_ms,
            )
        # Pool initializer doesn't accept kwargs; bind via functools.partial.
        init_fn = partial(
            _worker_init,
            obs_dim=OBS_DIM,
            num_actions=self.num_actions,
            hidden=traversal_hidden,
            blocks=traversal_blocks,
            action_space=self.action_space,
            num_players=self.cfg.num_players,
            starting_stack=self.cfg.starting_stack,
            small_blind=self.cfg.small_blind,
            big_blind=self.cfg.big_blind,
            worker_torch_threads=self.cfg.worker_torch_threads,
            script_worker_nets=self.cfg.script_worker_nets,
            inference_request_queue=(
                None if self._inference_server is None else self._inference_server.request_queue
            ),
            inference_response_queues=(
                None if self._inference_server is None else self._inference_server.response_queues
            ),
            inference_worker_counter=(
                None if self._inference_server is None else self._inference_server.worker_counter
            ),
            inference_worker_counter_lock=(
                None if self._inference_server is None else self._inference_server.worker_counter_lock
            ),
        )
        self._pool = ctx.Pool(
            processes=self.cfg.num_workers,
            initializer=init_fn,
            initargs=(blobs,),
        )
        self._pool_workers = self.cfg.num_workers
        logger.info("started worker pool: %d processes", self._pool_workers)
        return self._pool

    def _refresh_workers(self):
        if self._pool is None:
            return
        blobs = self._snapshot_state_dicts()
        if self._inference_server is not None:
            self._inference_server.update(blobs)
            return
        # Broadcast new state_dicts to every worker. Pool.map fans out one
        # task per process when chunksize=1 and the iterable has W items.
        self._pool.map(_worker_update_nets, [blobs] * self._pool_workers, chunksize=1)

    def _close_pool(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._inference_server is not None:
            self._inference_server.close()
            self._inference_server = None

    def _serial_vectorized_external_sampling(self, traverser: int, t: int) -> None:
        cfg = self.cfg
        traversal_nets = self._traversal_nets()
        for net in traversal_nets:
            if net is not None:
                net.eval()

        if any(net is not None for net in traversal_nets):
            strategy_fn = make_net_strategy_fn(traversal_nets, self.device)
            batch_strategy_fn = make_batched_net_strategy_fn(traversal_nets, self.device)
        else:
            strategy_fn = make_uniform_strategy_fn()
            batch_strategy_fn = make_batched_uniform_strategy_fn()

        seed = self.rng.randint(0, 2**31 - 1)
        deal_rng = random.Random(seed)
        sample_seed_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
        adv_all = []
        strat_all = []
        processed = 0
        batch_size = max(1, int(cfg.vectorized_traversal_batch_size))
        while processed < cfg.traversals_per_iter:
            current = min(batch_size, cfg.traversals_per_iter - processed)
            adv, strat = traverse_many_vectorized(
                traverser=traverser,
                strategy_fn=strategy_fn,
                batch_strategy_fn=batch_strategy_fn,
                iter_t=t,
                num_traversals=current,
                num_players=cfg.num_players,
                starting_stack=cfg.starting_stack,
                small_blind=cfg.small_blind,
                big_blind=cfg.big_blind,
                button_offset=processed,
                action_space=self.action_space,
                deal_rng=deal_rng,
                sample_seed_rng=sample_seed_rng,
                linear_weight=cfg.linear_cfr,
                adv_weight_power=cfg.discounted_cfr_alpha,
                strat_weight_power=cfg.discounted_cfr_gamma,
            )
            adv_all.extend(adv)
            strat_all.extend(strat)
            processed += current

        a_obs, a_legal, a_target, a_weight = samples_to_arrays(
            adv_all, OBS_DIM, self.num_actions
        )
        s_obs, s_legal, s_target, s_weight = samples_to_arrays(
            strat_all, OBS_DIM, self.num_actions
        )
        if a_obs.shape[0] > 0:
            self.advantage_buffers[traverser].add_arrays(a_obs, a_legal, a_target, a_weight)
        if s_obs.shape[0] > 0:
            self.strategy_buffer.add_arrays(s_obs, s_legal, s_target, s_weight)

    def _parallel_external_sampling(self, traverser: int, t: int) -> None:
        cfg = self.cfg
        pool = self._ensure_pool()
        if pool is None:
            if self._use_vectorized_traversal():
                self._serial_vectorized_external_sampling(traverser, t)
                return
            external_sampling(
                traverser=traverser,
                advantage_nets=self._traversal_nets(),
                advantage_buffer=self.advantage_buffers[traverser],
                strategy_buffer=self.strategy_buffer,
                iter_t=t,
                num_traversals=cfg.traversals_per_iter,
                num_players=cfg.num_players,
                starting_stack=cfg.starting_stack,
                small_blind=cfg.small_blind,
                big_blind=cfg.big_blind,
                action_space=self.action_space,
                rng=self.rng,
                device=self.device,
                linear_weight=cfg.linear_cfr,
                adv_weight_power=cfg.discounted_cfr_alpha,
                strat_weight_power=cfg.discounted_cfr_gamma,
            )
            return

        total = cfg.traversals_per_iter
        nw = self._pool_workers
        chunk = _traversal_chunk_size(total, nw, cfg.worker_chunk_min)
        tasks = []
        use_file_results = cfg.worker_result_transport == "file"
        if self._use_vectorized_traversal():
            worker_fn = (
                _worker_run_chunk_vectorized_to_file
                if use_file_results else _worker_run_chunk_vectorized
            )
        else:
            worker_fn = _worker_run_chunk_to_file if use_file_results else _worker_run_chunk
        offset = 0
        button_offset = 0
        while offset < total:
            sz = min(chunk, total - offset)
            seed = self.rng.randint(0, 2**31 - 1)
            if self._use_vectorized_traversal():
                args = [
                    traverser, sz, t, seed, button_offset, cfg.linear_cfr,
                    cfg.vectorized_traversal_batch_size,
                    cfg.discounted_cfr_alpha, cfg.discounted_cfr_gamma,
                ]
            else:
                args = [
                    traverser, sz, t, seed, button_offset, cfg.linear_cfr,
                    cfg.discounted_cfr_alpha, cfg.discounted_cfr_gamma,
                ]
            if use_file_results:
                args.append(self._worker_result_dir)
            tasks.append(tuple(args))
            offset += sz
            button_offset += sz
        results = pool.starmap(worker_fn, tasks)
        self._insert_worker_results(
            results,
            self.advantage_buffers[traverser],
            self.strategy_buffer,
        )

    def _dispatch_async_traversals(self, t: int):
        """Async dispatch: returns list of (traverser, AsyncResult) pairs.

        Workers use whatever CPU snapshots they currently hold (refreshed
        by the prior iter's :meth:`_refresh_workers` call).
        """
        cfg = self.cfg
        pool = self._ensure_pool()
        if pool is None:
            return None
        nw = self._pool_workers
        total = cfg.traversals_per_iter
        chunk = _traversal_chunk_size(total, nw, cfg.worker_chunk_min)
        pending = []
        use_file_results = cfg.worker_result_transport == "file"
        if self._use_vectorized_traversal():
            worker_fn = (
                _worker_run_chunk_vectorized_to_file
                if use_file_results else _worker_run_chunk_vectorized
            )
        else:
            worker_fn = _worker_run_chunk_to_file if use_file_results else _worker_run_chunk
        for p in range(cfg.num_players):
            tasks = []
            offset = 0
            button_offset = 0
            while offset < total:
                sz = min(chunk, total - offset)
                seed = self.rng.randint(0, 2**31 - 1)
                if self._use_vectorized_traversal():
                    args = [
                        p, sz, t, seed, button_offset, cfg.linear_cfr,
                        cfg.vectorized_traversal_batch_size,
                        cfg.discounted_cfr_alpha, cfg.discounted_cfr_gamma,
                    ]
                else:
                    args = [
                        p, sz, t, seed, button_offset, cfg.linear_cfr,
                        cfg.discounted_cfr_alpha, cfg.discounted_cfr_gamma,
                    ]
                if use_file_results:
                    args.append(self._worker_result_dir)
                tasks.append(tuple(args))
                offset += sz
                button_offset += sz
            pending.append((p, pool.starmap_async(worker_fn, tasks)))
        return PendingTraversalBatch(t, time.time(), pending)

    def _materialize_worker_result(self, result):
        if not isinstance(result, str):
            return result
        path = result
        try:
            with np.load(path) as data:
                return (
                    data["a_obs"].copy(),
                    data["a_legal"].copy(),
                    data["a_target"].copy(),
                    data["a_weight"].copy(),
                    data["s_obs"].copy(),
                    data["s_legal"].copy(),
                    data["s_target"].copy(),
                    data["s_weight"].copy(),
                )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _insert_worker_results(
        self,
        results,
        adv_buf: ReservoirBuffer,
        strat_buf: ReservoirBuffer,
    ) -> None:
        for result in results:
            (
                a_obs, a_legal, a_target, a_weight,
                s_obs, s_legal, s_target, s_weight,
            ) = self._materialize_worker_result(result)
            if a_obs.shape[0] > 0:
                adv_buf.add_arrays(a_obs, a_legal, a_target, a_weight)
            if s_obs.shape[0] > 0:
                strat_buf.add_arrays(s_obs, s_legal, s_target, s_weight)

    def _collect_async_traversals(self, pending: Optional[PendingTraversalBatch]) -> None:
        """Block on async results and insert samples into buffers."""
        if pending is None:
            return
        strat_buf = self.strategy_buffer
        for p, ar in pending.by_player:
            results = ar.get()
            adv_buf = self.advantage_buffers[p]
            self._insert_worker_results(results, adv_buf, strat_buf)

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        logger.info(
            "deep_cfr start | iters=%d | traversals/iter/player=%d | players=%d | "
            "stack=%d | bb=%d | sizes=%s | obs=%d | actions=%d | device=%s | backend=%s",
            cfg.num_iterations, cfg.traversals_per_iter, cfg.num_players,
            cfg.starting_stack, cfg.big_blind, cfg.bet_fractions, OBS_DIM,
            self.num_actions, self.device, cfg.traversal_backend,
        )
        pending_batches = deque()
        next_async_iter = 2
        async_depth = max(1, int(cfg.async_pipeline_depth))

        for t in range(1, cfg.num_iterations + 1):
            self.iter = t
            iter_start = time.time()

            use_async = (
                cfg.async_pipeline
                and cfg.num_workers > 0
                and t > 1   # iter 1 must be sync to populate buffers first
            )

            # 1) Run external sampling for each traverser
            if use_async:
                # Keep up to async_depth traversal batches queued so workers can
                # keep moving while GPU training runs. Queue in waves: refreshing
                # worker snapshots is synchronous, so we only refresh after the
                # current wave drains instead of blocking behind future tasks.
                if not pending_batches:
                    while (
                        next_async_iter <= cfg.num_iterations
                        and len(pending_batches) < async_depth
                    ):
                        pending_batches.append(self._dispatch_async_traversals(next_async_iter))
                        next_async_iter += 1
            else:
                for p in range(cfg.num_players):
                    self._parallel_external_sampling(p, t)

            # 2) Train per-player advantage nets (concurrent with async traversals)
            adv_losses = []
            adv_stats = self._train_advantage_nets()
            for p, stats in enumerate(adv_stats):
                adv_losses.append(stats["loss"])
                self.writer.add_scalar(f"loss/advantage_p{p}", stats["loss"], t)
                self.writer.add_scalar(f"buffer/advantage_p{p}",
                                       len(self.advantage_buffers[p]), t)

            # 1b) Wait for async traversals (if any) and insert into buffers.
            if use_async and pending_batches:
                self._collect_async_traversals(pending_batches.popleft())

            proxy_stats = self._refresh_proxy_nets(t)

            # 2b) Push refreshed CPU snapshots to workers for next iter.
            if self._pool is not None and (not use_async or not pending_batches):
                self._refresh_workers()

            # 3) Periodically train the average-policy net
            policy_loss = float("nan")
            if t % max(1, cfg.train_strategy_every) == 0:
                stats = self._train_net(
                    self.policy_net,
                    self.strategy_buffer,
                    cfg.strategy_train_steps,
                    "ce_soft",
                )
                policy_loss = stats["loss"]
                self.writer.add_scalar("loss/policy", policy_loss, t)
                self.writer.add_scalar("buffer/strategy",
                                       len(self.strategy_buffer), t)

            wallclock = time.time() - iter_start
            self.writer.add_scalar("time/iter_seconds", wallclock, t)

            # 4) Periodic evaluation + checkpoint
            eval_payload = {}
            if cfg.eval_interval > 0 and t % cfg.eval_interval == 0:
                eval_payload = evaluate_vs_baselines(
                    self.policy_net, cfg, self.device,
                    num_hands=cfg.eval_hands,
                    rng=random.Random(cfg.seed + 9000 + t),
                )
                for name, mbbg in eval_payload.items():
                    self.writer.add_scalar(f"eval/mbb_per_game/{name}", mbbg, t)
                # Use the *worst-case* baseline margin so "best" means the
                # checkpoint strongest against the toughest opponent. Beating
                # a random bot by 5000 mbb but losing to tight_aggressive
                # should NOT count as "best".
                score = float(min(eval_payload.values())) if eval_payload else 0.0
                self.writer.add_scalar("eval/score_min_mbbg", score, t)
                if score > self._best_score:
                    self._best_score = score
                    self.save_checkpoint(os.path.join(cfg.checkpoint_dir, "best.pt"),
                                         meta={"iter": t, "score_mbbg": score,
                                               "eval": eval_payload})

            latest_interval = max(1, int(cfg.latest_checkpoint_interval))
            if t % latest_interval == 0 or t == cfg.num_iterations:
                self.save_checkpoint(os.path.join(cfg.checkpoint_dir, "latest.pt"),
                                     meta={"iter": t})

            # 5) Periodic LBR exploitability (Phase G)
            lbr_mbbg = None
            if cfg.lbr_interval > 0 and t % cfg.lbr_interval == 0:
                lbr_mbbg = evaluate_lbr(
                    self.policy_net, cfg, self.device,
                    num_hands=cfg.lbr_hands,
                    equity_samples=cfg.lbr_equity_samples,
                    rng=random.Random(cfg.seed + 7000 + t),
                )
                self.writer.add_scalar("eval/lbr_mbb_per_game", lbr_mbbg, t)
                # Save best-by-LBR checkpoint (lower is less exploitable).
                if lbr_mbbg < self._best_lbr:
                    self._best_lbr = float(lbr_mbbg)
                    self.save_checkpoint(
                        os.path.join(cfg.checkpoint_dir, "best_lbr.pt"),
                        meta={"iter": t, "lbr_mbbg": float(lbr_mbbg)},
                    )

            if t % max(1, cfg.log_interval) == 0:
                eval_str = (
                    " | eval " + " ".join(f"{k}={v:+.1f}mbb" for k, v in eval_payload.items())
                    if eval_payload else ""
                )
                if lbr_mbbg is not None:
                    eval_str += f" | lbr={lbr_mbbg:+.1f}mbb"
                if proxy_stats:
                    proxy_l1 = [s["strategy_l1"] for s in proxy_stats]
                    eval_str += " | proxy_l1 " + ",".join(f"{x:.4f}" for x in proxy_l1)
                logger.info(
                    "iter %d | adv_loss %s | pol_loss %.4f | adv_buf %s | strat_buf %d | %.1fs%s",
                    t,
                    ",".join(f"{x:.3f}" for x in adv_losses),
                    policy_loss,
                    "/".join(str(len(b)) for b in self.advantage_buffers),
                    len(self.strategy_buffer),
                    wallclock,
                    eval_str,
                )

        self.writer.close()
        self._close_pool()

    # -------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------

    def save_checkpoint(self, path: str, meta: Optional[Dict] = None) -> None:
        def _strip(sd):
            out = {}
            for k, v in sd.items():
                if k.startswith("_orig_mod."):
                    k = k[len("_orig_mod."):]
                out[k] = v
            return out

        payload = {
            "policy_net": _strip(self.policy_net.state_dict()),
            "advantage_nets": [
                None if net is None else _strip(net.state_dict())
                for net in self.advantage_nets
            ],
            "proxy_advantage_nets": [
                None if net is None else _strip(net.state_dict())
                for net in self.proxy_advantage_nets
            ],
            "iter": self.iter,
            "config": self.cfg.__dict__,
            "meta": meta or {},
        }
        torch.save(payload, path)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
