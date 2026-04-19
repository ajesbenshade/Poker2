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
from .lbr import evaluate_lbr
from .network import AdvantageNet, PolicyNet
from .traversal import external_sampling
from . import worker as _worker_mod
from .worker import (
    _init_worker as _worker_init,
    update_nets as _worker_update_nets,
    run_chunk as _worker_run_chunk,
    serialize_state_dict as _worker_serialize,
)

logger = logging.getLogger(__name__)


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
        self.device = _select_device(cfg.device)
        self.amp_dtype = _select_amp_dtype(cfg.amp_dtype, self.device)
        self.num_actions = ActionSpace(cfg.bet_fractions).num_actions
        self.action_space = ActionSpace(cfg.bet_fractions)
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # One advantage net per player.
        self.advantage_nets: List[Optional[AdvantageNet]] = [None] * cfg.num_players
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
        self.writer = SummaryWriter(cfg.log_dir)
        self.iter = 0
        self._best_score = -float("inf")

        # Worker pool for parallel traversals (None until first use).
        self._pool = None
        self._pool_workers = 0

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
        last_steps = 0
        for step in range(steps):
            obs_np, legal_np, target_np, weight_np = buffer.sample(bs)
            obs = torch.from_numpy(obs_np).to(self.device)
            legal = torch.from_numpy(legal_np).to(self.device)
            target = torch.from_numpy(target_np).to(self.device)
            weight = torch.from_numpy(weight_np).to(self.device)
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
            total_loss += float(loss.detach())
            last_steps = step + 1
        net.eval()
        return {"loss": total_loss / max(1, last_steps), "steps": last_steps}

    # -------------------------------------------------------------------
    # Worker pool
    # -------------------------------------------------------------------

    def _snapshot_state_dicts(self):
        return [_worker_serialize(net) for net in self.advantage_nets]

    def _ensure_pool(self):
        if self.cfg.num_workers <= 0:
            return None
        if self._pool is not None:
            return self._pool
        import multiprocessing as mp
        from functools import partial
        ctx = mp.get_context("forkserver")
        blobs = self._snapshot_state_dicts()
        # Pool initializer doesn't accept kwargs; bind via functools.partial.
        init_fn = partial(
            _worker_init,
            obs_dim=OBS_DIM,
            num_actions=self.num_actions,
            hidden=self.cfg.hidden_size,
            blocks=self.cfg.num_blocks,
            action_space=self.action_space,
            num_players=self.cfg.num_players,
            starting_stack=self.cfg.starting_stack,
            small_blind=self.cfg.small_blind,
            big_blind=self.cfg.big_blind,
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
        # Broadcast new state_dicts to every worker. Pool.map fans out one
        # task per process when chunksize=1 and the iterable has W items.
        self._pool.map(_worker_update_nets, [blobs] * self._pool_workers, chunksize=1)

    def _close_pool(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _parallel_external_sampling(self, traverser: int, t: int) -> None:
        cfg = self.cfg
        pool = self._ensure_pool()
        if pool is None:
            external_sampling(
                traverser=traverser,
                advantage_nets=self.advantage_nets,
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
            )
            return

        total = cfg.traversals_per_iter
        nw = self._pool_workers
        chunk = max(cfg.worker_chunk_min, (total + nw - 1) // nw)
        tasks = []
        offset = 0
        button_offset = 0
        while offset < total:
            sz = min(chunk, total - offset)
            seed = self.rng.randint(0, 2**31 - 1)
            tasks.append((traverser, sz, t, seed, button_offset, cfg.linear_cfr))
            offset += sz
            button_offset += sz
        results = pool.starmap(_worker_run_chunk, tasks)
        # Bulk-insert results into reservoirs.
        adv_buf = self.advantage_buffers[traverser]
        strat_buf = self.strategy_buffer
        for (a_obs, a_legal, a_target, a_weight,
             s_obs, s_legal, s_target, s_weight) in results:
            if a_obs.shape[0] > 0:
                adv_buf.add_arrays(a_obs, a_legal, a_target, a_weight)
            if s_obs.shape[0] > 0:
                strat_buf.add_arrays(s_obs, s_legal, s_target, s_weight)

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
        chunk = max(cfg.worker_chunk_min, (total + nw - 1) // nw)
        pending = []
        for p in range(cfg.num_players):
            tasks = []
            offset = 0
            button_offset = 0
            while offset < total:
                sz = min(chunk, total - offset)
                seed = self.rng.randint(0, 2**31 - 1)
                tasks.append((p, sz, t, seed, button_offset, cfg.linear_cfr))
                offset += sz
                button_offset += sz
            pending.append((p, pool.starmap_async(_worker_run_chunk, tasks)))
        return pending

    def _collect_async_traversals(self, pending) -> None:
        """Block on async results and insert samples into buffers."""
        if pending is None:
            return
        strat_buf = self.strategy_buffer
        for p, ar in pending:
            results = ar.get()
            adv_buf = self.advantage_buffers[p]
            for (a_obs, a_legal, a_target, a_weight,
                 s_obs, s_legal, s_target, s_weight) in results:
                if a_obs.shape[0] > 0:
                    adv_buf.add_arrays(a_obs, a_legal, a_target, a_weight)
                if s_obs.shape[0] > 0:
                    strat_buf.add_arrays(s_obs, s_legal, s_target, s_weight)

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        logger.info(
            "deep_cfr start | iters=%d | traversals/iter/player=%d | players=%d | "
            "stack=%d | bb=%d | sizes=%s | obs=%d | actions=%d | device=%s",
            cfg.num_iterations, cfg.traversals_per_iter, cfg.num_players,
            cfg.starting_stack, cfg.big_blind, cfg.bet_fractions, OBS_DIM,
            self.num_actions, self.device,
        )
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
                # Dispatch iter t traversals async; they overlap with GPU train.
                pending = self._dispatch_async_traversals(t)
            else:
                pending = None
                for p in range(cfg.num_players):
                    self._parallel_external_sampling(p, t)

            # 2) Train per-player advantage nets (concurrent with async traversals)
            adv_losses = []
            for p in range(cfg.num_players):
                if cfg.reset_advantage_net_each_iter or self.advantage_nets[p] is None:
                    self.advantage_nets[p] = self._make_advantage_net()
                stats = self._train_net(
                    self.advantage_nets[p],
                    self.advantage_buffers[p],
                    cfg.advantage_train_steps,
                    "regression",
                )
                adv_losses.append(stats["loss"])
                self.writer.add_scalar(f"loss/advantage_p{p}", stats["loss"], t)
                self.writer.add_scalar(f"buffer/advantage_p{p}",
                                       len(self.advantage_buffers[p]), t)

            # 1b) Wait for async traversals (if any) and insert into buffers.
            if pending is not None:
                self._collect_async_traversals(pending)

            # 2b) Push refreshed CPU snapshots to workers for next iter.
            if self._pool is not None:
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

            if t % max(1, cfg.log_interval) == 0:
                eval_str = (
                    " | eval " + " ".join(f"{k}={v:+.1f}mbb" for k, v in eval_payload.items())
                    if eval_payload else ""
                )
                if lbr_mbbg is not None:
                    eval_str += f" | lbr={lbr_mbbg:+.1f}mbb"
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
