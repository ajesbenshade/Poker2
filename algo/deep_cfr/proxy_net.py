"""Proxy advantage-net distillation for Deep CFR traversal.

Proxy nets are optional, smaller traversal-only advantage networks. They are
trained to mimic the current main advantage nets (with improved strategy-aware
distillation) and can be broadcast to CPU workers or the inference server for
much faster strategy inference during the (now very fast) traversals.

The key improvement in distillation (Priority 5) is the ability to blend
advantage MSE with direct regret-matching strategy matching, which produces
proxies that behave much more like the teacher during actual external sampling.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch

from engine.encoder import OBS_DIM

from .buffer import ReservoirBuffer
from .network import AdvantageNet


def make_proxy_advantage_net(
    *,
    num_actions: int,
    hidden: int,
    num_blocks: int,
    dropout: float,
    device: torch.device,
) -> AdvantageNet:
    return AdvantageNet(
        obs_dim=OBS_DIM,
        num_actions=num_actions,
        hidden=hidden,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)


def regret_matching_tensor(advantages: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    pos = torch.clamp(advantages, min=0.0) * legal
    total = pos.sum(dim=-1, keepdim=True)
    uniform = legal / legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(total > 0, pos / total.clamp_min(1e-8), uniform)


def distill_proxy_net(
    *,
    proxy_net: torch.nn.Module,
    teacher_net: torch.nn.Module,
    buffer: ReservoirBuffer,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    loss_log_interval: int,
    array_to_device: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    strategy_weight: float = 0.0,   # New: weight on strategy distillation (0 = advantage MSE only)
) -> Dict[str, float]:
    if len(buffer) == 0 or steps <= 0:
        return {"loss": float("nan"), "strategy_l1": float("nan"), "steps": 0}

    def to_device(array: np.ndarray) -> torch.Tensor:
        if array_to_device is not None:
            return array_to_device(array)
        return torch.from_numpy(array).to(device)

    teacher_net.eval()
    proxy_net.train()
    opt = torch.optim.AdamW(
        proxy_net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    bs = min(batch_size, len(buffer))
    total_loss = 0.0
    total_strategy_l1 = 0.0
    samples = 0
    last_steps = 0
    loss_log_interval = max(1, int(loss_log_interval))

    for step in range(steps):
        obs_np, legal_np, _target_np, weight_np = buffer.sample(bs)
        obs = to_device(obs_np)
        legal = to_device(legal_np)
        weight = to_device(weight_np)

        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else _NullCtx()
        )
        with torch.no_grad():
            teacher_adv = teacher_net(obs, legal).float()
        with ctx:
            proxy_adv = proxy_net(obs, legal)
            # Advantage MSE loss (original behavior)
            adv_diff = (proxy_adv.float() - teacher_adv) ** 2 * legal
            adv_per = adv_diff.sum(dim=-1)
            w_sum = weight.sum().clamp_min(1e-8)
            adv_loss = (adv_per * weight).sum() / w_sum

            # Strategy distillation loss (regret matching output)
            if strategy_weight > 0:
                with torch.no_grad():
                    teacher_strat = regret_matching_tensor(teacher_adv, legal)
                proxy_strat = regret_matching_tensor(proxy_adv, legal)
                # L1 on strategies (stable and directly relevant to traversal sampling)
                strat_loss = (torch.abs(proxy_strat - teacher_strat) * legal).sum(dim=-1)
                strat_loss = (strat_loss * weight).sum() / w_sum
                loss = (1.0 - strategy_weight) * adv_loss + strategy_weight * strat_loss
            else:
                loss = adv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy_net.parameters(), grad_clip)
        opt.step()

        if (step + 1) % loss_log_interval == 0 or step + 1 == steps:
            with torch.no_grad():
                proxy_after = proxy_net(obs, legal).float()
                teacher_strategy = regret_matching_tensor(teacher_adv, legal)
                proxy_strategy = regret_matching_tensor(proxy_after, legal)
                strategy_l1 = torch.abs(teacher_strategy - proxy_strategy).sum(dim=-1).mean()
            total_loss += float(loss.detach())
            total_strategy_l1 += float(strategy_l1.detach())
            samples += 1
        last_steps = step + 1

    proxy_net.eval()
    return {
        "loss": total_loss / max(1, samples),
        "strategy_l1": total_strategy_l1 / max(1, samples),
        "steps": last_steps,
    }


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False