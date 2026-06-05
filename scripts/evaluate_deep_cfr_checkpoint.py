from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo.deep_cfr.config import DeepCFRConfig
from algo.deep_cfr.eval import evaluate_vs_baselines
from algo.deep_cfr.lbr import evaluate_lbr
from algo.deep_cfr.network import PolicyNet
from engine.actions import ActionSpace
from engine.encoder import OBS_DIM


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _cfg_from_payload(payload: dict, device_name: str | None) -> DeepCFRConfig:
    cfg = DeepCFRConfig()
    raw = payload.get("config", {}) or {}
    for key, val in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    cfg.bet_fractions = tuple(cfg.bet_fractions)
    if device_name is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = device_name
    return cfg


def _load_policy(path: Path, device_name: str | None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_payload(payload, device_name)
    device = torch.device(cfg.device)
    action_space = ActionSpace(cfg.bet_fractions)
    net = PolicyNet(
        obs_dim=OBS_DIM,
        num_actions=action_space.num_actions,
        hidden=cfg.hidden_size,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)
    net.load_state_dict(payload["policy_net"])
    net.eval()
    return payload, cfg, device, net


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) <= 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def evaluate_checkpoint(args, path: Path) -> None:
    payload, cfg, device, net = _load_policy(path, args.device)
    if args.policy_temperature is not None:
        cfg.policy_temperature = max(1e-6, float(args.policy_temperature))
    if args.policy_bet_mult is not None:
        cfg.policy_bet_multiplier = max(0.0, float(args.policy_bet_mult))
    if args.policy_all_in_mult is not None:
        cfg.policy_all_in_multiplier = max(0.0, float(args.policy_all_in_mult))
    seeds = _parse_seeds(args.seeds)
    metrics: dict[str, list[float]] = {}

    for seed in seeds:
        baseline_results = evaluate_vs_baselines(
            net,
            cfg,
            device,
            num_hands=args.eval_hands,
            rng=random.Random(seed),
            include_human_like=args.include_human_like,
        )
        for key, value in baseline_results.items():
            metrics.setdefault(key, []).append(float(value))
        if args.lbr_hands > 0:
            lbr = evaluate_lbr(
                net,
                cfg,
                device,
                num_hands=args.lbr_hands,
                equity_samples=args.lbr_equity_samples,
                rng=random.Random(seed + 100_000),
            )
            metrics.setdefault("lbr", []).append(float(lbr))

    meta = payload.get("meta", {}) or {}
    print(f"CHECKPOINT {path}")
    print(f"  iter={payload.get('iter')} meta={meta}")
    print(
        "  policy_transform="
        f"temp={cfg.policy_temperature:g} bet_mult={cfg.policy_bet_multiplier:g} "
        f"all_in_mult={cfg.policy_all_in_multiplier:g}"
    )
    for key in sorted(metrics):
        mean, std = _mean_std(metrics[key])
        print(f"  {key}: mean={mean:+.1f} std={std:.1f} n={len(metrics[key])}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved Deep CFR checkpoint(s).")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--eval-hands", type=int, default=6000)
    parser.add_argument("--lbr-hands", type=int, default=4000)
    parser.add_argument("--lbr-equity-samples", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="4242")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--include-human-like", action="store_true")
    parser.add_argument("--policy-temperature", type=float, default=None)
    parser.add_argument("--policy-bet-mult", type=float, default=None)
    parser.add_argument("--policy-all-in-mult", type=float, default=None)
    args = parser.parse_args()

    for path in args.checkpoints:
        evaluate_checkpoint(args, path)


if __name__ == "__main__":
    main()
