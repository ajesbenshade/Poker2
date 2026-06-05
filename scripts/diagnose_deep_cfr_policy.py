from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo.deep_cfr.eval import BASELINES, HUMAN_LIKE_BASELINES, policy_from_net
from engine import apply_action, is_terminal, legal_action_mask, new_hand, payoffs
from engine.actions import ActionSpace
from scripts.evaluate_deep_cfr_checkpoint import _load_policy


def _parse_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def _bucket_action(space: ActionSpace, action: int) -> str:
    if action == space.fold_id:
        return "fold"
    if action == space.check_call_id:
        return "check_call"
    if action == space.all_in_id:
        return "all_in"
    if space.is_bet(action):
        return "bet_raise"
    return "other"


def _play_one_orientation(net, cfg, device, baseline, trained_seat: int, hands: int, seed: int) -> dict:
    rng = random.Random(seed)
    space = ActionSpace(cfg.bet_fractions)
    trained = policy_from_net(net, device, deterministic=False)
    policies = [baseline] * cfg.num_players
    policies[trained_seat] = trained

    stats = {
        "hands": 0,
        "chips": 0.0,
        "showdown_chips": 0.0,
        "nonshowdown_chips": 0.0,
        "actions": Counter(),
        "buckets": Counter(),
        "stage_buckets": defaultdict(Counter),
        "facing_bet": Counter(),
        "unopened": Counter(),
        "folds_facing_bet": 0,
        "calls_facing_bet": 0,
        "raises_facing_bet": 0,
        "bets_unopened": 0,
        "checks_unopened": 0,
        "illegal_fallbacks": 0,
        "trained_action_count": 0,
    }

    for hand_i in range(hands):
        button = hand_i % cfg.num_players
        state = new_hand(
            num_players=cfg.num_players,
            starting_stack=cfg.starting_stack,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            button=button,
            rng=rng,
            action_space=space,
        )
        safety = 0
        while not is_terminal(state) and safety < 400:
            seat = state.to_act
            action = policies[seat](state, seat, rng)
            mask = legal_action_mask(state)
            if not mask[action]:
                stats["illegal_fallbacks"] += int(seat == trained_seat)
                action = next(i for i, legal in enumerate(mask) if legal)
            if seat == trained_seat:
                bucket = _bucket_action(space, action)
                action_name = space.name(action)
                stage_name = state.stage.name.lower()
                to_call = state.call_amount(seat)
                stats["actions"][action_name] += 1
                stats["buckets"][bucket] += 1
                stats["stage_buckets"][stage_name][bucket] += 1
                stats["trained_action_count"] += 1
                if to_call > 0:
                    stats["facing_bet"][bucket] += 1
                    if bucket == "fold":
                        stats["folds_facing_bet"] += 1
                    elif bucket == "check_call":
                        stats["calls_facing_bet"] += 1
                    else:
                        stats["raises_facing_bet"] += 1
                else:
                    stats["unopened"][bucket] += 1
                    if bucket == "check_call":
                        stats["checks_unopened"] += 1
                    else:
                        stats["bets_unopened"] += 1
            state = apply_action(state, action)
            safety += 1

        result = payoffs(state)[trained_seat]
        stats["hands"] += 1
        stats["chips"] += float(result)
        if sum(1 for folded in state.folded if not folded) > 1:
            stats["showdown_chips"] += float(result)
        else:
            stats["nonshowdown_chips"] += float(result)

    return stats


def _merge_stats(items: list[dict]) -> dict:
    merged = {
        "hands": 0,
        "chips": 0.0,
        "showdown_chips": 0.0,
        "nonshowdown_chips": 0.0,
        "actions": Counter(),
        "buckets": Counter(),
        "stage_buckets": defaultdict(Counter),
        "facing_bet": Counter(),
        "unopened": Counter(),
        "folds_facing_bet": 0,
        "calls_facing_bet": 0,
        "raises_facing_bet": 0,
        "bets_unopened": 0,
        "checks_unopened": 0,
        "illegal_fallbacks": 0,
        "trained_action_count": 0,
    }
    for item in items:
        for key in ("hands", "folds_facing_bet", "calls_facing_bet", "raises_facing_bet", "bets_unopened", "checks_unopened", "illegal_fallbacks", "trained_action_count"):
            merged[key] += item[key]
        for key in ("chips", "showdown_chips", "nonshowdown_chips"):
            merged[key] += item[key]
        for key in ("actions", "buckets", "facing_bet", "unopened"):
            merged[key].update(item[key])
        for stage, counter in item["stage_buckets"].items():
            merged["stage_buckets"][stage].update(counter)
    return merged


def _print_stats(name: str, cfg, stats: dict) -> None:
    hands = max(1, stats["hands"])
    actions = max(1, stats["trained_action_count"])
    facing = sum(stats["facing_bet"].values())
    unopened = sum(stats["unopened"].values())
    mbbg = stats["chips"] / hands * 1000.0 / cfg.big_blind
    showdown_mbbg = stats["showdown_chips"] / hands * 1000.0 / cfg.big_blind
    nonshowdown_mbbg = stats["nonshowdown_chips"] / hands * 1000.0 / cfg.big_blind

    print(f"BASELINE {name}")
    print(f"  result_mbbg={mbbg:+.1f} showdown={showdown_mbbg:+.1f} nonshowdown={nonshowdown_mbbg:+.1f} hands={stats['hands']}")
    print(
        "  facing_bet "
        f"fold={_pct(stats['folds_facing_bet'], facing):.1f}% "
        f"call={_pct(stats['calls_facing_bet'], facing):.1f}% "
        f"raise={_pct(stats['raises_facing_bet'], facing):.1f}% n={facing}"
    )
    print(
        "  unopened "
        f"check={_pct(stats['checks_unopened'], unopened):.1f}% "
        f"bet={_pct(stats['bets_unopened'], unopened):.1f}% n={unopened}"
    )
    bucket_parts = [f"{key}={_pct(value, actions):.1f}%" for key, value in stats["buckets"].most_common()]
    print(f"  action_buckets {' '.join(bucket_parts)}")
    action_parts = [f"{key}={_pct(value, actions):.1f}%" for key, value in stats["actions"].most_common()]
    print(f"  actions {' '.join(action_parts)}")
    for stage in ("preflop", "flop", "turn", "river"):
        stage_total = sum(stats["stage_buckets"][stage].values())
        if stage_total:
            parts = [f"{key}={_pct(value, stage_total):.1f}%" for key, value in stats["stage_buckets"][stage].most_common()]
            print(f"  {stage} {' '.join(parts)} n={stage_total}")
    if stats["illegal_fallbacks"]:
        print(f"  illegal_fallbacks={stats['illegal_fallbacks']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Deep CFR policy behavior against scripted baselines.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--hands", type=int, default=5000)
    parser.add_argument("--seeds", type=str, default="4242")
    parser.add_argument("--baselines", type=str, default="bluff_catcher,pot_pressure,loose_passive")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    payload, cfg, device, net = _load_policy(args.checkpoint, args.device)
    baseline_pool = dict(BASELINES)
    baseline_pool.update(HUMAN_LIKE_BASELINES)
    names = _parse_names(args.baselines)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    print(f"CHECKPOINT {args.checkpoint}")
    print(f"  iter={payload.get('iter')} meta={payload.get('meta', {})}")
    print(f"  hands_per_seed={args.hands} seeds={seeds} baselines={names}")

    summary_scores: dict[str, list[float]] = defaultdict(list)
    for name in names:
        if name not in baseline_pool:
            raise SystemExit(f"unknown baseline {name!r}; choices={sorted(baseline_pool)}")
        all_stats = []
        seed_scores = []
        for seed in seeds:
            seed_stats = []
            hands_per_seat = max(1, args.hands // cfg.num_players)
            for trained_seat in range(cfg.num_players):
                orientation_stats = _play_one_orientation(
                    net,
                    cfg,
                    device,
                    baseline_pool[name],
                    trained_seat,
                    hands_per_seat,
                    seed + trained_seat * 100_000,
                )
                seed_stats.append(orientation_stats)
                all_stats.append(orientation_stats)
            seed_merged = _merge_stats(seed_stats)
            seed_scores.append(
                seed_merged["chips"] / max(1, seed_merged["hands"]) * 1000.0 / cfg.big_blind
            )
        stats = _merge_stats(all_stats)
        summary_scores[name].extend(seed_scores)
        _print_stats(name, cfg, stats)

    if len(seeds) > 1:
        print("SUMMARY")
        for name, values in sorted(summary_scores.items()):
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"  {name}: mean={mean:+.1f} std={std:.1f} n={len(values)}")


if __name__ == "__main__":
    main()