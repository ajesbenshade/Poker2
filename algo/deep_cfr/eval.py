"""Evaluation utilities: head-to-head matches against scripted baselines.

We measure expected per-hand winnings of the trained policy in chips and
report it in **mbb/g** (milli-big-blinds per game), the standard HUNL
benchmark. Positive = trained policy wins.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from engine import (
    GameState,
    apply_action,
    encode_observation,
    is_terminal,
    legal_action_mask,
    new_hand,
    payoffs,
)
from engine.actions import ActionSpace, all_in_id, check_call_id, fold_id

from .config import DeepCFRConfig
from .network import PolicyNet


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

Policy = Callable[[GameState, int, random.Random], int]


def random_policy(state: GameState, seat: int, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    legal = [i for i, m in enumerate(mask) if m]
    return rng.choice(legal)


def call_station(state: GameState, seat: int, rng: random.Random) -> int:
    """Always call/check; never fold; never raise."""
    mask = legal_action_mask(state)
    cc = check_call_id()
    if mask[cc]:
        return cc
    legal = [i for i, m in enumerate(mask) if m]
    return rng.choice(legal)


def tight_aggressive(state: GameState, seat: int, rng: random.Random) -> int:
    """Very simple TAG heuristic: shove with strong cards, otherwise check/call,
    fold to large bets with weak hands."""
    from engine.cards import rank_of
    mask = legal_action_mask(state)
    h0, h1 = state.hole_cards[seat]
    r0, r1 = rank_of(h0), rank_of(h1)
    pair = r0 == r1
    high = max(r0, r1)
    strong = pair and high >= 8           # 99+
    medium = pair or high >= 10            # any pair or face
    to_call = state.call_amount(seat)
    pot = state.pot

    if strong and mask[all_in_id()]:
        return all_in_id()
    if to_call == 0:
        return check_call_id()
    if to_call <= 0.4 * pot and (medium or to_call <= state.big_blind * 2):
        return check_call_id()
    if mask[fold_id()]:
        return fold_id()
    return check_call_id()


BASELINES: Dict[str, Policy] = {
    "random": random_policy,
    "call_station": call_station,
    "tight_aggressive": tight_aggressive,
}


# ---------------------------------------------------------------------------
# Trained policy as a callable
# ---------------------------------------------------------------------------

def policy_from_net(
    net: PolicyNet,
    device: torch.device,
    deterministic: bool = False,
) -> Policy:
    @torch.no_grad()
    def _act(state: GameState, seat: int, rng: random.Random) -> int:
        legal = np.asarray(legal_action_mask(state), dtype=np.float32)
        obs = encode_observation(state, perspective_seat=seat)
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        legal_t = torch.from_numpy(legal).to(device).unsqueeze(0)
        probs = net.strategy(obs_t, legal_t).squeeze(0).float().cpu().numpy()
        legal_idx = [i for i, m in enumerate(legal) if m > 0.5]
        if not legal_idx:
            raise RuntimeError("no legal actions")
        if probs.sum() <= 0:
            return rng.choice(legal_idx)
        if deterministic:
            return int(np.argmax(probs))
        # Sample using cumulative distribution + rng.random() so we stay on the
        # caller's `random.Random` stream (deterministic given seed).
        legal_probs = np.array([probs[i] for i in legal_idx], dtype=np.float64)
        legal_probs /= legal_probs.sum()
        u = rng.random()
        cum = 0.0
        for i, p in zip(legal_idx, legal_probs):
            cum += p
            if u <= cum:
                return int(i)
        return int(legal_idx[-1])
    return _act


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def play_match(
    policies: List[Policy],
    num_hands: int,
    cfg: DeepCFRConfig,
    rng: Optional[random.Random] = None,
) -> List[float]:
    """Return per-seat average chip winnings over ``num_hands`` hands.

    Seats alternate the dealer button each hand to remove positional bias.
    """
    if len(policies) != cfg.num_players:
        raise ValueError("need one policy per seat")
    if rng is None:
        rng = random.Random()

    space = ActionSpace(cfg.bet_fractions)
    totals = [0.0] * cfg.num_players

    for hand_i in range(num_hands):
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
                # policy returned an illegal action (shouldn't happen, but be safe)
                action = next(i for i, m in enumerate(mask) if m)
            state = apply_action(state, action)
            safety += 1
        deltas = payoffs(state)
        for s in range(cfg.num_players):
            totals[s] += deltas[s]

    return [t / max(1, num_hands) for t in totals]


def evaluate_vs_baselines(
    net: PolicyNet,
    cfg: DeepCFRConfig,
    device: torch.device,
    num_hands: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """Run trained policy in seat 0 vs each baseline in seat 1 (HU only).

    Returns a dict ``{baseline_name: mbb_per_game}`` of trained policy's
    per-hand winnings in milli-big-blinds.
    """
    if cfg.num_players != 2:
        # Multi-way evaluation is implemented post-Phase-4.
        return {}
    n = num_hands or cfg.eval_hands
    bb = cfg.big_blind
    rng = rng or random.Random(0xBEEF)
    trained = policy_from_net(net, device, deterministic=False)
    results: Dict[str, float] = {}
    for name, baseline in BASELINES.items():
        # Average two orderings to remove positional advantage.
        a = play_match([trained, baseline], n, cfg, rng=random.Random(rng.random()))
        b = play_match([baseline, trained], n, cfg, rng=random.Random(rng.random()))
        chips_per_hand = (a[0] + b[1]) / 2.0
        results[name] = chips_per_hand * 1000.0 / bb
    return results
