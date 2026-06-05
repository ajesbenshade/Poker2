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
from engine.actions import ActionSpace, all_in_id, bet_id, check_call_id, fold_id

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


def _first_legal(mask, candidates):
    for action in candidates:
        if 0 <= action < len(mask) and mask[action]:
            return action
    return next(i for i, m in enumerate(mask) if m)


def _hand_features(state: GameState, seat: int) -> tuple[bool, int, bool]:
    from engine.cards import rank_of
    h0, h1 = state.hole_cards[seat]
    r0, r1 = rank_of(h0), rank_of(h1)
    pair = r0 == r1
    high = max(r0, r1)
    connectedish = abs(r0 - r1) <= 2
    return pair, high, connectedish


def loose_passive(state: GameState, seat: int, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    pair, high, connectedish = _hand_features(state, seat)
    to_call = state.call_amount(seat)
    if to_call == 0:
        return check_call_id()
    if to_call <= state.big_blind * 4 or pair or high >= 9 or connectedish:
        return check_call_id() if mask[check_call_id()] else _first_legal(mask, [])
    return fold_id() if mask[fold_id()] else check_call_id()


def loose_aggressive(state: GameState, seat: int, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    pair, high, connectedish = _hand_features(state, seat)
    pressure = pair or high >= 9 or connectedish or rng.random() < 0.25
    if pressure:
        return _first_legal(mask, [bet_id(3), bet_id(2), all_in_id(), check_call_id()])
    if state.call_amount(seat) <= state.big_blind * 2:
        return check_call_id() if mask[check_call_id()] else _first_legal(mask, [])
    return fold_id() if mask[fold_id()] else check_call_id()


def overfolder(state: GameState, seat: int, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    pair, high, _ = _hand_features(state, seat)
    to_call = state.call_amount(seat)
    if to_call == 0:
        return _first_legal(mask, [bet_id(1), check_call_id()]) if high >= 11 or pair else check_call_id()
    if pair and high >= 10 and to_call <= 0.5 * max(1, state.pot):
        return check_call_id()
    return fold_id() if mask[fold_id()] else check_call_id()


def bluff_catcher(state: GameState, seat: int, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    pair, high, _ = _hand_features(state, seat)
    to_call = state.call_amount(seat)
    if to_call == 0:
        return check_call_id()
    if pair or high >= 10 or to_call <= 0.33 * max(1, state.pot):
        return check_call_id() if mask[check_call_id()] else _first_legal(mask, [])
    return fold_id() if mask[fold_id()] else check_call_id()


def pot_pressure(state: GameState, seat: int, rng: random.Random) -> int:
    mask = legal_action_mask(state)
    pair, high, _ = _hand_features(state, seat)
    strong = pair or high >= 11
    if state.call_amount(seat) == 0:
        return _first_legal(mask, [bet_id(3), bet_id(2), check_call_id()])
    if strong:
        return _first_legal(mask, [bet_id(3), bet_id(2), all_in_id(), check_call_id()])
    if state.call_amount(seat) <= 0.25 * max(1, state.pot):
        return check_call_id() if mask[check_call_id()] else _first_legal(mask, [])
    return fold_id() if mask[fold_id()] else check_call_id()


BASELINES: Dict[str, Policy] = {
    "random": random_policy,
    "call_station": call_station,
    "tight_aggressive": tight_aggressive,
}

HUMAN_LIKE_BASELINES: Dict[str, Policy] = {
    "loose_passive": loose_passive,
    "loose_aggressive": loose_aggressive,
    "overfolder": overfolder,
    "bluff_catcher": bluff_catcher,
    "pot_pressure": pot_pressure,
}


# ---------------------------------------------------------------------------
# Trained policy as a callable
# ---------------------------------------------------------------------------

def policy_from_net(
    net: PolicyNet,
    device: torch.device,
    deterministic: bool = False,
    *,
    action_space: Optional[ActionSpace] = None,
    temperature: float = 1.0,
    bet_multiplier: float = 1.0,
    all_in_multiplier: float = 1.0,
) -> Policy:
    action_space = action_space or ActionSpace()
    temperature = max(1e-6, float(temperature))
    bet_multiplier = max(0.0, float(bet_multiplier))
    all_in_multiplier = max(0.0, float(all_in_multiplier))

    @torch.no_grad()
    def _act(state: GameState, seat: int, rng: random.Random) -> int:
        legal = np.asarray(legal_action_mask(state), dtype=np.float32)
        obs = encode_observation(state, perspective_seat=seat)
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        legal_t = torch.from_numpy(legal).to(device).unsqueeze(0)
        logits = net(obs_t, legal_t) / temperature
        masked = logits.masked_fill(legal_t < 0.5, float("-inf"))
        probs = torch.softmax(masked, dim=-1).squeeze(0).float().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if bet_multiplier != 1.0 or all_in_multiplier != 1.0:
            for action in range(len(probs)):
                if action_space.is_bet(action):
                    probs[action] *= bet_multiplier
            if 0 <= action_space.all_in_id < len(probs):
                probs[action_space.all_in_id] *= all_in_multiplier
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
    baselines: Optional[Dict[str, Policy]] = None,
    include_human_like: bool = False,
) -> Dict[str, float]:
    """Run trained policy vs each baseline. Returns ``{name: mbb_per_game}``.

    HU (num_players=2): trained plays both seats in turn (button rotated)
    against the baseline; results are averaged for positional symmetry.

    Multi-way (num_players>2): trained plays seat 0 vs N-1 baseline copies.
    Results are averaged across button rotations of the trained's seat
    so positional bias is removed (each seat plays the trained for
    1/N of hands).
    """
    n = num_hands or cfg.eval_hands
    bb = cfg.big_blind
    rng = rng or random.Random(0xBEEF)
    action_space = ActionSpace(cfg.bet_fractions)
    trained = policy_from_net(
        net,
        device,
        deterministic=False,
        action_space=action_space,
        temperature=cfg.policy_temperature,
        bet_multiplier=cfg.policy_bet_multiplier,
        all_in_multiplier=cfg.policy_all_in_multiplier,
    )
    np_ = cfg.num_players
    results: Dict[str, float] = {}
    chosen_baselines = dict(BASELINES if baselines is None else baselines)
    if include_human_like:
        chosen_baselines.update(HUMAN_LIKE_BASELINES)
    for name, baseline in chosen_baselines.items():
        if np_ == 2:
            a = play_match([trained, baseline], n, cfg, rng=random.Random(rng.random()))
            b = play_match([baseline, trained], n, cfg, rng=random.Random(rng.random()))
            chips_per_hand = (a[0] + b[1]) / 2.0
        else:
            # Multi-way: rotate trained's seat across all positions.
            per_seat = max(1, n // np_)
            total = 0.0
            for s in range(np_):
                policies = [baseline] * np_
                policies[s] = trained
                avgs = play_match(policies, per_seat, cfg,
                                  rng=random.Random(rng.random()))
                total += avgs[s]
            chips_per_hand = total / np_
        results[name] = chips_per_hand * 1000.0 / bb
    return results
