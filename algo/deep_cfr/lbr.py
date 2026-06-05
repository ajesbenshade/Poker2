"""Local Best Response (LBR) approximate exploitability.

LBR is a tractable lower bound on the exploitability of a poker policy.
At every LBR decision we restrict the responder's action set to {fold,
call/check} and compute the better choice using a Monte-Carlo equity
estimate against the opponent's *prior* range (here we use a uniform
range over un-blocked opponent hole cards as a baseline; tighter range
tracking is straightforward to add later).

Reference: Lisý & Bowling, "Equilibrium Approximation Quality of Current
No-Limit Poker Bots" (2017).

Returned metric is **mbb/g** (milli-big-blinds per game) won by LBR. A
positive number is the lower-bound exploitability of the trained policy:
larger = more exploitable. A truly unexploitable policy would yield
mbb/g <= 0.
"""
from __future__ import annotations

import random
from typing import Optional

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
from engine.actions import ActionSpace, check_call_id, fold_id
from engine.cards import evaluate_seven, CARD_COUNT

from .config import DeepCFRConfig
from .eval import policy_from_net
from .network import PolicyNet


def _equity_vs_random(state: GameState, seat: int,
                       samples: int, rng: random.Random) -> float:
    """Monte-Carlo showdown equity vs a uniform random un-blocked opp hand.

    Runs out the remaining board with a sampled opp hole if pre-river.
    Returns win probability in [0, 1] (counting ties as half).
    """
    hole = list(state.hole_cards[seat])
    board = list(state.board)
    blocked = set(hole) | set(board)
    deck = [c for c in range(CARD_COUNT) if c not in blocked]
    needed_board = 5 - len(board)
    wins = 0.0
    for _ in range(samples):
        # sample without replacement: 2 opp + (5 - len(board)) extra board cards.
        picks = rng.sample(deck, 2 + needed_board)
        opp = picks[:2]
        extra = picks[2:]
        full_board = board + extra
        my_rank = evaluate_seven(hole + full_board)
        op_rank = evaluate_seven(opp + full_board)
        if my_rank > op_rank:
            wins += 1.0
        elif my_rank == op_rank:
            wins += 0.5
    return wins / max(1, samples)


def lbr_action(
    state: GameState,
    seat: int,
    rng: random.Random,
    *,
    equity_samples: int = 100,
) -> int:
    """LBR-CC: pick fold vs call/check via expected-value comparison.

    EV(fold)        = -committed[seat]                  (lose what's already in)
    EV(call/check)  = equity * pot_after - committed_after
    """
    mask = legal_action_mask(state)
    cc = check_call_id()
    fd = fold_id()
    # If only one of the two basic actions is legal, take it.
    can_call = bool(mask[cc])
    can_fold = bool(mask[fd])
    if not can_fold and can_call:
        return cc
    if not can_call and can_fold:
        return fd

    to_call = state.call_amount(seat)
    committed_after = state.committed[seat] + to_call
    pot_after = state.pot + to_call

    eq = _equity_vs_random(state, seat, equity_samples, rng)
    ev_call = eq * pot_after - committed_after
    ev_fold = -state.committed[seat]
    return cc if ev_call >= ev_fold else fd


def evaluate_lbr(
    net: PolicyNet,
    cfg: DeepCFRConfig,
    device: torch.device,
    *,
    num_hands: int = 2000,
    equity_samples: int = 100,
    rng: Optional[random.Random] = None,
) -> float:
    """Run LBR vs the trained policy. Returns mbb/g won by LBR.

    The trained policy plays both seats in turn (button-rotation averaging).
    LBR is restricted to fold/call/check (LBR-CC variant).
    """
    if cfg.num_players != 2:
        return float("nan")
    if rng is None:
        rng = random.Random(0xCAFE)
    space = ActionSpace(cfg.bet_fractions)
    trained = policy_from_net(
        net,
        device,
        deterministic=False,
        action_space=space,
        temperature=cfg.policy_temperature,
        bet_multiplier=cfg.policy_bet_multiplier,
        all_in_multiplier=cfg.policy_all_in_multiplier,
    )

    def _play(lbr_seat: int, hand_i: int) -> float:
        button = hand_i % 2
        state = new_hand(
            num_players=2,
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
            if seat == lbr_seat:
                a = lbr_action(state, seat, rng, equity_samples=equity_samples)
            else:
                a = trained(state, seat, rng)
            mask = legal_action_mask(state)
            if not mask[a]:
                a = next(i for i, m in enumerate(mask) if m)
            state = apply_action(state, a)
            safety += 1
        return float(payoffs(state)[lbr_seat])

    # Average across seat assignments to remove positional bias.
    total = 0.0
    n_each = num_hands // 2
    for i in range(n_each):
        total += _play(0, i)
    for i in range(n_each):
        total += _play(1, i)
    chips_per_hand = total / max(1, 2 * n_each)
    return chips_per_hand * 1000.0 / cfg.big_blind
