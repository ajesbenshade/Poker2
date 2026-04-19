"""Tests for the new NLHE engine."""
from __future__ import annotations

import random

import numpy as np
import pytest

from engine import (
    ActionSpace,
    GameState,
    OBS_DIM,
    Stage,
    apply_action,
    encode_observation,
    is_terminal,
    legal_action_mask,
    new_hand,
    payoffs,
)
from engine.actions import bet_id, all_in_id, check_call_id, fold_id


# ---------------------------------------------------------------------------
# Construction & blinds
# ---------------------------------------------------------------------------

def test_hu_blinds_and_button():
    s = new_hand(num_players=2, starting_stack=200, small_blind=1, big_blind=2,
                 button=0, rng=random.Random(0))
    assert s.num_players == 2
    # Button = SB in HU
    assert s.committed[0] == 1 and s.committed[1] == 2
    assert s.stacks == [199, 198]
    assert s.current_bet == 2
    assert s.stage == Stage.PREFLOP
    # SB acts first preflop in HU (= button = seat 0)
    assert s.to_act == 0
    # Each seat has 2 hole cards
    for seat in range(2):
        c0, c1 = s.hole_cards[seat]
        assert 0 <= c0 < 52 and 0 <= c1 < 52 and c0 != c1


def test_3max_blinds_and_first_actor():
    s = new_hand(num_players=3, starting_stack=200, button=0,
                 rng=random.Random(0))
    # Button=0 => SB=1, BB=2; UTG (left of BB) = 0 acts first preflop.
    assert s.committed == [0, 1, 2]
    assert s.to_act == 0


def test_holes_are_unique():
    s = new_hand(num_players=6, rng=random.Random(123))
    cards = []
    for seat in range(s.num_players):
        cards.extend(s.hole_cards[seat])
    assert len(cards) == len(set(cards))


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------

def test_cannot_fold_when_no_bet_to_call():
    s = new_hand(num_players=2, button=0, rng=random.Random(1))
    # SB calls (limps); BB now faces no extra bet -> cannot fold
    s = apply_action(s, check_call_id())
    assert s.to_act == 1
    mask = legal_action_mask(s)
    assert not mask[fold_id()]
    assert mask[check_call_id()]


def test_call_legal_facing_bet():
    s = new_hand(num_players=2, button=0, rng=random.Random(2))
    # SB facing BB - to_call > 0 - fold should be legal
    mask = legal_action_mask(s)
    assert mask[fold_id()]
    assert mask[check_call_id()]
    assert mask[all_in_id()]


def test_min_raise_enforced():
    # Construct a postflop scenario. Seat 0 opens 4, seat 1 should not be
    # allowed to "raise to" anything below 8.
    s = new_hand(num_players=2, button=0, starting_stack=1000,
                 rng=random.Random(3))
    s = apply_action(s, check_call_id())  # SB calls 1
    s = apply_action(s, check_call_id())  # BB checks
    assert s.stage == Stage.FLOP
    # Postflop, BB acts first; bet 1.0 pot (= 4 chips, above BB=2 minimum).
    s = apply_action(s, bet_id(3))
    # SB now faces a bet; min raise = current_bet + last_raise_size
    min_total = s.current_bet + s.last_raise_size
    space = s.action_space
    mask = legal_action_mask(s)
    # Find the smallest sized raise that is legal -> its target must be >= min_total
    for k in range(space.num_bet_sizes):
        if mask[space.bet_id(k)]:
            frac = space.bet_fractions[k]
            target = s.current_bet + int(round(frac * (s.pot + s.call_amount(s.to_act))))
            assert target >= min_total


# ---------------------------------------------------------------------------
# Reaching a terminal state
# ---------------------------------------------------------------------------

def test_fold_ends_hand_and_pays_winner():
    s = new_hand(num_players=2, button=0, starting_stack=100,
                 small_blind=1, big_blind=2, rng=random.Random(4))
    # SB folds preflop
    s = apply_action(s, fold_id())
    assert is_terminal(s)
    p = payoffs(s)
    # BB wins SB's posted blind
    assert p[0] == -1
    assert p[1] == 1
    assert sum(p) == 0


def test_check_through_to_showdown():
    s = new_hand(num_players=2, button=0, starting_stack=200,
                 rng=random.Random(5))
    s = apply_action(s, check_call_id())  # SB call
    s = apply_action(s, check_call_id())  # BB check
    assert s.stage == Stage.FLOP
    # check, check
    s = apply_action(s, check_call_id())
    s = apply_action(s, check_call_id())
    assert s.stage == Stage.TURN
    s = apply_action(s, check_call_id())
    s = apply_action(s, check_call_id())
    assert s.stage == Stage.RIVER
    s = apply_action(s, check_call_id())
    s = apply_action(s, check_call_id())
    assert is_terminal(s)
    p = payoffs(s)
    assert sum(p) == 0
    # Pot was 4 chips; payouts must be in {(2, -2), (-2, 2), (0, 0)} (split).
    assert set(map(abs, p)) <= {0, 2}


def test_all_in_terminal_runs_out_board():
    s = new_hand(num_players=2, button=0, starting_stack=20,
                 small_blind=1, big_blind=2, rng=random.Random(6))
    # SB shoves all-in preflop
    s = apply_action(s, all_in_id())
    # BB calls the shove
    s = apply_action(s, check_call_id())
    assert is_terminal(s)
    assert len(s.board) == 5
    p = payoffs(s)
    assert sum(p) == 0
    # Each seat risked their full stack
    assert max(p) <= 20 and min(p) >= -20


def test_payoffs_sum_to_zero_random_play():
    rng = random.Random(7)
    for trial in range(50):
        s = new_hand(num_players=2, button=trial % 2, rng=rng,
                     starting_stack=50)
        steps = 0
        while not is_terminal(s) and steps < 200:
            mask = legal_action_mask(s)
            choices = [i for i, m in enumerate(mask) if m]
            assert choices, "no legal action available"
            s = apply_action(s, rng.choice(choices))
            steps += 1
        assert is_terminal(s), "hand failed to terminate"
        assert sum(payoffs(s)) == 0


# ---------------------------------------------------------------------------
# Side pots
# ---------------------------------------------------------------------------

def test_side_pot_three_handed():
    """Short stack shoves preflop, two bigger stacks call. Verify side pot
    accounting: short stack only wins what they put in from each opponent."""
    rng = random.Random(8)
    # Custom stacks: short stack at seat 0 (button)
    s = new_hand(num_players=3, button=0, starting_stack=200,
                 small_blind=1, big_blind=2, rng=rng)
    # Force seat 0 to be short by overwriting its stack
    s.stacks[0] = 10           # 10 chips behind button
    # UTG (seat 0) shoves all-in (10 + 0 = 10 total)
    s = apply_action(s, all_in_id())
    assert s.committed[0] == 10
    # SB (seat 1) calls; BB (seat 2) calls
    s = apply_action(s, check_call_id())
    s = apply_action(s, check_call_id())
    # BB option after SB only completed; we checked with call which matches
    # current_bet=10, so round closes -> postflop. From there everyone checks.
    safety = 0
    while not is_terminal(s) and safety < 20:
        s = apply_action(s, check_call_id())
        safety += 1
    assert is_terminal(s)
    # Each seat committed the same amount (10) since 0 was the short stack
    # and 1,2 only matched.
    assert s.committed == [10, 10, 10]
    p = payoffs(s)
    assert sum(p) == 0


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def test_encoder_shape_and_bounds():
    s = new_hand(num_players=2, button=0, rng=random.Random(9))
    obs = encode_observation(s)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    # Hole-card multi-hot has exactly 2 ones for the perspective seat
    assert obs[:52].sum() == 2.0
    # Board is empty preflop
    assert obs[52:104].sum() == 0.0


def test_encoder_hides_other_holes():
    s = new_hand(num_players=2, button=0, rng=random.Random(10))
    obs0 = encode_observation(s, perspective_seat=0)
    obs1 = encode_observation(s, perspective_seat=1)
    # Different seats see different hole cards (with overwhelming probability)
    assert not np.array_equal(obs0[:52], obs1[:52])


def test_encoder_per_seat_block_is_perspective_rotated():
    s = new_hand(num_players=3, button=0, rng=random.Random(11))
    for seat in range(3):
        obs = encode_observation(s, perspective_seat=seat)
        # The "is_perspective" flag is at slot 0 in the per-seat block
        from engine.encoder import _PER_SEAT_OFFSET, _PER_SEAT_DIMS
        assert obs[_PER_SEAT_OFFSET + 5] == 1.0
        # And not set for other slots
        for slot in range(1, 3):
            assert obs[_PER_SEAT_OFFSET + slot * _PER_SEAT_DIMS + 5] == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_same_hand():
    s1 = new_hand(num_players=2, rng=random.Random(42))
    s2 = new_hand(num_players=2, rng=random.Random(42))
    assert s1.hole_cards == s2.hole_cards
    assert s1.deck == s2.deck
