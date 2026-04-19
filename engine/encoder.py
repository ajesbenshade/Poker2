"""Dense observation encoder for the policy network.

Produces a fixed-length ``float32`` vector for the seat that is currently to act
(or any seat passed via ``perspective_seat``). The encoding is dense, fully
informative for a heads-up policy and supports up to ``MAX_PLAYERS`` seats
without changing dimensionality \u2014 this lets the same network be reused as we
scale from HU to 6-max.

Feature blocks (in order):

  hole_cards            52 dims (multi-hot, perspective seat only)
  board_cards           52 dims (multi-hot, public)
  stage                  4 dims (preflop / flop / turn / river one-hot)
  position_offset        MAX_PLAYERS dims (one-hot: seats clockwise from button)
  per-seat block         6 * MAX_PLAYERS dims
                         [stack/start, total_committed/start,
                          street_committed/start, folded, all_in, is_perspective]
  global scalars         9 dims
                         [pot/start, current_bet/start, to_call/start,
                          last_raise/start, pot_odds, spr_log,
                          n_active/MAX, n_all_in/MAX, num_players/MAX]
  history scalars        4 dims
                         [n_actions_this_hand/32, n_aggressive_this_street/8,
                          stage_progress, is_facing_bet]

Total = 52 + 52 + 4 + 9 + 54 + 9 + 4 = **184 dims**.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .actions import ActionSpace
from .cards import CARD_COUNT
from .state import GameState, Stage, is_terminal

MAX_PLAYERS = 9

_HOLE_OFFSET = 0
_BOARD_OFFSET = _HOLE_OFFSET + CARD_COUNT
_STAGE_OFFSET = _BOARD_OFFSET + CARD_COUNT
_POSITION_OFFSET = _STAGE_OFFSET + 4
_PER_SEAT_OFFSET = _POSITION_OFFSET + MAX_PLAYERS
_PER_SEAT_DIMS = 6
_GLOBAL_OFFSET = _PER_SEAT_OFFSET + _PER_SEAT_DIMS * MAX_PLAYERS
_GLOBAL_DIMS = 9
_HISTORY_OFFSET = _GLOBAL_OFFSET + _GLOBAL_DIMS
_HISTORY_DIMS = 4
OBS_DIM = _HISTORY_OFFSET + _HISTORY_DIMS


def encoder_feature_layout() -> List[Tuple[str, int, int]]:
    """Return ``(name, start, length)`` triples documenting the layout."""
    return [
        ("hole_cards", _HOLE_OFFSET, CARD_COUNT),
        ("board_cards", _BOARD_OFFSET, CARD_COUNT),
        ("stage", _STAGE_OFFSET, 4),
        ("position_offset", _POSITION_OFFSET, MAX_PLAYERS),
        ("per_seat", _PER_SEAT_OFFSET, _PER_SEAT_DIMS * MAX_PLAYERS),
        ("global", _GLOBAL_OFFSET, _GLOBAL_DIMS),
        ("history", _HISTORY_OFFSET, _HISTORY_DIMS),
    ]


def encode_observation(
    state: GameState,
    perspective_seat: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Encode ``state`` from the point of view of ``perspective_seat``.

    If ``perspective_seat`` is None, ``state.to_act`` is used. Hole cards are
    only revealed for the perspective seat (private information).
    """
    if perspective_seat is None:
        perspective_seat = state.to_act
    if not 0 <= perspective_seat < state.num_players:
        raise ValueError(f"bad perspective_seat {perspective_seat}")

    if out is None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
    else:
        if out.shape != (OBS_DIM,) or out.dtype != np.float32:
            raise ValueError(f"out must be float32 shape ({OBS_DIM},)")
        obs = out
        obs.fill(0.0)

    # Hole cards (perspective only)
    h0, h1 = state.hole_cards[perspective_seat]
    obs[_HOLE_OFFSET + int(h0)] = 1.0
    obs[_HOLE_OFFSET + int(h1)] = 1.0

    # Board cards
    for c in state.board:
        obs[_BOARD_OFFSET + int(c)] = 1.0

    # Stage one-hot (DONE collapses to river)
    stage_idx = min(int(state.stage), int(Stage.RIVER))
    obs[_STAGE_OFFSET + stage_idx] = 1.0

    # Position offset clockwise from the button
    pos_offset = (perspective_seat - state.button) % state.num_players
    obs[_POSITION_OFFSET + pos_offset] = 1.0

    start = max(1, state.starting_stack)
    inv_start = 1.0 / start

    # Per-seat block (rotated so perspective is index 0)
    for slot in range(state.num_players):
        seat = (perspective_seat + slot) % state.num_players
        base = _PER_SEAT_OFFSET + slot * _PER_SEAT_DIMS
        obs[base + 0] = state.stacks[seat] * inv_start
        obs[base + 1] = state.committed[seat] * inv_start
        obs[base + 2] = state.street_committed[seat] * inv_start
        obs[base + 3] = 1.0 if state.folded[seat] else 0.0
        obs[base + 4] = 1.0 if state.all_in[seat] else 0.0
        obs[base + 5] = 1.0 if seat == perspective_seat else 0.0

    # Global scalars
    pot = state.pot
    to_call = state.call_amount(perspective_seat)
    n_active = sum(1 for f in state.folded if not f)
    n_all_in = sum(1 for a in state.all_in if a)
    eff_stack = max(1, min(state.stacks[s] + state.street_committed[s]
                           for s in range(state.num_players)
                           if not state.folded[s]))

    obs[_GLOBAL_OFFSET + 0] = pot * inv_start
    obs[_GLOBAL_OFFSET + 1] = state.current_bet * inv_start
    obs[_GLOBAL_OFFSET + 2] = to_call * inv_start
    obs[_GLOBAL_OFFSET + 3] = state.last_raise_size * inv_start
    obs[_GLOBAL_OFFSET + 4] = (to_call / (pot + to_call)) if (pot + to_call) > 0 else 0.0
    obs[_GLOBAL_OFFSET + 5] = math.log1p(eff_stack / max(1, pot)) / 5.0
    obs[_GLOBAL_OFFSET + 6] = n_active / MAX_PLAYERS
    obs[_GLOBAL_OFFSET + 7] = n_all_in / MAX_PLAYERS
    obs[_GLOBAL_OFFSET + 8] = state.num_players / MAX_PLAYERS

    # History scalars
    n_actions = len(state.history)
    n_aggressive_street = sum(
        1 for (_, aid, _) in state.history
        if aid >= 2  # bet/raise/all-in
    )
    obs[_HISTORY_OFFSET + 0] = min(1.0, n_actions / 32.0)
    obs[_HISTORY_OFFSET + 1] = min(1.0, n_aggressive_street / 8.0)
    obs[_HISTORY_OFFSET + 2] = stage_idx / float(Stage.RIVER)
    obs[_HISTORY_OFFSET + 3] = 1.0 if to_call > 0 else 0.0

    return obs


def encode_legality(state: GameState, action_space: Optional[ActionSpace] = None) -> np.ndarray:
    """Float32 mask (1.0 legal / 0.0 illegal) over the action vocabulary."""
    from .state import legal_action_mask
    space = action_space or state.action_space
    if is_terminal(state):
        return np.zeros(space.num_actions, dtype=np.float32)
    return np.asarray(legal_action_mask(state), dtype=np.float32)
