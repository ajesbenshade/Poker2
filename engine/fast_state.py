"""Fast mutable poker game state for high-volume Deep CFR traversals.

This module provides a mutable alternative to the immutable GameState in
engine/state.py. It is intended for use *only* inside the Deep CFR traversal
hot paths (vectorized_traversal.py and traversal.py).

Key design:
- FastGameState has identical public attribute names to GameState.
- Existing pure functions (encode_observation, legal_action_mask, is_terminal,
  payoffs) work unchanged on FastGameState instances.
- We use a simple full-state snapshot + restore model for backtracking.
  This is dramatically cheaper than repeated dataclass + list() allocations
  while remaining trivial to keep 100% semantically identical to the slow path.
- Public immutable API (engine.GameState, apply_action, etc.) is untouched.

Typical usage in a recursive traversal:

    state = new_fast_hand(...)
    snap = apply_action_in_place(state, action_id)
    value = recurse(...)
    restore_state(state, snap)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .actions import ActionSpace, DEFAULT_BET_FRACTIONS
from .cards import deal_deck
from .state import (
    GameState,
    Stage,
    _BOARD_SIZE,
    _first_to_act,
    _no_more_betting,
    _round_complete,
)


# =============================================================================
# Fast (mutable) Game State
# =============================================================================

@dataclass
class FastGameState:
    """Mutable equivalent of GameState for solver hot paths.

    All attribute names and types are deliberately kept identical to GameState
    so that the rest of the engine (encoder, legal mask, payoffs, etc.) works
    without modification.
    """
    num_players: int
    button: int
    small_blind: int
    big_blind: int
    starting_stack: int
    stacks: List[int]
    committed: List[int]
    street_committed: List[int]
    folded: List[bool]
    all_in: List[bool]
    acted_this_street: List[bool]
    hole_cards: List[Tuple[int, int]]
    board: List[int]
    deck_pos: int
    deck: List[int]
    stage: Stage
    to_act: int
    current_bet: int
    last_raise_size: int
    aggressor: int
    history: List[Tuple[int, int, int]]
    action_space: ActionSpace = field(default_factory=ActionSpace)

    # --- compatibility helpers (same as slow path) ---
    @property
    def pot(self) -> int:
        return sum(self.committed)

    def call_amount(self, seat: int) -> int:
        return max(0, self.current_bet - self.street_committed[seat])


# =============================================================================
# Snapshot / Restore (the key to cheap backtracking)
# =============================================================================

def snapshot_state(fs: FastGameState) -> Dict[str, Any]:
    """Capture a cheap snapshot of everything we mutate during apply_action.

    Because max 9 players, copying a few small lists is very inexpensive
    compared with repeated dataclass construction + recursive object creation.
    """
    return {
        "to_act": fs.to_act,
        "current_bet": fs.current_bet,
        "last_raise_size": fs.last_raise_size,
        "stage": fs.stage,
        "aggressor": fs.aggressor,
        "deck_pos": fs.deck_pos,
        "history_len": len(fs.history),
        "board_len": len(fs.board),
        "stacks": list(fs.stacks),
        "committed": list(fs.committed),
        "street_committed": list(fs.street_committed),
        "folded": list(fs.folded),
        "all_in": list(fs.all_in),
        "acted_this_street": list(fs.acted_this_street),
    }


def restore_state(fs: FastGameState, snap: Dict[str, Any]) -> None:
    """Restore state from a snapshot produced by snapshot_state()."""
    fs.to_act = snap["to_act"]
    fs.current_bet = snap["current_bet"]
    fs.last_raise_size = snap["last_raise_size"]
    fs.stage = snap["stage"]
    fs.aggressor = snap["aggressor"]
    fs.deck_pos = snap["deck_pos"]

    # Truncate variable-length history/board
    del fs.history[snap["history_len"]:]
    del fs.board[snap["board_len"]:]

    # Restore per-player arrays
    fs.stacks[:] = snap["stacks"]
    fs.committed[:] = snap["committed"]
    fs.street_committed[:] = snap["street_committed"]
    fs.folded[:] = snap["folded"]
    fs.all_in[:] = snap["all_in"]
    fs.acted_this_street[:] = snap["acted_this_street"]


# =============================================================================
# Construction
# =============================================================================

def new_fast_hand(
    num_players: int = 2,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    button: int = 0,
    rng: Optional[random.Random] = None,
    action_space: Optional[ActionSpace] = None,
    deck: Optional[Sequence[int]] = None,
) -> FastGameState:
    """Fast-path equivalent of engine.state.new_hand."""
    if num_players < 2 or num_players > 9:
        raise ValueError("num_players must be in [2, 9]")
    if rng is None:
        rng = random.Random()
    if action_space is None:
        action_space = ActionSpace(DEFAULT_BET_FRACTIONS)

    full_deck = list(deck) if deck is not None else deal_deck(rng)
    if len(full_deck) < 2 * num_players + 5:
        raise ValueError("deck too short")

    # Deal hole cards (identical logic)
    holes: List[Tuple[int, int]] = [(0, 0)] * num_players
    pos = 0
    for round_ in range(2):
        for offset in range(num_players):
            seat = (button + 1 + offset) % num_players
            c = full_deck[pos]
            pos += 1
            holes[seat] = (c, holes[seat][1]) if round_ == 0 else (holes[seat][0], c)

    # Blinds
    if num_players == 2:
        sb_seat = button
        bb_seat = (button + 1) % 2
    else:
        sb_seat = (button + 1) % num_players
        bb_seat = (button + 2) % num_players

    stacks = [starting_stack] * num_players
    committed = [0] * num_players
    street_committed = [0] * num_players
    all_in = [False] * num_players

    def _post(seat: int, amount: int) -> None:
        amt = min(amount, stacks[seat])
        stacks[seat] -= amt
        committed[seat] += amt
        street_committed[seat] += amt
        if stacks[seat] == 0:
            all_in[seat] = True

    _post(sb_seat, small_blind)
    _post(bb_seat, big_blind)

    return FastGameState(
        num_players=num_players,
        button=button,
        small_blind=small_blind,
        big_blind=big_blind,
        starting_stack=starting_stack,
        stacks=stacks,
        committed=committed,
        street_committed=street_committed,
        folded=[False] * num_players,
        all_in=all_in,
        acted_this_street=[False] * num_players,
        hole_cards=holes,
        board=[],
        deck_pos=pos,
        deck=full_deck,
        stage=Stage.PREFLOP,
        to_act=_first_to_act(num_players, button, Stage.PREFLOP, all_in, [False] * num_players),
        current_bet=big_blind,
        last_raise_size=big_blind,
        aggressor=bb_seat,
        history=[],
        action_space=action_space,
    )


# =============================================================================
# In-place transition (the hot path)
# =============================================================================

def apply_action_in_place(fs: FastGameState, action_id: int) -> Dict[str, Any]:
    """Mutate fs in place and return a snapshot for later restore.

    This is the core primitive that eliminates the vast majority of allocations
    in the traversal hot loop. It mirrors the logic in engine/state.py:apply_action
    exactly (including side effects from _advance_after_action).
    """
    from .state import is_terminal, legal_action_mask  # use the authoritative versions

    if is_terminal(fs):
        raise RuntimeError("cannot act on a terminal state")

    mask = legal_action_mask(fs)
    if not mask[action_id]:
        space = fs.action_space
        raise ValueError(
            f"illegal action {action_id} ({space.name(action_id)}) "
            f"at seat {fs.to_act}; legal={mask}"
        )

    # Take snapshot BEFORE any mutation
    snap = snapshot_state(fs)

    seat = fs.to_act
    space = fs.action_space
    chips_added = 0

    # --- Exact mutation logic copied from engine/state.py:apply_action ---
    if action_id == space.fold_id:
        fs.folded[seat] = True

    elif action_id == space.check_call_id:
        to_call = fs.call_amount(seat)
        chips = min(to_call, fs.stacks[seat])
        if chips > 0:
            fs.stacks[seat] -= chips
            fs.committed[seat] += chips
            fs.street_committed[seat] += chips
            chips_added = chips
            if fs.stacks[seat] == 0:
                fs.all_in[seat] = True

    elif action_id == space.all_in_id:
        chips = fs.stacks[seat]
        fs.stacks[seat] = 0
        fs.committed[seat] += chips
        fs.street_committed[seat] += chips
        fs.all_in[seat] = True
        chips_added = chips
        new_total = fs.street_committed[seat]
        if new_total > fs.current_bet:
            raise_size = new_total - fs.current_bet
            fs.last_raise_size = max(fs.last_raise_size, raise_size)
            fs.current_bet = new_total
            fs.aggressor = seat
            for i in range(fs.num_players):
                if i != seat and not fs.folded[i] and not fs.all_in[i]:
                    fs.acted_this_street[i] = False

    else:  # sized bet/raise
        frac = space.bet_fraction(action_id)
        pot_after_call = fs.pot + fs.call_amount(seat)
        target_total = fs.current_bet + int(round(frac * pot_after_call))
        chips = target_total - fs.street_committed[seat]
        chips = min(chips, fs.stacks[seat])
        fs.stacks[seat] -= chips
        fs.committed[seat] += chips
        fs.street_committed[seat] += chips
        chips_added = chips
        if fs.stacks[seat] == 0:
            fs.all_in[seat] = True
        new_total = fs.street_committed[seat]
        raise_size = new_total - fs.current_bet
        fs.last_raise_size = raise_size
        fs.current_bet = new_total
        fs.aggressor = seat
        for i in range(fs.num_players):
            if i != seat and not fs.folded[i] and not fs.all_in[i]:
                fs.acted_this_street[i] = False

    fs.acted_this_street[seat] = True
    fs.history.append((seat, action_id, chips_added))

    # This call contains all the complex street-advance / board-dealing logic
    from .state import _advance_after_action
    _advance_after_action(fs)

    return snap


# =============================================================================
# Convenience: convert between slow and fast representations
# =============================================================================

def fast_to_slow(fs: FastGameState) -> GameState:
    """Convert a FastGameState into an immutable GameState (for eval/LBR/etc.)."""
    return GameState(
        num_players=fs.num_players,
        button=fs.button,
        small_blind=fs.small_blind,
        big_blind=fs.big_blind,
        starting_stack=fs.starting_stack,
        stacks=list(fs.stacks),
        committed=list(fs.committed),
        street_committed=list(fs.street_committed),
        folded=list(fs.folded),
        all_in=list(fs.all_in),
        acted_this_street=list(fs.acted_this_street),
        hole_cards=list(fs.hole_cards),
        board=list(fs.board),
        deck_pos=fs.deck_pos,
        deck=list(fs.deck),
        stage=fs.stage,
        to_act=fs.to_act,
        current_bet=fs.current_bet,
        last_raise_size=fs.last_raise_size,
        aggressor=fs.aggressor,
        history=list(fs.history),
        action_space=fs.action_space,
    )


def slow_to_fast(gs: GameState) -> FastGameState:
    """Convert an immutable GameState into a mutable FastGameState."""
    return FastGameState(
        num_players=gs.num_players,
        button=gs.button,
        small_blind=gs.small_blind,
        big_blind=gs.big_blind,
        starting_stack=gs.starting_stack,
        stacks=list(gs.stacks),
        committed=list(gs.committed),
        street_committed=list(gs.street_committed),
        folded=list(gs.folded),
        all_in=list(gs.all_in),
        acted_this_street=list(gs.acted_this_street),
        hole_cards=list(gs.hole_cards),
        board=list(gs.board),
        deck_pos=gs.deck_pos,
        deck=list(gs.deck),
        stage=gs.stage,
        to_act=gs.to_act,
        current_bet=gs.current_bet,
        last_raise_size=gs.last_raise_size,
        aggressor=gs.aggressor,
        history=list(gs.history),
        action_space=gs.action_space,
    )
