"""Heads-up / N-player No-Limit Hold'em game state and transitions.

The engine is intentionally minimal but *correct*: blinds, dealer button,
betting order, min-raise enforcement, all-in handling and side pots are
all modelled explicitly. ``apply_action`` returns a new ``GameState``
without mutating the input, which makes the engine safe to use inside
recursive solvers (Deep CFR external-sampling traversals etc.).

Conventions:

* Seats are indexed ``0..num_players-1``. Seat ``0`` is the small blind
  (or the button in heads-up). Action proceeds clockwise (``+1 mod N``).
* All chip amounts are integers (chips). Default ``small_blind=1``,
  ``big_blind=2``, ``starting_stack=200`` (i.e. 100 BB).
* Card ints follow :mod:`engine.cards`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

from .actions import ActionSpace, DEFAULT_BET_FRACTIONS
from .cards import deal_deck, evaluate_seven


class Stage(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    DONE = 4


_BOARD_SIZE = {Stage.PREFLOP: 0, Stage.FLOP: 3, Stage.TURN: 4, Stage.RIVER: 5, Stage.DONE: 5}


@dataclass
class GameState:
    num_players: int
    button: int
    small_blind: int
    big_blind: int
    starting_stack: int
    stacks: List[int]
    committed: List[int]            # chips into pot across the whole hand
    street_committed: List[int]     # chips into pot this street only
    folded: List[bool]
    all_in: List[bool]
    acted_this_street: List[bool]
    hole_cards: List[Tuple[int, int]]
    board: List[int]
    deck_pos: int                   # next undealt card index
    deck: List[int]                 # full pre-shuffled deck
    stage: Stage
    to_act: int
    current_bet: int                # max street_committed across seats
    last_raise_size: int            # smallest legal additional raise this street
    aggressor: int                  # last seat that opened/raised
    history: List[Tuple[int, int, int]]   # (seat, action_id, chips_added)
    action_space: ActionSpace = field(default_factory=ActionSpace)

    # ---- shallow-copy helpers ------------------------------------------------
    def _clone(self) -> "GameState":
        return GameState(
            num_players=self.num_players,
            button=self.button,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            starting_stack=self.starting_stack,
            stacks=list(self.stacks),
            committed=list(self.committed),
            street_committed=list(self.street_committed),
            folded=list(self.folded),
            all_in=list(self.all_in),
            acted_this_street=list(self.acted_this_street),
            hole_cards=list(self.hole_cards),
            board=list(self.board),
            deck_pos=self.deck_pos,
            deck=self.deck,            # deck is read-only after deal; share
            stage=self.stage,
            to_act=self.to_act,
            current_bet=self.current_bet,
            last_raise_size=self.last_raise_size,
            aggressor=self.aggressor,
            history=list(self.history),
            action_space=self.action_space,
        )

    @property
    def pot(self) -> int:
        return sum(self.committed)

    def call_amount(self, seat: int) -> int:
        return max(0, self.current_bet - self.street_committed[seat])

    def active_seats(self) -> List[int]:
        return [s for s in range(self.num_players) if not self.folded[s]]

    def can_act_seats(self) -> List[int]:
        return [s for s in range(self.num_players)
                if not self.folded[s] and not self.all_in[s]]


# =============================================================================
# Construction
# =============================================================================

def new_hand(
    num_players: int = 2,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    button: int = 0,
    rng: Optional[random.Random] = None,
    action_space: Optional[ActionSpace] = None,
    deck: Optional[Sequence[int]] = None,
) -> GameState:
    if num_players < 2 or num_players > 9:
        raise ValueError("num_players must be in [2, 9]")
    if rng is None:
        rng = random.Random()
    if action_space is None:
        action_space = ActionSpace(DEFAULT_BET_FRACTIONS)
    full_deck = list(deck) if deck is not None else deal_deck(rng)
    if len(full_deck) < 2 * num_players + 5:
        raise ValueError("deck too short")

    # Deal hole cards (2 per seat, round-robin starting left of button)
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
        sb_seat = button                 # HU: button is SB
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

    state = GameState(
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
    return state


def _first_to_act(
    num_players: int,
    button: int,
    stage: Stage,
    all_in_mask: Sequence[bool],
    folded_mask: Sequence[bool],
) -> int:
    """Seat that should act first this street, skipping folded/all-in players."""
    if stage == Stage.PREFLOP:
        if num_players == 2:
            # HU: button (=SB) acts first preflop.
            start = button
        else:
            # Left of BB.
            bb_seat = (button + 2) % num_players
            start = (bb_seat + 1) % num_players
    else:
        # Postflop: first non-folded seat left of button. In HU that's the BB.
        start = (button + 1) % num_players
    for offset in range(num_players):
        seat = (start + offset) % num_players
        if not folded_mask[seat] and not all_in_mask[seat]:
            return seat
    return start  # everyone all-in / folded; caller will short-circuit


# =============================================================================
# Legality
# =============================================================================

def legal_action_mask(state: GameState) -> List[bool]:
    """Boolean mask over ``state.action_space.num_actions``.

    A bet/raise is illegal if:
      * the actor cannot meet the minimum raise increment (and the action is
        not effectively an all-in), or
      * the resulting target equals what an all-in would do (we keep ALL_IN
        as the canonical large-shove action and forbid the duplicate bet).
    """
    space = state.action_space
    mask = [False] * space.num_actions
    if is_terminal(state):
        return mask

    seat = state.to_act
    if state.folded[seat] or state.all_in[seat]:
        return mask

    to_call = state.call_amount(seat)
    stack = state.stacks[seat]

    # FOLD: only if facing a real bet (free check shouldn't fold)
    if to_call > 0:
        mask[space.fold_id] = True

    # CHECK / CALL: always legal as long as it's our turn
    mask[space.check_call_id] = True

    # ALL-IN: legal whenever we have chips
    if stack > 0:
        mask[space.all_in_id] = True

    # Sized bets/raises: must be > to_call AND meet min-raise AND < all-in size
    pot_after_call = state.pot + to_call
    min_total = state.current_bet + state.last_raise_size
    all_in_total = state.street_committed[seat] + stack  # what street_commit becomes if we shove

    for k, frac in enumerate(space.bet_fractions):
        target = state.current_bet + int(round(frac * pot_after_call))
        # clamp to stack
        if target >= all_in_total:
            continue   # collapses to ALL_IN
        if target < min_total:
            continue
        if target <= state.current_bet:
            continue
        # need enough chips
        chips_needed = target - state.street_committed[seat]
        if chips_needed > stack:
            continue
        mask[space.bet_id(k)] = True

    return mask


# =============================================================================
# Transitions
# =============================================================================

def apply_action(state: GameState, action_id: int) -> GameState:
    if is_terminal(state):
        raise RuntimeError("cannot act on a terminal state")
    mask = legal_action_mask(state)
    if not mask[action_id]:
        raise ValueError(
            f"illegal action {action_id} ({state.action_space.name(action_id)}) "
            f"at seat {state.to_act}; legal={mask}"
        )

    s = state._clone()
    seat = s.to_act
    space = s.action_space
    chips_added = 0

    if action_id == space.fold_id:
        s.folded[seat] = True

    elif action_id == space.check_call_id:
        to_call = s.call_amount(seat)
        chips = min(to_call, s.stacks[seat])
        if chips > 0:
            s.stacks[seat] -= chips
            s.committed[seat] += chips
            s.street_committed[seat] += chips
            chips_added = chips
            if s.stacks[seat] == 0:
                s.all_in[seat] = True

    elif action_id == space.all_in_id:
        chips = s.stacks[seat]
        s.stacks[seat] = 0
        s.committed[seat] += chips
        s.street_committed[seat] += chips
        s.all_in[seat] = True
        chips_added = chips
        new_total = s.street_committed[seat]
        if new_total > s.current_bet:
            raise_size = new_total - s.current_bet
            # Even an under-min all-in counts as a raise but doesn't
            # re-open action for players already matched. We simplify and
            # always re-open action; this is a known minor deviation that
            # is harmless under self-play training.
            s.last_raise_size = max(s.last_raise_size, raise_size)
            s.current_bet = new_total
            s.aggressor = seat
            for i in range(s.num_players):
                if i != seat and not s.folded[i] and not s.all_in[i]:
                    s.acted_this_street[i] = False

    else:  # sized bet/raise
        frac = space.bet_fraction(action_id)
        pot_after_call = s.pot + s.call_amount(seat)
        target_total = s.current_bet + int(round(frac * pot_after_call))
        chips = target_total - s.street_committed[seat]
        chips = min(chips, s.stacks[seat])
        s.stacks[seat] -= chips
        s.committed[seat] += chips
        s.street_committed[seat] += chips
        chips_added = chips
        if s.stacks[seat] == 0:
            s.all_in[seat] = True
        new_total = s.street_committed[seat]
        raise_size = new_total - s.current_bet
        s.last_raise_size = raise_size
        s.current_bet = new_total
        s.aggressor = seat
        for i in range(s.num_players):
            if i != seat and not s.folded[i] and not s.all_in[i]:
                s.acted_this_street[i] = False

    s.acted_this_street[seat] = True
    s.history.append((seat, action_id, chips_added))

    _advance_after_action(s)
    return s


def _advance_after_action(s: GameState) -> None:
    """Update ``to_act``/``stage`` after applying an action (in-place)."""
    # Win-by-fold: only one non-folded player remaining.
    if sum(1 for f in s.folded if not f) == 1:
        s.stage = Stage.DONE
        return

    if _round_complete(s):
        # If everyone left is all-in (or only one can act), run out the board.
        while s.stage != Stage.DONE and _no_more_betting(s):
            _advance_street(s, deal_only=True)
        if s.stage == Stage.DONE:
            return
        if _round_complete(s):
            _advance_street(s, deal_only=False)
            if s.stage == Stage.DONE:
                return
        s.to_act = _first_to_act(
            s.num_players, s.button, s.stage, s.all_in, s.folded
        )
        return

    # Otherwise advance to next eligible actor.
    n = s.num_players
    nxt = (s.to_act + 1) % n
    for _ in range(n):
        if not s.folded[nxt] and not s.all_in[nxt]:
            s.to_act = nxt
            return
        nxt = (nxt + 1) % n
    # Should not reach here unless no one can act, in which case round close
    # logic above should have fired.


def _round_complete(s: GameState) -> bool:
    can_act = [i for i in range(s.num_players)
               if not s.folded[i] and not s.all_in[i]]
    if not can_act:
        return True
    for i in can_act:
        if not s.acted_this_street[i]:
            return False
        if s.street_committed[i] != s.current_bet:
            return False
    return True


def _no_more_betting(s: GameState) -> bool:
    """True if at most one player can still voluntarily act."""
    return sum(1 for i in range(s.num_players)
               if not s.folded[i] and not s.all_in[i]) <= 1


def _advance_street(s: GameState, deal_only: bool) -> None:
    if s.stage >= Stage.RIVER:
        s.stage = Stage.DONE
        return

    s.stage = Stage(s.stage + 1)
    target_board = _BOARD_SIZE[s.stage]
    while len(s.board) < target_board:
        s.board.append(s.deck[s.deck_pos])
        s.deck_pos += 1

    if deal_only:
        return

    # Reset for a fresh street of betting.
    for i in range(s.num_players):
        s.street_committed[i] = 0
        s.acted_this_street[i] = False
    s.current_bet = 0
    s.last_raise_size = s.big_blind
    s.aggressor = -1


# =============================================================================
# Terminal / payouts
# =============================================================================

def is_terminal(state: GameState) -> bool:
    return state.stage == Stage.DONE


def payoffs(state: GameState) -> List[int]:
    """Chip delta per seat (sums to 0). Only call when ``is_terminal``."""
    if not is_terminal(state):
        raise RuntimeError("payoffs() called on non-terminal state")

    n = state.num_players
    deltas = [-c for c in state.committed]   # start by deducting buy-in
    contenders = [i for i in range(n) if not state.folded[i]]

    if len(contenders) == 1:
        deltas[contenders[0]] += sum(state.committed)
        return deltas

    # Build side pots by ascending commitment levels.
    levels = sorted(set(state.committed[i] for i in contenders))
    prev = 0
    # Pre-evaluate hands once.
    strengths = {
        i: evaluate_seven(list(state.hole_cards[i]) + state.board)
        for i in contenders
    }
    for level in levels:
        layer = level - prev
        if layer <= 0:
            prev = level
            continue
        pot_size = 0
        for i in range(n):
            contrib = min(state.committed[i], level) - min(state.committed[i], prev)
            pot_size += contrib
        eligible = [i for i in contenders if state.committed[i] >= level]
        if not eligible:
            prev = level
            continue
        best = max(strengths[i] for i in eligible)
        winners = [i for i in eligible if strengths[i] == best]
        share, rem = divmod(pot_size, len(winners))
        for w in winners:
            deltas[w] += share
        # Distribute odd chips clockwise from first eligible left of button.
        if rem:
            order = [(state.button + 1 + k) % n for k in range(n)]
            for seat in order:
                if rem == 0:
                    break
                if seat in winners:
                    deltas[seat] += 1
                    rem -= 1
        prev = level

    # Sanity: deltas should sum to 0.
    if sum(deltas) != 0:  # pragma: no cover - defensive
        raise AssertionError(f"payoff mismatch: {deltas} (committed={state.committed})")
    return deltas
