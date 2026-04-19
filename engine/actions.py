"""Discretized No-Limit action space.

The network sees a fixed-size action vocabulary; the engine uses a
per-state legality mask to forbid impossible actions (e.g. checking when
facing a bet, or undersized raises).

Layout (id -> meaning), with ``K = len(DEFAULT_BET_FRACTIONS)``::

    0           : FOLD
    1           : CHECK_CALL
    2 .. 1+K    : BET / RAISE-TO sized as a fraction of "pot after call"
    2+K         : ALL_IN

The chip amount for a bet/raise of fraction ``f`` is computed as::

    pot_after_call = pot + (current_bet - actor.street_committed)
    target_street_commit = current_bet + f * pot_after_call

This is the standard solver convention (PIO / GTO+) and corresponds to a
"fraction of pot" bet sizing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

DEFAULT_BET_FRACTIONS: Tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 1.5, 2.5)


@dataclass(frozen=True)
class ActionSpace:
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS

    @property
    def num_bet_sizes(self) -> int:
        return len(self.bet_fractions)

    @property
    def num_actions(self) -> int:
        # FOLD + CHECK_CALL + K bet sizes + ALL_IN
        return 2 + self.num_bet_sizes + 1

    @property
    def fold_id(self) -> int:
        return 0

    @property
    def check_call_id(self) -> int:
        return 1

    @property
    def all_in_id(self) -> int:
        return 1 + self.num_bet_sizes + 1

    def bet_id(self, frac_index: int) -> int:
        if not 0 <= frac_index < self.num_bet_sizes:
            raise IndexError(frac_index)
        return 2 + frac_index

    def is_bet(self, action_id: int) -> bool:
        return 2 <= action_id < 2 + self.num_bet_sizes

    def bet_fraction(self, action_id: int) -> float:
        if not self.is_bet(action_id):
            raise ValueError(f"action {action_id} is not a sized bet")
        return self.bet_fractions[action_id - 2]

    def name(self, action_id: int) -> str:
        if action_id == self.fold_id:
            return "FOLD"
        if action_id == self.check_call_id:
            return "CHECK_CALL"
        if action_id == self.all_in_id:
            return "ALL_IN"
        if self.is_bet(action_id):
            return f"BET_{self.bet_fractions[action_id - 2]:.2f}P"
        raise ValueError(action_id)


_DEFAULT_SPACE = ActionSpace()
NUM_ACTIONS = _DEFAULT_SPACE.num_actions


def fold_id() -> int:
    return _DEFAULT_SPACE.fold_id


def check_call_id() -> int:
    return _DEFAULT_SPACE.check_call_id


def all_in_id() -> int:
    return _DEFAULT_SPACE.all_in_id


def bet_id(frac_index: int) -> int:
    return _DEFAULT_SPACE.bet_id(frac_index)


def action_name(action_id: int) -> str:
    return _DEFAULT_SPACE.name(action_id)
