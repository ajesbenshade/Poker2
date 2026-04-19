"""Card utilities for the engine.

Cards are encoded as integers in ``[0, 52)``::

    card_id = rank * 4 + suit
    rank \u2208 [0, 12]   # 0=2, 1=3, ..., 12=A
    suit \u2208 [0, 3]    # eval7 order: 0=s, 1=h, 2=d, 3=c

This matches the encoding used by the legacy ``game.py``/``evaluate_hand``
helpers, so card ints are interchangeable across the codebase.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Sequence

import eval7

CARD_COUNT = 52
_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "shdc"  # eval7 order

_EVAL7_CACHE: list[eval7.Card] = []


def _build_cache() -> None:
    if _EVAL7_CACHE:
        return
    for cid in range(CARD_COUNT):
        r = cid // 4
        s = cid % 4
        _EVAL7_CACHE.append(eval7.Card(_RANK_CHARS[r] + _SUIT_CHARS[s]))


def rank_of(card_id: int) -> int:
    return int(card_id) // 4


def suit_of(card_id: int) -> int:
    return int(card_id) % 4


def card_id_to_eval7(card_id: int) -> eval7.Card:
    _build_cache()
    return _EVAL7_CACHE[int(card_id)]


def card_ids_to_eval7(card_ids: Iterable[int]) -> List[eval7.Card]:
    _build_cache()
    return [_EVAL7_CACHE[int(c)] for c in card_ids]


def deal_deck(rng: random.Random) -> List[int]:
    """Return a freshly shuffled deck of 52 card ints."""
    deck = list(range(CARD_COUNT))
    rng.shuffle(deck)
    return deck


def evaluate_seven(card_ids: Sequence[int]) -> int:
    """Return eval7 hand strength for a 5\u20137 card holding (higher is better)."""
    return int(eval7.evaluate(card_ids_to_eval7(card_ids)))
