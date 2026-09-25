"""Cards and hole-card combos, matching engine/src/holdem/cards.h."""

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
NUM_COMBOS = 1326


def parse_card(text: str) -> int:
    rank = RANKS.index(text[0].upper())
    suit = SUITS.index(text[1].lower())
    return rank * 4 + suit


def parse_cards(text: str) -> list[int]:
    """"AsKd" -> [As, Kd]."""
    if len(text) % 2:
        raise ValueError(f"bad card list: {text}")
    return [parse_card(text[i:i + 2]) for i in range(0, len(text), 2)]


def card_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def _build_combos() -> np.ndarray:
    return np.array([(a, b) for a in range(52) for b in range(a + 1, 52)], dtype=np.int64)


COMBOS = _build_combos()  # [1326, 2], same order as the engine
COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(COMBOS)}


def combo_index(c1: int, c2: int) -> int:
    return COMBO_INDEX[(min(c1, c2), max(c1, c2))]


def combos_blocked_by(cards) -> np.ndarray:
    """Boolean [1326]: combo shares a card with `cards`."""
    dead = np.zeros(52, dtype=bool)
    dead[list(cards)] = True
    return dead[COMBOS[:, 0]] | dead[COMBOS[:, 1]]


def combos_overlap_matrix() -> np.ndarray:
    """Boolean [1326, 1326]: the two combos share a card (diagonal True)."""
    incidence = np.zeros((NUM_COMBOS, 52), dtype=np.int32)
    incidence[np.arange(NUM_COMBOS), COMBOS[:, 0]] = 1
    incidence[np.arange(NUM_COMBOS), COMBOS[:, 1]] = 1
    return (incidence @ incidence.T) > 0
