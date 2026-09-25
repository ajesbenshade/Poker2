"""Vectorized 5-7 card hand evaluator: a NumPy port of engine/src/holdem/evaluator.h.

Returns the same 32-bit strengths as the engine: category << 20 | five 4-bit
ranks in order of significance. Higher is better; equal is a tie.
"""

import numpy as np

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ["high card", "pair", "two pair", "trips", "straight", "flush", "full house",
                  "quads", "straight flush"]

_POPCOUNT = np.array([bin(i).count("1") for i in range(1 << 14)], dtype=np.int64)
_HIGHEST = np.array([max(i.bit_length() - 1, 0) for i in range(1 << 14)], dtype=np.int64)


def category_of(strength):
    return np.asarray(strength) >> 20


def _straight_high(ranks):
    """High rank of the best straight in each 13-bit rank mask, or -1 (wheel aware)."""
    m = (ranks << 1) | ((ranks >> 12) & 1)
    runs = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4)
    return np.where(runs > 0, _HIGHEST[runs] + 3, -1)


def _pack_top(mask, count, slot):
    """Top `count` ranks of each mask, highest first, starting at nibble `slot`."""
    mask = mask.copy()
    out = np.zeros_like(mask)
    for i in range(count):
        r = _HIGHEST[mask]
        out |= r << (4 * (slot - i))
        mask &= ~(np.int64(1) << r)
    return out


def evaluate(cards) -> np.ndarray:
    """cards: int array [N, n] with 5 <= n <= 7. Returns int64 strengths [N]."""
    cards = np.asarray(cards, dtype=np.int64)
    rank_bit = np.int64(1) << (cards >> 2)
    suit = cards & 3
    suit_mask = np.stack([np.bitwise_or.reduce(np.where(suit == s, rank_bit, 0), axis=1)
                          for s in range(4)])  # [4, N]

    counts = _POPCOUNT[suit_mask]
    has_flush = (counts >= 5).any(axis=0)
    flush_mask = np.where(counts >= 5, suit_mask, 0).max(axis=0)  # at most one flush suit

    ones = np.zeros(cards.shape[0], dtype=np.int64)
    twos = np.zeros_like(ones)
    threes = np.zeros_like(ones)
    fours = np.zeros_like(ones)
    for s in range(4):
        m = suit_mask[s]
        fours |= threes & m
        threes |= twos & m
        twos |= ones & m
        ones |= m
    trips = threes & ~fours
    pairs = twos & ~threes

    sf_high = _straight_high(flush_mask)
    st_high = _straight_high(ones)
    quad = _HIGHEST[fours]
    trip = _HIGHEST[trips]
    trip_bit = np.int64(1) << trip
    pair1 = _HIGHEST[pairs]
    pair2 = _HIGHEST[pairs & ~(np.int64(1) << pair1)]
    fh_pair = _HIGHEST[(trips & ~trip_bit) | pairs]

    def value(category, ranks):
        return (np.int64(category) << 20) | ranks

    candidates = [
        (has_flush & (sf_high >= 0), value(STRAIGHT_FLUSH, np.maximum(sf_high, 0) << 16)),
        (fours > 0, value(QUADS, (quad << 16) | _pack_top(ones & ~(np.int64(1) << quad), 1, 3))),
        ((trips > 0) & ((_POPCOUNT[trips] >= 2) | (pairs > 0)),
         value(FULL_HOUSE, (trip << 16) | (fh_pair << 12))),
        (has_flush, value(FLUSH, _pack_top(np.where(has_flush, flush_mask, 0x1F), 5, 4))),
        (st_high >= 0, value(STRAIGHT, np.maximum(st_high, 0) << 16)),
        (trips > 0, value(TRIPS, (trip << 16) | _pack_top(ones & ~trip_bit, 2, 3))),
        (_POPCOUNT[pairs] >= 2,
         value(TWO_PAIR, (pair1 << 16) | (pair2 << 12) |
               _pack_top(ones & ~(np.int64(1) << pair1) & ~(np.int64(1) << pair2), 1, 2))),
        (pairs > 0, value(PAIR, (pair1 << 16) | _pack_top(ones & ~(np.int64(1) << pair1), 3, 3))),
    ]
    result = value(HIGH_CARD, _pack_top(ones, 5, 4))
    for condition, v in reversed(candidates):
        result = np.where(condition, v, result)
    return result
