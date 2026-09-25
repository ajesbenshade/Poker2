"""The NumPy evaluator against published combinatorics, like the C++ tests."""

import itertools

import numpy as np

from search.cards import parse_cards
from search.evaluator import CATEGORY_NAMES, category_of, evaluate


def ev(text):
    return int(evaluate(np.array([parse_cards(text)]))[0])


def test_hand_picked_orderings():
    assert category_of(ev("AsKsQsJsTs")) == 8
    assert ev("As2s3s4s5s") < ev("2s3s4s5s6s"), "steel wheel is the lowest straight flush"
    assert ev("AsAhAdAc2s") > ev("KsKhKdKcAs")
    assert ev("3s3h3d2s2h") > ev("2s2h2dAsAh"), "full house ranked by trips first"
    assert ev("As2d3c4h5s") < ev("2d3c4h5s6h"), "wheel is the lowest straight"
    assert ev("AsAhKdKcQs") > ev("AsAhKdKc2s"), "two pair kicker matters"
    assert ev("AsAhAdKsKhKd2c") == ev("AsAhAdKsKh3c4d"), "two trips make the best full house"
    assert ev("AsAhKdKcQsQh2d") == ev("AsAhKdKc2sQh3d"), "third pair only counts as a kicker"
    assert category_of(ev("AsKsQsJs9s8h7h")) == 5, "flush beats straight"


def test_all_five_card_hands_match_published_counts():
    hands = np.array(list(itertools.combinations(range(52), 5)), dtype=np.int64)
    assert len(hands) == 2598960
    strengths = evaluate(hands)
    categories = category_of(strengths)
    expected_count = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    expected_distinct = [1277, 2860, 858, 858, 10, 1277, 156, 156, 10]
    for k in range(9):
        in_category = strengths[categories == k]
        assert len(in_category) == expected_count[k], CATEGORY_NAMES[k]
        assert len(np.unique(in_category)) == expected_distinct[k], CATEGORY_NAMES[k]
    assert len(np.unique(strengths)) == 7462


def test_seven_cards_equal_best_five_card_subset():
    rng = np.random.default_rng(2024)
    hands = np.array([rng.choice(52, 7, replace=False) for _ in range(20000)])
    best = np.zeros(len(hands), dtype=np.int64)
    for subset in itertools.combinations(range(7), 5):
        best = np.maximum(best, evaluate(hands[:, list(subset)]))
    assert np.array_equal(evaluate(hands), best)
