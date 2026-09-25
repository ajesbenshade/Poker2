"""River solver: tree rules, an analytic equilibrium, exact convergence, precision."""

import numpy as np
import pytest
import torch

from search.cards import NUM_COMBOS, combo_index, parse_cards
from search.river_solver import RiverSolver
from search.tree import RiverSpot, build_tree

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def range_of(weights: dict) -> np.ndarray:
    r = np.zeros(NUM_COMBOS)
    for text, w in weights.items():
        a, b = parse_cards(text)
        r[combo_index(a, b)] = w
    return r


def test_tree_follows_engine_sizing():
    spot = RiverSpot(contrib=(500, 500), stack=20000, first_to_act=1)
    nodes = build_tree(spot)
    root = nodes[0]
    assert root.player == 1
    # Pot 1000: 0.33 -> 330, 0.5 -> 500, 0.75 -> 750, 1.0 -> 1000, 1.5 -> 1500, plus all-in.
    assert root.actions == ["k", "b330", "b500", "b750", "b1000", "b1500", "b19500"]
    facing = nodes[root.children[root.actions.index("b1000")]]
    # Raises use (0.5, 1.0) of the pot after calling (3000): to 2500 and 4000.
    assert facing.actions == ["f", "c", "b2500", "b4000", "b19500"]
    assert all(n.kind in ("decision", "fold", "showdown") for n in nodes)


def test_polarized_river_matches_the_analytic_equilibrium():
    # Board 2c3d7h8sKd. Player 0 holds trips (KcKs) or air (4c5c); player 1
    # holds a bluff-catcher (AhQh) that beats air and loses to trips. Pot 1000,
    # one pot-size bet (= all-in). Theory: player 1 checks; player 0 bets all
    # trips and bluffs so that bluffs : value = B / (P + B) = 1/2, i.e. half its
    # air; player 1 calls a bet with probability P / (P + B) = 1/2.
    spot = RiverSpot(contrib=(500, 500), stack=1500, first_to_act=1, bet_sizes=(1.0,),
                     raise_sizes=(1.0,), max_raises=1)
    solver = RiverSolver(spot, parse_cards("2c3d7h8sKd"),
                         [range_of({"KcKs": 1.0, "4c5c": 1.0}), range_of({"AhQh": 1.0})], device=DEVICE)
    solver.iterate(3000)
    kk, air, bc = (combo_index(*parse_cards(t)) for t in ("KcKs", "4c5c", "AhQh"))

    root = solver.average_strategy(0)
    assert solver.nodes[0].actions == ["k", "b1000"]
    assert root[0, bc] > 0.99, "bluff-catcher checks"

    after_check = solver.node_after("k")
    s = solver.average_strategy(after_check)
    assert solver.nodes[after_check].actions == ["k", "b1000"]
    assert s[1, kk] > 0.99, "trips always bet"
    assert float(s[1, air]) == pytest.approx(0.5, abs=0.02), "air bluffs half the time"

    facing = solver.node_after("k", "b1000")
    s = solver.average_strategy(facing)
    assert solver.nodes[facing].actions == ["f", "c"]
    assert float(s[1, bc]) == pytest.approx(0.5, abs=0.02), "bluff-catcher calls half the time"

    chips, fraction = solver.exploitability()
    assert fraction < 1e-3


def test_realistic_river_converges():
    # Uniform ranges on a coordinated board, full blueprint river sizes.
    spot = RiverSpot(contrib=(500, 500), stack=20000, first_to_act=1)
    ranges = [np.ones(NUM_COMBOS), np.ones(NUM_COMBOS)]
    solver = RiverSolver(spot, parse_cards("Ts9s8d4h2c"), ranges, device=DEVICE)
    solver.iterate(20)
    early = solver.exploitability()[1]
    solver.iterate(480)
    late = solver.exploitability()[1]
    print(f"exploitability: {early:.4%} of pot after 20 iterations, {late:.4%} after 500")
    assert late < early / 10
    assert late < 0.005, "under 0.5% of the pot after 500 iterations"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_float32_agrees_with_cpu_float64():
    spot = RiverSpot(contrib=(1000, 1000), stack=20000, first_to_act=1,
                     bet_sizes=(0.5, 1.0), raise_sizes=(1.0,), max_raises=2)
    rng = np.random.default_rng(3)
    ranges = [rng.random(NUM_COMBOS), rng.random(NUM_COMBOS)]
    board = parse_cards("AhJd7c5s3h")
    gpu = RiverSolver(spot, board, ranges, device="cuda", dtype=torch.float32)
    cpu = RiverSolver(spot, board, ranges, device="cpu", dtype=torch.float64)
    gpu.iterate(200)
    cpu.iterate(200)
    # Compare exploitability, not strategies: equilibria are not unique (many
    # hands are indifferent between actions), so rounding differences steer the
    # two runs to different but equally good mixes. Measured: 0.2414% vs 0.2429%
    # of the pot, while single-combo frequencies differed by up to 0.28.
    assert gpu.exploitability()[1] == pytest.approx(cpu.exploitability()[1], abs=1e-4)
