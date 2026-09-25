"""River subgame betting trees, following the engine's no-limit rules.

Mirrors engine/src/holdem/game.h and action_abstraction.h for a single
street: raise amounts are "raise to" totals for the street, sizes are pot
fractions after calling, snapped up to the minimum raise and down to all-in,
duplicates removed, all-in always offered while raising is allowed, and no
raising once the cap is reached or the opponent is all-in.
"""

import math
from dataclasses import dataclass, field


@dataclass
class Node:
    kind: str                      # "decision", "fold" or "showdown"
    player: int = -1               # acting player at decision nodes
    contrib: tuple = (0, 0)        # total chips each player has put in the pot
    folder: int = -1               # fold terminals: who folded
    actions: list = field(default_factory=list)   # labels, e.g. "k", "c", "f", "b300"
    children: list = field(default_factory=list)  # node indices
    index: int = -1


@dataclass
class RiverSpot:
    """Betting state when the river starts."""
    contrib: tuple = (500, 500)    # chips already in the pot from each player
    stack: int = 20000             # starting stack (same for both players)
    big_blind: int = 100
    first_to_act: int = 1          # heads-up: the big blind acts first postflop
    bet_sizes: tuple = (0.33, 0.5, 0.75, 1.0, 1.5)   # first bet, pot fractions
    raise_sizes: tuple = (0.5, 1.0)                  # later raises
    max_raises: int = 3
    allow_all_in: bool = True

    @property
    def pot(self) -> int:
        return self.contrib[0] + self.contrib[1]


def build_tree(spot: RiverSpot) -> list[Node]:
    nodes: list[Node] = []

    def add(node: Node) -> int:
        node.index = len(nodes)
        nodes.append(node)
        return node.index

    def build(contrib, street, to_act, last_raise, raises, acted) -> int:
        opp = 1 - to_act
        to_call = street[opp] - street[to_act]
        remaining = [spot.stack - contrib[0], spot.stack - contrib[1]]
        node_id = add(Node("decision", player=to_act, contrib=tuple(contrib)))

        options = []  # (label, kind, payload)
        if to_call > 0:
            options.append(("f", "fold", None))
        options.append(("c" if to_call > 0 else "k", "call", None))
        can_raise = remaining[to_act] > to_call and remaining[opp] > 0 and raises < spot.max_raises
        if can_raise:
            max_to = street[to_act] + remaining[to_act]
            min_to = min(street[opp] + max(last_raise, spot.big_blind), max_to)
            fractions = spot.bet_sizes if raises == 0 else spot.raise_sizes
            pot_after_call = contrib[0] + contrib[1] + to_call
            # floor(x + 0.5) matches the engine's std::lround (Python's round() rounds halves to even).
            amounts = {min(max(street[opp] + math.floor(f * pot_after_call + 0.5), min_to), max_to)
                       for f in fractions}
            if spot.allow_all_in:
                amounts.add(max_to)
            for amount in sorted(amounts):
                options.append((f"b{amount}", "raise", amount))

        children, labels = [], []
        for label, kind, amount in options:
            labels.append(label)
            if kind == "fold":
                children.append(add(Node("fold", contrib=tuple(contrib), folder=to_act)))
            elif kind == "call":
                new_contrib, new_street = list(contrib), list(street)
                new_contrib[to_act] += to_call
                new_street[to_act] += to_call
                now_acted = list(acted)
                now_acted[to_act] = True
                if now_acted[opp]:  # street closes: the river is last, so showdown
                    children.append(add(Node("showdown", contrib=tuple(new_contrib))))
                else:
                    children.append(build(new_contrib, new_street, opp, last_raise, raises, now_acted))
            else:
                new_contrib, new_street = list(contrib), list(street)
                increment = amount - street[opp]
                new_last = increment if increment >= last_raise else last_raise
                new_contrib[to_act] += amount - street[to_act]
                new_street[to_act] = amount
                now_acted = [False, False]
                now_acted[to_act] = True
                children.append(build(new_contrib, new_street, opp, new_last, raises + 1, now_acted))
        nodes[node_id].actions = labels
        nodes[node_id].children = children
        return node_id

    build(list(spot.contrib), [0, 0], spot.first_to_act, spot.big_blind, 0, [False, False])
    return nodes
