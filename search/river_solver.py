"""Vectorized CFR for river subgames, on the GPU with PyTorch.

On the river every card is public, so each of the 1326 hole-card combos is its
own information set and a whole range is updated with tensor operations.

The tree is processed level by level rather than node by node, so one
iteration is ~100 GPU kernels regardless of tree size (a recursive version
launched thousands of tiny kernels and was no faster than the CPU):

  1. strategies for every edge from the regrets (regret matching);
  2. reach probabilities top-down, one tree depth at a time;
  3. terminal values in a few batched ops: all showdowns are one matrix
     product, values = stake * (opp_reach @ M^T) with M[h, j] =
     sign(strength_h - strength_j) for card-compatible combos, and folds use
     card removal (compatible reach = total - reach sharing a card + self);
  4. values bottom-up, one depth at a time;
  5. regret and strategy-sum updates for every edge at once.

The algorithm is Discounted CFR (Brown & Sandholm 2019) with the usual poker
parameters alpha = 1.5, beta = 0, gamma = 2 and alternating updates.
exploitability() computes an exact best response for each player.
"""

import numpy as np
import torch

from .cards import COMBOS, NUM_COMBOS, combos_blocked_by, combos_overlap_matrix
from .evaluator import evaluate
from .tree import RiverSpot, build_tree

_OVERLAP = None


def _overlap_matrix() -> np.ndarray:
    global _OVERLAP
    if _OVERLAP is None:
        _OVERLAP = combos_overlap_matrix()
    return _OVERLAP


class RiverSolver:
    def __init__(self, spot: RiverSpot, board, ranges, device="cuda", dtype=torch.float32,
                 alpha=1.5, beta=0.0, gamma=2.0):
        if len(board) != 5:
            raise ValueError("a river board has 5 cards")
        self.spot = spot
        self.nodes = build_tree(spot)
        self.device = torch.device(device)
        self.dtype = dtype
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        dev = self.device

        # Cards: valid combos, showdown matrix, card incidence for folds.
        valid = ~combos_blocked_by(board)
        self.ranges = torch.tensor(np.stack([np.where(valid, np.asarray(r, dtype=np.float64), 0.0)
                                             for r in ranges]), dtype=dtype, device=dev)
        cards7 = np.concatenate([COMBOS, np.tile(np.asarray(board, dtype=np.int64), (NUM_COMBOS, 1))], axis=1)
        strength = evaluate(cards7)
        compat = ~_overlap_matrix() & valid[:, None] & valid[None, :]
        showdown = np.where(compat, np.sign(strength[:, None] - strength[None, :]), 0)
        self.showdown_t = torch.tensor(showdown.T.copy(), dtype=dtype, device=dev)
        incidence = np.zeros((NUM_COMBOS, 52))
        incidence[np.arange(NUM_COMBOS), COMBOS[:, 0]] = 1
        incidence[np.arange(NUM_COMBOS), COMBOS[:, 1]] = 1
        self.incidence = torch.tensor(incidence, dtype=dtype, device=dev)

        # Tree structure as flat arrays. Edge e goes from parent[e] to child[e].
        depth = [0] * len(self.nodes)
        parent, child, actor, self.node_edges = [], [], [], {}
        for node in self.nodes:  # parents precede children in build order
            if node.kind != "decision":
                continue
            first = len(parent)
            for c in node.children:
                depth[c] = depth[node.index] + 1
                parent.append(node.index)
                child.append(c)
                actor.append(node.player)
            self.node_edges[node.index] = (first, len(parent))
        to_t = lambda x: torch.tensor(x, dtype=torch.long, device=dev)
        self.parent, self.child, self.actor = to_t(parent), to_t(child), to_t(actor)
        self.num_edges = len(parent)
        edge_depth = np.array([depth[p] for p in parent])
        self.levels = [to_t(np.nonzero(edge_depth == d)[0]) for d in range(edge_depth.max() + 1)]
        self.edges_of = [self.actor == p for p in (0, 1)]
        counts = np.zeros(len(self.nodes))
        np.add.at(counts, parent, 1)
        self.edge_count = torch.tensor(counts[parent], dtype=dtype, device=dev)[:, None]
        self.decision = to_t([n.index for n in self.nodes if n.kind == "decision"])

        showdowns = [n for n in self.nodes if n.kind == "showdown"]
        folds = [n for n in self.nodes if n.kind == "fold"]
        self.showdown_nodes = to_t([n.index for n in showdowns])
        self.showdown_stake = torch.tensor([float(n.contrib[0]) for n in showdowns], dtype=dtype, device=dev)[:, None]
        self.fold_nodes = to_t([n.index for n in folds])
        self.fold_payoff = [torch.tensor([float(-n.contrib[p] if n.folder == p else n.contrib[1 - p]) for n in folds],
                                         dtype=dtype, device=dev)[:, None] for p in (0, 1)]

        shape = (self.num_edges, NUM_COMBOS)
        self.regret = torch.zeros(shape, dtype=dtype, device=dev)
        self.strategy_sum = torch.zeros(shape, dtype=dtype, device=dev)
        self.iterations = 0

    # -- building blocks -------------------------------------------------------

    def _normalize(self, weights):
        """Per-node normalization of edge weights; uniform where a node's total is 0."""
        totals = torch.zeros(len(self.nodes), NUM_COMBOS, dtype=self.dtype, device=self.device)
        totals.index_add_(0, self.parent, weights)
        per_edge = totals[self.parent]
        return torch.where(per_edge > 0, weights / per_edge.clamp(min=1e-30), 1.0 / self.edge_count)

    def _current_strategy(self):
        return self._normalize(self.regret.clamp(min=0))

    def _reach(self, sigma):
        """Reach of both players at every node: [2, nodes, 1326]."""
        reach = torch.zeros(2, len(self.nodes), NUM_COMBOS, dtype=self.dtype, device=self.device)
        reach[:, 0] = self.ranges
        for level in self.levels:
            par, ch, act = self.parent[level], self.child[level], self.actor[level]
            s = sigma[level]
            reach[0, ch] = reach[0, par] * torch.where((act == 0)[:, None], s, 1.0)
            reach[1, ch] = reach[1, par] * torch.where((act == 1)[:, None], s, 1.0)
        return reach

    def _compatible_reach(self, reach):
        """reach: [..., 1326]. Sum of reach over combos sharing no card with each combo."""
        per_card = reach @ self.incidence
        return reach.sum(-1, keepdim=True) - per_card @ self.incidence.T + reach

    def _terminal_values(self, player, opp_reach):
        """Values for `player` at every node, filled in at terminals: [nodes, 1326]."""
        values = torch.zeros(len(self.nodes), NUM_COMBOS, dtype=self.dtype, device=self.device)
        if len(self.showdown_nodes):
            values[self.showdown_nodes] = self.showdown_stake * (opp_reach[self.showdown_nodes] @ self.showdown_t)
        if len(self.fold_nodes):
            values[self.fold_nodes] = self.fold_payoff[player] * self._compatible_reach(opp_reach[self.fold_nodes])
        return values

    # -- CFR -----------------------------------------------------------------

    def _traverse(self, player):
        sigma = self._current_strategy()
        reach = self._reach(sigma)
        values = self._terminal_values(player, reach[1 - player])
        own = self.edges_of[player][:, None]
        for level in reversed(self.levels):
            par, ch = self.parent[level], self.child[level]
            contribution = values[ch] * torch.where(own[level], sigma[level], 1.0)
            values.index_add_(0, par, contribution)  # decision rows start at zero
        mine = self.edges_of[player]
        regret_delta = values[self.child] - values[self.parent]
        self.regret += torch.where(mine[:, None], regret_delta, 0.0)
        self.strategy_sum += torch.where(mine[:, None], reach[player][self.parent] * sigma, 0.0)

    def _discount(self, player):
        t = self.iterations
        pos = t ** self.alpha / (t ** self.alpha + 1)
        neg = t ** self.beta / (t ** self.beta + 1)
        strat = (t / (t + 1)) ** self.gamma
        mine = self.edges_of[player][:, None]
        self.regret *= torch.where(mine, torch.where(self.regret > 0, pos, neg), 1.0)
        self.strategy_sum *= torch.where(mine, strat, 1.0)

    def iterate(self, count=1):
        with torch.no_grad():
            for _ in range(count):
                self.iterations += 1
                for player in (0, 1):
                    self._traverse(player)
                    self._discount(player)

    # -- strategies and evaluation ---------------------------------------------

    def average_strategy(self, index):
        """[actions, 1326] average strategy at decision node `index`."""
        first, last = self.node_edges[index]
        return self._normalize(self.strategy_sum)[first:last]

    def _best_response_values(self, player, sigma):
        opp = 1 - player
        reach = self._reach(sigma)
        values = self._terminal_values(player, reach[opp])
        mine = self.edges_of[player]
        for level in reversed(self.levels):
            par, ch = self.parent[level], self.child[level]
            own = mine[level]
            if (~own).any():  # opponent nodes: sum (its strategy is in the reach)
                values.index_add_(0, par[~own], values[ch[~own]])
            if own.any():     # best responder: max over actions, per combo
                idx = par[own][:, None].expand(-1, NUM_COMBOS)
                values.scatter_reduce_(0, idx, values[ch[own]], reduce="amax", include_self=False)
        return values[0]

    def exploitability(self):
        """(exploitability in chips per hand, as a fraction of the starting pot).

        Values are averaged over the card-compatible pairs of the two ranges.
        The game is zero-sum, so exploitability = (BR value 0 + BR value 1) / 2.
        """
        with torch.no_grad():
            sigma = self._normalize(self.strategy_sum)
            z = float(self.ranges[0] @ self._compatible_reach(self.ranges[1]))
            br = [float(self.ranges[p] @ self._best_response_values(p, sigma)) / z for p in (0, 1)]
        chips = (br[0] + br[1]) / 2.0
        return chips, chips / self.spot.pot

    def node_after(self, *labels):
        """Index of the node reached by following action labels from the root."""
        index = 0
        for label in labels:
            node = self.nodes[index]
            index = node.children[node.actions.index(label)]
        return index
