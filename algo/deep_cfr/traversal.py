"""External-sampling MCCFR traversal driven by neural advantage networks.

For each Deep CFR iteration ``t`` and each traverser player ``p``, we run
``traversals_per_iter`` independent traversals of randomly dealt hands.
Within a traversal:

* At a node where ``p`` is to act, recurse on **every legal action** to
  compute counterfactual values ``v(I, a)``. The instantaneous regret
  ``r(I, a) = v(I, a) - sum_a sigma(a) * v(I, a)`` is added to ``p``'s
  advantage buffer.
* At a node where the opponent is to act, **sample one action** from their
  current strategy ``sigma_o`` (computed via regret matching on the
  opponent's advantage net) and recurse only on that branch. Also push the
  full opponent strategy distribution into the strategy buffer.
* Chance is already baked into ``new_hand`` (cards are dealt up-front).

The instantaneous regret samples are weighted by the current iteration ``t``
when ``linear_cfr=True`` (Linear CFR; Brown & Sandholm 2019), which
empirically converges faster.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

from engine import (
    GameState,
    encode_observation,
    is_terminal,
    legal_action_mask,
    apply_action,
    payoffs,
)
from engine.encoder import OBS_DIM
from engine.actions import ActionSpace

from .buffer import ReservoirBuffer
from .network import AdvantageNet, PolicyNet


# ---------------------------------------------------------------------------
# Regret matching
# ---------------------------------------------------------------------------

def regret_matching(advantages: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Return a strategy distribution from raw per-action advantages.

    Standard CFR+ rule: positive part of regrets, normalized; uniform over
    legal actions if all regrets are non-positive.
    """
    pos = np.maximum(advantages, 0.0) * legal
    total = pos.sum()
    if total > 0:
        return pos / total
    # Uniform over legal actions
    legal_count = legal.sum()
    if legal_count <= 0:
        # Shouldn't happen \u2014 caller should not invoke regret matching at terminal.
        return np.zeros_like(legal)
    return legal / legal_count


# ---------------------------------------------------------------------------
# Strategy querying
# ---------------------------------------------------------------------------

@torch.no_grad()
def _strategy_from_net(
    net: Optional[AdvantageNet],
    obs: np.ndarray,
    legal: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if net is None:
        # Iteration 0: uniform over legal actions.
        legal_count = legal.sum()
        if legal_count <= 0:
            return np.zeros_like(legal)
        return legal / legal_count
    obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
    legal_t = torch.from_numpy(legal).to(device).unsqueeze(0)
    adv = net(obs_t, legal_t).squeeze(0).float().cpu().numpy()
    return regret_matching(adv, legal)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def _traverse(
    state: GameState,
    traverser: int,
    advantage_nets: List[Optional[AdvantageNet]],
    advantage_buffer: ReservoirBuffer,
    strategy_buffer: ReservoirBuffer,
    iter_t: int,
    rng: random.Random,
    device: torch.device,
    linear_weight: bool,
    big_blind: int = 1,
) -> float:
    """Recursive traversal returning the counterfactual value at ``state``
    for the traverser (in chips)."""
    if is_terminal(state):
        return float(payoffs(state)[traverser])

    seat = state.to_act
    legal_list = legal_action_mask(state)
    legal = np.asarray(legal_list, dtype=np.float32)
    obs = encode_observation(state, perspective_seat=seat)

    sigma = _strategy_from_net(advantage_nets[seat], obs, legal, device)

    if seat == traverser:
        # Recurse on every legal action.
        action_values = np.zeros_like(legal, dtype=np.float32)
        for a in range(len(legal_list)):
            if not legal_list[a]:
                continue
            action_values[a] = _traverse(
                apply_action(state, a),
                traverser,
                advantage_nets,
                advantage_buffer,
                strategy_buffer,
                iter_t,
                rng,
                device,
                linear_weight,
                big_blind,
            )
        node_value = float((sigma * action_values).sum())
        # Normalize regrets by big_blind to keep targets ~O(1).
        instantaneous_regret = ((action_values - node_value) / max(1, big_blind)) * legal
        weight = float(iter_t) if linear_weight else 1.0
        advantage_buffer.add(obs, legal, instantaneous_regret, weight)
        return node_value
    else:
        # External sampling: sample one opponent action.
        weight = float(iter_t) if linear_weight else 1.0
        strategy_buffer.add(obs, legal, sigma.astype(np.float32), weight)
        legal_actions = [a for a, ok in enumerate(legal_list) if ok]
        probs = sigma[legal_actions]
        s = probs.sum()
        if s <= 0:
            chosen = rng.choice(legal_actions)
        else:
            probs = probs / s
            u = rng.random()
            cum = 0.0
            chosen = legal_actions[-1]
            for a, p in zip(legal_actions, probs):
                cum += p
                if u <= cum:
                    chosen = a
                    break
        return _traverse(
            apply_action(state, chosen),
            traverser,
            advantage_nets,
            advantage_buffer,
            strategy_buffer,
            iter_t,
            rng,
            device,
            linear_weight,
            big_blind,
        )


def external_sampling(
    traverser: int,
    advantage_nets: List[Optional[AdvantageNet]],
    advantage_buffer: ReservoirBuffer,
    strategy_buffer: ReservoirBuffer,
    iter_t: int,
    *,
    num_traversals: int,
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    action_space: ActionSpace,
    rng: random.Random,
    device: torch.device,
    linear_weight: bool = True,
) -> None:
    """Run ``num_traversals`` external-sampling MCCFR traversals."""
    from engine import new_hand

    for k in range(num_traversals):
        button = k % num_players
        state = new_hand(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            button=button,
            rng=rng,
            action_space=action_space,
        )
        # Put each net into eval mode for inference safety.
        for net in advantage_nets:
            if net is not None:
                net.eval()
        _traverse(
            state,
            traverser,
            advantage_nets,
            advantage_buffer,
            strategy_buffer,
            iter_t,
            rng,
            device,
            linear_weight,
            big_blind,
        )
