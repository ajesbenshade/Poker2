"""Real heads-up / N-player No-Limit Hold'em engine.

This package replaces the heuristic ``game.py`` with an explicit game tree:
true betting rounds, side pots, blinds, button rotation, action legality and
showdown via ``eval7``. It is designed to support 2..9 seats from day one
even though only HU is exercised initially.

Public surface:

  - :class:`GameState`        immutable-ish dataclass holding all hand state
  - :func:`new_hand`          deal a fresh hand
  - :func:`legal_action_mask` boolean mask over the discretized action set
  - :func:`apply_action`      transition ``(state, action_id) -> next_state``
  - :func:`is_terminal`       True when the hand is decided
  - :func:`payoffs`           chip delta per seat (sums to zero)
  - :class:`ActionSpace`      enumerates fold/check-call/bet sizes/all-in
  - :func:`encode_observation`  dense per-actor feature vector for a network
"""

from .cards import (
    CARD_COUNT,
    card_id_to_eval7,
    card_ids_to_eval7,
    deal_deck,
    rank_of,
    suit_of,
)
from .actions import (
    ActionSpace,
    DEFAULT_BET_FRACTIONS,
    NUM_ACTIONS,
    action_name,
    fold_id,
    check_call_id,
    all_in_id,
    bet_id,
)
from .state import (
    GameState,
    Stage,
    new_hand,
    apply_action,
    legal_action_mask,
    is_terminal,
    payoffs,
)
from .encoder import (
    OBS_DIM,
    encode_observation,
    encoder_feature_layout,
)

__all__ = [
    "ActionSpace",
    "CARD_COUNT",
    "DEFAULT_BET_FRACTIONS",
    "GameState",
    "NUM_ACTIONS",
    "OBS_DIM",
    "Stage",
    "action_name",
    "all_in_id",
    "apply_action",
    "bet_id",
    "card_id_to_eval7",
    "card_ids_to_eval7",
    "check_call_id",
    "deal_deck",
    "encode_observation",
    "encoder_feature_layout",
    "fold_id",
    "is_terminal",
    "legal_action_mask",
    "new_hand",
    "payoffs",
    "rank_of",
    "suit_of",
]
