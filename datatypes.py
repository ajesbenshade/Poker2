from dataclasses import dataclass
from enum import Enum
import hashlib


class Suit(Enum):
    HEARTS, DIAMONDS, CLUBS, SPADES = range(4)


class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2


@dataclass(frozen=True)
class Card:
    value: int
    suit: int | Suit

    @property
    def rank(self):
        return self.value

    @property
    def suit_id(self):
        return self.suit.value if isinstance(self.suit, Suit) else int(self.suit)

    def to_id(self):
        return self.value * 4 + self.suit_id


class Infoset:
    KEY_MODE = "legacy"

    def __init__(
        self,
        bucket_id: int,
        history: tuple = (),
        acting_player: int = 0,
        street: Street | int = Street.PREFLOP,
        pot_size: float = 100.0,
        stack_sizes: tuple[float, float] = (1000.0, 1000.0),
        board_cards: tuple[int, ...] = (),
        private_cards: tuple[int, ...] = (),
        current_bet: float = 20.0,
    ):
        self.bucket_id = int(bucket_id)
        self.history = tuple(history)
        self.acting_player = int(acting_player)
        self.street = street if isinstance(street, Street) else Street(int(street))
        self.pot_size = float(pot_size)
        self.stack_sizes = tuple(float(stack) for stack in stack_sizes)
        self.board_cards = tuple(int(card_id) for card_id in board_cards)
        self.private_cards = tuple(int(card_id) for card_id in private_cards)
        self.current_bet = float(current_bet)
        self.legacy_key = (self.bucket_id, self._hash_tuple(self.history))
        self.state_key = (
            self.bucket_id,
            self.acting_player,
            self.street.value,
            round(self.pot_size, 4),
            tuple(round(stack, 4) for stack in self.stack_sizes),
            round(self.current_bet, 4),
            self._hash_tuple(self.board_cards),
            self._hash_tuple(self.private_cards),
            self._hash_tuple(self.history),
        )

    @staticmethod
    def _hash_tuple(values):
        return hashlib.sha256(str(tuple(values)).encode()).hexdigest()[:16]

    @property
    def key(self):
        if self.KEY_MODE == "legacy":
            return self.legacy_key
        if self.KEY_MODE == "state":
            return self.state_key
        raise ValueError(f"Unsupported infoset key mode: {self.KEY_MODE}")

    @property
    def effective_stack(self):
        return min(self.stack_sizes) if self.stack_sizes else 0.0

    @property
    def history_length(self):
        return len(self.history)

    def next_infoset(
        self,
        *,
        history=None,
        acting_player=None,
        street=None,
        pot_size=None,
        stack_sizes=None,
        board_cards=None,
        private_cards=None,
        current_bet=None,
    ):
        return Infoset(
            self.bucket_id,
            history=self.history if history is None else tuple(history),
            acting_player=self.acting_player if acting_player is None else int(acting_player),
            street=self.street if street is None else street,
            pot_size=self.pot_size if pot_size is None else float(pot_size),
            stack_sizes=self.stack_sizes if stack_sizes is None else tuple(float(stack) for stack in stack_sizes),
            board_cards=self.board_cards if board_cards is None else tuple(int(card_id) for card_id in board_cards),
            private_cards=self.private_cards if private_cards is None else tuple(int(card_id) for card_id in private_cards),
            current_bet=self.current_bet if current_bet is None else float(current_bet),
        )
