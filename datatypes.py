from enum import Enum
import hashlib


class Suit(Enum):
    HEARTS, DIAMONDS, CLUBS, SPADES = range(4)


class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2


class Card:
    def __init__(self, value: int, suit: Suit):
        self.value = value
        self.suit = suit


class Infoset:
    def __init__(self, bucket_id: int, history: tuple = ()):    
        h = hashlib.sha256(str(history).encode()).hexdigest()[:16]
        self.key = (int(bucket_id), h)
        self.history = history
