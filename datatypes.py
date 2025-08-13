from enum import Enum

class Suit(Enum):
    HEARTS, DIAMONDS, CLUBS, SPADES = range(4)

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class Card:
    def __init__(self, value: int, suit: Suit):
        self.value = value  # 2-14 (A=14)
        self.suit = suit

class Infoset:
    def __init__(self, bucket_id: int, history: tuple = ()):
        self.key = (bucket_id, hash(history))  # Abstracted state
        self.history = history  # Actions tuple for terminal check