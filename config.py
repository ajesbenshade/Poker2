import torch


class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.bfloat16
    NUM_BUCKETS = 5000
    NUM_SIMS = 100000
    ITERATIONS = 1000000
    SAMPLING_RATE = 0.5
    DISCOUNT = 0.99
    NUM_ACTIONS = 3
    NUM_OPPONENTS = 1
    BATCH_SIZE = 1024
    POT_SIZE = 100.0
    CALL_AMOUNT = 20.0
    RAISE_MULTIPLIER = 3.0
    FOLD_EQUITY_MEAN = 0.4
    FOLD_EQUITY_STD = 0.3
    EQUITY_STD = 0.1
    EQUITY_ROLLOUTS = 40
    BLUFF_FACTOR = 0.3
    FOLD_PENALTY = 0.5
