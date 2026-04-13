import atexit
import numpy as np
from environment import setup_rocmo

setup_rocmo()

import torch
from datatypes import Action
from config import Config
import eval7
import multiprocessing as mp

_HAND_RANK_POOL = None


def _shutdown_hand_rank_pool():
    global _HAND_RANK_POOL
    if _HAND_RANK_POOL is not None:
        _HAND_RANK_POOL.close()
        _HAND_RANK_POOL.join()
        _HAND_RANK_POOL = None


atexit.register(_shutdown_hand_rank_pool)


def _get_hand_rank_pool():
    global _HAND_RANK_POOL
    if _HAND_RANK_POOL is None:
        # Reuse one forkserver pool per process to avoid re-spawning workers every batch.
        _HAND_RANK_POOL = mp.get_context('forkserver').Pool(processes=Config.MP_PROCESSES)
    return _HAND_RANK_POOL


def _card_to_id(card):
    # Support both eval7 cards and lightweight project card objects with suit enums.
    rank = getattr(card, 'rank', getattr(card, 'value', 2) - 2)
    suit = getattr(card, 'suit', 0)
    suit_value = getattr(suit, 'value', suit)
    return int(rank) * 4 + int(suit_value)

def evaluate_hand(cards_ids):
    ranks_str = '23456789TJQKA'
    suits_str = 'shdc'  # eval7 suit order: s=0, h=1, d=2, c=3
    cards = []
    for card_id in cards_ids:
        rank_id = card_id // 4
        suit_id = card_id % 4
        card_str = ranks_str[rank_id] + suits_str[suit_id]
        cards.append(eval7.Card(card_str))
    return eval7.evaluate(cards)

def hand_rank_tensor(hands):  # hands: (batch, 7) tensor of card ints (int64)
    hands_list = hands.cpu().numpy().tolist()  # Move to CPU list for multiprocessing
    pool = _get_hand_rank_pool()
    ranks = pool.map(evaluate_hand, hands_list)
    return torch.tensor(ranks, dtype=torch.int32, device=Config.DEVICE)

def _sanitize_equity_tensor(equities):
    equities = torch.nan_to_num(equities, nan=0.5, posinf=1.0, neginf=0.0)
    return torch.clamp(equities, min=0.0, max=1.0)


def simulate_equity_batch(holes, boards, num_opponents=None):
    num_opponents = max(1, num_opponents or Config.NUM_OPPONENTS)
    num_hands = len(holes)
    equities = torch.zeros(num_hands, device=Config.DEVICE)
    batch_size = min(Config.BATCH_SIZE, num_hands)
    for start in range(0, num_hands, batch_size):
        end = min(start + batch_size, num_hands)
        local_holes = holes[start:end]
        local_boards = boards[start:end]
        board_groups = {}
        for offset, board in enumerate(local_boards):
            board_groups.setdefault(len(board), []).append(offset)

        for board_len, group_offsets in board_groups.items():
            group_holes = [local_holes[offset] for offset in group_offsets]
            group_boards = [local_boards[offset] for offset in group_offsets]
            group_size = len(group_offsets)
            batch_holes = torch.tensor([[_card_to_id(card) for card in hole] for hole in group_holes], dtype=torch.int64, device=Config.DEVICE)
            if board_len > 0:
                batch_boards = torch.tensor([[_card_to_id(card) for card in board] for board in group_boards], dtype=torch.int64, device=Config.DEVICE)
            else:
                batch_boards = torch.empty((group_size, 0), dtype=torch.int64, device=Config.DEVICE)

            full_deck = torch.arange(52, dtype=torch.int64, device=Config.DEVICE).unsqueeze(0).repeat(group_size, 1)
            used = torch.cat((batch_holes, batch_boards), dim=1)
            mask = torch.zeros((group_size, 52), device=Config.DEVICE).scatter_(1, used, 1).bool()
            remaining = full_deck[~mask].reshape(group_size, -1)
            num_rollouts = Config.EQUITY_ROLLOUTS
            rollout_wins = torch.zeros((group_size, num_rollouts), device=Config.DEVICE)
            board_cards_needed = max(0, 5 - board_len)

            for rollout_idx in range(num_rollouts):
                random_scores = torch.rand((group_size, remaining.shape[1]), device=Config.DEVICE)
                shuffled_indices = torch.argsort(random_scores, dim=1)
                shuffled_remaining = remaining.gather(1, shuffled_indices)
                opp_cards = shuffled_remaining[:, :2 * num_opponents].reshape(group_size, num_opponents, 2)
                board_add = shuffled_remaining[:, 2 * num_opponents:2 * num_opponents + board_cards_needed]
                full_boards = torch.cat((batch_boards, board_add), dim=1)
                my_hands = torch.cat((batch_holes, full_boards), dim=1)
                my_ranks = hand_rank_tensor(my_hands)
                opp_ranks = torch.stack(
                    [hand_rank_tensor(torch.cat((opp_cards[:, opponent_idx], full_boards), dim=1)) for opponent_idx in range(num_opponents)],
                    dim=1,
                )
                wins = (my_ranks.unsqueeze(1) > opp_ranks).float()
                ties = (my_ranks.unsqueeze(1) == opp_ranks).float() * 0.5
                rollout_wins[:, rollout_idx] = (wins + ties).mean(dim=1)

            wins = rollout_wins.mean(dim=1)
            noisy_equities = _sanitize_equity_tensor(wins + torch.randn(group_size, device=Config.DEVICE) * Config.EQUITY_STD)
            for offset_index, local_index in enumerate(group_offsets):
                equities[start + local_index] = noisy_equities[offset_index]
    return _sanitize_equity_tensor(equities).to(torch.float32).cpu().numpy()

def simulate_action_batch(infosets, actions):
    results = np.zeros(len(infosets))
    for i, (infoset, action) in enumerate(zip(infosets, actions)):
        deck = eval7.Deck()
        deck.shuffle()
        hole = deck.deal(2)
        board = deck.deal(len(infoset.history) % 5)
        equity = simulate_equity_batch([hole], [board])[0]
        pot = Config.POT_SIZE + np.random.uniform(0, 100)  # Wider pot variance
        call_amt = Config.CALL_AMOUNT + np.random.uniform(0, 20)
        fold_equity = np.random.normal(Config.FOLD_EQUITY_MEAN, Config.FOLD_EQUITY_STD)
        bluff = np.random.uniform(0, Config.BLUFF_FACTOR) if action == Action.RAISE else 0
        equity += bluff - (1 - equity) * 0.2 if action == Action.FOLD else 0  # Penalty for fold low equity
        if action is None:
            results[i] = float(np.clip(equity - 0.5, -1.0, 1.0))
        elif action == Action.FOLD:
            results[i] = -call_amt / pot * Config.FOLD_PENALTY  # Higher loss
        elif action == Action.CALL:
            results[i] = equity * pot / (pot + call_amt) - call_amt / pot
        else:
            raise_amt = call_amt * (Config.RAISE_MULTIPLIER + np.random.uniform(0, 2))
            new_pot = pot + raise_amt * (1 - fold_equity)
            results[i] = equity * new_pot / (new_pot + raise_amt) - raise_amt / pot + fold_equity * pot / new_pot * bluff
    return np.nan_to_num(results, nan=0.0, posinf=Config.UTILITY_CLAMP, neginf=-Config.UTILITY_CLAMP)

def simulate_action(infoset, action: Action):
    return simulate_action_batch([infoset], [action])[0]


def quick_simulate(infoset, action: Action, simulations=32):
    # Fast low-fidelity utility estimate used for cheap policy probes.
    values = []
    sims = max(4, int(simulations))
    for _ in range(sims):
        values.append(simulate_action(infoset, action))
    return float(np.mean(values))


def hybrid_equity(nn_equity, mc_equity, weight=None):
    weight = Config.HYBRID_EQUITY_WEIGHT if weight is None else float(weight)
    weight = float(np.clip(weight, 0.0, 1.0))
    return float(weight * nn_equity + (1.0 - weight) * mc_equity)

def terminal(infoset):
    return len(infoset.history) >= 4