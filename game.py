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
    if hands.numel() == 0:
        return torch.empty((0,), dtype=torch.int32, device=Config.DEVICE)

    hands_list = hands.detach().cpu().tolist()
    use_pool = Config.MP_PROCESSES > 1 and len(hands_list) >= max(16, Config.MP_PROCESSES * 2)
    if use_pool:
        pool = _get_hand_rank_pool()
        chunksize = max(1, len(hands_list) // max(1, Config.MP_PROCESSES * 4))
        ranks = pool.map(evaluate_hand, hands_list, chunksize=chunksize)
    else:
        ranks = [evaluate_hand(hand) for hand in hands_list]
    return torch.tensor(ranks, dtype=torch.int32, device=Config.DEVICE)


def _rank_rollout_hands(batch_holes, opp_cards, full_boards):
    group_size = batch_holes.shape[0]
    num_opponents = opp_cards.shape[1]
    player_hands = torch.cat((batch_holes.unsqueeze(1), opp_cards), dim=1)
    board_tiles = full_boards.unsqueeze(1).expand(group_size, num_opponents + 1, full_boards.shape[1])
    all_hands = torch.cat((player_hands, board_tiles), dim=2).reshape(group_size * (num_opponents + 1), -1)
    all_ranks = hand_rank_tensor(all_hands).view(group_size, num_opponents + 1)
    return all_ranks[:, 0], all_ranks[:, 1:]


def _sample_remaining_cards(remaining, cards_needed):
    if cards_needed <= 0:
        return torch.empty((remaining.shape[0], 0), dtype=remaining.dtype, device=remaining.device)

    random_scores = torch.rand((remaining.shape[0], remaining.shape[1]), device=remaining.device)
    sampled_indices = torch.topk(random_scores, k=cards_needed, dim=1, largest=False, sorted=True).indices
    return remaining.gather(1, sampled_indices)

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
            cards_needed = (2 * num_opponents) + board_cards_needed

            for rollout_idx in range(num_rollouts):
                sampled_cards = _sample_remaining_cards(remaining, cards_needed)
                opp_cards = sampled_cards[:, :2 * num_opponents].reshape(group_size, num_opponents, 2)
                board_add = sampled_cards[:, 2 * num_opponents:2 * num_opponents + board_cards_needed]
                full_boards = torch.cat((batch_boards, board_add), dim=1)
                my_ranks, opp_ranks = _rank_rollout_hands(batch_holes, opp_cards, full_boards)
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
        board = deck.deal(min(len(infoset.history), 5))
        equity = simulate_equity_batch([hole], [board])[0]
        pot = Config.POT_SIZE + np.random.uniform(0, 100)  # Wider pot variance
        call_amt = Config.CALL_AMOUNT + np.random.uniform(0, 20)
        fold_equity = np.random.normal(Config.FOLD_EQUITY_MEAN, Config.FOLD_EQUITY_STD)
        bluff = np.random.uniform(0, Config.BLUFF_FACTOR) if action == Action.RAISE else 0
        fold_penalty = (1.0 - equity) * 0.2 if action == Action.FOLD else 0.0
        equity = equity + bluff - fold_penalty
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
    history = infoset.history
    if not history:
        return False
    if history[-1] == Action.FOLD.value:
        return True
    if len(history) >= 2 and history[-1] == Action.CALL.value and history[-2] == Action.CALL.value:
        return True
    return len(history) >= 4