import torch
import numpy as np
import eval7
import multiprocessing as mp

from datatypes import Action
from config import Config


def evaluate_hand(cards_ids):
    ranks_str = '23456789TJQKA'
    suits_str = 'shdc'
    cards = []
    for card_id in cards_ids:
        rank_id = card_id // 4
        suit_id = card_id % 4
        card_str = ranks_str[rank_id] + suits_str[suit_id]
        cards.append(eval7.Card(card_str))
    return eval7.evaluate(cards)


def hand_rank_tensor(hands):
    batch_size = hands.shape[0]
    hands_list = hands.cpu().numpy().tolist()
    with mp.Pool(processes=24) as pool:
        ranks = pool.map(evaluate_hand, hands_list)
    return torch.tensor(ranks, dtype=torch.int32, device=Config.DEVICE)


def simulate_equity_batch(holes, boards, num_opponents=Config.NUM_OPPONENTS):
    num_hands = len(holes)
    equities = torch.zeros(num_hands, device=Config.DEVICE)
    batch_size = min(Config.BATCH_SIZE, num_hands)
    for start in range(0, num_hands, batch_size):
        end = min(start + batch_size, num_hands)
        batch_holes = torch.tensor([[c.rank * 4 + c.suit for c in h] for h in holes[start:end]], dtype=torch.int64, device=Config.DEVICE)
        batch_boards = torch.tensor([[c.rank * 4 + c.suit for c in b] for b in boards[start:end]], dtype=torch.int64, device=Config.DEVICE)
        full_deck = torch.arange(52, dtype=torch.int64, device=Config.DEVICE).unsqueeze(0).repeat(end-start, 1)
        used = torch.cat((batch_holes, batch_boards), dim=1)
        mask = torch.zeros((end-start, 52), device=Config.DEVICE).scatter_(1, used, 1).bool()
        remaining = full_deck[~mask].reshape(end-start, -1)
        num_rollouts = Config.EQUITY_ROLLOUTS
        rollout_wins = torch.zeros((end-start, num_rollouts), device=Config.DEVICE)
        for r in range(num_rollouts):
            perm = torch.randperm(remaining.shape[1], device=Config.DEVICE).repeat(end-start, 1)
            opp_cards = perm[:, :2*num_opponents].reshape(end-start, num_opponents, 2)
            board_add = perm[:, 2*num_opponents:2*num_opponents + max(0,5-batch_boards.shape[1])]
            full_boards = torch.cat((batch_boards, board_add), dim=1)
            my_hands = torch.cat((batch_holes, full_boards), dim=1)
            my_ranks = hand_rank_tensor(my_hands)
            opp_ranks = torch.stack([hand_rank_tensor(torch.cat((opp_cards[:,j], full_boards), dim=1)) for j in range(num_opponents)], dim=1)
            rollout_wins[:, r] = (my_ranks.unsqueeze(1) < opp_ranks).float().mean(dim=1)
        wins = rollout_wins.mean(dim=1)
        equities[start:end] = wins + torch.randn(end-start, device=Config.DEVICE) * Config.EQUITY_STD
    return equities.to(torch.float32).cpu().numpy()


def simulate_action_batch(infosets, actions):
    results = np.zeros(len(infosets))
    for i, (infoset, action) in enumerate(zip(infosets, actions)):
        if action is None:
            raise ValueError("simulate_action_batch requires an explicit Action")
        deck = eval7.Deck()
        deck.shuffle()
        hole = deck.deal(2)
        board = deck.deal(len(infoset.history) % 5)
        equity = simulate_equity_batch([hole], [board])[0]
        pot = Config.POT_SIZE + np.random.uniform(0, 100)
        call_amt = Config.CALL_AMOUNT + np.random.uniform(0, 20)
        fold_equity = np.random.normal(Config.FOLD_EQUITY_MEAN, Config.FOLD_EQUITY_STD)
        bluff = np.random.uniform(0, Config.BLUFF_FACTOR) if action == Action.RAISE else 0
        if action == Action.FOLD:
            equity -= (1 - equity) * 0.2
            results[i] = -call_amt / pot * Config.FOLD_PENALTY
        elif action == Action.CALL:
            results[i] = equity * pot / (pot + call_amt) - call_amt / pot
        elif action == Action.RAISE:
            raise_amt = call_amt * (Config.RAISE_MULTIPLIER + np.random.uniform(0, 2))
            new_pot = pot + raise_amt * (1 - fold_equity)
            results[i] = equity * new_pot / (new_pot + raise_amt) - raise_amt / pot + fold_equity * pot / new_pot * bluff
        else:
            raise ValueError(f"Unknown action: {action}")
    return results


def simulate_action(infoset, action: Action):
    return simulate_action_batch([infoset], [action])[0]


def terminal(infoset):
    return len(infoset.history) >= 4
