import torch
import numpy as np
import multiprocessing as mp

from datatypes import Action, Street
from config import Config

try:
    import eval7
except ImportError:
    eval7 = None
    from treys import Card as TreysCard, Deck as TreysDeck, Evaluator as TreysEvaluator

    treys_evaluator = TreysEvaluator()


RANKS_STR = '23456789TJQKA'
SUITS_STR = 'shdc'


def _card_string(card_id):
    rank_id = card_id // 4
    suit_id = card_id % 4
    return RANKS_STR[rank_id] + SUITS_STR[suit_id]


def evaluate_hand(cards_ids):
    if eval7 is not None:
        cards = [eval7.Card(_card_string(card_id)) for card_id in cards_ids]
        return eval7.evaluate(cards)

    hand = [TreysCard.new(_card_string(card_id)) for card_id in cards_ids[:2]]
    board = [TreysCard.new(_card_string(card_id)) for card_id in cards_ids[2:]]
    return -treys_evaluator.evaluate(board, hand)


def _card_to_id(card):
    if eval7 is not None:
        return card.rank * 4 + card.suit

    card_str = TreysCard.int_to_str(card)
    rank_id = RANKS_STR.index(card_str[0])
    suit_id = SUITS_STR.index(card_str[1].lower())
    return rank_id * 4 + suit_id


def _new_deck():
    return eval7.Deck() if eval7 is not None else TreysDeck()


def _deal_cards(deck, count):
    if eval7 is not None:
        return deck.deal(count)
    drawn = deck.draw(count)
    return drawn if isinstance(drawn, list) else [drawn]


def hand_rank_tensor(hands):
    batch_size = hands.shape[0]
    hands_list = hands.cpu().numpy().tolist()
    process_count = max(1, Config.HAND_EVAL_PROCESSES)
    if process_count == 1:
        ranks = [evaluate_hand(hand) for hand in hands_list]
    else:
        with mp.Pool(processes=process_count) as pool:
            ranks = pool.map(evaluate_hand, hands_list)
    return torch.tensor(ranks, dtype=torch.int32, device=Config.DEVICE)


def _card_from_id(card_id):
    card_str = _card_string(card_id)
    return eval7.Card(card_str) if eval7 is not None else TreysCard.new(card_str)


def _prepare_deck(infoset):
    deck = _new_deck()
    known_cards = []
    for card_id in getattr(infoset, 'private_cards', ()):
        known_cards.append(_card_from_id(card_id))
    for card_id in getattr(infoset, 'board_cards', ()):
        known_cards.append(_card_from_id(card_id))
    if known_cards:
        deck.cards = [card for card in deck.cards if card not in known_cards]
    deck.shuffle()
    return deck


def _board_cards_to_draw(infoset):
    street = getattr(infoset, 'street', None)
    if street is None:
        return len(getattr(infoset, 'history', ())) % 5
    board_cards_by_street = {
        Street.PREFLOP: 0,
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
        Street.SHOWDOWN: 5,
    }
    return board_cards_by_street[street]


def simulate_equity_batch(holes, boards, num_opponents=Config.NUM_OPPONENTS):
    num_hands = len(holes)
    equities = torch.zeros(num_hands, device=Config.DEVICE)
    batch_size = min(Config.BATCH_SIZE, num_hands)
    for start in range(0, num_hands, batch_size):
        end = min(start + batch_size, num_hands)
        batch_holes = torch.tensor([[_card_to_id(card) for card in hand] for hand in holes[start:end]], dtype=torch.int64, device=Config.DEVICE)
        batch_boards = torch.tensor([[_card_to_id(card) for card in board] for board in boards[start:end]], dtype=torch.int64, device=Config.DEVICE)
        full_deck = torch.arange(52, dtype=torch.int64, device=Config.DEVICE).unsqueeze(0).repeat(end-start, 1)
        used = torch.cat((batch_holes, batch_boards), dim=1)
        mask = torch.zeros((end-start, 52), device=Config.DEVICE).scatter_(1, used, 1).bool()
        remaining = full_deck[~mask].reshape(end-start, -1)
        num_rollouts = Config.EQUITY_ROLLOUTS
        rollout_wins = torch.zeros((end-start, num_rollouts), device=Config.DEVICE)
        for r in range(num_rollouts):
            perm = torch.stack([
                torch.randperm(remaining.shape[1], device=Config.DEVICE)
                for _ in range(end - start)
            ], dim=0)
            sampled_cards = remaining.gather(1, perm)
            opp_cards = sampled_cards[:, :2*num_opponents].reshape(end-start, num_opponents, 2)
            board_add = sampled_cards[:, 2*num_opponents:2*num_opponents + max(0,5-batch_boards.shape[1])]
            full_boards = torch.cat((batch_boards, board_add), dim=1)
            my_hands = torch.cat((batch_holes, full_boards), dim=1)
            my_ranks = hand_rank_tensor(my_hands)
            opp_ranks = torch.stack([hand_rank_tensor(torch.cat((opp_cards[:,j], full_boards), dim=1)) for j in range(num_opponents)], dim=1)
            rollout_wins[:, r] = (my_ranks.unsqueeze(1) > opp_ranks).float().mean(dim=1)
        wins = rollout_wins.mean(dim=1)
        equities[start:end] = wins + torch.randn(end-start, device=Config.DEVICE) * Config.EQUITY_STD
    return equities.to(torch.float32).cpu().numpy()


def simulate_action_batch(infosets, actions):
    results = np.zeros(len(infosets))
    for i, (infoset, action) in enumerate(zip(infosets, actions)):
        if action is None:
            raise ValueError("simulate_action_batch requires an explicit Action")
        deck = _prepare_deck(infoset)
        private_cards = getattr(infoset, 'private_cards', ())
        board_cards = getattr(infoset, 'board_cards', ())
        hole = [_card_from_id(card_id) for card_id in private_cards] if private_cards else _deal_cards(deck, 2)
        board = [_card_from_id(card_id) for card_id in board_cards] if board_cards else _deal_cards(deck, _board_cards_to_draw(infoset))
        equity = simulate_equity_batch([hole], [board])[0]
        pot = float(getattr(infoset, 'pot_size', Config.POT_SIZE))
        call_amt = float(getattr(infoset, 'current_bet', Config.CALL_AMOUNT))
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
    if getattr(infoset, 'street', None) == Street.SHOWDOWN:
        return True
    return len(infoset.history) >= 4 or getattr(infoset, 'effective_stack', 1.0) <= 0.0
