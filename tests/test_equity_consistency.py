import unittest

import eval7
import numpy as np
import torch

from environment import initialize_rocm_runtime

initialize_rocm_runtime()

from config import Config
from game import _card_to_id, _sample_remaining_cards, _sanitize_equity_tensor, hand_rank_tensor, simulate_equity_batch


def _reference_sample_remaining_cards(remaining, cards_needed):
    if cards_needed <= 0:
        return torch.empty((remaining.shape[0], 0), dtype=remaining.dtype, device=remaining.device)

    random_scores = torch.rand((remaining.shape[0], remaining.shape[1]), device=remaining.device)
    shuffled_indices = torch.argsort(random_scores, dim=1)
    return remaining.gather(1, shuffled_indices[:, :cards_needed])


def _reference_simulate_equity_batch(holes, boards, num_opponents):
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
            board_cards_needed = max(0, 5 - board_len)
            cards_needed = (2 * num_opponents) + board_cards_needed
            rollout_wins = torch.zeros((group_size, Config.EQUITY_ROLLOUTS), device=Config.DEVICE)

            for rollout_idx in range(Config.EQUITY_ROLLOUTS):
                sampled_cards = _reference_sample_remaining_cards(remaining, cards_needed)
                opp_cards = sampled_cards[:, :2 * num_opponents].reshape(group_size, num_opponents, 2)
                board_add = sampled_cards[:, 2 * num_opponents:2 * num_opponents + board_cards_needed]
                full_boards = torch.cat((batch_boards, board_add), dim=1)
                player_hands = torch.cat((batch_holes.unsqueeze(1), opp_cards), dim=1)
                board_tiles = full_boards.unsqueeze(1).expand(group_size, num_opponents + 1, full_boards.shape[1])
                all_hands = torch.cat((player_hands, board_tiles), dim=2).reshape(group_size * (num_opponents + 1), -1)
                all_ranks = hand_rank_tensor(all_hands).view(group_size, num_opponents + 1)
                my_ranks = all_ranks[:, 0]
                opp_ranks = all_ranks[:, 1:]
                wins = (my_ranks.unsqueeze(1) > opp_ranks).float()
                ties = (my_ranks.unsqueeze(1) == opp_ranks).float() * 0.5
                rollout_wins[:, rollout_idx] = (wins + ties).mean(dim=1)

            wins = rollout_wins.mean(dim=1)
            noisy_equities = _sanitize_equity_tensor(wins + torch.randn(group_size, device=Config.DEVICE) * Config.EQUITY_STD)
            for offset_index, local_index in enumerate(group_offsets):
                equities[start + local_index] = noisy_equities[offset_index]

    return _sanitize_equity_tensor(equities).to(torch.float32).cpu().numpy()


class EquityConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.prev_rollouts = Config.EQUITY_ROLLOUTS
        self.prev_std = Config.EQUITY_STD
        Config.EQUITY_ROLLOUTS = 4
        Config.EQUITY_STD = 0.0

    def tearDown(self):
        Config.EQUITY_ROLLOUTS = self.prev_rollouts
        Config.EQUITY_STD = self.prev_std

    def test_sample_remaining_cards_matches_reference(self):
        remaining = torch.arange(52, dtype=torch.int64, device=Config.DEVICE).view(1, 52).repeat(8, 1)
        cards_needed = 9

        torch.manual_seed(1234)
        sampled = _sample_remaining_cards(remaining, cards_needed)
        torch.manual_seed(1234)
        reference = _reference_sample_remaining_cards(remaining, cards_needed)

        self.assertTrue(torch.equal(sampled.cpu(), reference.cpu()))

    def test_simulate_equity_batch_matches_reference(self):
        holes = []
        boards = []
        for _ in range(6):
            deck = eval7.Deck()
            deck.shuffle()
            holes.append(deck.deal(2))
            boards.append(deck.deal(3))

        torch.manual_seed(4321)
        np.random.seed(4321)
        optimized = simulate_equity_batch(holes, boards, num_opponents=2)

        torch.manual_seed(4321)
        np.random.seed(4321)
        reference = _reference_simulate_equity_batch(holes, boards, num_opponents=2)

        self.assertTrue(np.allclose(optimized, reference, atol=1e-6))


if __name__ == "__main__":
    unittest.main()