import argparse
import os
import time

from environment import setup_rocmo

setup_rocmo()

import eval7
import numpy as np
import torch
import torch.nn.functional as F

from config import Config, EquityNet
from game import simulate_equity_batch


def card_to_id(card):
    # Convert eval7 cards into the 0-51 encoding already used throughout the project.
    return int(card.rank) * 4 + int(card.suit)


def encode_samples(holes, boards, num_opponents):
    # Encode hole cards and boards into separate multi-hot planes plus street/opponent context.
    features = torch.zeros((len(holes), Config.EQUITY_FEATURE_DIM), dtype=torch.float32)
    for index, (hole, board) in enumerate(zip(holes, boards)):
        for card in hole:
            features[index, card_to_id(card)] = 1.0
        for card in board:
            features[index, 52 + card_to_id(card)] = 1.0
        features[index, 104] = len(board) / 5.0
        features[index, 105] = num_opponents / max(1, Config.MAX_CURRICULUM_OPPONENTS)
    return features


def sample_hand_batch(batch_size):
    # Mix preflop, flop, turn, and river samples so EquityNet generalizes across streets.
    holes = []
    boards = []
    street_sizes = np.random.choice([0, 3, 4, 5], size=batch_size, p=[0.35, 0.35, 0.15, 0.15])
    for board_size in street_sizes:
        deck = eval7.Deck()
        deck.shuffle()
        holes.append(deck.deal(2))
        boards.append(deck.deal(int(board_size)))
    return holes, boards


def build_labeled_batch(batch_size, num_opponents):
    # Reuse the existing Monte Carlo equity simulator so labels match CFR rollouts.
    holes, boards = sample_hand_batch(batch_size)
    features = encode_samples(holes, boards, num_opponents).to(Config.DEVICE)
    labels = torch.tensor(
        simulate_equity_batch(holes, boards, num_opponents=num_opponents),
        dtype=torch.float32,
        device=Config.DEVICE,
    ).unsqueeze(1)
    return features, labels


def evaluate_model(model, batch_size, num_batches, num_opponents, amp_enabled, autocast_device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            features, labels = build_labeled_batch(batch_size, num_opponents)
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                predictions = model(features)
                loss = F.mse_loss(predictions, labels)
            losses.append(loss.detach().float().item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def train_equity_model(args):
    torch.set_float32_matmul_precision('high')
    amp_enabled = Config.DEVICE == 'cuda'
    autocast_device = 'cuda' if amp_enabled else 'cpu'

    model = EquityNet(Config.EQUITY_FEATURE_DIM, Config.EQUITY_HIDDEN_DIM).to(Config.DEVICE)
    if amp_enabled:
        try:
            # Compile the standalone trainer too so long runs benefit from ROCm kernel tuning.
            model = torch.compile(model, mode='max-autotune', fullgraph=False)
        except Exception as compile_error:
            print(f'EquityNet compile fallback: {compile_error}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val_loss = float('inf')
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for _ in range(args.steps_per_epoch):
            features, labels = build_labeled_batch(args.batch_size, args.num_opponents)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=amp_enabled):
                predictions = model(features)
                loss = F.mse_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().float().item())
            global_step += args.batch_size

        val_loss = evaluate_model(
            model,
            batch_size=args.validation_batch_size,
            num_batches=args.validation_batches,
            num_opponents=args.num_opponents,
            amp_enabled=amp_enabled,
            autocast_device=autocast_device,
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        elapsed = time.time() - start_time
        samples_per_second = global_step / max(elapsed, 1e-6)
        print(
            f'Epoch {epoch + 1}/{args.epochs} | Train Loss {train_loss:.6f} | '
            f'Val Loss {val_loss:.6f} | Samples {global_step:,} | {samples_per_second:,.0f} samples/s'
        )

        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_to_save.state_dict(), 'best_equity_model.pth')
            print('Saved improved equity model to best_equity_model.pth')

    print(f'Finished equity training with best validation loss {best_val_loss:.6f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train EquityNet with Monte Carlo equity labels.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--steps-per-epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--validation-batch-size', type=int, default=256)
    parser.add_argument('--validation-batches', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-opponents', type=int, default=Config.NUM_OPPONENTS)
    return parser.parse_args()


if __name__ == '__main__':
    train_equity_model(parse_args())
