"""Tests for the Deep CFR pieces."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from algo.deep_cfr import (
    AdvantageNet,
    DeepCFRConfig,
    DeepCFRTrainer,
    PolicyNet,
    ReservoirBuffer,
    external_sampling,
    regret_matching,
)
from engine import OBS_DIM, NUM_ACTIONS
from engine.actions import ActionSpace


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

def test_reservoir_holds_all_below_capacity():
    buf = ReservoirBuffer(capacity=10, obs_dim=4, num_actions=3, seed=0)
    for i in range(7):
        buf.add(np.full(4, i, np.float32), np.ones(3, np.float32),
                np.full(3, i, np.float32), float(i))
    assert len(buf) == 7
    obs, legal, target, weight = buf.sample(7)
    assert obs.shape == (7, 4)
    assert legal.shape == (7, 3)
    assert target.shape == (7, 3)
    assert weight.shape == (7,)


def test_reservoir_replaces_above_capacity():
    buf = ReservoirBuffer(capacity=5, obs_dim=2, num_actions=2, seed=42)
    for i in range(50):
        buf.add(np.full(2, i, np.float32), np.ones(2, np.float32),
                np.full(2, i, np.float32), 1.0)
    assert len(buf) == 5
    assert buf.total_seen == 50


# ---------------------------------------------------------------------------
# Regret matching
# ---------------------------------------------------------------------------

def test_regret_matching_uniform_when_all_negative():
    legal = np.array([1, 1, 0, 1], np.float32)
    advantages = np.array([-1, -2, 5, -0.5], np.float32)
    sigma = regret_matching(advantages, legal)
    # Illegal action gets 0
    assert sigma[2] == 0.0
    # The other three split uniformly = 1/3 each
    np.testing.assert_allclose(sigma[[0, 1, 3]], 1 / 3, atol=1e-6)


def test_regret_matching_normalizes_positive():
    legal = np.array([1, 1, 1], np.float32)
    advantages = np.array([1, 3, 0], np.float32)
    sigma = regret_matching(advantages, legal)
    np.testing.assert_allclose(sigma, [0.25, 0.75, 0.0])


def test_regret_matching_masks_illegal():
    legal = np.array([1, 0, 1], np.float32)
    advantages = np.array([1, 100, 1], np.float32)
    sigma = regret_matching(advantages, legal)
    # Illegal action 1 should be 0 even though it has the largest advantage
    assert sigma[1] == 0.0
    np.testing.assert_allclose(sigma[[0, 2]], 0.5)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def test_advantage_net_shape():
    net = AdvantageNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    obs = torch.randn(4, OBS_DIM)
    legal = torch.ones(4, NUM_ACTIONS)
    out = net(obs, legal)
    assert out.shape == (4, NUM_ACTIONS)


def test_policy_net_strategy_masks_illegal():
    net = PolicyNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    obs = torch.randn(2, OBS_DIM)
    legal = torch.zeros(2, NUM_ACTIONS)
    legal[:, 0] = 1.0
    legal[:, 3] = 1.0
    probs = net.strategy(obs, legal)
    assert probs.shape == (2, NUM_ACTIONS)
    # Probabilities of illegal actions are zero
    illegal_mask = legal < 0.5
    assert torch.all(probs[illegal_mask] == 0.0)
    # Probabilities of legal actions sum to 1
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Traversal smoke
# ---------------------------------------------------------------------------

def test_external_sampling_populates_buffers():
    space = ActionSpace()
    adv_buf = ReservoirBuffer(1024, OBS_DIM, space.num_actions, seed=0)
    strat_buf = ReservoirBuffer(1024, OBS_DIM, space.num_actions, seed=1)
    nets = [None, None]   # iter 0: uniform strategies
    external_sampling(
        traverser=0,
        advantage_nets=nets,
        advantage_buffer=adv_buf,
        strategy_buffer=strat_buf,
        iter_t=1,
        num_traversals=20,
        num_players=2,
        starting_stack=20,
        small_blind=1,
        big_blind=2,
        action_space=space,
        rng=random.Random(0),
        device=torch.device("cpu"),
    )
    assert len(adv_buf) > 0
    assert len(strat_buf) > 0


# ---------------------------------------------------------------------------
# Tiny end-to-end smoke
# ---------------------------------------------------------------------------

def test_trainer_runs_one_iteration(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=1,
        traversals_per_iter=8,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=512,
        strategy_buffer_size=512,
        advantage_train_steps=2,
        strategy_train_steps=2,
        train_batch_size=16,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 1
    # latest.pt should exist
    import os
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))
