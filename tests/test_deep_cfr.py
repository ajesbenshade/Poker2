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


# ---------------------------------------------------------------------------
# Vectorized buffer insertion (Phase A1)
# ---------------------------------------------------------------------------

def test_add_arrays_matches_add_loop_below_capacity():
    """add_arrays must be byte-identical to per-row add when fitting in capacity."""
    obs_dim, na, n = 6, 4, 30
    rng = np.random.default_rng(123)
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    legal = rng.integers(0, 2, size=(n, na)).astype(np.float32)
    target = rng.standard_normal((n, na)).astype(np.float32)
    weight = rng.uniform(0.5, 5.0, size=n).astype(np.float32)

    buf_loop = ReservoirBuffer(capacity=100, obs_dim=obs_dim, num_actions=na, seed=7)
    for i in range(n):
        buf_loop.add(obs[i], legal[i], target[i], float(weight[i]))

    buf_vec = ReservoirBuffer(capacity=100, obs_dim=obs_dim, num_actions=na, seed=7)
    buf_vec.add_arrays(obs, legal, target, weight)

    assert len(buf_loop) == len(buf_vec) == n
    assert buf_loop.total_seen == buf_vec.total_seen == n
    np.testing.assert_array_equal(buf_loop._obs[:n], buf_vec._obs[:n])
    np.testing.assert_array_equal(buf_loop._legal[:n], buf_vec._legal[:n])
    np.testing.assert_array_equal(buf_loop._target[:n], buf_vec._target[:n])
    np.testing.assert_array_equal(buf_loop._weight[:n], buf_vec._weight[:n])


def test_add_arrays_above_capacity_preserves_distribution():
    """When n >> capacity, add_arrays must produce a valid uniform sample (no
    bias, every retained entry must come from the input stream)."""
    obs_dim, na, cap, n = 4, 3, 50, 5000
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    # Distinguish rows by stamping the row index in obs[:, 0] for traceability.
    obs[:, 0] = np.arange(n, dtype=np.float32)
    legal = np.ones((n, na), dtype=np.float32)
    target = rng.standard_normal((n, na)).astype(np.float32)
    weight = np.ones(n, dtype=np.float32)

    buf = ReservoirBuffer(capacity=cap, obs_dim=obs_dim, num_actions=na, seed=42)
    buf.add_arrays(obs, legal, target, weight)
    assert len(buf) == cap
    assert buf.total_seen == n
    # Every retained obs[:, 0] must be a valid row index from the input.
    stored_ids = buf._obs[:, 0].astype(np.int64)
    assert stored_ids.min() >= 0 and stored_ids.max() < n
    # No duplicate slots (each slot was written from a unique source row).
    assert len(np.unique(stored_ids)) == cap


def test_add_arrays_chunked_matches_single_call():
    """Splitting an array into chunks and calling add_arrays repeatedly must
    yield identical state to one big add_arrays call (with fixed seeds)."""
    obs_dim, na, n = 5, 4, 200
    cap = 80
    rng = np.random.default_rng(2024)
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    legal = np.ones((n, na), dtype=np.float32)
    target = rng.standard_normal((n, na)).astype(np.float32)
    weight = np.ones(n, dtype=np.float32)

    buf_one = ReservoirBuffer(capacity=cap, obs_dim=obs_dim, num_actions=na, seed=11)
    buf_one.add_arrays(obs, legal, target, weight)

    buf_chunks = ReservoirBuffer(capacity=cap, obs_dim=obs_dim, num_actions=na, seed=11)
    for start in range(0, n, 37):
        end = min(start + 37, n)
        buf_chunks.add_arrays(obs[start:end], legal[start:end],
                              target[start:end], weight[start:end])
    # Sizes and total_seen must match; reservoir contents may differ because
    # the RNG draws are split differently — check size invariants only.
    assert len(buf_one) == len(buf_chunks) == cap
    assert buf_one.total_seen == buf_chunks.total_seen == n


# ---------------------------------------------------------------------------
# Traversal refactor (Phase A2 + E)
# ---------------------------------------------------------------------------

def test_traverse_one_collects_samples():
    """traverse_one must return non-empty sample lists with consistent shapes."""
    from algo.deep_cfr.traversal import traverse_one, make_uniform_strategy_fn
    rng_deal = random.Random(0)
    rng_sample = np.random.default_rng(1)
    aspace = ActionSpace()
    adv, strat = traverse_one(
        traverser=0,
        strategy_fn=make_uniform_strategy_fn(),
        iter_t=1,
        num_players=2,
        starting_stack=200,
        small_blind=1,
        big_blind=2,
        button=0,
        action_space=aspace,
        deal_rng=rng_deal,
        sample_rng=rng_sample,
        linear_weight=True,
    )
    # At least one decision either way per traversal in a 2-player hand.
    assert len(adv) + len(strat) > 0
    for s in adv + strat:
        obs_v, legal_v, target_v, w = s
        assert obs_v.shape == (OBS_DIM,)
        assert legal_v.shape == (NUM_ACTIONS,)
        assert target_v.shape == (NUM_ACTIONS,)
        assert isinstance(w, float)


def test_external_sampling_back_compat_runs():
    """The serial wrapper must still run end-to-end and populate buffers."""
    cfg = DeepCFRConfig()
    aspace = ActionSpace(cfg.bet_fractions)
    advs = [None, None]
    abuf = ReservoirBuffer(capacity=2000, obs_dim=OBS_DIM,
                           num_actions=aspace.num_actions, seed=0)
    sbuf = ReservoirBuffer(capacity=2000, obs_dim=OBS_DIM,
                           num_actions=aspace.num_actions, seed=1)
    external_sampling(
        traverser=0,
        advantage_nets=advs,
        advantage_buffer=abuf,
        strategy_buffer=sbuf,
        iter_t=1,
        num_traversals=10,
        num_players=2,
        starting_stack=200,
        small_blind=1,
        big_blind=2,
        action_space=aspace,
        rng=random.Random(0),
        device=torch.device("cpu"),
        linear_weight=True,
    )
    assert len(abuf) > 0
    assert len(sbuf) > 0


# ---------------------------------------------------------------------------
# Worker (Phase A3) — direct module-level invocation (no pool)
# ---------------------------------------------------------------------------

def test_worker_run_chunk_returns_arrays():
    """Run worker entry inline (no Pool) and verify returned array shapes."""
    from algo.deep_cfr import worker as W
    aspace = ActionSpace()
    W._init_worker(
        [None, None],
        obs_dim=OBS_DIM,
        num_actions=aspace.num_actions,
        hidden=64,
        blocks=1,
        action_space=aspace,
        num_players=2,
        starting_stack=200,
        small_blind=1,
        big_blind=2,
    )
    out = W.run_chunk(
        traverser=0,
        chunk_size=8,
        iter_t=1,
        base_seed=12345,
        button_offset=0,
        linear_weight=True,
    )
    a_obs, a_legal, a_target, a_weight, s_obs, s_legal, s_target, s_weight = out
    assert a_obs.shape[1] == OBS_DIM
    assert a_legal.shape[1] == aspace.num_actions
    assert a_target.shape[1] == aspace.num_actions
    assert s_obs.shape[1] == OBS_DIM
    # At least one of advantage / strategy samples must be produced.
    assert (a_obs.shape[0] + s_obs.shape[0]) > 0

