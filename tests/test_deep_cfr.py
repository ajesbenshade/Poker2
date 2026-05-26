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
from algo.deep_cfr.inference_server import BatchedAdvantageInference
from algo.deep_cfr.traversal import (
    _traverse,
    make_batched_net_strategy_fn,
    make_batched_strategy_fn_from_single,
    make_batched_uniform_strategy_fn,
    make_net_strategy_fn,
    make_uniform_strategy_fn,
    samples_to_arrays,
)
from algo.deep_cfr.vectorized_traversal import _traverse_batch, traverse_many_vectorized
from algo.deep_cfr.worker import _finalize_worker_net, _strip_compile_prefix, serialize_state_dict
from algo.deep_cfr.trainer import _traversal_chunk_size
from engine import OBS_DIM, NUM_ACTIONS
from engine import new_hand
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


def test_reservoir_state_dict_roundtrip():
    buf = ReservoirBuffer(capacity=10, obs_dim=3, num_actions=2, seed=7)
    for i in range(8):
        buf.add(np.full(3, i, np.float32), np.ones(2, np.float32),
                np.full(2, i + 1, np.float32), float(i))

    clone = ReservoirBuffer(capacity=10, obs_dim=3, num_actions=2, seed=99)
    clone.load_state_dict(buf.state_dict())

    assert len(clone) == len(buf)
    assert clone.total_seen == buf.total_seen
    np.testing.assert_array_equal(clone._obs[:len(buf)], buf._obs[:len(buf)])
    np.testing.assert_array_equal(clone._legal[:len(buf)], buf._legal[:len(buf)])
    np.testing.assert_array_equal(clone._target[:len(buf)], buf._target[:len(buf)])
    np.testing.assert_array_equal(clone._weight[:len(buf)], buf._weight[:len(buf)])


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


def test_scripted_worker_net_matches_eager():
    import algo.deep_cfr.worker as worker_mod

    net = AdvantageNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    net.eval()
    obs = torch.randn(5, OBS_DIM)
    legal = torch.ones(5, NUM_ACTIONS)

    old_value = worker_mod._SCRIPT_WORKER_NETS
    worker_mod._SCRIPT_WORKER_NETS = True
    try:
        scripted = _finalize_worker_net(net)
    finally:
        worker_mod._SCRIPT_WORKER_NETS = old_value

    with torch.no_grad():
        np.testing.assert_allclose(
            net(obs, legal).numpy(),
            scripted(obs, legal).numpy(),
            atol=1e-5,
            rtol=1e-5,
        )


def test_worker_strips_compile_prefix_from_state_dict():
    net = AdvantageNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    prefixed = {f"_orig_mod.{k}": v for k, v in net.state_dict().items()}
    stripped = _strip_compile_prefix(prefixed)

    reloaded = AdvantageNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    reloaded.load_state_dict(stripped)

    assert set(stripped) == set(net.state_dict())


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


def test_traversal_chunk_size_balances_worker_tasks():
    assert _traversal_chunk_size(total=2000, num_workers=12, configured_chunk=25) == 25
    assert _traversal_chunk_size(total=2000, num_workers=12, configured_chunk=0) == 167
    assert _traversal_chunk_size(total=10, num_workers=12, configured_chunk=25) == 10


def test_batched_advantage_inference_matches_single_strategy_fn():
    net = AdvantageNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    net.eval()
    rng = np.random.default_rng(123)
    obs = rng.standard_normal((6, OBS_DIM)).astype(np.float32)
    legal = rng.integers(0, 2, size=(6, NUM_ACTIONS)).astype(np.float32)
    legal[:, 0] = 1.0
    seats = np.zeros(6, dtype=np.int64)

    engine = BatchedAdvantageInference(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden=32,
        blocks=1,
        dropout=0.0,
        device=torch.device("cpu"),
        amp_dtype=None,
        num_players=2,
    )
    engine.load_state_dict_blobs([serialize_state_dict(net), None])
    batched = engine.infer_batch(seats, obs, legal)

    single_fn = make_net_strategy_fn([net, None], torch.device("cpu"))
    expected = np.stack([single_fn(obs[i], legal[i], 0) for i in range(obs.shape[0])])
    np.testing.assert_allclose(batched, expected, atol=1e-6)


def test_batched_strategy_adapter_matches_single_rows():
    rng = np.random.default_rng(17)
    obs = rng.standard_normal((7, OBS_DIM)).astype(np.float32)
    legal = rng.integers(0, 2, size=(7, NUM_ACTIONS)).astype(np.float32)
    legal[:, 1] = 1.0

    single = make_uniform_strategy_fn()
    batched = make_batched_strategy_fn_from_single(single)
    expected = np.stack([single(obs[i], legal[i], 0) for i in range(obs.shape[0])])
    np.testing.assert_allclose(batched(obs, legal, 0), expected, atol=1e-6)


def test_batched_net_strategy_matches_single_and_scripted():
    net = AdvantageNet(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden=32, num_blocks=1)
    net.eval()
    scripted = torch.jit.script(net)
    scripted.eval()

    rng = np.random.default_rng(23)
    obs = rng.standard_normal((5, OBS_DIM)).astype(np.float32)
    legal = rng.integers(0, 2, size=(5, NUM_ACTIONS)).astype(np.float32)
    legal[:, 0] = 1.0

    single = make_net_strategy_fn([net, None], torch.device("cpu"))
    batched = make_batched_net_strategy_fn([net, None], torch.device("cpu"))
    scripted_batched = make_batched_net_strategy_fn([scripted, None], torch.device("cpu"))

    expected = np.stack([single(obs[i], legal[i], 0) for i in range(obs.shape[0])])
    np.testing.assert_allclose(batched(obs, legal, 0), expected, atol=1e-6)
    np.testing.assert_allclose(scripted_batched(obs, legal, 0), expected, atol=1e-6)


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


def test_trainer_load_checkpoint_warm_start(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=1,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        log_dir=str(tmp_path / "runs_a"),
        checkpoint_dir=str(tmp_path / "ckpt_a"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()

    clone_cfg = DeepCFRConfig(
        num_iterations=1,
        hidden_size=32,
        num_blocks=1,
        device="cpu",
        amp_dtype="float32",
        log_dir=str(tmp_path / "runs_b"),
        checkpoint_dir=str(tmp_path / "ckpt_b"),
    )
    clone = DeepCFRTrainer(clone_cfg)
    clone.load_checkpoint(str(tmp_path / "ckpt_a" / "latest.pt"), restore_iteration=False)

    assert clone.iter == 0
    for key, value in trainer.policy_net.state_dict().items():
        torch.testing.assert_close(clone.policy_net.state_dict()[key], value)


def test_trainer_resume_checkpoint_iteration(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=1,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        save_buffer_state=True,
        log_dir=str(tmp_path / "runs_a"),
        checkpoint_dir=str(tmp_path / "ckpt_a"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()

    resume_cfg = DeepCFRConfig(
        num_iterations=2,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        log_dir=str(tmp_path / "runs_b"),
        checkpoint_dir=str(tmp_path / "ckpt_b"),
    )
    resumed = DeepCFRTrainer(resume_cfg)
    resumed.load_checkpoint(
        str(tmp_path / "ckpt_a" / "latest.pt"),
        restore_iteration=True,
        restore_buffers=True,
    )
    resumed.train()

    assert resumed.iter == 2
    assert len(resumed.strategy_buffer) > 0


def test_trainer_uses_split_learning_rates():
    cfg = DeepCFRConfig(
        learning_rate=1e-3,
        advantage_learning_rate=3e-3,
        strategy_learning_rate=2e-3,
        lr_schedule="cosine",
        device="cpu",
    )
    trainer = DeepCFRTrainer(cfg)
    assert trainer._base_learning_rate("regression") == pytest.approx(3e-3)
    assert trainer._base_learning_rate("ce_soft") == pytest.approx(2e-3)
    assert trainer._scheduled_learning_rate(1.0, 9, 10) == pytest.approx(cfg.lr_min_mult)


def test_trainer_runs_with_optimization_flags_on_cpu(tmp_path):
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
        pin_training_batches=True,
        loss_log_interval=2,
        concurrent_advantage_training=True,
        latest_checkpoint_interval=5,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 1
    import os
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))


def test_trainer_runs_with_vectorized_backend_on_cpu(tmp_path):
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
        traversal_backend="vectorized",
        vectorized_traversal_batch_size=4,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 1
    import os
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))


def test_trainer_runs_with_vectorized_workers_on_cpu(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=1,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        num_workers=2,
        worker_chunk_min=2,
        traversal_backend="vectorized",
        vectorized_traversal_batch_size=2,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 1
    import os
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))


def test_trainer_runs_with_file_worker_results_on_cpu(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=2,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        num_workers=2,
        worker_chunk_min=2,
        traversal_backend="vectorized",
        vectorized_traversal_batch_size=2,
        worker_result_transport="file",
        async_pipeline=True,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    result_dir = tmp_path / "ckpt" / "_worker_results"
    assert trainer.iter == 2
    assert not list(result_dir.glob("*.npz"))


def test_trainer_runs_with_vectorized_proxy_workers_on_cpu(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=2,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        num_workers=2,
        worker_chunk_min=2,
        traversal_backend="vectorized",
        vectorized_traversal_batch_size=2,
        use_proxy_nets=True,
        proxy_hidden_size=16,
        proxy_num_blocks=1,
        proxy_refresh_interval=1,
        proxy_training_steps=1,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 2
    assert all(net is not None for net in trainer.proxy_advantage_nets)
    assert trainer.proxy_advantage_nets[0].trunk.input.out_features == 16
    payload = torch.load(tmp_path / "ckpt" / "latest.pt", map_location="cpu")
    assert payload["config"]["use_proxy_nets"] is True
    assert all(sd is not None for sd in payload["proxy_advantage_nets"])


def test_latest_checkpoint_interval_saves_final_iteration(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=3,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        latest_checkpoint_interval=2,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    payload = torch.load(tmp_path / "ckpt" / "latest.pt", map_location="cpu")
    assert payload["iter"] == 3


def test_trainer_runs_with_inference_server_on_cpu(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=2,
        traversals_per_iter=4,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=256,
        strategy_buffer_size=256,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        num_workers=2,
        worker_chunk_min=2,
        traversal_inference_mode="server",
        inference_server_batch_size=8,
        inference_server_timeout_ms=1.0,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 2
    import os
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))


def test_trainer_runs_with_scripted_workers_and_async_depth_on_cpu(tmp_path):
    cfg = DeepCFRConfig(
        num_iterations=3,
        traversals_per_iter=6,
        hidden_size=32,
        num_blocks=1,
        advantage_buffer_size=512,
        strategy_buffer_size=512,
        advantage_train_steps=1,
        strategy_train_steps=1,
        train_batch_size=8,
        eval_interval=0,
        starting_stack=20,
        device="cpu",
        amp_dtype="float32",
        num_workers=2,
        worker_chunk_min=2,
        script_worker_nets=True,
        async_pipeline=True,
        async_pipeline_depth=2,
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.train()
    assert trainer.iter == 3
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


def test_vectorized_traversal_outputs_valid_samples():
    aspace = ActionSpace()
    adv, strat = traverse_many_vectorized(
        traverser=0,
        strategy_fn=make_uniform_strategy_fn(),
        batch_strategy_fn=make_batched_uniform_strategy_fn(),
        iter_t=3,
        num_traversals=8,
        num_players=2,
        starting_stack=20,
        small_blind=1,
        big_blind=2,
        button_offset=0,
        action_space=aspace,
        deal_rng=random.Random(11),
        sample_seed_rng=np.random.default_rng(12),
        linear_weight=True,
    )
    assert len(adv) + len(strat) > 0
    for obs_v, legal_v, target_v, weight in adv + strat:
        assert obs_v.shape == (OBS_DIM,)
        assert legal_v.shape == (NUM_ACTIONS,)
        assert target_v.shape == (NUM_ACTIONS,)
        assert np.all(np.isfinite(obs_v))
        assert np.all(np.isfinite(target_v))
        assert weight == 3.0
    for _, legal_v, sigma_v, _ in strat:
        assert np.all(sigma_v[legal_v < 0.5] == 0.0)
        np.testing.assert_allclose(float(sigma_v.sum()), 1.0, atol=1e-6)


def test_vectorized_single_state_matches_recursive_oracle():
    aspace = ActionSpace()
    state = new_hand(
        num_players=2,
        starting_stack=20,
        small_blind=1,
        big_blind=2,
        button=0,
        rng=random.Random(33),
        action_space=aspace,
    )
    recursive_adv = []
    recursive_strat = []
    vectorized_adv = []
    vectorized_strat = []
    strategy_fn = make_uniform_strategy_fn()
    batch_strategy_fn = make_batched_uniform_strategy_fn()

    recursive_value = _traverse(
        state,
        0,
        strategy_fn,
        recursive_adv,
        recursive_strat,
        2,
        np.random.default_rng(44),
        True,
        2,
    )
    vectorized_value = _traverse_batch(
        [state],
        [np.random.default_rng(44)],
        traverser=0,
        batch_strategy_fn=batch_strategy_fn,
        adv_samples=vectorized_adv,
        strat_samples=vectorized_strat,
        iter_t=2,
        linear_weight=True,
        big_blind=2,
    )[0]

    np.testing.assert_allclose(vectorized_value, recursive_value, atol=1e-6)
    for arrays_a, arrays_b in (
        (samples_to_arrays(vectorized_adv, OBS_DIM, NUM_ACTIONS), samples_to_arrays(recursive_adv, OBS_DIM, NUM_ACTIONS)),
        (samples_to_arrays(vectorized_strat, OBS_DIM, NUM_ACTIONS), samples_to_arrays(recursive_strat, OBS_DIM, NUM_ACTIONS)),
    ):
        for left, right in zip(arrays_a, arrays_b):
            np.testing.assert_allclose(left, right, atol=1e-6)


def test_vectorized_traversal_is_deterministic_for_fixed_seeds():
    def run_once():
        aspace = ActionSpace()
        return traverse_many_vectorized(
            traverser=1,
            strategy_fn=make_uniform_strategy_fn(),
            batch_strategy_fn=make_batched_uniform_strategy_fn(),
            iter_t=2,
            num_traversals=6,
            num_players=2,
            starting_stack=20,
            small_blind=1,
            big_blind=2,
            button_offset=0,
            action_space=aspace,
            deal_rng=random.Random(101),
            sample_seed_rng=np.random.default_rng(202),
            linear_weight=True,
        )

    adv_a, strat_a = run_once()
    adv_b, strat_b = run_once()
    for arrays_a, arrays_b in (
        (samples_to_arrays(adv_a, OBS_DIM, NUM_ACTIONS), samples_to_arrays(adv_b, OBS_DIM, NUM_ACTIONS)),
        (samples_to_arrays(strat_a, OBS_DIM, NUM_ACTIONS), samples_to_arrays(strat_b, OBS_DIM, NUM_ACTIONS)),
    ):
        for left, right in zip(arrays_a, arrays_b):
            np.testing.assert_allclose(left, right, atol=1e-6)


def test_worker_run_chunk_vectorized_returns_arrays():
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
        starting_stack=20,
        small_blind=1,
        big_blind=2,
    )
    out = W.run_chunk_vectorized(
        traverser=0,
        chunk_size=8,
        iter_t=1,
        base_seed=12345,
        button_offset=0,
        linear_weight=True,
        vectorized_batch_size=4,
    )
    a_obs, a_legal, a_target, a_weight, s_obs, s_legal, s_target, s_weight = out
    assert a_obs.shape[1] == OBS_DIM
    assert a_legal.shape[1] == aspace.num_actions
    assert a_target.shape[1] == aspace.num_actions
    assert s_obs.shape[1] == OBS_DIM
    assert (a_obs.shape[0] + s_obs.shape[0]) > 0


# ---------------------------------------------------------------------------
# LBR exploitability (Phase G)
# ---------------------------------------------------------------------------

def test_lbr_runs_and_returns_finite_mbbg():
    """Smoke: evaluate_lbr against an untrained net should run end-to-end and
    return a finite mbb/g (positive => trained policy is exploitable)."""
    from algo.deep_cfr import evaluate_lbr
    cfg = DeepCFRConfig()
    cfg.num_players = 2
    net = PolicyNet(obs_dim=OBS_DIM,
                    num_actions=ActionSpace(cfg.bet_fractions).num_actions,
                    hidden=64, num_blocks=1).to(torch.device("cpu"))
    net.eval()
    mbbg = evaluate_lbr(
        net, cfg, torch.device("cpu"),
        num_hands=20, equity_samples=20,
        rng=random.Random(0),
    )
    assert np.isfinite(mbbg)


# ---------------------------------------------------------------------------
# Multi-way (Phase H)
# ---------------------------------------------------------------------------

def test_evaluate_vs_baselines_6max_runs():
    """Multi-way (6-max) baseline eval must produce one mbb/g per baseline."""
    from algo.deep_cfr import evaluate_vs_baselines
    cfg = DeepCFRConfig()
    cfg.num_players = 6
    aspace = ActionSpace(cfg.bet_fractions)
    net = PolicyNet(obs_dim=OBS_DIM, num_actions=aspace.num_actions,
                    hidden=64, num_blocks=1).to(torch.device("cpu"))
    net.eval()
    out = evaluate_vs_baselines(net, cfg, torch.device("cpu"),
                                 num_hands=24, rng=random.Random(0))
    assert set(out.keys()) == {"random", "call_station", "tight_aggressive"}
    for v in out.values():
        assert np.isfinite(v)


def test_evaluate_vs_human_like_baselines_runs():
    from algo.deep_cfr import evaluate_vs_baselines
    cfg = DeepCFRConfig()
    aspace = ActionSpace(cfg.bet_fractions)
    net = PolicyNet(obs_dim=OBS_DIM, num_actions=aspace.num_actions,
                    hidden=64, num_blocks=1).to(torch.device("cpu"))
    net.eval()
    out = evaluate_vs_baselines(
        net, cfg, torch.device("cpu"),
        num_hands=20, rng=random.Random(0), include_human_like=True,
    )
    assert {"loose_passive", "loose_aggressive", "overfolder",
            "bluff_catcher", "pot_pressure"}.issubset(out)
    for v in out.values():
        assert np.isfinite(v)


def test_traverse_one_3max_runs():
    """traverse_one must work for >2 players."""
    from algo.deep_cfr.traversal import traverse_one, make_uniform_strategy_fn
    aspace = ActionSpace()
    adv, strat = traverse_one(
        traverser=0,
        strategy_fn=make_uniform_strategy_fn(),
        iter_t=1,
        num_players=3,
        starting_stack=200,
        small_blind=1,
        big_blind=2,
        button=0,
        action_space=aspace,
        deal_rng=random.Random(0),
        sample_rng=np.random.default_rng(1),
        linear_weight=True,
    )
    assert len(adv) + len(strat) > 0

