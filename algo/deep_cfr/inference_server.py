"""Batched inference server for Deep CFR traversal workers.

Traversal workers make many single-observation advantage-net calls while walking
game trees. This module lets workers send those requests to one server process,
which batches them by seat and evaluates the advantage nets on the configured
device. The existing CPU worker-local path remains the fallback.
"""
from __future__ import annotations

import io
import logging
import queue
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .network import AdvantageNet

logger = logging.getLogger(__name__)


def _strip_compile_prefix(state_dict):
    return {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in state_dict.items()
    }


def _load_state_dict_blob(blob: bytes):
    return _strip_compile_prefix(torch.load(io.BytesIO(blob), map_location="cpu"))


def _select_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name not in ("cuda", "cpu"):
        try:
            device = torch.device(name)
            if device.type != "cuda" or torch.cuda.is_available():
                return device
        except Exception:
            pass
    return torch.device("cpu")


def _select_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if name in ("float32", "fp32"):
        return None
    if name in ("bfloat16", "bf16"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if name in ("float16", "fp16"):
        return torch.float16
    return None


class BatchedAdvantageInference:
    """Holds per-seat advantage nets and evaluates batched regret matching."""

    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        hidden: int,
        blocks: int,
        dropout: float,
        device: torch.device,
        amp_dtype: Optional[torch.dtype],
        num_players: int,
    ) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.blocks = blocks
        self.dropout = dropout
        self.device = device
        self.amp_dtype = amp_dtype
        self.nets: List[Optional[AdvantageNet]] = [None] * num_players

    def load_state_dict_blobs(self, blobs: Sequence[Optional[bytes]]) -> None:
        nets: List[Optional[AdvantageNet]] = []
        for blob in blobs:
            if blob is None:
                nets.append(None)
                continue
            net = AdvantageNet(
                self.obs_dim,
                self.num_actions,
                hidden=self.hidden,
                num_blocks=self.blocks,
                dropout=self.dropout,
            )
            net.load_state_dict(_load_state_dict_blob(blob))
            net.to(self.device)
            net.eval()
            nets.append(net)
        self.nets = nets

    def infer_batch(
        self,
        seats: np.ndarray,
        obs: np.ndarray,
        legal: np.ndarray,
    ) -> np.ndarray:
        if obs.shape[0] == 0:
            return np.zeros((0, self.num_actions), dtype=np.float32)

        out = np.zeros((obs.shape[0], self.num_actions), dtype=np.float32)
        for seat in np.unique(seats.astype(np.int64, copy=False)):
            idx = np.nonzero(seats == seat)[0]
            seat_legal = legal[idx].astype(np.float32, copy=False)
            net = self.nets[int(seat)] if 0 <= int(seat) < len(self.nets) else None
            if net is None:
                legal_count = seat_legal.sum(axis=1, keepdims=True)
                legal_count = np.maximum(legal_count, 1.0)
                out[idx] = seat_legal / legal_count
                continue

            obs_t = torch.from_numpy(obs[idx].astype(np.float32, copy=False)).to(self.device)
            legal_t = torch.from_numpy(seat_legal).to(self.device)
            ctx = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if self.amp_dtype is not None
                else _NullCtx()
            )
            with torch.no_grad(), ctx:
                adv = net(obs_t, legal_t).float()
                pos = torch.clamp(adv, min=0.0) * legal_t
                total = pos.sum(dim=-1, keepdim=True)
                uniform = legal_t / legal_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
                probs = torch.where(total > 0, pos / total.clamp_min(1e-8), uniform)
            out[idx] = probs.cpu().numpy().astype(np.float32, copy=False)
        return out


class QueueInferenceClient:
    """Small worker-side client used as a synchronous strategy callable."""

    def __init__(self, request_queue, response_queue, worker_id: int) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = int(worker_id)
        self._next_request_id = 0

    def infer(self, obs: np.ndarray, legal: np.ndarray, seat: int) -> np.ndarray:
        request_id = self._next_request_id
        self._next_request_id += 1
        self.request_queue.put((
            "infer",
            self.worker_id,
            request_id,
            int(seat),
            obs.astype(np.float32, copy=True),
            legal.astype(np.float32, copy=True),
        ))
        while True:
            response_id, probs = self.response_queue.get()
            if response_id == request_id:
                return probs


def make_queue_strategy_fn(client: QueueInferenceClient):
    def _fn(obs: np.ndarray, legal: np.ndarray, seat: int) -> np.ndarray:
        return client.infer(obs, legal, seat)
    return _fn


class InferenceServerHandle:
    """Owns queues and lifecycle for a batched inference server process."""

    def __init__(
        self,
        ctx,
        *,
        num_workers: int,
        state_dict_blobs: Sequence[Optional[bytes]],
        obs_dim: int,
        num_actions: int,
        hidden: int,
        blocks: int,
        dropout: float,
        num_players: int,
        device_name: str,
        amp_dtype_name: str,
        batch_size: int,
        timeout_ms: float,
        queue_size: int,
    ) -> None:
        maxsize = max(0, int(queue_size))
        self.request_queue = ctx.Queue(maxsize=maxsize)
        self.control_response_queue = ctx.Queue(maxsize=8)
        self.response_queues = [ctx.Queue(maxsize=maxsize) for _ in range(num_workers)]
        self.worker_counter = ctx.Value("i", 0)
        self.worker_counter_lock = ctx.Lock()
        self._next_command_id = 0
        self.process = ctx.Process(
            target=run_inference_server,
            args=(self.request_queue, self.response_queues, self.control_response_queue),
            kwargs={
                "state_dict_blobs": list(state_dict_blobs),
                "obs_dim": obs_dim,
                "num_actions": num_actions,
                "hidden": hidden,
                "blocks": blocks,
                "dropout": dropout,
                "num_players": num_players,
                "device_name": device_name,
                "amp_dtype_name": amp_dtype_name,
                "batch_size": batch_size,
                "timeout_ms": timeout_ms,
            },
        )
        self.process.start()

    def update(self, state_dict_blobs: Sequence[Optional[bytes]], timeout: float = 120.0) -> None:
        command_id = self._next_command_id
        self._next_command_id += 1
        self.request_queue.put(("update", command_id, list(state_dict_blobs)))
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("timed out waiting for inference server update")
            ack_id, ok, message = self.control_response_queue.get(timeout=remaining)
            if ack_id != command_id:
                continue
            if not ok:
                raise RuntimeError(f"inference server update failed: {message}")
            return

    def close(self, timeout: float = 10.0) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.request_queue.put(("stop",))
            self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=timeout)


def run_inference_server(
    request_queue,
    response_queues,
    control_response_queue,
    *,
    state_dict_blobs: Sequence[Optional[bytes]],
    obs_dim: int,
    num_actions: int,
    hidden: int,
    blocks: int,
    dropout: float,
    num_players: int,
    device_name: str,
    amp_dtype_name: str,
    batch_size: int,
    timeout_ms: float,
) -> None:
    torch.set_num_threads(1)
    device = _select_device(device_name)
    amp_dtype = _select_amp_dtype(amp_dtype_name, device)
    engine = BatchedAdvantageInference(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden=hidden,
        blocks=blocks,
        dropout=dropout,
        device=device,
        amp_dtype=amp_dtype,
        num_players=num_players,
    )
    engine.load_state_dict_blobs(state_dict_blobs)
    max_batch = max(1, int(batch_size))
    timeout_s = max(0.0, float(timeout_ms)) / 1000.0

    while True:
        try:
            item = request_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        tag = item[0]
        if tag == "stop":
            return
        if tag == "update":
            command_id = item[1]
            try:
                engine.load_state_dict_blobs(item[2])
                control_response_queue.put((command_id, True, ""))
            except Exception as exc:
                control_response_queue.put((command_id, False, repr(exc)))
            continue
        if tag != "infer":
            continue

        batch = [item]
        deadline = time.monotonic() + timeout_s
        deferred = []
        while len(batch) < max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_item = request_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_item[0] == "infer":
                batch.append(next_item)
            else:
                deferred.append(next_item)
                break

        _serve_inference_batch(engine, batch, response_queues)

        for control_item in deferred:
            if control_item[0] == "stop":
                return
            if control_item[0] == "update":
                command_id = control_item[1]
                try:
                    engine.load_state_dict_blobs(control_item[2])
                    control_response_queue.put((command_id, True, ""))
                except Exception as exc:
                    control_response_queue.put((command_id, False, repr(exc)))


def _serve_inference_batch(engine: BatchedAdvantageInference, batch, response_queues) -> None:
    worker_ids = np.asarray([item[1] for item in batch], dtype=np.int64)
    request_ids = [item[2] for item in batch]
    seats = np.asarray([item[3] for item in batch], dtype=np.int64)
    obs = np.stack([item[4] for item in batch]).astype(np.float32, copy=False)
    legal = np.stack([item[5] for item in batch]).astype(np.float32, copy=False)
    probs = engine.infer_batch(seats, obs, legal)
    for row, worker_id in enumerate(worker_ids):
        response_queues[int(worker_id)].put((request_ids[row], probs[row]))


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
