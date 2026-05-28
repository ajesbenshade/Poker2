"""Shared memory transport for Deep CFR worker results.

This provides a high-performance, low-copy path for returning large numpy
arrays from traversal workers back to the main trainer process.

Usage:
- Workers allocate a SharedMemory block, write the 8 arrays into it
    (with a small header describing shapes/dtypes), and return a small
    descriptor (name + size).
- Trainer maps the same SharedMemory, copies the arrays into process-local
    numpy arrays, inserts into buffers, then unlinks the block.

This is dramatically faster than pickle (current "ipc") or disk+npz ("file")
for the large observation/advantage/strategy sample payloads.

Only suitable for local workers on the same machine (same as the current pool).
"""

from __future__ import annotations

import multiprocessing.shared_memory as shm
import os
import struct
from typing import Tuple

import numpy as np


# Header format: 8 arrays, we store for each: ndim(1B), shape up to 2 dims (2x i64),
# dtype code (1B). We use a fixed simple header for our exact 8 arrays.
# Layout (little endian):
#   magic (4 bytes "CFR1")
#   For each of 8 arrays:
#     ndim (uint8, we only use 1 or 2)
#     dim0 (int64)
#     dim1 (int64, 0 if 1D)
#     dtype_code (uint8): 0=float32
#   Then the raw data for all arrays concatenated (float32 only for our case)

_MAGIC = b"CFR1"
_HEADER_SIZE = 4 + 8 * (1 + 8 + 8 + 1)   # magic + 8 * (ndim + dim0 + dim1 + dtype)


_DTYPE_CODE = {np.dtype("float32"): 0}
_DTYPE_FROM_CODE = {0: np.float32}


def _pack_header(arrays: Tuple[np.ndarray, ...]) -> bytes:
    """Pack metadata for our exact 8 arrays into a compact header."""
    if len(arrays) != 8:
        raise ValueError("expected exactly 8 result arrays")
    buf = bytearray(_MAGIC)
    for arr in arrays:
        if arr.dtype != np.float32:
            # We currently only support float32 payloads from traversals
            raise ValueError(f"unsupported dtype {arr.dtype}")
        buf.append(arr.ndim & 0xFF)
        shape = arr.shape
        buf.extend(struct.pack("<q", shape[0]))
        buf.extend(struct.pack("<q", shape[1] if len(shape) > 1 else 0))
        buf.append(0)  # dtype code for float32
    return bytes(buf)


def _unpack_header(header: bytes) -> list[tuple]:
    """Return list of (shape, dtype) for the 8 arrays."""
    if header[:4] != _MAGIC:
        raise ValueError("bad magic in sharedmem header")
    meta = []
    off = 4
    for _ in range(8):
        ndim = header[off]
        off += 1
        d0 = struct.unpack_from("<q", header, off)[0]
        off += 8
        d1 = struct.unpack_from("<q", header, off)[0]
        off += 8
        code = header[off]
        off += 1
        dtype = _DTYPE_FROM_CODE.get(code, np.float32)
        shape = (d0,) if ndim == 1 else (d0, d1)
        meta.append((shape, dtype))
    return meta


def pack_results_to_sharedmem(
    result: Tuple[np.ndarray, ...],
) -> tuple[str, int]:
    """Write the 8 result arrays into a new SharedMemory block.

    Returns (shm_name, total_size) that the receiver can use to attach.
    The caller (worker) is responsible for calling unlink after the
    receiver has mapped it (or we can rely on process exit cleanup).
    """
    header = _pack_header(result)
    data_size = sum(a.nbytes for a in result)
    total_size = len(header) + data_size

    shm_obj = shm.SharedMemory(create=True, size=total_size)
    try:
        buf = shm_obj.buf
        buf[: len(header)] = header
        offset = len(header)
        for arr in result:
            buf[offset : offset + arr.nbytes] = arr.tobytes()
            offset += arr.nbytes
        name = shm_obj.name
    finally:
        shm_obj.close()  # close in this process; receiver will attach

    return name, total_size


def load_results_from_sharedmem(name: str, size: int) -> Tuple[np.ndarray, ...]:
    """Attach to existing shared memory, copy out the 8 arrays, and unlink it."""
    shm_obj = shm.SharedMemory(name=name, create=False, size=size)
    buf = None
    try:
        buf = shm_obj.buf
        header = bytes(buf[:_HEADER_SIZE])
        meta = _unpack_header(header)

        offset = _HEADER_SIZE
        arrays = []
        for shape, dtype in meta:
            dtype = np.dtype(dtype)
            view = np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)
            arrays.append(view.copy())
            offset += view.nbytes
            del view

        return tuple(arrays)
    finally:
        if buf is not None:
            buf.release()
        shm_obj.close()
        # Unlink so the OS can free the memory once all attachments are gone.
        try:
            shm.SharedMemory(name=name, create=False).unlink()
        except FileNotFoundError:
            pass


def cleanup_sharedmem(name: str) -> None:
    """Best-effort cleanup if something went wrong."""
    try:
        shm.SharedMemory(name=name, create=False).unlink()
    except Exception:
        pass
