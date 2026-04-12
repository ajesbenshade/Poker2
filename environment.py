import gc
import os

import psutil


ROCM_ENV_DEFAULTS = {
    'HIP_VISIBLE_DEVICES': '0',
    'HSA_OVERRIDE_GFX_VERSION': '11_0_0',
    'PYTORCH_HIP_ALLOC_CONF': 'garbage_collection_threshold:0.8,expandable_segments:True',
    'PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING': '1',
}


def setup_rocmo():
    for key, value in ROCM_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    return dict(ROCM_ENV_DEFAULTS)


def setup_rocm():
    return setup_rocmo()


def _cuda_ready(torch_module):
    return hasattr(torch_module, 'cuda') and torch_module.cuda.is_available()


def get_vram_usage():
    try:
        import torch
    except Exception:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0,
            'used_pct': 0.0,
        }

    if not _cuda_ready(torch):
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0,
            'used_pct': 0.0,
        }

    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    total_bytes = float(device_properties.total_memory)
    allocated_bytes = float(torch.cuda.memory_allocated(device_index))
    reserved_bytes = float(torch.cuda.memory_reserved(device_index))
    free_bytes = max(total_bytes - reserved_bytes, 0.0)

    if hasattr(torch.cuda, 'mem_get_info'):
        try:
            mem_free, mem_total = torch.cuda.mem_get_info(device_index)
            free_bytes = float(mem_free)
            total_bytes = float(mem_total)
        except Exception:
            pass

    used_bytes = max(allocated_bytes, reserved_bytes, total_bytes - free_bytes)
    total_gb = total_bytes / (1024 ** 3) if total_bytes > 0 else 0.0
    used_gb = used_bytes / (1024 ** 3) if used_bytes > 0 else 0.0

    return {
        'allocated_gb': allocated_bytes / (1024 ** 3),
        'reserved_gb': reserved_bytes / (1024 ** 3),
        'used_gb': used_gb,
        'free_gb': free_bytes / (1024 ** 3),
        'total_gb': total_gb,
        'used_pct': (used_gb / total_gb * 100.0) if total_gb > 0 else 0.0,
    }


def get_memory_snapshot():
    virtual_memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    vram = get_vram_usage()
    return {
        **vram,
        'ram_pct': float(virtual_memory.percent),
        'ram_used_gb': float(virtual_memory.used) / (1024 ** 3),
        'ram_total_gb': float(virtual_memory.total) / (1024 ** 3),
        'process_rss_gb': float(process.memory_info().rss) / (1024 ** 3),
    }


def clear_runtime_caches():
    gc.collect()
    try:
        import torch
    except Exception:
        return

    if _cuda_ready(torch):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if hasattr(torch.cuda, 'ipc_collect'):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass