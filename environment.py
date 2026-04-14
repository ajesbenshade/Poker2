import gc
import json
import os
import subprocess
import sys

import psutil


ROCM_ENV_DEFAULTS = {
    'HIP_VISIBLE_DEVICES': '0',
    'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
    'PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING': '1',
    'HSA_ENABLE_SDMA': '0',
    'TORCH_CUDNN_ENABLE': '0',
    'OMP_NUM_THREADS': '1',
}

ROCM_HSA_FALLBACK = '10.3.0'

SUPPORTED_ALLOCATOR_CONF_KEYS = {
    'expandable_segments',
    'max_split_size_mb',
}

ALLOCATOR_CONF_KEY_ALIASES = {
    'expandable_segment': 'expandable_segments',
}


def _normalize_allocator_conf(raw_value):
    normalized_parts = []
    for part in str(raw_value or '').split(','):
        entry = part.strip()
        if not entry or ':' not in entry:
            continue
        key, value = entry.split(':', 1)
        canonical_key = ALLOCATOR_CONF_KEY_ALIASES.get(key.strip(), key.strip())
        if canonical_key in SUPPORTED_ALLOCATOR_CONF_KEYS:
            normalized_parts.append(f'{canonical_key}:{value.strip()}')
    return ','.join(normalized_parts)


def _resolve_allocator_conf():
    configured_value = os.environ.get('PYTORCH_ALLOC_CONF')
    legacy_value = os.environ.get('PYTORCH_HIP_ALLOC_CONF')
    source_value = configured_value or legacy_value or ROCM_ENV_DEFAULTS['PYTORCH_ALLOC_CONF']
    normalized_value = _normalize_allocator_conf(source_value)
    if not normalized_value:
        normalized_value = ROCM_ENV_DEFAULTS['PYTORCH_ALLOC_CONF']
    os.environ['PYTORCH_ALLOC_CONF'] = normalized_value
    os.environ.pop('PYTORCH_HIP_ALLOC_CONF', None)
    return normalized_value


def setup_rocmo():
    resolved_env = {'PYTORCH_ALLOC_CONF': _resolve_allocator_conf()}
    for key, value in ROCM_ENV_DEFAULTS.items():
        if key == 'PYTORCH_ALLOC_CONF':
            continue
        os.environ.setdefault(key, value)
        resolved_env[key] = os.environ[key]
    resolved_env['PYTORCH_ALLOC_CONF'] = os.environ['PYTORCH_ALLOC_CONF']
    return resolved_env


def apply_hsa_fallback():
    # Use fallback only after a failed device initialization attempt.
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = ROCM_HSA_FALLBACK
    return ROCM_HSA_FALLBACK


def setup_rocm():
    return setup_rocmo()


def _default_startup_info():
    return {
        'device': 'cpu',
        'device_available': False,
        'gpu_name': 'none (CPU fallback)',
        'vram_total_gb': 0.0,
        'fallback_applied': False,
        'probe_error': '',
        'hsa_override': os.environ.get('HSA_OVERRIDE_GFX_VERSION', ROCM_ENV_DEFAULTS['HSA_OVERRIDE_GFX_VERSION']),
        'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES', ROCM_ENV_DEFAULTS['HIP_VISIBLE_DEVICES']),
    }


def _probe_rocm_device(env):
    code = """
import json

result = {
    'device': 'cpu',
    'device_available': False,
    'gpu_name': 'none (CPU fallback)',
    'vram_total_gb': 0.0,
    'probe_error': '',
}

try:
    import torch
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        result.update({
            'device': 'cuda',
            'device_available': True,
            'gpu_name': torch.cuda.get_device_name(device_index),
            'vram_total_gb': float(properties.total_memory) / (1024 ** 3),
        })
except Exception as exc:
    result['probe_error'] = f'{type(exc).__name__}: {exc}'

print(json.dumps(result))
""".strip()

    try:
        completed = subprocess.run(
            [sys.executable, '-c', code],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except Exception as exc:
        result = _default_startup_info()
        result['probe_error'] = f'{type(exc).__name__}: {exc}'
        return result

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        result = _default_startup_info()
        result['probe_error'] = (completed.stderr or stdout or 'ROCm probe failed').strip()
        return result

    try:
        result = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        result = _default_startup_info()
        result['probe_error'] = f'Invalid ROCm probe output: {stdout or completed.stderr}'.strip()
        return result

    startup_info = _default_startup_info()
    startup_info.update(result)
    return startup_info


def initialize_rocm_runtime():
    applied_env = setup_rocmo()
    startup_info = _default_startup_info()
    startup_info['hsa_override'] = applied_env['HSA_OVERRIDE_GFX_VERSION']
    startup_info['hip_visible_devices'] = applied_env['HIP_VISIBLE_DEVICES']

    initial_probe = _probe_rocm_device(dict(os.environ))
    startup_info.update(initial_probe)

    if initial_probe.get('probe_error'):
        fallback = apply_hsa_fallback()
        fallback_probe = _probe_rocm_device(dict(os.environ))
        startup_info.update(fallback_probe)
        startup_info['fallback_applied'] = True
        startup_info['hsa_override'] = fallback
        if fallback_probe.get('probe_error'):
            startup_info['probe_error'] = fallback_probe['probe_error']

    return startup_info


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

    try:
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
    except Exception:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0,
            'used_pct': 0.0,
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