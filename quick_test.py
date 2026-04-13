#!/usr/bin/env python3
"""
Minimal quick-test runner for Deep CFR on AMD ROCm.
Targets: complete in <15 min, no crashes, clear progress metrics.
Uses SMOKE_TEST config by default but with even more aggressive scaling.
"""
import os
import sys
import time
import logging
import argparse

os.environ.setdefault('OMP_NUM_THREADS', '1')

from environment import ROCM_ENV_DEFAULTS, setup_rocmo, get_memory_snapshot, clear_runtime_caches

setup_rocmo()

import torch
import numpy as np
from config import Config
from abstractions import create_buckets, simulate_features
from datatypes import Infoset
from game import terminal
from deep_cfr import DeepCFRAgent
import ray

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class QuickTestConfig(Config):
    """Ultra-minimal config for <15 min validation."""
    # Extreme reduction for speed
    ITERATIONS = 4  # Just 4 iterations (was 2 for smoke test)
    SMOKE_TEST_ITERATIONS = 4
    NUM_BUCKETS = 16  # Was 32
    SMOKE_TEST_NUM_BUCKETS = 16
    BATCH_SIZE = 128  # Was 256
    SMOKE_TEST_BATCH_SIZE = 128
    NUM_TRAVERSALS = 32  # Was 64
    SMOKE_TEST_TRAVERSALS = 32
    REPLAY_WARMUP_SAMPLES = 16
    SMOKE_TEST_REPLAY_WARMUP = 16
    REPLAY_BUFFER_SIZE = 4096  # Tiny
    SMOKE_TEST_REPLAY_BUFFER_SIZE = 4096
    LOG_INTERVAL = 1
    SMOKE_TEST_LOG_INTERVAL = 1
    CHECKPOINT_INTERVAL = 4  # Save at end
    SMOKE_TEST_CHECKPOINT_INTERVAL = 4
    EQUITY_ROLLOUTS = 2  # Minimal
    SMOKE_TEST_EQUITY_ROLLOUTS = 2

    # Slightly safer memory
    VRAM_SOFT_LIMIT_GB = 8.0
    RAM_SOFT_LIMIT_PCT = 85.0

    # Disable all advanced features for speed
    SAFE_HARDWARE_MODE = True
    GRADIENT_CHECKPOINTING = False
    REPLAY_BUFFER_SIZE_MIN = 64


def init_ray_minimal():
    """Initialize Ray with minimal settings."""
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        num_cpus=4,
        num_gpus=0,  # Avoid GPU in Ray workers for this test
        ignore_reinit_error=True,
        runtime_env={
            'env_vars': {
                'HIP_VISIBLE_DEVICES': '0',
                'PYTORCH_HIP_ALLOC_CONF': ROCM_ENV_DEFAULTS['PYTORCH_HIP_ALLOC_CONF'],
                'HSA_OVERRIDE_GFX_VERSION': ROCM_ENV_DEFAULTS['HSA_OVERRIDE_GFX_VERSION'],
                'OMP_NUM_THREADS': '1',
            }
        },
    )


def quick_test_deep_cfr():
    """Run a minimal Deep CFR test that completes in <15 minutes."""
    start_time = time.time()
    
    logger.info('=' * 80)
    logger.info('QUICK TEST: Deep CFR on AMD ROCm (Ryzen 9 7900X + RX 7900 XT)')
    logger.info('=' * 80)
    logger.info('Config: ITERATIONS=%d, BUCKETS=%d, BATCH=%d, TRAVERSALS=%d',
                QuickTestConfig.ITERATIONS,
                QuickTestConfig.NUM_BUCKETS,
                QuickTestConfig.BATCH_SIZE,
                QuickTestConfig.NUM_TRAVERSALS)
    
    # Apply quick config
    Config.ITERATIONS = QuickTestConfig.ITERATIONS
    Config.NUM_BUCKETS = QuickTestConfig.NUM_BUCKETS
    Config.BATCH_SIZE = QuickTestConfig.BATCH_SIZE
    Config.NUM_TRAVERSALS = QuickTestConfig.NUM_TRAVERSALS
    Config.REPLAY_BUFFER_SIZE = QuickTestConfig.REPLAY_BUFFER_SIZE
    Config.GRADIENT_CHECKPOINTING = False
    Config.VRAM_SOFT_LIMIT_GB = QuickTestConfig.VRAM_SOFT_LIMIT_GB
    Config.EQUITY_ROLLOUTS = QuickTestConfig.EQUITY_ROLLOUTS
    
    # Create minimal abstractions
    logger.info('Creating abstractions...')
    # Generate features with minimal sims for speed
    features = simulate_features(num_sims=Config.NUM_SIMULATIONS)
    buckets, centroids = create_buckets(features, num_buckets=Config.NUM_BUCKETS)
    infosets = [Infoset(bucket_id) for bucket_id in range(len(centroids))]
    logger.info('Abstractions created: %d infosets, %d centroids', len(infosets), len(centroids))
    
    # Initialize agent
    logger.info('Initializing Deep CFR agent...')
    clear_runtime_caches()
    agent = DeepCFRAgent(infosets, centroids, writer=None, resume=False)
    
    # Quick training loop
    logger.info('Starting training loop...')
    iteration_times = []
    
    try:
        iter_start = time.time()
        
        # Train all iterations at once
        metrics = agent.train(iterations=Config.ITERATIONS, start_iteration=0)
        
        total_train_time = time.time() - iter_start
        iteration_times.append(total_train_time)
        
        logger.info('Training complete - Final metrics:')
        logger.info('  Avg Utility: %.4f', metrics.get('avg_utility', 0.0))
        logger.info('  Exploitability: %.4f', metrics.get('exploitability_proxy', 0.0))
        advantage_loss = metrics.get('advantage_loss')
        if advantage_loss is not None:
            logger.info('  Advantage Loss: %.4f', advantage_loss)
        strategy_loss = metrics.get('strategy_loss')
        if strategy_loss is not None:
            logger.info('  Strategy Loss: %.4f', strategy_loss)
        logger.info('  Train Time: %.2f sec', total_train_time)
        logger.info('  Buffer sizes: adv=%d, strat=%d',
                    len(agent.advantage_buffer),
                    len(agent.strategy_buffer))
        logger.info('  Backoff events: %d', metrics.get('backoff_events', 0))
        logger.info('  Elapsed seconds: %.2f', metrics.get('elapsed_seconds', 0.0))
        
        # Clear caches
        clear_runtime_caches()
        
    except Exception as e:
        logger.error('Error during training: %s', e, exc_info=True)
        return False
    
    # Final stats
    total_time = time.time() - start_time
    snapshot_final = get_memory_snapshot()
    
    logger.info('=' * 80)
    logger.info('QUICK TEST COMPLETE')
    logger.info('=' * 80)
    logger.info('Total time: %.2f seconds (%.2f minutes)', total_time, total_time / 60.0)
    logger.info('Avg time per iteration: %.2f seconds', np.mean(iteration_times))
    logger.info('Final VRAM: %.2f / %.2f GB (%.1f%%)', 
                snapshot_final['used_gb'], snapshot_final['total_gb'], snapshot_final['used_pct'])
    logger.info('Final RAM: %.1f%%', snapshot_final['ram_pct'])
    logger.info('No crashes! ✓')
    
    # Save checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'quick_test_checkpoint.pt')
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    try:
        agent.save_checkpoint(Config.ITERATIONS - 1, {}, is_best=False)
        logger.info('Checkpoint saved to %s', checkpoint_path)
    except Exception as e:
        logger.warning('Failed to save checkpoint: %s', e)
    
    return total_time < 900  # 15 minutes


def main():
    parser = argparse.ArgumentParser(description='Quick test for Deep CFR on ROCm.')
    parser.add_argument('--no-ray', action='store_true', help='Skip Ray initialization.')
    args = parser.parse_args()
    
    success = True
    
    if not args.no_ray:
        try:
            init_ray_minimal()
            logger.info('Ray initialized')
        except Exception as e:
            logger.warning('Ray initialization failed: %s', e)
    
    try:
        success = quick_test_deep_cfr()
    except Exception as e:
        logger.error('Quick test failed: %s', e, exc_info=True)
        success = False
    finally:
        if ray.is_initialized():
            ray.shutdown()
    
    logger.info('Test %s', 'PASSED ✓' if success else 'FAILED ✗')
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
