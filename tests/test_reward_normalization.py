"""
Test script for reward normalization functionality.

Tests the RunningMeanStd and reward normalization in env.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.observation_normalizer import RunningMeanStd, RewardNormalizer


def test_running_mean_std_basic():
    """Test basic RunningMeanStd functionality."""
    print("Testing RunningMeanStd basic functionality...")
    
    rms = RunningMeanStd(shape=())
    
    # Test with scalar values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        rms.update(np.array(v))
    
    expected_mean = np.mean(values)
    expected_var = np.var(values)
    
    print(f"  Expected mean: {expected_mean:.4f}, Got: {float(rms.mean):.4f}")
    print(f"  Expected var: {expected_var:.4f}, Got: {float(rms.var):.4f}")
    
    assert abs(float(rms.mean) - expected_mean) < 0.1, "Mean mismatch"
    print("  ✓ Basic test passed")


def test_running_mean_std_batch():
    """Test batch update."""
    print("\nTesting RunningMeanStd batch update...")
    
    rms = RunningMeanStd(shape=())
    
    # Batch of values
    batch = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rms.update(batch)
    
    print(f"  Mean: {float(rms.mean):.4f}")
    print(f"  Var: {float(rms.var):.4f}")
    print(f"  Count: {rms.count:.4f}")
    
    assert rms.count >= 5, "Count should be at least 5"
    print("  ✓ Batch test passed")


def test_nan_handling():
    """Test NaN handling in RunningMeanStd."""
    print("\nTesting NaN handling...")
    
    rms = RunningMeanStd(shape=())
    
    # Add some valid values first
    rms.update(np.array([1.0, 2.0, 3.0]))
    old_mean = float(rms.mean)
    old_count = rms.count
    
    # Try to add NaN values - should be ignored
    rms.update(np.array([np.nan, np.inf, -np.inf]))
    
    print(f"  Mean before NaN: {old_mean:.4f}, after: {float(rms.mean):.4f}")
    print(f"  Count before NaN: {old_count:.4f}, after: {rms.count:.4f}")
    
    # Mean and count should not change
    assert abs(float(rms.mean) - old_mean) < 0.001, "Mean should not change with NaN input"
    assert rms.count == old_count, "Count should not change with NaN input"
    
    print("  ✓ NaN handling test passed")


def test_normalization():
    """Test normalization output."""
    print("\nTesting normalization output...")
    
    rms = RunningMeanStd(shape=())
    
    # Add enough samples to build statistics
    np.random.seed(42)
    values = np.random.randn(100) * 10 + 50  # Mean ~50, std ~10
    rms.update(values)
    
    print(f"  Running mean: {float(rms.mean):.4f}")
    print(f"  Running std: {np.sqrt(float(rms.var)):.4f}")
    
    # Normalize a value
    test_val = 60.0
    normalized = (test_val - float(rms.mean)) / (np.sqrt(float(rms.var)) + 1e-8)
    
    print(f"  Original value: {test_val}")
    print(f"  Normalized value: {normalized:.4f}")
    
    # Normalized value should be close to 1 (60 is about 1 std above mean of 50)
    assert abs(normalized) < 5, "Normalized value should be within reasonable range"
    assert not np.isnan(normalized), "Normalized value should not be NaN"
    
    print("  ✓ Normalization test passed")


def test_reward_normalizer():
    """Test RewardNormalizer class."""
    print("\nTesting RewardNormalizer...")
    
    normalizer = RewardNormalizer(clip=10.0)
    
    # Simulate episode rewards
    np.random.seed(42)
    for _ in range(5):
        # Simulate 16-agent rewards (like grid4x4)
        rewards = {f"agent_{i}": np.random.randn() * 3 - 1 for i in range(16)}
        normalized = normalizer.normalize(rewards)
        
        # Check all values are valid
        for k, v in normalized.items():
            assert not np.isnan(v), f"NaN in normalized reward for {k}"
            assert not np.isinf(v), f"Inf in normalized reward for {k}"
            assert -10 <= v <= 10, f"Normalized reward {v} for {k} out of clip range"
    
    print(f"  Mean: {float(normalizer.rms.mean):.4f}")
    print(f"  Std: {np.sqrt(float(normalizer.rms.var)):.4f}")
    print("  ✓ RewardNormalizer test passed")


def test_extreme_values():
    """Test handling of extreme values."""
    print("\nTesting extreme values...")
    
    rms = RunningMeanStd(shape=())
    
    # Add normal values first
    rms.update(np.array([1.0, 2.0, 3.0]))
    
    # Add very large values
    rms.update(np.array([1e6, 1e6, 1e6]))
    
    # Check that statistics are still valid
    assert not np.isnan(rms.mean), "Mean should not be NaN"
    assert not np.isnan(rms.var), "Var should not be NaN"
    assert not np.isinf(rms.mean), "Mean should not be Inf"
    assert not np.isinf(rms.var), "Var should not be Inf"
    
    print(f"  Mean after extreme values: {float(rms.mean):.4f}")
    print(f"  Var after extreme values: {float(rms.var):.4f}")
    print("  ✓ Extreme values test passed")


def test_zero_variance():
    """Test handling of zero variance (all same values)."""
    print("\nTesting zero variance case...")
    
    rms = RunningMeanStd(shape=())
    
    # All same values
    rms.update(np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
    
    print(f"  Mean: {float(rms.mean):.4f}")
    print(f"  Var: {float(rms.var):.4f}")
    
    # Normalization with near-zero variance
    test_val = 5.0
    std = np.sqrt(float(rms.var) + 1e-8)
    normalized = (test_val - float(rms.mean)) / std
    
    print(f"  Normalized 5.0: {normalized:.4f}")
    
    assert not np.isnan(normalized), "Normalized value should not be NaN"
    assert abs(normalized) < 100, "Normalized value should be reasonable"
    
    print("  ✓ Zero variance test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("REWARD NORMALIZATION TESTS")
    print("=" * 60)
    
    try:
        test_running_mean_std_basic()
        test_running_mean_std_batch()
        test_nan_handling()
        test_normalization()
        test_reward_normalizer()
        test_extreme_values()
        test_zero_variance()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
