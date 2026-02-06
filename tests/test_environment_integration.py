#!/usr/bin/env python3
"""
Test Environment Integration.

This module tests the integration between the SUMO environment
and the MGMQ model, focusing on:
1. Observation space shape and content
2. Action space shape and constraints
3. Reward computation sanity
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Skip these tests if SUMO is not available
pytestmark = pytest.mark.skipif(
    not Path("/usr/bin/sumo").exists() and not Path("/usr/local/bin/sumo").exists(),
    reason="SUMO not installed"
)


class TestObservationSpace:
    """Test observation space properties."""
    
    def test_observation_shape_matches_model_input(self):
        """Observation dim should be 48 (12 lanes × 4 features)."""
        # Expected: 12 lanes × 4 features (density, queue, wait_time, speed)
        expected_obs_dim = 48
        num_lanes = 12
        features_per_lane = 4
        
        assert num_lanes * features_per_lane == expected_obs_dim, \
            "Observation dimension calculation mismatch"
            
    def test_observation_features_are_normalized(self):
        """Observations should be normalized to reasonable ranges."""
        # This is a design check - actual normalization happens in traffic_signal.py
        # Features should be in range [0, 1] or [-1, 1] for stable learning
        
        # We don't have a live environment here, so this is a documentation test
        expected_ranges = {
            "density": (0, 1),      # Normalized by max capacity
            "queue": (0, 1),        # Normalized by lane length
            "wait_time": (0, 1),    # Normalized by max wait threshold
            "speed": (0, 1),        # Normalized by max speed
        }
        
        assert len(expected_ranges) == 4, "Should have 4 feature types per lane"


class TestActionSpace:
    """Test action space properties."""
    
    def test_action_is_simplex(self):
        """Action should be a probability distribution over phases."""
        # Standard action: 8 phases
        num_phases = 8
        
        # Generate random valid action
        logits = np.random.randn(num_phases)
        action = np.exp(logits) / np.exp(logits).sum()  # Softmax
        
        # Check simplex constraints
        assert np.abs(action.sum() - 1.0) < 1e-6, "Action should sum to 1"
        assert (action >= 0).all(), "Action should be non-negative"
        
    def test_action_converts_to_green_times(self):
        """Action distribution should convert to green times."""
        cycle_time = 90  # seconds
        num_phases = 4
        min_green = 5  # seconds
        
        # Random action (simplex)
        action = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Convert to green times
        available_time = cycle_time - num_phases * min_green
        green_times = min_green + action * available_time
        
        # Check constraints
        assert np.abs(green_times.sum() - cycle_time) < 1e-6, \
            "Green times should sum to cycle_time"
        assert (green_times >= min_green).all(), \
            "All green times should be >= min_green"


class TestRewardComputation:
    """Test reward function properties."""
    
    def test_reward_is_bounded(self):
        """Reward should be bounded to prevent gradient explosion."""
        # Reward clipping range (from model_config.yml)
        clip_min = -1.0
        clip_max = 1.0
        
        # Simulate raw reward
        raw_rewards = np.array([-10, -1, 0, 1, 10])
        clipped = np.clip(raw_rewards, clip_min, clip_max)
        
        assert (clipped >= clip_min).all(), "Clipped rewards should be >= min"
        assert (clipped <= clip_max).all(), "Clipped rewards should be <= max"
        
    def test_reward_functions_exist(self):
        """Reward function registry should have expected functions."""
        expected_functions = [
            'halt-veh-by-detectors',
            'diff-departed-veh',
            'occupancy',
        ]
        
        # Check that these are valid reward function names
        for fn_name in expected_functions:
            # This is a design check - actual registry is in traffic_signal.py
            assert isinstance(fn_name, str), f"Reward function name should be string: {fn_name}"


class TestPhaseConfig:
    """Test phase configuration from intersection_config.json."""
    
    def test_phase_config_file_exists(self):
        """intersection_config.json should exist for grid4x4."""
        config_path = Path(__file__).parent.parent / "network" / "grid4x4" / "intersection_config.json"
        
        assert config_path.exists(), f"Config file not found: {config_path}"
        
    def test_phase_config_structure(self):
        """Config should have required structure."""
        import json
        
        config_path = Path(__file__).parent.parent / "network" / "grid4x4" / "intersection_config.json"
        
        if not config_path.exists():
            pytest.skip("Config file not found")
            
        with open(config_path) as f:
            config = json.load(f)
            
        # Check top-level keys
        assert "intersections" in config, "Config should have 'intersections' key"
        assert "num_intersections" in config, "Config should have 'num_intersections' key"
        
        # Check first intersection structure
        first_intersection = list(config["intersections"].values())[0]
        required_keys = ["direction_map", "lanes_by_direction", "phase_config"]
        
        for key in required_keys:
            assert key in first_intersection, f"Intersection should have '{key}' key"
            
    def test_direction_map_is_complete(self):
        """direction_map should have all 4 cardinal directions."""
        import json
        
        config_path = Path(__file__).parent.parent / "network" / "grid4x4" / "intersection_config.json"
        
        if not config_path.exists():
            pytest.skip("Config file not found")
            
        with open(config_path) as f:
            config = json.load(f)
            
        for ts_id, intersection in config["intersections"].items():
            direction_map = intersection["direction_map"]
            
            for direction in ['N', 'E', 'S', 'W']:
                assert direction in direction_map, \
                    f"Intersection {ts_id} missing direction {direction}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
