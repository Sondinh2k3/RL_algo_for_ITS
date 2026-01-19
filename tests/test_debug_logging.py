"""Test script for debug logging functionality.

This script tests the debug logging for rewards and actions in TrafficSignal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from unittest.mock import MagicMock


def create_mock_data_provider():
    """Create a mock data provider for testing."""
    mock = MagicMock()
    mock.get_sim_time.return_value = 100.0
    mock.should_act.return_value = True
    mock.get_controlled_lanes.return_value = ["lane_1", "lane_2"]
    mock.get_detector_length.return_value = 50.0
    mock.get_detector_vehicle_count.return_value = 5
    mock.get_detector_vehicle_ids.return_value = ["veh_1", "veh_2"]
    mock.get_detector_jam_length.return_value = 10.0
    mock.get_detector_halting_number.return_value = 2
    mock.get_detector_occupancy.return_value = 30.0
    mock.get_detector_mean_speed.return_value = 8.0
    mock.get_detector_lane_id.return_value = "lane_1"
    mock.get_lane_max_speed.return_value = 13.89
    mock.get_lane_vehicles.return_value = ["veh_1", "veh_2"]
    mock.get_vehicle_length.return_value = 5.0
    mock.get_vehicle_speed.return_value = 8.0
    mock.get_vehicle_allowed_speed.return_value = 13.89
    mock.get_vehicle_waiting_time.return_value = 10.0
    mock.set_traffic_light_phase.return_value = None
    return mock


def test_debug_logging():
    """Test that debug logging works correctly."""
    from environment.drl_algo.traffic_signal import TrafficSignal
    from environment.drl_algo.observations import DefaultObservationFunction
    
    # Create mock data provider
    mock_dp = create_mock_data_provider()
    
    # Create TrafficSignal with debug logging enabled
    ts = TrafficSignal(
        ts_id="test_ts",
        delta_time=60,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=False,
        begin_time=0,
        reward_fn=["diff-waiting-time", "halt-veh-by-detectors"],
        reward_weights=[0.5, 0.5],
        data_provider=mock_dp,
        num_green_phases=2,
        observation_class=DefaultObservationFunction,
        detectors=[[], ["det_1", "det_2"]],
        window_size=1,
    )
    
    # Enable debug logging at level 2 (detailed)
    ts.enable_debug_logging(True, level=2)
    
    print("\n" + "="*80)
    print("Testing Debug Logging for TrafficSignal")
    print("="*80)
    
    # Test action logging
    print("\n>>> Testing ACTION logging:")
    action = np.array([0.4, 0.6])  # 40% phase 1, 60% phase 2
    ts.set_next_phase(action)
    
    # Test reward logging
    print("\n>>> Testing REWARD logging:")
    reward = ts.compute_reward()
    print(f"Final computed reward: {reward}")
    
    # Test disabling logging
    print("\n>>> Disabling debug logging...")
    ts.enable_debug_logging(False)
    
    # These should NOT produce debug output
    print(">>> Testing with logging disabled (should be silent):")
    ts.set_next_phase(action)
    reward = ts.compute_reward()
    print(f"Final computed reward (no debug output): {reward}")
    
    print("\n" + "="*80)
    print("Debug logging test completed successfully!")
    print("="*80)


def test_logging_levels():
    """Test different logging levels."""
    from environment.drl_algo.traffic_signal import TrafficSignal
    from environment.drl_algo.observations import DefaultObservationFunction
    
    mock_dp = create_mock_data_provider()
    
    ts = TrafficSignal(
        ts_id="level_test_ts",
        delta_time=60,
        yellow_time=3,
        min_green=5,
        max_green=50,
        enforce_max_green=False,
        begin_time=0,
        reward_fn="diff-waiting-time",
        reward_weights=None,
        data_provider=mock_dp,
        num_green_phases=2,
        observation_class=DefaultObservationFunction,
        detectors=[[], ["det_1", "det_2"]],
        window_size=1,
    )
    
    action = np.array([0.5, 0.5])
    
    print("\n" + "="*80)
    print("Testing Different Logging Levels")
    print("="*80)
    
    for level in [1, 2, 3]:
        print(f"\n>>> Testing Level {level}:")
        ts.enable_debug_logging(True, level=level)
        ts.set_next_phase(action)
        ts.compute_reward()
    
    print("\n" + "="*80)
    print("Logging levels test completed!")
    print("="*80)


if __name__ == "__main__":
    test_debug_logging()
    test_logging_levels()
