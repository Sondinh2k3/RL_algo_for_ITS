"""Test Case 1: Ngã tư chuẩn (4 hướng)"""

import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from environment.drl_algo.traffic_signal import TrafficSignal
from environment.drl_algo.observations import DefaultObservationFunction

class TestStandardIntersectionObservation(unittest.TestCase):
    def test_standard_intersection_shape_and_padding(self):
        """
        Test Case 1: Ngã tư chuẩn (4 hướng)
        Input: Dữ liệu giả lập ngã tư đầy đủ.
        Expectation: Tensor đầu ra có shape [12, 5] (12 làn chuẩn, 5 đặc trưng). Không có giá trị 0 do padding.
        """
        # Mock TrafficSignal
        ts = MagicMock()
        ts.id = "center"
        # 12 Detectors (12 lanes implies 4 approaches * 3 lanes)
        ts.detectors_e2 = [f"e2_det_{i}" for i in range(12)]
        
        # Mock data return values for features.
        # We assume 5 features: density, queue, occupancy, speed, last_green_ratio.
        # We provide data for all potential getters to simulate "full intersection data".
        
        # Using non-zero values to verify "Not 0 due to padding"
        ts.get_lanes_density_by_detectors.return_value = [0.1] * 12
        ts.get_lanes_queue_by_detectors.return_value = [0.2] * 12
        ts.get_lanes_occupancy_by_detectors.return_value = [0.3] * 12
        ts.get_lanes_average_speed_by_detectors.return_value = [0.4] * 12
        # Use a distinguishable value for the new feature
        ts.get_lanes_last_green_ratio_by_detectors.return_value = [0.5] * 12

        # Initialize Observation Function
        obs_fn = DefaultObservationFunction(ts)
        
        # Execute
        observation = obs_fn()
        
        # Check Shape
        # Expectation: [12, 5]
        # Since standard Gym spaces are usually flat, we might need to reshape to interpret it as [12, 5]
        # OR the user expects the observation function to return a 2D array (less common for gym but possible for custom models).
        
        # If the observation is flat, we check if it has 12*5 = 60 elements.
        expected_elements = 12 * 5
        
        if observation.shape == (12, 5):
            reshaped_obs = observation
        elif observation.size == expected_elements:
             # Assume it's flat and we need to view it as [12, 5]
             reshaped_obs = observation.reshape(12, 5)
        else:
            self.fail(f"Observation shape {observation.shape} (size {observation.size}) does not match expectation [12, 5] (size 60). "
                      f"Current features per lane seems to be {observation.size / 12}")
            
        self.assertEqual(reshaped_obs.shape, (12, 5), "Output tensor shape mismatch")
        
        # Check padding
        # "Không có giá trị 0 do padding" -> check that no element is 0 (since we provided non-zero inputs)
        # If there are zeros, it implies either we fed zeros (we didn't) or there's padding logic active.
        self.assertTrue(np.all(reshaped_obs != 0), "Observation contains zeros, which might indicate padding or missing data")
        
        # Check padding
        # "Không có giá trị 0 do padding" -> check that no element is 0 (since we provided non-zero inputs)
        # If there are zeros, it implies either we fed zeros (we didn't) or there's padding logic active.
        self.assertTrue(np.all(reshaped_obs != 0), "Observation contains zeros, which might indicate padding or missing data")

if __name__ == '__main__':
    unittest.main()
