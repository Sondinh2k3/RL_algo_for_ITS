
import unittest
from unittest.mock import MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.drl_algo.traffic_signal import TrafficSignal

class TestDetectorAlignment(unittest.TestCase):
    def setUp(self):
        self.ts_id = "J1"
        self.mock_data_provider = MagicMock()
        
        # Mock detectors list (ordered as passed from SumoSimulator)
        # Assume SumoSimulator passed them in correct order: [N_L, N_T, N_R, E_L, ...]
        self.detectors = [[], ["det_N_L", "det_N_T", "det_N_R", "det_E_L"]]
        
        self.ts = TrafficSignal(
            ts_id=self.ts_id,
            delta_time=90, # Increased delta_time to satisfy min_green * num_phases <= total_green
            yellow_time=2,
            min_green=5,
            max_green=30,
            enforce_max_green=False,
            begin_time=0,
            reward_fn="diff-waiting-time",
            reward_weights=None,
            data_provider=self.mock_data_provider,
            num_green_phases=4,
            observation_class=MagicMock(),
            detectors=self.detectors
        )
        
        # Mock detector history to have some data
        self.ts.detector_history["density"]["det_N_L"] = [0.1]
        self.ts.detector_history["density"]["det_N_T"] = [0.2]
        self.ts.detector_history["density"]["det_N_R"] = [0.3]
        self.ts.detector_history["density"]["det_E_L"] = [0.4]
        
    def test_observation_order(self):
        """Test that observation vector preserves detector order."""
        # Call get_lanes_density_by_detectors
        densities = self.ts.get_lanes_density_by_detectors()
        
        # Expected order: [val_N_L, val_N_T, val_N_R, val_E_L]
        expected = [0.1, 0.2, 0.3, 0.4]
        
        self.assertEqual(densities, expected)
        
        # Verify it iterated self.detectors_e2 in order
        # self.detectors_e2 is set in __init__ from detectors[1]
        self.assertEqual(self.ts.detectors_e2, ["det_N_L", "det_N_T", "det_N_R", "det_E_L"])

if __name__ == '__main__':
    unittest.main()
