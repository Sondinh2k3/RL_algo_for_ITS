
import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch, mock_open

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock sumolib and traci before importing SumoSimulator
sys.modules['sumolib'] = MagicMock()
sys.modules['traci'] = MagicMock()
sys.modules['pyvirtualdisplay'] = MagicMock()
sys.modules['pyvirtualdisplay.smartdisplay'] = MagicMock()

from src.sim.Sumo_sim import SumoSimulator

class TestSumoSimLogic(unittest.TestCase):
    def setUp(self):
        self.sim = SumoSimulator(
            net_file="dummy.net.xml",
            route_file="dummy.rou.xml",
            preprocessing_config="dummy_config.json"
        )
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"intersections": {"J1": {"lanes_by_direction": {"N": ["N_0", "N_1"], "E": ["E_0", "E_1"], "S": ["S_0", "S_1"], "W": ["W_0", "W_1"]}}}}')
    @patch('os.path.exists', return_value=True)
    @patch('src.sim.Sumo_sim.TrafficSignal')
    def test_lane_ordering_reversal(self, mock_traffic_signal, mock_exists, mock_file):
        """Test that lanes are reversed (Right-to-Left -> Left-to-Right) when loading from config."""
        
        # Mock connection
        mock_conn = MagicMock()
        mock_conn.trafficlight.getIDList.return_value = ["J1"]
        mock_conn.trafficlight.getAllProgramLogics.return_value = [MagicMock(phases=[1, 2])] # 1 green phase
        
        # Mock lanearea (detectors)
        mock_conn.lanearea.getIDList.return_value = ["det_N_0", "det_N_1"]
        mock_conn.lanearea.getLaneID.side_effect = lambda d: d.replace("det_", "")
        
        # Run _build_traffic_signals
        self.sim._build_traffic_signals(mock_conn)
        
        # Check TrafficSignal initialization
        # We expect lanes to be reversed for each direction
        # Input N: ["N_0", "N_1"] (Right to Left)
        # Expected N: ["N_1", "N_0"] (Left to Right)
        
        # Total expected order: N(rev), E(rev), S(rev), W(rev)
        expected_lanes = ["N_1", "N_0", "E_1", "E_0", "S_1", "S_0", "W_1", "W_0"]
        
        # Verify TrafficSignal was called
        self.assertTrue(mock_traffic_signal.called)
        
        # Verify that SumoSimulator stored the correct lanes
        self.assertIn("J1", self.sim.ts_lanes)
        self.assertEqual(self.sim.ts_lanes["J1"], expected_lanes)
        
        # Verify get_controlled_lanes returns the stored lanes
        self.assertEqual(self.sim.get_controlled_lanes("J1"), expected_lanes)

if __name__ == '__main__':
    unittest.main()
