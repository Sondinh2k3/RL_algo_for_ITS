
import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.standardizer import IntersectionStandardizer
from src.preprocessing.frap import PhaseStandardizer, MovementType

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.junction_id = "J1"
        self.mock_data_provider = MagicMock()
        
    def test_gpi_direction_mapping(self):
        """Test that GPI correctly maps lanes to N/E/S/W based on vectors."""
        # Setup mock data
        # Lane coming from North (pointing South) -> Angle ~270 (pointing down)
        # Wait, GPI logic says:
        # North (225-315): Vector points downward (from north)
        # So a lane from North heading South has vector (0, -1).
        # atan2(-1, 0) = -90 deg = 270 deg. Correct.
        
        self.mock_data_provider.get_incoming_edges.return_value = ["edge_N", "edge_E", "edge_S", "edge_W"]
        
        # Mock lane shapes
        # edge_N_0: (0, 10) -> (0, 0) (Points South, comes from North)
        # edge_E_0: (10, 0) -> (0, 0) (Points West, comes from East)
        # edge_S_0: (0, -10) -> (0, 0) (Points North, comes from South)
        # edge_W_0: (-10, 0) -> (0, 0) (Points East, comes from West)
        
        def get_lane_shape(lane_id):
            if "edge_N" in lane_id: return [(0, 10), (0, 0)]
            if "edge_E" in lane_id: return [(10, 0), (0, 0)]
            if "edge_S" in lane_id: return [(0, -10), (0, 0)]
            if "edge_W" in lane_id: return [(-10, 0), (0, 0)]
            return []
            
        self.mock_data_provider.get_lane_shape.side_effect = get_lane_shape
        
        # Mock get_edge_lanes to return 1 lane per edge
        self.mock_data_provider.get_edge_lanes.side_effect = lambda e: [f"{e}_0"]
        
        gpi = IntersectionStandardizer(self.junction_id, self.mock_data_provider)
        direction_map = gpi.map_intersection()
        
        self.assertEqual(direction_map['N'], "edge_N")
        self.assertEqual(direction_map['E'], "edge_E")
        self.assertEqual(direction_map['S'], "edge_S")
        self.assertEqual(direction_map['W'], "edge_W")

    def test_frap_phase_mapping(self):
        """Test that FRAP correctly identifies standard phases."""
        # Mock GPI
        mock_gpi = MagicMock()
        mock_gpi.get_edge_direction.side_effect = lambda e: e.split('_')[1] # edge_N -> N
        
        frap = PhaseStandardizer(self.junction_id, gpi_standardizer=mock_gpi, data_provider=self.mock_data_provider)
        
        # Mock signal program
        # Phase 0: NS Green (G)
        # Phase 1: EW Green (G)
        mock_program = MagicMock()
        mock_phase0 = MagicMock()
        mock_phase0.state = "GGrr" # Indices 0,1 Green
        mock_phase0.duration = 30
        
        mock_phase1 = MagicMock()
        mock_phase1.state = "rrGG" # Indices 2,3 Green
        mock_phase1.duration = 30
        
        # Use list for phases to match sumolib/traci hybrid handling
        mock_program.phases = [mock_phase0, mock_phase1]
        # Also support getPhases()
        mock_program.getPhases.return_value = [mock_phase0, mock_phase1]
        
        self.mock_data_provider.get_traffic_light_program.return_value = mock_program
        
        # Mock controlled links
        # Link 0: N -> S (Through)
        # Link 1: S -> N (Through)
        # Link 2: E -> W (Through)
        # Link 3: W -> E (Through)
        
        # Format: [[(from_lane, to_lane, via_lane)], ...]
        links = [
            [("edge_N_0", "edge_S_out_0", "via0")], # Index 0
            [("edge_S_0", "edge_N_out_0", "via1")], # Index 1
            [("edge_E_0", "edge_W_out_0", "via2")], # Index 2
            [("edge_W_0", "edge_E_out_0", "via3")], # Index 3
        ]
        self.mock_data_provider.get_controlled_links.return_value = links
        
        # Configure FRAP
        frap.configure()
        
        # Check phases
        self.assertEqual(frap.num_phases, 2)
        
        # Check movement mapping
        # Phase 0 should have NS Through movements
        p0_movements = frap.phases[0].movements
        self.assertTrue(MovementType.NORTH_THROUGH in p0_movements)
        self.assertTrue(MovementType.SOUTH_THROUGH in p0_movements)
        
        # Phase 1 should have EW Through movements
        p1_movements = frap.phases[1].movements
        self.assertTrue(MovementType.EAST_THROUGH in p1_movements)
        self.assertTrue(MovementType.WEST_THROUGH in p1_movements)

    def test_standardize_action(self):
        """Test that standardize_action correctly translates standard actions to actual phases."""
        # Mock GPI
        mock_gpi = MagicMock()
        mock_gpi.get_edge_direction.side_effect = lambda e: e.split('_')[1]  # edge_N -> N
        
        frap = PhaseStandardizer(self.junction_id, gpi_standardizer=mock_gpi, data_provider=self.mock_data_provider)
        
        # Mock signal program with 2 phases
        mock_program = MagicMock()
        mock_phase0 = MagicMock()
        mock_phase0.state = "GGrr"
        mock_phase0.duration = 30
        
        mock_phase1 = MagicMock()
        mock_phase1.state = "rrGG"
        mock_phase1.duration = 30
        
        mock_program.phases = [mock_phase0, mock_phase1]
        mock_program.getPhases.return_value = [mock_phase0, mock_phase1]
        
        self.mock_data_provider.get_traffic_light_program.return_value = mock_program
        
        # Mock controlled links
        links = [
            [("edge_N_0", "edge_S_out_0", "via0")],
            [("edge_S_0", "edge_N_out_0", "via1")],
            [("edge_E_0", "edge_W_out_0", "via2")],
            [("edge_W_0", "edge_E_out_0", "via3")],
        ]
        self.mock_data_provider.get_controlled_links.return_value = links
        
        # Configure FRAP
        frap.configure()
        
        # Test standardize_action
        # Standard action format: [NS_through_ratio, NS_left_ratio, EW_through_ratio, EW_left_ratio]
        standard_action = np.array([0.6, 0.0, 0.4, 0.0])
        
        actual_action = frap.standardize_action(standard_action)
        
        # Should have 2 actual phases
        self.assertEqual(len(actual_action), 2)
        
        # The actual values should be mapped from standard phases
        # Phase 0 (NS Through) should get standard[0] = 0.6
        # Phase 1 (EW Through) should get standard[2] = 0.4
        print(f"Standard action: {standard_action}")
        print(f"Actual action: {actual_action}")
        print(f"Phase mapping: {frap.actual_to_standard}")

if __name__ == '__main__':
    unittest.main()
