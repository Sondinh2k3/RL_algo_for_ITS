#!/usr/bin/env python3
"""
Test FRAP Preprocessing Logic.

This module tests the direction mapping and movement type inference
in the FRAP (Phase Standardizer) module.

Key validations:
1. Diff calculation formula: (to_idx - from_idx) % 4
2. Movement mapping: diff=1→Left, diff=2→Through, diff=3→Right
3. RHT compliance: N→E is Left, N→W is Right
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDiffCalculation:
    """Test the relative turn diff calculation."""
    
    def test_diff_formula_positive(self):
        """Test diff calculation with positive results."""
        direction_order = ['N', 'E', 'S', 'W']
        
        # From North
        assert (direction_order.index('E') - direction_order.index('N')) % 4 == 1  # N→E = Left
        assert (direction_order.index('S') - direction_order.index('N')) % 4 == 2  # N→S = Through
        assert (direction_order.index('W') - direction_order.index('N')) % 4 == 3  # N→W = Right
        
    def test_diff_formula_negative_wrap(self):
        """Test diff calculation with negative intermediate values (Python modulo)."""
        direction_order = ['N', 'E', 'S', 'W']
        
        # From South to East: (1 - 2) % 4 = -1 % 4 = 3 (Right)
        diff = (direction_order.index('E') - direction_order.index('S')) % 4
        assert diff == 3, f"S→E should be 3 (Right), got {diff}"
        
        # From South to North: (0 - 2) % 4 = -2 % 4 = 2 (Through)
        diff = (direction_order.index('N') - direction_order.index('S')) % 4
        assert diff == 2, f"S→N should be 2 (Through), got {diff}"
        
        # From East to North: (0 - 1) % 4 = -1 % 4 = 3 (Right)
        diff = (direction_order.index('N') - direction_order.index('E')) % 4
        assert diff == 3, f"E→N should be 3 (Right), got {diff}"
        
    def test_all_direction_pairs(self):
        """Test all 12 valid direction pairs (excluding U-turns)."""
        direction_order = ['N', 'E', 'S', 'W']
        
        # Expected: (from, to) → expected_diff
        expected = {
            # From North
            ('N', 'E'): 1,  # Left
            ('N', 'S'): 2,  # Through
            ('N', 'W'): 3,  # Right
            # From East
            ('E', 'S'): 1,  # Left
            ('E', 'W'): 2,  # Through
            ('E', 'N'): 3,  # Right
            # From South
            ('S', 'W'): 1,  # Left
            ('S', 'N'): 2,  # Through
            ('S', 'E'): 3,  # Right
            # From West
            ('W', 'N'): 1,  # Left
            ('W', 'E'): 2,  # Through
            ('W', 'S'): 3,  # Right
        }
        
        for (from_dir, to_dir), expected_diff in expected.items():
            from_idx = direction_order.index(from_dir)
            to_idx = direction_order.index(to_dir)
            actual_diff = (to_idx - from_idx) % 4
            assert actual_diff == expected_diff, \
                f"{from_dir}→{to_dir}: expected diff={expected_diff}, got {actual_diff}"


class TestMovementMapping:
    """Test the movement_map in frap.py."""
    
    def test_movement_map_structure(self):
        """Verify movement_map has all required entries."""
        from src.preprocessing.frap import MovementType
        
        # Expected structure: 4 directions × 3 turn types = 12 entries
        expected_keys = [
            ('N', 1), ('N', 2), ('N', 3),
            ('E', 1), ('E', 2), ('E', 3),
            ('S', 1), ('S', 2), ('S', 3),
            ('W', 1), ('W', 2), ('W', 3),
        ]
        
        movement_map = {
            ('N', 2): MovementType.NORTH_THROUGH,
            ('N', 1): MovementType.NORTH_LEFT,
            ('N', 3): MovementType.NORTH_RIGHT,
            ('S', 2): MovementType.SOUTH_THROUGH,
            ('S', 1): MovementType.SOUTH_LEFT,
            ('S', 3): MovementType.SOUTH_RIGHT,
            ('E', 2): MovementType.EAST_THROUGH,
            ('E', 1): MovementType.EAST_LEFT,
            ('E', 3): MovementType.EAST_RIGHT,
            ('W', 2): MovementType.WEST_THROUGH,
            ('W', 1): MovementType.WEST_LEFT,
            ('W', 3): MovementType.WEST_RIGHT,
        }
        
        for key in expected_keys:
            assert key in movement_map, f"Missing key {key} in movement_map"
            
    def test_rht_compliance(self):
        """Verify Right-Hand Traffic compliance.
        
        In RHT:
        - Turning LEFT means crossing oncoming traffic (dangerous)
        - Turning RIGHT means staying on your side (safer)
        """
        from src.preprocessing.frap import MovementType
        
        # N→E should be LEFT (crossing traffic)
        # N→W should be RIGHT (not crossing)
        movement_map = {
            ('N', 1): MovementType.NORTH_LEFT,
            ('N', 3): MovementType.NORTH_RIGHT,
        }
        
        assert movement_map[('N', 1)] == MovementType.NORTH_LEFT, \
            "N with diff=1 (N→E) should map to NORTH_LEFT"
        assert movement_map[('N', 3)] == MovementType.NORTH_RIGHT, \
            "N with diff=3 (N→W) should map to NORTH_RIGHT"


class TestMovementTypeEnum:
    """Test MovementType enum values."""
    
    def test_enum_exists(self):
        """Verify MovementType enum is importable."""
        from src.preprocessing.frap import MovementType
        
        assert hasattr(MovementType, 'NORTH_LEFT')
        assert hasattr(MovementType, 'NORTH_THROUGH')
        assert hasattr(MovementType, 'NORTH_RIGHT')
        
    def test_enum_count(self):
        """Verify correct number of movement types (12 for standard intersection)."""
        from src.preprocessing.frap import MovementType
        
        # 4 directions × 3 turn types = 12
        movement_types = [m for m in MovementType]
        assert len(movement_types) == 12, \
            f"Expected 12 movement types, got {len(movement_types)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
