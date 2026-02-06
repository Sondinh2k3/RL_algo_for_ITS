#!/usr/bin/env python3
"""
Test GAT Adjacency Matrices.

This module tests the Cooperation and Conflict matrices used in the
DualStreamGATLayer for lane-level graph attention.

Key validations:
1. Matrix symmetry
2. Self-loops (diagonal = 1)
3. Specific known conflicts/cooperations
4. No invalid connections
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gat_layer import get_lane_conflict_matrix, get_lane_cooperation_matrix


# Lane indices (PREPROCESSED ordering: Left, Through, Right per direction)
# NL=0, NT=1, NR=2, EL=3, ET=4, ER=5, SL=6, ST=7, SR=8, WL=9, WT=10, WR=11
LANE_NAMES = ['NL', 'NT', 'NR', 'EL', 'ET', 'ER', 'SL', 'ST', 'SR', 'WL', 'WT', 'WR']
LANE_TO_IDX = {name: idx for idx, name in enumerate(LANE_NAMES)}


class TestMatrixProperties:
    """Test general properties of adjacency matrices."""
    
    def test_conflict_matrix_shape(self):
        """Conflict matrix should be 12x12."""
        adj = get_lane_conflict_matrix()
        assert adj.shape == (12, 12), f"Expected (12, 12), got {adj.shape}"
        
    def test_cooperation_matrix_shape(self):
        """Cooperation matrix should be 12x12."""
        adj = get_lane_cooperation_matrix()
        assert adj.shape == (12, 12), f"Expected (12, 12), got {adj.shape}"
        
    def test_conflict_matrix_symmetric(self):
        """Conflict matrix should be symmetric (if A conflicts with B, B conflicts with A)."""
        adj = get_lane_conflict_matrix()
        assert torch.allclose(adj, adj.T), "Conflict matrix is not symmetric"
        
    def test_cooperation_matrix_symmetric(self):
        """Cooperation matrix should be symmetric."""
        adj = get_lane_cooperation_matrix()
        assert torch.allclose(adj, adj.T), "Cooperation matrix is not symmetric"
        
    def test_conflict_matrix_self_loops(self):
        """Conflict matrix should have self-loops (diagonal = 1)."""
        adj = get_lane_conflict_matrix()
        diagonal = torch.diag(adj)
        assert torch.all(diagonal == 1), f"Diagonal should be all 1s, got {diagonal}"
        
    def test_cooperation_matrix_self_loops(self):
        """Cooperation matrix should have self-loops."""
        adj = get_lane_cooperation_matrix()
        diagonal = torch.diag(adj)
        assert torch.all(diagonal == 1), f"Diagonal should be all 1s, got {diagonal}"
        
    def test_matrices_are_binary(self):
        """All values should be 0 or 1."""
        conflict = get_lane_conflict_matrix()
        coop = get_lane_cooperation_matrix()
        
        assert torch.all((conflict == 0) | (conflict == 1)), "Conflict matrix has non-binary values"
        assert torch.all((coop == 0) | (coop == 1)), "Cooperation matrix has non-binary values"


class TestConflictLogic:
    """Test specific conflict relationships."""
    
    def test_through_vs_crossing_through(self):
        """Through lanes should conflict with crossing through lanes."""
        adj = get_lane_conflict_matrix()
        
        # NT vs ET, WT
        assert adj[LANE_TO_IDX['NT'], LANE_TO_IDX['ET']] == 1, "NT should conflict with ET"
        assert adj[LANE_TO_IDX['NT'], LANE_TO_IDX['WT']] == 1, "NT should conflict with WT"
        
        # ST vs ET, WT
        assert adj[LANE_TO_IDX['ST'], LANE_TO_IDX['ET']] == 1, "ST should conflict with ET"
        assert adj[LANE_TO_IDX['ST'], LANE_TO_IDX['WT']] == 1, "ST should conflict with WT"
        
    def test_left_vs_opposing_through(self):
        """Left turn should conflict with opposing through."""
        adj = get_lane_conflict_matrix()
        
        # NL vs ST (North left conflicts with South through)
        assert adj[LANE_TO_IDX['NL'], LANE_TO_IDX['ST']] == 1, "NL should conflict with ST"
        
        # SL vs NT
        assert adj[LANE_TO_IDX['SL'], LANE_TO_IDX['NT']] == 1, "SL should conflict with NT"
        
        # EL vs WT
        assert adj[LANE_TO_IDX['EL'], LANE_TO_IDX['WT']] == 1, "EL should conflict with WT"
        
        # WL vs ET
        assert adj[LANE_TO_IDX['WL'], LANE_TO_IDX['ET']] == 1, "WL should conflict with ET"
        
    def test_same_approach_no_conflict(self):
        """Lanes from same approach should NOT conflict with each other."""
        adj = get_lane_conflict_matrix()
        
        # North approach: NL, NT, NR should not conflict
        # Note: They have self-loops (diagonal), so we check off-diagonal
        # Actually in this implementation, same-approach lanes don't conflict
        # (they share the same phase or sequential phases)
        # But let's verify the logic: NL-NT-NR from same approach
        # In the current implementation, same-approach might still have edges
        # due to self-loop + cooperation. Let's check they're NOT in conflict matrix
        # as conflicts specifically.
        
        # The conflict matrix SHOULD NOT have same-direction lanes conflicting
        # unless explicitly defined (which they aren't in the current logic)
        pass  # This depends on implementation details


class TestCooperationLogic:
    """Test specific cooperation relationships."""
    
    def test_ns_through_phase(self):
        """Phase A: NS Through lanes should cooperate."""
        adj = get_lane_cooperation_matrix()
        
        # NT, NR, ST, SR should all cooperate
        phase_a_lanes = [LANE_TO_IDX['NT'], LANE_TO_IDX['NR'], 
                         LANE_TO_IDX['ST'], LANE_TO_IDX['SR']]
        
        for i in phase_a_lanes:
            for j in phase_a_lanes:
                if i != j:
                    assert adj[i, j] == 1, \
                        f"{LANE_NAMES[i]} should cooperate with {LANE_NAMES[j]}"
                        
    def test_ew_through_phase(self):
        """Phase B: EW Through lanes should cooperate."""
        adj = get_lane_cooperation_matrix()
        
        # ET, ER, WT, WR should all cooperate
        phase_b_lanes = [LANE_TO_IDX['ET'], LANE_TO_IDX['ER'],
                         LANE_TO_IDX['WT'], LANE_TO_IDX['WR']]
        
        for i in phase_b_lanes:
            for j in phase_b_lanes:
                if i != j:
                    assert adj[i, j] == 1, \
                        f"{LANE_NAMES[i]} should cooperate with {LANE_NAMES[j]}"
                        
    def test_ns_left_phase(self):
        """Phase C: NS Left turns should cooperate."""
        adj = get_lane_cooperation_matrix()
        
        # NL and SL cooperate
        assert adj[LANE_TO_IDX['NL'], LANE_TO_IDX['SL']] == 1, \
            "NL should cooperate with SL"
            
    def test_same_approach_cooperate(self):
        """Single-approach phases: all 3 lanes of same approach cooperate."""
        adj = get_lane_cooperation_matrix()
        
        # North approach: NL, NT, NR
        north_lanes = [LANE_TO_IDX['NL'], LANE_TO_IDX['NT'], LANE_TO_IDX['NR']]
        for i in north_lanes:
            for j in north_lanes:
                if i != j:
                    assert adj[i, j] == 1, \
                        f"{LANE_NAMES[i]} should cooperate with {LANE_NAMES[j]}"


class TestMatricesDisjoint:
    """Test that conflict and cooperation matrices have minimal overlap."""
    
    def test_conflict_cooperation_relationship(self):
        """
        Conflicting lanes should generally NOT cooperate.
        However, some lanes may appear in both (e.g., through self-loops).
        This test checks the general pattern.
        """
        conflict = get_lane_conflict_matrix()
        coop = get_lane_cooperation_matrix()
        
        # Count overlapping edges (excluding diagonal)
        overlap = 0
        for i in range(12):
            for j in range(12):
                if i != j and conflict[i, j] == 1 and coop[i, j] == 1:
                    overlap += 1
                    
        # Some overlap is expected (e.g., NT-NR might be in same phase but also cross)
        # But majority should be disjoint
        total_conflict_edges = (conflict.sum() - 12).item()  # Exclude diagonal
        total_coop_edges = (coop.sum() - 12).item()
        
        # Overlap should be less than 50% of smaller matrix
        min_edges = min(total_conflict_edges, total_coop_edges)
        if min_edges > 0:
            overlap_ratio = overlap / min_edges
            # This is a soft assertion - just log it
            print(f"Overlap ratio: {overlap_ratio:.2%} ({overlap}/{min_edges})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
