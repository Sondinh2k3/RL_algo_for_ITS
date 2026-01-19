"""General Plug-In (GPI) Module for Intersection Standardization.

This module implements the GPI component from the GESA architecture.
It standardizes intersection approaches to cardinal directions (N, E, S, W)
regardless of the actual network geometry.

Reference: GESA paper - General Plug-In Module

The GPI module enables:
1. Geometry-agnostic state representation
2. Shared policy across different intersection layouts
3. Transfer learning between networks
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class IntersectionStandardizer:
    """Standardizes intersection approaches to cardinal directions.
    
    The GPI module maps physical road approaches to standard directions
    (North, East, South, West) based on their geometric angles. This enables
    a shared policy to work across intersections with different geometries.
    
    Direction Assignment Logic:
    - Compute the direction vector of each incoming lane (pointing INTO junction)
    - Calculate angle from positive X-axis using arctan2
    - Assign to cardinal direction based on angle ranges:
        * North (225°-315°): Vector points downward (from north)
        * East (135°-225°): Vector points leftward (from east)
        * South (45°-135°): Vector points upward (from south)
        * West (315°-45°): Vector points rightward (from west)
    
    Attributes:
        junction_id: SUMO junction/intersection ID
        standard_map: Mapping from direction (N/E/S/W) to edge ID
        data_provider: Interface for getting network data
    """
    
    # Direction angle ranges (degrees, based on vector pointing INTO junction)
    DIRECTION_RANGES = {
        'N': (225, 315),
        'E': (135, 225),
        'S': (45, 135),
        'W': None,  # 315-360 and 0-45 (wraps around)
    }

    def __init__(self, junction_id: str, data_provider: Any = None):
        """Initialize GPI module for a junction.
        
        Args:
            junction_id: SUMO junction/intersection ID
            data_provider: Object providing network data methods:
                - get_incoming_edges(junction_id) -> List[str]
                - get_lane_shape(lane_id) -> List[Tuple[float, float]]
                If None, uses traci directly.
        """
        self.junction_id = junction_id
        self.data_provider = data_provider
        
        # Standard map: direction -> edge_id
        self.standard_map: Dict[str, Optional[str]] = {
            'N': None,
            'E': None,
            'S': None,
            'W': None
        }
        
        # Cache for computed values
        self._edge_vectors: Dict[str, np.ndarray] = {}
        self._edge_angles: Dict[str, float] = {}
        self._mapped = False
        
    def _get_incoming_edges(self) -> List[str]:
        """Get incoming edges for the junction."""
        if self.data_provider is not None:
            return self.data_provider.get_incoming_edges(self.junction_id)
        else:
            import traci
            return traci.junction.getIncomingEdges(self.junction_id)
    
    def _get_lane_shape(self, lane_id: str) -> List[Tuple[float, float]]:
        """Get lane shape (list of coordinate points)."""
        if self.data_provider is not None:
            return self.data_provider.get_lane_shape(lane_id)
        else:
            import traci
            return traci.lane.getShape(lane_id)

    def _compute_lane_vector(self, lane_id: str) -> np.ndarray:
        """Compute normalized direction vector of a lane.
        
        Uses the last segment of the lane shape to determine the direction
        the lane points (towards the junction stop line).
        
        Args:
            lane_id: Lane identifier
            
        Returns:
            Normalized 2D direction vector
        """
        shape = self._get_lane_shape(lane_id)
        
        if len(shape) < 2:
            return np.array([0.0, 0.0])
        
        # Last two points: approaching the junction
        p2 = np.array(shape[-1])  # End point (stop line)
        p1 = np.array(shape[-2])  # Previous point
        
        vector = p2 - p1
        norm = np.linalg.norm(vector)
        
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        
        return vector / norm

    def _vector_to_angle(self, vector: np.ndarray) -> float:
        """Convert direction vector to angle in degrees [0, 360).
        
        Args:
            vector: 2D direction vector
            
        Returns:
            Angle in degrees from positive X-axis, range [0, 360)
        """
        angle_rad = np.arctan2(vector[1], vector[0])
        angle_deg = np.degrees(angle_rad)
        return angle_deg % 360

    def _angle_to_direction(self, angle: float) -> str:
        """Map angle to cardinal direction.
        
        Args:
            angle: Angle in degrees [0, 360)
            
        Returns:
            Direction string: 'N', 'E', 'S', or 'W'
        """
        # Normalize to [0, 360)
        angle = angle % 360
        
        if 225 <= angle < 315:
            return 'N'
        elif 135 <= angle < 225:
            return 'E'
        elif 45 <= angle < 135:
            return 'S'
        else:  # 315-360 or 0-45
            return 'W'

    def map_intersection(self) -> Dict[str, Optional[str]]:
        """Map incoming edges to standard cardinal directions.
        
        This is the main GPI algorithm that assigns each incoming edge
        to a cardinal direction based on its geometric approach angle.
        
        Returns:
            Dictionary mapping directions ('N', 'E', 'S', 'W') to edge IDs.
            Missing approaches are mapped to None.
        """
        if self._mapped:
            return self.standard_map
            
        # Get all incoming edges
        incoming_edges = self._get_incoming_edges()
        
        if not incoming_edges:
            self._mapped = True
            return self.standard_map
        
        # Compute vectors and angles for each edge
        edge_directions: Dict[str, Tuple[float, str]] = {}
        
        for edge in incoming_edges:
            # Use first lane of edge to represent edge direction
            lane_id = f"{edge}_0"
            
            try:
                vector = self._compute_lane_vector(lane_id)
                self._edge_vectors[edge] = vector
                
                angle = self._vector_to_angle(vector)
                self._edge_angles[edge] = angle
                
                direction = self._angle_to_direction(angle)
                edge_directions[edge] = (angle, direction)
                
            except Exception as e:
                print(f"Warning: Could not compute direction for edge {edge}: {e}")
                continue
        
        # Assign edges to directions
        # If multiple edges map to same direction, use the one with angle closest to ideal
        ideal_angles = {'N': 270, 'E': 180, 'S': 90, 'W': 0}
        
        direction_candidates: Dict[str, List[Tuple[str, float]]] = {
            'N': [], 'E': [], 'S': [], 'W': []
        }
        
        for edge, (angle, direction) in edge_directions.items():
            direction_candidates[direction].append((edge, angle))
        
        # Select best edge for each direction
        for direction, candidates in direction_candidates.items():
            if not candidates:
                continue
                
            if len(candidates) == 1:
                self.standard_map[direction] = candidates[0][0]
            else:
                # Multiple candidates: choose closest to ideal angle
                ideal = ideal_angles[direction]
                best_edge = min(
                    candidates,
                    key=lambda x: min(abs(x[1] - ideal), 360 - abs(x[1] - ideal))
                )[0]
                self.standard_map[direction] = best_edge
        
        self._mapped = True
        return self.standard_map

    def load_config(self, direction_map: Dict[str, Optional[str]], observation_mask: Optional[List[float]] = None):
        """Load direction mapping from configuration (intersection_config.json).
        
        This avoids calling map_intersection() which queries SUMO.
        """
        self.standard_map = direction_map
        if observation_mask is not None:
            # Add self.observation_mask attribute if it doesn't exist (it should be in __init__)
            self._loaded_observation_mask = np.array(observation_mask, dtype=np.float32)
        self._mapped = True
        # print(f"   [IntersectionStandardizer] Loaded config for {self.junction_id}")

    def get_observation_mask(self) -> np.ndarray:
        """Generate binary mask indicating existing approaches.
        
        Returns:
            Binary array [N, E, S, W] where 1 = approach exists, 0 = missing
        """
        # If we loaded a mask from config, use it
        if hasattr(self, '_loaded_observation_mask') and self._loaded_observation_mask is not None:
            return self._loaded_observation_mask
            
        mapping = self.map_intersection()
        return np.array([
            1 if mapping[d] is not None else 0
            for d in ['N', 'E', 'S', 'W']
        ], dtype=np.float32)

    def get_standardized_edges(self) -> List[Optional[str]]:
        """Get edges in standard order [N, E, S, W].
        
        Returns:
            List of edge IDs in order [N, E, S, W], None for missing
        """
        mapping = self.map_intersection()
        return [mapping['N'], mapping['E'], mapping['S'], mapping['W']]

    def get_edge_direction(self, edge_id: str) -> Optional[str]:
        """Get the cardinal direction assigned to an edge.
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            Direction ('N', 'E', 'S', 'W') or None if not mapped
        """
        mapping = self.map_intersection()
        for direction, edge in mapping.items():
            if edge == edge_id:
                return direction
        return None

    def get_direction_edge(self, direction: str) -> Optional[str]:
        """Get the edge assigned to a cardinal direction.
        
        Args:
            direction: Direction string ('N', 'E', 'S', 'W')
            
        Returns:
            Edge ID or None if direction has no approach
        """
        mapping = self.map_intersection()
        return mapping.get(direction.upper())

    def reset(self):
        """Clear cached mappings (e.g., for new simulation)."""
        self.standard_map = {'N': None, 'E': None, 'S': None, 'W': None}
        self._edge_vectors = {}
        self._edge_angles = {}
        self._mapped = False

    def get_lanes_by_direction(self) -> Dict[str, List[str]]:
        """Get all lanes grouped by their cardinal direction.
        
        Returns:
            Dict mapping direction ('N', 'E', 'S', 'W') to list of lane IDs
        """
        mapping = self.map_intersection()
        result = {}
        
        for direction in ['N', 'E', 'S', 'W']:
            edge_id = mapping.get(direction)
            if edge_id is not None and self.data_provider is not None:
                # Get all lanes for this edge
                if hasattr(self.data_provider, 'get_edge_lanes'):
                    result[direction] = self.data_provider.get_edge_lanes(edge_id)
                else:
                    # Fallback: try to get lane count from shape
                    result[direction] = [f"{edge_id}_0"]
            else:
                result[direction] = []
        
        return result

    def export_config(self) -> Dict[str, Any]:
        """Export standardization configuration as a dictionary.
        
        This can be saved to JSON and loaded later for training,
        avoiding the need to re-run standardization.
        
        Returns:
            Dict with direction mappings and observation mask
        """
        return {
            "junction_id": self.junction_id,
            "direction_map": self.map_intersection(),
            "lanes_by_direction": self.get_lanes_by_direction(),
            "observation_mask": self.get_observation_mask().tolist(),
            "edge_angles": {k: float(v) for k, v in self._edge_angles.items()},
        }

    def __repr__(self) -> str:
        mapping = self.map_intersection()
        return (
            f"IntersectionStandardizer(junction='{self.junction_id}', "
            f"N={mapping['N']}, E={mapping['E']}, S={mapping['S']}, W={mapping['W']})"
        )
