"""FRAP (Feature Relation Attention Processing) Module for Phase Standardization.

This module implements the FRAP component from the GESA architecture.
It standardizes traffic signal phases based on movement patterns,
enabling shared policy learning across different phase configurations.

Reference: GESA paper - FRAP Module
           IntelliLight: FRAP concept for phase representation

The FRAP module enables:
1. Phase-agnostic state representation
2. Consistent action space across different signal programs
3. Movement-based phase encoding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum


class MovementType(Enum):
    """Standard traffic movements at an intersection."""
    # Through movements
    NORTH_THROUGH = "NT"
    SOUTH_THROUGH = "ST"
    EAST_THROUGH = "ET"
    WEST_THROUGH = "WT"
    
    # Left turn movements
    NORTH_LEFT = "NL"
    SOUTH_LEFT = "SL"
    EAST_LEFT = "EL"
    WEST_LEFT = "WL"
    
    # Right turn movements (often permitted/overlap)
    NORTH_RIGHT = "NR"
    SOUTH_RIGHT = "SR"
    EAST_RIGHT = "ER"
    WEST_RIGHT = "WR"


@dataclass
class Movement:
    """Represents a traffic movement at an intersection."""
    movement_type: MovementType
    from_direction: str  # N, E, S, W
    to_direction: str    # N, E, S, W
    lanes: List[str]     # Lane IDs serving this movement
    is_protected: bool = True  # Protected vs permitted


@dataclass
class Phase:
    """Represents a traffic signal phase."""
    phase_id: int
    movements: Set[MovementType]  # Movements served by this phase
    duration_range: Tuple[int, int]  # (min, max) duration in seconds
    is_yellow: bool = False
    state: str = ""
    duration: float = 0.0
    green_indices: List[int] = None


class PhaseStandardizer:
    """Standardizes traffic signal phases based on movement patterns.
    
    The FRAP module maps actual signal phases to 8 standard movement combinations.
    This allows the RL agent to learn phase selection based on traffic demand
    for specific movements, regardless of the actual phase definitions.
    
    8 Standard Phase Patterns:
    ============================================
    Group 1: Dual-Through Phases (High throughput)
    - Phase A (0): N-S Through (NT + ST + NR + SR) - North-South through + right
    - Phase B (1): E-W Through (ET + WT + ER + WR) - East-West through + right
    
    Group 2: Dual-Left Phases (Protected left turns)
    - Phase C (2): N-S Left (NL + SL) - North-South left turns
    - Phase D (3): E-W Left (EL + WL) - East-West left turns
    
    Group 3: Single-Approach Phases (One direction green)
    - Phase E (4): North Green (NT + NL + NR) - All North movements
    - Phase F (5): South Green (ST + SL + SR) - All South movements
    - Phase G (6): East Green (ET + EL + ER) - All East movements
    - Phase H (7): West Green (WT + WL + WR) - All West movements
    
    Note: Right turns (NR, SR, ER, WR) are often treated as "free movements"
    and accompany their corresponding through phase.
    
    Action Masking: A phase is VALID only if ALL its required movements exist.
    Example: T-junction missing West -> Phases B, D, H are MASKED.
    
    Attributes:
        junction_id: Traffic signal/junction ID
        phases: List of Phase objects
        movement_to_phase: Mapping from movement to serving phase
        data_provider: Interface for getting signal data
    """
    
    # Number of standard phases
    NUM_STANDARD_PHASES = 8
    
    # Standard 8-phase pattern (comprehensive)
    # Each phase defines REQUIRED movements that must ALL exist for the phase to be valid
    STANDARD_PHASES = {
        # Group 1: Dual-Through (high throughput for main corridors)
        0: {MovementType.NORTH_THROUGH, MovementType.SOUTH_THROUGH},  # Phase A: NS Through
        1: {MovementType.EAST_THROUGH, MovementType.WEST_THROUGH},    # Phase B: EW Through
        
        # Group 2: Dual-Left (protected left turns)
        2: {MovementType.NORTH_LEFT, MovementType.SOUTH_LEFT},        # Phase C: NS Left
        3: {MovementType.EAST_LEFT, MovementType.WEST_LEFT},          # Phase D: EW Left
        
        # Group 3: Single-Approach (one direction gets full green)
        4: {MovementType.NORTH_THROUGH, MovementType.NORTH_LEFT},     # Phase E: North Green
        5: {MovementType.SOUTH_THROUGH, MovementType.SOUTH_LEFT},     # Phase F: South Green
        6: {MovementType.EAST_THROUGH, MovementType.EAST_LEFT},       # Phase G: East Green
        7: {MovementType.WEST_THROUGH, MovementType.WEST_LEFT},       # Phase H: West Green
    }
    
    # Extended phase info with optional right-turn movements
    # Right turns are "free" and accompany their through/approach phase
    STANDARD_PHASES_EXTENDED = {
        0: {MovementType.NORTH_THROUGH, MovementType.SOUTH_THROUGH, 
            MovementType.NORTH_RIGHT, MovementType.SOUTH_RIGHT},       # Phase A + Right
        1: {MovementType.EAST_THROUGH, MovementType.WEST_THROUGH,
            MovementType.EAST_RIGHT, MovementType.WEST_RIGHT},         # Phase B + Right
        2: {MovementType.NORTH_LEFT, MovementType.SOUTH_LEFT},         # Phase C (no right)
        3: {MovementType.EAST_LEFT, MovementType.WEST_LEFT},           # Phase D (no right)
        4: {MovementType.NORTH_THROUGH, MovementType.NORTH_LEFT,
            MovementType.NORTH_RIGHT},                                  # Phase E + Right
        5: {MovementType.SOUTH_THROUGH, MovementType.SOUTH_LEFT,
            MovementType.SOUTH_RIGHT},                                  # Phase F + Right
        6: {MovementType.EAST_THROUGH, MovementType.EAST_LEFT,
            MovementType.EAST_RIGHT},                                   # Phase G + Right
        7: {MovementType.WEST_THROUGH, MovementType.WEST_LEFT,
            MovementType.WEST_RIGHT},                                   # Phase H + Right
    }
    
    # Standard 2-phase pattern (simpler intersections - fallback)
    STANDARD_PHASES_2 = {
        0: {MovementType.NORTH_THROUGH, MovementType.SOUTH_THROUGH,
            MovementType.NORTH_LEFT, MovementType.SOUTH_LEFT},        # All NS
        1: {MovementType.EAST_THROUGH, MovementType.WEST_THROUGH,
            MovementType.EAST_LEFT, MovementType.WEST_LEFT},          # All EW
    }

    def __init__(
        self, 
        junction_id: str, 
        gpi_standardizer: Any = None,
        data_provider: Any = None
    ):
        """Initialize FRAP module.
        
        Args:
            junction_id: Traffic signal ID
            gpi_standardizer: GPI module for direction standardization
            data_provider: Object providing signal program data
        """
        self.junction_id = junction_id
        self.gpi = gpi_standardizer
        self.data_provider = data_provider
        
        # Phase configuration
        self.phases: List[Phase] = []
        self.movements: List[Movement] = []
        self.num_phases = 0
        
        # Mappings
        self.movement_to_phase: Dict[MovementType, int] = {}
        self.phase_to_movements: Dict[int, Set[MovementType]] = {}
        self.lane_to_movement: Dict[str, MovementType] = {}
        
        # Standard phase mapping
        self.actual_to_standard: Dict[int, int] = {}
        self.standard_to_actual: Dict[int, int] = {}
        
        self.phase_config: Dict[str, Any] = {}
        self._configured = False

    def load_config(self, phase_config: Dict[str, Any]):
        """Load phase configuration from dictionary (intersection_config.json).
        
        This avoids calling configure() which makes many TraCI requests.
        """
        self.phase_config = phase_config
        self.num_phases = phase_config.get("num_phases", 0)
        
        # Load actual to standard mapping (ensure keys are ints)
        actual_to_std = phase_config.get("actual_to_standard", {})
        self.actual_to_standard = {int(k): v for k, v in actual_to_std.items()}
        
        # Load standard to actual mapping (ensure keys are ints)
        std_to_actual = phase_config.get("standard_to_actual", {})
        self.standard_to_actual = {int(k): v for k, v in std_to_actual.items()}
        
        # Note: standardized_action only needs actual_to_standard
        
        self._configured = True

    def _get_signal_program(self) -> Any:
        """Get traffic light program/logic."""
        if self.data_provider is not None:
            return self.data_provider.get_traffic_light_program(self.junction_id)
        else:
            import traci
            return traci.trafficlight.getAllProgramLogics(self.junction_id)[0]

    def _get_controlled_links(self) -> List[List[Tuple[str, str, int]]]:
        """Get controlled links (movements) for the signal."""
        if self.data_provider is not None:
            return self.data_provider.get_controlled_links(self.junction_id)
        else:
            import traci
            return traci.trafficlight.getControlledLinks(self.junction_id)

    def _infer_movement_type(
        self, 
        from_lane: str, 
        to_lane: str,
        from_direction: str,
        to_direction: str
    ) -> Optional[MovementType]:
        """Infer movement type from lane connection and directions.
        
        Args:
            from_lane: Incoming lane ID
            to_lane: Outgoing lane ID  
            from_direction: Standard direction of incoming approach (N/E/S/W)
            to_direction: Standard direction of outgoing approach (N/E/S/W)
            
        Returns:
            MovementType or None if cannot determine
        """
        if from_direction is None:
            return None
            
        # Determine turn type based on direction change
        direction_order = ['N', 'E', 'S', 'W']
        
        if from_direction == to_direction:
            # U-turn (usually not a standard movement)
            return None
            
        from_idx = direction_order.index(from_direction)
        to_idx = direction_order.index(to_direction) if to_direction in direction_order else -1
        
        if to_idx == -1:
            return None
        
        # Calculate relative turn
        diff = (to_idx - from_idx) % 4
        
        # Map direction + turn type to movement
        movement_map = {
            ('N', 2): MovementType.NORTH_THROUGH,  # N -> S (through)
            ('N', 1): MovementType.NORTH_RIGHT,    # N -> E (right)
            ('N', 3): MovementType.NORTH_LEFT,     # N -> W (left)
            
            ('S', 2): MovementType.SOUTH_THROUGH,  # S -> N
            ('S', 1): MovementType.SOUTH_RIGHT,    # S -> W
            ('S', 3): MovementType.SOUTH_LEFT,     # S -> E
            
            ('E', 2): MovementType.EAST_THROUGH,   # E -> W
            ('E', 1): MovementType.EAST_RIGHT,     # E -> S
            ('E', 3): MovementType.EAST_LEFT,      # E -> N
            
            ('W', 2): MovementType.WEST_THROUGH,   # W -> E
            ('W', 1): MovementType.WEST_RIGHT,     # W -> N
            ('W', 3): MovementType.WEST_LEFT,      # W -> S
        }
        
        return movement_map.get((from_direction, diff))

    def configure(self):
        """Configure FRAP module by analyzing signal program.
        
        This method:
        1. Extracts phases from signal program
        2. Maps lanes to movements using GPI directions
        3. Determines which movements are served by each phase
        4. Creates standard phase mapping
        """
        if self._configured:
            return
            
        # Get signal program
        try:
            program = self._get_signal_program()
            controlled_links = self._get_controlled_links()
        except Exception as e:
            print(f"Warning: Could not get signal program for {self.junction_id}: {e}")
            self._configured = True
            return
        
        # Extract phases (skip yellow phases)
        phases_data = []
        
        # Handle both sumolib (getPhases()) and traci (.phases) objects
        if hasattr(program, "phases"):
            phases_list = program.phases
        elif hasattr(program, "getPhases"):
            phases_list = program.getPhases()
        else:
            print(f"Warning: Unknown program object type for {self.junction_id}: {type(program)}")
            phases_list = []
            
        for i, phase in enumerate(phases_list):
            state = phase.state
            duration = phase.duration
            
            # Yellow phase detection
            is_yellow = 'y' in state.lower()
            
            if not is_yellow:
                phases_data.append({
                    'index': i,
                    'state': state,
                    'duration': duration,
                    'green_indices': [j for j, c in enumerate(state) if c.upper() == 'G']
                })
        
        self.num_phases = len(phases_data)
        
        # Map controlled links to movements
        link_movements = []
        for link_idx, link_group in enumerate(controlled_links):
            if not link_group:
                continue
                
            # Each link is (from_lane, to_lane, via_lane)
            from_lane_obj = link_group[0][0] if link_group else None
            to_lane_obj = link_group[0][1] if link_group else None
            
            if from_lane_obj is None:
                link_movements.append(None)
                continue
                
            # Handle sumolib Lane objects vs traci string IDs
            from_lane = from_lane_obj.getID() if hasattr(from_lane_obj, "getID") else from_lane_obj
            to_lane = to_lane_obj.getID() if hasattr(to_lane_obj, "getID") else to_lane_obj
            
            # Get edge from lane
            from_edge = from_lane.rsplit('_', 1)[0]
            to_edge = to_lane.rsplit('_', 1)[0] if to_lane else None
            
            # Get standard directions from GPI
            from_dir = None
            to_dir = None
            if self.gpi is not None:
                from_dir = self.gpi.get_edge_direction(from_edge)
                to_dir = self.gpi.get_edge_direction(to_edge) if to_edge else None
            
            movement = self._infer_movement_type(from_lane, to_lane, from_dir, to_dir)
            link_movements.append(movement)
            
            if movement is not None:
                self.lane_to_movement[from_lane] = movement
        
        # Determine movements per phase
        for phase_data in phases_data:
            phase_movements = set()
            for green_idx in phase_data['green_indices']:
                if green_idx < len(link_movements) and link_movements[green_idx]:
                    phase_movements.add(link_movements[green_idx])
            
            phase = Phase(
                phase_id=phase_data['index'],
                movements=phase_movements,
                duration_range=(5, 60),  # Default range
                is_yellow=False,
                state=phase_data['state'],
                duration=phase_data['duration'],
                green_indices=phase_data['green_indices']
            )
            self.phases.append(phase)
            self.phase_to_movements[len(self.phases) - 1] = phase_movements
        
        # Create standard phase mapping
        self._create_standard_mapping()
        
        self._configured = True

    def _create_standard_mapping(self):
        """Create mapping between actual and standard phases.
        
        This method maps each actual phase to a standard phase (0-7) based on:
        1. Movement overlap if movements are identified
        2. Fallback heuristics based on phase index and green patterns
        
        Standard 8 phases:
        - 0: NS Through (Phase A) - N-S direction through
        - 1: EW Through (Phase B) - E-W direction through
        - 2: NS Left (Phase C) - N-S direction left turns
        - 3: EW Left (Phase D) - E-W direction left turns
        - 4: North Green (Phase E) - All North movements
        - 5: South Green (Phase F) - All South movements
        - 6: East Green (Phase G) - All East movements
        - 7: West Green (Phase H) - All West movements
        """
        # Use extended phases for better matching (includes right turns)
        standard = self.STANDARD_PHASES_EXTENDED
        
        # Track which phases have valid movement mappings
        phases_with_movements = [p for p in self.phases if len(p.movements) > 0]
        
        if len(phases_with_movements) > 0:
            # Method 1: Use movement overlap (preferred)
            for actual_idx, phase in enumerate(self.phases):
                best_standard = 0
                best_overlap = 0
                
                for std_idx, std_movements in standard.items():
                    overlap = len(phase.movements & std_movements)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_standard = std_idx
                
                self.actual_to_standard[actual_idx] = best_standard
                self.standard_to_actual[best_standard] = actual_idx
        else:
            # Method 2: Fallback - use phase index pattern
            # Common patterns:
            # - 2 phases: alternate NS (0) and EW (1)
            # - 4 phases: NS-T(0), EW-T(1), NS-L(2), EW-L(3)
            # - 8 phases: direct mapping to standard phases
            self._create_fallback_mapping()
    
    def _create_fallback_mapping(self):
        """Create fallback mapping when movement info is unavailable.
        
        Uses heuristics based on common signal timing patterns:
        - 2-phase: NS-through (0), EW-through (1)
        - 4-phase: NS-T (0), EW-T (1), NS-L (2), EW-L (3)
        - 8-phase: Direct mapping to all 8 standard phases
        """
        if self.num_phases <= 2:
            # Simple 2-phase: NS-through then EW-through
            for i in range(self.num_phases):
                self.actual_to_standard[i] = i  # 0 -> 0 (NS-T), 1 -> 1 (EW-T)
                self.standard_to_actual[i] = i
        elif self.num_phases <= 4:
            # 4-phase NEMA-like: map to first 4 standard phases
            # Typical order: NS-Through, EW-Through, NS-Left, EW-Left
            for i in range(self.num_phases):
                std_idx = i % 4
                self.actual_to_standard[i] = std_idx
                self.standard_to_actual[std_idx] = i
        else:
            # 8+ phases: distribute across 8 standard phases
            num_groups = self.NUM_STANDARD_PHASES  # 8
            
            for i in range(self.num_phases):
                # Map to one of 8 standard phases
                # Using modulo to cycle through: 0,1,2,3,4,5,6,7,0,1,...
                std_idx = i % num_groups
                self.actual_to_standard[i] = std_idx
                
                # Standard to actual: pick representative phase from each group
                if std_idx not in self.standard_to_actual:
                    self.standard_to_actual[std_idx] = i

    def get_phase_demand_features(
        self, 
        density_by_direction: Dict[str, float],
        queue_by_direction: Dict[str, float]
    ) -> np.ndarray:
        """Compute standardized phase-based demand features.
        
        This creates a fixed-size feature vector representing traffic demand
        for each of the 8 standard phases, enabling shared policy learning.
        
        Args:
            density_by_direction: Traffic density per direction {N/E/S/W: value}
            queue_by_direction: Queue length per direction {N/E/S/W: value}
            
        Returns:
            Feature vector [phase_0_demand, ..., phase_7_demand] for density,
            then [phase_0_queue, ..., phase_7_queue] for queue
            Total: 16 values (8 phases * 2 metrics)
        """
        # Aggregate demand by standard phase
        num_standard_phases = self.NUM_STANDARD_PHASES  # 8 phases
        demands = np.zeros(num_standard_phases * 2)  # density + queue per phase
        
        direction_to_demand = {
            'N': (density_by_direction.get('N', 0), queue_by_direction.get('N', 0)),
            'S': (density_by_direction.get('S', 0), queue_by_direction.get('S', 0)),
            'E': (density_by_direction.get('E', 0), queue_by_direction.get('E', 0)),
            'W': (density_by_direction.get('W', 0), queue_by_direction.get('W', 0)),
        }
        
        # Phase 0 (A): NS Through - average of N and S through traffic
        demands[0] = (direction_to_demand['N'][0] + direction_to_demand['S'][0]) / 2
        demands[8] = (direction_to_demand['N'][1] + direction_to_demand['S'][1]) / 2
        
        # Phase 1 (B): EW Through - average of E and W through traffic
        demands[1] = (direction_to_demand['E'][0] + direction_to_demand['W'][0]) / 2
        demands[9] = (direction_to_demand['E'][1] + direction_to_demand['W'][1]) / 2
        
        # Phase 2 (C): NS Left - estimate as portion of NS demand (typically ~30%)
        demands[2] = demands[0] * 0.3
        demands[10] = demands[8] * 0.3
        
        # Phase 3 (D): EW Left - estimate as portion of EW demand
        demands[3] = demands[1] * 0.3
        demands[11] = demands[9] * 0.3
        
        # Phase 4 (E): North Green - all North movements
        demands[4] = direction_to_demand['N'][0]
        demands[12] = direction_to_demand['N'][1]
        
        # Phase 5 (F): South Green - all South movements
        demands[5] = direction_to_demand['S'][0]
        demands[13] = direction_to_demand['S'][1]
        
        # Phase 6 (G): East Green - all East movements
        demands[6] = direction_to_demand['E'][0]
        demands[14] = direction_to_demand['E'][1]
        
        # Phase 7 (H): West Green - all West movements
        demands[7] = direction_to_demand['W'][0]
        demands[15] = direction_to_demand['W'][1]
        
        return demands.astype(np.float32)

    def standardize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert standard action to actual phase durations.
        
        Maps actions from standard 8-phase format to actual signal phases.
        Handles cases where:
        - num_phases < 8: aggregates standard phase values
        - num_phases == 8: direct mapping
        - num_phases > 8: multiple actual phases may share same standard index
        
        Args:
            action: Standard phase durations/ratios (8 values)
                   [NS_Through, EW_Through, NS_Left, EW_Left,
                    North_Green, South_Green, East_Green, West_Green]
            
        Returns:
            Actual phase durations matching signal program (num_phases values)
        """
        action = np.asarray(action).flatten()
        actual_action = np.zeros(self.num_phases)
        
        # Map each actual phase to its corresponding standard action value
        for actual_idx in range(self.num_phases):
            std_idx = self.actual_to_standard.get(actual_idx, 0)
            
            if std_idx < len(action):
                actual_action[actual_idx] = action[std_idx]
            else:
                # Fallback: wrap around if std_idx exceeds action length
                actual_action[actual_idx] = action[std_idx % len(action)]
        
        # Normalize to ensure ratios sum to 1.0 (if they're meant to be ratios)
        total = actual_action.sum()
        if total > 0:
            actual_action = actual_action / total
        else:
            # Equal distribution if all zeros
            actual_action = np.ones(self.num_phases) / self.num_phases
        
        return actual_action

    def get_movement_mask(self) -> np.ndarray:
        """Get binary mask indicating which standard movements exist.
        
        Returns:
            Binary array for 8 standard movements (4 through + 4 left)
        """
        all_movements = set()
        for phase in self.phases:
            all_movements.update(phase.movements)
        
        standard_movements = [
            MovementType.NORTH_THROUGH, MovementType.SOUTH_THROUGH,
            MovementType.EAST_THROUGH, MovementType.WEST_THROUGH,
            MovementType.NORTH_LEFT, MovementType.SOUTH_LEFT,
            MovementType.EAST_LEFT, MovementType.WEST_LEFT,
        ]
        
        return np.array([
            1 if m in all_movements else 0
            for m in standard_movements
        ], dtype=np.float32)

    def get_phase_mask(self) -> np.ndarray:
        """Get mask indicating which standard phases are available/valid.
        
        A phase is VALID (mask=1) if and only if ALL its required movements exist.
        This implements the Action Masking described in the GESA specification.
        
        Example for T-junction missing West:
        - Phase B (EW Through): MASKED (needs WT which doesn't exist)
        - Phase D (EW Left): MASKED (needs WL which doesn't exist)
        - Phase H (West Green): MASKED (needs WT, WL)
        - Result: [1, 0, 1, 0, 1, 1, 1, 0]
        
        Returns:
            Binary array for 8 standard phases [A, B, C, D, E, F, G, H]
        """
        mask = np.zeros(self.NUM_STANDARD_PHASES)  # 8 phases
        
        # Collect all movements that exist at this intersection
        all_movements = set()
        for phase in self.phases:
            all_movements.update(phase.movements)
        
        # Also check lane_to_movement for additional coverage
        all_movements.update(self.lane_to_movement.values())
        
        # Check each standard phase
        for std_idx, required_movements in self.STANDARD_PHASES.items():
            if std_idx < self.NUM_STANDARD_PHASES:
                # Phase is valid only if ALL required movements exist
                if required_movements.issubset(all_movements):
                    mask[std_idx] = 1
                # Special case: if we have the actual mapping, also mark as valid
                elif std_idx in self.standard_to_actual:
                    mask[std_idx] = 1
        
        # If no phases are valid (fallback), enable basic phases 0 and 1
        if mask.sum() == 0:
            mask[0] = 1  # NS Through always available as fallback
            mask[1] = 1  # EW Through always available as fallback
        
        return mask.astype(np.float32)

    def reset(self):
        """Reset module state."""
        self._configured = False
        self.phases = []
        self.movements = []
        self.movement_to_phase = {}
        self.phase_to_movements = {}
        self.actual_to_standard = {}
        self.standard_to_actual = {}

    def __repr__(self) -> str:
        return (
            f"PhaseStandardizer(junction='{self.junction_id}', "
            f"num_phases={self.num_phases}, "
            f"mapping={self.actual_to_standard})"
        )
