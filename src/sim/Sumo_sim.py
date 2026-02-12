"""SUMO Simulator Implementation - Complete SUMO backend for traffic simulation.

This module implements the SimulatorAPI for SUMO. It encapsulates all SUMO-specific
logic (traffic signal management, observation, reward, etc.) and provides a clean API
that the environment can use without knowing about SUMO internals.

Multi-worker support: Uses PID in connection labels to avoid conflicts between Ray workers.
"""

import os
import json
import sys
import sumolib
import libsumo
import traci
import numpy as np
import numpy as nplibsumo
from typing import Dict, List, Optional, Tuple, Any, Union
from pyvirtualdisplay.smartdisplay import SmartDisplay

# Add SUMO tools to path
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

from gymnasium import spaces
from .simulator_api import SimulatorAPI

# Import traffic signal and observation functions
# Try multiple import paths for compatibility
try:
    from src.environment.drl_algo.traffic_signal import TrafficSignal
    from src.environment.drl_algo.observations import (
        DefaultObservationFunction, 
        SpatioTemporalObservationFunction,
        NeighborTemporalObservationFunction
    )
except ImportError:
    try:
        # Fallback: try relative import
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment', 'drl_algo'))
        from environment.drl_algo.traffic_signal import TrafficSignal
        from environment.drl_algo.observations import (
            DefaultObservationFunction, 
            SpatioTemporalObservationFunction,
            NeighborTemporalObservationFunction
        )
    except ImportError as e:
        print(f"Warning: Failed to import traffic signal modules: {e}")
        TrafficSignal = None
        DefaultObservationFunction = None
        SpatioTemporalObservationFunction = None
        NeighborTemporalObservationFunction = None

# Import preprocessing modules for phase standardization
try:
    from src.preprocessing.standardizer import IntersectionStandardizer
    from src.preprocessing.frap import PhaseStandardizer
except ImportError:
    try:
        # Fallback: try without src prefix
        from preprocessing.standardizer import IntersectionStandardizer
        from preprocessing.frap import PhaseStandardizer
    except ImportError as e:
        print(f"Warning: Failed to import preprocessing modules: {e}")
        IntersectionStandardizer = None
        PhaseStandardizer = None

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class NeighborProvider:
    """Provides neighbor information for NeighborTemporalObservationFunction.
    
    This class enables pre-packaged observations with neighbor features,
    which is essential for Local GNN processing with RLlib batching.
    
    It provides:
    - Neighbor IDs for each traffic signal
    - Observation history access for neighbors
    """
    
    def __init__(self, traffic_signals: dict, adjacency_map: dict, 
                 direction_map: dict = None, max_neighbors: int = 4):
        """Initialize NeighborProvider.
        
        Args:
            traffic_signals: Dict mapping ts_id -> TrafficSignal object
            adjacency_map: Dict mapping ts_id -> list of neighbor ts_ids
            direction_map: Dict mapping ts_id -> {neighbor_id: direction_index}
                          where direction_index is 0=N, 1=E, 2=S, 3=W
            max_neighbors: Maximum number of neighbors (K)
        """
        self.traffic_signals = traffic_signals
        self.adjacency_map = adjacency_map
        self.direction_map = direction_map or {}
        self.max_neighbors = max_neighbors
        
    def get_neighbor_ids(self, ts_id: str) -> List[str]:
        """Get list of neighbor IDs for a traffic signal.
        
        Args:
            ts_id: Traffic signal ID
            
        Returns:
            List of neighbor ts_ids (up to max_neighbors)
        """
        neighbors = self.adjacency_map.get(ts_id, [])
        return neighbors[:self.max_neighbors]
    
    def get_neighbor_directions(self, ts_id: str) -> List[int]:
        """Get direction indices for neighbors of a traffic signal.
        
        Direction indices follow: 0=North, 1=East, 2=South, 3=West.
        Padded with -1 for missing neighbors.
        
        Args:
            ts_id: Traffic signal ID
            
        Returns:
            List of direction indices [K] (0-3 for valid, -1 for padding)
        """
        neighbors = self.get_neighbor_ids(ts_id)
        ts_dirs = self.direction_map.get(ts_id, {})
        directions = []
        for neighbor_id in neighbors:
            directions.append(ts_dirs.get(neighbor_id, 0))
        # Pad with -1 for missing neighbors
        while len(directions) < self.max_neighbors:
            directions.append(-1)
        return directions[:self.max_neighbors]
    
    def get_observation_history(self, ts_id: str, window_size: int) -> Optional[List]:
        """Get observation history for a traffic signal.
        
        Args:
            ts_id: Traffic signal ID
            window_size: Number of historical timesteps (T)
            
        Returns:
            List of observations [T, feature_dim] or None if not available
        """
        if ts_id not in self.traffic_signals:
            return None
            
        ts = self.traffic_signals[ts_id]
        if hasattr(ts, 'get_observation_history'):
            return ts.get_observation_history(window_size)
        return None
    
    def update_traffic_signals(self, traffic_signals: dict):
        """Update traffic signals reference (called after _build_traffic_signals).
        
        Args:
            traffic_signals: Updated dict of TrafficSignal objects
        """
        self.traffic_signals = traffic_signals


def build_adjacency_map_from_network(
    net_file: str, ts_ids: List[str]
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Build adjacency map with directional info from SUMO network file.
    
    Two controlled intersections are considered neighbors if they are
    connected directly or via non-controlled junctions.
    
    Args:
        net_file: Path to SUMO .net.xml file
        ts_ids: List of traffic signal IDs (controlled intersections)
        
    Returns:
        Tuple of:
        - adjacency_map: Dict mapping ts_id -> list of neighbor ts_ids
        - direction_map: Dict mapping ts_id -> {neighbor_id: direction_index}
          where direction_index is 0=North, 1=East, 2=South, 3=West
    """
    import math
    from collections import defaultdict
    import xml.etree.ElementTree as ET
    
    adjacency_map = {ts_id: [] for ts_id in ts_ids}
    direction_map = {ts_id: {} for ts_id in ts_ids}
    ts_set = set(ts_ids)
    
    if not net_file or not os.path.exists(net_file):
        print(f"[NeighborProvider] Warning: net_file not found, returning empty adjacency")
        return adjacency_map, direction_map
        
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Get junction coordinates for direction computation
        junction_coords = {}
        for junction in root.findall('junction'):
            junc_id = junction.get('id')
            x = float(junction.get('x', 0))
            y = float(junction.get('y', 0))
            junction_coords[junc_id] = (x, y)
        
        # Build graph of all junctions
        graph = defaultdict(set)
        
        for edge in root.findall('.//edge'):
            # Skip internal edges
            if edge.get('id', '').startswith(':'):
                continue
                
            from_junction = edge.get('from')
            to_junction = edge.get('to')
            
            if from_junction and to_junction:
                graph[from_junction].add(to_junction)
                graph[to_junction].add(from_junction)  # Undirected
        
        def get_direction_index(from_id: str, to_id: str) -> int:
            """Get direction index (0=N, 1=E, 2=S, 3=W) from source to target."""
            if from_id not in junction_coords or to_id not in junction_coords:
                return -1
            x1, y1 = junction_coords[from_id]
            x2, y2 = junction_coords[to_id]
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return -1
            angle = math.degrees(math.atan2(dy, dx)) % 360
            if 45 <= angle < 135:
                return 0  # North
            elif 135 <= angle < 225:
                return 3  # West
            elif 225 <= angle < 315:
                return 2  # South
            else:
                return 1  # East
        
        # For each controlled intersection, find neighbors
        def find_controlled_neighbors(start_ts: str) -> List[str]:
            """BFS to find controlled intersections reachable."""
            neighbors = []
            visited = {start_ts}
            queue = list(graph[start_ts])
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                if current in ts_set:
                    # Found a controlled intersection - it's a neighbor
                    neighbors.append(current)
                else:
                    # Non-controlled junction - continue searching
                    for next_junction in graph[current]:
                        if next_junction not in visited:
                            queue.append(next_junction)
            
            return neighbors
        
        # Build adjacency map with direction info
        for ts_id in ts_ids:
            neighbors = find_controlled_neighbors(ts_id)
            adjacency_map[ts_id] = neighbors
            for neighbor_id in neighbors:
                dir_idx = get_direction_index(ts_id, neighbor_id)
                direction_map[ts_id][neighbor_id] = dir_idx if dir_idx >= 0 else 0
        
    except Exception as e:
        print(f"[NeighborProvider] Error building adjacency: {e}")
        
    return adjacency_map, direction_map


class SumoSimulator(SimulatorAPI):
    """SUMO traffic simulator implementing SimulatorAPI interface.
    
    This class contains ALL SUMO-related logic:
    - SUMO connection management (start/stop/step)
    - Traffic signal creation and control
    - Observation and reward computation
    - State management
    
    Multi-worker support: Uses os.getpid() to create unique connection labels
    so multiple Ray workers can run SUMO instances simultaneously.
    
    Note: The environment (env.py) should NEVER import traci or sumolib directly.
    It only uses this class through SimulatorAPI methods.
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        label: str = "0",
        use_gui: bool = False,
        virtual_display: Optional[Tuple[int, int]] = None,
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = 100,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = 120,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        enforce_max_green: bool = False,
        reward_fn: Union[str, Dict] = "diff-waiting-time",
        reward_weights: Optional[List[float]] = None,
        observation_class = None,
        sumo_seed: Union[str, int] = "random",
        ts_ids: Optional[List[str]] = None,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        preprocessing_config: Optional[str] = None,
        window_size: int = 1,  # Added for Spatio-Temporal
        use_phase_standardizer: bool = False,  # Enable phase standardization
        use_neighbor_obs: bool = False,  # Enable neighbor observation for Local GNN
        max_neighbors: int = 4,  # Maximum neighbors (K) for neighbor observation
    ):
        """Initialize SUMO simulator with all parameters."""
        # Configuration
        self.net_file = net_file
        self.route_file = route_file
        self.label = label
        self.use_gui = use_gui
        self.virtual_display = virtual_display
        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        
        if observation_class is None:
            if window_size > 1:
                self.observation_class = SpatioTemporalObservationFunction
            else:
                self.observation_class = DefaultObservationFunction
        else:
            self.observation_class = observation_class
            
        self.sumo_seed = sumo_seed
        self.ts_ids = ts_ids
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd or ""
        self.preprocessing_config = preprocessing_config
        self.window_size = window_size
        self.use_phase_standardizer = use_phase_standardizer
        self.use_neighbor_obs = use_neighbor_obs
        self.max_neighbors = max_neighbors
        
        # State
        self.sumo = None
        self.conn = None
        self.disp = None
        self._started = False
        self._unique_label = None  # Store unique label for multi-worker
        self.traffic_signals = {}
        self.ts_lanes = {}  # Store controlled lanes per TS (ordered)
        self.gpi_standardizers = {}  # Store GPI modules per TS
        self.phase_standardizers = {}  # Store FRAP modules per TS
        self.vehicles = {}
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0
        self.neighbor_provider = None  # Will be created in initialize()
        self.adjacency_map = {}  # Map ts_id -> neighbor ts_ids
        self.direction_map = {}  # Map ts_id -> {neighbor_id: direction_index}

    # =====================================================================
    # SimulatorAPI Implementation
    # =====================================================================

    def initialize(self) -> Dict[str, Any]:
        """Initialize simulator and get initial observations.
        
        Steps:
        1. Start SUMO simulation (start once only)
        2. Get agent IDs (traffic light IDs)
        3. Build TrafficSignal objects for each agent
        4. Return initial observations for all agents
        """
        # Start simulation (handles both fresh start and reload)
        self._start_simulation()
        
        # Get agent IDs if not provided
        if self.ts_ids is None:
            self.ts_ids = list(self.conn.trafficlight.getIDList())
        
        # Build adjacency map for neighbor observation
        if self.use_neighbor_obs:
            self.adjacency_map, self.direction_map = build_adjacency_map_from_network(
                self.net_file, self.ts_ids
            )
            # Create NeighborProvider (traffic_signals will be updated later)
            self.neighbor_provider = NeighborProvider(
                traffic_signals={},
                adjacency_map=self.adjacency_map,
                direction_map=self.direction_map,
                max_neighbors=self.max_neighbors
            )
        
        # Build traffic signals
        self._build_traffic_signals(self.conn)
        
        # Update neighbor_provider with built traffic signals
        if self.neighbor_provider is not None:
            self.neighbor_provider.update_traffic_signals(self.traffic_signals)
        
        # Initialize vehicles dict and counters
        self.vehicles = {}
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0
        
        # Get initial observations
        initial_obs = {}
        for ts_id in self.ts_ids:
            if ts_id in self.traffic_signals:
                initial_obs[ts_id] = self.traffic_signals[ts_id].compute_observation()
        
        return initial_obs

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """Execute one simulation step.
        
        Steps:
        1. Apply actions to traffic signals
        2. Run SUMO simulation until next decision time
        3. Update detector history for all agents
        4. Collect observations, rewards, dones
        5. Return tuple of (obs, rewards, dones, info)
        """
        if self.sumo is None:
            raise RuntimeError("Simulator not initialized. Call initialize() or reset() first.")
        
        # Safety check: ensure we have traffic signals to control
        if not self.traffic_signals:
            raise RuntimeError(
                f"[SumoSimulator] No traffic signals found! "
                f"ts_ids={self.ts_ids}. "
                f"Check if preprocessing_config.json exists and contains valid E2 detector mappings."
            )
        
        # Apply actions for agents that are ready to act
        actions_applied = 0
        for ts_id, action in actions.items():
            if ts_id in self.traffic_signals and self.traffic_signals[ts_id].time_to_act:
                self.traffic_signals[ts_id].set_next_phase(action)
                actions_applied += 1
        
        # CRITICAL FIX: If an agent is ready to act but no action was provided,
        # apply a default action (equal phase distribution) to prevent infinite loop.
        # This can happen when RLlib is initializing or sampling.
        # NOTE: If fixed_ts is enabled, we DO NOT want to interfere with SUMO logic.
        if not self.fixed_ts:
            for ts_id in self.ts_ids:
                if ts_id in self.traffic_signals:
                    ts = self.traffic_signals[ts_id]
                    if ts.time_to_act and ts_id not in actions:
                        # Apply default action: equal distribution across 8 standard phases
                        # CRITICAL: set_next_phase() expects 8 standard phase ratios,
                        # NOT num_green_phases (actual phases). The FRAP pipeline
                        # handles mapping from standard → actual phases.
                        default_action = np.ones(ts.NUM_STANDARD_PHASES) / ts.NUM_STANDARD_PHASES
                        ts.set_next_phase(default_action)
                        actions_applied += 1
        else:
            # For fixed_ts (Baseline), we need to manually update the timing
            # so that the simulation steps forward by delta_time (e.g. 90s)
            # instead of 1s, ensuring consistent step counting with AI training.
            for ts_id in self.ts_ids:
                if ts_id in self.traffic_signals:
                    ts = self.traffic_signals[ts_id]
                    if ts.time_to_act:
                        ts.update_timing()
                        actions_applied += 1
        
        # Debug: Log if no actions were applied (potential issue indicator)
        # Disabled during normal training to reduce log noise
        # if actions_applied == 0:
        #     sim_time = self.get_sim_time()
        #     sample_ts = next(iter(self.traffic_signals.values()), None)
        #     if sample_ts:
        #         print(f"[DEBUG] No actions applied at sim_time={sim_time:.1f}. "
        #               f"Sample TS next_action_time={sample_ts.next_action_time:.1f}, "
        #               f"time_to_act={sample_ts.time_to_act}")
        
        # Run simulation until next decision time
        time_to_act = False
        loop_count = 0
        max_loop_iterations = 100000  # Safety limit: ~10000 seconds at 0.1s step
        start_sim_time = self.get_sim_time()
        
        # Wall-clock timeout to prevent infinite hang
        import time as _time
        _step_wall_start = _time.time()
        _wall_timeout = 300  # 300 seconds wall-clock timeout
        
        while not time_to_act:
            loop_count += 1
            
            # Safety check: prevent infinite loop
            if loop_count > max_loop_iterations:
                sim_time = self.get_sim_time()
                sample_ts = next(iter(self.traffic_signals.values()), None)
                if sample_ts:
                    raise RuntimeError(
                        f"[SumoSimulator] Infinite loop detected in step()! "
                        f"Loop count: {loop_count}, sim_time: {sim_time:.1f}, "
                        f"TS next_action_time: {sample_ts.next_action_time:.1f}, "
                        f"delta_time: {sample_ts.delta_time}, "
                        f"time_to_act: {sample_ts.time_to_act}"
                    )
                else:
                    raise RuntimeError(f"[SumoSimulator] Infinite loop detected in step()! No traffic signals found.")
            
            # Wall-clock timeout check
            if _time.time() - _step_wall_start > _wall_timeout:
                sim_time = self.get_sim_time()
                sample_ts = next(iter(self.traffic_signals.values()), None)
                raise RuntimeError(
                    f"[SumoSimulator] Wall-clock timeout ({_wall_timeout}s) in step()! "
                    f"Loop count: {loop_count}, sim_time: {sim_time:.1f}, "
                    f"TS next_action_time: {sample_ts.next_action_time if sample_ts else 'N/A'}"
                )
            
            # Step SUMO simulation
            self.sumo.simulationStep()
            
            # Update vehicle counters
            self.num_arrived_vehicles += self.sumo.simulation.getArrivedNumber()
            self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()
            self.num_teleported_vehicles += self.sumo.simulation.getEndingTeleportNumber()
            
            # Update detector history and departed vehicles for all traffic signals
            for ts_id in self.ts_ids:
                if ts_id in self.traffic_signals:
                    self.traffic_signals[ts_id].update_detectors_history()
                    self.traffic_signals[ts_id].update_departed_vehicles()
            
            # Check if any agent can act
            # CRITICAL FIX: Even with fixed_ts=True, we must wait for delta_time
            # before returning obs/rewards to match training step counting
            for ts_id in self.ts_ids:
                if ts_id in self.traffic_signals:
                    # Always check time_to_act (based on delta_time), regardless of fixed_ts
                    if self.traffic_signals[ts_id].time_to_act:
                        time_to_act = True
                        break
            
            # Check if simulation ended (prevent loop if no agents can ever act)
            if self.get_sim_step() >= self.sim_max_time:
                break
        
        # Collect observations and rewards for agents that acted
        observations = {}
        rewards = {}
        
        for ts_id in self.ts_ids:
            if ts_id in self.traffic_signals:
                if self.traffic_signals[ts_id].time_to_act or self.fixed_ts:
                    observations[ts_id] = self.traffic_signals[ts_id].compute_observation()
                    rewards[ts_id] = self.traffic_signals[ts_id].compute_reward()
        
        # Determine dones
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.get_sim_step() >= self.sim_max_time
        
        # Prepare info dict
        info = {
            "step": self.get_sim_step(),
            "num_arrived": self.num_arrived_vehicles,
            "num_departed": self.num_departed_vehicles,
            "num_teleported": self.num_teleported_vehicles,
        }
        
        return observations, rewards, dones, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reset simulator to initial state.
        
        Args:
            seed: Random seed for simulation
            options: Additional options
        
        Steps:
        1. Close current simulation if running
        2. Clear traffic signals and state
        3. Update seed if provided
        4. Call initialize() to start fresh
        """
        # Close current simulation if running
        if self._started:
            self._close_connection()
        
        # Reset state
        self.traffic_signals = {}
        self.vehicles = {}
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0
        
        # Update seed if provided
        if seed is not None:
            self.sumo_seed = seed
        
        # Initialize fresh simulation
        return self.initialize()

    def close(self):
        """Clean up and close simulator.
        
        Steps:
        1. Close SUMO connection
        2. Stop virtual display if running
        3. Clean up resources
        """
        self._close_connection()
        self.traffic_signals = {}
        self.vehicles = {}

    def get_agent_ids(self) -> List[str]:
        """Get list of all agent IDs (traffic signal IDs)."""
        return self.ts_ids or []

    def get_observation_space(self, agent_id: str):
        """Get observation space for agent."""
        if agent_id not in self.traffic_signals:
            return None
        return self.traffic_signals[agent_id].observation_space

    def get_action_space(self, agent_id: str):
        """Get action space for agent."""
        if agent_id not in self.traffic_signals:
            return None
        return self.traffic_signals[agent_id].action_space

    def get_sim_step(self) -> float:
        """Get current simulation time."""
        if self.sumo is None:
            return 0.0
        try:
            return float(self.sumo.simulation.getTime())
        except Exception:
            return 0.0

    @property
    def sim_step(self) -> float:
        """Property for current simulation step."""
        return self.get_sim_step()

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate counters for the current episode."""
        return {
            "num_arrived": self.num_arrived_vehicles,
            "num_departed": self.num_departed_vehicles,
            "num_teleported": self.num_teleported_vehicles,
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Return system-level information for the current state."""
        if self.sumo is None:
            return {}

        try:
            vehicles = list(self.sumo.vehicle.getIDList())
            speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
            waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
            num_backlogged = self.sumo.simulation.getPendingVehiclesNumber()

            return {
                "system_total_running": len(vehicles),
                "system_total_backlogged": num_backlogged,
                "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
                "system_total_arrived": self.num_arrived_vehicles,
                "system_total_departed": self.num_departed_vehicles,
                "system_total_teleported": self.num_teleported_vehicles,
                "system_total_waiting_time": float(sum(waiting_times)),
                "system_mean_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
                "system_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            }
        except Exception:
            return {}

    def get_per_agent_info(self) -> Dict[str, Any]:
        """Return per-agent (traffic signal) information."""
        if not self.traffic_signals or not self.ts_ids:
            return {}

        info: Dict[str, Any] = {}
        try:
            stopped: List[int] = []
            accumulated_waiting: List[float] = []
            average_speed: List[float] = []

            for ts in self.ts_ids:
                signal = self.traffic_signals.get(ts)
                if signal is None:
                    continue

                stopped.append(signal.get_total_queued())
                accumulated_waiting.append(sum(signal.get_accumulated_waiting_time_per_lane()))
                average_speed.append(signal.get_average_speed())

                info[f"{ts}_stopped"] = stopped[-1]
                info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting[-1]
                info[f"{ts}_average_speed"] = average_speed[-1]

            info["agents_total_stopped"] = sum(stopped)
            info["agents_total_accumulated_waiting_time"] = float(sum(accumulated_waiting))
        except Exception:
            return {}

        return info

    def get_rgb_array(self):  # pragma: no cover - depends on GUI availability
        """Return an RGB array representation of the current frame if available."""
        if not self.use_gui or self.disp is None:
            return None

        try:
            image = self.disp.grab()
            if image is None:
                return None
            return np.array(image)
        except Exception:
            return None

    def enable_debug_logging(self, enable: bool = True, level: int = 1, ts_ids: list = None):
        """Enable or disable debug logging for traffic signals.
        
        Args:
            enable: True to enable logging, False to disable
            level: Logging detail level (1=basic, 2=detailed, 3=verbose)
            ts_ids: Optional list of specific traffic signal IDs to enable logging for.
                   If None, enables for all traffic signals.
        
        Example:
            simulator.enable_debug_logging(True, level=2)  # Enable for all
            simulator.enable_debug_logging(True, level=1, ts_ids=['tl_1', 'tl_2'])  # Specific ones
        """
        if ts_ids is None:
            ts_ids = list(self.traffic_signals.keys())
        
        for ts_id in ts_ids:
            if ts_id in self.traffic_signals:
                self.traffic_signals[ts_id].enable_debug_logging(enable, level)
        
        if enable:
            print(f"[SumoSim] Debug logging enabled for {len(ts_ids)} traffic signals at level {level}")
        else:
            print(f"[SumoSim] Debug logging disabled for {len(ts_ids)} traffic signals")

    # =====================================================================
    # Internal SUMO Management (Private Methods)
    # =====================================================================

    def _binary(self, gui: bool) -> str:
        """Get SUMO binary path."""
        return sumolib.checkBinary("sumo-gui" if gui else "sumo")

    def _build_cmd(self, gui: bool = False, net_only: bool = False) -> List[str]:
        """Build SUMO command line."""
        binp = self._binary(gui)
        cmd = [binp, "-n", self.net_file]
        
        if not net_only:
            cmd.extend(["-r", self.route_file])
            # cmd.extend(["--max-depart-delay", str(self.max_depart_delay)])
            # cmd.extend(["--waiting-time-memory", str(self.waiting_time_memory)])
            cmd.extend(["--time-to-teleport", str(self.time_to_teleport)])
        
        if self.begin_time > 0:
            cmd.extend(["-b", str(self.begin_time)])
        
        if self.sumo_seed == "random":
            cmd.append("--random")
        elif self.sumo_seed is not None:
            cmd.extend(["--seed", str(self.sumo_seed)])
        
        if not self.sumo_warnings:
            cmd.append("--no-warnings")
        
        # NOTE: collision.action and collision.mingap-factor should be set via additional_sumo_cmd
        # to avoid duplicate option errors when user also specifies them
        
        if self.additional_sumo_cmd:
            if isinstance(self.additional_sumo_cmd, str):
                cmd.extend(self.additional_sumo_cmd.split())
            else:
                cmd.extend(self.additional_sumo_cmd)
        
        # Always add --no-step-log to suppress SUMO step logs
        cmd.append("--no-step-log")

        if gui:
            cmd.extend(["--start", "--quit-on-end"])
        
        return cmd

    def _start_simulation(self):
        """Start SUMO simulation (handles both fresh start and reload/reset)."""
        cmd = self._build_cmd(gui=self.use_gui, net_only=False)
        
        # Setup virtual display if needed
        if self.use_gui and self.virtual_display and not LIBSUMO:
            if self.disp is None:
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()

        if LIBSUMO:
            # Handle libsumo lifecycle: single instance per process
            if self._started:
                try:
                    # Reload simulation using load() - args start from index 1 (skip binary)
                    libsumo.load(cmd[1:])
                except Exception as e:
                    print(f"[SumoSim] Warning: libsumo.load() failed, trying start(): {e}")
                    pass
            else:
                libsumo.start(cmd)
                self._started = True
            
            self.conn = libsumo
            self.sumo = libsumo
            
        else:
            # Handle traci lifecycle (multi-process safe via labels)
            # Use PID for unique label to avoid conflicts between Ray workers
            label = f"sim_{self.label}_{os.getpid()}"
            
            if self._started and self.conn:
                try:
                    # Try to reload existing connection
                    traci.switch(self._unique_label)
                    traci.load(cmd[1:])
                except Exception:
                    # Connection lost, restart
                    try:
                        traci.close()
                    except:
                        pass
                    traci.start(cmd, label=label)
                    self._unique_label = label
                    self.conn = traci.getConnection(label)
                    self._started = True
            else:
                # Fresh start
                try:
                    traci.start(cmd, label=label)
                    self._unique_label = label
                    self.conn = traci.getConnection(label)
                    self._started = True
                except traci.FatalTraCIError:
                    # If label exists but connection lost (zombie), try getting it or force close
                    try:
                        traci.close() # Close any active default
                    except:
                        pass
                    # Retry
                    traci.start(cmd, label=label)
                    self._unique_label = label
                    self.conn = traci.getConnection(label)
                    self._started = True
            
            self.sumo = self.conn

        # Setup GUI display (common for both if GUI enabled)
        if self.use_gui:
            try:
                gui_interface = self.sumo.gui
                if "DEFAULT_VIEW" not in dir(gui_interface):
                    gui_interface.DEFAULT_VIEW = "View #0"
                gui_interface.setSchema(gui_interface.DEFAULT_VIEW, "real world")
            except Exception:
                pass

    def _close_connection(self):
        """Close SUMO connection and cleanup."""
        # For libsumo, we generally avoid closing in Ray workers to allow reload()
        if LIBSUMO:
            return
            
        if not self._started and self.conn is None:
            return
        
        try:
            if self.conn:
                label_to_use = getattr(self, '_unique_label', self.label)
                traci.switch(label_to_use)
                traci.close()
        finally:
            if self.disp is not None:
                try:
                    self.disp.stop()
                except Exception:
                    pass
                self.disp = None
            if not LIBSUMO:
                self.conn = None
                self.sumo = None
                self._started = False

    def _build_traffic_signals(self, conn):
        """Build TrafficSignal objects for each traffic light.
        
        Args:
            conn: SUMO TraCI connection (temporary or full)
        
        Steps:
        1. Get list of all traffic light IDs from SUMO
        2. For each traffic light:
           - Create TrafficSignal object with data_provider interface
           - Setup observation and action spaces
           - Initialize detector connections
        3. Store in self.traffic_signals dict
        """
        self.traffic_signals = {}
        self.ts_lanes = {}
        
        # Load preprocessing config FIRST to get ts_ids
        intersection_config = None
        if self.preprocessing_config and os.path.exists(self.preprocessing_config):
            try:
                with open(self.preprocessing_config, 'r') as f:
                    full_config = json.load(f)
                    intersection_config = full_config.get("intersections", {})
            except Exception as e:
                print(f"[SumoSim] Warning: Failed to load preprocessing config: {e}")
        
        # PRIORITIZE: Use ts_ids from intersection_config if available
        # This ensures we only use the signals explicitly defined by user
        if intersection_config:
            ts_ids = list(intersection_config.keys())
        else:
            # Fallback: Get all traffic light IDs from SUMO
            ts_ids = conn.trafficlight.getIDList()
        
        if not ts_ids:
            self.traffic_signals = {}
            return

        # OPTIMIZATION: Build lane -> detectors map ONCE to avoid O(N_TS * N_Det) TRACI calls
        # This is only used as fallback if intersection_config doesn't have detector info
        lane_to_detectors = {}
        use_config_detectors = False
        
        # Check if intersection_config has detector info
        if intersection_config:
            sample_ts = next(iter(intersection_config.values()), {})
            if "detectors_e2" in sample_ts:
                use_config_detectors = True
        
        if not use_config_detectors:
            # Fallback: Build mapping from SUMO API
            try:
                all_e2 = conn.lanearea.getIDList()
                for det_id in all_e2:
                    det_lane = conn.lanearea.getLaneID(det_id)
                    if det_lane not in lane_to_detectors:
                        lane_to_detectors[det_lane] = []
                    lane_to_detectors[det_lane].append(det_id)
                print(f"[SumoSim] Fallback: Mapped {len(all_e2)} detectors to {len(lane_to_detectors)} lanes via SUMO API")
            except Exception as e:
                print(f"[SumoSim] Error mapping detectors: {e}")
                return
        
        for ts_id in ts_ids:
            try:
                # Get number of green phases
                logic = conn.trafficlight.getAllProgramLogics(ts_id)[0]
                num_green_phases = len(logic.phases) // 2
                
                # Determine controlled lanes
                controlled_lanes = []
                
                # Strategy 1: Use JSON config if available (Preferred)
                if intersection_config and ts_id in intersection_config:
                    ts_config = intersection_config[ts_id]
                    # Flatten lanes from all directions in standard order (N, E, S, W)
                    # and standard lane order (Left to Right)
                    if "lanes_by_direction" in ts_config:
                        lanes_dict = ts_config["lanes_by_direction"]
                        for direction in ['N', 'E', 'S', 'W']:
                            if direction in lanes_dict:
                                # SUMO returns lanes Right to Left (0 is rightmost)
                                # GAT expects Left to Right (0 is leftmost)
                                # So we reverse the list
                                lanes = lanes_dict[direction]
                                ordered = lanes[::-1]
                                controlled_lanes.extend(ordered)
                                # print(f"[SumoSim] {ts_id} {direction} Lanes: {ordered} (Expect L->T->R)")
                
                # Strategy 2: Use GPI to Auto-Sort Lanes (New Fallback)
                # This ensures correct [N, E, S, W] and [Left, Through, Right] order
                # even without a config file.
                if not controlled_lanes and IntersectionStandardizer is not None:
                    try:
                        # Create GPI standardizer
                        gpi = IntersectionStandardizer(ts_id, data_provider=self)
                        # Get lanes grouped by direction {N: [l1, l2...], E: ...}
                        lanes_map = gpi.get_lanes_by_direction()
                        
                        for direction in ['N', 'E', 'S', 'W']:
                            if direction in lanes_map and lanes_map[direction]:
                                # SUMO returns lanes Right->Left (0 is rightmost)
                                # GAT expects Left->Right (0 is leftmost/median)
                                # So we reverse the list for each direction
                                lanes = lanes_map[direction]
                                controlled_lanes.extend(lanes[::-1])
                        
                        if controlled_lanes:
                            print(f"[SumoSim] Auto-sorted {len(controlled_lanes)} lanes for {ts_id} using GPI (N->E->S->W)")
                            pass
                            
                    except Exception as e:
                        print(f"[SumoSim] Warning: GPI auto-sort failed for {ts_id}: {e}")
                
                # Strategy 3: Use SUMO API (Raw Fallback)
                if not controlled_lanes:
                    controlled_lanes = conn.trafficlight.getControlledLanes(ts_id)
                
                # Store controlled lanes for this TS
                self.ts_lanes[ts_id] = controlled_lanes
                
                # Get detectors for this traffic signal
                e1_detectors = []
                e2_detectors = []
                
                if use_config_detectors and ts_id in intersection_config:
                    # NEW: Read detectors directly from intersection_config.json
                    ts_config = intersection_config[ts_id]
                    e1_detectors = ts_config.get("detectors_e1", [])
                    e2_detectors = ts_config.get("detectors_e2", [])
                else:
                    # Fallback: Use pre-built lane_to_detectors map
                    for lane in controlled_lanes:
                        if lane in lane_to_detectors:
                            e2_detectors.extend(lane_to_detectors[lane])
                    # Remove duplicates if any
                    e2_detectors = list(dict.fromkeys(e2_detectors))
                
                if not e2_detectors:
                    # Skip traffic signals without detectors (e.g., pedestrian crossings, internal junctions)
                    continue
                
                detectors = [e1_detectors, e2_detectors]  # [e1_detectors, e2_detectors]
                
                # Create PhaseStandardizer if enabled
                phase_std = None
                if self.use_phase_standardizer and PhaseStandardizer is not None:
                    try:
                        # Create GPI (IntersectionStandardizer) first if available
                        gpi = None
                        if IntersectionStandardizer is not None:
                            gpi = IntersectionStandardizer(ts_id, data_provider=self)
                            # Optimization: Load direction map from config to avoid TraCI calls
                            if ts_id in intersection_config and "direction_map" in intersection_config[ts_id]:
                                gpi.load_config(
                                    intersection_config[ts_id]["direction_map"],
                                    intersection_config[ts_id].get("observation_mask")
                                )
                            else:
                                gpi.map_intersection()
                            self.gpi_standardizers[ts_id] = gpi
                        
                        # Create FRAP (PhaseStandardizer)
                        phase_std = PhaseStandardizer(
                            junction_id=ts_id,
                            gpi_standardizer=gpi,
                            data_provider=self
                        )
                        
                        # Optimization: Load mapping from config if available to avoid TraCI calls
                        if ts_id in intersection_config and "phase_config" in intersection_config[ts_id]:
                            phase_std.load_config(intersection_config[ts_id]["phase_config"])
                        else:
                            # Fallback: Configure using TraCI
                            pass
                            
                        self.phase_standardizers[ts_id] = phase_std
                    except Exception as e:
                        print(f"[SumoSim] Warning: Failed to create PhaseStandardizer for {ts_id}: {e}")
                        phase_std = None
                
                # Determine observation class based on configuration
                obs_class = self.observation_class
                if self.use_neighbor_obs and NeighborTemporalObservationFunction is not None:
                    obs_class = NeighborTemporalObservationFunction
                
                # Create TrafficSignal object with data_provider (self)
                ts = TrafficSignal(
                    ts_id=ts_id,
                    delta_time=self.delta_time,
                    yellow_time=self.yellow_time,
                    min_green=self.min_green,
                    max_green=self.max_green,
                    enforce_max_green=self.enforce_max_green,
                    begin_time=self.begin_time,
                    reward_fn=self.reward_fn.get(ts_id, "diff-waiting-time") if isinstance(self.reward_fn, dict) else self.reward_fn,
                    reward_weights=self.reward_weights,
                    data_provider=self,  # Pass self as data provider
                    num_green_phases=num_green_phases,
                    observation_class=obs_class,
                    detectors=detectors,
                    window_size=self.window_size,
                    phase_standardizer=phase_std,
                    use_phase_standardizer=self.use_phase_standardizer,
                    detectors_e2_length=ts_config.get("e2_detector_lengths") if ts_id in intersection_config else None,
                    neighbor_provider=self.neighbor_provider,  # Pass neighbor provider for Local GNN
                    max_neighbors=self.max_neighbors,
                )
                
                self.traffic_signals[ts_id] = ts
                
            except Exception as e:
                # Log only unexpected errors, not detector-related skips
                if "detector" not in str(e).lower():
                    print(f"Warning: Failed to create TrafficSignal for {ts_id}: {e}")
                continue
        
        # Summary log (clean, single-line output)
        skipped_count = len(ts_ids) - len(self.traffic_signals)
        if skipped_count > 0:
            print(f"[SumoSim] Created {len(self.traffic_signals)} traffic signals ({skipped_count} skipped - no E2 detectors)")
        else:
            print(f"[SumoSim] Created {len(self.traffic_signals)} traffic signals")

    # =====================================================================
    # Data Provider Interface Implementation
    # These methods provide traffic data to TrafficSignal agents
    # =====================================================================

    def get_sim_time(self) -> float:
        """Return current simulation time in seconds."""
        if self.sumo is None:
            return 0.0
        try:
            return float(self.sumo.simulation.getTime())
        except Exception:
            return 0.0

    def should_act(self, ts_id: str, next_action_time: float) -> bool:
        """Check if traffic signal should act at current time."""
        return self.get_sim_time() >= next_action_time

    def set_traffic_light_phase(self, ts_id: str, green_times: List[float]):
        """Set traffic light phase durations and synchronize to cycle start.
        
        IMPORTANT: After setting new phase durations, we reset the traffic light
        to phase 0 to ensure the new timing takes effect immediately and the
        total cycle time remains consistent (e.g., 90s).
        
        Without resetting to phase 0, if we change durations mid-cycle, the
        actual observed cycle time will vary because:
        - Phases that already completed keep their old duration
        - Only future phases use the new duration
        
        Args:
            ts_id: Traffic signal ID
            green_times: List of green phase durations (one per green phase)
        """
        try:
            logic = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0]
            num_phases = len(green_times)
            for i in range(num_phases):
                logic.phases[2 * i].duration = green_times[i]
            self.sumo.trafficlight.setProgramLogic(ts_id, logic)
            
            # CRITICAL: Reset to phase 0 to start a fresh cycle with new timings
            # This ensures the total cycle time = sum(green_times) + sum(yellow_times)
            # Without this, cycle time varies depending on when action is applied
            self.sumo.trafficlight.setPhase(ts_id, 0)
            
        except Exception as e:
            print(f"Warning: Failed to set traffic light phase for {ts_id}: {e}")

    def get_controlled_lanes(self, ts_id: str) -> List[str]:
        """Get list of lanes controlled by traffic signal."""
        if ts_id in self.ts_lanes:
            return self.ts_lanes[ts_id]
        try:
            return list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(ts_id)))
        except Exception:
            return []

    # Detector methods
    def get_detector_length(self, detector_id: str) -> float:
        """Get detector length in meters."""
        try:
            return self.sumo.lanearea.getLength(detector_id)
        except Exception:
            return 0.0

    def get_detector_vehicle_count(self, detector_id: str) -> int:
        """Get number of vehicles in detector."""
        try:
            return self.sumo.lanearea.getLastIntervalVehicleNumber(detector_id)
        except Exception:
            return 0

    def get_detector_vehicle_ids(self, detector_id: str) -> List[str]:
        """Get IDs of vehicles currently present in detector.
        
        Uses getLastStepVehicleIDs (current step) instead of 
        getLastIntervalVehicleIDs (last completed aggregation period).
        The interval-based method returns empty when no period has completed,
        causing diff-departed-veh reward to always be zero.
        """
        try:
            return self.sumo.lanearea.getLastStepVehicleIDs(detector_id)
        except Exception:
            return []

    def get_detector_jam_length(self, detector_id: str) -> float:
        """Get jam length in detector (meters)."""
        try:
            return self.sumo.lanearea.getJamLengthMeters(detector_id)
        except Exception:
            return 0.0

    def get_detector_halting_number(self, detector_id: str) -> int:
        """Get number of halting vehicles in E2 detector."""
        try:
            return self.sumo.lanearea.getLastStepHaltingNumber(detector_id)
        except Exception:
            return 0

    def get_detector_occupancy(self, detector_id: str) -> float:
        """Get detector occupancy percentage."""
        try:
            return self.sumo.lanearea.getLastIntervalOccupancy(detector_id)
        except Exception:
            return 0.0

    def get_detector_mean_speed(self, detector_id: str) -> float:
        """Get mean speed in detector (m/s)."""
        try:
            return self.sumo.lanearea.getLastIntervalMeanSpeed(detector_id)
        except Exception:
            return 0.0

    def get_detector_lane_id(self, detector_id: str) -> str:
        """Get lane ID associated with detector."""
        try:
            return self.sumo.lanearea.getLaneID(detector_id)
        except Exception:
            return ""

    # Lane methods
    def get_lane_vehicles(self, lane_id: str) -> List[str]:
        """Get list of vehicle IDs in lane."""
        try:
            return self.sumo.lane.getLastStepVehicleIDs(lane_id)
        except Exception:
            return []

    def get_lane_halting_number(self, lane_id: str) -> int:
        """Get number of halting vehicles in lane."""
        try:
            return self.sumo.lane.getLastStepHaltingNumber(lane_id)
        except Exception:
            return 0

    def get_lane_max_speed(self, lane_id: str) -> float:
        """Get maximum allowed speed in lane (m/s)."""
        try:
            return self.sumo.lane.getMaxSpeed(lane_id)
        except Exception:
            return 50.0  # Default speed

    # Vehicle methods
    def get_vehicle_length(self, vehicle_id: str) -> float:
        """Get vehicle length in meters."""
        try:
            return self.sumo.vehicle.getLength(vehicle_id)
        except Exception:
            return 5.0  # Default vehicle length

    def get_vehicle_speed(self, vehicle_id: str) -> float:
        """Get vehicle speed (m/s)."""
        try:
            return self.sumo.vehicle.getSpeed(vehicle_id)
        except Exception:
            return 0.0

    def get_vehicle_allowed_speed(self, vehicle_id: str) -> float:
        """Get vehicle's allowed speed (m/s)."""
        try:
            return self.sumo.vehicle.getAllowedSpeed(vehicle_id)
        except Exception:
            return 50.0  # Default speed

    def get_vehicle_waiting_time(self, vehicle_id: str, lane_id: str) -> float:
        """Get vehicle accumulated waiting time in lane."""
        try:
            # Get accumulated waiting time
            acc = self.sumo.vehicle.getAccumulatedWaitingTime(vehicle_id)
            
            # Track per-lane waiting time
            if not hasattr(self, 'vehicle_waiting_times'):
                self.vehicle_waiting_times = {}
            
            veh_lane = self.sumo.vehicle.getLaneID(vehicle_id)
            if vehicle_id not in self.vehicle_waiting_times:
                self.vehicle_waiting_times[vehicle_id] = {veh_lane: acc}
            else:
                self.vehicle_waiting_times[vehicle_id][veh_lane] = acc - sum(
                    [self.vehicle_waiting_times[vehicle_id][l] 
                     for l in self.vehicle_waiting_times[vehicle_id].keys() if l != veh_lane]
                )
            
            return self.vehicle_waiting_times[vehicle_id].get(lane_id, 0.0)
        except Exception:
            return 0.0

    # =====================================================================
    # Data Provider Methods for Preprocessing Modules (GPI, FRAP)
    # =====================================================================

    def get_incoming_edges(self, junction_id: str) -> List[str]:
        """Get list of incoming edges for a junction (for GPI module)."""
        try:
            controlled_lanes = self.sumo.trafficlight.getControlledLanes(junction_id)
            edges = set()
            for lane in controlled_lanes:
                edge = lane.rsplit('_', 1)[0]
                edges.add(edge)
            return list(edges)
        except Exception:
            return []

    def get_lane_shape(self, lane_id: str) -> List[Tuple[float, float]]:
        """Get lane shape as list of coordinate points (for GPI module)."""
        try:
            return list(self.sumo.lane.getShape(lane_id))
        except Exception:
            return []

    def get_edge_lanes(self, edge_id: str) -> List[str]:
        """Get list of lanes for an edge (for GPI module)."""
        try:
            return list(self.sumo.edge.getLaneIDs(edge_id))
        except Exception:
            return []

    def get_traffic_light_program(self, ts_id: str):
        """Get traffic light program logic (for FRAP module)."""
        try:
            return self.sumo.trafficlight.getAllProgramLogics(ts_id)[0]
        except Exception:
            return None

    def get_controlled_links(self, ts_id: str) -> List:
        """Get controlled links/movements for traffic signal (for FRAP module)."""
        try:
            return self.sumo.trafficlight.getControlledLinks(ts_id)
        except Exception:
            return []

    def get_teleport_count_this_step(self) -> int:
        """Get number of vehicles that completed teleport in the current simulation step.
        
        This is useful for penalizing teleportation in rewards, as teleportation
        indicates severe congestion where vehicles are stuck for too long.
        
        Returns:
            int: Number of vehicles that teleported in this step
        """
        try:
            return self.sumo.simulation.getEndingTeleportNumber()
        except Exception:
            return 0
    
    def get_total_teleport_count(self) -> int:
        """Get total number of teleported vehicles since simulation start.
        
        Returns:
            int: Total number of teleported vehicles
        """
        return self.num_teleported_vehicles
