import copy
"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation.

This class is now independent of SUMO/traci - it only handles RL logic (observation, reward, action).
All simulation-specific data is provided through a data provider interface.
"""

from typing import Callable, List, Union, Dict, Any, Optional
import logging

import numpy as np
from gymnasium import spaces

# Import preprocessing modules (optional)
try:
    from preprocessing.frap import PhaseStandardizer
except ImportError:
    PhaseStandardizer = None

# Configure debug logger with explicit stdout handler
import sys

logger = logging.getLogger("TrafficSignal")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent root logger from blocking DEBUG messages

# Add handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout for Ray workers
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: 
    Currently it is not supporting all-red phases (but should be easy to implement it). It's meant that the agent decides only the green phase durations, the yellow time and all-red phases are fixed.

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [ lane_1_density, lane_2_density, ..., lane_n_density,
            lane_1_queue, lane_2_queue, ..., lane_n_queue, 
            lane_1_occupancy, lane_2_occupancy, ..., lane_n_occupancy,
            lane_1_average_speed, lane_2_average_speed, ..., lane_n_average_speed ]

    where:
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
    - ```lane_i_occupancy``` is the sum of lengths of vehicles in incoming lane i divided by the length of the lane
    - ```lane_i_average_speed``` is the average speed of vehicles in incoming lane i divided

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is a continuous vector of size equal to the number of green phases 
    of the traffic signal. Each element of the vector represents the proportion of 
    green time allocated to the corresponding green phase in the next cycle. The 
    actual green time for each phase is computed by normalizing the action vector 
    and scaling it to fit

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        data_provider: Any,  # Interface to get traffic data (replaces direct SUMO access)
        num_green_phases: int,
        observation_class: type,
        detectors: List = None,
        window_size: int = 1,  # Added for Spatio-Temporal
        phase_standardizer: Optional["PhaseStandardizer"] = None,  # FRAP module for phase standardization
        use_phase_standardizer: bool = False,  # Flag to enable/disable phase standardization
        detectors_e2_length: Optional[Dict[str, float]] = None,  # Pre-computed detector lengths
        neighbor_provider=None,  # NeighborProvider for Local GNN observation
        max_neighbors: int = 4,  # Max neighbors for NeighborTemporalObservationFunction
    ):
        """Initializes a TrafficSignal object.

        Args:
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            data_provider: Object that provides traffic data (replaces direct SUMO/traci access).
            num_green_phases (int): Number of green phases for this traffic signal.
            observation_class: Class for computing observations.
            detectors (List): List of detector IDs [e1_detectors, e2_detectors].
            window_size (int): Size of observation history window.
        """
        self.id = ts_id
        self.data_provider = data_provider  # Replaces self.sumo
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.num_green_phases = num_green_phases
        self.window_size = window_size
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.detectors = detectors if detectors else [[], []]
        self.avg_veh_length = 3.0
        self.sampling_interval_s = 10
        self.aggregation_interval_s = delta_time
        
        # Debug logging configuration
        self.debug_logging = False  # Set to True to enable detailed logging
        self.debug_log_level = 1    # 1=basic, 2=detailed, 3=verbose

        # Calculate total green and yellow time in a cycle
        self.total_yellow_time = self.yellow_time * self.num_green_phases
        self.total_green_time = self.delta_time - self.total_yellow_time

        if type(self.reward_fn) is list:
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [self._get_reward_fn_from_string(reward_fn) for reward_fn in self.reward_fn]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        # Create observation function
        # Check if observation class needs neighbor_provider (like NeighborTemporalObservationFunction)
        self.neighbor_provider = neighbor_provider
        self.max_neighbors = max_neighbors
        
        import inspect
        obs_class_params = inspect.signature(observation_class.__init__).parameters
        if 'neighbor_provider' in obs_class_params:
            self.observation_fn = observation_class(
                self, 
                neighbor_provider=neighbor_provider,
                max_neighbors=max_neighbors,
                window_size=window_size
            )
        else:
            self.observation_fn = observation_class(self)

        # Get lanes and detectors from data provider
        self.lanes = self.data_provider.get_controlled_lanes(self.id)
        
        self.detectors_e1 = self.detectors[0]
        self.detectors_e2 = self.detectors[1]
        
        # Use provided lengths or fetch via data provider (TraCI)
        if detectors_e2_length:
            self.detectors_e2_length = detectors_e2_length
        else:
            self.detectors_e2_length = {
                e2: self.data_provider.get_detector_length(e2) for e2 in self.detectors_e2
            }

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Box(
            low=np.array([(self.min_green / self.total_green_time) for _ in range(self.num_green_phases)], dtype=np.float32), 
            high=np.array([1.0 for _ in range(self.num_green_phases)], dtype=np.float32), 
            dtype=np.float32
        )
        assert (self.min_green * self.num_green_phases) <= self.total_green_time, (
            "Minimum green time too high for traffic signal " + self.id + " cycle time"
        )
        
        # Initialize detector history
        self.detector_history = {
            "density": {det_id: [] for det_id in self.detectors_e2},
            "queue": {det_id: [] for det_id in self.detectors_e2},
            "occupancy": {det_id: [] for det_id in self.detectors_e2},
            "average_speed": {det_id: [] for det_id in self.detectors_e2},
        }
        self._last_sampling_time = -self.sampling_interval_s
        
        # History of observations for Spatio-Temporal model
        self.observation_history = []
        self.max_history_size = 50  # Keep enough history

        # Tracking for halt_veh and diff_departed_veh rewards
        self.max_veh = 0
        self._compute_max_veh()  # Calculate max_veh based on detectors
        self.initial_vehicles_this_cycle = 0
        self.departed_vehicles_this_cycle = 0
        self.halting_vehicles_samples = []
        
        # Track unique vehicle IDs for accurate departed vehicle counting
        self._vehicles_at_cycle_start = set()  # Vehicles present at start of cycle
        self._vehicles_seen_this_cycle = set()  # All vehicles seen during the cycle
        
        # Reward metrics history - aggregated over cycle (6 samples with 10s interval = 60s)
        self.reward_metrics_history = {
            "halting_vehicles": [],    # Track halting vehicles per sample
            "total_queued": [],        # Track queued vehicles per sample
            "average_speed": [],       # Track average speed per sample
            "waiting_time": [],        # Track total waiting time per sample
        }
        
        # Action tracking for debugging policy behavior
        self.action_history = []  # Store all actions in episode
        self.action_count = 0     # Number of actions in episode

        # Phase standardization (FRAP module)
        self.phase_standardizer = phase_standardizer
        self.use_phase_standardizer = use_phase_standardizer and phase_standardizer is not None
        if self.use_phase_standardizer and self.phase_standardizer is not None:
            # Configure phase standardizer if not already configured
            if hasattr(self.phase_standardizer, 'configure') and not self.phase_standardizer._configured:
                self.phase_standardizer.configure()

        
    def _get_reward_fn_from_string(self, reward_fn):
        if type(reward_fn) is str:
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        return reward_fn

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.data_provider.should_act(self.id, self.next_action_time)

    def update(self):
        """Updates the traffic signal state. (No-op since simulator handles the cycle)."""
        pass

    def update_detectors_history(self):
        """
        Cập nhật lịch sử dữ liệu từ các detector mỗi sampling_interval_s (10s).
        
        1. Mỗi 10s, thu thập một mẫu dữ liệu từ tất cả detectors
        2. Lưu mẫu vào lịch sử, giữ lại tối đa max_samples mẫu
        """
        current_time = self.data_provider.get_sim_time()
        
        # Kiểm tra xem có cần cập nhật lịch sử không (mỗi sampling_interval_s = 10s)
        if current_time - self._last_sampling_time >= self.sampling_interval_s - 0.1:
            self._last_sampling_time = current_time
            
            # Số lượng mẫu tối đa để giữ trong cửa sổ delta_time
            max_samples = max(1, int(self.aggregation_interval_s / self.sampling_interval_s))
            
            # Define metrics and their compute functions for detectors
            detector_metrics = [
                ("density", self._compute_detector_density),
                ("queue", self._compute_detector_queue),
                ("occupancy", self._compute_detector_occupancy),
                ("average_speed", self._compute_detector_average_speed),
            ]
            
            # Update detector history for each metric
            for det_id in self.detectors_e2:
                for metric_name, compute_fn in detector_metrics:
                    value = compute_fn(det_id)
                    self._update_history_buffer(
                        self.detector_history[metric_name][det_id], value, max_samples
                    )
            
            # Update reward metrics history
            reward_metrics = [
                ("halting_vehicles", self._get_total_halting_veh_instant),
                ("total_queued", self._get_total_queued_instant),
                ("average_speed", self._get_average_speed_instant),
                ("waiting_time", self._get_waiting_time_from_detectors),
            ]
            
            for metric_name, compute_fn in reward_metrics:
                value = compute_fn()
                self._update_history_buffer(
                    self.reward_metrics_history[metric_name], value, max_samples
                )
    
    def _update_history_buffer(self, buffer: list, value: float, max_size: int):
        """Append value to buffer and trim to max_size."""
        buffer.append(value)
        if len(buffer) > max_size:
            del buffer[:-max_size]
    
    def _compute_detector_density(self, det_id: str) -> float:
        """Tính mật độ giao thông chuẩn hóa [0,1] cho một detector."""
        try:
            vehicle_count = self.data_provider.get_detector_vehicle_count(det_id)
            
            if vehicle_count == 0:
                return 0.0
            
            detector_length_meters = self.data_provider.get_detector_length(det_id)
            if detector_length_meters <= 0:
                return 0.0
            
            vehicle_ids = self.data_provider.get_detector_vehicle_ids(det_id)
            
            if len(vehicle_ids) > 0:
                total_length = sum(self.data_provider.get_vehicle_length(veh_id) for veh_id in vehicle_ids)
                avg_vehicle_length = total_length / len(vehicle_ids)
            else:
                avg_vehicle_length = 5.0
            
            max_vehicle_capacity = detector_length_meters / (self.MIN_GAP + avg_vehicle_length)
            density = vehicle_count / max_vehicle_capacity
            
            return min(1.0, density)
        except Exception:
            return 0.0
    
    def _compute_detector_queue(self, det_id: str) -> float:
        """Tính độ dài hàng đợi chuẩn hóa [0,1] cho một detector."""
        try:
            queue_length_meters = self.data_provider.get_detector_jam_length(det_id)
            
            if queue_length_meters == 0:
                return 0.0
            
            detector_length_meters = self.data_provider.get_detector_length(det_id)
            if detector_length_meters <= 0:
                return 0.0
            
            normalized_queue = queue_length_meters / detector_length_meters
            return min(1.0, normalized_queue)
        except Exception:
            return 0.0
    
    def _compute_detector_occupancy(self, det_id: str) -> float:
        """Tính độ chiếm dụng chuẩn hóa [0,1] cho một detector."""
        try:
            occupancy = self.data_provider.get_detector_occupancy(det_id)
            normalized_occupancy = occupancy / 100.0
            return min(1.0, max(0.0, normalized_occupancy))
        except Exception:
            return 0.0
    
    def _compute_detector_average_speed(self, det_id: str) -> float:
        """Tính tốc độ trung bình chuẩn hóa [0,1] cho một detector."""
        try:
            mean_speed = self.data_provider.get_detector_mean_speed(det_id)
            
            if mean_speed <= 0:
                return 0.0
            
            lane_id = self.data_provider.get_detector_lane_id(det_id)
            max_speed = self.data_provider.get_lane_max_speed(lane_id)
            
            if max_speed > 0:
                normalized_speed = mean_speed / max_speed
                return min(1.0, normalized_speed)
            else:
                return 1.0
        except Exception:
            return 1.0

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase.

        Args:
            new_phase (int): Action representing green time ratios for each phase.
                            This is the STANDARD action from the model.
                            If use_phase_standardizer is True, it will be translated
                            to actual phase order before applying.
        """
        # Translate standard action to actual phase order if phase standardizer is enabled
        if self.use_phase_standardizer and self.phase_standardizer is not None:
            try:
                action_array = np.array(new_phase) if not isinstance(new_phase, np.ndarray) else new_phase
                actual_action = self.phase_standardizer.standardize_action(action_array)
                if self.debug_logging:
                    print(f"[PhaseStd] {self.id}: Standard action {action_array} -> Actual action {actual_action}")
            except Exception as e:
                if self.debug_logging:
                    print(f"[PhaseStd] Warning: Failed to standardize action for {self.id}: {e}")
                actual_action = new_phase
        else:
            actual_action = new_phase
        
        self.green_times = self._get_green_time_from_ratio(actual_action)

        # Delegate phase setting to data provider (simulator)
        self.data_provider.set_traffic_light_phase(self.id, self.green_times)

        # Set the next action time
        current_time = self.data_provider.get_sim_time()
        self.next_action_time = current_time + self.delta_time
        
        # Update vehicle tracking for diff_departed_veh reward at start of new cycle
        self.update_cycle_vehicle_tracking()
        
        # Track action for distribution analysis
        if self.debug_logging:
            action_array = np.array(new_phase) if not isinstance(new_phase, np.ndarray) else new_phase
            self.action_history.append(action_array.copy())
            self.action_count += 1
        
        # Debug logging for action
        if self.debug_logging:
            self._log_action_debug(new_phase, current_time)

    def _get_green_time_from_ratio(self, green_time_set: np.ndarray):
        """
        Computes the green time for each phase based on the provided green time rates.
        Args:
            green_time_set (np.ndarray): An array of green time rates for each phase.
        Returns:
            List[int]: A list of green times for each phase.
        """
        green_time_set = np.array(green_time_set, dtype=float, copy=True)
        if np.sum(green_time_set) == 0:
            green_time_set = np.ones(self.num_green_phases)
        green_time_set /= np.sum(green_time_set)
        green_times = green_time_set * self.total_green_time
        
        return green_times

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        if hasattr(self.observation_fn, "compute_current_observation"):
            current_obs = self.observation_fn.compute_current_observation()
            
            # Store observation in history
            self.observation_history.append(current_obs)
            if len(self.observation_history) > self.max_history_size:
                self.observation_history.pop(0)
                
            # Return the stacked observation (history)
            obs = self.observation_fn()
        else:
            obs = self.observation_fn()
            
            # Store observation in history
            self.observation_history.append(obs)
            if len(self.observation_history) > self.max_history_size:
                self.observation_history.pop(0)

        return self._validate_and_clip_observation(obs)

    def _validate_and_clip_observation(self, obs: Any) -> Any:
        """Validate and clip observation to observation_space bounds and ensure float32.
        
        Handles both Box (numpy array) and Dict (dictionary of arrays) observation spaces.
        """
        # Case 1: Observation is a dictionary (common for GNN/Complex models)
        if isinstance(obs, dict):
            clipped_obs = {}
            for key, value in obs.items():
                # Recursively validate items, passing appropriate subspace if possible
                # For simplicity, we just validate the value type here
                # A full implementation would traverse self.observation_space.spaces[key]
                clipped_obs[key] = self._validate_and_clip_value(value, key)
            return clipped_obs
            
        # Case 2: Observation is a list/tuple/array (Standard Box space)
        return self._validate_and_clip_value(obs)

    def _validate_and_clip_value(self, value: Any, key: str = None) -> np.ndarray:
        """Helper to validate a single value against observation space."""
        # Convert to numpy float32
        try:
            arr = np.asarray(value, dtype=np.float32)
        except (ValueError, TypeError):
            # If conversion fails (e.g. non-numeric), return as is
            return value
            
        # Identify the relevant space for clipping
        target_space = self.observation_space
        
        # If we are inside a Dict space, try to find the subspace
        if key is not None and isinstance(self.observation_space, spaces.Dict):
            if key in self.observation_space.spaces:
                target_space = self.observation_space.spaces[key]
            else:
                target_space = None # Key not in space, cannot clip
                
        # If we have a Box space (or subspace), apply clipping
        if target_space is not None and hasattr(target_space, 'low') and hasattr(target_space, 'high'):
            try:
                # Handle shapes matching
                low = np.asarray(target_space.low, dtype=np.float32)
                high = np.asarray(target_space.high, dtype=np.float32)
                
                # Only clip if shapes are compatible
                if arr.shape == low.shape:
                    msg_pre = f"[ObsClip] key='{key}'" if key else "[ObsClip]"
                    
                    # Optional: warning if out of bounds (can be noisy)
                    # if np.any(arr < low) or np.any(arr > high):
                    #    if self.debug_logging:
                    #        print(f"{msg_pre} Clipping value range [{arr.min()}, {arr.max()}] to [{low.min()}, {high.max()}]")
                            
                    arr = np.clip(arr, low, high)
            except Exception:
                # If shapes mismatch or other error, skip clipping
                pass
                
        return arr

    def compute_reward(self) -> Union[float, np.ndarray]:
        """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        reward_components = {}  # Store individual rewards for logging
        
        if self.reward_dim == 1:
            self.last_reward = self.reward_list[0](self)
            if self.debug_logging:
                reward_components[self._get_reward_fn_name(self.reward_list[0])] = self.last_reward
        else:
            rewards_array = []
            for i, reward_fn in enumerate(self.reward_list):
                reward_val = reward_fn(self)
                rewards_array.append(reward_val)
                if self.debug_logging:
                    reward_components[self._get_reward_fn_name(reward_fn)] = reward_val
            
            self.last_reward = np.array(rewards_array, dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = np.dot(self.last_reward, self.reward_weights)  # Linear combination of rewards
                if self.debug_logging:
                    reward_components["weighted_sum"] = self.last_reward

        # Debug logging for reward
        if self.debug_logging:
            self._log_reward_debug(reward_components)

        return self.last_reward
    
    def _get_reward_fn_name(self, reward_fn) -> str:
        """Get the name of a reward function for logging."""
        # Check if it's a registered reward function
        for name, fn in TrafficSignal.reward_fns.items():
            if fn == reward_fn:
                return name
        # Fallback to function name
        return getattr(reward_fn, '__name__', str(reward_fn))
    
    def _log_action_debug(self, action, current_time):
        """Log detailed action information for debugging."""
        log_msg = f"\n{'='*60}\n"
        log_msg += f"[ACTION DEBUG] Traffic Signal: {self.id}\n"
        log_msg += f"{'='*60}\n"
        log_msg += f"  Simulation Time: {current_time:.1f}s\n"
        log_msg += f"  Next Action Time: {self.next_action_time:.1f}s\n"
        log_msg += f"  Delta Time: {self.delta_time}s\n"
        log_msg += f"  ---\n"
        log_msg += f"  Raw Action (ratio): {action}\n"
        log_msg += f"  Computed Green Times (seconds):\n"
        for i, gt in enumerate(self.green_times):
            log_msg += f"    Phase {i}: {gt:.2f}s ({gt/self.total_green_time*100:.1f}%)\n"
        log_msg += f"  Total Green Time: {self.total_green_time}s\n"
        log_msg += f"  Total Yellow Time: {self.total_yellow_time}s\n"
        
        if self.debug_log_level >= 2:
            log_msg += f"  ---\n"
            log_msg += f"  Initial Vehicles This Cycle: {self.initial_vehicles_this_cycle}\n"
            log_msg += f"  Departed Vehicles: {self.departed_vehicles_this_cycle}\n"
            log_msg += f"  Max Vehicles Capacity: {self.max_veh:.2f}\n"
            
            # Episode action statistics
            if self.action_count > 1:
                stats = self.get_action_stats()
                log_msg += f"  ---\n"
                log_msg += f"  Episode Action Stats ({stats['count']} actions so far):\n"
                log_msg += f"    Mean per phase: {np.array2string(stats['mean'], precision=3)}\n"
                log_msg += f"    Std per phase:  {np.array2string(stats['std'], precision=3)}\n"
                log_msg += f"    Range: [{np.array2string(stats['min'], precision=3)} - {np.array2string(stats['max'], precision=3)}]\n"
        
        log_msg += f"{'='*60}\n"
        print(log_msg, flush=True)  # Use print for Ray worker compatibility
    
    def _log_reward_debug(self, reward_components: Dict[str, float]):
        """Log detailed reward information for debugging."""
        current_time = self.data_provider.get_sim_time()
        
        log_msg = f"\n{'='*60}\n"
        log_msg += f"[REWARD DEBUG] Traffic Signal: {self.id}\n"
        log_msg += f"{'='*60}\n"
        log_msg += f"  Simulation Time: {current_time:.1f}s\n"
        log_msg += f"  ---\n"
        log_msg += f"  Reward Components:\n"
        
        for name, value in reward_components.items():
            if isinstance(value, (float, int)):
                log_msg += f"    {name}: {value:.4f}\n"
            else:
                log_msg += f"    {name}: {value}\n"
        
        log_msg += f"  ---\n"
        log_msg += f"  Final Reward: {self.last_reward}\n"
        
        if self.debug_log_level >= 2:
            log_msg += f"  ---\n"
            log_msg += f"  Traffic State Info:\n"
            log_msg += f"    Total Queued Vehicles: {self.get_total_queued()}\n"
            log_msg += f"    Total Halting Vehicles: {self.get_total_halting_veh_by_detectors()}\n"
            log_msg += f"    Average Speed: {self.get_average_speed():.4f}\n"
            log_msg += f"    Last Waiting Time: {self.last_ts_waiting_time:.2f}s\n"
        
        if self.debug_log_level >= 3:
            log_msg += f"  ---\n"
            log_msg += f"  Detector History (last sample):\n"
            for det_id in self.detectors_e2[:3]:  # Show first 3 detectors only
                density_hist = self.detector_history["density"].get(det_id, [])
                queue_hist = self.detector_history["queue"].get(det_id, [])
                log_msg += f"    {det_id}:\n"
                log_msg += f"      Density: {density_hist[-1]:.3f}\n" if density_hist else "      Density: N/A\n"
                log_msg += f"      Queue: {queue_hist[-1]:.3f}\n" if queue_hist else "      Queue: N/A\n"
            if len(self.detectors_e2) > 3:
                log_msg += f"    ... and {len(self.detectors_e2) - 3} more detectors\n"
        
        log_msg += f"{'='*60}\n"
        print(log_msg, flush=True)  # Use print for Ray worker compatibility
    
    def enable_debug_logging(self, enable: bool = True, level: int = 1):
        """Enable or disable debug logging for this traffic signal.
        
        Args:
            enable: True to enable logging, False to disable
            level: Logging detail level (1=basic, 2=detailed, 3=verbose)
        """
        self.debug_logging = enable
        self.debug_log_level = level
        if enable:
            print(f"[TrafficSignal] Debug logging enabled for {self.id} at level {level}", flush=True)

    def get_action_stats(self) -> dict:
        """Return action statistics for the episode.
        
        Returns:
            dict: Statistics including mean, std, min, max of actions,
                  or None values if no actions recorded.
        """
        if not self.action_history:
            return {
                "mean": None, 
                "std": None, 
                "min": None,
                "max": None,
                "count": 0
            }
        
        actions = np.array(self.action_history)
        return {
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0),
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
            "count": len(actions),
        }
    
    def reset_action_tracking(self):
        """Reset action tracking for new episode."""
        self.action_history = []
        self.action_count = 0

    @staticmethod
    def _clip_reward(value: float, low: float = -3.0, high: float = 3.0) -> float:
        """Clip reward to valid range."""
        return max(low, min(high, value))

    def _pressure_reward(self):
        """Computes pressure-based reward using E2 detectors. Range: [-3, 3]."""
        pressure = self.get_pressure_from_detectors()
        # Pressure dương = ùn tắc → reward âm
        return self._clip_reward(-pressure * 3.0)

    def _average_speed_reward(self):
        avg_speed = self.get_aggregated_average_speed()
        # Map [0, 1] to [-3, 3]: 0 -> -3, 1 -> 3
        return self._clip_reward((avg_speed * 6.0) - 3.0)

    def _queue_reward(self):
        """Computes queue-based reward. Range: [-3, 3]."""
        total_queued = self.get_aggregated_queued()
        if self.max_veh == 0:
            return 0.0
        # Fewer queued vehicles = higher reward
        return self._clip_reward(-(total_queued / self.max_veh) * 3.0)

    def _occupancy_reward(self):
        """Computes occupancy-based reward. Range: [-3, 0]."""
        avg_occupancy = self.get_aggregated_occupancy()
        # Map [0, 1] to [0, -3]: 0 -> 0, 1 -> -3
        return self._clip_reward(-avg_occupancy * 3.0, low=-3.0, high=0.0)

    def _diff_waiting_time_reward(self):
        """Computes normalized difference in waiting time reward in range [-3, 3].
        
        Logic:
        - Tính tổng thời gian chờ TRUNG BÌNH trong chu kỳ từ các mẫu thu thập
        - Reward = waiting_time_cũ - waiting_time_mới (giảm waiting time → reward dương)
        - Chuẩn hóa bằng cách chia cho (max_veh * delta_time) - là giá trị thay đổi
          waiting time tối đa có thể xảy ra trong một chu kỳ
        
        Returns:
            float: Normalized reward where:
                   3.0 = waiting time giảm tối đa (tốt nhất)
                   0.0 = waiting time không đổi
                   -3.0 = waiting time tăng tối đa (tệ nhất)
        """
        # Tổng thời gian chờ TRUNG BÌNH trong chu kỳ (aggregated over 6 samples)
        ts_wait = self.get_aggregated_waiting_time()
        
        # Chênh lệch: dương nếu waiting time giảm (tốt), âm nếu tăng (xấu)
        reward = self.last_ts_waiting_time - ts_wait
        
        # Lưu lại để so sánh ở bước tiếp theo
        self.last_ts_waiting_time = ts_wait
        
        # Chuẩn hóa: max_waiting_change = max_veh * delta_time
        # (mỗi xe có thể đợi tối đa delta_time giây trong một chu kỳ)
        if self.max_veh > 0 and self.delta_time > 0:
            max_waiting_change = self.max_veh * self.delta_time
            normalized_reward = (reward / max_waiting_change) * 3.0
        else:
            normalized_reward = 0.0
        
        return max(-3.0, min(3.0, normalized_reward))

    def _halt_veh_reward_by_detectors(self):
        """Computes penalty for halting vehicles. Range: [-3.0, 0.0].
        
        Logic:
        - Calculate saturation ratio = total_halt / max_capacity.
        - Reward is negative of this ratio, scaled by 3.0.
        - 0.0 = No halting (Best).
        - -3.0 = Total gridlock (Worst).
        """
        if self.max_veh == 0:
            return 0.0
        
        # Get AGGREGATED halting vehicles over the cycle
        total_halt = self.get_aggregated_halting_vehicles()
        
        # Calculate saturation ratio
        ratio = min(1.0, total_halt / self.max_veh)
        
        # Penalty: -3.0 * ratio
        return -3.0 * float(ratio)
        
    
    def _diff_departed_veh_reward(self):
        """Computes Outflow Efficiency reward. Range: [0.0, 3.0].
        
        Logic:
        - Measures the proportion of initial vehicles that successfully departed.
        - Reward = (Departed / Initial) * 3.0
        - Promotes throughput: clearing vehicles from the intersection.
        """
        initial = float(self.initial_vehicles_this_cycle)
        departed = float(self.departed_vehicles_this_cycle)
        
        # Normalize by initial vehicles
        if initial > 0:
            # Ratio of vehicles cleared
            # Can be > 1.0 if new vehicles arrived and departed in same cycle
            ratio = departed / initial
        else:
            # If no vehicles at start:
            if departed > 0:
                # Vehicles arrived and departed -> Good flow
                ratio = 1.0
            else:
                # No demand, no flow -> Neutral
                return 0.0
        
        # Scale to 3.0 to match other reward magnitudes
        reward = ratio * 3.0
        
        # Clip to max 3.0 (100% clearance is excellent already)
        return max(0.0, min(3.0, reward))

    def _observation_fn_default(self):
        """Default observation function returning comprehensive traffic state information.
        
        Returns:
            np.ndarray: Observation vector containing:
                - phase_id: one-hot encoding of current green phase
                - min_green: whether minimum green time has passed
                - density: normalized vehicle density per incoming lane
                - queue: normalized queue length per incoming lane  
                - occupancy: normalized lane occupancy per incoming lane
                - average_speed: normalized average speed per incoming lane
        """
        # Traffic state information
        density = self.get_lanes_density_by_detectors()
        queue = self.get_lanes_queue_by_detectors()
        occupancy = self.get_lanes_occupancy_by_detectors()
        average_speed = self.get_lanes_average_speed_by_detectors()
        
        # Combine all information
        observation = np.array(density + queue + occupancy + average_speed, dtype=np.float32)
        # CRITICAL: Clip to [0, 1] to ensure observation is within observation_space
        observation = np.clip(observation, 0.0, 1.0)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.data_provider.get_lane_vehicles(lane)
            wait_time = 0.0
            for veh in veh_list:
                wait_time += self.data_provider.get_vehicle_waiting_time(veh, lane)
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            speed = self.data_provider.get_vehicle_speed(v)
            allowed_speed = self.data_provider.get_vehicle_allowed_speed(v)
            avg_speed += speed / allowed_speed if allowed_speed > 0 else 0.0
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection.
        
        Uses E2 detector data to estimate pressure based on occupancy and speed.
        """
        return self.get_pressure_from_detectors()

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection measured by E2 detectors."""
        total_queued = 0
        for det_id in self.detectors_e2:
            try:
                total_queued += self.data_provider.get_detector_halting_number(det_id)
            except Exception:
                pass
        return total_queued

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.data_provider.get_lane_vehicles(lane)
        return veh_list

    def _compute_max_veh(self):
        """Calculate the maximum possible number of vehicles detected by detectors e2."""
        self.max_veh = 0
        for det_id in self.detectors_e2:
            detector_length = self.detectors_e2_length.get(det_id, 0)
            if detector_length > 0:
                # Maximum vehicles = detector length / (minimum gap + average vehicle length)
                max_vehicles = detector_length / (self.MIN_GAP + self.avg_veh_length)
                self.max_veh += max_vehicles

    def get_total_halting_veh_by_detectors(self) -> int:
        """Returns the total number of vehicles halting in the intersection measured by detectors.
        
        Uses the jam length from E2 detectors to estimate halting vehicles.
        """
        total_halt = 0
        for det_id in self.detectors_e2:
            try:
                # Get jam length in meters from detector
                jam_length = self.data_provider.get_detector_jam_length(det_id)
                # Convert to vehicle count using average vehicle length + gap
                if jam_length > 0:
                    halt_vehicles = jam_length / (self.MIN_GAP + self.avg_veh_length)
                    total_halt += halt_vehicles
            except Exception:
                pass
        return int(total_halt)

    # =========================================================================
    # Instant Helper Functions - collect metrics at each sample (for history)
    # =========================================================================
    
    def _get_total_halting_veh_instant(self) -> float:
        """Returns instantaneous halting vehicle count (for history tracking)."""
        return float(self.get_total_halting_veh_by_detectors())
    
    def _get_total_queued_instant(self) -> float:
        """Returns instantaneous queued vehicle count (for history tracking)."""
        return float(self.get_total_queued())
    
    def _get_average_speed_instant(self) -> float:
        """Returns instantaneous average speed from detectors (for history tracking).
        
        Uses aggregated detector data to compute average speed across all lanes.
        """
        total_speed = 0.0
        count = 0
        for det_id in self.detectors_e2:
            try:
                speed = self.data_provider.get_detector_mean_speed(det_id)
                if speed >= 0:  # Valid speed reading
                    lane_id = self.data_provider.get_detector_lane_id(det_id)
                    max_speed = self.data_provider.get_lane_max_speed(lane_id)
                    if max_speed > 0:
                        total_speed += speed / max_speed
                        count += 1
            except Exception:
                pass
        
        if count > 0:
            return min(1.0, total_speed / count)
        return 1.0  # Default to max speed if no data
    
    def _get_waiting_time_from_detectors(self) -> float:
        """Estimates waiting time from E2 detectors using jam length and occupancy.
        
        Logic:
        - Sử dụng jam_length (meters) và sampling_interval để ước tính waiting time
        - Mỗi xe trong hàng đợi (jam) được coi là đang chờ
        - waiting_time = (jam_length / avg_veh_length) * sampling_interval
        
        Returns:
            float: Estimated total waiting time in seconds
        """
        total_waiting_time = 0.0
        
        for det_id in self.detectors_e2:
            try:
                # Lấy jam length (meters) từ detector
                jam_length = self.data_provider.get_detector_jam_length(det_id)
                
                if jam_length > 0:
                    # Ước tính số xe trong hàng đợi
                    vehicles_in_jam = jam_length / (self.MIN_GAP + self.avg_veh_length)
                    # Mỗi xe đợi trong khoảng sampling_interval
                    total_waiting_time += vehicles_in_jam * self.sampling_interval_s
            except Exception:
                pass
        
        return total_waiting_time
    
    def get_pressure_from_detectors(self) -> float:
        """Calculates pressure from E2 detectors. Range: [-1, 1].
        
        Pressure = (incoming vehicles - outgoing vehicles) / max_capacity
        
        Sử dụng:
        - occupancy cao → nhiều xe đến (incoming)
        - speed cao + occupancy thấp → nhiều xe đi (outgoing)
        
        Returns:
            float: Normalized pressure [-1, 1]. 
                   Positive = congested, Negative = free flow
        """
        if self.max_veh == 0:
            return 0.0
        
        total_occupancy = 0.0
        total_speed_factor = 0.0
        count = 0
        
        for det_id in self.detectors_e2:
            try:
                # Occupancy (0-100) normalized to [0, 1]
                occupancy = self.data_provider.get_detector_occupancy(det_id) / 100.0
                
                # Speed normalized to [0, 1]
                speed = self.data_provider.get_detector_mean_speed(det_id)
                lane_id = self.data_provider.get_detector_lane_id(det_id)
                max_speed = self.data_provider.get_lane_max_speed(lane_id)
                
                if max_speed > 0:
                    normalized_speed = min(1.0, speed / max_speed)
                else:
                    normalized_speed = 1.0
                
                total_occupancy += occupancy
                total_speed_factor += normalized_speed
                count += 1
            except Exception:
                pass
        
        if count == 0:
            return 0.0
        
        avg_occupancy = total_occupancy / count  # [0, 1]
        avg_speed = total_speed_factor / count   # [0, 1]
        
        # Pressure = occupancy - speed (simplified model)
        # High occupancy + low speed = positive pressure (congestion)
        # Low occupancy + high speed = negative pressure (free flow)
        pressure = avg_occupancy - avg_speed
        
        return max(-1.0, min(1.0, pressure))
    
    def _get_occupancy_instant(self) -> float:
        """Returns instantaneous average occupancy from E2 detectors.
        
        Returns:
            float: Normalized occupancy [0, 1]
        """
        total_occupancy = 0.0
        count = 0
        
        for det_id in self.detectors_e2:
            try:
                occupancy = self.data_provider.get_detector_occupancy(det_id) / 100.0
                total_occupancy += occupancy
                count += 1
            except Exception:
                pass
        
        if count > 0:
            return min(1.0, total_occupancy / count)
        return 0.0
    
    # =========================================================================
    # Aggregated Getter Functions - return mean values over the cycle (60s)
    # =========================================================================
    
    def get_aggregated_halting_vehicles(self) -> float:
        """Returns mean halting vehicle count over the cycle.
        
        Aggregates samples collected during the cycle (6 samples with 10s interval = 60s).
        Falls back to instantaneous value if no history available.
        """
        history = self.reward_metrics_history.get("halting_vehicles", [])
        if history:
            return float(np.mean(history))
        return float(self.get_total_halting_veh_by_detectors())
    
    def get_aggregated_queued(self) -> float:
        """Returns mean queued vehicle count over the cycle.
        
        Aggregates samples collected during the cycle (6 samples with 10s interval = 60s).
        Falls back to instantaneous value if no history available.
        """
        history = self.reward_metrics_history.get("total_queued", [])
        if history:
            return float(np.mean(history))
        return float(self.get_total_queued())
    
    def get_aggregated_occupancy(self) -> float:
        """Returns mean occupancy over the cycle from E2 detectors.
        
        Aggregates occupancy samples collected during the cycle.
        Returns normalized value [0, 1].
        
        Falls back to instantaneous value if no history available.
        """
        # Get occupancy from detector history (stored during update_detectors_history)
        total_occupancy = 0.0
        count = 0
        
        for det_id in self.detectors_e2:
            occ_history = self.detector_history.get("occupancy", {}).get(det_id, [])
            if occ_history:
                total_occupancy += np.mean(occ_history)
                count += 1
        
        if count > 0:
            return min(1.0, total_occupancy / count)
        
        # Fallback: get instantaneous occupancy
        return self._get_occupancy_instant()
    
    def get_aggregated_average_speed(self) -> float:
        """Returns mean average speed over the cycle.
        
        Aggregates samples collected during the cycle (6 samples with 10s interval = 60s).
        Falls back to instantaneous value if no history available.
        """
        history = self.reward_metrics_history.get("average_speed", [])
        if history:
            return float(np.mean(history))
        return self._get_average_speed_instant()
    
    def get_aggregated_waiting_time(self) -> float:
        """Returns mean total waiting time over the cycle.
        
        Aggregates samples collected during the cycle (6 samples with 10s interval = 60s).
        Falls back to instantaneous value if no history available.
        """
        history = self.reward_metrics_history.get("waiting_time", [])
        if history:
            return float(np.mean(history))
        return float(sum(self.get_accumulated_waiting_time_per_lane()))

    def get_current_vehicle_count(self) -> int:
        """Returns the current number of vehicles in the intersection's detectors."""
        total = 0
        for det_id in self.detectors_e2:
            try:
                count = self.data_provider.get_detector_vehicle_count(det_id)
                total += count
            except Exception:
                pass
        return total

    def update_cycle_vehicle_tracking(self):
        """Update vehicle tracking for diff_departed_veh reward.
        
        Should be called when a new cycle starts (in set_next_phase).
        Computes departed vehicles from the PREVIOUS cycle before resetting.
        
        IMPORTANT: Both departed_vehicles_this_cycle and initial_vehicles_this_cycle
        refer to the PREVIOUS cycle for reward calculation. They are updated together
        so the reward can use consistent values from the same cycle.
        """
        # Get current vehicles in detection area
        current_vehicles = self._get_current_vehicle_ids()
        
        if self._vehicles_seen_this_cycle:  # Skip on first cycle
            # Compute departed vehicles from the PREVIOUS cycle
            # Departed = vehicles that were seen during the cycle but are no longer present
            vehicles_that_left = self._vehicles_seen_this_cycle - current_vehicles
            self.departed_vehicles_this_cycle = len(vehicles_that_left)
            
            # initial_vehicles_this_cycle was set at the START of the previous cycle
            # (in _vehicles_at_cycle_start), so it's already correct for reward
            self.initial_vehicles_this_cycle = len(self._vehicles_at_cycle_start)
        else:
            # First cycle - no previous data
            self.departed_vehicles_this_cycle = 0
            self.initial_vehicles_this_cycle = 0
        
        # Now prepare for the NEW cycle
        # Store current vehicles as the starting point for next cycle's reward
        self._vehicles_at_cycle_start = current_vehicles.copy()
        
        # Reset the seen vehicles set for the new cycle (start with current vehicles)
        self._vehicles_seen_this_cycle = current_vehicles.copy()
        
        self.halting_vehicles_samples = []
        
        # Reset reward metrics history at start of new cycle
        # This ensures fresh aggregation for the new cycle
        for key in self.reward_metrics_history:
            self.reward_metrics_history[key] = []

    def update_departed_vehicles(self):
        """Track unique vehicles seen during the cycle.
        
        Should be called each simulation step to accumulate all vehicles
        that pass through the intersection during this cycle.
        The actual departed count is computed at the end of the cycle
        in update_cycle_vehicle_tracking().
        """
        # Get all unique vehicle IDs currently in detection area
        current_vehicles = self._get_current_vehicle_ids()
        # Add them to the set of vehicles seen this cycle
        self._vehicles_seen_this_cycle.update(current_vehicles)
    
    def _get_current_vehicle_ids(self) -> set:
        """Get the set of unique vehicle IDs currently in the detection area.
        
        Returns:
            set: Set of vehicle IDs currently detected by E2 detectors.
        """
        vehicle_ids = set()
        for det_id in self.detectors_e2:
            try:
                ids = self.data_provider.get_detector_vehicle_ids(det_id)
                vehicle_ids.update(ids)
            except Exception:
                pass
        return vehicle_ids


    def get_lanes_density_by_detectors(self) -> List[float]:
        """Trả về mật độ trung bình trong khoảng delta_time cho mỗi detector.
        
        Returns:
            List[float]: Danh sách chứa mật độ chuẩn hóa [0,1] trung bình cho mỗi detector.
        """
        avg_densities = []
        for det_id in self.detectors_e2:
            history = self.detector_history["density"].get(det_id, [])
            if history:
                # Clip to [0, 1] and ensure float32
                val = float(np.clip(np.mean(history), 0.0, 1.0))
                avg_densities.append(val)
            else:
                avg_densities.append(0.0)
        return avg_densities

    def get_lanes_queue_by_detectors(self) -> List[float]:
        """Trả về hàng đợi trung bình trong khoảng delta_time cho mỗi detector.

        Returns:
            List[float]: Danh sách chứa độ dài hàng đợi chuẩn hóa [0,1] trung bình cho mỗi detector.
        """
        avg_queues = []
        for det_id in self.detectors_e2:
            history = self.detector_history["queue"].get(det_id, [])
            if history:
                # Clip to [0, 1] and ensure float32
                val = float(np.clip(np.mean(history), 0.0, 1.0))
                avg_queues.append(val)
            else:
                avg_queues.append(0.0)
        return avg_queues

    def get_lanes_occupancy_by_detectors(self) -> List[float]:
        """Trả về độ chiếm dụng trung bình trong khoảng delta_time cho mỗi detector.
        
        Returns:
            List[float]: Danh sách chứa độ chiếm dụng chuẩn hóa [0,1] trung bình cho mỗi detector.
        """
        avg_occupancies = []
        for det_id in self.detectors_e2:
            history = self.detector_history["occupancy"].get(det_id, [])
            if history:
                # Clip to [0, 1] and ensure float32
                val = float(np.clip(np.mean(history), 0.0, 1.0))
                avg_occupancies.append(val)
            else:
                avg_occupancies.append(0.0)
        return avg_occupancies
    
    def get_lanes_average_speed_by_detectors(self) -> List[float]:
        """Trả về tốc độ trung bình trong khoảng delta_time cho mỗi detector.
        
        Returns:
            List[float]: Danh sách chứa tốc độ trung bình chuẩn hóa [0,1] cho mỗi detector.
        """
        avg_speeds = []
        for det_id in self.detectors_e2:
            history = self.detector_history["average_speed"].get(det_id, [])
            if history:
                # Clip to [0, 1] and ensure float32
                val = float(np.clip(np.mean(history), 0.0, 1.0))
                avg_speeds.append(val)
            else:
                avg_speeds.append(1.0)
        return avg_speeds

    def get_observation_history(self, window_size: int) -> List[Any]:
        """
        Trả về lịch sử quan sát trong window_size bước gần nhất.
        
        Args:
            window_size: Số lượng bước thời gian quá khứ cần lấy.
            
        Returns:
            List[Any]: Danh sách các vector/dict quan sát, độ dài = window_size.
                       Nếu lịch sử chưa đủ, sẽ padding bằng quan sát đầu tiên hoặc 0.
        """
        if not self.observation_history:
            # Construct a default observation if history is empty
            
            # SPECIAL CASE: Wrapper functions (like NeighborTemporalObservationFunction)
            # define compute_current_observation() which returns a Vector (Box),
            # but their observation_space() returns a Dict.
            # In this case, the history MUST store Vectors (matching compute_current_observation),
            # so the padding must be Vectors, not Dicts.
            if hasattr(self.observation_fn, "compute_current_observation"):
                obs_dim = 4 * len(self.detectors_e2) if self.detectors_e2 else 48
                default_obs = np.zeros(obs_dim, dtype=np.float32)
                return [default_obs for _ in range(window_size)]
            
            # Standard case: Use observation_space structure
            if self.observation_space is not None:
                if isinstance(self.observation_space, spaces.Dict):
                    default_obs = {}
                    for k, space in self.observation_space.spaces.items():
                        if hasattr(space, 'low'):
                            default_obs[k] = np.zeros(space.shape, dtype=np.float32)
                        elif hasattr(space, 'shape'):
                            default_obs[k] = np.zeros(space.shape, dtype=np.float32)
                        else:
                            default_obs[k] = np.zeros((1,), dtype=np.float32)
                elif hasattr(self.observation_space, 'shape') and self.observation_space.shape is not None:
                    default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                else:
                    # Fallback
                    default_obs = np.zeros((1,), dtype=np.float32)
            else:
                # No space defined, fallback to estimated size
                obs_dim = 4 * len(self.detectors_e2) if self.detectors_e2 else 48
                default_obs = np.zeros((obs_dim,), dtype=np.float32)
                
            return [default_obs for _ in range(window_size)]
            
        # Ensure all history elements are validated/clipped
        history = [self._validate_and_clip_observation(obs) for obs in self.observation_history]
        
        # Padding
        if len(history) < window_size:
            # Use the first available observation for padding structure
            padding_value = copy.deepcopy(history[0])
            
            # Zero out the padding value
            if isinstance(padding_value, dict):
                for k in padding_value:
                    if isinstance(padding_value[k], np.ndarray):
                        padding_value[k] = np.zeros_like(padding_value[k])
            elif isinstance(padding_value, np.ndarray):
                padding_value = np.zeros_like(padding_value)
                
            padding = [padding_value] * (window_size - len(history))
            history = padding + history
            
        # Return last window_size elements
        return history[-window_size:]

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "occupancy": _occupancy_reward,
        "pressure": _pressure_reward,
        "halt-veh-by-detectors": _halt_veh_reward_by_detectors,
        "diff-departed-veh": _diff_departed_veh_reward,
    }