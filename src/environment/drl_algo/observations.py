"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        density = self.ts.get_lanes_density_by_detectors()
        queue = self.ts.get_lanes_queue_by_detectors()
        occupancy = self.ts.get_lanes_occupancy_by_detectors()
        average_speed = self.ts.get_lanes_average_speed_by_detectors()
        # CRITICAL FIX: Reorder to Lane-major [Lane0_Feats, Lane1_Feats, ...]
        # This matches the model's expectation: .view(-1, 12, 4)
        obs_data = []
        num_lanes = len(self.ts.detectors_e2)
        for i in range(num_lanes):
            obs_data.extend([density[i], queue[i], occupancy[i], average_speed[i]])
            
        observation = np.array(obs_data, dtype=np.float32)
        # CRITICAL: Clip to [0, 1] AND ensure dtype is float32 to match observation_space
        observation = np.clip(observation, 0.0, 1.0).astype(np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(4 * len(self.ts.detectors_e2), dtype=np.float32),
            high=np.ones(4 * len(self.ts.detectors_e2), dtype=np.float32),
        )


class SpatioTemporalObservationFunction(ObservationFunction):
    """Observation function that returns a history of observations."""

    def __init__(self, ts: TrafficSignal, window_size: int = 5):
        """Initialize spatio-temporal observation function."""
        super().__init__(ts)
        self.window_size = window_size
        # Try to get window_size from TrafficSignal if available (will be added later)
        if hasattr(ts, "window_size"):
            self.window_size = ts.window_size

    def compute_current_observation(self) -> np.ndarray:
        """Return the current single-step observation."""
        density = self.ts.get_lanes_density_by_detectors()
        queue = self.ts.get_lanes_queue_by_detectors()
        occupancy = self.ts.get_lanes_occupancy_by_detectors()
        average_speed = self.ts.get_lanes_average_speed_by_detectors()
        # CRITICAL FIX: Reorder to Lane-major [Lane0_Feats, Lane1_Feats, ...]
        # This matches the model's expectation: .view(-1, 12, 4)
        obs_data = []
        num_lanes = len(self.ts.detectors_e2)
        for i in range(num_lanes):
            obs_data.extend([density[i], queue[i], occupancy[i], average_speed[i]])

        observation = np.array(obs_data, dtype=np.float32)
        # CRITICAL: Clip to [0, 1] AND ensure dtype is float32 to match observation_space
        observation = np.clip(observation, 0.0, 1.0).astype(np.float32)
        return observation

    def __call__(self) -> np.ndarray:
        """Return the stacked observation history."""
        # Get history from TrafficSignal
        history = self.ts.get_observation_history(self.window_size)
        
        # Stack into a single array [window_size, features]
        # Flattening might be needed depending on how the model expects it, 
        # but usually we want to keep the time dimension separate or flatten it.
        # Here we return flattened array [window_size * features] to be compatible with Gym spaces
        # The model will reshape it back to [window_size, features]
        stacked_obs = np.array(history, dtype=np.float32).flatten()
        # CRITICAL: Clip to [0, 1] AND ensure dtype is float32 to match observation_space
        stacked_obs = np.clip(stacked_obs, 0.0, 1.0).astype(np.float32)
        return stacked_obs

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        # Base feature size
        feature_size = 4 * len(self.ts.detectors_e2)
        
        return spaces.Box(
            low=np.zeros(self.window_size * feature_size, dtype=np.float32),
            high=np.ones(self.window_size * feature_size, dtype=np.float32),
        )


class NeighborTemporalObservationFunction(ObservationFunction):
    """Spatio-Temporal observation with pre-packaged neighbor features.
    
    This observation function is designed for Local GNN processing with RLlib.
    Instead of requiring global graph structure, each observation contains:
    - Self features history: [T, feature_dim]
    - Neighbor features history: [K, T, feature_dim]
    - Neighbor mask: [K] indicating which neighbors are valid
    
    This allows the model to perform local GNN operations without
    needing to reconstruct the global graph from shuffled batches.
    """

    def __init__(
        self, 
        ts: TrafficSignal, 
        neighbor_provider=None,
        max_neighbors: int = 4,
        window_size: int = 5
    ):
        """Initialize neighbor temporal observation function.
        
        Args:
            ts: TrafficSignal instance for this agent
            neighbor_provider: Object that provides neighbor info and observations
            max_neighbors: Maximum number of neighbors (pad if less)
            window_size: Number of historical timesteps (T)
        """
        super().__init__(ts)
        self.neighbor_provider = neighbor_provider
        self.max_neighbors = max_neighbors
        self.window_size = window_size
        
        # Override from TrafficSignal if available
        if hasattr(ts, "window_size"):
            self.window_size = ts.window_size
            
    def compute_current_observation(self) -> np.ndarray:
        """Return the current single-step observation (for history tracking)."""
        density = self.ts.get_lanes_density_by_detectors()
        queue = self.ts.get_lanes_queue_by_detectors()
        occupancy = self.ts.get_lanes_occupancy_by_detectors()
        average_speed = self.ts.get_lanes_average_speed_by_detectors()
        # CRITICAL FIX: Reorder to Lane-major [Lane0_Feats, Lane1_Feats, ...]
        obs_data = []
        num_lanes = len(self.ts.detectors_e2)
        for i in range(num_lanes):
            obs_data.extend([density[i], queue[i], occupancy[i], average_speed[i]])
            
        observation = np.array(obs_data, dtype=np.float32)
        observation = np.clip(observation, 0.0, 1.0).astype(np.float32)
        return observation

    def __call__(self) -> dict:
        """Return Dict observation with self, neighbor features and mask.
        
        Returns:
            Dict with keys:
                - self_features: np.ndarray of shape [T, feature_dim]
                - neighbor_features: np.ndarray of shape [K, T, feature_dim]  
                - neighbor_mask: np.ndarray of shape [K]
        """
        feature_dim = 4 * len(self.ts.detectors_e2)  # Typically 48
        
        # Get T-step history for self
        # Note: self.ts.get_observation_history now returns validated/clipped observations
        # which might be Dicts if using nested observations.
        # However, for Local GNN, we expect the BASE observation to be a vector (Box)
        # We need to extract the raw features vector if the history contains Dicts.
        
        self_history = self.ts.get_observation_history(self.window_size)
        
        # Helper to extract feature vector from observation history element
        def extract_features(obs):
            if isinstance(obs, dict):
                # If observation is already a dict (e.g. from previous step's NeighborObs),
                # we need to extract the "self_features" part or flatten it.
                # BUT: The base observation function usually returns a vector.
                # If we get a dict here, it means we are using a complex observation class
                # that wraps another one.
                # For NeighborTemporalObservationFunction, the base internal observation
                # should be the DEFAULT vector observation.
                if "self_features" in obs:
                    return obs["self_features"][-1] # Take most recent if it's a sequence
                # Fallback: flatten values
                return np.concatenate([v.flatten() for v in obs.values()])
            return np.asarray(obs, dtype=np.float32)

        processed_history = [extract_features(obs) for obs in self_history]
        self_features = np.array(processed_history, dtype=np.float32)  # [T, feature_dim]
        # Ensure shape [T, feature_dim] (handle if extract_features returns [1, feature_dim])
        if self_features.ndim > 2:
             self_features = self_features.reshape(self.window_size, -1)
             
        self_features = np.clip(self_features, 0.0, 1.0)
        
        # Get neighbor features
        neighbor_features, neighbor_mask = self._get_neighbor_features(feature_dim)
        
        return {
            "self_features": self_features,
            "neighbor_features": neighbor_features,
            "neighbor_mask": neighbor_mask,
        }
        
    def _get_neighbor_features(self, feature_dim: int):
        """Get T-step history for all neighbors with padding.
        
        Args:
            feature_dim: Single observation feature dimension
            
        Returns:
            Tuple of (neighbor_features, neighbor_mask)
            - neighbor_features: [K, T, feature_dim]
            - neighbor_mask: [K] with 1.0 for valid neighbors, 0.0 for padding
        """
        K = self.max_neighbors
        T = self.window_size
        
        # Initialize with zeros (padding)
        neighbor_features = np.zeros((K, T, feature_dim), dtype=np.float32)
        neighbor_mask = np.zeros(K, dtype=np.float32)
        
        if self.neighbor_provider is None:
            return neighbor_features, neighbor_mask
            
        # Get list of neighbor IDs
        neighbor_ids = self.neighbor_provider.get_neighbor_ids(self.ts.id)

        # === FIX: SORT NEIGHBOR IDS TO ENSURE CONSISTENT ORDER FOR BiGRU ===
        # BiGRU is sequence-sensitive. Random order from Set causes fluctuation.
        if neighbor_ids:
            neighbor_ids = sorted(neighbor_ids)
        # ===================================================================
        
        for i, neighbor_id in enumerate(neighbor_ids[:K]):
            if neighbor_id is None:
                continue
                
            # Get neighbor's observation history
            neighbor_history = self.neighbor_provider.get_observation_history(
                neighbor_id, T
            )
            
            if neighbor_history is not None and len(neighbor_history) > 0:
                # Helper to extract feature vector from observation history element
                # Same logic as above: handle if neighbor history contains Dicts
                def extract_features(obs):
                    if isinstance(obs, dict):
                        if "self_features" in obs:
                            feat = obs["self_features"]
                            # If shape is [T, F], take last? Or is it [F]?
                            # Usually history items are individual time steps.
                            # If obs is a Dict from NeighborObs, it has 'self_features' as [T, F]
                            # BUT neighbor history should store raw per-step observations from BaseObs
                            # If it stores NeighborObs, we have recursion.
                            # ASSUMPTION: Neighbor provider returns the history of the NEIGHBOR'S
                            # observation function. If neighbor also uses NeighborObs, it returns Dicts.
                            # We need the RAW feature vector (density, queue, etc.)
                            
                            # If the neighbor uses NeighborTemporalObservationFunction, 
                            # its compute_observation() returns a Dict.
                            # We want the 'self_features' part of it, which represents its own state.
                            # 'self_features' in NeighborObs is [T, F] (history).
                            # We just want the most recent one? Or the whole history?
                            
                            # Wait, get_observation_history returns list of observations.
                            # If Neighbor uses NeighborObs, each item in history is a Dict.
                            # Inside that Dict, 'self_features' is ALREADY a history [T, F].
                            # This is redundant.
                            
                            # CORRECT APPROACH:
                            # The neighbor_provider should provide access to the neighbor's BASE features.
                            # But currently it likely calls get_observation_history() on the neighbor TS.
                            
                            # If neighbor supports 'get_lanes_density...' etc, we can reconstruct? No.
                            
                            # Simplification: extracting 'self_features' from the Dict
                            # If obs is {self: [T,F], ...}, we likely want the last step of self.
                            # But wait, history contains T steps.
                            # If each step is a Dict, it means we stored Dicts in history.
                            
                            # Let's look at TrafficSignal.compute_observation():
                            # It appends current_obs to history.
                            # If obs_fn returns Dict, history has Dicts.
                            
                            if "self_features" in obs: # It's a NeighborObs output
                                # It contains a history window [T, F]. 
                                # We probably just want the most recent frame from it?
                                # Or is this a single step obs that just happens to be shaped [1, F]?
                                val = obs["self_features"]
                                if val.ndim > 1: return val[-1] # Take last time step
                                return val
                        return np.concatenate([v.flatten() for v in obs.values()])
                    return np.asarray(obs, dtype=np.float32)

                try:
                    processed_history = [extract_features(obs) for obs in neighbor_history]
                    hist_array = np.array(processed_history, dtype=np.float32)
                    
                    # Ensure shape [T, feature_dim]
                    if hist_array.ndim != 2:
                        # Try to reshape if total elements match
                        if hist_array.size == T * feature_dim:
                            hist_array = hist_array.reshape(T, feature_dim)
                        else:
                            # Log warning or skip?
                            # Fallback: slice or pad
                            if hist_array.shape[0] > T: hist_array = hist_array[-T:]
                            # Dimension mismatch is hard to fix blindly
                            pass

                    hist_array = np.clip(hist_array, 0.0, 1.0)
                    
                    # Only assign if shapes match
                    if hist_array.shape == neighbor_features[i].shape:
                        neighbor_features[i] = hist_array
                        neighbor_mask[i] = 1.0
                except Exception:
                    pass # Skip this neighbor if parsing fails
                
        return neighbor_features, neighbor_mask

    def observation_space(self) -> spaces.Dict:
        """Return the Dict observation space.
        
        Returns:
            spaces.Dict with:
                - self_features: Box [T, feature_dim]
                - neighbor_features: Box [K, T, feature_dim]
                - neighbor_mask: Box [K]
        """
        feature_dim = 4 * len(self.ts.detectors_e2)
        T = self.window_size
        K = self.max_neighbors
        
        return spaces.Dict({
            "self_features": spaces.Box(
                low=0.0, high=1.0, 
                shape=(T, feature_dim), 
                dtype=np.float32
            ),
            "neighbor_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(K, T, feature_dim),
                dtype=np.float32
            ),
            "neighbor_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(K,),
                dtype=np.float32
            ),
        })