"""Observation functions for traffic signals.

Each observation function produces a Dict with at least:
  * ``features``:    lane metrics + green-time ratios  [56]
  * ``action_mask``: binary mask for valid phases      [8]

Feature layout (56 dims):
  [Lane0_density, Lane0_queue, Lane0_occupancy, Lane0_speed,
   Lane1_density, ..., Lane11_speed,               # 48 dims  (12 lanes × 4)
   GreenRatio_Phase0, ..., GreenRatio_Phase7]       #  8 dims
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from .traffic_signal import TrafficSignal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_NUM_LANES = 12
NUM_LANE_FEATURES = 4   # density, queue, occupancy, avg_speed
NUM_GREEN_TIME_FEATURES = 8  # one per standard phase
NUM_STANDARD_PHASES = 8
LANE_FEATURE_DIM = TARGET_NUM_LANES * NUM_LANE_FEATURES          # 48
TOTAL_FEATURE_DIM = LANE_FEATURE_DIM + NUM_GREEN_TIME_FEATURES   # 56


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_or_trim(values: list, target_len: int, fill: float = 0.0) -> list:
    """Pad with *fill* or trim *values* to exactly *target_len*."""
    if len(values) >= target_len:
        return values[:target_len]
    return values + [fill] * (target_len - len(values))


def _build_lane_features(ts: "TrafficSignal") -> np.ndarray:
    """Build lane-major feature vector [48] = 12 lanes × 4 metrics.

    If the intersection has fewer than 12 detectors, the missing lanes are
    zero-padded.  If it has more, they are trimmed.
    """
    density = _pad_or_trim(ts.get_lanes_density_by_detectors(), TARGET_NUM_LANES)
    queue = _pad_or_trim(ts.get_lanes_queue_by_detectors(), TARGET_NUM_LANES)
    occupancy = _pad_or_trim(ts.get_lanes_occupancy_by_detectors(), TARGET_NUM_LANES)
    avg_speed = _pad_or_trim(ts.get_lanes_average_speed_by_detectors(), TARGET_NUM_LANES, fill=1.0)

    data = []
    for i in range(TARGET_NUM_LANES):
        data.extend([density[i], queue[i], occupancy[i], avg_speed[i]])

    return np.clip(np.array(data, dtype=np.float32), 0.0, 1.0)


def _build_green_time_features(ts: "TrafficSignal") -> np.ndarray:
    """Green-time ratio features [8], each in [0, 1]."""
    return ts.get_green_time_ratios()  # already float32, clipped


def _build_full_features(ts: "TrafficSignal") -> np.ndarray:
    """Concatenate lane features [48] + green-time ratios [8] → [56]."""
    lane = _build_lane_features(ts)
    green = _build_green_time_features(ts)
    return np.concatenate([lane, green])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: "TrafficSignal"):
        self.ts = ts

    @abstractmethod
    def __call__(self) -> dict:
        ...

    @abstractmethod
    def observation_space(self) -> spaces.Dict:
        ...


# ---------------------------------------------------------------------------
# Default (single-step)
# ---------------------------------------------------------------------------

class DefaultObservationFunction(ObservationFunction):
    """Single-step observation: features [56] + action_mask [8]."""

    def __call__(self) -> dict:
        return {
            "features": _build_full_features(self.ts),
            "action_mask": np.array(self.ts.get_action_mask(), dtype=np.float32),
        }

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "features": spaces.Box(0.0, 1.0, shape=(TOTAL_FEATURE_DIM,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(NUM_STANDARD_PHASES,), dtype=np.float32),
        })


# ---------------------------------------------------------------------------
# Spatio-Temporal (history window)
# ---------------------------------------------------------------------------

class SpatioTemporalObservationFunction(ObservationFunction):
    """Stacked history observation: features [T × 56] + action_mask [8]."""

    def __init__(self, ts: "TrafficSignal", window_size: int = 5):
        super().__init__(ts)
        self.window_size = getattr(ts, "window_size", window_size)

    def compute_current_observation(self) -> np.ndarray:
        """Single-step feature vector [56] for history tracking."""
        return _build_full_features(self.ts)

    def __call__(self) -> dict:
        history = self.ts.get_observation_history(self.window_size)
        stacked = np.array(history, dtype=np.float32).flatten()
        features = np.clip(stacked, 0.0, 1.0).astype(np.float32)
        return {
            "features": features,
            "action_mask": np.array(self.ts.get_action_mask(), dtype=np.float32),
        }

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "features": spaces.Box(
                0.0, 1.0,
                shape=(self.window_size * TOTAL_FEATURE_DIM,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(0.0, 1.0, shape=(NUM_STANDARD_PHASES,), dtype=np.float32),
        })


# ---------------------------------------------------------------------------
# Neighbor-Temporal (for Local GNN)
# ---------------------------------------------------------------------------

class NeighborTemporalObservationFunction(ObservationFunction):
    """Observation with pre-packaged neighbour features for Local GNN.

    Returns:
        self_features       [T, 56]
        neighbor_features   [K, T, 56]
        neighbor_mask       [K]
        neighbor_directions [K]
        action_mask         [8]
    """

    def __init__(
        self,
        ts: "TrafficSignal",
        neighbor_provider=None,
        max_neighbors: int = 4,
        window_size: int = 5,
    ):
        super().__init__(ts)
        self.neighbor_provider = neighbor_provider
        self.max_neighbors = max_neighbors
        self.window_size = getattr(ts, "window_size", window_size)

    def compute_current_observation(self) -> np.ndarray:
        """Single-step feature vector [56]."""
        return _build_full_features(self.ts)

    def __call__(self) -> dict:
        feature_dim = TOTAL_FEATURE_DIM
        T = self.window_size
        K = self.max_neighbors

        # --- Self history ---
        self_history = self.ts.get_observation_history(T)
        processed = [self._extract_features(o) for o in self_history]
        self_features = np.clip(
            np.array(processed, dtype=np.float32).reshape(T, -1)[:, :feature_dim],
            0.0, 1.0,
        )
        # Ensure exactly [T, feature_dim]
        if self_features.shape != (T, feature_dim):
            padded = np.zeros((T, feature_dim), dtype=np.float32)
            rows = min(self_features.shape[0], T)
            cols = min(self_features.shape[1], feature_dim)
            padded[:rows, :cols] = self_features[:rows, :cols]
            self_features = padded

        # --- Neighbour features ---
        neighbor_features = np.zeros((K, T, feature_dim), dtype=np.float32)
        neighbor_mask = np.zeros(K, dtype=np.float32)

        if self.neighbor_provider is not None:
            neighbor_ids = self.neighbor_provider.get_neighbor_ids(self.ts.id)
            if neighbor_ids:
                neighbor_ids = sorted(neighbor_ids)
            for i, nid in enumerate(neighbor_ids[:K]):
                if nid is None:
                    continue
                nh = self.neighbor_provider.get_observation_history(nid, T)
                if nh and len(nh) > 0:
                    try:
                        arr = np.array(
                            [self._extract_features(o) for o in nh], dtype=np.float32
                        )
                        arr = np.clip(arr.reshape(-1, feature_dim)[-T:], 0.0, 1.0)
                        if arr.shape == neighbor_features[i].shape:
                            neighbor_features[i] = arr
                            neighbor_mask[i] = 1.0
                    except Exception:
                        pass

        # --- Directions ---
        neighbor_directions = self._get_neighbor_directions()

        return {
            "self_features": self_features,
            "neighbor_features": neighbor_features,
            "neighbor_mask": neighbor_mask,
            "neighbor_directions": neighbor_directions,
            "action_mask": np.array(self.ts.get_action_mask(), dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(obs) -> np.ndarray:
        """Pull a flat feature vector from whatever the history stores."""
        if isinstance(obs, dict):
            if "self_features" in obs:
                val = obs["self_features"]
                if hasattr(val, "ndim") and val.ndim > 1:
                    return val[-1]
                return np.asarray(val, dtype=np.float32)
            if "features" in obs:
                return np.asarray(obs["features"], dtype=np.float32)
            return np.concatenate([np.asarray(v).flatten() for v in obs.values()])
        return np.asarray(obs, dtype=np.float32)

    def _get_neighbor_directions(self) -> np.ndarray:
        """Direction indices: 0.0=N, 0.25=E, 0.5=S, 0.75=W, -1=pad."""
        K = self.max_neighbors
        dirs = np.full(K, -1.0, dtype=np.float32)
        if self.neighbor_provider is not None and hasattr(self.neighbor_provider, "get_neighbor_directions"):
            raw = self.neighbor_provider.get_neighbor_directions(self.ts.id)
            for i, d in enumerate(raw[:K]):
                if d >= 0:
                    dirs[i] = d / 4.0
        return dirs

    def observation_space(self) -> spaces.Dict:
        feature_dim = TOTAL_FEATURE_DIM
        T = self.window_size
        K = self.max_neighbors
        return spaces.Dict({
            "self_features": spaces.Box(0.0, 1.0, shape=(T, feature_dim), dtype=np.float32),
            "neighbor_features": spaces.Box(0.0, 1.0, shape=(K, T, feature_dim), dtype=np.float32),
            "neighbor_mask": spaces.Box(0.0, 1.0, shape=(K,), dtype=np.float32),
            "neighbor_directions": spaces.Box(-1.0, 1.0, shape=(K,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(NUM_STANDARD_PHASES,), dtype=np.float32),
        })
