"""TrafficSignal – RL agent wrapper for a single signalised intersection.

Responsibilities:
  * Observation computation (delegates to an ObservationFunction)
  * Action translation  (ratio → green times, or discrete ±Δ adjustment)
  * Reward computation   (delegates to ``rewards`` module)
  * Detector-history bookkeeping

Design notes:
  * **No SUMO dependency** – all simulator data flows through ``data_provider``.
  * Action masking via FRAP ``PhaseStandardizer`` ensures only physically-valid
    phases receive green time.
"""

from __future__ import annotations

import copy
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces

from .rewards import REWARD_REGISTRY, get_reward_fn

# Optional FRAP import
try:
    from preprocessing.frap import PhaseStandardizer
except ImportError:
    PhaseStandardizer = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("TrafficSignal")
logger.setLevel(logging.WARNING)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.WARNING)
    _handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_STANDARD_PHASES = 8
MIN_GAP = 2.5            # SUMO default gap
DEFAULT_AVG_VEH_LEN = 3.0
SAMPLING_INTERVAL_S = 10  # detector sampling every 10 s
MAX_HISTORY_SAMPLES = 5   # samples kept per cycle


class TrafficSignal:
    """RL agent that controls one traffic signal.

    Action Modes
    -------------
    ``ratio`` (default)
        Agent outputs an 8-dim continuous vector of phase-time ratios
        (Gaussian + softmax via MaskedSoftmax distribution). Ratios are
        masked, normalised, and converted to integer green times.

    ``cycle_level_continuous``
        Same translation pipeline as ``ratio``, but the policy uses a
        Dirichlet distribution directly on the 8-simplex. Sampled actions
        already satisfy ``sum=1`` by construction, which removes the need
        for a softmax layer and typically yields a lower-variance policy
        gradient. Invalid phases are masked to ~zero concentration.

    ``discrete_adjustment``
        Agent outputs an 8-dim vector with values in {0..6} per phase:
        0=-15s, 1=-10s, 2=-5s, 3=keep, 4=+5s, 5=+10s, 6=+15s.
        Invalid phases are forced to *keep* (action 3).

    Observation
    -----------
    Delegated to an ``ObservationFunction`` subclass (``DefaultObservation``,
    ``SpatioTemporal``, ``NeighborTemporal``).  The traffic signal exposes
    aggregated detector metrics and a green-time ratio vector for the
    observation function to compose.
    """

    def __init__(
        self,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: Optional[List[float]],
        data_provider: Any,
        num_green_phases: int,
        observation_class: type,
        *,
        detectors: Optional[List] = None,
        window_size: int = 1,
        phase_standardizer: Optional[Any] = None,
        use_phase_standardizer: bool = False,
        detectors_e2_length: Optional[Dict[str, float]] = None,
        neighbor_provider: Any = None,
        max_neighbors: int = 4,
        action_mode: str = "ratio",
        green_time_step: int = 5,
        enforce_max_green: bool = False,
        fixed_transition_time: Optional[float] = None,
    ):
        # Identity & timing
        self.id = ts_id
        self.data_provider = data_provider
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.num_green_phases = num_green_phases
        self.fixed_transition_time = fixed_transition_time
        self.window_size = window_size
        self.next_action_time = begin_time

        # Phase state
        self.green_phase = 0
        self.is_yellow = False

        # Cycle timing
        self.total_yellow_time = (
            float(self.fixed_transition_time)
            if self.fixed_transition_time is not None
            else self.yellow_time * self.num_green_phases
        )
        self.total_green_time = int(round(self.delta_time - self.total_yellow_time))

        # Action mode
        self.action_mode = action_mode
        self.green_time_step = green_time_step
        # Discrete adjustment steps: {-15, -10, -5, 0, +5, +10, +15}
        self.discrete_deltas = [-15, -10, -5, 0, +5, +10, +15]
        self.num_discrete_actions = len(self.discrete_deltas)

        # Current green times (for observation & discrete adjustment)
        self._init_equal_green_times()

        # Detectors
        detectors = detectors or [[], []]
        self.detectors_e1: List[str] = detectors[0]
        self.detectors_e2: List[str] = detectors[1]

        self.lanes = self.data_provider.get_controlled_lanes(self.id)

        if detectors_e2_length:
            self.detectors_e2_length = detectors_e2_length
        else:
            self.detectors_e2_length = {
                e2: self.data_provider.get_detector_length(e2) for e2 in self.detectors_e2
            }

        # Max vehicle capacity (for reward normalisation)
        self.avg_veh_length = DEFAULT_AVG_VEH_LEN
        self.max_veh: float = 0.0
        self._compute_max_veh()

        # Phase standardisation (FRAP)
        self.phase_standardizer = phase_standardizer
        self.use_phase_standardizer = use_phase_standardizer and phase_standardizer is not None
        if self.use_phase_standardizer and hasattr(self.phase_standardizer, "configure"):
            if not self.phase_standardizer._configured:
                self.phase_standardizer.configure()

        # --- Reward setup ---
        self.reward_weights = reward_weights
        self.last_reward: Optional[float] = None
        self.last_ts_waiting_time = 0.0
        self._setup_reward_fns(reward_fn)

        # --- Observation setup ---
        self.neighbor_provider = neighbor_provider
        self.max_neighbors = max_neighbors

        import inspect
        params = inspect.signature(observation_class.__init__).parameters
        if "neighbor_provider" in params:
            self.observation_fn = observation_class(
                self,
                neighbor_provider=neighbor_provider,
                max_neighbors=max_neighbors,
                window_size=window_size,
            )
        else:
            self.observation_fn = observation_class(self)

        self.observation_space = self.observation_fn.observation_space()

        # --- Action space ---
        if self.action_mode == "discrete_adjustment":
            # 7 choices per phase: {-15, -10, -5, 0, +5, +10, +15}
            self.action_space = spaces.MultiDiscrete(
                [self.num_discrete_actions] * NUM_STANDARD_PHASES, dtype=np.int64
            )
        else:
            # ratio and cycle_level_continuous share the same Box space [0, 1]^8.
            # The difference is how the policy produces the sample (Gaussian +
            # softmax vs. Dirichlet).
            self.action_space = spaces.Box(
                low=np.zeros(NUM_STANDARD_PHASES, dtype=np.float32),
                high=np.ones(NUM_STANDARD_PHASES, dtype=np.float32),
                dtype=np.float32,
            )

        assert (self.min_green * self.num_green_phases) <= self.total_green_time, (
            f"min_green too high for {self.id}: "
            f"{self.min_green}*{self.num_green_phases} > {self.total_green_time}"
        )
        if self.max_green and self.max_green > 0:
            assert (self.max_green * self.num_green_phases) >= self.total_green_time, (
                f"max_green too low for {self.id}: "
                f"{self.max_green}*{self.num_green_phases} < {self.total_green_time}"
            )

        # --- Detector history ---
        self.detector_history: Dict[str, Dict[str, list]] = {
            metric: {det: [] for det in self.detectors_e2}
            for metric in ("density", "queue", "occupancy", "average_speed")
        }
        self._last_sampling_time = -SAMPLING_INTERVAL_S

        # Observation history for spatio-temporal models
        self.observation_history: List[Any] = []
        self._max_obs_history = 50

        # Reward-metric history (cycle-level aggregation)
        self.reward_metrics_history: Dict[str, list] = {
            "halting_vehicles": [],
            "total_queued": [],
            "average_speed": [],
            "waiting_time": [],
        }

        # Vehicle tracking for throughput reward
        self._vehicles_at_cycle_start: set = set()
        self._vehicles_seen_this_cycle: set = set()
        self.initial_vehicles_this_cycle: int = 0
        self.departed_vehicles_this_cycle: int = 0

        # Teleport tracking
        self.teleported_vehicles_this_cycle: int = 0
        self._last_total_teleport: int = 0

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_equal_green_times(self) -> None:
        """Set initial green times to an equal distribution within [min, max]."""
        n = self.num_green_phases
        if n <= 0:
            self.current_green_times = []
            return

        base = self.total_green_time // n
        green_per_phase = max(self.min_green, base)
        if self.max_green and self.max_green > 0:
            green_per_phase = min(green_per_phase, self.max_green)
        self.current_green_times = [green_per_phase] * n

        # Distribute any remainder one second at a time, respecting max_green.
        remainder = self.total_green_time - sum(self.current_green_times)
        idx = 0
        while remainder > 0 and idx < 10 * n:
            i = idx % n
            if self.max_green is None or self.max_green <= 0 or self.current_green_times[i] < self.max_green:
                self.current_green_times[i] += 1
                remainder -= 1
            idx += 1

    def _setup_reward_fns(self, reward_fn: Union[str, Callable, List]) -> None:
        """Resolve reward function(s) from names or callables."""
        if isinstance(reward_fn, list):
            self.reward_list = [
                get_reward_fn(fn) if isinstance(fn, str) else fn
                for fn in reward_fn
            ]
        else:
            self.reward_list = [
                get_reward_fn(reward_fn) if isinstance(reward_fn, str) else reward_fn
            ]

        if self.reward_weights is not None:
            self.reward_dim = 1  # scalarised
        else:
            self.reward_dim = len(self.reward_list)

        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32
        )

    def _compute_max_veh(self) -> None:
        """Compute maximum vehicle capacity across all E2 detectors."""
        self.max_veh = 0.0
        for det_id in self.detectors_e2:
            length = self.detectors_e2_length.get(det_id, 0)
            if length > 0:
                self.max_veh += length / (MIN_GAP + self.avg_veh_length)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @property
    def time_to_act(self) -> bool:
        return self.data_provider.should_act(self.id, self.next_action_time)

    def update_timing(self) -> None:
        """Advance next-action time without changing phase (fixed-ts mode)."""
        self.next_action_time = self.data_provider.get_sim_time() + self.delta_time
        self.update_cycle_vehicle_tracking()

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def set_next_phase(self, action: np.ndarray) -> None:
        """Apply an action and schedule the next decision time.

        Args:
            action: Either an 8-dim ratio vector or an 8-dim discrete vector
                    depending on ``self.action_mode``.
        """
        action = np.asarray(action)

        if self.action_mode == "discrete_adjustment":
            # _apply_discrete_adjustment updates self.current_green_times internally
            self.green_times = self._apply_discrete_adjustment(action)
        else:
            # Both "ratio" and "cycle_level_continuous" feed into the same
            # ratio-to-green-time translator. A Dirichlet sample is already on
            # the simplex, but _get_green_time_from_ratio safely re-normalises,
            # so sharing the code path is fine.
            self.green_times = self._get_green_time_from_ratio(action)
            # Persist for observation (green-time ratio features)
            self.current_green_times = list(self.green_times)

        self.data_provider.set_traffic_light_phase(self.id, self.green_times)
        self.next_action_time = self.data_provider.get_sim_time() + self.delta_time
        self.update_cycle_vehicle_tracking()

    # -- Ratio mode --

    def _get_green_time_from_ratio(self, ratios: np.ndarray) -> List[int]:
        """Convert 8 standard-phase ratios → actual green times.

        Pipeline:
        1. Mask invalid phases.
        2. Normalise to sum=1.
        3. FRAP conversion (8 standard → actual phases).
        4. Enforce min_green; distribute remaining time.
        """
        std_ratios = np.array(ratios, dtype=float, copy=True)

        # Pad / truncate to 8
        if len(std_ratios) < NUM_STANDARD_PHASES:
            padded = np.zeros(NUM_STANDARD_PHASES)
            padded[: len(std_ratios)] = std_ratios
            std_ratios = padded
        elif len(std_ratios) > NUM_STANDARD_PHASES:
            std_ratios = std_ratios[:NUM_STANDARD_PHASES]

        # 1. Mask
        mask = self.get_action_mask()
        std_ratios *= mask

        # 2. Normalise
        total = std_ratios.sum()
        if total == 0:
            valid = mask / mask.sum() if mask.sum() > 0 else np.array([0.5, 0.5] + [0] * 6)
            std_ratios = valid
        else:
            std_ratios /= total

        # 3. FRAP → actual ratios
        if self.use_phase_standardizer and self.phase_standardizer is not None:
            actual_ratios = self.phase_standardizer.standardize_action(std_ratios)
        else:
            actual_ratios = np.zeros(self.num_green_phases)
            for i in range(min(self.num_green_phases, NUM_STANDARD_PHASES)):
                actual_ratios[i] = std_ratios[i]
            s = actual_ratios.sum()
            actual_ratios = actual_ratios / s if s > 0 else np.ones(self.num_green_phases) / self.num_green_phases

        # 4. Distribute green time subject to [min_green, max_green]
        min_total = self.min_green * self.num_green_phases
        remaining = self.total_green_time - min_total
        if remaining < 0:
            return [self.min_green] * self.num_green_phases

        green_times = self.min_green + actual_ratios * remaining
        int_gt = np.floor(green_times).astype(int)

        # Distribute rounding deficit to phases with the largest fractional part
        deficit = int(self.total_green_time - int_gt.sum())
        if deficit > 0:
            frac = green_times - int_gt
            for idx in np.argsort(frac)[::-1][:deficit]:
                int_gt[idx] += 1

        # Enforce [min_green, max_green] and rebalance to keep cycle total
        min_g = int(self.min_green)
        max_g = int(self.max_green) if self.max_green and self.max_green > 0 else None
        int_gt = np.maximum(int_gt, min_g)
        if max_g is not None:
            int_gt = np.minimum(int_gt, max_g)

        diff = int(self.total_green_time - int_gt.sum())
        guard = 0
        max_iter = 4 * self.num_green_phases * max(1, max_g if max_g is not None else self.total_green_time)
        while diff != 0 and guard < max_iter:
            if diff > 0:
                order = np.argsort(int_gt)
                added = False
                for idx in order:
                    if max_g is None or int_gt[idx] < max_g:
                        int_gt[idx] += 1
                        diff -= 1
                        added = True
                        break
                if not added:
                    break
            else:
                order = np.argsort(-int_gt)
                removed = False
                for idx in order:
                    if int_gt[idx] > min_g:
                        int_gt[idx] -= 1
                        diff += 1
                        removed = True
                        break
                if not removed:
                    break
            guard += 1

        return int_gt.tolist()

    # -- Discrete adjustment mode --

    def _apply_discrete_adjustment(self, action: np.ndarray) -> List[int]:
        """Apply integer second-deltas to the current cycle's green times.

        Key invariants preserved here:
          * The *order* of green phases is never changed (we only edit durations).
          * The *cycle length* is never changed: sum(new_green) == total_green_time.
          * Each green duration stays in [min_green, max_green].
          * Outputs remain integer seconds (no float rescaling that would turn
            {±5, ±10, ±15} deltas into odd numbers).

        Pipeline:
          1. Convert agent's discrete indices → integer second deltas per
             *standard* phase; force invalid phases to keep (delta = 0).
          2. Project 8 standard deltas → num_green_phases *actual* deltas.
             We pick one representative standard phase per actual phase (via
             ``standard_to_actual``) so a single standard action maps to
             exactly one actual phase — this avoids the double-apply bug
             where two actual phases share the same standard index.
          3. Re-center deltas so they sum to 0 (integer-safe). This keeps
             the cycle length constant without any float rescaling.
          4. Apply deltas, clip to [min_green, max_green], then redistribute
             any shortfall/excess one second at a time among phases that
             still have headroom.
        """
        action = np.asarray(action, dtype=int)
        keep_action = self.num_discrete_actions // 2  # middle index = delta 0

        if len(action) < NUM_STANDARD_PHASES:
            padded = np.full(NUM_STANDARD_PHASES, keep_action, dtype=int)
            padded[: len(action)] = action
            action = padded

        # ---- 1. Standard-phase integer deltas, masked ----
        mask = self.get_action_mask().astype(bool)
        deltas_std = np.zeros(NUM_STANDARD_PHASES, dtype=int)
        for i in range(NUM_STANDARD_PHASES):
            if mask[i]:
                idx = int(np.clip(action[i], 0, self.num_discrete_actions - 1))
                deltas_std[i] = self.discrete_deltas[idx]

        # ---- 2. Standard → actual, one-to-one via standard_to_actual ----
        deltas_actual = np.zeros(self.num_green_phases, dtype=int)
        std_to_actual: Dict[int, int] = {}
        if self.use_phase_standardizer and self.phase_standardizer is not None:
            std_to_actual = getattr(self.phase_standardizer, "standard_to_actual", {}) or {}

        if std_to_actual:
            for std_idx, actual_idx in std_to_actual.items():
                if 0 <= actual_idx < self.num_green_phases and 0 <= std_idx < NUM_STANDARD_PHASES:
                    deltas_actual[actual_idx] = deltas_std[std_idx]
        else:
            # Fallback: identity mapping for the first num_green_phases entries
            n = min(self.num_green_phases, NUM_STANDARD_PHASES)
            deltas_actual[:n] = deltas_std[:n]

        # ---- 3. Re-center so sum(deltas) == 0 (integer-safe) ----
        total_delta = int(deltas_actual.sum())
        if total_delta != 0:
            n = self.num_green_phases
            share = total_delta // n
            deltas_actual -= share
            # Distribute the remainder (sign-aware) across phases with the
            # largest/smallest deltas so we nudge the extremes first.
            residual = int(deltas_actual.sum())
            if residual != 0:
                step = 1 if residual > 0 else -1
                # Order: for positive residual, take from the biggest deltas;
                # for negative, add to the smallest.
                order = np.argsort(-deltas_actual) if residual > 0 else np.argsort(deltas_actual)
                for k in range(abs(residual)):
                    deltas_actual[order[k % n]] -= step

        # ---- 4. Apply, clip to [min_green, max_green], rebalance to keep total ----
        new_green = np.array(self.current_green_times, dtype=int) + deltas_actual
        min_g = int(self.min_green)
        max_g = int(self.max_green) if self.max_green and self.max_green > 0 else None

        new_green = np.maximum(new_green, min_g)
        if max_g is not None:
            new_green = np.minimum(new_green, max_g)

        # After clipping, the total may drift. Redistribute 1s at a time on
        # phases that still have headroom, preserving integer granularity.
        diff = int(self.total_green_time - new_green.sum())
        if diff != 0:
            n = self.num_green_phases
            guard = 0
            max_iter = 4 * n * (max(1, max_g if max_g is not None else self.total_green_time))
            while diff != 0 and guard < max_iter:
                if diff > 0:
                    # Need to add seconds → pick phase with smallest value that can still grow
                    order = np.argsort(new_green)
                    added = False
                    for idx in order:
                        if max_g is None or new_green[idx] < max_g:
                            new_green[idx] += 1
                            diff -= 1
                            added = True
                            break
                    if not added:
                        break  # Nowhere to add — stop gracefully
                else:
                    # Need to remove seconds → pick phase with largest value above min
                    order = np.argsort(-new_green)
                    removed = False
                    for idx in order:
                        if new_green[idx] > min_g:
                            new_green[idx] -= 1
                            diff += 1
                            removed = True
                            break
                    if not removed:
                        break
                guard += 1

        result = new_green.astype(int).tolist()
        # Persist for next cycle's baseline and for observation features.
        self.current_green_times = result
        return result

    def _actual_to_standard_greens(self) -> List[float]:
        """Map current actual green times back to 8 standard phases.

        Uses FRAP's ``standardize_action`` in reverse: distributes actual
        green times proportionally across the standard phases that map to
        each actual phase.  Falls back to a 1:1 mapping when no FRAP
        module is present.
        """
        std = [0.0] * NUM_STANDARD_PHASES
        if self.use_phase_standardizer and self.phase_standardizer is not None:
            # Build inverse: for each actual phase, find which standard phases map to it.
            # We do this by probing standardize_action with one-hot vectors.
            mask = self.get_action_mask()
            for i in range(NUM_STANDARD_PHASES):
                if mask[i] < 0.5:
                    continue
                # Probe: all weight on standard phase i
                probe = np.zeros(NUM_STANDARD_PHASES)
                probe[i] = 1.0
                actual_ratios = self.phase_standardizer.standardize_action(probe)
                # Find which actual phase(s) got non-zero ratio
                for j, r in enumerate(actual_ratios):
                    if r > 0.01 and j < len(self.current_green_times):
                        std[i] = float(self.current_green_times[j])
                        break
                else:
                    std[i] = float(self.min_green)
        else:
            for i in range(min(self.num_green_phases, NUM_STANDARD_PHASES)):
                std[i] = float(self.current_green_times[i])
        return std

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """Binary mask [8] for valid standard phases."""
        if self.use_phase_standardizer and self.phase_standardizer is not None:
            return self.phase_standardizer.get_phase_mask()
        return np.ones(NUM_STANDARD_PHASES, dtype=np.float32)

    # ------------------------------------------------------------------
    # Green-time ratio features (for observation)
    # ------------------------------------------------------------------

    def get_green_time_ratios(self) -> np.ndarray:
        """Return normalised green-time ratios [8] for the 8 standard phases.

        Each value is ``current_green / max_green``, clipped to [0, 1].
        Invalid phases are 0.
        """
        std_greens = self._actual_to_standard_greens()
        mask = self.get_action_mask()
        ratios = np.zeros(NUM_STANDARD_PHASES, dtype=np.float32)
        for i in range(NUM_STANDARD_PHASES):
            if mask[i] > 0.5 and self.max_green > 0:
                ratios[i] = np.clip(std_greens[i] / self.max_green, 0.0, 1.0)
        return ratios

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def compute_observation(self) -> Any:
        """Compute and store observation, return validated result."""
        if hasattr(self.observation_fn, "compute_current_observation"):
            current_obs = self.observation_fn.compute_current_observation()
            self._push_history(current_obs)
            obs = self.observation_fn()
        else:
            obs = self.observation_fn()
            self._push_history(obs)

        return self._validate_observation(obs)

    def _push_history(self, obs: Any) -> None:
        self.observation_history.append(obs)
        if len(self.observation_history) > self._max_obs_history:
            self.observation_history.pop(0)

    def _validate_observation(self, obs: Any) -> Any:
        """Clip observation values to declared space bounds."""
        if isinstance(obs, dict):
            return {k: self._clip_to_space(v, k) for k, v in obs.items()}
        return self._clip_to_space(obs)

    def _clip_to_space(self, value: Any, key: str = None) -> np.ndarray:
        try:
            arr = np.asarray(value, dtype=np.float32)
        except (ValueError, TypeError):
            return value

        space = self.observation_space
        if key and isinstance(space, spaces.Dict) and key in space.spaces:
            space = space.spaces[key]

        if hasattr(space, "low") and hasattr(space, "high"):
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            if arr.shape == low.shape:
                arr = np.clip(arr, low, high)
        return arr

    def get_observation_history(self, window_size: int) -> List[Any]:
        """Return the last *window_size* observations, zero-padded if needed."""
        if not self.observation_history:
            if hasattr(self.observation_fn, "compute_current_observation"):
                obs_dim = 4 * len(self.detectors_e2) + NUM_STANDARD_PHASES
                default = np.zeros(obs_dim, dtype=np.float32)
            elif isinstance(self.observation_space, spaces.Dict):
                default = {
                    k: np.zeros(sp.shape, dtype=np.float32)
                    for k, sp in self.observation_space.spaces.items()
                    if hasattr(sp, "shape")
                }
            elif hasattr(self.observation_space, "shape"):
                default = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                default = np.zeros(56, dtype=np.float32)
            return [default] * window_size

        history = [self._validate_observation(o) for o in self.observation_history]

        if len(history) < window_size:
            pad = copy.deepcopy(history[0])
            if isinstance(pad, dict):
                pad = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else v for k, v in pad.items()}
            elif isinstance(pad, np.ndarray):
                pad = np.zeros_like(pad)
            history = [pad] * (window_size - len(history)) + history

        return history[-window_size:]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self) -> Union[float, np.ndarray]:
        """Compute the scalar reward for this cycle."""
        values = []
        for fn in self.reward_list:
            v = float(fn(self))
            if not np.isfinite(v):
                v = 0.0
            values.append(v)

        if len(values) == 1:
            self.last_reward = values[0]
        else:
            arr = np.array(values, dtype=np.float32)
            if self.reward_weights is not None:
                self.last_reward = float(np.dot(arr, self.reward_weights))
            else:
                self.last_reward = arr

        if isinstance(self.last_reward, float) and not np.isfinite(self.last_reward):
            self.last_reward = 0.0

        return self.last_reward

    # ------------------------------------------------------------------
    # Detector history & metrics
    # ------------------------------------------------------------------

    def update_detectors_history(self) -> None:
        """Sample detector metrics every ``SAMPLING_INTERVAL_S`` seconds."""
        current_time = self.data_provider.get_sim_time()
        if current_time - self._last_sampling_time < SAMPLING_INTERVAL_S - 0.1:
            return
        self._last_sampling_time = current_time

        metric_fns = [
            ("density", self._compute_detector_density),
            ("queue", self._compute_detector_queue),
            ("occupancy", self._compute_detector_occupancy),
            ("average_speed", self._compute_detector_average_speed),
        ]
        for det_id in self.detectors_e2:
            for name, fn in metric_fns:
                self._append_history(self.detector_history[name][det_id], fn(det_id))

        reward_fns = [
            ("halting_vehicles", self._get_total_halting_instant),
            ("total_queued", self._get_total_queued_instant),
            ("average_speed", self._get_average_speed_instant),
            ("waiting_time", self._get_waiting_time_from_detectors),
        ]
        for name, fn in reward_fns:
            self._append_history(self.reward_metrics_history[name], fn())

    @staticmethod
    def _append_history(buf: list, value: float, max_size: int = MAX_HISTORY_SAMPLES) -> None:
        buf.append(value)
        if len(buf) > max_size:
            del buf[:-max_size]

    # -- Individual detector metrics --

    def _compute_detector_density(self, det_id: str) -> float:
        try:
            count = self.data_provider.get_detector_vehicle_count(det_id)
            if count == 0:
                return 0.0
            length = self.data_provider.get_detector_length(det_id)
            if length <= 0:
                return 0.0
            ids = self.data_provider.get_detector_vehicle_ids(det_id)
            if ids:
                avg_len = sum(self.data_provider.get_vehicle_length(v) for v in ids) / len(ids)
            else:
                avg_len = 5.0
            cap = length / (MIN_GAP + avg_len)
            return min(1.0, count / cap)
        except Exception:
            return 0.0

    def _compute_detector_queue(self, det_id: str) -> float:
        try:
            jam = self.data_provider.get_detector_jam_length(det_id)
            if jam == 0:
                return 0.0
            length = self.data_provider.get_detector_length(det_id)
            return min(1.0, jam / length) if length > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_detector_occupancy(self, det_id: str) -> float:
        try:
            occ = self.data_provider.get_detector_occupancy(det_id)
            return np.clip(occ / 100.0, 0.0, 1.0)
        except Exception:
            return 0.0

    def _compute_detector_average_speed(self, det_id: str) -> float:
        try:
            speed = self.data_provider.get_detector_mean_speed(det_id)
            if speed <= 0:
                return 0.0
            lane_id = self.data_provider.get_detector_lane_id(det_id)
            max_speed = self.data_provider.get_lane_max_speed(lane_id)
            return min(1.0, speed / max_speed) if max_speed > 0 else 1.0
        except Exception:
            return 1.0

    # -- Instant reward-metric helpers --

    def _get_total_halting_instant(self) -> float:
        return float(self.get_total_halting_veh_by_detectors())

    def _get_total_queued_instant(self) -> float:
        return float(self.get_total_queued())

    def _get_average_speed_instant(self) -> float:
        total, count = 0.0, 0
        for det_id in self.detectors_e2:
            try:
                speed = self.data_provider.get_detector_mean_speed(det_id)
                if speed >= 0:
                    lane_id = self.data_provider.get_detector_lane_id(det_id)
                    mx = self.data_provider.get_lane_max_speed(lane_id)
                    if mx > 0:
                        total += speed / mx
                        count += 1
            except Exception:
                pass
        return min(1.0, total / count) if count else 1.0

    def _get_waiting_time_from_detectors(self) -> float:
        total = 0.0
        for det_id in self.detectors_e2:
            try:
                jam = self.data_provider.get_detector_jam_length(det_id)
                if jam > 0:
                    total += (jam / (MIN_GAP + self.avg_veh_length)) * SAMPLING_INTERVAL_S
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    # Aggregated metrics (cycle-mean)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_mean(values: list, fallback: float = 0.0) -> float:
        if not values:
            return fallback
        arr = np.array(values, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        return float(np.mean(valid)) if len(valid) else fallback

    def get_aggregated_halting_vehicles(self) -> float:
        h = self.reward_metrics_history.get("halting_vehicles", [])
        return self._safe_mean(h) if h else float(self.get_total_halting_veh_by_detectors())

    def get_aggregated_queued(self) -> float:
        h = self.reward_metrics_history.get("total_queued", [])
        return self._safe_mean(h) if h else float(self.get_total_queued())

    def get_aggregated_occupancy(self) -> float:
        total, count = 0.0, 0
        for det_id in self.detectors_e2:
            hist = self.detector_history.get("occupancy", {}).get(det_id, [])
            if hist:
                total += self._safe_mean(hist)
                count += 1
        if count:
            return min(1.0, total / count)
        return self._get_occupancy_instant()

    def get_aggregated_average_speed(self) -> float:
        h = self.reward_metrics_history.get("average_speed", [])
        val = self._safe_mean(h, fallback=-1.0) if h else -1.0
        return val if val >= 0 else self._get_average_speed_instant()

    def get_aggregated_waiting_time(self) -> float:
        h = self.reward_metrics_history.get("waiting_time", [])
        val = self._safe_mean(h, fallback=-1.0) if h else -1.0
        return val if val >= 0 else float(sum(self.get_accumulated_waiting_time_per_lane()))

    def _get_occupancy_instant(self) -> float:
        total, count = 0.0, 0
        for det_id in self.detectors_e2:
            try:
                total += self.data_provider.get_detector_occupancy(det_id) / 100.0
                count += 1
            except Exception:
                pass
        return min(1.0, total / count) if count else 0.0

    # ------------------------------------------------------------------
    # Lane-level getters (for observation functions)
    # ------------------------------------------------------------------

    def get_lanes_density_by_detectors(self) -> List[float]:
        return self._get_detector_metric("density")

    def get_lanes_queue_by_detectors(self) -> List[float]:
        return self._get_detector_metric("queue")

    def get_lanes_occupancy_by_detectors(self) -> List[float]:
        return self._get_detector_metric("occupancy")

    def get_lanes_average_speed_by_detectors(self) -> List[float]:
        return self._get_detector_metric("average_speed", fallback=1.0)

    def _get_detector_metric(self, metric: str, fallback: float = 0.0) -> List[float]:
        result = []
        for det_id in self.detectors_e2:
            hist = self.detector_history[metric].get(det_id, [])
            if hist:
                result.append(float(np.clip(self._safe_mean(hist, fallback), 0.0, 1.0)))
            else:
                result.append(fallback)
        return result

    # ------------------------------------------------------------------
    # Vehicle / pressure helpers
    # ------------------------------------------------------------------

    def get_current_vehicle_count(self) -> int:
        total = 0
        for det_id in self.detectors_e2:
            try:
                total += self.data_provider.get_detector_vehicle_count(det_id)
            except Exception:
                pass
        return total

    def get_total_queued(self) -> int:
        total = 0
        for det_id in self.detectors_e2:
            try:
                total += self.data_provider.get_detector_halting_number(det_id)
            except Exception:
                pass
        return total

    def get_total_halting_veh_by_detectors(self) -> int:
        total = 0
        for det_id in self.detectors_e2:
            try:
                jam = self.data_provider.get_detector_jam_length(det_id)
                if jam > 0:
                    total += jam / (MIN_GAP + self.avg_veh_length)
            except Exception:
                pass
        return int(total)

    def get_pressure_from_detectors(self) -> float:
        if self.max_veh == 0:
            return 0.0
        total_occ, total_spd, count = 0.0, 0.0, 0
        for det_id in self.detectors_e2:
            try:
                occ = self.data_provider.get_detector_occupancy(det_id) / 100.0
                speed = self.data_provider.get_detector_mean_speed(det_id)
                lane_id = self.data_provider.get_detector_lane_id(det_id)
                mx = self.data_provider.get_lane_max_speed(lane_id)
                norm_spd = min(1.0, speed / mx) if mx > 0 else 1.0
                total_occ += occ
                total_spd += norm_spd
                count += 1
            except Exception:
                pass
        if count == 0:
            return 0.0
        return np.clip((total_occ - total_spd) / count, -1.0, 1.0)

    def get_average_speed(self) -> float:
        vehs = []
        for lane in self.lanes:
            vehs.extend(self.data_provider.get_lane_vehicles(lane))
        if not vehs:
            return 1.0
        total = 0.0
        for v in vehs:
            s = self.data_provider.get_vehicle_speed(v)
            a = self.data_provider.get_vehicle_allowed_speed(v)
            total += s / a if a > 0 else 0.0
        return total / len(vehs)

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        result = []
        for lane in self.lanes:
            wt = 0.0
            for veh in self.data_provider.get_lane_vehicles(lane):
                wt += self.data_provider.get_vehicle_waiting_time(veh, lane)
            result.append(wt)
        return result

    # ------------------------------------------------------------------
    # Cycle vehicle tracking
    # ------------------------------------------------------------------

    def update_cycle_vehicle_tracking(self) -> None:
        """Update departed-vehicle counters at cycle boundary."""
        current = self._get_current_vehicle_ids()

        if self._vehicles_seen_this_cycle:
            left = self._vehicles_seen_this_cycle - current
            self.departed_vehicles_this_cycle = len(left)
            self.initial_vehicles_this_cycle = len(self._vehicles_at_cycle_start)
        else:
            self.departed_vehicles_this_cycle = 0
            self.initial_vehicles_this_cycle = 0

        self._vehicles_at_cycle_start = current.copy()
        self._vehicles_seen_this_cycle = current.copy()

        for key in self.reward_metrics_history:
            self.reward_metrics_history[key] = []

        self._update_teleport_tracking()

    def update_departed_vehicles(self) -> None:
        """Accumulate vehicle IDs seen during this cycle (call every sim step)."""
        self._vehicles_seen_this_cycle.update(self._get_current_vehicle_ids())

    def _get_current_vehicle_ids(self) -> set:
        ids: set = set()
        for det_id in self.detectors_e2:
            try:
                ids.update(self.data_provider.get_detector_vehicle_ids(det_id))
            except Exception:
                pass
        return ids

    def _update_teleport_tracking(self) -> None:
        try:
            total = self.data_provider.get_total_teleport_count()
            self.teleported_vehicles_this_cycle = max(0, total - self._last_total_teleport)
            self._last_total_teleport = total
        except Exception:
            self.teleported_vehicles_this_cycle = 0
