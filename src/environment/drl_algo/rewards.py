"""Reward functions for traffic signal control.

Each reward function takes a TrafficSignal instance and returns a scalar reward.
All rewards are clipped to a consistent range for stable training.

Registry pattern allows easy addition of new reward functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

import numpy as np

if TYPE_CHECKING:
    from .traffic_signal import TrafficSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(value: float, low: float = -3.0, high: float = 3.0) -> float:
    """Clip reward to valid range."""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Reward functions  (all accept a single TrafficSignal, return float)
# ---------------------------------------------------------------------------

def pressure_reward(ts: "TrafficSignal") -> float:
    """Pressure-based reward.  Range: [-3, 3].

    Positive pressure (congestion) → negative reward.
    """
    if ts.get_current_vehicle_count() == 0:
        return 0.0
    pressure = ts.get_pressure_from_detectors()
    return _clip(-pressure * 3.0)


def average_speed_reward(ts: "TrafficSignal") -> float:
    """Average speed reward.  Range: [-3, 3].

    Maps normalised speed [0, 1] → reward [-3, 3].
    """
    if ts.get_current_vehicle_count() == 0:
        return 0.0
    avg_speed = ts.get_aggregated_average_speed()
    return _clip(avg_speed * 6.0 - 3.0)


def queue_reward(ts: "TrafficSignal") -> float:
    """Queue-length penalty.  Range: [-3, 0].

    Fewer queued vehicles → higher (less negative) reward.
    """
    if ts.get_current_vehicle_count() == 0:
        return 0.0
    total_queued = ts.get_aggregated_queued()
    if ts.max_veh == 0:
        return 0.0
    return _clip(-(total_queued / ts.max_veh) * 3.0, low=-3.0, high=0.0)


def occupancy_reward(ts: "TrafficSignal") -> float:
    """Occupancy penalty.  Range: [-3, 0]."""
    if ts.get_current_vehicle_count() == 0:
        return 0.0
    avg_occ = ts.get_aggregated_occupancy()
    return _clip(-avg_occ * 3.0, low=-3.0, high=0.0)


def diff_waiting_time_reward(ts: "TrafficSignal") -> float:
    """Normalised difference in waiting time.  Range: [-3, 3].

    Positive when waiting time decreases (good), negative when it increases.
    """
    if ts.get_current_vehicle_count() == 0:
        ts.last_ts_waiting_time = 0.0
        return 0.0

    ts_wait = ts.get_aggregated_waiting_time()
    reward = ts.last_ts_waiting_time - ts_wait
    ts.last_ts_waiting_time = ts_wait

    if ts.max_veh > 0 and ts.delta_time > 0:
        max_change = ts.max_veh * ts.delta_time
        normalised = (reward / max_change) * 3.0
    else:
        normalised = 0.0
    return _clip(normalised)


def halt_veh_reward(ts: "TrafficSignal") -> float:
    """Halting-vehicle penalty.  Range: [-3, 0]."""
    if ts.get_current_vehicle_count() == 0:
        return 0.0
    if ts.max_veh == 0:
        return 0.0
    total_halt = ts.get_aggregated_halting_vehicles()
    ratio = min(1.0, total_halt / ts.max_veh)
    return -3.0 * float(ratio)


def throughput_reward(ts: "TrafficSignal") -> float:
    """Throughput (departed vehicles) reward.  Range: [0, 3].

    Measures the proportion of initial vehicles that departed during the cycle.
    """
    if ts.get_current_vehicle_count() == 0:
        return 3.0

    initial = float(ts.initial_vehicles_this_cycle)
    departed = float(ts.departed_vehicles_this_cycle)

    MIN_THRESHOLD = 1.0
    if initial >= MIN_THRESHOLD:
        ratio = departed / initial
    elif departed >= MIN_THRESHOLD:
        ratio = 0.5
    else:
        return 0.0

    return _clip(ratio * 3.0, low=0.0, high=3.0)


def teleport_penalty_reward(ts: "TrafficSignal") -> float:
    """Penalty for teleported vehicles.  Range: [-3, 0]."""
    if ts.get_current_vehicle_count() == 0:
        return 0.0
    if ts.max_veh == 0:
        return 0.0
    teleported = float(ts.teleported_vehicles_this_cycle)
    if teleported == 0:
        return 0.0
    ratio = min(1.0, teleported / (ts.max_veh * 0.1))
    return -3.0 * ratio


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_REGISTRY: Dict[str, Callable[["TrafficSignal"], float]] = {
    "pressure": pressure_reward,
    "average-speed": average_speed_reward,
    "queue": queue_reward,
    "occupancy": occupancy_reward,
    "diff-waiting-time": diff_waiting_time_reward,
    "halt-veh-by-detectors": halt_veh_reward,
    "throughput": throughput_reward,
    "diff-departed-veh": throughput_reward,  # alias for backward compat
    "teleport-penalty": teleport_penalty_reward,
}


def get_reward_fn(name: str) -> Callable[["TrafficSignal"], float]:
    """Look up a reward function by name.

    Raises:
        KeyError: if *name* is not registered.
    """
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward function '{name}'. "
            f"Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]


def register_reward_fn(name: str, fn: Callable[["TrafficSignal"], float]) -> None:
    """Register a custom reward function."""
    REWARD_REGISTRY[name] = fn
