"""Helpers cho sim trainer (SUMO) để map detector output → vars chuẩn.

Sim training pipeline có detector e1 (point) và e2 (lane area) từ SUMO. Để
distribution observation match runtime, **sim trainer phải dùng cùng formula
và cùng tập biến** mà runtime dùng.

Hàm `sumo_detector_to_road_vars` convert detector reading + lane metadata
sang vars `(occupancy, speed)` mà formula expect. Static vars (lanes, length,
...) lấy từ network XML của SUMO.

Service-ai (runtime) KHÔNG cần import module này — chỉ sim trainer dùng.
"""

from __future__ import annotations

from typing import Optional


def sumo_detector_to_road_vars(
    e1_occupancy_percent: Optional[float] = None,
    e2_occupancy_percent: Optional[float] = None,
    e1_mean_speed_ms: Optional[float] = None,
) -> dict[str, float]:
    """Convert SUMO detector reading sang dict {occupancy, speed} mà formula expect.

    Args:
        e1_occupancy_percent: e1 detector occupancy (%), thường đo gần đèn.
                              Nếu None, fallback e2.
        e2_occupancy_percent: e2 detector occupancy (%), đo cả lane area.
        e1_mean_speed_ms: e1 detector mean speed (m/s). Convert sang km/h.

    Returns:
        dict với keys 'occupancy' (% [0, 100]) và 'speed' (km/h).
        Đây là cùng format mà real API gửi runtime — formula áp lên 2 dict
        này phải ra cùng output.
    """
    # Occupancy: ưu tiên e1 (gần đèn, đại diện queue tốt hơn), fallback e2.
    occ = e1_occupancy_percent if e1_occupancy_percent is not None else e2_occupancy_percent
    occ_val = float(occ) if occ is not None else 0.0
    # Clamp về [0, 100] để khớp ràng buộc real API.
    occ_val = max(0.0, min(100.0, occ_val))

    # Speed: m/s -> km/h.
    speed_kmh = (float(e1_mean_speed_ms) * 3.6) if e1_mean_speed_ms is not None else 0.0
    speed_kmh = max(0.0, speed_kmh)

    return {"occupancy": occ_val, "speed": speed_kmh}


def road_static_from_sumo_lane(
    num_lanes: int,
    lane_length_meters: float,
    speed_limit_ms: Optional[float] = None,
    saturation_flow_vph: Optional[float] = None,
) -> dict[str, float]:
    """Build dict roads_static entry từ SUMO network info.

    Sim trainer call hàm này 1 lần lúc init env, output dùng để init FeatureBuilder.
    """
    out: dict[str, float] = {
        "lanes": int(num_lanes),
        "length_meters": float(lane_length_meters),
    }
    if speed_limit_ms is not None:
        out["speed_design_kmh"] = float(speed_limit_ms) * 3.6
    if saturation_flow_vph is not None:
        out["saturation_flow"] = float(saturation_flow_vph)
    else:
        # Heuristic chuẩn HCM: 1800 veh/h/lane.
        out["saturation_flow"] = float(num_lanes) * 1800.0
    return out
