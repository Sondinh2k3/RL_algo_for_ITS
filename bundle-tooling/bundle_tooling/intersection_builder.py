"""Translate sim's intersection_config.json + deployment_map.json → bundle files.

Output:
  - `intersections/cross_<real_cross_id>.json` cho mỗi cross
  - `network.json` với real IDs + neighbor đã translate
  - `feature_formula.json` (copy từ deployment_map.feature_formula)

Bundle config keyed bằng **real DB IDs** — runtime không bao giờ thấy sim IDs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bundle_tooling.deployment_map import (
    CrossMapping,
    DeploymentMap,
)


NUM_DIRECTIONS = 4
LANES_PER_DIRECTION = 3
TOTAL_LANES = NUM_DIRECTIONS * LANES_PER_DIRECTION  # 12

# Sim direction string → standardized direction index 0..3.
# Khớp với DB convention (1=N → 0, 2=E → 1, 3=S → 2, 4=W → 3).
_SIM_DIR_TO_IDX: Dict[str, int] = {"N": 0, "E": 1, "S": 2, "W": 3}


def _direction_idx(direction: str) -> int:
    idx = _SIM_DIR_TO_IDX.get(direction)
    if idx is None:
        raise ValueError(f"Hướng không hợp lệ: {direction!r}")
    return idx


def _build_observation_masks(
    cm: CrossMapping, sim_cross: dict
) -> tuple[List[int], List[int]]:
    """Build 2 mask: per-direction (4-dim) + per-lane (12-dim)."""
    mask_dir = [0, 0, 0, 0]
    mask_lane = [0] * TOTAL_LANES

    for direction, road in cm.roads_by_direction.items():
        if road is None:
            continue
        d_idx = _direction_idx(direction)
        mask_dir[d_idx] = 1
        n_lanes_with_data = min(int(road.real_lanes), LANES_PER_DIRECTION)
        for lane_offset in range(n_lanes_with_data):
            mask_lane[d_idx * LANES_PER_DIRECTION + lane_offset] = 1

    return mask_dir, mask_lane


def _build_cycles_section(cm: CrossMapping) -> tuple[int, Dict[str, dict]]:
    """Build mapping cycles → stage info. Trả (primary_cycle_id, cycles_dict)."""
    cycles_dict: Dict[str, dict] = {}
    primary_id: Optional[int] = None

    for cy in cm.cycles:
        stage_to_std: Dict[str, int] = {}
        std_to_stage: Dict[str, int] = {}
        for mapping in cy.phase_to_stage:
            stage_to_std[str(mapping.real_stage_id)] = mapping.std_phase_idx
            std_to_stage[str(mapping.std_phase_idx)] = mapping.real_stage_id

        cycles_dict[str(cy.real_cycle_id)] = {
            "is_primary": cy.is_primary,
            "stage_to_standard_phase": stage_to_std,
            "standard_phase_to_stage": std_to_stage,
            "num_stages": len(cy.phase_to_stage),
        }
        if cy.is_primary:
            primary_id = cy.real_cycle_id

    if primary_id is None:
        raise ValueError(
            f"Cross sim_tls_id={cm.sim_tls_id}: không có cycle primary."
        )

    return primary_id, cycles_dict


def build_cross_config(cm: CrossMapping, sim_cross: dict) -> dict:
    """Build `intersections/cross_<real_cross_id>.json` cho 1 cross."""
    direction_map: Dict[str, int] = {}
    lanes_per_dir: Dict[str, int] = {str(i): 0 for i in range(NUM_DIRECTIONS)}
    roads_static: Dict[str, dict] = {}

    for direction, road in cm.roads_by_direction.items():
        if road is None:
            continue
        d_idx = _direction_idx(direction)
        rid_key = str(road.real_road_id)
        direction_map[rid_key] = d_idx
        lanes_per_dir[str(d_idx)] = int(road.real_lanes)
        roads_static[rid_key] = {
            "lanes": int(road.real_lanes),
            "length_meters": road.length_meters,
            "speed_design_kmh": road.speed_design_kmh,
            "saturation_flow": road.saturation_flow,
        }

    mask_dir, mask_lane = _build_observation_masks(cm, sim_cross)
    primary_cycle_id, cycles_section = _build_cycles_section(cm)

    return {
        "cross_id": cm.real_cross_id,
        "sim_tls_id": cm.sim_tls_id,
        "direction_map": direction_map,
        "lanes_per_direction": lanes_per_dir,
        "roads_static": roads_static,
        "observation_mask_direction": mask_dir,
        "observation_mask": mask_lane,
        "primary_cycle_id": primary_cycle_id,
        "cycles": cycles_section,
    }


def _build_neighbors(
    deployment_map: DeploymentMap, sim_config: dict
) -> Dict[str, List[Dict[str, Any]]]:
    """Build neighbor map keyed by real_cross_id (str)."""
    sim_to_real: Dict[str, int] = {
        c.sim_tls_id: c.real_cross_id for c in deployment_map.crosses
    }

    sim_adjacency: Dict[str, Any] = sim_config.get("adjacency") or {}
    neighbors: Dict[str, List[Dict[str, Any]]] = {}

    for cm in deployment_map.crosses:
        sim_nbrs = sim_adjacency.get(cm.sim_tls_id) or []
        out: List[Dict[str, Any]] = []
        for nbr in sim_nbrs:
            sim_nbr_id = nbr.get("neighbor_id")
            real_nbr_id = sim_to_real.get(sim_nbr_id)
            if real_nbr_id is None:
                continue
            direction_str = nbr.get("direction")
            try:
                d_idx = _direction_idx(direction_str) if direction_str else None
            except ValueError:
                d_idx = None
            entry: Dict[str, Any] = {"neighbor_id": real_nbr_id}
            if d_idx is not None:
                entry["direction"] = d_idx
            out.append(entry)
        neighbors[str(cm.real_cross_id)] = out

    return neighbors


def build_network_json(deployment_map: DeploymentMap, sim_config: dict) -> dict:
    """Build `network.json` cho bundle. Keyed bằng real_cross_id."""
    cross_ids = [cm.real_cross_id for cm in deployment_map.crosses]
    neighbors = _build_neighbors(deployment_map, sim_config)

    return {
        "area_id": deployment_map.area_id,
        "network_id": deployment_map.network_id,
        "cross_ids": cross_ids,
        "neighbors": neighbors,
        "max_neighbors": NUM_DIRECTIONS,
    }


def build_feature_formula_json(deployment_map: DeploymentMap) -> dict:
    """Extract feature_formula thành file riêng cho bundle."""
    return deployment_map.feature_formula.model_dump()


def build_all_intersection_configs(
    deployment_map: DeploymentMap, sim_config: dict
) -> Dict[int, dict]:
    """Build tất cả cross_<real_id>.json. Returns {real_cross_id: dict}."""
    sim_intersections = sim_config.get("intersections") or {}
    out: Dict[int, dict] = {}
    for cm in deployment_map.crosses:
        sim_cross = sim_intersections.get(cm.sim_tls_id)
        if sim_cross is None:
            raise ValueError(
                f"Sim config thiếu cross sim_tls_id={cm.sim_tls_id}. "
                f"Chạy validator trước khi build bundle."
            )
        out[cm.real_cross_id] = build_cross_config(cm, sim_cross)
    return out
