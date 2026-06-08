"""Cross-validate deployment_map.json against sim's intersection_config.json.

Pydantic schema (`deployment_map.py`) đã validate cấu trúc nội tại. Validator
này check **consistency với sim config** — bắt lỗi commissioning sớm trước khi
build bundle.

Trả về danh sách issue (errors + warnings). Caller (packager) quyết định fail
hay tiếp tục dựa trên severity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional  # noqa: F401

from bundle_tooling.deployment_map import CrossMapping, DeploymentMap


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    severity: IssueSeverity
    sim_tls_id: Optional[str]
    code: str
    message: str

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}] "
        if self.sim_tls_id:
            prefix += f"cross={self.sim_tls_id} "
        return f"{prefix}{self.code}: {self.message}"


def _ensure_sim_cross(sim_config: dict, tls_id: str) -> Optional[dict]:
    intersections = sim_config.get("intersections") or {}
    return intersections.get(tls_id)


def _check_cross_coverage(
    dm: DeploymentMap, sim_config: dict, issues: List[ValidationIssue]
) -> None:
    sim_ids = set((sim_config.get("intersections") or {}).keys())
    mapped_ids = {c.sim_tls_id for c in dm.crosses}

    missing = sim_ids - mapped_ids
    for sim_id in sorted(missing):
        issues.append(ValidationIssue(
            IssueSeverity.ERROR, sim_id, "CROSS_NOT_MAPPED",
            f"Sim cross '{sim_id}' không có trong deployment_map.",
        ))

    extra = mapped_ids - sim_ids
    for sim_id in sorted(extra):
        issues.append(ValidationIssue(
            IssueSeverity.ERROR, sim_id, "CROSS_NOT_IN_SIM",
            f"deployment_map có sim_tls_id '{sim_id}' nhưng sim không khai báo.",
        ))


def _check_directions(
    cm: CrossMapping, sim_cross: dict, issues: List[ValidationIssue]
) -> None:
    sim_dir_map = sim_cross.get("direction_map") or {}
    sim_edges_by_dir: Dict[str, Optional[str]] = {
        d: sim_dir_map.get(d) for d in ("N", "E", "S", "W")
    }

    for direction in ("N", "E", "S", "W"):
        sim_edge = sim_edges_by_dir.get(direction)
        real_road = cm.roads_by_direction.get(direction)

        if sim_edge is None and real_road is not None:
            issues.append(ValidationIssue(
                IssueSeverity.WARNING, cm.sim_tls_id, "DIRECTION_EXTRA_IN_REAL",
                f"Hướng {direction}: sim không có edge nhưng deployment_map khai báo road.",
            ))
            continue

        if sim_edge is not None and real_road is None:
            issues.append(ValidationIssue(
                IssueSeverity.ERROR, cm.sim_tls_id, "DIRECTION_MISSING_IN_REAL",
                f"Hướng {direction}: sim có edge '{sim_edge}' nhưng deployment_map không map.",
            ))
            continue

        if sim_edge is not None and real_road is not None:
            if real_road.sim_edge_id != sim_edge:
                issues.append(ValidationIssue(
                    IssueSeverity.ERROR, cm.sim_tls_id, "SIM_EDGE_MISMATCH",
                    f"Hướng {direction}: sim edge '{sim_edge}' != map sim_edge_id "
                    f"'{real_road.sim_edge_id}'.",
                ))
            sim_lanes_list = (sim_cross.get("lanes_by_direction") or {}).get(direction) or []
            sim_lane_count = len(sim_lanes_list)
            if real_road.sim_lanes is not None and real_road.sim_lanes != sim_lane_count:
                issues.append(ValidationIssue(
                    IssueSeverity.WARNING, cm.sim_tls_id, "SIM_LANE_COUNT_MISMATCH",
                    f"Hướng {direction}: sim_lanes={real_road.sim_lanes} != "
                    f"sim config lane count={sim_lane_count}.",
                ))
            if real_road.real_lanes != sim_lane_count:
                issues.append(ValidationIssue(
                    IssueSeverity.WARNING, cm.sim_tls_id, "LANE_COUNT_MISMATCH",
                    f"Hướng {direction}: real_lanes={real_road.real_lanes} != "
                    f"sim lane count={sim_lane_count}. Policy có thể cần adapter.",
                ))


def _check_phases(
    cm: CrossMapping, sim_cross: dict, issues: List[ValidationIssue]
) -> None:
    phase_cfg = sim_cross.get("phase_config") or {}
    sim_phases = phase_cfg.get("phases") or []
    # sim_phase_idx mà operator điền trong deployment_map = phase["index"] (vd 0, 2, 4).
    # actual_to_standard của sim lại dùng **ORDINAL POSITION** trong list phases.
    sim_phase_indices: List[int] = [int(p.get("index", -1)) for p in sim_phases]
    phase_idx_to_ordinal: Dict[int, int] = {
        pi: ordinal for ordinal, pi in enumerate(sim_phase_indices) if pi >= 0
    }

    sim_actual_to_std: Dict[Any, Any] = phase_cfg.get("actual_to_standard") or {}
    ordinal_to_std: Dict[int, int] = {}
    for k, v in sim_actual_to_std.items():
        try:
            ordinal_to_std[int(k)] = int(v)
        except (TypeError, ValueError):
            continue

    primary = next((c for c in cm.cycles if c.is_primary), None)
    if primary is None:
        return

    mapped_sim_indices = {m.sim_phase_idx for m in primary.phase_to_stage}
    sim_phase_set = set(phase_idx_to_ordinal.keys())

    missing = sim_phase_set - mapped_sim_indices
    for idx in sorted(missing):
        issues.append(ValidationIssue(
            IssueSeverity.ERROR, cm.sim_tls_id, "PHASE_NOT_MAPPED",
            f"Sim phase index {idx} (cycle primary) không có trong deployment_map.",
        ))

    extra = mapped_sim_indices - sim_phase_set
    for idx in sorted(extra):
        issues.append(ValidationIssue(
            IssueSeverity.ERROR, cm.sim_tls_id, "PHASE_NOT_IN_SIM",
            f"deployment_map map sim_phase_idx={idx} nhưng sim không có pha này.",
        ))

    for mapping in primary.phase_to_stage:
        ordinal = phase_idx_to_ordinal.get(mapping.sim_phase_idx)
        if ordinal is None:
            continue
        if ordinal not in ordinal_to_std:
            issues.append(ValidationIssue(
                IssueSeverity.WARNING, cm.sim_tls_id, "STD_PHASE_NOT_IN_SIM",
                f"sim_phase_idx={mapping.sim_phase_idx} (ordinal {ordinal}) không có "
                f"trong actual_to_standard. Không verify được std_phase_idx.",
            ))
            continue
        expected = ordinal_to_std[ordinal]
        if expected != mapping.std_phase_idx:
            issues.append(ValidationIssue(
                IssueSeverity.ERROR, cm.sim_tls_id, "STD_PHASE_INDEX_MISMATCH",
                f"sim_phase_idx={mapping.sim_phase_idx} (ordinal {ordinal}): "
                f"deployment_map nói std={mapping.std_phase_idx} nhưng "
                f"sim's actual_to_standard nói {expected}.",
            ))


def validate(
    deployment_map: DeploymentMap,
    sim_config: dict,
) -> List[ValidationIssue]:
    """Cross-validate DeploymentMap với sim's intersection_config dict."""
    issues: List[ValidationIssue] = []

    if not isinstance(sim_config, dict) or "intersections" not in sim_config:
        issues.append(ValidationIssue(
            IssueSeverity.ERROR, None, "INVALID_SIM_CONFIG",
            "sim_config phải là dict có key 'intersections'.",
        ))
        return issues

    _check_cross_coverage(deployment_map, sim_config, issues)

    for cm in deployment_map.crosses:
        sim_cross = _ensure_sim_cross(sim_config, cm.sim_tls_id)
        if sim_cross is None:
            continue
        _check_directions(cm, sim_cross, issues)
        _check_phases(cm, sim_cross, issues)

    return issues


def has_errors(issues: List[ValidationIssue]) -> bool:
    return any(i.severity is IssueSeverity.ERROR for i in issues)


def format_report(issues: List[ValidationIssue]) -> str:
    if not issues:
        return "Validation passed — không có issue."
    lines = ["Validation issues:"]
    for issue in issues:
        lines.append(f"  {issue}")
    err_count = sum(1 for i in issues if i.severity is IssueSeverity.ERROR)
    warn_count = sum(1 for i in issues if i.severity is IssueSeverity.WARNING)
    lines.append(f"Total: {err_count} error(s), {warn_count} warning(s).")
    return "\n".join(lines)
