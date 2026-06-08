"""Deployment Map schema — bridge sim ID ↔ real DB ID lúc commissioning.

Operator điền file `deployment_map.json` một lần cho mỗi area khi triển khai.
File này KẾT HỢP với `intersection_config.json` (output của training pipeline,
mô tả mạng SUMO) để Bundle Packager sinh ra `intersections/cross_<real_id>.json`
trong bundle — runtime KHÔNG cần biết về sim IDs nữa.

Cấu trúc 3 tầng mapping:
  1. Cross-level:  sim_tls_id   → real_cross_id
  2. Road-level:   sim_edge_id  → real_road_id  (per direction)
  3. Stage-level:  (cycle_id, sim_phase_idx) → (real_stage_id, std_phase_idx)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from traffic_rl_features import ALLOWED_VARS, FormulaError, validate_formula_syntax
from traffic_rl_features.spec import DEFAULT_CHANNELS


SCHEMA_VERSION = "1.0"
_VALID_DIRECTIONS = frozenset({"N", "E", "S", "W"})
_NUM_STANDARD_PHASES = 8


class RoadMapping(BaseModel):
    """Map 1 road giữa sim và real, kèm thông số có thể lệch.

    Static road properties (length, speed_design, saturation_flow) là **biến cho
    feature_formula** — phải có ở bundle để runtime eval công thức suy diễn 4
    channel. Operator điền từ v_road.length / v_road.speed_design / v_road.capacity_design.
    """

    sim_edge_id: str = Field(..., min_length=1, description="SUMO edge ID, vd '4999334'.")
    real_road_id: int = Field(..., ge=1, description="v_road.id trong DB.")
    real_lanes: int = Field(..., gt=0, description="Số làn thực tế (v_road.number_of_lanes).")
    sim_lanes: Optional[int] = Field(default=None)
    length_meters: Optional[float] = Field(default=None, gt=0)
    speed_design_kmh: Optional[float] = Field(default=None, gt=0)
    saturation_flow: Optional[float] = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class PhaseStageMapping(BaseModel):
    sim_phase_idx: int = Field(..., ge=0)
    real_stage_id: int = Field(..., ge=1)
    std_phase_idx: int = Field(..., ge=0, lt=_NUM_STANDARD_PHASES)
    model_config = ConfigDict(extra="forbid")


class CycleMapping(BaseModel):
    real_cycle_id: int = Field(..., ge=1)
    is_primary: bool = Field(default=True)
    phase_to_stage: List[PhaseStageMapping] = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")

    @field_validator("phase_to_stage")
    @classmethod
    def _check_unique_sim_phase(cls, v: List[PhaseStageMapping]) -> List[PhaseStageMapping]:
        sim_indices = [m.sim_phase_idx for m in v]
        if len(sim_indices) != len(set(sim_indices)):
            raise ValueError("sim_phase_idx bị trùng trong cùng 1 cycle.")
        real_stage_ids = [m.real_stage_id for m in v]
        if len(real_stage_ids) != len(set(real_stage_ids)):
            raise ValueError("real_stage_id bị trùng trong cùng 1 cycle.")
        return v


class CrossMapping(BaseModel):
    sim_tls_id: str = Field(..., min_length=1)
    real_cross_id: int = Field(..., ge=1)
    roads_by_direction: Dict[str, Optional[RoadMapping]] = Field(...)
    cycles: List[CycleMapping] = Field(..., min_length=1)
    notes: Optional[str] = Field(default=None)
    model_config = ConfigDict(extra="forbid")

    @field_validator("roads_by_direction")
    @classmethod
    def _check_directions(cls, v: Dict[str, Optional[RoadMapping]]) -> Dict[str, Optional[RoadMapping]]:
        invalid = set(v.keys()) - _VALID_DIRECTIONS
        if invalid:
            raise ValueError(f"Hướng không hợp lệ: {invalid}. Cho phép: {sorted(_VALID_DIRECTIONS)}.")
        if not any(r is not None for r in v.values()):
            raise ValueError("Cross phải có ít nhất 1 hướng có road mapping.")
        return v

    @field_validator("cycles")
    @classmethod
    def _exactly_one_primary(cls, v: List[CycleMapping]) -> List[CycleMapping]:
        primary_count = sum(1 for c in v if c.is_primary)
        if primary_count != 1:
            raise ValueError(
                f"Phải có đúng 1 cycle is_primary=true, hiện có {primary_count}."
            )
        cycle_ids = [c.real_cycle_id for c in v]
        if len(cycle_ids) != len(set(cycle_ids)):
            raise ValueError("real_cycle_id bị trùng trong cùng 1 cross.")
        return v


class FeatureFormula(BaseModel):
    """Công thức suy diễn N channel từ data real.

    Sim training PHẢI áp đúng những công thức này lên detector output để
    distribution observation khớp giữa sim ↔ runtime. Cùng formula eval qua
    `traffic_rl_features.FeatureBuilder` ở cả hai phía.
    """
    channels: List[str] = Field(default=list(DEFAULT_CHANNELS))
    formulas: Dict[str, str] = Field(...)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> "FeatureFormula":
        if len(self.channels) != len(set(self.channels)):
            raise ValueError("channels bị trùng tên.")
        missing = [c for c in self.channels if c not in self.formulas]
        if missing:
            raise ValueError(f"Thiếu formula cho channel: {missing}")
        extra = [k for k in self.formulas if k not in self.channels]
        if extra:
            raise ValueError(f"formulas có channel ngoài danh sách channels: {extra}")
        for ch, expr in self.formulas.items():
            try:
                validate_formula_syntax(expr, set(ALLOWED_VARS))
            except FormulaError as e:
                raise ValueError(f"Formula cho '{ch}' lỗi: {e}") from e
        return self


class DeploymentMap(BaseModel):
    """Top-level deployment_map.json."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    area_id: int = Field(..., ge=1)
    network_id: str = Field(..., min_length=1)
    sim_config_path: Optional[str] = Field(default=None)
    sim_config_sha256: Optional[str] = Field(default=None)
    feature_formula: FeatureFormula = Field(...)
    crosses: List[CrossMapping] = Field(..., min_length=1)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    notes: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("crosses")
    @classmethod
    def _check_unique_ids(cls, v: List[CrossMapping]) -> List[CrossMapping]:
        sim_ids = [c.sim_tls_id for c in v]
        if len(sim_ids) != len(set(sim_ids)):
            raise ValueError("sim_tls_id bị trùng trong deployment_map.")
        real_ids = [c.real_cross_id for c in v]
        if len(real_ids) != len(set(real_ids)):
            raise ValueError("real_cross_id bị trùng trong deployment_map.")
        all_road_ids: List[int] = []
        for c in v:
            for road in c.roads_by_direction.values():
                if road is not None:
                    all_road_ids.append(road.real_road_id)
        if len(all_road_ids) != len(set(all_road_ids)):
            raise ValueError("real_road_id bị trùng giữa các cross.")
        return v

    def primary_cycle(self, sim_tls_id: str) -> Optional[CycleMapping]:
        for c in self.crosses:
            if c.sim_tls_id == sim_tls_id:
                for cy in c.cycles:
                    if cy.is_primary:
                        return cy
        return None
