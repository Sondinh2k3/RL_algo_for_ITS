"""FeatureBuilder — core compute engine, không phụ thuộc bundle structure.

Tách core khỏi runtime service: ai-algorithm-service wrap thêm layer cache +
bundle file IO, sim trainer dùng trực tiếp với detector data của SUMO.

Workflow điển hình:

  # Sim trainer (SUMO):
    from traffic_rl_features import FeatureSpec, FeatureBuilder
    spec = FeatureSpec.from_file("network/cologne3/feature_formula.json")
    builder = FeatureBuilder(spec, roads_static)
    obs = builder.compute(road_id, occupancy=det_e2_occ, speed=det_e1_speed)

  # Runtime service:
    spec = FeatureSpec.from_file(bundle_root / "feature_formula.json")
    builder = FeatureBuilder(spec, roads_static)
    feat = builder.compute(road.id, occupancy=road.occupancySpace,
                            speed=road.averageSpeed)

Cùng spec + cùng vars → cùng output. KHÔNG còn drift logic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from traffic_rl_features.formula import compile_formula, eval_formula
from traffic_rl_features.spec import FeatureSpec


# Default cho biến static nếu road không có (operator chưa điền lúc commissioning).
# An toàn: không divide-by-zero, không phá distribution training nếu road được
# bundle hóa đúng.
DEFAULT_ROAD_LENGTH_M: float = 100.0
DEFAULT_ROAD_SPEED_DESIGN_KMH: float = 50.0
DEFAULT_ROAD_LANES: int = 1
DEFAULT_ROAD_SATURATION_FLOW: float = 1800.0


@dataclass(frozen=True)
class CompiledSpec:
    """Spec với formula đã compile AST. Immutable, share được giữa nhiều builder."""
    spec: FeatureSpec
    compiled_by_channel: Dict[str, ast.Expression]


def compile_spec(spec: FeatureSpec) -> CompiledSpec:
    """Compile mọi formula trong spec. Idempotent."""
    return CompiledSpec(
        spec=spec,
        compiled_by_channel={ch: compile_formula(spec.formulas[ch]) for ch in spec.channels},
    )


class FeatureBuilder:
    """Eval N-channel feature cho mỗi road.

    Thread-safe (compiled AST không mutate, dict lookup is GIL-safe).
    """

    def __init__(
        self,
        spec: FeatureSpec,
        roads_static: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        self._compiled = compile_spec(spec)
        # Normalize key to str để consistent với JSON deserialization.
        self._roads_static: Dict[str, Dict[str, float]] = {}
        if roads_static:
            for rid, props in roads_static.items():
                self._roads_static[str(rid)] = dict(props)

    @property
    def spec(self) -> FeatureSpec:
        return self._compiled.spec

    @property
    def channels(self) -> tuple[str, ...]:
        return self._compiled.spec.channels

    @property
    def channel_count(self) -> int:
        return self._compiled.spec.num_channels

    def _vars_for_road(
        self,
        real_road_id: int | str,
        occupancy: float,
        speed: float,
        density: float | None,
        queue: float | None,
    ) -> Dict[str, float]:
        static = self._roads_static.get(str(real_road_id), {})
        occ = float(occupancy)
        den = occ if density is None else float(density)
        que = occ if queue is None else float(queue)
        return {
            "occupancy": occ,
            "speed": float(speed),
            "density": den,
            "queue": que,
            "lanes": float(static.get("lanes") or DEFAULT_ROAD_LANES),
            "length": float(static.get("length_meters") or DEFAULT_ROAD_LENGTH_M),
            "speed_design": float(
                static.get("speed_design_kmh") or DEFAULT_ROAD_SPEED_DESIGN_KMH
            ),
            "saturation_flow": float(
                static.get("saturation_flow") or DEFAULT_ROAD_SATURATION_FLOW
            ),
        }

    def compute(
        self,
        real_road_id: int | str,
        occupancy: float,
        speed: float,
        density: float | None = None,
        queue: float | None = None,
    ) -> np.ndarray:
        """Eval N-channel cho 1 road. Trả về ndarray shape (N,) float32."""
        vars_ = self._vars_for_road(real_road_id, occupancy, speed, density, queue)
        out = np.zeros(self.channel_count, dtype=np.float32)
        for i, ch in enumerate(self._compiled.spec.channels):
            tree = self._compiled.compiled_by_channel[ch]
            out[i] = eval_formula(tree, vars_)
        return out

    def compute_batch(
        self,
        road_data: list[tuple[int | str, float, float, float | None, float | None]],
    ) -> np.ndarray:
        """Eval cho nhiều road. road_data = [(id, occupancy, speed, density, queue), ...]. Shape (B, N)."""
        if not road_data:
            return np.zeros((0, self.channel_count), dtype=np.float32)
        out = np.zeros((len(road_data), self.channel_count), dtype=np.float32)
        for b, (rid, occ, spd, den, que) in enumerate(road_data):
            out[b] = self.compute(rid, occ, spd, den, que)
        return out
