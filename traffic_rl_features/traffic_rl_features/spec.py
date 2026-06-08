"""FeatureSpec — schema chuẩn cho feature_formula.json.

Đây là **single source of truth** về cấu trúc spec. Cả sim trainer và runtime
service đều dùng `FeatureSpec.from_dict / to_dict` để parse/emit nhất quán.

Tách spec khỏi `FeatureBuilder` (compile + eval) vì spec là pure data — có thể
serialize, validate, transmit qua mạng/file — còn builder là compute engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from traffic_rl_features.formula import (
    ALLOWED_VARS,
    FormulaError,
    validate_formula_syntax,
)


DEFAULT_CHANNELS: tuple[str, ...] = ("density", "queue", "occupancy", "speed")

# Default formula — fallback nếu bundle không có feature_formula.json. Map 2
# real measurements (occupancy + speed) vào 4 channel mà 4-channel policy có
# thể consume. Distribution KHÔNG đảm bảo match training nếu sim trainer dùng
# detector khác — phải override bằng formula riêng trong commissioning.
DEFAULT_FORMULAS: Dict[str, str] = {
    "density":   "clip(occupancy / 100.0, 0.0, 1.0)",
    "queue":     "clip(occupancy / 100.0, 0.0, 1.0)",
    "occupancy": "clip(occupancy / 100.0, 0.0, 1.0)",
    "speed":     "clip(speed / max(speed_design, 50.0), 0.0, 1.5)",
}


@dataclass(frozen=True)
class FeatureSpec:
    """Định nghĩa N kênh feature + công thức tính từ tập biến chuẩn.

    Attributes:
        channels: danh sách tên kênh. Thứ tự cố định — sim trainer + runtime
                  phải dùng cùng thứ tự để policy ăn đúng index.
        formulas: map tên kênh → Python expression. Phải đầy đủ cho mọi channel.
    """
    channels: tuple[str, ...]
    formulas: Dict[str, str]

    def __post_init__(self) -> None:
        # Frozen dataclass: validate sau init.
        if not self.channels:
            raise FormulaError("channels rỗng.")
        if len(self.channels) != len(set(self.channels)):
            raise FormulaError("channels có tên trùng.")
        missing = [c for c in self.channels if c not in self.formulas]
        if missing:
            raise FormulaError(f"Thiếu formula cho channel: {missing}")
        extra = [k for k in self.formulas if k not in self.channels]
        if extra:
            raise FormulaError(f"formulas có channel ngoài danh sách: {extra}")
        allowed = set(ALLOWED_VARS)
        for ch, expr in self.formulas.items():
            try:
                validate_formula_syntax(expr, allowed)
            except FormulaError as e:
                raise FormulaError(f"Formula cho '{ch}' lỗi: {e}") from e

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeatureSpec":
        channels_raw = data.get("channels") or DEFAULT_CHANNELS
        formulas_raw = data.get("formulas") or {}
        if not isinstance(channels_raw, Sequence) or isinstance(channels_raw, (str, bytes)):
            raise FormulaError("channels phải là list.")
        if not isinstance(formulas_raw, Mapping):
            raise FormulaError("formulas phải là dict.")
        return cls(
            channels=tuple(str(c) for c in channels_raw),
            formulas={str(k): str(v) for k, v in formulas_raw.items()},
        )

    @classmethod
    def from_json(cls, payload: str) -> "FeatureSpec":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_file(cls, path: Path | str) -> "FeatureSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> Dict[str, object]:
        return {
            "channels": list(self.channels),
            "formulas": dict(self.formulas),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def default_spec() -> FeatureSpec:
    """Default 4-channel spec dùng làm fallback."""
    return FeatureSpec(channels=DEFAULT_CHANNELS, formulas=dict(DEFAULT_FORMULAS))
