"""traffic_rl_features — shared feature preprocessing cho sim training + runtime.

Public API:

    from traffic_rl_features import (
        FeatureBuilder, FeatureSpec, default_spec,
        compile_formula, eval_formula, FormulaError,
        ALLOWED_VARS, DEFAULT_CHANNELS, PACKAGE_VERSION,
    )

Sim trainer + runtime service import cùng module này → cùng kết quả compute
khi feed cùng vars. Bump PACKAGE_VERSION (major) khi thay đổi không tương
thích — bundle manifest nhúng version, runtime fail-fast nếu mismatch.
"""

from traffic_rl_features.builder import (
    DEFAULT_ROAD_LANES,
    DEFAULT_ROAD_LENGTH_M,
    DEFAULT_ROAD_SATURATION_FLOW,
    DEFAULT_ROAD_SPEED_DESIGN_KMH,
    CompiledSpec,
    FeatureBuilder,
    compile_spec,
)
from traffic_rl_features.formula import (
    ALLOWED_VARS,
    FormulaError,
    compile_formula,
    eval_formula,
    validate_formula_syntax,
)
from traffic_rl_features.spec import (
    DEFAULT_CHANNELS,
    DEFAULT_FORMULAS,
    FeatureSpec,
    default_spec,
)
from traffic_rl_features.version import (
    PACKAGE_VERSION,
    is_compatible,
    major_version,
)

__all__ = [
    "ALLOWED_VARS",
    "CompiledSpec",
    "DEFAULT_CHANNELS",
    "DEFAULT_FORMULAS",
    "DEFAULT_ROAD_LANES",
    "DEFAULT_ROAD_LENGTH_M",
    "DEFAULT_ROAD_SATURATION_FLOW",
    "DEFAULT_ROAD_SPEED_DESIGN_KMH",
    "FeatureBuilder",
    "FeatureSpec",
    "FormulaError",
    "PACKAGE_VERSION",
    "compile_formula",
    "compile_spec",
    "default_spec",
    "eval_formula",
    "is_compatible",
    "major_version",
    "validate_formula_syntax",
]
