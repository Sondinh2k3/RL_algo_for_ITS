"""Tests cho traffic_rl_features package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traffic_rl_features import (
    ALLOWED_VARS,
    DEFAULT_CHANNELS,
    PACKAGE_VERSION,
    FeatureBuilder,
    FeatureSpec,
    FormulaError,
    compile_formula,
    default_spec,
    eval_formula,
    is_compatible,
    major_version,
    validate_formula_syntax,
)
from traffic_rl_features.sim_helpers import (
    road_static_from_sumo_lane,
    sumo_detector_to_road_vars,
)


# --------------------------- formula -------------------------------------


def test_formula_basic_arithmetic():
    tree = compile_formula("occupancy / 100.0 + lanes")
    assert eval_formula(tree, {"occupancy": 50.0, "lanes": 2.0}) == 2.5


def test_formula_clip():
    tree = compile_formula("clip(speed / 50.0, 0.0, 1.0)")
    assert eval_formula(tree, {"speed": 70.0}) == 1.0
    assert eval_formula(tree, {"speed": -5.0}) == 0.0


def test_formula_reject_attribute_access():
    with pytest.raises(FormulaError):
        compile_formula("occupancy.foo")


def test_formula_reject_unknown_var():
    with pytest.raises(FormulaError):
        validate_formula_syntax("foo + 1", set(ALLOWED_VARS))


def test_formula_allowed_vars_complete():
    expected = {
        "occupancy", "speed", "density", "queue",
        "lanes", "length", "speed_design", "saturation_flow",
    }
    assert set(ALLOWED_VARS) == expected


# --------------------------- spec ----------------------------------------


def test_default_spec_has_4_channels():
    spec = default_spec()
    assert spec.channels == DEFAULT_CHANNELS
    assert spec.num_channels == 4


def test_spec_validates_formula_at_construction():
    with pytest.raises(FormulaError):
        FeatureSpec(channels=("x",), formulas={"x": "occupancy.attr"})


def test_spec_reject_missing_formula():
    with pytest.raises(FormulaError, match="Thiếu"):
        FeatureSpec(channels=("a", "b"), formulas={"a": "occupancy"})


def test_spec_reject_extra_formula():
    with pytest.raises(FormulaError, match="ngoài"):
        FeatureSpec(channels=("a",), formulas={"a": "occupancy", "b": "speed"})


def test_spec_reject_duplicate_channels():
    with pytest.raises(FormulaError, match="trùng"):
        FeatureSpec(channels=("a", "a"), formulas={"a": "occupancy"})


def test_spec_round_trip_json():
    spec = default_spec()
    payload = spec.to_json()
    loaded = FeatureSpec.from_json(payload)
    assert loaded.channels == spec.channels
    assert loaded.formulas == spec.formulas


def test_spec_from_file(tmp_path: Path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({
        "channels": ["a"],
        "formulas": {"a": "occupancy / 100.0"},
    }))
    spec = FeatureSpec.from_file(p)
    assert spec.channels == ("a",)


# --------------------------- builder -------------------------------------


def test_builder_compute_basic():
    spec = FeatureSpec(
        channels=("c1", "c2"),
        formulas={
            "c1": "occupancy / 100.0",
            "c2": "speed / 50.0",
        },
    )
    builder = FeatureBuilder(spec)
    out = builder.compute(real_road_id=1, occupancy=50.0, speed=25.0)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)


def test_builder_uses_road_static_lookup():
    spec = FeatureSpec(
        channels=("x",),
        formulas={"x": "lanes * length / speed_design"},
    )
    roads_static = {
        "100": {"lanes": 3, "length_meters": 200, "speed_design_kmh": 60},
    }
    builder = FeatureBuilder(spec, roads_static=roads_static)
    out = builder.compute(real_road_id=100, occupancy=0, speed=0)
    assert out[0] == pytest.approx(3 * 200 / 60)


def test_builder_defaults_for_missing_road():
    spec = FeatureSpec(
        channels=("x",),
        formulas={"x": "lanes * length / speed_design"},
    )
    builder = FeatureBuilder(spec, roads_static={})
    out = builder.compute(real_road_id=999, occupancy=0, speed=0)
    # Defaults: lanes=1, length=100, speed_design=50
    assert out[0] == pytest.approx(1 * 100 / 50)


def test_builder_compute_batch():
    spec = default_spec()
    builder = FeatureBuilder(spec)
    batch = builder.compute_batch([
        (1, 50.0, 30.0),
        (2, 20.0, 40.0),
        (3, 0.0, 60.0),
    ])
    assert batch.shape == (3, 4)


def test_builder_normalizes_road_id_to_str():
    """compute(123) và compute('123') phải trả về cùng output."""
    spec = FeatureSpec(channels=("x",), formulas={"x": "lanes"})
    builder = FeatureBuilder(spec, roads_static={"42": {"lanes": 5}})
    assert builder.compute(42, 0, 0)[0] == 5
    assert builder.compute("42", 0, 0)[0] == 5


# --------------------------- version contract ----------------------------


def test_package_version_semver_format():
    parts = PACKAGE_VERSION.split(".")
    assert len(parts) == 3
    for p in parts:
        int(p)  # raise if not int


def test_is_compatible_same_major():
    assert is_compatible("1.5.3", "1.0.0")
    assert is_compatible("1.0.0", "1.0.0")


def test_is_compatible_different_major():
    assert not is_compatible("2.0.0", "1.0.0")
    assert not is_compatible("1.0.0", "2.0.0")


def test_is_compatible_invalid_version():
    assert not is_compatible("abc", "1.0.0")
    assert not is_compatible("", "1.0.0")


def test_major_version():
    assert major_version("3.7.2") == 3
    assert major_version() == int(PACKAGE_VERSION.split(".")[0])


# --------------------------- sim_helpers ---------------------------------


def test_sumo_detector_prefers_e1_occupancy():
    out = sumo_detector_to_road_vars(
        e1_occupancy_percent=30.0,
        e2_occupancy_percent=50.0,
        e1_mean_speed_ms=10.0,
    )
    assert out["occupancy"] == 30.0
    assert out["speed"] == pytest.approx(36.0)  # 10 m/s * 3.6


def test_sumo_detector_fallback_to_e2():
    out = sumo_detector_to_road_vars(
        e1_occupancy_percent=None,
        e2_occupancy_percent=42.0,
        e1_mean_speed_ms=None,
    )
    assert out["occupancy"] == 42.0
    assert out["speed"] == 0.0


def test_sumo_detector_clamps_occupancy():
    out = sumo_detector_to_road_vars(
        e1_occupancy_percent=150.0,
        e1_mean_speed_ms=5.0,
    )
    assert out["occupancy"] == 100.0


def test_road_static_from_sumo_lane_default_saturation():
    static = road_static_from_sumo_lane(num_lanes=2, lane_length_meters=200)
    assert static["lanes"] == 2
    assert static["length_meters"] == 200.0
    assert static["saturation_flow"] == 3600.0  # 2 * 1800


def test_road_static_from_sumo_lane_with_speed_limit():
    static = road_static_from_sumo_lane(
        num_lanes=1, lane_length_meters=100,
        speed_limit_ms=13.89,  # ~50 km/h
        saturation_flow_vph=1500,
    )
    assert static["speed_design_kmh"] == pytest.approx(13.89 * 3.6)
    assert static["saturation_flow"] == 1500.0


# --------------------------- end-to-end consistency ----------------------


def test_sim_and_runtime_produce_same_features():
    """Cốt lõi của package: cùng spec + cùng vars → cùng output.

    Mô phỏng scenario: sim feed (det_occ, det_speed_ms), runtime feed
    (api_occ, api_speed_kmh). Nếu thông qua sim_helpers convert đúng → output
    builder ở hai bên giống nhau.
    """
    spec = default_spec()

    # Sim trainer side
    sim_vars = sumo_detector_to_road_vars(
        e1_occupancy_percent=45.0,
        e1_mean_speed_ms=30.0 / 3.6,  # 30 km/h ở m/s
    )
    sim_static = road_static_from_sumo_lane(num_lanes=2, lane_length_meters=150)
    sim_builder = FeatureBuilder(spec, roads_static={"r1": sim_static})
    sim_feat = sim_builder.compute("r1", **sim_vars)

    # Runtime side: API gửi cùng giá trị logical
    runtime_static = {"lanes": 2, "length_meters": 150, "speed_design_kmh": 50, "saturation_flow": 3600}
    runtime_builder = FeatureBuilder(spec, roads_static={"r1": runtime_static})
    runtime_feat = runtime_builder.compute("r1", occupancy=45.0, speed=30.0)

    np.testing.assert_allclose(sim_feat, runtime_feat, rtol=1e-6)
