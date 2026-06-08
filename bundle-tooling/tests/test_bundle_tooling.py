"""Tests cho bundle-tooling package — deployment map + validator + intersection_builder + packager."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from bundle_tooling import (
    CommissioningError,
    CrossMapping,
    CycleMapping,
    DeploymentMap,
    FeatureFormula,
    PhaseStageMapping,
    RoadMapping,
    build_all_intersection_configs,
    build_cross_config,
    build_feature_formula_json,
    build_network_json,
    build_v2_bundle_zip,
    format_report,
    has_errors,
    validate,
)
from bundle_tooling.deployment_validator import IssueSeverity
from bundle_tooling.intersection_builder import TOTAL_LANES


# Repo-relative paths — examples đã copy vào bundle-tooling.
_EXAMPLES = Path(__file__).parent.parent / "examples" / "cologne3"
_SIM_CONFIG_PATH = _EXAMPLES / "intersection_config.json"
_DEPLOY_EXAMPLE_PATH = _EXAMPLES / "deployment_map.example.json"


def _load_sim_config() -> dict:
    return json.loads(_SIM_CONFIG_PATH.read_text(encoding="utf-8"))


def _load_example_dm() -> DeploymentMap:
    return DeploymentMap.model_validate_json(
        _DEPLOY_EXAMPLE_PATH.read_text(encoding="utf-8")
    )


def _make_minimal_dm() -> DeploymentMap:
    return DeploymentMap(
        area_id=1,
        network_id="cologne3",
        feature_formula=FeatureFormula(
            channels=["density", "queue", "occupancy", "speed"],
            formulas={
                "density": "occupancy / 100.0",
                "queue": "occupancy / 100.0 * lanes",
                "occupancy": "occupancy / 100.0",
                "speed": "speed / 50.0",
            },
        ),
        crosses=[
            CrossMapping(
                sim_tls_id="33202549",
                real_cross_id=567001,
                roads_by_direction={
                    "N": RoadMapping(sim_edge_id="4999334", real_road_id=100001, real_lanes=1),
                    "E": RoadMapping(sim_edge_id="-241660955#6", real_road_id=100002, real_lanes=2),
                    "S": None,
                    "W": RoadMapping(sim_edge_id="241660955#4", real_road_id=100003, real_lanes=2),
                },
                cycles=[
                    CycleMapping(
                        real_cycle_id=700001,
                        is_primary=True,
                        phase_to_stage=[
                            PhaseStageMapping(sim_phase_idx=0, real_stage_id=800001, std_phase_idx=1),
                            PhaseStageMapping(sim_phase_idx=2, real_stage_id=800002, std_phase_idx=4),
                        ],
                    )
                ],
            )
        ],
    )


# --------------------------- deployment_map schema -----------------------


def test_minimal_deployment_map_ok():
    dm = _make_minimal_dm()
    assert dm.area_id == 1


def test_reject_no_primary_cycle():
    base = _make_minimal_dm().model_dump()
    base["crosses"][0]["cycles"][0]["is_primary"] = False
    with pytest.raises(ValueError, match="primary"):
        DeploymentMap(**base)


def test_reject_invalid_formula_syntax():
    base = _make_minimal_dm().model_dump()
    base["feature_formula"]["formulas"]["density"] = "occupancy.attr"
    with pytest.raises(ValueError):
        DeploymentMap(**base)


# --------------------------- validator -----------------------------------


def test_example_deployment_map_parses():
    dm = _load_example_dm()
    assert dm.network_id == "cologne3"
    assert len(dm.crosses) == 5


def test_example_deployment_map_validates_against_sim():
    dm = _load_example_dm()
    sim = _load_sim_config()
    issues = validate(dm, sim)
    errors = [i for i in issues if i.severity is IssueSeverity.ERROR]
    assert not errors, f"Errors: {[str(e) for e in errors]}"


def test_validator_detects_missing_cross():
    dm = _load_example_dm()
    base = dm.model_dump()
    base["crosses"] = base["crosses"][:-1]
    dm_partial = DeploymentMap(**base)
    issues = validate(dm_partial, _load_sim_config())
    assert has_errors(issues)
    assert any(i.code == "CROSS_NOT_MAPPED" for i in issues)


def test_validator_invalid_sim_config():
    dm = _make_minimal_dm()
    issues = validate(dm, {"foo": "bar"})
    assert has_errors(issues)


# --------------------------- intersection_builder ------------------------


def test_build_cross_config_keyed_by_real_id():
    dm = _load_example_dm()
    sim = _load_sim_config()
    cm = dm.crosses[0]
    cfg = build_cross_config(cm, sim["intersections"][cm.sim_tls_id])
    assert cfg["cross_id"] == 567001
    assert cfg["sim_tls_id"] == "33202549"
    assert cfg["direction_map"]["100001"] == 0
    assert "S" not in cfg["direction_map"]


def test_build_cross_config_observation_masks():
    dm = _load_example_dm()
    sim = _load_sim_config()
    cm = dm.crosses[0]
    cfg = build_cross_config(cm, sim["intersections"][cm.sim_tls_id])
    assert cfg["observation_mask_direction"] == [1, 1, 0, 1]
    assert cfg["observation_mask"] == [1, 0, 0,  1, 1, 0,  0, 0, 0,  1, 1, 0]
    assert len(cfg["observation_mask"]) == TOTAL_LANES


def test_build_network_json_translates_neighbors():
    dm = _load_example_dm()
    sim = _load_sim_config()
    network = build_network_json(dm, sim)
    assert network["network_id"] == "cologne3"
    nbrs_567001 = network["neighbors"]["567001"]
    assert any(n["neighbor_id"] == 567004 and n["direction"] == 3 for n in nbrs_567001)


def test_build_all_intersection_configs():
    dm = _load_example_dm()
    sim = _load_sim_config()
    configs = build_all_intersection_configs(dm, sim)
    assert set(configs.keys()) == {567001, 567002, 567003, 567004, 567005}


# --------------------------- packager end-to-end -------------------------


def _make_fake_policy_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"\x00fake-onnx")
    meta = tmp_path / "policy_meta.json"
    meta.write_text(json.dumps({
        "use_local_gnn": True,
        "obs_dim": 336, "base_obs_dim": 56, "window_size": 6,
        "num_actions_per_phase": 5, "keep_action_index": 2,
        "input_names": ["self_features"],
        "output_name": "logits",
    }))
    return onnx, meta


def test_build_v2_bundle_zip_end_to_end(tmp_path: Path):
    onnx, meta = _make_fake_policy_artifacts(tmp_path)
    output_zip = tmp_path / "bundle.zip"

    from traffic_rl_features import PACKAGE_VERSION

    manifest = build_v2_bundle_zip(
        sim_config_path=_SIM_CONFIG_PATH,
        deployment_map_path=_DEPLOY_EXAMPLE_PATH,
        policy_onnx_path=onnx,
        policy_meta_path=meta,
        output_zip=output_zip,
        tenant_id="hcm_pilot",
        version="v2026.05.14",
    )

    assert output_zip.exists()
    assert manifest.network_id == "cologne3"
    assert manifest.feature_pkg_version == PACKAGE_VERSION
    assert manifest.deployment_map_sha256 is not None
    assert manifest.feature_formula_sha256 is not None

    with zipfile.ZipFile(output_zip) as zf:
        names = set(zf.namelist())
        assert "policy.onnx" in names
        assert "network.json" in names
        assert "feature_formula.json" in names
        assert "deployment_map.json" in names
        assert "model_manifest.json" in names
        for real_id in (567001, 567002, 567003, 567004, 567005):
            assert f"intersections/cross_{real_id}.json" in names


def test_build_v2_bundle_zip_rejects_bad_deployment_map(tmp_path: Path):
    dm_raw = json.loads(_DEPLOY_EXAMPLE_PATH.read_text(encoding="utf-8"))
    dm_raw["crosses"] = dm_raw["crosses"][:-1]
    bad_path = tmp_path / "deployment_map_bad.json"
    bad_path.write_text(json.dumps(dm_raw), encoding="utf-8")

    onnx, meta = _make_fake_policy_artifacts(tmp_path)

    with pytest.raises(CommissioningError):
        build_v2_bundle_zip(
            sim_config_path=_SIM_CONFIG_PATH,
            deployment_map_path=bad_path,
            policy_onnx_path=onnx,
            policy_meta_path=meta,
            output_zip=tmp_path / "out.zip",
            tenant_id="hcm",
            version="v0.1",
        )
