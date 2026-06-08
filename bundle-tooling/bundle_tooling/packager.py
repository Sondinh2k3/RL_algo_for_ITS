"""Bundle Packager v2 — build Model Bundle ZIP từ sim output + deployment_map.

Quy trình 7 bước:
  1. Parse + validate deployment_map.json (Pydantic schema).
  2. Cross-validate với sim_config (validator). Strict → raise nếu có error.
  3. Translate sim ID → real ID → emit intersections/cross_<real_id>.json + network.json.
  4. Copy policy.onnx + policy_meta.json vào staging.
  5. Emit feature_formula.json + deployment_map.json snapshot.
  6. Tính topology_hash, file_checksums, build manifest.
  7. ZIP toàn bộ staging.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from traffic_rl_features import PACKAGE_VERSION as FEATURE_PKG_VERSION
from traffic_rl_features.bundle import (
    MANIFEST_FILENAME,
    ModelManifest,
    compute_bundle_checksum,
    compute_dir_checksums,
    compute_file_sha256,
    compute_topology_hash,
)

from bundle_tooling.deployment_map import DeploymentMap
from bundle_tooling.deployment_validator import (
    format_report,
    has_errors,
    validate as validate_deployment_map,
)
from bundle_tooling.intersection_builder import (
    build_all_intersection_configs,
    build_feature_formula_json,
    build_network_json,
)


# File commissioning v2.
FEATURE_FORMULA_FILENAME = "feature_formula.json"
DEPLOYMENT_MAP_FILENAME = "deployment_map.json"


class CommissioningError(ValueError):
    """deployment_map không pass validate với sim_config."""


def _gen_bundle_id(network_id: str, version: str) -> str:
    return f"{network_id}-{version}-{uuid.uuid4().hex[:8]}"


def build_v2_bundle_zip(
    *,
    sim_config_path: Path,
    deployment_map_path: Path,
    policy_onnx_path: Path,
    policy_meta_path: Path,
    output_zip: Path,
    tenant_id: str,
    version: str,
    bundle_id: Optional[str] = None,
    config_version: str = "1",
    training_run_id: Optional[str] = None,
    training_dataset_id: Optional[str] = None,
    training_pipeline_commit: Optional[str] = None,
    commissioned_by: Optional[str] = None,
    extras: Optional[dict] = None,
    strict: bool = True,
) -> ModelManifest:
    """Build bundle ZIP v2 từ sim config + deployment_map."""
    sim_config_path = Path(sim_config_path).resolve()
    deployment_map_path = Path(deployment_map_path).resolve()
    policy_onnx_path = Path(policy_onnx_path).resolve()
    policy_meta_path = Path(policy_meta_path).resolve()
    output_zip = Path(output_zip).resolve()

    for p, label in (
        (sim_config_path, "sim_config"),
        (deployment_map_path, "deployment_map"),
        (policy_onnx_path, "policy.onnx"),
        (policy_meta_path, "policy_meta.json"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Thiếu input '{label}': {p}")

    # 1. Parse deployment_map
    deployment_map = DeploymentMap.model_validate_json(
        deployment_map_path.read_text(encoding="utf-8")
    )
    # 2. Parse sim_config + cross-validate
    sim_config = json.loads(sim_config_path.read_text(encoding="utf-8"))
    issues = validate_deployment_map(deployment_map, sim_config)
    if issues:
        print(f"[packager] {format_report(issues)}")
    if strict and has_errors(issues):
        raise CommissioningError(
            "Deployment_map có error khi validate với sim_config. "
            "Sửa lỗi rồi build lại, hoặc dùng strict=False (chỉ recommend cho dev)."
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "intersections").mkdir()

        # 3. Translate sim_id → real_id
        cross_configs = build_all_intersection_configs(deployment_map, sim_config)
        intersection_files: List[str] = []
        for real_cross_id, cfg in cross_configs.items():
            rel = f"intersections/cross_{real_cross_id}.json"
            with open(staging / rel, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            intersection_files.append(rel)
        intersection_files.sort()

        network_dict = build_network_json(deployment_map, sim_config)
        with open(staging / "network.json", "w", encoding="utf-8") as f:
            json.dump(network_dict, f, ensure_ascii=False, indent=2)

        # 4. Copy policy artifacts
        shutil.copy2(policy_onnx_path, staging / "policy.onnx")
        shutil.copy2(policy_meta_path, staging / "policy_meta.json")

        # 5. Emit feature_formula.json + deployment_map.json snapshot
        feature_formula_dict = build_feature_formula_json(deployment_map)
        with open(staging / FEATURE_FORMULA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(feature_formula_dict, f, ensure_ascii=False, indent=2)

        deploy_dump = deployment_map.model_dump()
        with open(staging / DEPLOYMENT_MAP_FILENAME, "w", encoding="utf-8") as f:
            json.dump(deploy_dump, f, ensure_ascii=False, indent=2)

        # 6. Tính hash + manifest
        topology_hash = compute_topology_hash(staging / "network.json")
        deployment_map_sha = compute_file_sha256(staging / DEPLOYMENT_MAP_FILENAME)
        feature_formula_sha = compute_file_sha256(staging / FEATURE_FORMULA_FILENAME)

        file_checksums = compute_dir_checksums(
            staging, exclude=(MANIFEST_FILENAME,)
        )
        agg_checksum = compute_bundle_checksum(file_checksums)

        manifest = ModelManifest(
            bundle_id=bundle_id or _gen_bundle_id(deployment_map.network_id, version),
            tenant_id=tenant_id,
            network_id=deployment_map.network_id,
            version=version,
            topology_hash=topology_hash,
            checksum=agg_checksum,
            policy_version=version,
            config_version=config_version,
            training_run_id=training_run_id,
            training_dataset_id=training_dataset_id,
            training_pipeline_commit=training_pipeline_commit,
            intersection_files=intersection_files,
            file_checksums=file_checksums,
            sim_network_id=deployment_map.network_id,
            deployment_map_sha256=deployment_map_sha,
            feature_formula_sha256=feature_formula_sha,
            feature_pkg_version=FEATURE_PKG_VERSION,
            commissioned_at=deployment_map.created_at,
            commissioned_by=commissioned_by or deployment_map.created_by,
            extras=dict(extras or {}),
        )

        with open(staging / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

        # 7. ZIP
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(staging.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(staging).as_posix())

    return manifest
