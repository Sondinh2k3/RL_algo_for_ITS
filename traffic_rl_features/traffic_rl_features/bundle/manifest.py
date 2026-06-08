"""Model Bundle manifest schema.

Một Model Bundle chuẩn chứa:
  policy.onnx                       # trọng số ONNX
  policy_meta.json                  # hyperparameters + obs_stats + input/output names
  network.json                      # topology graph (cross + neighbor)
  intersections/cross_<id>.json     # config từng nút giao
  model_manifest.json               # metadata bundle (file này)

`model_manifest.json` là single source of truth khi validate bundle. Không
được auto-generate trên Edge — phải sinh tại CI/CD Packager (Lớp 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


MANIFEST_FILENAME = "model_manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"

# File bắt buộc tồn tại trong bundle (relative paths). intersections/ là folder
# nên kiểm tra sau khi load manifest.intersection_files.
BUNDLE_FILES_REQUIRED: tuple[str, ...] = (
    "policy.onnx",
    "policy_meta.json",
    "network.json",
    MANIFEST_FILENAME,
)


@dataclass
class ModelManifest:
    """Schema model_manifest.json.

    Attributes:
      schema_version: phiên bản schema manifest (forward-compat).
      bundle_id: ID duy nhất cho mỗi lần build (UUID hoặc slug).
      tenant_id: định danh khách hàng (vd 'hcm_city').
      network_id: định danh mạng lưới (vd 'quan_1_corridor'). Phải khớp với
        network_id mà Core Controller gửi trong inference request.
      version: phiên bản policy. Format khuyến nghị 'vMAJOR.MINOR.PATCH' hoặc
        'YYYY.MM.DD-N'.
      topology_hash: SHA-256 hex của network.json đã canonicalize (xem
        topology_hash.compute_topology_hash). Phát hiện drift cấu trúc đường.
      checksum: SHA-256 hex tổng hợp các file con (xem checksum.compute_bundle_checksum).
      policy_version: alias của version, dùng tương thích ngược với cấu trúc cũ.
      config_version: phiên bản config (direction_map/phase_mapping/mask).
      created_at: ISO-8601 UTC. CI/CD Packager set lúc build.
      training_run_id: tham chiếu MLflow run ID (nếu có).
      training_dataset_id: tham chiếu dataset đã dùng để train.
      training_pipeline_commit: git SHA của pipeline training.
      intersection_files: danh sách relative path của intersections/cross_<id>.json
        có trong bundle.
      file_checksums: map relative_path -> sha256 hex của từng file con.
      sim_network_id: network_id phía sim training (vd 'cologne3'). v2 only.
      deployment_map_sha256: SHA-256 của deployment_map.json snapshot. v2 only.
      feature_formula_sha256: SHA-256 của feature_formula.json. v2 only.
      feature_pkg_version: SemVer của shared package `traffic_rl_features`
        lúc build bundle. Runtime check major-compatibility.
      commissioned_at: ISO-8601 thời điểm operator commission.
      commissioned_by: tên operator commission.
      extras: trường tự do để mở rộng tương lai.
    """

    bundle_id: str
    tenant_id: str
    network_id: str
    version: str
    topology_hash: str
    checksum: str
    schema_version: str = MANIFEST_SCHEMA_VERSION
    policy_version: Optional[str] = None
    config_version: str = "1"
    created_at: str = ""
    training_run_id: Optional[str] = None
    training_dataset_id: Optional[str] = None
    training_pipeline_commit: Optional[str] = None
    intersection_files: List[str] = field(default_factory=list)
    file_checksums: Dict[str, str] = field(default_factory=dict)
    # Commissioning metadata (v2 bundle — bundle build từ sim_config + deployment_map).
    # v1 legacy bundle để các trường này None để tránh break extractor cũ.
    sim_network_id: Optional[str] = None
    deployment_map_sha256: Optional[str] = None
    feature_formula_sha256: Optional[str] = None
    feature_pkg_version: Optional[str] = None
    commissioned_at: Optional[str] = None
    commissioned_by: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_version:
            self.policy_version = self.version
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        if not isinstance(data, dict):
            raise ValueError("Manifest payload phải là dict.")

        required = ("bundle_id", "tenant_id", "network_id", "version",
                    "topology_hash", "checksum")
        missing = [k for k in required if not data.get(k)]
        if missing:
            raise ValueError(f"Manifest thiếu trường bắt buộc: {missing}")

        return cls(
            bundle_id=str(data["bundle_id"]),
            tenant_id=str(data["tenant_id"]),
            network_id=str(data["network_id"]),
            version=str(data["version"]),
            topology_hash=str(data["topology_hash"]),
            checksum=str(data["checksum"]),
            schema_version=str(data.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            policy_version=data.get("policy_version") or str(data["version"]),
            config_version=str(data.get("config_version", "1")),
            created_at=str(data.get("created_at", "")),
            training_run_id=data.get("training_run_id"),
            training_dataset_id=data.get("training_dataset_id"),
            training_pipeline_commit=data.get("training_pipeline_commit"),
            intersection_files=list(data.get("intersection_files") or []),
            file_checksums=dict(data.get("file_checksums") or {}),
            sim_network_id=data.get("sim_network_id"),
            deployment_map_sha256=data.get("deployment_map_sha256"),
            feature_formula_sha256=data.get("feature_formula_sha256"),
            feature_pkg_version=data.get("feature_pkg_version"),
            commissioned_at=data.get("commissioned_at"),
            commissioned_by=data.get("commissioned_by"),
            extras=dict(data.get("extras") or {}),
        )

    # Path helpers
    def manifest_path(self, bundle_root: Path) -> Path:
        return Path(bundle_root) / MANIFEST_FILENAME

    def policy_path(self, bundle_root: Path) -> Path:
        return Path(bundle_root) / "policy.onnx"

    def meta_path(self, bundle_root: Path) -> Path:
        return Path(bundle_root) / "policy_meta.json"

    def network_path(self, bundle_root: Path) -> Path:
        return Path(bundle_root) / "network.json"

    def intersection_path(self, bundle_root: Path, cross_id: int) -> Path:
        return Path(bundle_root) / "intersections" / f"cross_{cross_id}.json"
