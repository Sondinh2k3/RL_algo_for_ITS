"""Bundle format submodule — shared giữa bundle-tooling và ai-algorithm-service.

Mục đích: định nghĩa CHUẨN của bundle (manifest, checksum, topology hash) ở
một nơi duy nhất, cả phía build (tooling) và phía runtime (service) cùng import.

NỘI DUNG:
  - manifest.py        : ModelManifest dataclass + parse/serialize
  - checksum.py        : SHA-256 helpers cho file + aggregate
  - topology_hash.py   : canonical hash của network.json

KHÔNG có ở đây: extraction (runtime concern), packaging (build-time concern),
deployment_map (commissioning concern). Mỗi project tự xử các phần đó.
"""

from traffic_rl_features.bundle.checksum import (
    compute_bundle_checksum,
    compute_dir_checksums,
    compute_file_sha256,
)
from traffic_rl_features.bundle.manifest import (
    BUNDLE_FILES_REQUIRED,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ModelManifest,
)
from traffic_rl_features.bundle.topology_hash import compute_topology_hash

__all__ = [
    "BUNDLE_FILES_REQUIRED",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ModelManifest",
    "compute_bundle_checksum",
    "compute_dir_checksums",
    "compute_file_sha256",
    "compute_topology_hash",
]
