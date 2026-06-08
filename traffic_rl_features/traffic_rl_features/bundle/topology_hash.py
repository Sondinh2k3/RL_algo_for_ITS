"""Topology hash — phát hiện drift cấu trúc đường.

Tính SHA-256 trên `network.json` đã canonicalize:
  - Sắp xếp keys deterministic.
  - Chỉ giữ trường mô tả cấu trúc (id, neighbors, direction, lane count).
  - Loại trường ephemeral (timestamps, comments).

Nếu hash trên Edge khác hash trong manifest → cấu trúc đường thay đổi →
ai-ops dừng activate, ai-runtime fallback heuristic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union


# Trường có ý nghĩa cấu trúc. Trường khác bị bỏ qua khi hash.
_STRUCTURAL_KEYS = {
    "intersections",
    "id",
    "neighbors",
    "neighbor_id",
    "direction",
    "lanes",
    "num_lanes",
    "incoming",
    "outgoing",
    "edges",
    "from",
    "to",
    "road_id",
    "lane_index",
}


def _canonicalize(value: Any) -> Any:
    """Lọc giữ structural keys + sort dict + sort danh sách dict theo id/neighbor_id."""
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k in _STRUCTURAL_KEYS:
                out[k] = _canonicalize(v)
        return dict(sorted(out.items()))
    if isinstance(value, (list, tuple)):
        items = [_canonicalize(v) for v in value]
        # Nếu là list of dict có key id / neighbor_id → sort theo nó.
        if items and all(isinstance(x, dict) for x in items):
            for k in ("id", "neighbor_id", "road_id"):
                if all(k in x for x in items):
                    items = sorted(items, key=lambda x: str(x[k]))
                    break
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compute_topology_hash(network: Union[dict, Path, str]) -> str:
    """Compute deterministic SHA-256 hash từ network.json.

    Có thể truyền dict đã parse hoặc đường dẫn file.
    """
    if isinstance(network, (str, Path)):
        with open(network, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(network, dict):
        data = network
    else:
        raise TypeError(f"network phải là dict hoặc path, got {type(network)}")

    canon = _canonicalize(data)
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
