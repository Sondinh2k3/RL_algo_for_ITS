"""Checksum helpers cho Model Bundle.

Bundle checksum = SHA-256 của chuỗi ghép `relpath:sha256` đã sort theo relpath
(không phụ thuộc thứ tự duyệt FS). Cho phép verify nhanh mà không cần lưu
chuỗi raw.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable


_CHUNK = 1024 * 1024


def compute_file_sha256(path: Path) -> str:
    """SHA-256 hex của 1 file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_bundle_checksum(file_checksums: Dict[str, str]) -> str:
    """Aggregate checksum từ map relpath -> sha256.

    Sort theo relpath để bất biến với thứ tự thêm vào dict.
    """
    h = hashlib.sha256()
    for relpath in sorted(file_checksums):
        h.update(relpath.encode("utf-8"))
        h.update(b":")
        h.update(file_checksums[relpath].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def compute_dir_checksums(
    root: Path,
    *,
    exclude: Iterable[str] = (),
) -> Dict[str, str]:
    """Tính sha256 cho mọi file trong thư mục, trả map relpath -> hex.

    relpath dùng forward slash để bất biến giữa OS.
    """
    root = Path(root).resolve()
    exclude_set = set(exclude)
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in exclude_set:
            continue
        out[rel] = compute_file_sha256(p)
    return out
