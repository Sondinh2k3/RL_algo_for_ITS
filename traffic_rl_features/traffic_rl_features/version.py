"""Version constant cho contract giữa sim trainer ↔ runtime service.

Bump khi thay đổi:
  - Cú pháp formula được phép (thêm/bớt operator, function)
  - Tập biến cho phép (ALLOWED_VARS)
  - Default formula
  - Signature của FeatureBuilder.compute

Bump theo SemVer:
  - MAJOR: thay đổi không tương thích (sim training cũ + runtime mới = mismatch silent)
  - MINOR: thêm tính năng giữ tương thích ngược (vd thêm function clip2)
  - PATCH: bug fix evaluator

ai-algorithm-service nhúng version này vào bundle manifest. Khi load bundle,
runtime check version major khớp với version đang chạy — nếu không, fail-fast
với contract mismatch.
"""

PACKAGE_VERSION = "1.0.0"


def major_version(version: str = PACKAGE_VERSION) -> int:
    return int(version.split(".", 1)[0])


def is_compatible(bundle_version: str, runtime_version: str = PACKAGE_VERSION) -> bool:
    """True nếu major version khớp. Khác major → KHÔNG compatible."""
    try:
        return major_version(bundle_version) == major_version(runtime_version)
    except (ValueError, IndexError):
        return False
