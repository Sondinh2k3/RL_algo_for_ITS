"""CLI: build-bundle — đóng gói model bundle từ sim outputs + deployment_map.

Cách dùng:

  build-bundle v2 \\
    --sim-config       network/cologne3/intersection_config.json \\
    --deployment-map   network/cologne3/deployment_map.json \\
    --policy-onnx      <sim-output>/policy.onnx \\
    --policy-meta      <sim-output>/policy_meta.json \\
    --tenant-id        hcm_pilot \\
    --version          v2026.05.15 \\
    --output-zip       dist/cologne3-v2026.05.15.zip

Không tự activate. Activate qua API ai-ops sau khi push:
  POST /ops/bundles/pull body={"sourceUri":"s3://...","activate":true}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-zip", required=True, type=Path)
    parser.add_argument("--config-version", default="1")
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--training-run-id", default=None)
    parser.add_argument("--training-dataset-id", default=None)
    parser.add_argument("--training-pipeline-commit", default=None)


def _build_v2(args: argparse.Namespace) -> int:
    from bundle_tooling.packager import CommissioningError, build_v2_bundle_zip

    try:
        manifest = build_v2_bundle_zip(
            sim_config_path=args.sim_config,
            deployment_map_path=args.deployment_map,
            policy_onnx_path=args.policy_onnx,
            policy_meta_path=args.policy_meta,
            output_zip=args.output_zip,
            tenant_id=args.tenant_id,
            version=args.version,
            bundle_id=args.bundle_id,
            config_version=args.config_version,
            training_run_id=args.training_run_id,
            training_dataset_id=args.training_dataset_id,
            training_pipeline_commit=args.training_pipeline_commit,
            commissioned_by=args.commissioned_by,
            strict=not args.no_strict,
        )
    except CommissioningError as e:
        print(f"[build v2] Commissioning failed: {e}", file=sys.stderr)
        return 3

    print(
        f"[build] OK bundle_id={manifest.bundle_id} "
        f"network={manifest.network_id} version={manifest.version}"
    )
    print(f"[build] zip: {args.output_zip}")
    print(
        f"[build] checksum: {manifest.checksum[:16]}.. "
        f"topo_hash: {manifest.topology_hash[:16]}.."
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Model Bundle ZIP từ sim outputs + operator's deployment_map. "
            "Output ZIP đem upload MinIO để ai-algorithm-service tự pick up."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_v2 = subparsers.add_parser(
        "v2",
        help="Build bundle từ sim_config + deployment_map + policy artifacts.",
    )
    _add_common_args(p_v2)
    p_v2.add_argument("--sim-config", required=True, type=Path,
                      help="Đường dẫn sim's intersection_config.json.")
    p_v2.add_argument("--deployment-map", required=True, type=Path,
                      help="Đường dẫn deployment_map.json (operator điền).")
    p_v2.add_argument("--policy-onnx", required=True, type=Path)
    p_v2.add_argument("--policy-meta", required=True, type=Path)
    p_v2.add_argument("--commissioned-by", default=None)
    p_v2.add_argument("--no-strict", action="store_true",
                      help="Cho phép build dù deployment_map có validation error (chỉ DEV).")

    args = parser.parse_args()
    if args.mode == "v2":
        return _build_v2(args)
    parser.error(f"Unknown mode: {args.mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
