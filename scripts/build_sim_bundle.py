"""Build a training-side `.sim.zip` model bundle.

The backend/ai-ops service later composes this sim bundle with the real network
snapshot and sim-to-real mapping.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _copy_required(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing {label}: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def build_sim_bundle(args: argparse.Namespace) -> Path:
    policy_onnx = Path(args.policy_onnx).resolve()
    policy_meta = Path(args.policy_meta).resolve()
    sim_config_path = Path(args.sim_config).resolve()
    output_zip = Path(args.output_zip).resolve()
    staging_dir = Path(args.staging_dir).resolve() if args.staging_dir else (
        output_zip.parent / "staging" / output_zip.stem
    )

    for path, label in (
        (policy_onnx, "policy.onnx"),
        (policy_meta, "policy_meta.json"),
        (sim_config_path, "sim network/intersection config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if staging_dir.exists() and args.clean:
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    _copy_required(policy_onnx, staging_dir / "policy.onnx", "policy.onnx")
    _copy_required(policy_meta, staging_dir / "policy_meta.json", "policy_meta.json")
    _copy_required(sim_config_path, staging_dir / "sim_network.json", "sim_network.json")

    sim_bundle_id = args.sim_bundle_id or f"sim-{args.network_id}-{uuid.uuid4().hex[:8]}"
    manifest = {
        "sim_bundle_id": sim_bundle_id,
        "tenant_id": args.tenant_id,
        "network_id": args.network_id,
        "version": args.version,
        "sim_network_path": "sim_network.json",
        "policy_onnx_path": "policy.onnx",
        "policy_meta_path": "policy_meta.json",
        "training_run_id": args.training_run_id,
        "training_dataset_id": args.training_dataset_id,
        "training_pipeline_commit": args.training_pipeline_commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "sim_config_path": "sim_network.json",
    }

    (staging_dir / "sim_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in (
            "sim_bundle_manifest.json",
            "sim_network.json",
            "policy.onnx",
            "policy_meta.json",
        ):
            zf.write(staging_dir / rel, rel)

    return output_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training-side MGMQ .sim.zip bundle.")
    parser.add_argument("--policy-onnx", required=True)
    parser.add_argument("--policy-meta", required=True)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--network-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--training-config")
    parser.add_argument("--normalizer-state")
    parser.add_argument("--checkpoint")
    parser.add_argument("--training-run-id")
    parser.add_argument("--training-dataset-id")
    parser.add_argument("--training-pipeline-commit")
    parser.add_argument("--sim-bundle-id")
    parser.add_argument("--staging-dir")
    parser.add_argument("--clean", action="store_true", help="Delete staging dir before building.")
    return parser.parse_args()


def main() -> int:
    out_zip = build_sim_bundle(parse_args())
    print(f"Built sim bundle: {out_zip}")
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("sim_bundle_manifest.json"))
    print(f"sim_bundle_id={manifest.get('sim_bundle_id')}")
    print(f"network_id={manifest.get('network_id')}")
    print(f"version={manifest.get('version')}")
    print("required_files_ok=", all(
        name in names
        for name in ("sim_bundle_manifest.json", "policy.onnx", "policy_meta.json", "sim_network.json")
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
