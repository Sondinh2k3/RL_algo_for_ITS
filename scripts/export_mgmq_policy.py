"""Export an MGMQ/RLlib checkpoint to policy.onnx + policy_meta.json.

This script is intended for the sim bundle pipeline. It converts a trained
RLlib PPO checkpoint into the two policy artifacts expected by
docs/Model_Bundle_Packaging_Guide.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _register_project_components() -> None:
    from ray.rllib.models import ModelCatalog

    from src.models.mgmq_model import LocalMGMQTorchModel, MGMQTorchModel
    from src.models.mlp_model import MLPTorchModel
    from src.models.masked_dirichlet import register_masked_dirichlet
    from src.models.masked_multi_categorical import register_masked_multi_categorical

    ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
    ModelCatalog.register_custom_model("local_mgmq_model", LocalMGMQTorchModel)
    ModelCatalog.register_custom_model("mlp_model", MLPTorchModel)
    register_masked_multi_categorical()
    register_masked_dirichlet()


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_training_config(checkpoint: Path, explicit: Optional[Path]) -> Optional[Dict[str, Any]]:
    if explicit:
        return _load_json(explicit)

    for parent in [checkpoint.parent, checkpoint.parent.parent, checkpoint.parent.parent.parent, *checkpoint.parents]:
        candidate = parent / "mgmq_training_config.json"
        if candidate.exists():
            return _load_json(candidate)
    return None


def _config_to_dict(obj: Any) -> Dict[str, Any]:
    cfg = getattr(obj, "config", {}) or {}
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()
    return cfg if isinstance(cfg, dict) else {}


def _model_config_from_policy(policy: Any) -> Dict[str, Any]:
    cfg = _config_to_dict(policy)
    model_cfg = (cfg.get("model") or {}).get("custom_model_config") or {}
    return dict(model_cfg)


def _safe_policy_config(policy: Any) -> Dict[str, Any]:
    return _config_to_dict(policy)


def _write_policy_meta(
    *,
    output_dir: Path,
    checkpoint: Path,
    policy: Any,
    training_config: Optional[Dict[str, Any]],
    output_names: Sequence[str],
) -> None:
    policy_cfg = _safe_policy_config(policy)
    model_cfg = dict((training_config or {}).get("mgmq_config") or {})
    if not model_cfg:
        model_cfg = _model_config_from_policy(policy)

    model_section = policy_cfg.get("model") or {}

    custom_model = model_section.get("custom_model") or "unknown"
    use_local_gnn = custom_model == "local_mgmq_model"
    base_obs_dim = int(model_cfg.get("obs_dim") or 56)
    window_size = int(model_cfg.get("window_size") or 1)
    num_actions = int(model_cfg.get("num_discrete_actions") or 7)

    input_names = (
        [
            "self_features",
            "neighbor_features",
            "neighbor_mask",
            "neighbor_directions",
            "action_mask",
        ]
        if use_local_gnn
        else ["obs", "action_mask"]
    )

    meta = {
        "use_local_gnn": use_local_gnn,
        "obs_dim": base_obs_dim * window_size,
        "base_obs_dim": base_obs_dim,
        "window_size": window_size,
        "max_neighbors": int(model_cfg.get("max_neighbors") or 4),
        "num_standard_phases": 8,
        "num_actions_per_phase": num_actions,
        "keep_action_index": num_actions // 2,
        "input_names": input_names,
        "output_name": list(output_names)[0],
        "checkpoint": checkpoint.as_posix(),
    }

    (output_dir / "policy_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _export_local_mgmq_onnx(
    *,
    policy: Any,
    output_path: Path,
    training_config: Optional[Dict[str, Any]],
    opset: int,
) -> None:
    """Export LocalMGMQTorchModel with clean inference inputs.

    RLlib's generic export_model path may fail on recent PyTorch versions due
    to dynamic_axes -> dynamic_shapes conversion. This wrapper exports the
    actual inference surface the backend needs.
    """
    import torch

    policy_cfg = _safe_policy_config(policy)
    model_section = policy_cfg.get("model") or {}
    custom_model = model_section.get("custom_model")
    if custom_model != "local_mgmq_model":
        raise RuntimeError(
            "Fallback ONNX export currently supports custom_model="
            f"'local_mgmq_model' only; checkpoint has {custom_model!r}."
        )

    model_cfg = dict((training_config or {}).get("mgmq_config") or {})
    if not model_cfg:
        model_cfg = _model_config_from_policy(policy)

    obs_dim = int(model_cfg.get("obs_dim") or 56)
    max_neighbors = int(model_cfg.get("max_neighbors") or 4)
    window_size = int(model_cfg.get("window_size") or 1)

    class LocalPolicyOnnxWrapper(torch.nn.Module):
        def __init__(self, rllib_model: torch.nn.Module):
            super().__init__()
            self.rllib_model = rllib_model

        def forward(
            self,
            self_features: torch.Tensor,
            neighbor_features: torch.Tensor,
            neighbor_mask: torch.Tensor,
            neighbor_directions: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> torch.Tensor:
            logits, _ = self.rllib_model(
                {
                    "obs": {
                        "self_features": self_features,
                        "neighbor_features": neighbor_features,
                        "neighbor_mask": neighbor_mask,
                        "neighbor_directions": neighbor_directions,
                        "action_mask": action_mask,
                    }
                },
                [],
                None,
            )
            return logits

    model = policy.model
    model.eval()
    wrapper = LocalPolicyOnnxWrapper(model).eval()

    if window_size > 1:
        self_shape = (1, window_size, obs_dim)
        neighbor_shape = (1, max_neighbors, window_size, obs_dim)
    else:
        self_shape = (1, obs_dim)
        neighbor_shape = (1, max_neighbors, obs_dim)

    dummy_inputs = (
        torch.zeros(*self_shape, dtype=torch.float32),
        torch.zeros(*neighbor_shape, dtype=torch.float32),
        torch.ones(1, max_neighbors, dtype=torch.float32),
        torch.zeros(1, max_neighbors, dtype=torch.float32),
        torch.ones(1, 8, dtype=torch.float32),
    )
    input_names = [
        "self_features",
        "neighbor_features",
        "neighbor_mask",
        "neighbor_directions",
        "action_mask",
    ]

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={
            name: {0: "batch_size"}
            for name in [*input_names, "logits"]
        },
        dynamo=False,
    )


def _validate_policy_contract(output_dir: Path) -> None:
    """Fail fast if policy.onnx and policy_meta.json disagree.

    The runtime relies on `window_size` in policy_meta.json to decide whether
    local-GNN inputs are rank-2/rank-3 or rank-3/rank-4. This catches the exact
    mismatch that causes ONNXRuntime "Invalid rank" errors.
    """
    import onnx

    meta_path = output_dir / "policy_meta.json"
    onnx_path = output_dir / "policy.onnx"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = onnx.load(onnx_path.as_posix())

    inputs = {
        item.name: [
            dim.dim_value or dim.dim_param or "?"
            for dim in item.type.tensor_type.shape.dim
        ]
        for item in model.graph.input
    }
    outputs = {item.name for item in model.graph.output}

    output_name = meta.get("output_name")
    if output_name not in outputs:
        raise RuntimeError(
            f"policy_meta.output_name={output_name!r} not found in ONNX outputs={sorted(outputs)}"
        )

    if not meta.get("use_local_gnn"):
        return

    required_inputs = [
        "self_features",
        "neighbor_features",
        "neighbor_mask",
        "neighbor_directions",
        "action_mask",
    ]
    missing = [name for name in required_inputs if name not in inputs]
    if missing:
        raise RuntimeError(f"ONNX missing local-GNN inputs: {missing}")

    window_size = int(meta.get("window_size") or 1)
    expected_ranks = {
        "self_features": 3 if window_size > 1 else 2,
        "neighbor_features": 4 if window_size > 1 else 3,
        "neighbor_mask": 2,
        "neighbor_directions": 2,
        "action_mask": 2,
    }
    bad = {
        name: {"expected_rank": rank, "actual_shape": inputs[name]}
        for name, rank in expected_ranks.items()
        if len(inputs[name]) != rank
    }
    if bad:
        raise RuntimeError(
            "ONNX input ranks do not match policy_meta contract: "
            + json.dumps(bad, ensure_ascii=False)
        )


def export_policy(args: argparse.Namespace) -> None:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'onnx'. Install it in the active venv first:\n"
            "  python -m pip install onnx"
        ) from exc

    from ray.rllib.policy.policy import Policy

    _register_project_components()

    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = _find_training_config(
        checkpoint,
        Path(args.training_config).resolve() if args.training_config else None,
    )

    # Restore the policy directly instead of PPO.from_checkpoint(). The latter
    # reconstructs EnvRunner/RolloutWorker instances and therefore needs the
    # SUMO env to be registered and startable, which is unnecessary for export.
    restored = Policy.from_checkpoint(str(checkpoint), policy_ids=[args.policy_id])
    if isinstance(restored, dict):
        if args.policy_id not in restored:
            raise ValueError(
                f"Policy '{args.policy_id}' not found. Available: {sorted(restored)}"
            )
        policy = restored[args.policy_id]
    else:
        policy = restored

    try:
        raw_export_dir = output_dir / "_rllib_onnx_export"
        if raw_export_dir.exists():
            shutil.rmtree(raw_export_dir)
        raw_export_dir.mkdir(parents=True)

        policy_onnx = output_dir / "policy.onnx"
        policy_cfg = _safe_policy_config(policy)
        custom_model = (policy_cfg.get("model") or {}).get("custom_model")

        if custom_model == "local_mgmq_model":
            _export_local_mgmq_onnx(
                policy=policy,
                output_path=policy_onnx,
                training_config=training_config,
                opset=args.opset,
            )
            output_names = ["logits"]
        else:
            output_names = ["output", "state_outs"]
            try:
                policy.export_model(str(raw_export_dir), onnx=args.opset)
                raw_model = raw_export_dir / "model.onnx"
                if not raw_model.exists():
                    raise FileNotFoundError(
                        f"RLlib did not create expected ONNX file: {raw_model}"
                    )
                shutil.copy2(raw_model, policy_onnx)
            except Exception as exc:
                raise RuntimeError(
                    "RLlib ONNX export failed and clean fallback supports only "
                    f"custom_model='local_mgmq_model' (got {custom_model!r})."
                ) from exc

        _write_policy_meta(
            output_dir=output_dir,
            checkpoint=checkpoint,
            policy=policy,
            training_config=training_config,
            output_names=output_names,
        )
        _validate_policy_contract(output_dir)
    finally:
        stop = getattr(policy, "stop", None)
        if callable(stop):
            stop()

    print(f"Exported: {output_dir / 'policy.onnx'}")
    print(f"Exported: {output_dir / 'policy_meta.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MGMQ/RLlib checkpoint to policy.onnx + policy_meta.json.",
    )
    parser.add_argument("--checkpoint", required=True, help="RLlib checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for policy artifacts.")
    parser.add_argument("--training-config", default=None, help="Optional mgmq_training_config.json path.")
    parser.add_argument("--network-id", default=None, help="Accepted for bundle workflow compatibility.")
    parser.add_argument("--version", default=None, help="Accepted for bundle workflow compatibility.")
    parser.add_argument("--policy-id", default="default_policy")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    export_policy(parse_args())
