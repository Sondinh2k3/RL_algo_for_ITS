#!/usr/bin/env python3
"""
Ablation Experiment Runner — Systematic Root-Cause Isolation.

Methodology:
  Each experiment changes EXACTLY ONE variable from the baseline.
  "Root cause là thứ mà khi bạn loại bỏ nó, collapse biến mất."

  If an experiment does NOT collapse, the changed variable was necessary
  for the collapse → likely root cause (or co-factor).

Test order (optimal):
  T0  Baseline            – reproduce the collapse (control experiment)
  T1  normalize OFF       – remove reward normalization
  T2  vf_loss_coeff 0.001 – near-zero critic weight on policy path
  T3  clip_param 0.05     – tighter trust region
  T4  vf_loss_coeff 0     – freeze critic completely
  T5  vf_share_coeff 0    – fully detach encoder from value path

Usage:
  python scripts/run_ablation.py --test T0           # run one test
  python scripts/run_ablation.py --test T0 T1 T2     # run selected tests
  python scripts/run_ablation.py --all                # run all sequentially
  python scripts/run_ablation.py --list               # show test definitions
  python scripts/run_ablation.py --summary            # summarize completed results

Each run outputs to:
  results_ablation/<test_name>_<timestamp>/
    ├── mgmq_training_config.json
    ├── diagnostics/
    │   └── diagnostic_log.json
    └── checkpoints/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────
# BASELINE: exact reproduction of latest config
# All experiments inherit from this, then override ONE field.
# ─────────────────────────────────────────────────────────
BASELINE_CLI = {
    "--iterations": "20",
    "--workers": "2",
    "--use-local-gnn": True,         # flag (bool = present/absent)
    "--seed": "42",
    "--output-dir": "./results_ablation",
}

# ─────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# Each entry: description + dict of CLI overrides (relative to baseline).
# Only the CHANGED key is listed.  Everything else = baseline.
# ─────────────────────────────────────────────────────────
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "T0_baseline": {
        "desc": "Baseline (control) — reproduce the collapse",
        "overrides": {},  # no changes
    },
    "T1_no_normalize": {
        "desc": "Disable reward normalization (normalize_reward=False)",
        "overrides": {
            "--no-normalize-reward": True,  # flag
        },
    },
    "T2_vf_loss_0001": {
        "desc": "Reduce vf_loss_coeff from 0.02 → 0.001 (near-zero critic weight)",
        "overrides": {
            # Override via temporary YAML or env-var.
            # Since vf_loss_coeff comes from YAML, we create a patched config.
            "__yaml_overrides__": {"ppo": {"vf_loss_coeff": 0.001}},
        },
    },
    "T3_tight_clip": {
        "desc": "Tighter trust region: clip_param 0.1 → 0.05, kl_target 0.03 → 0.01",
        "overrides": {
            "__yaml_overrides__": {
                "ppo": {
                    "clip_param": 0.05,
                    "kl_target": 0.01,
                },
            },
        },
    },
    "T4_freeze_critic": {
        "desc": "Freeze critic: vf_loss_coeff = 0 (policy-only sanity check)",
        "overrides": {
            "__yaml_overrides__": {"ppo": {"vf_loss_coeff": 0.0}},
        },
    },
    "T5_split_encoder": {
        "desc": "Detach encoder from value path: vf_share_coeff = 0.0",
        "overrides": {
            "--vf-share-coeff": "0.0",
        },
    },
}

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_mgmq_ppo.py"
BASE_CONFIG = PROJECT_ROOT / "src" / "config" / "model_config.yml"
ABLATION_DIR = PROJECT_ROOT / "results_ablation"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (non-destructive copy)."""
    import copy
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def create_patched_config(yaml_overrides: dict, test_name: str) -> Path:
    """Create a temporary YAML config with overrides applied on top of baseline."""
    import yaml
    with open(BASE_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    config = deep_merge(config, yaml_overrides)
    
    patched_dir = ABLATION_DIR / "configs"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_path = patched_dir / f"{test_name}.yml"
    with open(patched_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return patched_path


def build_command(test_name: str, experiment: dict) -> List[str]:
    """Build the shell command for a given experiment."""
    overrides = experiment["overrides"]
    yaml_overrides = overrides.pop("__yaml_overrides__", None)
    
    # Determine config file
    if yaml_overrides:
        config_path = create_patched_config(yaml_overrides, test_name)
    else:
        config_path = BASE_CONFIG
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{test_name}_{timestamp}"
    
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(config_path),
        "--experiment-name", exp_name,
    ]
    
    # Add baseline CLI args
    for key, val in BASELINE_CLI.items():
        if isinstance(val, bool):
            if val:
                cmd.append(key)
        else:
            cmd.extend([key, str(val)])
    
    # Add experiment-specific overrides
    for key, val in overrides.items():
        if isinstance(val, bool):
            if val:
                cmd.append(key)
        else:
            cmd.extend([key, str(val)])
    
    return cmd, exp_name


def run_experiment(test_name: str, experiment: dict, dry_run: bool = False) -> dict:
    """Run a single experiment and return summary info."""
    # Re-insert yaml_overrides if needed (build_command pops it)
    import copy
    exp_copy = copy.deepcopy(experiment)
    
    cmd, exp_name = build_command(test_name, exp_copy)
    
    print("\n" + "=" * 80)
    print(f"  ABLATION TEST: {test_name}")
    print(f"  {experiment['desc']}")
    print("=" * 80)
    print(f"  Command: {' '.join(cmd)}")
    print()
    
    if dry_run:
        print("  [DRY RUN] Skipping actual execution.")
        return {"test": test_name, "status": "dry_run", "exp_name": exp_name}
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_time
    
    status = "success" if result.returncode == 0 else "failed"
    
    summary = {
        "test": test_name,
        "desc": experiment["desc"],
        "exp_name": exp_name,
        "status": status,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save per-test summary
    summary_path = ABLATION_DIR / exp_name / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def collect_results() -> List[dict]:
    """Collect results from all completed ablation experiments."""
    results = []
    if not ABLATION_DIR.exists():
        return results
    
    for exp_dir in sorted(ABLATION_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_file = exp_dir / "ablation_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            
            # Try to extract key metrics from diagnostic log
            diag_file = exp_dir / "diagnostics" / "diagnostic_log.json"
            if diag_file.exists():
                try:
                    with open(diag_file) as f:
                        diag = json.load(f)
                    if isinstance(diag, list) and len(diag) > 0:
                        last = diag[-1]
                        summary["final_raw_reward"] = last.get("diag/raw_reward_mean", "N/A")
                        summary["final_vf_loss"] = last.get("diag/vf_loss", "N/A")
                        summary["final_vf_ev"] = last.get("diag/vf_explained_var", "N/A")
                        summary["final_grad_cosine"] = last.get("diag/grad_cosine_policy_value", "N/A")
                        
                        # Check for collapse: look at reward trend
                        rewards = [
                            d.get("diag/raw_reward_mean")
                            for d in diag
                            if d.get("diag/raw_reward_mean") is not None
                        ]
                        if len(rewards) >= 5:
                            peak = max(rewards[:len(rewards)//2 + 1])  # peak in first half
                            final = rewards[-1]
                            if peak > 0 and final < peak * 0.7:
                                summary["collapsed"] = True
                                summary["collapse_drop_pct"] = round(100 * (1 - final / peak), 1)
                            else:
                                summary["collapsed"] = False
                except Exception as e:
                    summary["diag_error"] = str(e)
            
            results.append(summary)
    return results


def print_comparison_table(results: List[dict]):
    """Print a formatted comparison table of all ablation results."""
    if not results:
        print("No results found in results_ablation/")
        return
    
    print("\n" + "=" * 120)
    print("  ABLATION EXPERIMENT COMPARISON TABLE")
    print("=" * 120)
    
    # Header
    fmt = "{:<22} {:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>12}"
    print(fmt.format(
        "Test", "Status", "Collapsed?", "RawReward", "VF_Loss",
        "VF_ExpVar", "GradCos", "Time(s)"
    ))
    print("-" * 120)
    
    for r in results:
        test = r.get("test", r.get("exp_name", "???"))[:22]
        status = "✅" if r.get("status") == "success" else "❌"
        collapsed = "🔴 YES" if r.get("collapsed") else ("🟢 NO" if r.get("collapsed") is False else "???")
        raw_reward = f"{r['final_raw_reward']:.1f}" if isinstance(r.get("final_raw_reward"), (int, float)) else "N/A"
        vf_loss = f"{r['final_vf_loss']:.1f}" if isinstance(r.get("final_vf_loss"), (int, float)) else "N/A"
        vf_ev = f"{r['final_vf_ev']:.4f}" if isinstance(r.get("final_vf_ev"), (int, float)) else "N/A"
        grad_cos = f"{r['final_grad_cosine']:.3f}" if isinstance(r.get("final_grad_cosine"), (int, float)) else "N/A"
        elapsed = f"{r.get('elapsed_seconds', 0):.0f}"
        
        print(fmt.format(test, status, collapsed, raw_reward, vf_loss, vf_ev, grad_cos, elapsed))
    
    print("=" * 120)
    
    # Interpretation guide
    collapsed_tests = [r["test"] for r in results if r.get("collapsed")]
    stable_tests = [r["test"] for r in results if r.get("collapsed") is False]
    
    if stable_tests:
        print("\n  🔑 STABILITY ACHIEVED in:", ", ".join(stable_tests))
        print("     → The variable changed in these tests is likely a root cause of collapse.")
    if collapsed_tests:
        print("\n  ❌ STILL COLLAPSED in:", ", ".join(collapsed_tests))
        print("     → The variable changed was NOT the root cause.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for root-cause isolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--test", nargs="+", choices=list(EXPERIMENTS.keys()),
                        help="Run specific test(s)")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL tests sequentially")
    parser.add_argument("--list", action="store_true",
                        help="List all test definitions and exit")
    parser.add_argument("--summary", action="store_true",
                        help="Show comparison table from completed experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override number of iterations for all tests")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of workers for all tests")
    
    args = parser.parse_args()
    
    # Override baseline iterations/workers if requested
    if args.iterations is not None:
        BASELINE_CLI["--iterations"] = str(args.iterations)
    if args.workers is not None:
        BASELINE_CLI["--workers"] = str(args.workers)
    
    if args.list:
        print("\nAblation Experiment Definitions:")
        print("=" * 70)
        for name, exp in EXPERIMENTS.items():
            print(f"\n  {name}:")
            print(f"    {exp['desc']}")
            if exp["overrides"]:
                for k, v in exp["overrides"].items():
                    if k == "__yaml_overrides__":
                        print(f"    YAML: {json.dumps(v, indent=6)}")
                    else:
                        print(f"    CLI:  {k} = {v}")
            else:
                print("    (no changes — pure baseline)")
        print()
        return
    
    if args.summary:
        results = collect_results()
        print_comparison_table(results)
        return
    
    # Determine which tests to run
    if args.all:
        test_names = list(EXPERIMENTS.keys())
    elif args.test:
        test_names = args.test
    else:
        parser.print_help()
        return
    
    # Run experiments
    all_summaries = []
    for test_name in test_names:
        experiment = EXPERIMENTS[test_name]
        summary = run_experiment(test_name, experiment, dry_run=args.dry_run)
        all_summaries.append(summary)
        
        if summary.get("status") == "failed":
            print(f"\n⚠️  {test_name} FAILED (returncode={summary['returncode']})")
            print("    Continuing with next test...\n")
    
    # Print final comparison table
    if not args.dry_run:
        results = collect_results()
        print_comparison_table(results)
    else:
        print("\n[DRY RUN] No results to display.")


if __name__ == "__main__":
    main()
