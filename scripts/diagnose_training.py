"""
GESA-MGMQ-PPO Diagnostic Training Script.

Runs a short training session (5-10 iterations) with comprehensive diagnostic
logging enabled. Used for Step 0 of the diagnostic protocol.

Usage:
    python scripts/diagnose_training.py --iterations 10
    python scripts/diagnose_training.py --iterations 10 --analyze-only  # Just analyze latest results
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_latest_progress(results_dir: str = "./results_mgmq"):
    """Analyze the latest progress.csv to compute diagnostic metrics."""
    results_path = Path(results_dir)
    
    # Find all result directories
    result_dirs = sorted(
        [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("mgmq_")],
        key=lambda x: x.name
    )
    
    if not result_dirs:
        print("❌ No result directories found!")
        return
    
    latest = result_dirs[-1]
    print(f"\n📂 Analyzing: {latest.name}")
    
    # Find progress.csv
    progress_files = list(latest.rglob("progress.csv"))
    if not progress_files:
        print("❌ No progress.csv found!")
        return
    
    progress_file = progress_files[0]
    print(f"📄 Progress file: {progress_file}")
    
    # Read all data
    with open(progress_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("❌ No data in progress.csv!")
        return
    
    print(f"📊 Total iterations: {len(rows)}")
    
    # Compute diagnostic metrics across all iterations
    iterations = []
    rewards = []
    policy_losses = []
    vf_losses = []
    entropies = []
    kls = []
    grad_norms = []
    kl_coeffs = []
    vf_explained_vars = []
    lrs = []
    
    for r in rows:
        iterations.append(int(r.get("training_iteration", 0)))
        rewards.append(float(r.get("env_runners/episode_reward_mean", 0)))
        policy_losses.append(float(r.get("info/learner/default_policy/learner_stats/policy_loss", 0)))
        vf_losses.append(float(r.get("info/learner/default_policy/learner_stats/vf_loss", 0)))
        entropies.append(float(r.get("info/learner/default_policy/learner_stats/entropy", 0)))
        kls.append(float(r.get("info/learner/default_policy/learner_stats/kl", 0)))
        grad_norms.append(float(r.get("info/learner/default_policy/learner_stats/grad_gnorm", 0)))
        kl_coeffs.append(float(r.get("info/learner/default_policy/learner_stats/cur_kl_coeff", 0)))
        vf_explained_vars.append(float(r.get("info/learner/default_policy/learner_stats/vf_explained_var", 0)))
        lrs.append(float(r.get("info/learner/default_policy/learner_stats/cur_lr", 0)))
    
    # Print comprehensive analysis
    print("\n" + "=" * 90)
    print("  GESA-MGMQ-PPO DIAGNOSTIC ANALYSIS — Step 0: Measurement Phase")
    print("=" * 90)
    
    # ===== REWARD TRAJECTORY =====
    print(f"\n  📊 REWARD TRAJECTORY (over {len(rows)} iterations)")
    print(f"     Start:  {rewards[0]:.2f}")
    print(f"     End:    {rewards[-1]:.2f}")
    print(f"     Max:    {max(rewards):.2f} (iter {iterations[rewards.index(max(rewards))]})")
    print(f"     Min:    {min(rewards):.2f}")
    print(f"     Change: {rewards[-1] - rewards[0]:+.2f}")
    
    # Trend analysis
    if len(rewards) >= 10:
        first_10_mean = np.mean(rewards[:10])
        last_10_mean = np.mean(rewards[-10:])
        print(f"     Trend:  first_10_avg={first_10_mean:.2f}, last_10_avg={last_10_mean:.2f}")
        if last_10_mean > first_10_mean + 5:
            print(f"     ✅ Reward IMPROVING ({last_10_mean - first_10_mean:+.2f})")
        elif last_10_mean < first_10_mean - 5:
            print(f"     ⚠️  Reward DECLINING ({last_10_mean - first_10_mean:+.2f})")
        else:
            print(f"     ⚠️  Reward FLAT (change: {last_10_mean - first_10_mean:+.2f})")
    
    # ===== TRUST REGION HEALTH =====
    print(f"\n  🛡️  TRUST REGION HEALTH")
    print(f"     KL divergence: mean={np.mean(kls):.6f}, max={max(kls):.6f}")
    print(f"     KL coeff:      start={kl_coeffs[0]:.4f}, end={kl_coeffs[-1]:.4f}")
    if kl_coeffs[-1] > 1.0:
        print(f"     🔴 KL coeff EXPLODED: {kl_coeffs[0]:.4f} → {kl_coeffs[-1]:.4f}")
        print(f"        This means PPO detected large policy updates and is heavily penalizing them!")
        print(f"        Consequence: Policy gradient signal is SUPPRESSED → flat reward")
    elif kl_coeffs[-1] > 0.5:
        print(f"     🟡 KL coeff elevated: {kl_coeffs[-1]:.4f}")
    else:
        print(f"     🟢 KL coeff healthy")
    
    # Note: clip_fraction not available in old RLlib API - need custom logging
    print(f"     ⚠️  clip_fraction: NOT AVAILABLE (requires custom callback)")
    
    # ===== POLICY DYNAMICS =====
    print(f"\n  🔄 POLICY DYNAMICS")
    print(f"     Entropy: start={entropies[0]:.4f}, end={entropies[-1]:.4f}")
    entropy_decay = (entropies[0] - entropies[-1]) / max(entropies[0], 1e-8) * 100
    print(f"     Entropy decay: {entropy_decay:.1f}%")
    if entropy_decay > 80:
        print(f"     🔴 Entropy collapsed by {entropy_decay:.0f}% → exploration almost dead")
    elif entropy_decay > 50:
        print(f"     🟡 Entropy declining significantly ({entropy_decay:.0f}%)")
    else:
        print(f"     🟢 Entropy declining normally")
    
    # ===== LOSS MAGNITUDES =====
    print(f"\n  📉 LOSS MAGNITUDES")
    print(f"     Policy loss: start={policy_losses[0]:.6f}, end={policy_losses[-1]:.6f}")
    print(f"     VF loss:     start={vf_losses[0]:.4f}, end={vf_losses[-1]:.4f}")
    
    vl_pl_ratios = [vl / max(abs(pl), 1e-8) for vl, pl in zip(vf_losses, policy_losses)]
    print(f"     VL/|PL| ratio: start={vl_pl_ratios[0]:.1f}, end={vl_pl_ratios[-1]:.1f}")
    if vl_pl_ratios[-1] > 100:
        print(f"     🔴 Value loss DOMINATES policy loss by {vl_pl_ratios[-1]:.0f}x")
        print(f"        With vf_loss_coeff, effective gradient dominated by value function")
    
    # ===== VF EXPLAINED VARIANCE =====
    print(f"\n  📈 VALUE FUNCTION")
    print(f"     vf_explained_var: start={vf_explained_vars[0]:.4f}, end={vf_explained_vars[-1]:.4f}")
    if vf_explained_vars[-1] > 0.8:
        print(f"     ✅ Value function learned well ({vf_explained_vars[-1]:.4f})")
    elif vf_explained_vars[-1] > 0.3:
        print(f"     🟡 Value function partially learned ({vf_explained_vars[-1]:.4f})")
    else:
        print(f"     🔴 Value function poorly learned ({vf_explained_vars[-1]:.4f})")
    
    # ===== GRADIENT STATISTICS =====
    print(f"\n  🔬 GRADIENTS")
    print(f"     grad_norm: mean={np.mean(grad_norms):.4f}, max={max(grad_norms):.4f}")
    print(f"     grad_norm: start={grad_norms[0]:.4f}, end={grad_norms[-1]:.4f}")
    if max(grad_norms) > 50:
        print(f"     🔴 Gradient spikes detected (max={max(grad_norms):.2f})")
    else:
        print(f"     🟢 Gradient norms stable")
    
    # ===== LEARNING RATE =====
    print(f"\n  📐 LEARNING RATE")
    print(f"     LR: start={lrs[0]:.6f}, end={lrs[-1]:.6f}")
    
    # ===== TRIGGER ASSESSMENT =====
    print(f"\n  {'─' * 70}")
    print(f"  🔍 STEP TRIGGER ASSESSMENT")
    print(f"  {'─' * 70}")
    
    # Step 1 triggers
    step1_triggers = []
    if kl_coeffs[-1] > 1.0:
        step1_triggers.append(f"kl_coeff={kl_coeffs[-1]:.2f} > 1.0")
    if vl_pl_ratios[-1] > 100:
        step1_triggers.append(f"vl/|pl|={vl_pl_ratios[-1]:.0f} >> 100")
    if step1_triggers:
        print(f"  🔴 Step 1 (Advantage & Scale): TRIGGERED")
        for t in step1_triggers:
            print(f"     → {t}")
    else:
        print(f"  🟢 Step 1 (Advantage & Scale): OK")
    
    # Step 2 triggers (need model access for std stats - not available from CSV)
    print(f"  ⚪ Step 2 (Distribution Bounds): Needs model access (run with --iterations)")
    
    # Step 3 triggers
    step3_triggers = []
    if max(grad_norms) > 50:
        step3_triggers.append(f"grad_norm_max={max(grad_norms):.2f}")
    if step3_triggers:
        print(f"  🔴 Step 3 (GNN Architecture): TRIGGERED")
        for t in step3_triggers:
            print(f"     → {t}")
    else:
        print(f"  🟢 Step 3 (GNN Architecture): OK")
    
    # Step 4 assessment
    if (last_10_mean > first_10_mean + 5 and 
        vf_explained_vars[-1] > 0.8 and
        kl_coeffs[-1] < 1.0):
        print(f"  🟢 Step 4 (Macro Stability): READY for optimization")
    else:
        print(f"  ⚪ Step 4 (Macro Stability): NOT READY (resolve earlier steps first)")
    
    print("\n" + "=" * 90)
    
    # ===== RECOMMENDATION =====
    print(f"\n  💡 RECOMMENDED ACTIONS")
    print(f"  {'─' * 70}")
    
    recommendations = []
    
    if kl_coeffs[-1] > 1.0:
        recommendations.append(
            "1. [CRITICAL] KL coefficient explosion ({:.2f}). The adaptive KL penalty is "
            "suppressing policy updates.\n"
            "     → Enable per-minibatch advantage normalization (mean=0, std=1)\n"
            "     → This stabilizes the policy gradient scale and prevents large KL".format(kl_coeffs[-1])
        )
    
    if vl_pl_ratios[-1] > 100:
        recommendations.append(
            "2. [HIGH] Value loss dominates policy loss ({:.0f}x).\n"
            "     → Reduce vf_loss_coeff (current likely too high)\n"
            "     → Or apply PopArt/scale-invariant value head".format(vl_pl_ratios[-1])
        )
    
    if entropy_decay > 70:
        recommendations.append(
            "3. [MEDIUM] Entropy collapsed by {:.0f}%.\n"
            "     → Slow down entropy annealing schedule\n"
            "     → Consider raising entropy_coeff minimum".format(entropy_decay)
        )
    
    if rewards[-1] - rewards[0] < 10 and len(rewards) > 50:
        recommendations.append(
            "4. [HIGH] Reward flat after {} iterations ({:+.2f} change).\n"
            "     → Policy is not learning. Address Steps 1-2 first.".format(
                len(rewards), rewards[-1] - rewards[0])
        )
    
    if not recommendations:
        recommendations.append("No critical issues detected. Proceed with normal training.")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 90)


def run_diagnostic_training(iterations: int = 10, config_path: str = None):
    """Run a short diagnostic training with the callback enabled."""
    import torch
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    
    from src.environment.rllib_utils import (
        SumoMultiAgentEnv, get_network_ts_ids, register_sumo_env,
    )
    from src.models.mgmq_model import MGMQTorchModel, LocalMGMQTorchModel
    from src.models.dirichlet_distribution import register_dirichlet_distribution
    from src.models.masked_softmax_distribution import register_masked_softmax_distribution
    from src.config import (
        load_model_config, get_mgmq_config, get_ppo_config, get_training_config,
        get_reward_config, get_env_config, get_network_config, is_local_gnn_enabled,
    )
    from src.callbacks.diagnostic_callback import DiagnosticCallback
    from scripts.train_mgmq_ppo import create_mgmq_ppo_config
    
    # Register models and distributions
    ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
    ModelCatalog.register_custom_model("local_mgmq_model", LocalMGMQTorchModel)
    register_dirichlet_distribution()
    register_masked_softmax_distribution()
    
    # Load config
    config = load_model_config(config_path)
    mgmq_cfg = get_mgmq_config(config)
    ppo_cfg = get_ppo_config(config)
    training_cfg = get_training_config(config)
    reward_cfg = get_reward_config(config)
    env_cfg = get_env_config(config)
    
    project_root = Path(__file__).parent.parent
    network_cfg = get_network_config(config, project_root)
    
    # Create output dir for diagnostics
    diag_output = Path("./results_mgmq/diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    diag_output.mkdir(parents=True, exist_ok=True)
    
    ts_ids = get_network_ts_ids(network_cfg["network_name"])
    num_agents = len(ts_ids)
    
    # Build env_config (same as train script)
    reward_fn = reward_cfg["reward_fn"]
    reward_weights = reward_cfg["reward_weights"]
    if reward_weights is None and isinstance(reward_fn, list) and len(reward_fn) > 1:
        reward_weights = [1.0 / len(reward_fn)] * len(reward_fn)
    
    detector_file = network_cfg.get("detector_file", "")
    additional_sumo_cmd = "--step-length 1 --lateral-resolution 0.5 --ignore-route-errors"
    if detector_file and Path(detector_file).exists():
        additional_sumo_cmd = f"-a {detector_file} {additional_sumo_cmd}"
    
    preprocessing_config = network_cfg.get("intersection_config", None)
    if preprocessing_config and not Path(preprocessing_config).exists():
        preprocessing_config = None
    
    use_local_gnn = is_local_gnn_enabled(config)
    
    env_config = {
        "net_file": network_cfg["net_file"],
        "route_file": network_cfg["route_file"],
        "use_gui": False,
        "render_mode": None,
        "num_seconds": env_cfg["num_seconds"],
        "max_green": env_cfg["max_green"],
        "min_green": env_cfg["min_green"],
        "cycle_time": env_cfg["cycle_time"],
        "yellow_time": env_cfg["yellow_time"],
        "time_to_teleport": env_cfg["time_to_teleport"],
        "single_agent": False,
        "window_size": mgmq_cfg.get("window_size", 1),
        "preprocessing_config": preprocessing_config,
        "additional_sumo_cmd": additional_sumo_cmd,
        "reward_fn": reward_fn,
        "reward_weights": reward_weights,
        "use_phase_standardizer": env_cfg.get("use_phase_standardizer", True),
        "use_neighbor_obs": use_local_gnn,
        "max_neighbors": mgmq_cfg.get("max_neighbors", 4),
        "normalize_reward": False,
        "normalizer_state_file": str(diag_output / "normalizer_state.json"),
    }
    
    mgmq_config = {
        "num_agents": num_agents,
        "ts_ids": ts_ids,
        "net_file": network_cfg["net_file"],
        "gat_hidden_dim": mgmq_cfg["gat_hidden_dim"],
        "gat_output_dim": mgmq_cfg["gat_output_dim"],
        "gat_num_heads": mgmq_cfg["gat_num_heads"],
        "graphsage_hidden_dim": mgmq_cfg["graphsage_hidden_dim"],
        "gru_hidden_dim": mgmq_cfg["gru_hidden_dim"],
        "policy_hidden_dims": mgmq_cfg.get("policy_hidden_dims", [128, 64]),
        "value_hidden_dims": mgmq_cfg.get("value_hidden_dims", [256, 128, 64]),
        "dropout": mgmq_cfg["dropout"],
        "window_size": mgmq_cfg.get("window_size", 1),
        "obs_dim": 48,
        "max_neighbors": mgmq_cfg.get("max_neighbors", 4),
    }
    
    custom_model_name = "local_mgmq_model" if use_local_gnn else "mgmq_model"
    
    # Register environment
    register_sumo_env(env_config)
    
    # Create PPO config with diagnostic callback
    ppo_config = create_mgmq_ppo_config(
        env_config=env_config,
        mgmq_config=mgmq_config,
        num_workers=min(2, training_cfg.get("num_workers", 2)),  # Use fewer workers for diagnostics
        learning_rate=ppo_cfg["learning_rate"],
        gamma=ppo_cfg["gamma"],
        lambda_=ppo_cfg["lambda_"],
        clip_param=ppo_cfg["clip_param"],
        entropy_coeff=ppo_cfg.get("entropy_coeff", 0.01),
        entropy_coeff_schedule=ppo_cfg.get("entropy_coeff_schedule", None),
        train_batch_size=ppo_cfg["train_batch_size"],
        minibatch_size=ppo_cfg["minibatch_size"],
        num_sgd_iter=ppo_cfg["num_sgd_iter"],
        grad_clip=ppo_cfg["grad_clip"],
        vf_clip_param=ppo_cfg.get("vf_clip_param", 100.0),
        vf_loss_coeff=ppo_cfg.get("vf_loss_coeff", 0.5),
        use_gpu=False,
        custom_model_name=custom_model_name,
        lr_schedule=ppo_cfg.get("lr_schedule", None),
    )
    
    # Add diagnostic callback
    ppo_config.callbacks(DiagnosticCallback)
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(500e6),
        _memory=int(500e6),
        include_dashboard=False,
        log_to_driver=True,
        logging_level="warning",
    )
    
    try:
        print(f"\n{'='*80}")
        print(f"  GESA-MGMQ-PPO DIAGNOSTIC TRAINING — {iterations} iterations")
        print(f"{'='*80}")
        print(f"  Output: {diag_output}")
        print(f"  Workers: {min(2, training_cfg.get('num_workers', 2))}")
        print(f"{'='*80}\n")
        
        # Build algorithm
        algo = ppo_config.build()
        
        for i in range(iterations):
            result = algo.train()
            
            # The DiagnosticCallback.on_train_result() will print the report
            
        # Save final diagnostic summary
        print(f"\n✓ Diagnostic training complete. Results saved to: {diag_output}")
        algo.stop()
        
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="GESA-MGMQ-PPO Diagnostic Tool")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of diagnostic training iterations")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results (no training)")
    parser.add_argument("--results-dir", type=str, default="./results_mgmq",
                        help="Results directory to analyze")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_latest_progress(args.results_dir)
    else:
        # First analyze existing data
        analyze_latest_progress(args.results_dir)
        
        # Then run diagnostic training
        print("\n\n" + "🔄" * 40)
        print("  Starting diagnostic training...")
        print("🔄" * 40 + "\n")
        run_diagnostic_training(args.iterations, args.config)


if __name__ == "__main__":
    main()
