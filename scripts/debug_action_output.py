#!/usr/bin/env python3
"""
Deep debug script: Analyze action distribution output of trained MGMQ model.

Loads the latest checkpoint and runs inference to check:
1. Raw logits from model
2. Action mask values  
3. Softmax probabilities (after masking + temperature)
4. Final action output per agent per step
5. Whether phases are still uniform or differentiated

Usage:
    python scripts/debug_action_output.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
)
from src.models.mgmq_model import MGMQTorchModel, LocalMGMQTorchModel
from src.models.masked_softmax_distribution import register_masked_softmax_distribution
from src.config import load_model_config, get_network_config, get_env_config, get_mgmq_config


# ==============================================================================
# Configuration
# ==============================================================================
NETWORK = "grid4x4"
# Find latest checkpoint automatically
RESULTS_DIR = Path(__file__).parent.parent / "results_mgmq"
NUM_DEBUG_STEPS = 3  # Number of environment steps to analyze


def find_latest_checkpoint():
    """Find the most recent checkpoint in results directory."""
    checkpoints = []
    for trial_dir in RESULTS_DIR.glob("mgmq_ppo_grid4x4_*/PPO_*/checkpoint_*"):
        if trial_dir.is_dir():
            checkpoints.append(trial_dir)
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def print_separator(title="", char="=", width=80):
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char*padding} {title} {char*padding}")
    else:
        print(char * width)


def analyze_single_action(action, agent_id, mask=None):
    """Analyze a single agent's action vector."""
    print(f"\n  Agent: {agent_id}")
    print(f"    Action vector: [{', '.join([f'{a:.4f}' for a in action])}]")
    print(f"    Sum: {sum(action):.6f}")
    print(f"    Min: {min(action):.6f}, Max: {max(action):.6f}")
    print(f"    Std: {np.std(action):.6f}")
    
    # Check uniformity
    n_phases = len(action)
    uniform_val = 1.0 / n_phases
    deviation = np.std(action - uniform_val)
    print(f"    Deviation from uniform (1/{n_phases}={uniform_val:.4f}): {deviation:.6f}")
    
    if mask is not None:
        valid_mask = np.array(mask)
        n_valid = int(sum(valid_mask))
        valid_actions = action[valid_mask > 0.5]
        invalid_actions = action[valid_mask < 0.5]
        
        print(f"    Mask: [{', '.join([str(int(m)) for m in mask])}]")
        print(f"    Valid phases: {n_valid}, Invalid phases: {n_phases - n_valid}")
        if len(valid_actions) > 0:
            print(f"    Valid action values:   [{', '.join([f'{a:.4f}' for a in valid_actions])}]")
            print(f"    Valid sum: {sum(valid_actions):.6f}")
            print(f"    Valid std: {np.std(valid_actions):.6f}")
        if len(invalid_actions) > 0:
            print(f"    Invalid action values: [{', '.join([f'{a:.4f}' for a in invalid_actions])}]")
            invalid_sum = sum(invalid_actions)
            print(f"    Invalid sum: {invalid_sum:.8f} {'✓ ~0' if invalid_sum < 1e-4 else '⚠ NOT ZERO!'}")


def debug_model_internals(algo, obs_dict, env):
    """Deep debug: inspect model internals (logits, mask, softmax).
    
    Uses compute_single_action first, then reads cached model internals.
    """
    print_separator("MODEL INTERNALS DEBUG")
    
    policy = algo.get_policy("default_policy")
    model = policy.model
    
    sample_agents = list(obs_dict.keys())[:3]  # Debug first 3 agents
    
    for agent_id in sample_agents:
        agent_obs = obs_dict[agent_id]
        
        print(f"\n  >>> Agent: {agent_id}")
        
        # Print observation structure
        if isinstance(agent_obs, dict):
            for k, v in agent_obs.items():
                if isinstance(v, np.ndarray):
                    print(f"    obs['{k}']: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
        
        # Get action mask from obs
        action_mask_np = None
        if isinstance(agent_obs, dict) and 'action_mask' in agent_obs:
            action_mask_np = agent_obs['action_mask']
            n_valid = int(sum(action_mask_np))
            print(f"    Action mask: [{', '.join([str(int(m)) for m in action_mask_np])}] ({n_valid}/8 valid)")
        
        # Use compute_single_action to go through the full RLlib pipeline
        action = algo.compute_single_action(agent_obs, policy_id="default_policy")
        
        # Now read model internals that were cached during forward()
        print(f"\n    === After compute_single_action ===")
        print(f"    Output action: [{', '.join([f'{a:.6f}' for a in action])}]")
        print(f"    Action sum: {sum(action):.6f}")
        
        # Read cached model output (the raw logits before distribution)
        if hasattr(model, '_last_action_mask') and model._last_action_mask is not None:
            mask = model._last_action_mask
            if mask.dim() > 1:
                mask_np = mask[0].cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()
            print(f"    Model's _last_action_mask: [{', '.join([str(int(m)) for m in mask_np])}]")
        
        # Now do a manual forward pass with torch to inspect logits
        model.eval()
        with torch.no_grad():
            # Prepare obs dict with proper tensors
            obs_tensors = {}
            for k, v in agent_obs.items():
                obs_tensors[k] = torch.FloatTensor(v).unsqueeze(0)
            
            input_dict = {"obs": obs_tensors}
            model_output, state = model(input_dict, [], None)
            
            print(f"\n    Model raw output shape: {model_output.shape}")
            
            # Split into logits and log_std (MaskedSoftmax expects 2*action_dim)
            action_dim = model_output.shape[-1] // 2
            logits = model_output[0, :action_dim].cpu()
            log_std = model_output[0, action_dim:].cpu()
            
            print(f"    Logits ({action_dim}D): [{', '.join([f'{v:.4f}' for v in logits.numpy()])}]")
            print(f"    Log_std ({action_dim}D): [{', '.join([f'{v:.4f}' for v in log_std.numpy()])}]")
            print(f"    Std = exp(log_std): [{', '.join([f'{v:.4f}' for v in torch.exp(log_std).numpy()])}]")
            
            # Logit range analysis
            print(f"\n    Logit stats: min={logits.min():.4f}, max={logits.max():.4f}, range={logits.max()-logits.min():.4f}")
            
            # Get action mask from model
            if hasattr(model, '_last_action_mask') and model._last_action_mask is not None:
                mask = model._last_action_mask.cpu()
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
            else:
                mask = torch.ones(1, action_dim)
            
            # Apply masking + softmax manually (reproduce MaskedSoftmax logic)
            MASK_VALUE = -1e9
            TEMPERATURE = 0.3
            
            # Without noise (deterministic)
            logits_masked = logits.unsqueeze(0) + (1.0 - mask) * MASK_VALUE
            probs_det = F.softmax(logits_masked / TEMPERATURE, dim=-1)
            
            print(f"\n    === Softmax Pipeline (deterministic, T={TEMPERATURE}) ===")
            print(f"    Masked logits: [{', '.join([f'{v:.4f}' if v > -1e6 else '-INF' for v in logits_masked[0].numpy()])}]")
            print(f"    Softmax output: [{', '.join([f'{v:.6f}' for v in probs_det[0].numpy()])}]")
            print(f"    Sum: {probs_det[0].sum().item():.6f}")
            
            # Check differentiation on valid phases
            valid_probs = probs_det[0][mask[0] > 0.5].numpy()
            invalid_probs = probs_det[0][mask[0] < 0.5].numpy()
            
            if len(valid_probs) > 1:
                ratio = max(valid_probs) / (min(valid_probs) + 1e-10)
                print(f"\n    Valid phases probs: [{', '.join([f'{v:.6f}' for v in valid_probs])}]")
                print(f"    Max/Min ratio (valid only): {ratio:.2f}x")
                print(f"    Valid probs std: {np.std(valid_probs):.6f}")
                if ratio < 1.5:
                    print(f"    ⚠ LOW DIFFERENTIATION: phases are nearly uniform!")
                elif ratio < 3.0:
                    print(f"    ~ MODERATE differentiation between phases")
                else:
                    print(f"    ✓ GOOD differentiation between phases")
            
            if len(invalid_probs) > 0:
                print(f"    Invalid phases probs: [{', '.join([f'{v:.8f}' for v in invalid_probs])}]")
                inv_sum = sum(invalid_probs)
                print(f"    Invalid sum: {inv_sum:.10f} {'✓ ~0' if inv_sum < 1e-6 else '⚠ NOT ZERO!'}")
            
            # Compare with T=1.0
            probs_t1 = F.softmax(logits_masked / 1.0, dim=-1)
            print(f"\n    [Compare T=1.0] Softmax: [{', '.join([f'{v:.6f}' for v in probs_t1[0].numpy()])}]")
            
            # Multiple noise samples to show stochasticity
            print(f"\n    === Stochastic samples (5 different noise draws) ===")
            for trial in range(5):
                noise = torch.randn_like(logits.unsqueeze(0)) * torch.exp(log_std.unsqueeze(0))
                logits_noisy = logits.unsqueeze(0) + noise
                logits_noisy_masked = logits_noisy + (1.0 - mask) * MASK_VALUE
                probs_noisy = F.softmax(logits_noisy_masked / TEMPERATURE, dim=-1)
                valid_noisy = probs_noisy[0][mask[0] > 0.5].numpy()
                print(f"      Sample {trial+1}: [{', '.join([f'{v:.4f}' for v in valid_noisy])}]  std={np.std(valid_noisy):.4f}")


def main():
    print_separator("MGMQ ACTION OUTPUT DEEP DEBUG")
    print(f"Network: {NETWORK}")
    print(f"Debug steps: {NUM_DEBUG_STEPS}")
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    else:
        print("⚠ No checkpoint found. Will use random policy.")
    
    # Load config
    config = load_model_config()
    project_root = Path(__file__).parent.parent
    # Override network name in config before getting network config
    if "network" not in config:
        config["network"] = {}
    config["network"]["name"] = NETWORK
    network_cfg = get_network_config(config, project_root)
    env_cfg = get_env_config(config)
    mgmq_cfg = get_mgmq_config(config)
    
    ts_ids = get_network_ts_ids(NETWORK)
    num_agents = len(ts_ids)
    print(f"Traffic signals: {num_agents} ({', '.join(ts_ids[:4])}...)")
    
    # Environment config (short simulation for debug)
    env_config = {
        "net_file": network_cfg["net_file"],
        "route_file": network_cfg["route_file"],
        "use_gui": False,
        "num_seconds": 1000,
        "max_green": env_cfg["max_green"],
        "min_green": env_cfg["min_green"],
        "cycle_time": env_cfg["cycle_time"],
        "yellow_time": env_cfg["yellow_time"],
        "time_to_teleport": 500,
        "single_agent": False,
        "window_size": mgmq_cfg["window_size"],
        "preprocessing_config": network_cfg.get("intersection_config"),
        "additional_sumo_cmd": (
            f"-a {network_cfg.get('detector_file', '')} "
            "--step-length 1 --lateral-resolution 0.5 "
            "--ignore-route-errors --tls.actuated.jam-threshold 30 "
            "--device.rerouting.adaptation-steps 18 "
            "--device.rerouting.adaptation-interval 10"
        ) if network_cfg.get("detector_file") else (
            "--step-length 1 --lateral-resolution 0.5 "
            "--ignore-route-errors"
        ),
        "reward_fn": ["halt-veh-by-detectors", "diff-departed-veh", "occupancy"],
        "reward_weights": [0.333, 0.333, 0.333],
        "use_phase_standardizer": True,
        "use_neighbor_obs": True,
        "max_neighbors": 4,
        "normalize_reward": False,
    }
    
    # Register
    ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
    ModelCatalog.register_custom_model("local_mgmq_model", LocalMGMQTorchModel)
    register_masked_softmax_distribution()
    register_sumo_env(env_config)
    
    # Init Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
    
    try:
        # Build PPO config
        mgmq_model_config = {
            "num_agents": num_agents,
            "ts_ids": ts_ids,
            "net_file": network_cfg["net_file"],
            "gat_hidden_dim": mgmq_cfg["gat_hidden_dim"],
            "gat_output_dim": mgmq_cfg["gat_output_dim"],
            "gat_num_heads": mgmq_cfg["gat_num_heads"],
            "graphsage_hidden_dim": mgmq_cfg["graphsage_hidden_dim"],
            "gru_hidden_dim": mgmq_cfg["gru_hidden_dim"],
            "policy_hidden_dims": mgmq_cfg.get("policy_hidden_dims", [256, 128]),
            "value_hidden_dims": mgmq_cfg.get("value_hidden_dims", [256, 128]),
            "dropout": mgmq_cfg["dropout"],
            "window_size": mgmq_cfg["window_size"],
            "obs_dim": 48,
            "max_neighbors": 4,
        }
        
        ppo_config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="sumo_mgmq_v0", env_config=env_config)
            .env_runners(num_env_runners=0)  # No workers for debug
            .framework("torch")
            .training(
                model={
                    "custom_model": "local_mgmq_model",
                    "custom_model_config": mgmq_model_config,
                    "vf_share_layers": False,
                    "custom_action_dist": "masked_softmax",
                },
            )
        )
        # CRITICAL: disable normalize_actions
        ppo_config.normalize_actions = False
        ppo_config.clip_actions = False
        
        # Build algorithm
        print("\nBuilding PPO algorithm...")
        algo = ppo_config.build()
        
        # Try to load checkpoint if available
        if checkpoint_path:
            print(f"Attempting to load checkpoint: {checkpoint_path}")
            try:
                algo.restore(str(checkpoint_path))
                print("✓ Checkpoint restored")
            except Exception as e:
                print(f"⚠ Could not load checkpoint: {e}")
                print("  Continuing with untrained policy (this still tests the distribution pipeline)")
        else:
            print("Using untrained (random) policy")
        
        # Create environment
        print("\nCreating SUMO environment...")
        env = SumoMultiAgentEnv(**env_config)
        
        # Reset
        obs, info = env.reset()
        
        print_separator("INITIAL OBSERVATIONS")
        
        # Show observation structure
        sample_agent = list(obs.keys())[0]
        sample_obs = obs[sample_agent]
        print(f"\nObservation type: {type(sample_obs)}")
        if isinstance(sample_obs, dict):
            for k, v in sample_obs.items():
                if isinstance(v, np.ndarray):
                    print(f"  '{k}': shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")
                else:
                    print(f"  '{k}': {type(v)}")
        
        # Collect stats across all steps
        all_step_actions = []
        all_step_masks = []
        
        for step in range(NUM_DEBUG_STEPS):
            print_separator(f"STEP {step + 1}/{NUM_DEBUG_STEPS}")
            
            # ==============================================================
            # 1. Debug model internals (logits, mask, softmax) for first step
            # ==============================================================
            if step == 0:
                debug_model_internals(algo, obs, env)
            
            # ==============================================================
            # 2. Compute actions via algo (the normal way)
            # ==============================================================
            print_separator(f"ACTIONS (Step {step + 1})", char="-")
            
            actions = {}
            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(agent_obs, policy_id="default_policy")
                actions[agent_id] = action
            
            # Analyze all agents' actions
            all_actions = np.array(list(actions.values()))  # [num_agents, 8]
            
            # Get masks if available
            masks = {}
            for agent_id, agent_obs in obs.items():
                if isinstance(agent_obs, dict) and 'action_mask' in agent_obs:
                    masks[agent_id] = agent_obs['action_mask']
            
            # Print per-agent details (first 4 agents)
            for i, (agent_id, action) in enumerate(list(actions.items())[:4]):
                mask = masks.get(agent_id, None)
                analyze_single_action(action, agent_id, mask)
            
            if len(actions) > 4:
                print(f"\n  ... ({len(actions) - 4} more agents omitted)")
            
            # ==============================================================
            # 3. Aggregate statistics
            # ==============================================================
            print_separator(f"AGGREGATE STATS (Step {step + 1})", char="-")
            
            # Per-phase mean across all agents
            phase_means = np.mean(all_actions, axis=0)
            phase_stds = np.std(all_actions, axis=0)
            
            print(f"\n  Phase means (across {num_agents} agents):")
            for p in range(len(phase_means)):
                bar = "█" * int(phase_means[p] * 100)
                print(f"    Phase {p}: mean={phase_means[p]:.4f} ±{phase_stds[p]:.4f}  {bar}")
            
            print(f"\n  Cross-phase statistics:")
            print(f"    Mean of phase means: {np.mean(phase_means):.6f}")
            print(f"    Std of phase means:  {np.std(phase_means):.6f}")
            print(f"    Max/Min ratio:       {max(phase_means) / (min(phase_means) + 1e-10):.2f}x")
            
            # Within-agent variance
            within_agent_vars = [np.var(actions[aid]) for aid in actions]
            print(f"\n  Within-agent variance (how much each agent differentiates phases):")
            print(f"    Mean: {np.mean(within_agent_vars):.6f}")
            print(f"    Min:  {np.min(within_agent_vars):.6f}")
            print(f"    Max:  {np.max(within_agent_vars):.6f}")
            
            # Uniformity test
            uniform_val = 1.0 / all_actions.shape[1]
            uniformity_score = np.mean(np.abs(all_actions - uniform_val))
            print(f"\n  Uniformity score (MAE from uniform {uniform_val:.4f}): {uniformity_score:.6f}")
            
            if uniformity_score < 0.01:
                print(f"  ⚠⚠⚠ CRITICAL: Actions are VERY UNIFORM (nearly identical phases)")
            elif uniformity_score < 0.03:
                print(f"  ⚠ WARNING: Actions show LOW differentiation")
            elif uniformity_score < 0.05:
                print(f"  ~ MODERATE differentiation between phases")
            else:
                print(f"  ✓ GOOD differentiation between phases")
            
            all_step_actions.append(all_actions)
            all_step_masks.append(masks)
            
            # Step environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Print rewards
            reward_vals = list(rewards.values())
            if reward_vals:
                print(f"\n  Rewards: mean={np.mean(reward_vals):.4f}, std={np.std(reward_vals):.4f}")
            
            if terminateds.get('__all__', False) or truncateds.get('__all__', False):
                print("\n  [Episode ended]")
                break
        
        # ==============================================================
        # FINAL SUMMARY
        # ==============================================================
        print_separator("FINAL SUMMARY")
        
        all_actions_concat = np.concatenate(all_step_actions, axis=0)
        
        print(f"\n  Total action samples: {all_actions_concat.shape[0]} (agents × steps)")
        print(f"  Action dimension: {all_actions_concat.shape[1]}")
        
        # Overall phase distribution
        overall_means = np.mean(all_actions_concat, axis=0)
        overall_stds = np.std(all_actions_concat, axis=0)
        
        print(f"\n  Overall phase distribution:")
        for p in range(len(overall_means)):
            bar_len = int(overall_means[p] * 200)
            bar = "█" * bar_len
            print(f"    Phase {p}: {overall_means[p]:.4f} ±{overall_stds[p]:.4f}  {bar}")
        
        print(f"\n  Overall cross-phase std:  {np.std(overall_means):.6f}")
        print(f"  Overall max/min ratio:    {max(overall_means) / (min(overall_means) + 1e-10):.2f}x")
        
        # Entropy of average distribution
        p_avg = np.clip(overall_means, 1e-10, 1.0)
        p_avg = p_avg / p_avg.sum()
        entropy = -np.sum(p_avg * np.log(p_avg))
        max_entropy = np.log(len(p_avg))
        print(f"\n  Entropy of avg distribution: {entropy:.4f} (max={max_entropy:.4f})")
        print(f"  Normalized entropy: {entropy/max_entropy:.4f} (1.0 = perfectly uniform)")
        
        if entropy / max_entropy > 0.98:
            print(f"\n  ⚠⚠⚠ CONCLUSION: Policy output is NEARLY UNIFORM across phases.")
            print(f"  The model is NOT differentiating between phases effectively.")
        elif entropy / max_entropy > 0.90:
            print(f"\n  ⚠ CONCLUSION: Policy shows WEAK differentiation between phases.")
        else:
            print(f"\n  ✓ CONCLUSION: Policy shows MEANINGFUL differentiation between phases.")
        
        print_separator()
        
    finally:
        try:
            env.close()
        except:
            pass
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
