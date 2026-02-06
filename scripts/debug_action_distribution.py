#!/usr/bin/env python3
"""
Debug script to analyze action distribution and observation variance.

This script helps diagnose why green times are uniform across phases.

Usage:
    python scripts/debug_action_distribution.py --checkpoint <path_to_checkpoint>
    python scripts/debug_action_distribution.py --network grid4x4  # Run without checkpoint (random policy)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
)
from src.models.mgmq_model import MGMQTorchModel
from src.models.masked_softmax_distribution import register_masked_softmax_distribution
from src.config import load_model_config, get_network_config, get_env_config, get_mgmq_config

# Register models
ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
register_masked_softmax_distribution()


def analyze_actions(actions_dict: dict) -> dict:
    """Analyze action distribution across agents and phases."""
    stats = {}
    
    all_actions = []
    for agent_id, action in actions_dict.items():
        all_actions.append(action)
    
    all_actions = np.array(all_actions)  # [num_agents, num_phases]
    
    # Per-phase statistics
    for phase_idx in range(all_actions.shape[1]):
        phase_vals = all_actions[:, phase_idx]
        stats[f'phase_{phase_idx}'] = {
            'mean': float(np.mean(phase_vals)),
            'std': float(np.std(phase_vals)),
            'min': float(np.min(phase_vals)),
            'max': float(np.max(phase_vals)),
        }
    
    # Overall variance across phases
    phase_means = np.mean(all_actions, axis=0)
    stats['phase_variance'] = float(np.var(phase_means))
    stats['phase_std'] = float(np.std(phase_means))
    
    # Within-agent variance (how much does each agent differentiate phases)
    within_agent_vars = [np.var(agent_action) for agent_action in all_actions]
    stats['mean_within_agent_variance'] = float(np.mean(within_agent_vars))
    
    return stats


def analyze_observations(obs_dict: dict) -> dict:
    """Analyze observation variance across agents and lanes."""
    stats = {}
    
    all_obs = []
    for agent_id, obs in obs_dict.items():
        # Extract features from dict observation
        if isinstance(obs, dict):
            features = obs.get('features', obs.get('obs', None))
            if features is None:
                features = np.concatenate([v.flatten() for k, v in obs.items() if k != 'action_mask'])
        else:
            features = obs
        all_obs.append(features)
    
    all_obs = np.array(all_obs)  # [num_agents, obs_dim]
    
    # Reshape to [num_agents, 12 lanes, 4 features]
    num_agents, obs_dim = all_obs.shape
    if obs_dim == 48:  # 12 lanes * 4 features
        lane_obs = all_obs.reshape(num_agents, 12, 4)
        
        # Per-lane statistics
        for lane_idx in range(12):
            lane_vals = lane_obs[:, lane_idx, :]  # [num_agents, 4]
            stats[f'lane_{lane_idx}'] = {
                'mean_density': float(np.mean(lane_vals[:, 0])),
                'mean_queue': float(np.mean(lane_vals[:, 1])),
                'mean_occupancy': float(np.mean(lane_vals[:, 2])),
                'mean_speed': float(np.mean(lane_vals[:, 3])),
            }
        
        # Variance across lanes (should be high if traffic is unbalanced)
        lane_means = np.mean(lane_obs, axis=0)  # [12, 4]
        stats['lane_density_variance'] = float(np.var(lane_means[:, 0]))
        stats['lane_queue_variance'] = float(np.var(lane_means[:, 1]))
        
    stats['overall_obs_variance'] = float(np.var(all_obs))
    
    return stats


def run_debug_episode(env, algo=None, num_steps=10):
    """Run a debug episode and collect action/observation data."""
    obs, info = env.reset()
    
    action_history = []
    obs_history = []
    reward_history = []
    
    print("\n" + "="*80)
    print("DEBUG EPISODE")
    print("="*80)
    
    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Get actions (from policy or random)
        if algo is not None:
            actions = {}
            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(agent_obs, policy_id="default_policy")
                actions[agent_id] = action
        else:
            # Random uniform actions
            actions = {agent_id: env.action_space[agent_id].sample() 
                      for agent_id in obs.keys()}
        
        # Analyze actions
        action_stats = analyze_actions(actions)
        action_history.append(action_stats)
        
        # Analyze observations
        obs_stats = analyze_observations(obs)
        obs_history.append(obs_stats)
        
        # Print summary
        print(f"  Action phase_std: {action_stats['phase_std']:.4f}")
        print(f"  Action within_agent_var: {action_stats['mean_within_agent_variance']:.4f}")
        print(f"  Obs lane_density_var: {obs_stats.get('lane_density_variance', 0):.4f}")
        print(f"  Obs lane_queue_var: {obs_stats.get('lane_queue_variance', 0):.4f}")
        
        # Print sample action (first agent)
        sample_agent = list(actions.keys())[0]
        sample_action = actions[sample_agent]
        print(f"  Sample action ({sample_agent}): {np.round(sample_action, 3)}")
        
        # Step environment
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Collect rewards
        reward_vals = list(rewards.values())
        if reward_vals:
            reward_history.append({
                'mean': float(np.mean(reward_vals)),
                'std': float(np.std(reward_vals)),
            })
            print(f"  Reward mean: {reward_history[-1]['mean']:.4f}")
        
        # Check if episode is done
        if terminateds.get('__all__', False) or truncateds.get('__all__', False):
            print("\n[Episode ended]")
            break
    
    return action_history, obs_history, reward_history


def main():
    parser = argparse.ArgumentParser(description="Debug action distribution")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to RLlib checkpoint")
    parser.add_argument("--network", type=str, default="grid4x4",
                        help="Network name")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Number of steps to run")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml")
    
    args = parser.parse_args()
    
    # Load config
    config = load_model_config(args.config)
    project_root = Path(__file__).parent.parent
    network_cfg = get_network_config(config, project_root, args.network)
    env_cfg = get_env_config(config)
    mgmq_cfg = get_mgmq_config(config)
    
    # Get traffic signal IDs
    ts_ids = get_network_ts_ids(args.network)
    num_agents = len(ts_ids)
    
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION DEBUG")
    print("="*80)
    print(f"Network: {args.network}")
    print(f"Agents: {num_agents}")
    print(f"Checkpoint: {args.checkpoint or 'None (random policy)'}")
    
    # Environment config
    env_config = {
        "net_file": network_cfg["net_file"],
        "route_file": network_cfg["route_file"],
        "use_gui": False,
        "num_seconds": 1000,  # Short episode for debugging
        "max_green": env_cfg["max_green"],
        "min_green": env_cfg["min_green"],
        "cycle_time": env_cfg["cycle_time"],
        "yellow_time": env_cfg["yellow_time"],
        "time_to_teleport": -1,  # Disable teleport for debug
        "single_agent": False,
        "window_size": mgmq_cfg["window_size"],
        "preprocessing_config": network_cfg.get("intersection_config"),
        "reward_fn": ["halt-veh-by-detectors", "diff-departed-veh", "occupancy"],
        "reward_weights": [0.333, 0.333, 0.333],
        "use_phase_standardizer": True,
        "normalize_reward": False,  # Disable normalization for debug
    }
    
    # Register environment
    register_sumo_env(env_config)
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    try:
        # Create environment
        env = SumoMultiAgentEnv(env_config)
        
        # Load checkpoint if provided
        algo = None
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                print(f"\nLoading checkpoint: {checkpoint_path}")
                
                # Build config for loading
                from ray.rllib.algorithms.ppo import PPOConfig
                ppo_config = (
                    PPOConfig()
                    .api_stack(
                        enable_rl_module_and_learner=False,
                        enable_env_runner_and_connector_v2=False,
                    )
                    .environment(env="sumo_mgmq_v0", env_config=env_config)
                    .framework("torch")
                    .training(
                        model={
                            "custom_model": "mgmq_model",
                            "custom_model_config": {
                                "num_agents": num_agents,
                                "ts_ids": ts_ids,
                                "net_file": network_cfg["net_file"],
                                "gat_hidden_dim": mgmq_cfg["gat_hidden_dim"],
                                "gat_output_dim": mgmq_cfg["gat_output_dim"],
                                "gat_num_heads": mgmq_cfg["gat_num_heads"],
                                "graphsage_hidden_dim": mgmq_cfg["graphsage_hidden_dim"],
                                "gru_hidden_dim": mgmq_cfg["gru_hidden_dim"],
                            },
                            "custom_action_dist": "masked_softmax",
                        },
                    )
                )
                
                algo = PPO(config=ppo_config.to_dict())
                algo.restore(str(checkpoint_path))
                print("✓ Checkpoint loaded")
            else:
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
        
        # Run debug episode
        action_history, obs_history, reward_history = run_debug_episode(
            env, algo, num_steps=args.num_steps
        )
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if action_history:
            mean_phase_std = np.mean([h['phase_std'] for h in action_history])
            mean_within_var = np.mean([h['mean_within_agent_variance'] for h in action_history])
            print(f"\nAction Distribution:")
            print(f"  Mean phase std across steps: {mean_phase_std:.4f}")
            print(f"  Mean within-agent variance: {mean_within_var:.4f}")
            
            if mean_phase_std < 0.05:
                print("\n  ⚠ WARNING: Phase actions are nearly uniform!")
                print("    This suggests the policy is not differentiating between phases.")
            elif mean_phase_std < 0.1:
                print("\n  ⚠ WARNING: Phase actions show low variance!")
                print("    The policy may not be learning meaningful differences.")
            else:
                print("\n  ✓ Actions show reasonable variance across phases.")
        
        if obs_history:
            mean_density_var = np.mean([h.get('lane_density_variance', 0) for h in obs_history])
            mean_queue_var = np.mean([h.get('lane_queue_variance', 0) for h in obs_history])
            print(f"\nObservation Variance:")
            print(f"  Mean lane density variance: {mean_density_var:.4f}")
            print(f"  Mean lane queue variance: {mean_queue_var:.4f}")
            
            if mean_density_var < 0.01 and mean_queue_var < 0.01:
                print("\n  ⚠ WARNING: Observations are nearly uniform across lanes!")
                print("    Traffic may be too balanced, or observations lack directional info.")
        
        if reward_history:
            mean_reward = np.mean([h['mean'] for h in reward_history])
            print(f"\nRewards:")
            print(f"  Mean reward: {mean_reward:.4f}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if action_history and np.mean([h['phase_std'] for h in action_history]) < 0.1:
            print("""
1. Check entropy coefficient:
   - Current: 0.005 (hardcoded in train_mgmq_ppo.py:230)
   - Try reducing to 0.001 for less exploration

2. Check softmax temperature:
   - Current: 0.3 in masked_softmax_distribution.py
   - Lower temperature = sharper differentiation

3. Add direction-aware features:
   - Lane position encoding (North/South/East/West)
   - Phase-specific traffic metrics

4. Train longer:
   - Current training may be insufficient
   - Increase num_iterations to 100+
""")
        
    finally:
        env.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
