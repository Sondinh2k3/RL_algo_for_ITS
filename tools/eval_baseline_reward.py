"""
Baseline Reward Evaluation Tool.

This script runs the SUMO simulation using the DEFAULT traffic signal logic
(defined in .net.xml) but calculates the Reinforcement Learning REWARD
function in the background.

This allows for a fair comparison:
"What score (Reward) does the standard traffic light system achieve?"

IMPORTANT: This script uses the SAME environment configuration as training/evaluation
to ensure the reward calculation is consistent.

The ONLY difference from eval_mgmq_ppo.py is:
- Instead of loading AI policy and computing actions, we use fixed_ts=True
- This lets SUMO's default traffic light program control the signals

Author: Son Dinh
Date: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
)
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_env_config,
    get_reward_config,
    get_network_config,
    is_local_gnn_enabled
)


def evaluate_baseline(
    network_name: str = "grid4x4",
    num_episodes: int = 5,
    use_gui: bool = False,
    render: bool = False,
    output_file: str = None,
    seed: int = 42,
    config_path: Optional[str] = None,
):
    """
    Evaluate baseline (no AI) traffic signal control.
    
    Uses the SAME environment as training/evaluation, but with fixed_ts=True
    so SUMO's default traffic light program runs without AI intervention.
    
    Args:
        network_name: Network name (grid4x4, zurich, etc.)
        num_episodes: Number of evaluation episodes
        use_gui: Use SUMO GUI for visualization
        render: Render environment
        output_file: Output file for results
        seed: Random seed
        config_path: Path to model_config.yml
    """
    print("\n" + "="*80)
    print("BASELINE (NO-AI) EVALUATION")
    print("="*80)
    print(f"Network: {network_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Config: {config_path or 'default (src/config/model_config.yml)'}")
    print("="*80 + "\n")
    
    np.random.seed(seed)
    
    # Load YAML config (same as training/eval_mgmq_ppo.py)
    yaml_config = load_model_config(config_path)
    yaml_env_cfg = get_env_config(yaml_config)
    yaml_reward_cfg = get_reward_config(yaml_config)
    yaml_mgmq_cfg = get_mgmq_config(yaml_config)
    
    # Get network configuration from YAML
    project_root = Path(__file__).parent.parent
    network_cfg = get_network_config(yaml_config, project_root)
    
    # Override with CLI network name if different
    if network_name != network_cfg.get("network_name", "grid4x4"):
        yaml_config["network"]["name"] = network_name
        network_cfg = get_network_config(yaml_config, project_root)
    
    # Get network files
    net_file = network_cfg["net_file"]
    route_file = network_cfg["route_file"]
    preprocessing_config = network_cfg["intersection_config"]
    detector_file = network_cfg["detector_file"]
    network_name = network_cfg["network_name"]
    
    # Validate network files
    if not Path(net_file).exists():
        raise FileNotFoundError(f"Network file not found: {net_file}")
    
    print(f"✓ Network: {network_name}")
    print(f"✓ Network file: {net_file}")
    print(f"✓ Route file: {route_file}")
    
    if preprocessing_config and Path(preprocessing_config).exists():
        print(f"✓ Preprocessing config: {preprocessing_config}")
    else:
        preprocessing_config = None
        print("⚠ Warning: No preprocessing config found")
    
    # Get traffic signal IDs
    ts_ids = get_network_ts_ids(network_name)
    print(f"✓ Traffic signals: {len(ts_ids)} agents")
    
    # Build additional SUMO command (SAME as training/eval_mgmq_ppo.py)
    additional_sumo_cmd = (
        "--step-length 1 "
        "--lateral-resolution 0.5 "
        "--ignore-route-errors "
        "--tls.actuated.jam-threshold 30 "
        "--device.rerouting.adaptation-steps 18 "
        "--device.rerouting.adaptation-interval 10"
    )
    if detector_file and Path(detector_file).exists():
        additional_sumo_cmd = f"-a {detector_file} {additional_sumo_cmd}"
        print(f"✓ Detector file: {detector_file}")
    
    # Print reward config
    print(f"✓ Reward Function: {yaml_reward_cfg['reward_fn']}")
    print(f"✓ Reward Weights: {yaml_reward_cfg['reward_weights']}")
    
    # Build environment config (SAME as eval_mgmq_ppo.py, but with fixed_ts=True)
    env_config = {
        "net_file": net_file,
        "route_file": route_file,
        "use_gui": use_gui,
        "virtual_display": None,
        "render_mode": "human" if render else None,
        "num_seconds": yaml_env_cfg["num_seconds"],
        "max_green": yaml_env_cfg["max_green"],
        "min_green": yaml_env_cfg["min_green"],
        "cycle_time": yaml_env_cfg["cycle_time"],
        "yellow_time": yaml_env_cfg["yellow_time"],
        # Match eval_mgmq_ppo.py: Force enable teleport for evaluation to prevent permanent deadlocks
        "time_to_teleport": 500,
        "single_agent": False,
        "window_size": yaml_mgmq_cfg["window_size"],
        "preprocessing_config": preprocessing_config,
        "additional_sumo_cmd": additional_sumo_cmd,
        "reward_fn": yaml_reward_cfg["reward_fn"],
        "reward_weights": yaml_reward_cfg["reward_weights"],
        "use_phase_standardizer": yaml_env_cfg["use_phase_standardizer"],
        "use_neighbor_obs": False,  # Not needed for baseline
        "max_neighbors": yaml_mgmq_cfg["max_neighbors"],
        # CRITICAL DIFFERENCE: fixed_ts=True means NO agent actions are applied
        # SUMO's default traffic light program (from .net.xml) controls the signals
        "fixed_ts": True,
    }
    
    expected_steps = yaml_env_cfg["num_seconds"] // yaml_env_cfg["cycle_time"]
    print(f"\n✓ Cycle time: {yaml_env_cfg['cycle_time']}s")
    print(f"✓ Simulation time: {yaml_env_cfg['num_seconds']}s")
    print(f"✓ Expected steps per episode: ~{expected_steps}")
    print("")
    
    # Create environment (SAME as eval_mgmq_ppo.py)
    env = SumoMultiAgentEnv(**env_config)
    
    # Evaluation metrics (SAME structure as eval_mgmq_ppo.py)
    episode_rewards = []
    episode_lengths = []
    episode_waiting_times = []
    episode_avg_speeds = []
    episode_total_halts = []
    per_agent_rewards = {ts_id: [] for ts_id in ts_ids}
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = {"__all__": False}
        total_reward = 0
        agent_rewards = {ts_id: 0 for ts_id in ts_ids}
        step_count = 0
        
        while not done.get("__all__", False):
            # Baseline: Empty actions dict
            # With fixed_ts=True, the simulator will NOT apply any actions
            # and let SUMO's default traffic light program run
            actions = {}
            
            # Step environment (SAME as eval_mgmq_ppo.py)
            obs, rewards, terminateds, truncateds, info = env.step(actions)
            
            # Accumulate rewards (SAME as eval_mgmq_ppo.py)
            for agent_id, reward in rewards.items():
                total_reward += reward
                if agent_id in agent_rewards:
                    agent_rewards[agent_id] += reward
            
            step_count += 1
            
            # Check if episode is done (SAME as eval_mgmq_ppo.py)
            done = truncateds
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Store per-agent rewards (SAME as eval_mgmq_ppo.py)
        for ts_id in ts_ids:
            if ts_id in agent_rewards:
                per_agent_rewards[ts_id].append(agent_rewards[ts_id])
        
        # Get system metrics if available (SAME as eval_mgmq_ppo.py)
        sample_info = info.get(ts_ids[0], {}) if ts_ids else {}
        if "system_total_waiting_time" in sample_info:
            episode_waiting_times.append(sample_info["system_total_waiting_time"])
        if "system_mean_speed" in sample_info:
            episode_avg_speeds.append(sample_info["system_mean_speed"])
        if "system_total_stopped" in sample_info:
            episode_total_halts.append(sample_info["system_total_stopped"])
        
        print(f"Episode {ep+1}/{num_episodes}: Total Reward={total_reward:.2f}, Steps={step_count}")
    
    env.close()
    
    # Calculate statistics (SAME as eval_mgmq_ppo.py)
    results = {
        "network": network_name,
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
    }
    
    # Per-agent statistics (SAME as eval_mgmq_ppo.py)
    per_agent_stats = {}
    for ts_id in ts_ids:
        if per_agent_rewards[ts_id]:
            per_agent_stats[ts_id] = {
                "mean_reward": float(np.mean(per_agent_rewards[ts_id])),
                "std_reward": float(np.std(per_agent_rewards[ts_id])),
            }
    results["per_agent_stats"] = per_agent_stats
    
    if episode_waiting_times:
        results["mean_waiting_time"] = float(np.mean(episode_waiting_times))
        results["std_waiting_time"] = float(np.std(episode_waiting_times))
    
    if episode_avg_speeds:
        results["mean_avg_speed"] = float(np.mean(episode_avg_speeds))
        results["std_avg_speed"] = float(np.std(episode_avg_speeds))
    
    if episode_total_halts:
        results["mean_total_halts"] = float(np.mean(episode_total_halts))
        results["std_total_halts"] = float(np.std(episode_total_halts))
    
    # Print results (SAME format as eval_mgmq_ppo.py)
    print("\n" + "="*80)
    print("BASELINE RESULTS")
    print("="*80)
    print(f"Mean Total Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    
    if "mean_waiting_time" in results:
        print(f"Mean Waiting Time: {results['mean_waiting_time']:.2f} ± {results.get('std_waiting_time', 0):.2f}")
    if "mean_avg_speed" in results:
        print(f"Mean Average Speed: {results['mean_avg_speed']:.2f} ± {results.get('std_avg_speed', 0):.2f}")
    if "mean_total_halts" in results:
        print(f"Mean Total Halts: {results['mean_total_halts']:.2f} ± {results.get('std_total_halts', 0):.2f}")
    
    print("\nPer-Agent Mean Rewards:")
    for ts_id, stats in per_agent_stats.items():
        print(f"  {ts_id}: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    
    print("\n" + "-"*40)
    print("COMPARISON GUIDELINE:")
    print("-"*40)
    print(f"Baseline Mean Reward: {results['mean_reward']:.2f}")
    print(f"If AI's mean_reward > {results['mean_reward']:.2f} → AI is BETTER")
    print(f"If AI's mean_reward < {results['mean_reward']:.2f} → AI is WORSE")
    print("="*80 + "\n")
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline (no AI) traffic signal control"
    )
    parser.add_argument("--network", type=str, default="grid4x4",
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc"],
                        help="Network name")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml (default: src/config/model_config.yml)")
    
    args = parser.parse_args()
    
    evaluate_baseline(
        network_name=args.network,
        num_episodes=args.episodes,
        use_gui=args.gui,
        render=args.render,
        output_file=args.output,
        seed=args.seed,
        config_path=args.config,
    )
