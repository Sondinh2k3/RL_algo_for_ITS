"""
MGMQ-PPO Evaluation Script.

This script evaluates trained MGMQ-PPO models on traffic signal control.
It loads a checkpoint and runs evaluation episodes, collecting metrics.

IMPORTANT: This script uses the same preprocessing configuration as training
to ensure the policy is applied correctly to the actual network.

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
import torch

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
)
from src.models.mgmq_model import MGMQTorchModel, LocalTemporalMGMQTorchModel
from src.models.dirichlet_distribution import register_dirichlet_distribution
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_env_config,
    get_reward_config,
    get_network_config,
    is_local_gnn_enabled
)


# Register custom models (same as training)
ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
ModelCatalog.register_custom_model("local_temporal_mgmq_model", LocalTemporalMGMQTorchModel)

# Register Dirichlet distribution for action space
register_dirichlet_distribution()





def load_training_config(checkpoint_path: str) -> Optional[Dict]:
    """
    Load training configuration from the experiment directory.
    
    Args:
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Training configuration dict or None if not found
    """
    checkpoint_dir = Path(checkpoint_path)
    
    # Try to find mgmq_training_config.json in parent directories
    search_dirs = [
        checkpoint_dir.parent,  # checkpoint parent
        checkpoint_dir.parent.parent,  # experiment dir
        checkpoint_dir.parent.parent.parent,  # results dir
    ]
    
    for search_dir in search_dirs:
        config_file = search_dir / "mgmq_training_config.json"
        if config_file.exists():
            print(f"✓ Found training config: {config_file}")
            with open(config_file, "r") as f:
                return json.load(f)
    
    # Also try to find in the checkpoint directory itself
    for parent in checkpoint_dir.parents:
        config_file = parent / "mgmq_training_config.json"
        if config_file.exists():
            print(f"✓ Found training config: {config_file}")
            with open(config_file, "r") as f:
                return json.load(f)
    
    # Fallback: Try to load from params.json (RLlib default checkpoint config)
    params_file = checkpoint_dir.parent / "params.json"
    if params_file.exists():
        print(f"✓ Found RLlib params.json: {params_file}")
        with open(params_file, "r") as f:
            full_config = json.load(f)
            # Extract env_config from params.json
            if "env_config" in full_config:
                return {"env_config": full_config["env_config"]}
    
    print("⚠ Warning: Training config not found, using default parameters")
    return None





def evaluate_mgmq(
    checkpoint_path: str,
    network_name: str = "grid4x4",
    num_episodes: int = 10,
    use_gui: bool = False,
    render: bool = False,
    output_file: str = None,
    seed: int = 42,
    use_training_config: bool = True,
    config_path: Optional[str] = None,
):
    """
    Evaluate a trained MGMQ-PPO model.
    
    Args:
        checkpoint_path: Path to checkpoint
        network_name: Network name
        num_episodes: Number of evaluation episodes
        use_gui: Use SUMO GUI
        render: Render environment
        output_file: Output file for results
        seed: Random seed
        use_training_config: Whether to load and use training configuration
    """
    print("\n" + "="*80)
    print("MGMQ-PPO EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Network: {network_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Use Training Config: {use_training_config}")
    print("="*80 + "\n")
    
    # Initialize Ray with memory-efficient settings (same as training)
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(500e6),  # 500MB object store
        _memory=int(500e6),  # 500MB for tasks/actors
        include_dashboard=False,  # Disable dashboard to save memory
        _temp_dir=None,
        log_to_driver=True,  # Forward worker stdout/stderr to driver terminal
        logging_level="warning",  # Reduce Ray internal logs
    )
    
    try:
        np.random.seed(seed)
        
        # Convert relative path to absolute path for PyArrow compatibility
        checkpoint_path = str(Path(checkpoint_path).resolve())
        
        # Load training config if available (moved up to infer network name)
        training_config = None
        if use_training_config:
            training_config = load_training_config(checkpoint_path)
            
        # Infer network name from training config if not explicitly provided (default is grid4x4)
        if training_config and "network_name" in training_config and network_name == "grid4x4":
            print(f"✓ Inferred network name from checkpoint: {training_config['network_name']}")
            network_name = training_config["network_name"]

        # Load YAML config for defaults
        yaml_config = load_model_config(config_path)
        yaml_env_cfg = get_env_config(yaml_config)
        yaml_reward_cfg = get_reward_config(yaml_config)
        yaml_mgmq_cfg = get_mgmq_config(yaml_config)
        
        # Get network configuration from YAML
        project_root = Path(__file__).parent.parent
        network_cfg = get_network_config(yaml_config, project_root)
        
        # Override with CLI network name (or inferred name)
        if network_name != "grid4x4":  # If specific network is needed
            override_config = {"network": {"name": network_name}}
            network_cfg = get_network_config(override_config, project_root)
        
        # Use network config from YAML (or CLI override)
        net_file = network_cfg["net_file"]
        route_file = network_cfg["route_file"]
        preprocessing_config = network_cfg["intersection_config"]
        detector_file = network_cfg["detector_file"]
        network_name = network_cfg["network_name"]  # Update network_name from config

        # Validate network files
        if not Path(net_file).exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        
        print(f"✓ Network: {network_name}")
        print(f"✓ Network file: {net_file}")
        print(f"✓ Route file: {route_file}")
        
        if preprocessing_config and Path(preprocessing_config).exists():
            print(f"✓ Preprocessing config: {preprocessing_config}")
            print("  (This ensures proper phase/intersection normalization)")
        else:
            preprocessing_config = None
            print("⚠ Warning: No preprocessing config found")
        
        # Get traffic signal IDs
        ts_ids = get_network_ts_ids(network_name)
        
        # Build environment config - use training config if available
        if training_config and "env_config" in training_config:
            # Use stored env_config from training
            stored_env_config = training_config["env_config"]
            
            # FORCE OVERRIDE PATHS from local environment to fix Cloud vs Local path issues
            # We use network paths from YAML config (already resolved above)
            print(f"⚠ Overriding network paths in config to match local environment...")
            
            # Build additional SUMO command with detector file
            # Match training config (0.5s) and apply strict network settings
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
            
            env_config = {
                "net_file": net_file,  # FROM YAML CONFIG
                "route_file": route_file,  # FROM YAML CONFIG
                "use_gui": use_gui,  # Override with current setting
                "virtual_display": None,  # Disable virtual display for local evaluation
                "render_mode": "human" if render else None,
                "num_seconds": int(stored_env_config.get("num_seconds", 8000)),
                "max_green": int(stored_env_config.get("max_green", 90)),
                "min_green": int(stored_env_config.get("min_green", 5)),
                "cycle_time": int(stored_env_config.get("cycle_time", stored_env_config.get("delta_time", 90))),
                "yellow_time": int(stored_env_config.get("yellow_time", 3)),
                # Force enable teleport for evaluation to prevent permanent deadlocks
                # Training might use -1 (disabled) to punish agents, but eval needs to keep moving
                "time_to_teleport": 500, 
                "single_agent": False,
                "window_size": int(stored_env_config.get("window_size", 1)),
                "preprocessing_config": preprocessing_config, # FROM YAML CONFIG
                "additional_sumo_cmd": additional_sumo_cmd, # FROM YAML CONFIG
                "reward_fn": stored_env_config.get("reward_fn", yaml_reward_cfg["reward_fn"]),
                "reward_weights": stored_env_config.get("reward_weights", yaml_reward_cfg["reward_weights"]),
                "use_phase_standardizer": stored_env_config.get("use_phase_standardizer", yaml_env_cfg["use_phase_standardizer"]),
                "use_neighbor_obs": stored_env_config.get("use_neighbor_obs", is_local_gnn_enabled(yaml_config)),
                "max_neighbors": stored_env_config.get("max_neighbors", yaml_mgmq_cfg["max_neighbors"]),
            }
            print("\n✓ Using environment config from training:")
            print(f"  num_seconds: {env_config['num_seconds']}")
            print(f"  cycle_time: {env_config['cycle_time']}")
            print(f"  reward_fn: {env_config['reward_fn']}")
            print(f"  reward_weights: {env_config['reward_weights']}")
            print(f"  window_size: {env_config['window_size']}")
            print(f"  use_phase_standardizer: {env_config['use_phase_standardizer']}")
            print(f"  use_neighbor_obs: {env_config['use_neighbor_obs']}")
        else:
            # Build additional SUMO command with detector file
            # Match training config (0.5s) and apply strict network settings
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
            
            # Use default config from YAML
            env_config = {
                "net_file": net_file,  # FROM YAML CONFIG
                "route_file": route_file,  # FROM YAML CONFIG
                "use_gui": use_gui,
                "virtual_display": None,
                "render_mode": "human" if render else None,
                "num_seconds": yaml_env_cfg["num_seconds"],
                "max_green": yaml_env_cfg["max_green"],
                "min_green": yaml_env_cfg["min_green"],
                "cycle_time": yaml_env_cfg["cycle_time"],
                "yellow_time": yaml_env_cfg["yellow_time"],
                "time_to_teleport": yaml_env_cfg["time_to_teleport"],
                "single_agent": False,
                "window_size": yaml_mgmq_cfg["window_size"],
                "preprocessing_config": preprocessing_config,
                "additional_sumo_cmd": additional_sumo_cmd,
                "reward_fn": yaml_reward_cfg["reward_fn"],
                "reward_weights": yaml_reward_cfg["reward_weights"],
                "use_phase_standardizer": yaml_env_cfg["use_phase_standardizer"],
                "use_neighbor_obs": is_local_gnn_enabled(yaml_config),
                "max_neighbors": yaml_mgmq_cfg["max_neighbors"],
            }
            print("\n✓ Using environment config from YAML defaults")
        
        print("")
        
        # Register environment with MultiAgentEnv wrapper
        register_sumo_env(env_config)
        
        # Load algorithm from checkpoint
        print("Loading trained model from checkpoint...")
        try:
            # Try efficient Policy loading first (faster, ignores worker config)
            # This works best if we don't need the full EnvRunner/Worker setup
            # However, compute_single_action expects Policy or Algo. 
            # Let's try full Algo load first, but catch the specific GPU error.
            if torch.cuda.is_available():
                algo = PPO.from_checkpoint(checkpoint_path)
            else:
                # If no GPU, force CPU load by modifying config
                raise RuntimeError("Force CPU fallback")
                
        except Exception as e:
            print(f"⚠ Standard load failed or CPU forced: {e}")
            print("↺ Attempting to reconstruct Algorithm with CPU configuration...")
            
            # Find config file
            checkpoint_dir = Path(checkpoint_path)
            # Checkpoint structure: /path/to/experiment/PPO_.../checkpoint_000000
            params_path = checkpoint_dir.parent / "params.json"
            
            if not params_path.exists():
                print(f"❌ Could not find params.json at {params_path}")
                raise e
            
            with open(params_path, "r") as f:
                config = json.load(f)
            
            # Extract only the essential config we need
            model_config = config.get("model", {})
            
            print("⚠ Building fresh PPOConfig with OLD API stack (ModelV2)...")
            
            from ray.rllib.algorithms.ppo import PPOConfig
            
            # Create a FRESH config - don't use from_dict which brings in problematic values
            algo_config = PPOConfig()
            
            # CRITICAL: Disable new API stack FIRST before anything else
            algo_config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
            
            # Set environment
            algo_config.environment(env="sumo_mgmq_v0", env_config=env_config)
            
            # Set resources for CPU
            algo_config.resources(num_gpus=0)
            algo_config.env_runners(num_env_runners=0)
            
            # Set framework
            algo_config.framework(config.get("framework", "torch"))
            
            # Set model config (custom_model is in here)
            algo_config.training(model=model_config)
            
            # For multi-agent: Let RLlib infer policies from environment
            # Just set the policy_mapping_fn. RLlib will create policies based on env.
            algo_config.multi_agent(
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
            )
            
            # Build the algorithm
            algo = algo_config.build()
            
            # Restore weights from checkpoint
            print(f"  Restoring weights from {checkpoint_path}...")
            algo.restore(checkpoint_path)
            
        print("✓ Model loaded successfully\n")
        
        # Create evaluation environment
        env = SumoMultiAgentEnv(**env_config)
        
        # Evaluation metrics
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
                # Get actions from policy for all agents
                actions = {}
                if not obs:  # Safety check for empty observation
                    break
                
                for agent_id in obs.keys():
                    action = algo.compute_single_action(
                        obs[agent_id],
                        policy_id="default_policy"
                    )
                    actions[agent_id] = action
                    
                # DEBUG: Check if actions are uniform (indicating unlearned policy)
                if ep == 0 and step_count == 0:
                    sample_action = list(actions.values())[0]
                    print(f"\n[DEBUG Eval] Sample Action from Policy: {sample_action}")
                    if np.allclose(sample_action, sample_action[0], atol=1e-2):
                        print("[DEBUG Eval] ⚠ WARNING: Action is uniform! Policy might be random or unlearned.")

                # Step environment
                obs, rewards, terminateds, truncateds, info = env.step(actions)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    total_reward += reward
                    if agent_id in agent_rewards:
                        agent_rewards[agent_id] += reward
                
                step_count += 1
                
                # Check if episode is done
                done = truncateds
            
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            # Store per-agent rewards
            for ts_id in ts_ids:
                if ts_id in agent_rewards:
                    per_agent_rewards[ts_id].append(agent_rewards[ts_id])
            
            # Get system metrics if available
            sample_info = info.get(ts_ids[0], {}) if ts_ids else {}
            if "system_total_waiting_time" in sample_info:
                episode_waiting_times.append(sample_info["system_total_waiting_time"])
            if "system_mean_speed" in sample_info:
                episode_avg_speeds.append(sample_info["system_mean_speed"])
            if "system_total_stopped" in sample_info:
                episode_total_halts.append(sample_info["system_total_stopped"])
            
            print(f"Episode {ep+1}/{num_episodes}: Total Reward={total_reward:.2f}, Steps={step_count}")
        
        env.close()
        
        # Calculate statistics
        results = {
            "checkpoint": checkpoint_path,
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
        
        # Per-agent statistics
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
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
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
        
        print("="*80 + "\n")
        
        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to: {output_file}")
        
        return results
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MGMQ-PPO model on traffic signal control"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--network", type=str, default="grid4x4",
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc"],
                        help="Network name")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-training-config", action="store_true",
                        help="Do not load training config, use defaults")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml (default: src/config/model_config.yml)")
    
    args = parser.parse_args()
    
    evaluate_mgmq(
        checkpoint_path=args.checkpoint,
        network_name=args.network,
        num_episodes=args.episodes,
        use_gui=args.gui,
        render=args.render,
        output_file=args.output,
        seed=args.seed,
        use_training_config=not args.no_training_config,
        config_path=args.config,
    )
