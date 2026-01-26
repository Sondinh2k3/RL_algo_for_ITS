"""
Baseline Reward Evaluation Tool.

This script runs the SUMO simulation using the DEFAULT traffic signal logic
(defined in .net.xml or .rou.xml) but calculates the Reinforcement Learning REWARD
function in the background.

This allows for a fair comparison:
"What score (Reward) does the standard traffic light system achieve?"

Usage:
    python tools/eval_baseline_reward.py --network grid4x4 --config src/config/model_config.yml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import traci

# Add parent directory to path to import src modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
)
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_env_config,
    get_reward_config,
    get_network_config,
    is_local_gnn_enabled
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BaselineEval")

class BaselineSumoEnv(SumoMultiAgentEnv):
    """
    A specialized SUMO Environment that DOES NOT apply actions.
    It allows SUMO's internal logic to control the traffic lights,
    but still tracks metrics and calculates rewards.
    """
    def _apply_actions(self, actions):
        """
        OVERRIDE: Do not apply any actions. 
        Ignore the 'actions' dict provided by the agent.
        Let SUMO generic program (Fixed-time or Actuated) runs its course.
        """
        # We perform no traci.trafficlight.setPhase() calls here.
        pass

def evaluate_baseline(
    network_name: str = "grid4x4",
    num_episodes: int = 5,
    use_gui: bool = False,
    render: bool = False,
    output_file: str = None,
    seed: int = 42,
    config_path: Optional[str] = None,
    reward_fn_override: List[str] = None,
    reward_weights_override: List[float] = None,
):
    """
    Evaluate the reward function on the baseline (no-AI) traffic logic.
    """
    print("\n" + "="*80)
    print("BASELINE (NO-AI) REWARD EVALUATION")
    print("="*80)
    print(f"Network: {network_name}")
    print(f"Episodes: {num_episodes}")
    print("Config used: Default SUMO logic (net.xml)")
    print("="*80 + "\n")

    try:
        np.random.seed(seed)
        
        # Load YAML config for defaults
        yaml_config = load_model_config(config_path)
        yaml_env_cfg = get_env_config(yaml_config)
        yaml_reward_cfg = get_reward_config(yaml_config)
        yaml_mgmq_cfg = get_mgmq_config(yaml_config)
        
        # Get network configuration
        # Make sure we incorporate the CLI network name into the config
        # BEFORE calling get_network_config, so that we preserve any other
        # settings in the YAML (like custom route_file paths)
        if network_name:
            if "network" not in yaml_config:
                yaml_config["network"] = {}
            yaml_config["network"]["name"] = network_name
        
        network_cfg = get_network_config(yaml_config, project_root)
        
        net_file = network_cfg["net_file"]
        route_file = network_cfg["route_file"]
        preprocessing_config = network_cfg["intersection_config"]
        detector_file = network_cfg["detector_file"]
        network_name = network_cfg["network_name"]

        # Validate network files
        if not Path(net_file).exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        
        print(f"✓ Network file: {net_file}")
        print(f"✓ Route file: {route_file}")
        
        # Build additional SUMO command
        # Match the step-length used in Training/Eval for fair comparison
        additional_sumo_cmd = (
            "--step-length 0.5 " 
            "--lateral-resolution 0.5 "
            # We don't want jam threshold to teleport vehicles too aggressively in baseline
            # unless we want to match exactly. Let's keep it standard.
        )
        if detector_file and Path(detector_file).exists():
            additional_sumo_cmd = f"-a {detector_file} {additional_sumo_cmd}"
            print(f"✓ Detector file: {detector_file}")

        # Determine Reward Function
        # Priority: CLI Override > YAML Config
        final_reward_fn = reward_fn_override if reward_fn_override else yaml_reward_cfg["reward_fn"]
        final_reward_weights = reward_weights_override if reward_weights_override else yaml_reward_cfg["reward_weights"]

        print(f"✓ Reward Function: {final_reward_fn}")
        print(f"✓ Reward Weights: {final_reward_weights}")

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
            "time_to_teleport": yaml_env_cfg["time_to_teleport"],
            "single_agent": False,
            "window_size": yaml_mgmq_cfg["window_size"],
            "preprocessing_config": preprocessing_config,
            "additional_sumo_cmd": additional_sumo_cmd,
            "reward_fn": final_reward_fn,
            "reward_weights": final_reward_weights,
            "use_phase_standardizer": yaml_env_cfg["use_phase_standardizer"],
            "use_neighbor_obs": False, # Not needed for baseline eval
            "max_neighbors": 4, 
        }

        # Initialize the PASSIVE environment
        # We do NOT use register_sumo_env because we want to stick our Custom Class in directly
        # But SumoMultiAgentEnv structure assumes it's creating TrafficSignals inside.
        
        # Instantiate directly
        env = BaselineSumoEnv(**env_config)
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        episode_waiting_times = []
        episode_avg_speeds = []
        episode_total_halts = []
        
        ts_ids = get_network_ts_ids(network_name)
        per_agent_rewards = {ts_id: [] for ts_id in ts_ids}
        
        for ep in range(num_episodes):
            print(f"Starting Episode {ep+1}...")
            obs, info = env.reset(seed=seed + ep)
            done = {"__all__": False}
            total_reward = 0
            agent_rewards = {ts_id: 0 for ts_id in ts_ids}
            step_count = 0
            
            while not done.get("__all__", False):
                # We send Dummy Actions (empty dict or random)
                # The BaselineSumoEnv._apply_actions will IGNORE them.
                # However, env.step() triggers _compute_rewards which is what we want.
                dummy_actions = {ts_id: 0 for ts_id in ts_ids}
                
                # Step environment
                obs, rewards, terminateds, truncateds, info = env.step(dummy_actions)
                
                # Accumulate rewards (Calculating score of the Baseline behavior)
                for agent_id, reward in rewards.items():
                    total_reward += reward
                    if agent_id in agent_rewards:
                        agent_rewards[agent_id] += reward
                
                step_count += 1
                done = truncateds
            
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            # Store per-agent rewards
            for ts_id in ts_ids:
                if ts_id in agent_rewards:
                    per_agent_rewards[ts_id].append(agent_rewards[ts_id])
            
            # Get system metrics from the last info
            # Usually info contains stats from the last step or aggregated
            sample_info = info.get(ts_ids[0], {}) if ts_ids else {}
            
            # Note: SumoMultiAgentEnv might not aggregate system stats perfectly in info
            # But TrafficSignal class tracks them.
            
            if "system_total_waiting_time" in sample_info:
                episode_waiting_times.append(sample_info["system_total_waiting_time"])
            if "system_mean_speed" in sample_info:
                episode_avg_speeds.append(sample_info["system_mean_speed"])
            if "system_total_stopped" in sample_info:
                episode_total_halts.append(sample_info["system_total_stopped"])

            print(f"Episode {ep+1}/{num_episodes}: Total Reward={total_reward:.2f}, Steps={step_count}")

        env.close()

        # Compile Results
        results = {
            "network": network_name,
            "type": "BASELINE_NO_AI",
            "num_episodes": num_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "metrics": {},
            "episode_rewards": [float(r) for r in episode_rewards],
        }

        if episode_waiting_times:
            results["metrics"]["mean_waiting_time"] = float(np.mean(episode_waiting_times))
        if episode_avg_speeds:
            results["metrics"]["mean_avg_speed"] = float(np.mean(episode_avg_speeds))
        if episode_total_halts:
            results["metrics"]["mean_total_halts"] = float(np.mean(episode_total_halts))

        print("\n" + "="*80)
        print("BASELINE RESULTS SUMMARY")
        print("="*80)
        print(f"Mean Total Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Comparision Guideline:")
        print(f"  If Training Reward ({results['mean_reward']:.2f}) < AI Reward -> AI is improving objective")
        print(f"  If Training Reward ({results['mean_reward']:.2f}) > AI Reward -> AI is worse than static")
        print("-" * 40)
        if "metrics" in results and "mean_waiting_time" in results["metrics"]:
             print(f"Mean Waiting Time: {results['metrics']['mean_waiting_time']:.2f}")

        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        # Ensure Ray/Traci is closed if possible
        try:
            traci.close()
        except:
            pass
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline (Fixed-Time/Default) Reward"
    )
    parser.add_argument("--network", type=str, default="grid4x4",
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc"],
                        help="Network name")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--config", type=str, default="src/config/model_config.yml",
                        help="Path to model_config.yml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Allow overriding reward function CLI style to match training script inputs if needed
    parser.add_argument("--reward-fn", type=str, nargs='+', default=None,
                        help="Override reward function(s)")
    parser.add_argument("--reward-weights", type=float, nargs='+', default=None,
                        help="Override reward weights")

    args = parser.parse_args()
    
    evaluate_baseline(
        network_name=args.network,
        num_episodes=args.episodes,
        use_gui=args.gui,
        output_file=args.output,
        seed=args.seed,
        config_path=args.config,
        reward_fn_override=args.reward_fn,
        reward_weights_override=args.reward_weights
    )
