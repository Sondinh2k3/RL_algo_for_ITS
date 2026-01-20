"""
MGMQ-PPO Training Script for Adaptive Traffic Signal Control.

This script trains PPO agents with MGMQ (Multi-Layer graph masking Q-Learning)
architecture for traffic signal control. The MGMQ model uses:
1. GAT (Graph Attention Network) for intersection embedding
2. GraphSAGE + Bi-GRU for network embedding
3. Joint embedding for continuous action PPO

The environment (state space, action space, reward function) remains unchanged.
Only the model architecture is enhanced with GNN layers.

Author: Son Dinh
Date: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np
import torch

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.stopper import Stopper

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
    override_config,
)
from src.models.mgmq_model import MGMQTorchModel, LocalTemporalMGMQTorchModel
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_ppo_config,
    get_training_config,
    get_reward_config,
    get_env_config,
    get_network_config,
    is_local_gnn_enabled,
    get_history_length,
)


# Register custom MGMQ models with RLlib
ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
ModelCatalog.register_custom_model("local_temporal_mgmq_model", LocalTemporalMGMQTorchModel)



class MGMQStopper(Stopper):
    """Custom stopper for MGMQ training."""
    
    def __init__(
        self, 
        max_iter: int = 1000, 
        reward_threshold: float = None,
        patience: int = 500
    ):
        """
        Args:
            max_iter: Maximum training iterations
            reward_threshold: Stop if mean reward exceeds this
            patience: Early stopping patience (iterations without improvement)
        """
        self.max_iter = max_iter
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
    
    def __call__(self, trial_id, result):
        # Get current reward
        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", float('-inf'))
        
        # Check for improvement
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Stop conditions
        if result["training_iteration"] >= self.max_iter:
            print(f"\n✓ MGMQ Training completed: reached max iterations ({self.max_iter})")
            return True
        
        if self.reward_threshold is not None and mean_reward > self.reward_threshold:
            print(f"\n✓ MGMQ Training completed: reward threshold reached ({mean_reward:.2f})")
            return True
        
        if self.no_improvement_count >= self.patience:
            print(f"\n✓ MGMQ Training completed: early stopping after {self.patience} iterations without improvement")
            return True
        
        return False
    
    def stop_all(self):
        return False


def create_mgmq_ppo_config(
    env_config: dict,
    mgmq_config: dict,
    num_workers: int = 2,
    num_envs_per_worker: int = 1,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_coeff: float = 0.01,
    clip_param: float = 0.2,
    use_gpu: bool = False,
    custom_model_name: str = "mgmq_model",
) -> PPOConfig:
    """
    Create PPO config with MGMQ custom model.
    
    Args:
        env_config: Environment configuration
        mgmq_config: MGMQ model configuration
        num_workers: Number of parallel workers
        num_envs_per_worker: Environments per worker
        learning_rate: Learning rate
        gamma: Discount factor
        lambda_: GAE lambda
        entropy_coeff: Entropy coefficient
        clip_param: PPO clip parameter
        use_gpu: Whether to use GPU
        
    Returns:
        Configured PPOConfig
    """
    config = (
        PPOConfig()
        # Use the old API stack for custom model compatibility
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="sumo_mgmq_v0", env_config=env_config)
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            # Episode-based training: collect complete episodes before updating
            # With num_seconds=8000 and cycle_time=90, each episode ~89 env steps
            rollout_fragment_length="auto",  # Auto-calculate based on batch size
            batch_mode="complete_episodes",  # Wait for full episode before training
            sample_timeout_s=3600,  # Increased timeout for SUMO simulation (1 hour)
        )
        .multi_agent(
            count_steps_by="agent_steps",  # Count agent steps (samples) instead of env steps
        )
        .training(
            lr=learning_rate,
            gamma=gamma,
            lambda_=lambda_,
            entropy_coeff=entropy_coeff,
            clip_param=clip_param,
            vf_clip_param=10.0,
            # Episode-based training: 1 episode = ~89 env steps × 16 agents = ~1424 samples
            # With 4 workers, we get ~5696 samples per iteration (4 complete episodes)
            train_batch_size=1424,  # Match single episode size
            minibatch_size=64,      # Reduced from 256 to prevent CUDA OOM
            num_sgd_iter=4,         # Fewer SGD iterations for faster updates
            grad_clip=0.5,
            # Use custom MGMQ model
            model={
                "custom_model": custom_model_name,
                "custom_model_config": mgmq_config,
                "vf_share_layers": False,  # Separate policy and value networks
            },
        )
        .resources(num_gpus=1 if use_gpu else 0)
        .debugging(log_level="WARN")  # Reduce log verbosity
    )
    
    return config


def train_mgmq_ppo(
    network_name: str = None,  # From YAML config or CLI override
    net_file: str = None,  # From YAML config
    route_file: str = None,  # From YAML config
    detector_file: str = None,  # From YAML config
    preprocessing_config: str = None,  # From YAML config
    num_iterations: int = 200,
    num_workers: int = 2,
    checkpoint_interval: int = 5,
    reward_threshold: float = None,
    experiment_name: str = None,
    use_gui: bool = False,
    use_gpu: bool = False,
    seed: int = 42,
    output_dir: str = "./results_mgmq",
    # Environment parameters from YAML config
    num_seconds: int = 8000,  # Simulation duration
    max_green: int = 90,
    min_green: int = 10,
    cycle_time: int = 90,
    yellow_time: int = 3,
    time_to_teleport: int = 500,
    use_phase_standardizer: bool = True,
    # MGMQ model hyperparameters
    gat_hidden_dim: int = 256,
    gat_output_dim: int = 128,
    gat_num_heads: int = 4,
    graphsage_hidden_dim: int = 256,
    gru_hidden_dim: int = 128,
    policy_hidden_dims: list = None,
    value_hidden_dims: list = None,
    dropout: float = 0.3,
    learning_rate: float = 3e-4,
    patience: int = 50,
    history_length: int = 4,
    reward_fn = None,  # Default: ["halt-veh-by-detectors", "diff-departed-veh"]
    reward_weights: list = None,  # Default: equal weights for all reward functions
    use_local_gnn: bool = False,  # Use LocalTemporalMGMQTorchModel with pre-packaged neighbor obs
    max_neighbors: int = 4,  # Max neighbors (K) for local GNN
):
    """
    Main training function for MGMQ-PPO.
    
    Args:
        network_name: Name of traffic network
        num_iterations: Training iterations
        num_workers: Parallel workers
        checkpoint_interval: Checkpoint frequency
        reward_threshold: Early stopping threshold
        experiment_name: Experiment name
        use_gui: Use SUMO GUI
        use_gpu: Use GPU
        seed: Random seed
        output_dir: Output directory
        gat_hidden_dim: GAT hidden dimension
        gat_output_dim: GAT output dimension per head
        gat_num_heads: Number of GAT attention heads
        graphsage_hidden_dim: GraphSAGE hidden dimension
        gru_hidden_dim: Bi-GRU hidden dimension
        policy_hidden_dims: Policy network hidden dimensions
        value_hidden_dims: Value network hidden dimensions
        dropout: Dropout rate
        learning_rate: Learning rate
        patience: Early stopping patience
    """
    if policy_hidden_dims is None:
        policy_hidden_dims = [256, 128]
    if value_hidden_dims is None:
        value_hidden_dims = [256, 128]
    if reward_fn is None:
        reward_fn = ["halt-veh-by-detectors", "diff-departed-veh"]
    
    # Set default reward weights if not provided (equal weights, normalized to sum=1)
    if reward_weights is None and isinstance(reward_fn, list) and len(reward_fn) > 1:
        reward_weights = [1.0 / len(reward_fn)] * len(reward_fn)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set experiment name
    if experiment_name is None:
        experiment_name = f"mgmq_ppo_{network_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "="*80)
    print("Modify MGMQ-PPO TRAINING")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Network: {network_name}")
    print(f"Iterations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"GPU: {use_gpu}")
    print(f"Seed: {seed}")
    print("-"*80)
    print("MGMQ Model Configuration:")
    print(f"  GAT: hidden={gat_hidden_dim}, output={gat_output_dim}, heads={gat_num_heads}")
    print(f"  GraphSAGE: hidden={graphsage_hidden_dim}")
    print(f"  Bi-GRU: hidden={gru_hidden_dim}")
    print(f"  Policy: {policy_hidden_dims}")
    print(f"  Value: {value_hidden_dims}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  History Length: {history_length}")
    print(f"  Reward Function: {reward_fn}")
    print(f"  Reward Weights: {reward_weights}")
    if use_local_gnn:
        print(f"  Local GNN: ENABLED (neighbors={max_neighbors})")
    print("="*80 + "\n")
    
    # Initialize Ray with memory-efficient settings
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(200e6),  # 500MB object store (reduced for low-memory systems)
        _memory=int(200e6),  # 500MB for tasks/actors
        include_dashboard=False,  # Disable dashboard to save memory
        _temp_dir=None,
        log_to_driver=True,  # Forward worker stdout/stderr to driver terminal
        logging_level="warning",  # Reduce Ray internal logs, but keep user prints
    )
    
    try:
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Network files are now provided from YAML config or CLI override
        # Validate that required files exist
        if not Path(net_file).exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        
        # Validate route files
        for rf in route_file.split(","):
            if rf.strip() and not Path(rf.strip()).exists():
                raise FileNotFoundError(f"Route file not found: {rf}")
        
        print(f"✓ Network: {network_name}")
        print(f"✓ Network file: {net_file}")
        print(f"✓ Route file: {route_file}")
        
        if preprocessing_config and Path(preprocessing_config).exists():
            print(f"✓ Preprocessing config: {preprocessing_config}")
        else:
            preprocessing_config = None
            print("⚠ No preprocessing config found")
        print("")
        
        # Get traffic signal IDs
        ts_ids = get_network_ts_ids(network_name)
        num_agents = len(ts_ids)
        print(f"✓ Traffic signals: {num_agents} ({', '.join(ts_ids[:4])}{'...' if len(ts_ids) > 4 else ''})\n")
        
        # Phase standardizer: maps model's 4-value action to actual signal phases
        # When movements can't be identified, uses fallback heuristics based on phase patterns
        use_phase_standardizer = True
        print(f"✓ Phase Standardizer: {'ENABLED' if use_phase_standardizer else 'DISABLED'}")
        
        # Build additional SUMO command with detector file
        additional_sumo_cmd = (
            "--step-length 0.5 "
            "--lateral-resolution 0.5 "
            "--ignore-route-errors "
            "--tls.actuated.jam-threshold 30 "
            "--device.rerouting.adaptation-steps 18 "
            "--device.rerouting.adaptation-interval 10"
        )
        if detector_file and Path(detector_file).exists():
            additional_sumo_cmd = f"-a {detector_file} {additional_sumo_cmd}"
            print(f"✓ Detector file: {detector_file}")
        
        # Environment configuration - using values from YAML config
        env_config = {
            "net_file": net_file,
            "route_file": route_file,
            "use_gui": use_gui,
            "render_mode": "rgb_array" if use_gui else None,
            "num_seconds": num_seconds,  # From config (default: 3600s = 1 hour)
            "max_green": max_green,
            "min_green": min_green,
            "cycle_time": cycle_time,  # Agent makes exactly one decision per traffic light cycle
            "yellow_time": yellow_time,
            "time_to_teleport": time_to_teleport,  # -1 to disable teleporting
            "single_agent": False,  # Multi-agent mode for grid4x4
            "window_size": history_length,
            "preprocessing_config": preprocessing_config,
            # SUMO params: step-length=0.1 for smoother simulation
            "additional_sumo_cmd": additional_sumo_cmd,
            "reward_fn": reward_fn,
            "reward_weights": reward_weights,  # Weights for combining multiple reward functions
            "use_phase_standardizer": use_phase_standardizer,
            # Local GNN config
            "use_neighbor_obs": use_local_gnn,  # Enable pre-packaged neighbor observation
            "max_neighbors": max_neighbors,
        }
        
        # MGMQ model configuration
        mgmq_config = {
            "num_agents": num_agents,
            "ts_ids": ts_ids,
            "net_file": net_file,  # CRITICAL: Pass net_file for building network adjacency matrix
            "gat_hidden_dim": gat_hidden_dim,
            "gat_output_dim": gat_output_dim,
            "gat_num_heads": gat_num_heads,
            "graphsage_hidden_dim": graphsage_hidden_dim,
            "gru_hidden_dim": gru_hidden_dim,
            "policy_hidden_dims": policy_hidden_dims,
            "value_hidden_dims": value_hidden_dims,
            "dropout": dropout,
            "window_size": history_length,
            # Local GNN specific params
            "obs_dim": 48,  # 4 features * 12 detectors
            "max_neighbors": max_neighbors,
        }
        
        # Select custom model based on use_local_gnn flag
        custom_model_name = "local_temporal_mgmq_model" if use_local_gnn else "mgmq_model"
        
        # Register environment
        register_sumo_env(env_config)
        
        # Create PPO config with MGMQ model
        ppo_config = create_mgmq_ppo_config(
            env_config=env_config,
            mgmq_config=mgmq_config,
            num_workers=num_workers,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            custom_model_name=custom_model_name,
        )
        
        # Create stopper
        stopper = MGMQStopper(
            max_iter=num_iterations,
            reward_threshold=reward_threshold,
            patience=patience,
        )
        
        # Get absolute path for storage
        storage_path = output_dir.resolve()
        
        # Create Tuner
        tuner = tune.Tuner(
            "PPO",
            param_space=ppo_config.to_dict(),
            run_config=tune.RunConfig(  # Use tune.RunConfig instead of air.RunConfig
                name=experiment_name,
                storage_path=str(storage_path),  # Must be absolute path
                stop=stopper,
                checkpoint_config=tune.CheckpointConfig(  # Use tune.CheckpointConfig
                    checkpoint_frequency=checkpoint_interval,
                    num_to_keep=5,
                    checkpoint_score_attribute="env_runners/episode_reward_mean",
                    checkpoint_score_order="max",
                ),
                verbose=1,
            ),
        )
        
        # Run training
        print("Starting MGMQ-PPO training...\n")
        print("Model Architecture:")
        print("  Input -> GAT (Intersection Embedding) -> GraphSAGE+BiGRU (Network Embedding)")
        print("       -> Joint Embedding -> Policy/Value Networks -> Action\n")
        
        results = tuner.fit()
        
        # Get best results
        best_checkpoint = results.get_best_result(
            metric="env_runners/episode_reward_mean", 
            mode="max"
        ).checkpoint
        best_result = results.get_best_result(
            metric="env_runners/episode_reward_mean", 
            mode="max"
        )
        
        print("\n" + "="*80)
        print("MGMQ-PPO TRAINING COMPLETED")
        print("="*80)
        print(f"Best Checkpoint: {best_checkpoint}")
        best_reward = best_result.metrics.get("env_runners", {}).get("episode_reward_mean", 0)
        print(f"Best Episode Reward Mean: {best_reward:.2f}")
        print("="*80 + "\n")
        
        # Save configuration
        config_file = output_dir / experiment_name / "mgmq_training_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump({
                "experiment_name": experiment_name,
                "network_name": network_name,
                "num_iterations": num_iterations,
                "num_workers": num_workers,
                "use_gpu": use_gpu,
                "seed": seed,
                "mgmq_config": mgmq_config,
                "env_config": {k: str(v) if isinstance(v, Path) else v for k, v in env_config.items()},
                "best_checkpoint": str(best_checkpoint),
                "best_reward": float(best_reward),
            }, f, indent=2)
        
        print(f"✓ Configuration saved to: {config_file}")
        print(f"✓ Results saved to: {output_dir / experiment_name}")
        
        return best_checkpoint, best_result
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MGMQ-PPO agents for adaptive traffic signal control"
    )
    
    # Config file argument - load defaults from YAML file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml (default: src/config/model_config.yml)")
    
    # Basic arguments
    parser.add_argument("--network", type=str, default=None,
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc"],
                        help="Network name")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Checkpoint interval")
    parser.add_argument("--reward-threshold", type=float, default=None,
                        help="Stop if reward exceeds threshold")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    
    # MGMQ model arguments
    parser.add_argument("--gat-hidden-dim", type=int, default=None,
                        help="GAT hidden dimension")
    parser.add_argument("--gat-output-dim", type=int, default=None,
                        help="GAT output dimension per head")
    parser.add_argument("--gat-num-heads", type=int, default=None,
                        help="Number of GAT attention heads")
    parser.add_argument("--graphsage-hidden-dim", type=int, default=None,
                        help="GraphSAGE hidden dimension")
    parser.add_argument("--gru-hidden-dim", type=int, default=None,
                        help="Bi-GRU hidden dimension")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience")
    parser.add_argument("--history-length", type=int, default=None,
                        help="Observation history length (window size)")
    parser.add_argument("--reward-fn", type=str, nargs='+', default=None,
                        help="Reward function(s) for training. Can specify multiple. Available: diff-waiting-time, queue, average-speed, pressure, halt-veh-by-detectors, diff-departed-veh")
    parser.add_argument("--reward-weights", type=float, nargs='+', default=None,
                        help="Weights for reward functions. Must match number of reward functions. Default: equal weights.")
    
    # Local GNN arguments
    parser.add_argument("--use-local-gnn", action="store_true",
                        help="Use LocalTemporalMGMQTorchModel with pre-packaged neighbor observations")
    parser.add_argument("--max-neighbors", type=int, default=None,
                        help="Maximum neighbors (K) for local GNN. Default: 4")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_model_config(args.config)
    
    # Extract configs with defaults from file
    mgmq_cfg = get_mgmq_config(config)
    ppo_cfg = get_ppo_config(config)
    training_cfg = get_training_config(config)
    reward_cfg = get_reward_config(config)
    
    # Get network configuration from YAML
    project_root = Path(__file__).parent.parent
    network_cfg = get_network_config(config, project_root)
    
    # CLI --network override: merge network name into full config to preserve route_files
    if args.network:
        # Merge CLI network name into the full config (preserves route_files from YAML)
        config["network"]["name"] = args.network
        network_cfg = get_network_config(config, project_root)
    
    # Get environment configuration from YAML
    env_cfg = get_env_config(config)
    
    # CLI args override config file (if provided)
    train_mgmq_ppo(
        network_name=network_cfg["network_name"],
        net_file=network_cfg["net_file"],
        route_file=network_cfg["route_file"],
        detector_file=network_cfg["detector_file"],
        preprocessing_config=network_cfg["intersection_config"],
        num_iterations=args.iterations if args.iterations is not None else training_cfg["num_iterations"],
        num_workers=args.workers if args.workers is not None else training_cfg["num_workers"],
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else training_cfg["checkpoint_interval"],
        reward_threshold=args.reward_threshold,
        experiment_name=args.experiment_name,
        use_gui=args.gui,
        use_gpu=args.gpu or training_cfg["use_gpu"],
        seed=args.seed if args.seed is not None else training_cfg["seed"],
        output_dir=args.output_dir or training_cfg["output_dir"],
        # Environment config from YAML
        num_seconds=env_cfg["num_seconds"],
        max_green=env_cfg["max_green"],
        min_green=env_cfg["min_green"],
        cycle_time=env_cfg["cycle_time"],
        yellow_time=env_cfg["yellow_time"],
        time_to_teleport=env_cfg["time_to_teleport"],
        use_phase_standardizer=env_cfg["use_phase_standardizer"],
        # MGMQ model config
        gat_hidden_dim=args.gat_hidden_dim if args.gat_hidden_dim is not None else mgmq_cfg["gat_hidden_dim"],
        gat_output_dim=args.gat_output_dim if args.gat_output_dim is not None else mgmq_cfg["gat_output_dim"],
        gat_num_heads=args.gat_num_heads if args.gat_num_heads is not None else mgmq_cfg["gat_num_heads"],
        graphsage_hidden_dim=args.graphsage_hidden_dim if args.graphsage_hidden_dim is not None else mgmq_cfg["graphsage_hidden_dim"],
        gru_hidden_dim=args.gru_hidden_dim if args.gru_hidden_dim is not None else mgmq_cfg["gru_hidden_dim"],
        dropout=args.dropout if args.dropout is not None else mgmq_cfg["dropout"],
        learning_rate=args.learning_rate if args.learning_rate is not None else ppo_cfg["learning_rate"],
        patience=args.patience if args.patience is not None else training_cfg["patience"],
        history_length=args.history_length if args.history_length is not None else mgmq_cfg["window_size"],
        reward_fn=args.reward_fn or reward_cfg["reward_fn"],
        reward_weights=args.reward_weights or reward_cfg["reward_weights"],
        use_local_gnn=args.use_local_gnn or is_local_gnn_enabled(config),
        max_neighbors=args.max_neighbors if args.max_neighbors is not None else mgmq_cfg["max_neighbors"],
    )

