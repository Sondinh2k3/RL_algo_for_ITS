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
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

import numpy as np
import torch

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.stopper import Stopper
from ray.tune import register_trainable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
    override_config,
)
from src.models.mgmq_model import MGMQTorchModel, LocalMGMQTorchModel
from src.models.mlp_model import MLPTorchModel
from src.models.masked_multi_categorical import register_masked_multi_categorical
from src.models.masked_dirichlet import register_masked_dirichlet
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_ppo_config,
    get_scaling_config,
    get_training_config,
    get_reward_config,
    get_env_config,
    get_network_config,
    get_history_length,
    is_local_gnn_enabled,
)
from src.algorithm.mgmq_ppo import MGMQPPO, MGMQPPOConfig, MGMQPPOTorchPolicy
from src.callbacks.diagnostic_callback import DiagnosticCallback


# Register custom MGMQ models with RLlib
ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
ModelCatalog.register_custom_model("local_mgmq_model", LocalMGMQTorchModel)
ModelCatalog.register_custom_model("mlp_model", MLPTorchModel)

register_masked_multi_categorical()
register_masked_dirichlet()


def _auto_num_workers() -> int:
    """Pick a sensible default for `num_env_runners`.

    Leaves 2 cores for the driver/learner and caps at 8 so we don't oversubscribe
    laptops. Override by setting `training.num_workers` in the YAML config.
    """
    return max(1, min((os.cpu_count() or 4) - 2, 8))


def derive_runtime_scaling(
    num_agents: int,
    num_seconds: int,
    cycle_time: int,
    scaling_cfg: Dict[str, Any],
    num_iterations_cfg: Any,
    num_workers_cfg: Any,
) -> Dict[str, int]:
    """Derive batch / minibatch / iteration count from network shape.

    Returns a dict with: train_batch_size, minibatch_size, num_iterations,
    num_workers, episode_steps, episodes_per_iter.

    The scaling rules:
        train_batch_size = num_agents · T_episode · target_episodes_per_iter
        minibatch_size   = train_batch_size // minibatch_per_iter
        num_iterations   = ceil(target_agent_steps / train_batch_size)
    """
    episode_steps = max(1, num_seconds // max(1, cycle_time))
    default_eps = max(1, int(scaling_cfg["target_episodes_per_iter"]))
    # Use a larger episode multiplier only for tiny networks (<=3 agents).
    episodes_per_iter = default_eps if num_agents <= 3 else 2
    train_batch_size = num_agents * episode_steps * episodes_per_iter

    mb_per_iter = max(1, int(scaling_cfg["minibatch_per_iter"]))
    minibatch_size = max(1, train_batch_size // mb_per_iter)

    if num_iterations_cfg in (None, "auto"):
        num_iterations = max(
            1,
            math.ceil(int(scaling_cfg["target_agent_steps"]) / train_batch_size),
        )
    else:
        num_iterations = int(num_iterations_cfg)

    if num_workers_cfg in (None, "auto"):
        num_workers = _auto_num_workers()
    else:
        num_workers = int(num_workers_cfg)

    return {
        "train_batch_size": train_batch_size,
        "minibatch_size": minibatch_size,
        "num_iterations": num_iterations,
        "num_workers": num_workers,
        "episode_steps": episode_steps,
        "episodes_per_iter": episodes_per_iter,
    }


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


class NormalizerStateCallback:
    """Callback to save/restore reward normalizer state with checkpoints.
    
    This ensures that when training is resumed, the reward normalization
    statistics (mean, variance, count) are restored to maintain consistency.
    
    The normalizer state is saved to a JSON file alongside the mgmq_training_config.json.
    """
    
    def __init__(self, output_dir: Path, experiment_name: str):
        """
        Args:
            output_dir: Base output directory for training results
            experiment_name: Name of the experiment
        """
        self.state_file = output_dir / experiment_name / "normalizer_state.json"
    
    def save_state(self, env):
        """Save normalizer state from environment to file."""
        try:
            state = env.get_normalizer_state()
            if state:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                print(f"✓ Saved normalizer state to: {self.state_file}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save normalizer state: {e}")
    
    def restore_state(self, env):
        """Restore normalizer state from file to environment."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                env.set_normalizer_state(state)
                return True
            return False
        except Exception as e:
            print(f"⚠ Warning: Failed to restore normalizer state: {e}")
            return False


def _action_dist_for_mode(action_mode: str) -> dict:
    """Map action_mode to the RLlib custom_action_dist registration name."""
    if action_mode == "discrete_adjustment":
        return {"custom_action_dist": "masked_multi_categorical"}
    if action_mode == "cycle_level_continuous":
        return {"custom_action_dist": "masked_dirichlet"}
    # Default: ratio mode uses the Gaussian + softmax MaskedSoftmax distribution.
    return {"custom_action_dist": "masked_softmax"}


def create_mgmq_ppo_config(
    env_config: dict,
    mgmq_config: dict,
    num_workers: int = 2,
    num_envs_per_worker: int = 1,
    learning_rate: float = 5e-5,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    kl_target: float = 0.01,
    kl_coeff_floor: float = 0.01,
    entropy_coeff: float = 0.01,
    # entropy_coeff_schedule: list = None,
    train_batch_size: int = 4096,
    minibatch_size: int = 256,
    num_sgd_iter: int = 10,
    grad_clip: float = 10.0,
    vf_clip_param: float = 100.0,
    vf_loss_coeff: float = 0.01,
    use_gpu: bool = False,
    custom_model_name: str = "mgmq_model",
    lr_schedule: list = None,
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
        clip_param: PPO clip parameter
        entropy_coeff: Entropy coefficient for exploration (from config file)
        use_gpu: Whether to use GPU
        
    Returns:
        Configured PPOConfig
    """
    config = (
        MGMQPPOConfig()
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
            kl_target=kl_target,
            # Symmetric Δvalue clip applied in MGMQPPOTorchPolicy.loss().
            vf_clip_param=vf_clip_param,
            vf_loss_coeff=vf_loss_coeff,
            use_gae=True,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_sgd_iter,
            grad_clip=grad_clip,
            # Use custom MGMQ model
            model={
                "custom_model": custom_model_name,
                "custom_model_config": mgmq_config,
                "vf_share_layers": False,  # Separate policy and value networks
                # Custom action distributions with proper masking:
                # - ratio:                 MaskedSoftmax (Gaussian -> softmax on simplex)
                # - cycle_level_continuous: MaskedDirichlet (true simplex sampling)
                # - discrete_adjustment:   MaskedMultiCategorical (8 independent categoricals)
                **_action_dist_for_mode(mgmq_config.get("action_mode", "ratio")),
            },
        )
        .resources(num_gpus=1 if use_gpu else 0)
        .debugging(log_level="WARN")  # Reduce log verbosity
    )
    # Continuous simplex modes (ratio, cycle_level_continuous) must keep
    # normalize_actions=False so RLlib does not rescale the [0, 1] simplex
    # output into [-1, 1].
    if mgmq_config.get("action_mode", "ratio") in ("ratio", "cycle_level_continuous"):
        config.normalize_actions = False
    
    # CRITICAL: Set KL coeff floor to prevent trust-region collapse
    # Without this, kl_coeff can decay to 0 via adaptive KL halving,
    # disabling the trust-region constraint entirely.
    config["kl_coeff_floor"] = kl_coeff_floor
    
    # Add diagnostic callback for comprehensive training monitoring
    config.callbacks(DiagnosticCallback)
    
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
    # Resume training from previous experiment
    resume_path: str = None,  # Path to experiment dir to resume from
    # Environment parameters from YAML config
    num_seconds: int = 8000,  # Simulation duration
    max_green: int = 90,
    min_green: int = 5,
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
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    kl_target: float = 0.01,
    kl_coeff_floor: float = 0.01,
    entropy_coeff: float = 0.01,
    # entropy_coeff_schedule: list = None,
    train_batch_size: int = 4096,
    minibatch_size: int = 256,
    num_sgd_iter: int = 10,
    grad_clip: float = 10.0,
    vf_clip_param: float = 100.0,
    vf_loss_coeff: float = 0.5,
    # lr_schedule: list = None,
    patience: int = 50,
    history_length: int = 1,
    reward_fn = None,  # Default: ["halt-veh-by-detectors", "diff-departed-veh"]
    reward_weights: list = None,  # Default: equal weights for all reward functions
    use_local_gnn: bool = True,  # Use LocalMGMQTorchModel (star-graph) instead of global graph
    use_mlp: bool = False,  # Use MLPTorchModel (no GNN) as diagnostic baseline
    max_neighbors: int = 4,  # Max neighbors (K) for local GNN
    # Action mode
    action_mode: str = "ratio",      # "ratio", "cycle_level_continuous", or "discrete_adjustment"
    green_time_step: int = 5,        # Step size for discrete adjustment (legacy, unused with new 7-action)
    # Ablation / experiment overrides
    normalize_reward: bool = True,   # Scale-only return-based normalization
    clip_rewards: float = 10.0,      # Clip normalized rewards to [-clip, +clip]
    vf_share_coeff: float = 1.0,     # 1.0 = shared encoder (baseline), 0.0 = detach value
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
        reward_fn = ["halt-veh-by-detectors", "diff-departed-veh", "diff-waiting-time"]
    
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
    
    # Auto-detect GPU: if user requested GPU but none available, fallback to CPU
    if use_gpu and not torch.cuda.is_available():
        print("⚠ --gpu requested but no GPU found (torch.cuda.is_available()=False)")
        print("  → Falling back to CPU training")
        use_gpu = False
    
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
    print(f"  Action Mode: {action_mode}")
    print("-"*80)
    print("PPO Hyperparameters:")
    print(f"  gamma: {gamma}, lambda: {lambda_}, clip_param: {clip_param}")
    print(f"  entropy_coeff: {entropy_coeff}")
    # print(f"  entropy_coeff_schedule: {entropy_coeff_schedule}")
    print(f"  train_batch_size: {train_batch_size}, minibatch_size: {minibatch_size}")
    print(f"  num_sgd_iter: {num_sgd_iter}")
    print(f"  grad_clip: {grad_clip}")
    print(f"  vf_clip_param: {vf_clip_param}, vf_loss_coeff: {vf_loss_coeff}")
    print(f"  kl_target: {kl_target}, kl_coeff_floor: {kl_coeff_floor}")
    # print(f"  lr_schedule: {lr_schedule}")
    if use_local_gnn:
        print(f"  Local GNN: ENABLED (neighbors={max_neighbors})")
    print(f"  normalize_reward: {normalize_reward}, clip_rewards: {clip_rewards}")
    print(f"  vf_share_coeff: {vf_share_coeff}")
    print("="*80 + "\n")
    
    # Initialize Ray with memory-efficient settings
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(500e6),  # 500MB object store (reduced for low-memory systems)
        _memory=int(500e6),  # 500MB for tasks/actors
        include_dashboard=False,  # Disable dashboard to save memory
        _temp_dir=None,
        log_to_driver=True,  # Forward worker stdout/stderr to driver terminal
        logging_level="warning",  # Reduce Ray internal logs, but keep user prints
    )
    
    try:
        # Register custom MGMQ-PPO algorithm with Ray Tune
        register_trainable("MGMQPPO", MGMQPPO)
        
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
            "--step-length 1 "
            "--lateral-resolution 0.5 "
            "--ignore-route-errors "
            "--tls.actuated.jam-threshold 30 "
            "--no-internal-links true "
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
            "use_neighbor_obs": use_local_gnn,
            "max_neighbors": max_neighbors,
            # Action mode
            "action_mode": action_mode,
            "green_time_step": green_time_step,
            # Reward normalization: scale-only return-based RMS (see env.py).
            # `reward_norm_gamma` must match the PPO gamma so that the running
            # return tracked here matches the discount used in the GAE target.
            "normalize_reward": normalize_reward,
            "clip_rewards": clip_rewards,
            "reward_norm_gamma": gamma,
            # Resume training will reload RMS state + return accumulators here.
            "normalizer_state_file": str(output_dir / experiment_name / "normalizer_state.json"),
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
            # Local GNN fallback obs_dim when obs_space is unavailable.
            # 56 = 48 lane features (4 × 12 detectors) + 8 green-time ratios.
            "obs_dim": 56,
            "max_neighbors": max_neighbors,
            # Gradient isolation for shared encoder (ablation parameter)
            "vf_share_coeff": vf_share_coeff,
            # Action mode: "ratio" or "discrete_adjustment"
            "action_mode": action_mode,
            # Number of discrete actions per phase (7 = {-15,-10,-5,0,+5,+10,+15})
            "num_discrete_actions": 7,
        }
        
        # Select custom model based on flags
        if use_mlp:
            custom_model_name = "mlp_model"
            print(f"  Model: MLP BASELINE (no GNN)")
        elif use_local_gnn:
            custom_model_name = "local_mgmq_model"
            print(f"  Model: LOCAL MGMQ (star-graph, K={max_neighbors} neighbors)")
        else:
            custom_model_name = "mgmq_model"
        
        # Register environment
        register_sumo_env(env_config)
        
        # Create PPO config with MGMQ model
        ppo_config = create_mgmq_ppo_config(
            env_config=env_config,
            mgmq_config=mgmq_config,
            num_workers=num_workers,
            learning_rate=learning_rate,
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            kl_target=kl_target,
            kl_coeff_floor=kl_coeff_floor,
            entropy_coeff=entropy_coeff,
            # entropy_coeff_schedule=entropy_coeff_schedule,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_sgd_iter=num_sgd_iter,
            grad_clip=grad_clip,
            vf_clip_param=vf_clip_param,
            vf_loss_coeff=vf_loss_coeff,
            use_gpu=use_gpu,
            custom_model_name=custom_model_name,
            # lr_schedule=lr_schedule,
        )
        
        # Create stopper
        stopper = MGMQStopper(
            max_iter=num_iterations,
            reward_threshold=reward_threshold,
            patience=patience,
        )
        
        # Get absolute path for storage
        storage_path = output_dir.resolve()
        
        # Resume training or create new Tuner
        if resume_path:
            # Resume from previous experiment
            resume_path = Path(resume_path).resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume path not found: {resume_path}")
            
            print(f"\n{'='*80}")
            print("RESUMING TRAINING FROM PREVIOUS EXPERIMENT")
            print(f"{'='*80}")
            print(f"Resume path: {resume_path}")
            print(f"New max iterations: {num_iterations}")
            print(f"{'='*80}\n")
            
            # Use Tuner.restore() to resume from checkpoint
            tuner = tune.Tuner.restore(
                path=str(resume_path),
                trainable="MGMQPPO",
                resume_unfinished=True,  # Resume trials that haven't finished
                resume_errored=True,  # Retry trials that errored
                param_space=ppo_config.to_dict(),  # Allow parameter overrides
                # Override stopper with new iteration count
                restart_errored=False,  # Don't restart errored trials from scratch
            )
            # Note: stopper is reinitialized, so the iteration count starts fresh
            # The model weights are restored from the last checkpoint
        else:
            # Create new Tuner for fresh training using MGMQ-PPO
            # (includes per-minibatch advantage normalization + clip_fraction tracking)
            tuner = tune.Tuner(
                "MGMQPPO",
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
        
        # ========================================
        # SAVE CONFIG EARLY (before training starts)
        # This ensures config is saved even if training is interrupted
        # ========================================
        config_file = output_dir / experiment_name / "mgmq_training_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        initial_config = {
            "experiment_name": experiment_name,
            "network_name": network_name,
            "num_iterations": num_iterations,
            "num_workers": num_workers,
            "use_gpu": use_gpu,
            "seed": seed,
            "mgmq_config": mgmq_config,
            "env_config": {k: str(v) if isinstance(v, Path) else v for k, v in env_config.items()},
            "training_status": "in_progress",
            "started_at": datetime.now().isoformat(),
        }
        with open(config_file, "w") as f:
            json.dump(initial_config, f, indent=2)
        print(f"✓ Training config saved to: {config_file}")
        
        # Run training
        mode_str = "RESUMING" if resume_path else "Starting"
        print(f"{mode_str} MGMQ-PPO training...\n")
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
        
        # Update configuration with training results
        config_file = output_dir / experiment_name / "mgmq_training_config.json"
        # Load existing config and update with results
        existing_config = initial_config.copy()
        existing_config.update({
            "training_status": "completed",
            "completed_at": datetime.now().isoformat(),
            "best_checkpoint": str(best_checkpoint),
            "best_reward": float(best_reward),
        })
        with open(config_file, "w") as f:
            json.dump(existing_config, f, indent=2)
        
        print(f"✓ Configuration updated with results: {config_file}")
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
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc", "test", "test1", "osm", "leductho"],
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
    
    # Resume training argument
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to experiment directory to resume training from. "
                             "Example: results_mgmq/mgmq_ppo_grid4x4_20260127_003407")
    
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
    
    # Model arguments
    parser.add_argument("--use-local-gnn", action="store_true", default=True,
                        help="Use LocalMGMQTorchModel with star-graph topology (default: True)")
    parser.add_argument("--use-mlp", action="store_true",
                        help="Use MLPTorchModel (no GNN) as baseline diagnostic")
    parser.add_argument("--max-neighbors", type=int, default=None,
                        help="Maximum neighbors (K) for local GNN. Default: 4")
    parser.add_argument("--action-mode", type=str, default=None,
                        choices=["ratio", "cycle_level_continuous", "discrete_adjustment"],
                        help="Action mode. Overrides `environment.action_mode` in config.")
    
    # Ablation / experiment arguments
    parser.add_argument("--no-normalize-reward", action="store_true",
                        help="Disable reward normalization (ablation test)")
    parser.add_argument("--clip-rewards", type=float, default=None,
                        help="Clip normalized rewards to [-clip, +clip]. Default: 10.0")
    parser.add_argument("--vf-share-coeff", type=float, default=None,
                        help="Value-function encoder sharing coefficient. 1.0=shared (baseline), 0.0=detached")
    
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
    scaling_cfg = get_scaling_config(config)

    # Derive batch size, minibatch size, num_iterations and num_workers from
    # network shape so the same YAML works across networks of any size.
    num_agents_for_scaling = len(get_network_ts_ids(network_cfg["network_name"]))
    runtime = derive_runtime_scaling(
        num_agents=num_agents_for_scaling,
        num_seconds=env_cfg["num_seconds"],
        cycle_time=env_cfg["cycle_time"],
        scaling_cfg=scaling_cfg,
        num_iterations_cfg=(
            args.iterations if args.iterations is not None else training_cfg["num_iterations"]
        ),
        num_workers_cfg=(
            args.workers if args.workers is not None else training_cfg["num_workers"]
        ),
    )

    print(
        f"\n[runtime] N_agents={num_agents_for_scaling}, "
        f"T_episode={runtime['episode_steps']}, "
        f"episodes/iter={runtime['episodes_per_iter']} "
        f"→ train_batch_size={runtime['train_batch_size']}, "
        f"minibatch_size={runtime['minibatch_size']}, "
        f"num_iterations={runtime['num_iterations']}, "
        f"num_workers={runtime['num_workers']}"
    )

    train_mgmq_ppo(
        network_name=network_cfg["network_name"],
        net_file=network_cfg["net_file"],
        route_file=network_cfg["route_file"],
        detector_file=network_cfg["detector_file"],
        preprocessing_config=network_cfg["intersection_config"],
        num_iterations=runtime["num_iterations"],
        num_workers=runtime["num_workers"],
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else training_cfg["checkpoint_interval"],
        reward_threshold=args.reward_threshold,
        experiment_name=args.experiment_name,
        use_gui=args.gui,
        use_gpu=args.gpu or training_cfg["use_gpu"],
        seed=args.seed if args.seed is not None else training_cfg["seed"],
        output_dir=args.output_dir or training_cfg["output_dir"],
        resume_path=args.resume,
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
        gamma=ppo_cfg["gamma"],
        lambda_=ppo_cfg["lambda_"],
        clip_param=ppo_cfg["clip_param"],
        kl_target=ppo_cfg["kl_target"],
        kl_coeff_floor=ppo_cfg["kl_coeff_floor"],
        entropy_coeff=ppo_cfg["entropy_coeff"],
        train_batch_size=runtime["train_batch_size"],
        minibatch_size=runtime["minibatch_size"],
        num_sgd_iter=ppo_cfg["num_sgd_iter"],
        grad_clip=ppo_cfg["grad_clip"],
        vf_clip_param=ppo_cfg["vf_clip_param"],
        vf_loss_coeff=ppo_cfg["vf_loss_coeff"],
        patience=args.patience if args.patience is not None else training_cfg["patience"],
        history_length=args.history_length if args.history_length is not None else mgmq_cfg["window_size"],
        reward_fn=args.reward_fn or reward_cfg["reward_fn"],
        reward_weights=args.reward_weights or reward_cfg["reward_weights"],
        use_local_gnn=args.use_local_gnn or is_local_gnn_enabled(config),
        use_mlp=args.use_mlp,
        max_neighbors=args.max_neighbors if args.max_neighbors is not None else mgmq_cfg["max_neighbors"],
        action_mode=args.action_mode or env_cfg.get("action_mode", "ratio"),
        green_time_step=env_cfg.get("green_time_step", 5),
        # Reward normalization: scale-only return-based RMS (see env.py).
        normalize_reward=(not args.no_normalize_reward) and env_cfg["reward_norm_enabled"],
        clip_rewards=args.clip_rewards if args.clip_rewards is not None else env_cfg["reward_norm_clip"],
        vf_share_coeff=args.vf_share_coeff if args.vf_share_coeff is not None else mgmq_cfg.get("vf_share_coeff", 1.0),
    )

