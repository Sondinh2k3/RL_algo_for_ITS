"""
Configuration Loader for MGMQ-PPO.

This module provides utilities to load and merge configuration from YAML files.
Supports loading model_config.yml and simulation.yml files.

Author: Son Dinh
Date: 2025
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_config_dir() -> Path:
    """Get the config directory path."""
    return Path(__file__).parent


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model configuration from model_config.yml.
    
    Args:
        config_path: Optional custom path. Default: src/config/model_config.yml
        
    Returns:
        Model configuration dictionary
    """
    if config_path is None:
        config_path = get_config_dir() / "model_config.yml"
    return load_yaml_config(str(config_path))


def load_simulation_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load simulation configuration from simulation.yml.
    
    Args:
        config_path: Optional custom path. Default: src/config/simulation.yml
        
    Returns:
        Simulation configuration dictionary
    """
    if config_path is None:
        config_path = get_config_dir() / "simulation.yml"
    return load_yaml_config(str(config_path))


def get_mgmq_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MGMQ model configuration from loaded config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        MGMQ model configuration dictionary ready for model initialization
    """
    mgmq = config.get("mgmq", {})
    
    return {
        "gat_hidden_dim": mgmq.get("gat", {}).get("hidden_dim", 256),
        "gat_output_dim": mgmq.get("gat", {}).get("output_dim", 128),
        "gat_num_heads": mgmq.get("gat", {}).get("num_heads", 4),
        "graphsage_hidden_dim": mgmq.get("graphsage", {}).get("hidden_dim", 256),
        "gru_hidden_dim": mgmq.get("gru", {}).get("hidden_dim", 128),
        "policy_hidden_dims": mgmq.get("policy", {}).get("hidden_dims", [256, 128]),
        "value_hidden_dims": mgmq.get("value", {}).get("hidden_dims", [256, 128]),
        "dropout": mgmq.get("dropout", 0.3),
        "window_size": mgmq.get("history_length", 4),
        "obs_dim": mgmq.get("local_gnn", {}).get("obs_dim", 48),
        "max_neighbors": mgmq.get("local_gnn", {}).get("max_neighbors", 4),
    }


def get_ppo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PPO training configuration from loaded config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        PPO configuration dictionary
    """
    ppo = config.get("ppo", {})
    
    return {
        "learning_rate": ppo.get("learning_rate", 3e-5),
        "gamma": ppo.get("gamma", 0.99),
        "lambda_": ppo.get("lambda_", 0.95),
        "entropy_coeff": ppo.get("entropy_coeff", 0.01),
        "entropy_coeff_schedule": ppo.get("entropy_coeff_schedule", None),
        "clip_param": ppo.get("clip_param", 0.2),
        "vf_clip_param": ppo.get("vf_clip_param", 100.0),
        "vf_loss_coeff": ppo.get("vf_loss_coeff", 0.5),
        "train_batch_size": ppo.get("train_batch_size", 2048),
        "minibatch_size": ppo.get("minibatch_size", 256),
        "num_sgd_iter": ppo.get("num_sgd_iter", 10),
        "grad_clip": ppo.get("grad_clip", 5.0),
    }


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training settings from loaded config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        Training configuration dictionary
    """
    training = config.get("training", {})
    
    return {
        "num_iterations": training.get("num_iterations", 200),
        "num_workers": training.get("num_workers", 2),
        "num_envs_per_worker": training.get("num_envs_per_worker", 1),
        "checkpoint_interval": training.get("checkpoint_interval", 20),
        "patience": training.get("patience", 50),
        "seed": training.get("seed", 42),
        "use_gpu": training.get("use_gpu", False),
        "output_dir": training.get("output_dir", "./results_mgmq"),
    }


def get_reward_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract reward configuration from loaded config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        Reward configuration dictionary
    """
    reward = config.get("reward", {})
    
    reward_fn = reward.get("functions", ["halt-veh-by-detectors", "diff-departed-veh"])
    reward_weights = reward.get("weights", None)
    
    # Auto-compute equal weights if not provided
    if reward_weights is None and isinstance(reward_fn, list) and len(reward_fn) > 1:
        reward_weights = [1.0 / len(reward_fn)] * len(reward_fn)
    
    return {
        "reward_fn": reward_fn,
        "reward_weights": reward_weights,
    }


def get_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract environment configuration from loaded config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        Environment configuration dictionary
    """
    env = config.get("environment", {})
    
    return {
        "num_seconds": env.get("num_seconds", 8000),
        "max_green": env.get("max_green", 90),
        "min_green": env.get("min_green", 5),
        "cycle_time": env.get("cycle_time", 90),
        "yellow_time": env.get("yellow_time", 3),
        "time_to_teleport": env.get("time_to_teleport", -1),
        "use_phase_standardizer": env.get("use_phase_standardizer", True),
    }


def get_network_config(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extract network configuration from loaded config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        project_root: Project root directory for resolving relative paths.
                     If None, paths are returned as-is.
        
    Returns:
        Network configuration dictionary with resolved paths
    """
    network = config.get("network", {})
    
    network_name = network.get("name", "grid4x4")
    base_path = network.get("base_path")
    
    # Resolve base_path
    if base_path is None:
        if project_root:
            base_path = project_root / "network" / network_name
        else:
            base_path = Path("network") / network_name
    else:
        base_path = Path(base_path)
        if project_root and not base_path.is_absolute():
            base_path = project_root / base_path
    
    # Resolve net_file
    net_file = network.get("net_file", f"{network_name}.net.xml")
    if not Path(net_file).is_absolute():
        net_file = str(base_path / net_file)
    
    # Resolve route_files - can be a list or single string
    route_files_config = network.get("route_files", [f"{network_name}.rou.xml"])
    if isinstance(route_files_config, str):
        route_files_config = [route_files_config]
    
    route_files = []
    for rf in route_files_config:
        if not Path(rf).is_absolute():
            rf_path = base_path / rf
            if rf_path.exists():
                route_files.append(str(rf_path))
        else:
            if Path(rf).exists():
                route_files.append(rf)
    
    # If no route files found, use default
    if not route_files:
        default_route = base_path / f"{network_name}.rou.xml"
        route_files = [str(default_route)]
    
    # Join route files with comma for SUMO
    route_file = ",".join(route_files)
    
    # Resolve detector_file
    detector_file = network.get("detector_file", "detector.add.xml")
    if not Path(detector_file).is_absolute():
        detector_file = str(base_path / detector_file)
    
    # Resolve intersection_config
    intersection_config = network.get("intersection_config", "intersection_config.json")
    if not Path(intersection_config).is_absolute():
        intersection_config_path = base_path / intersection_config
        intersection_config = str(intersection_config_path) if intersection_config_path.exists() else None
    
    return {
        "network_name": network_name,
        "base_path": str(base_path),
        "net_file": net_file,
        "route_file": route_file,
        "detector_file": detector_file,
        "intersection_config": intersection_config,
    }


def is_local_gnn_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if Local GNN mode is enabled.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        True if local GNN is enabled
    """
    return config.get("mgmq", {}).get("local_gnn", {}).get("enabled", False)


def get_history_length(config: Dict[str, Any]) -> int:
    """
    Get history length (window size) from config.
    
    Args:
        config: Full configuration dictionary from model_config.yml
        
    Returns:
        History length value
    """
    return config.get("mgmq", {}).get("history_length", 4)


def load_training_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load training configuration from the experiment directory.
    
    Args:
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Training configuration dict or None if not found
    """
    try:
        checkpoint_dir = Path(checkpoint_path)
        
        # Add support for simple string paths that might handle PyArrow paths
        if not checkpoint_dir.exists():
            print(f"⚠ Warning: Checkpoint path does not exist: {checkpoint_path}")
            return None
        
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
                    # Also try to retrieve network name if possible
                    result = {"env_config": full_config["env_config"]}
                    
                    # Try to infer network name from path
                    path_str = str(checkpoint_path)
                    for net in ["grid4x4", "4x4loop", "zurich", "PhuQuoc"]:
                        if net in path_str:
                            result["network_name"] = net
                            break
                            
                    return result
        
        print("⚠ Warning: Training config not found")
        return None
        
    except Exception as e:
        print(f"⚠ Error loading training config: {e}")
        return None
