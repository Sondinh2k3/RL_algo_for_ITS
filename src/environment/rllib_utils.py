"""Shared utilities for RLlib integration with SUMO environment.

This module contains common classes and functions used by both training and evaluation scripts.
It eliminates code duplication between train_mgmq_ppo.py and eval_mgmq_ppo.py.

Author: Son Dinh
Date: 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from .drl_algo.env import SumoEnvironment


class SumoMultiAgentEnv(SumoEnvironment, MultiAgentEnv):
    """Wrapper for SumoEnvironment to be recognized as a MultiAgentEnv by RLlib.
    
    This ensures RLlib treats the dictionary observations correctly for 
    multi-agent traffic signal control scenarios.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure we are in multi-agent mode
        self.single_agent = False

    def step(self, action_dict):
        obs, rewards, dones, info = super().step(action_dict)
        
        # Convert dones (dict) to terminateds and truncateds for Gymnasium API
        truncateds = {"__all__": dones.pop("__all__", False)}
        terminateds = {"__all__": False}
        
        for agent_id in self.ts_ids:
            terminateds[agent_id] = False
            truncateds[agent_id] = truncateds["__all__"]
            
        # Restructure info for MultiAgentEnv
        # RLlib expects info to be a dict mapping agent_ids to info dicts
        new_info = {}
        common_info = {}
        
        # Initialize info dicts for all agents present in obs
        active_agents = list(obs.keys())
        for agent_id in active_agents:
            new_info[agent_id] = {}

        for key, value in info.items():
            # Check if key belongs to a specific agent
            found_agent = False
            for agent_id in self.ts_ids:
                if key.startswith(f"{agent_id}_"):
                    if agent_id in new_info:
                        metric_name = key[len(agent_id)+1:]
                        new_info[agent_id][metric_name] = value
                    found_agent = True
                    break
            
            if not found_agent:
                common_info[key] = value
        
        # Add common info to every agent's info dict
        for agent_id in new_info:
            new_info[agent_id].update(common_info)
            
        # Check for NaNs in rewards or obs
        if any(np.isnan(r) for r in rewards.values()):
            print(f"WARNING: NaN rewards detected: {rewards}")
            rewards = {k: (0.0 if np.isnan(v) else v) for k, v in rewards.items()}
            
        return obs, rewards, terminateds, truncateds, new_info


def get_network_ts_ids(network_name: str, project_root: Optional[Path] = None) -> List[str]:
    """Get traffic signal IDs for a given network.
    
    Try to read from intersection_config if available, else use defaults.
    
    Args:
        network_name: Name of the network (grid4x4, 4x4loop, etc.)
        project_root: Project root directory. If None, uses current file's parent.parent.
        
    Returns:
        List of traffic signal IDs
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    # Try to load from intersection_config.json
    config_path = project_root / "network" / network_name / "intersection_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            if "ts_ids" in config:
                return config["ts_ids"]
            elif "intersections" in config:
                return list(config["intersections"].keys())

    # Default traffic signal IDs for known networks
    network_ts_ids = {
        "grid4x4": [
            "A0", "A1", "A2", "A3",
            "B0", "B1", "B2", "B3",
            "C0", "C1", "C2", "C3",
            "D0", "D1", "D2", "D3"
        ],
        "4x4loop": [
            "0", "1", "2", "3",
            "4", "5", "6", "7",
            "8", "9", "10", "11",
            "12", "13", "14", "15"
        ],
        "network_test": ["gneJ1", "gneJ2", "gneJ3", "gneJ4"],
    }
    
    return network_ts_ids.get(network_name, ["ts_0"])


def register_sumo_env(config_dict: Dict[str, Any], env_name: str = "sumo_mgmq_v0"):
    """Register SUMO environment with RLlib.
    
    Args:
        config_dict: Environment configuration dictionary
        env_name: Name to register the environment under
    """
    register_env(
        env_name,
        lambda env_config: SumoMultiAgentEnv(**{**config_dict, **env_config})
    )


def override_config(cli_value: Any, config_value: Any) -> Any:
    """Return CLI value if not None, else config value.
    
    Helper for merging CLI arguments with config file values.
    
    Args:
        cli_value: Value from command line argument
        config_value: Value from configuration file
        
    Returns:
        cli_value if it's not None, otherwise config_value
    """
    return cli_value if cli_value is not None else config_value
