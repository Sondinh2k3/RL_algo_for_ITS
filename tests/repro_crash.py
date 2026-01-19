import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.drl_algo.env import SumoEnvironment
from src.sim.Sumo_sim import SumoSimulator

def test_repro():
    project_root = Path(__file__).parent.parent
    net_file = project_root / "network" / "grid4x4" / "grid4x4.net.xml"
    route_file = project_root / "network" / "grid4x4" / "grid4x4.rou.xml"
    
    # Load config if exists
    config_file = project_root / "network" / "grid4x4" / "intersection_config.json"
    
    print(f"Testing with network: {net_file}")
    
    env = SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        use_gui=False,
        num_seconds=1000,
        delta_time=60,
        yellow_time=3,
        min_green=5,
        max_green=60,
        preprocessing_config=str(config_file) if config_file.exists() else None,
        additional_sumo_cmd=f"-a {str(project_root / 'network' / 'grid4x4' / 'detector.add.xml')} --step-length 0.1",
        use_phase_standardizer=True,
    )
    
    print("Environment created. Initializing...")
    obs, info = env.reset()
    print(f"Reset successful. Observation keys: {obs.keys()}")
    
    print("Running 1 step...")
    # env.action_space for multi-agent usually returns a dict or has a different API
    import numpy as np
    actions = {ts_id: env.action_space.sample() for ts_id in obs.keys()}
    obs, rewards, dones, info = env.step(actions)
    print(f"Step successful. Reward mean: {sum(rewards.values())/len(rewards)}")
    
    env.close()
    print("Simulation finished successfully.")

if __name__ == "__main__":
    test_repro()
