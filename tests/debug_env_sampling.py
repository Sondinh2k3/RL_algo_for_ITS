#!/usr/bin/env python
"""
Debug test script for SUMO environment (without RLlib).

Run this to verify the environment works correctly before debugging RLlib integration.
This script will help identify if the issue is in SUMO environment or in RLlib.

Usage:
    cd /home/sondinh2k3/Documents/Working_ITS/Project/MGMQ_v8
    python tests/debug_env_sampling.py
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set environment variable for SUMO if not set
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

from pathlib import Path

def test_sumo_environment():
    """Test SUMO environment without RLlib."""
    print("=" * 60)
    print("SUMO Environment Debug Test")
    print("=" * 60)
    
    # Import after path setup - use SumoEnvironment directly (no RLlib dependency)
    print("\n[1] Importing SumoEnvironment...")
    start = time.time()
    try:
        from src.environment.drl_algo.env import SumoEnvironment
        print(f"    ✓ Import successful ({time.time() - start:.2f}s)")
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Build paths
    print("\n[2] Setting up paths...")
    network_dir = Path(project_root) / "network" / "grid4x4"
    net_file = str(network_dir / "grid4x4.net.xml")
    route_file = f"{network_dir}/grid4x4.rou.xml,{network_dir}/grid4x4-demo.rou.xml"
    detector_file = str(network_dir / "detector.add.xml")
    preprocessing_config = str(network_dir / "intersection_config.json")
    
    print(f"    net_file: {net_file}")
    print(f"    route_file: {route_file}")
    print(f"    detector_file: {detector_file}")
    print(f"    preprocessing_config: {preprocessing_config}")
    
    # Check files exist
    if not Path(net_file).exists():
        print(f"    ✗ net_file not found!")
        return False
    if not Path(preprocessing_config).exists():
        print(f"    ✗ preprocessing_config not found!")
        return False
    print("    ✓ All files exist")
    
    # Create environment
    print("\n[3] Creating SumoEnvironment...")
    start = time.time()
    try:
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=500,  # Short simulation for testing
            cycle_time=90,
            yellow_time=3,
            min_green=15,
            max_green=90,
            single_agent=False,  # Multi-agent mode
            preprocessing_config=preprocessing_config,
            additional_sumo_cmd=f"-a {detector_file} --step-length 0.1",
            reward_fn=["halt-veh-by-detectors", "diff-departed-veh"],
            reward_weights=[0.5, 0.5],
            use_phase_standardizer=True,
            use_neighbor_obs=True,
            max_neighbors=4,
            window_size=4,
        )
        print(f"    ✓ Environment created ({time.time() - start:.2f}s)")
        print(f"    ts_ids: {env.ts_ids}")
        print(f"    num_agents: {len(env.ts_ids)}")
    except Exception as e:
        import traceback
        print(f"    ✗ Failed to create env: {e}")
        traceback.print_exc()
        return False
    
    # Reset environment  
    print("\n[4] Calling env.reset()...")
    start = time.time()
    try:
        obs, info = env.reset()
        print(f"    ✓ Reset successful ({time.time() - start:.2f}s)")
        print(f"    obs keys: {list(obs.keys())}")
        print(f"    info keys: {list(info.keys())[:5]}...")
        
        # Check observations structure
        sample_ts = list(obs.keys())[0]
        sample_obs = obs[sample_ts]
        if isinstance(sample_obs, dict):
            print(f"    obs['{sample_ts}'] is dict with keys: {list(sample_obs.keys())}")
            for k, v in sample_obs.items():
                print(f"      {k}: shape={getattr(v, 'shape', 'N/A')}, dtype={getattr(v, 'dtype', type(v).__name__)}")
        else:
            print(f"    obs['{sample_ts}']: shape={getattr(sample_obs, 'shape', 'N/A')}")
    except Exception as e:
        import traceback
        print(f"    ✗ Reset failed: {e}")
        traceback.print_exc()
        env.close()
        return False
    
    # Run step loop
    print("\n[5] Running step loop (5 steps)...")
    for i in range(5):
        start = time.time()
        try:
            # Sample random actions
            actions = {}
            for ts_id in env.ts_ids:
                action_space = env.action_spaces(ts_id)
                if action_space is None:
                    print(f"    Warning: No action space for {ts_id}")
                    continue
                actions[ts_id] = action_space.sample()
            
            # Step environment (multi-agent returns 4 values: obs, rewards, dones, info)
            result = env.step(actions)
            if len(result) == 5:
                obs, rewards, terminateds, truncateds, info = result
                done_all = truncateds.get("__all__", False)
            else:
                obs, rewards, dones, info = result
                done_all = dones.get("__all__", False)
                
            elapsed = time.time() - start
            
            total_reward = sum(rewards.values())
            sim_time = info.get("step", "N/A")
            
            print(f"    Step {i+1}: reward_sum={total_reward:.3f}, "
                  f"sim_time={sim_time}, elapsed={elapsed:.3f}s")
            
            # Check for truncation
            if done_all:
                print(f"    → Episode ended at step {i+1}")
                break
                
        except Exception as e:
            import traceback
            print(f"    ✗ Step {i+1} failed: {e}")
            traceback.print_exc()
            env.close()
            return False
    
    # Clean up
    print("\n[6] Closing environment...")
    env.close()
    print("    ✓ Environment closed")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Environment works correctly!")
    print("=" * 60)
    print("\nIf training still fails, the issue is in RLlib integration.")
    print("Next step: Check Ray worker logs at ~/ray_results/ or /tmp/ray/")
    
    return True


if __name__ == "__main__":
    success = test_sumo_environment()
    sys.exit(0 if success else 1)
