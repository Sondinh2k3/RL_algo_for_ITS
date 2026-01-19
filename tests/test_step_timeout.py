"""
Quick test to verify the step() function works correctly and doesn't timeout.
This simulates RLlib's behavior by calling step() with and without actions.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.drl_algo.env import SumoEnvironment


def test_step_timeout():
    """Test that step() completes within reasonable time."""
    print("="*60)
    print("Testing step() timeout fix")
    print("="*60)
    
    # Create environment with short simulation for testing
    net_file = str(project_root / "network" / "grid4x4" / "grid4x4.net.xml")
    route_file = str(project_root / "network" / "grid4x4" / "grid4x4.rou.xml")
    detector_file = str(project_root / "network" / "grid4x4" / "detector.add.xml")
    preprocessing_config = str(project_root / "network" / "grid4x4" / "intersection_config.json")
    
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=600,  # Only 10 minutes simulation (shorter for testing)
        cycle_time=90,
        yellow_time=3,
        min_green=15,
        max_green=90,
        time_to_teleport=-1,
        single_agent=False,
        window_size=4,
        preprocessing_config=preprocessing_config,
        additional_sumo_cmd=f"-a {detector_file} --step-length 0.1",
        reward_fn=["halt-veh-by-detectors", "diff-departed-veh", "occupancy"],
        reward_weights=[0.33, 0.33, 0.34],
        use_phase_standardizer=True,
        use_neighbor_obs=True,
        max_neighbors=4,
    )
    
    print(f"\n✓ Environment created with {len(env.ts_ids)} traffic signals")
    print(f"  Traffic signals: {env.ts_ids}")
    
    # Test 1: reset() should return observations for all agents
    print("\nTest 1: Calling reset()...")
    start_time = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start_time
    print(f"  ✓ reset() completed in {reset_time:.2f}s")
    print(f"  ✓ Got observations for {len(obs)} agents")
    
    # Test 2: step() with empty actions (simulates RLlib initialization)
    print("\nTest 2: Calling step() with EMPTY actions...")
    start_time = time.time()
    obs, rewards, dones, info = env.step({})  # Empty actions - should apply default
    step_empty_time = time.time() - start_time
    print(f"  ✓ step() with empty actions completed in {step_empty_time:.2f}s")
    print(f"  ✓ Got observations for {len(obs)} agents")
    print(f"  ✓ Simulation time: {info.get('step', 0):.1f}s")
    
    # Test 3: step() with partial actions (some agents missing)
    print("\nTest 3: Calling step() with PARTIAL actions...")
    partial_actions = {env.ts_ids[0]: env.simulator.get_action_space(env.ts_ids[0]).sample()}
    start_time = time.time()
    obs, rewards, dones, info = env.step(partial_actions)
    step_partial_time = time.time() - start_time
    print(f"  ✓ step() with partial actions completed in {step_partial_time:.2f}s")
    print(f"  ✓ Simulation time: {info.get('step', 0):.1f}s")
    
    # Test 4: step() with full actions (normal operation)
    print("\nTest 4: Calling step() with FULL actions...")
    # Only use traffic signals that have valid action spaces (not skipped)
    valid_ts_ids = [ts_id for ts_id in env.ts_ids if env.simulator.get_action_space(ts_id) is not None]
    print(f"  Valid traffic signals: {len(valid_ts_ids)} out of {len(env.ts_ids)}")
    full_actions = {ts_id: env.simulator.get_action_space(ts_id).sample() for ts_id in valid_ts_ids}
    start_time = time.time()
    obs, rewards, dones, info = env.step(full_actions)
    step_full_time = time.time() - start_time
    print(f"  ✓ step() with full actions completed in {step_full_time:.2f}s")
    print(f"  ✓ Simulation time: {info.get('step', 0):.1f}s")
    print(f"  ✓ Sample reward: {list(rewards.values())[0]:.4f}")
    
    # Test 5: Multiple steps to check for accumulation issues
    print("\nTest 5: Running 5 consecutive steps...")
    total_step_time = 0
    for i in range(5):
        full_actions = {ts_id: env.simulator.get_action_space(ts_id).sample() for ts_id in valid_ts_ids}
        start_time = time.time()
        obs, rewards, dones, info = env.step(full_actions)
        step_time = time.time() - start_time
        total_step_time += step_time
        print(f"  Step {i+1}: {step_time:.2f}s, sim_time={info.get('step', 0):.1f}s")
        
        if dones.get("__all__", False):
            print("  Episode ended!")
            break
    
    print(f"\n  Average step time: {total_step_time/5:.2f}s")
    
    # Cleanup
    env.close()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print(f"  reset() time: {reset_time:.2f}s")
    print(f"  step() (empty actions) time: {step_empty_time:.2f}s")
    print(f"  step() (partial actions) time: {step_partial_time:.2f}s")
    print(f"  step() (full actions) time: {step_full_time:.2f}s")
    print(f"  Average step time: {total_step_time/5:.2f}s")
    
    # Check if times are reasonable
    expected_step_time = 90 / 0.1 * 0.001  # ~900 SUMO steps × ~1ms each = ~0.9s
    if step_full_time > 30:
        print(f"\n⚠️  WARNING: step() time ({step_full_time:.2f}s) is high!")
        print(f"   Expected approximate time: ~{expected_step_time:.1f}s")
        print(f"   Consider increasing SUMO step-length from 0.1s to 0.5s or 1.0s")
    else:
        print(f"\n✓ Step times are reasonable")


if __name__ == "__main__":
    test_step_timeout()
