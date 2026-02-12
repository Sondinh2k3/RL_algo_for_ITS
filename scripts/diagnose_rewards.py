#!/usr/bin/env python3
"""Diagnose why reward functions return near-zero values."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_loader import load_model_config, get_network_config, get_reward_config
from src.sim.Sumo_sim import SumoSimulator

def main():
    config = load_model_config("src/config/model_config.yml")
    net_cfg = get_network_config(config)
    reward_cfg = get_reward_config(config)
    env_cfg = config.get("environment", {})

    sim = SumoSimulator(
        net_file=net_cfg["net_file"],
        route_file=net_cfg["route_file"],
        use_gui=False,
        num_seconds=env_cfg.get("num_seconds", 6000),
        delta_time=env_cfg.get("cycle_time", 90),
        yellow_time=env_cfg.get("yellow_time", 3),
        min_green=env_cfg.get("min_green", 5),
        max_green=env_cfg.get("max_green", 90),
        reward_fn=reward_cfg["reward_fn"],
        reward_weights=reward_cfg.get("reward_weights"),
        time_to_teleport=env_cfg.get("time_to_teleport", -1),
        use_phase_standardizer=env_cfg.get("use_phase_standardizer", True),
        preprocessing_config=net_cfg.get("intersection_config"),
        additional_sumo_cmd=f"-a {net_cfg['detector_file']}" if net_cfg.get("detector_file") else None,
        fixed_ts=True,  # Use SUMO default timing
    )

    obs = sim.reset()
    ts_ids = list(sim.traffic_signals.keys())
    print(f"Traffic signals: {ts_ids}")
    print(f"Reward functions: {reward_cfg['reward_fn']}")
    print(f"Reward weights: {reward_cfg.get('reward_weights')}")

    # Pick a representative intersection
    test_ts_id = ts_ids[0]
    ts = sim.traffic_signals[test_ts_id]
    
    print(f"\n=== Intersection: {test_ts_id} ===")
    print(f"  Detectors E2: {ts.detectors_e2}")
    print(f"  Num detectors: {len(ts.detectors_e2)}")
    print(f"  max_veh: {ts.max_veh:.2f}")
    print(f"  delta_time: {ts.delta_time}")
    print(f"  max_waiting_change = max_veh * delta_time = {ts.max_veh * ts.delta_time:.2f}")
    print(f"  num_green_phases: {ts.num_green_phases}")
    print(f"  lanes: {ts.lanes}")
    
    # Check detector lengths
    print(f"\n  Detector lengths:")
    for det_id in ts.detectors_e2:
        length = ts.detectors_e2_length.get(det_id, 0)
        max_v = length / (ts.MIN_GAP + ts.avg_veh_length)
        print(f"    {det_id}: length={length:.1f}m, max_veh={max_v:.1f}")

    # Run several cycles and inspect reward components
    print(f"\n{'='*80}")
    print(f"RUNNING SIMULATION - 5 CYCLES")
    print(f"{'='*80}")
    
    for step in range(5):
        # Apply default actions (equal distribution)
        import numpy as np
        actions = {}
        for ts_id_i in ts_ids:
            ts_i = sim.traffic_signals[ts_id_i]
            if ts_i.time_to_act:
                actions[ts_id_i] = np.ones(8) / 8.0
        
        obs, rewards, dones, infos = sim.step(actions)
        
        if test_ts_id not in rewards:
            print(f"\n--- Step {step}: {test_ts_id} NOT in rewards (not acting yet) ---")
            continue
        
        ts = sim.traffic_signals[test_ts_id]
        sim_time = sim.get_sim_time()
        
        print(f"\n{'='*60}")
        print(f"STEP {step} | SimTime: {sim_time:.0f}s | TS: {test_ts_id}")
        print(f"{'='*60}")
        print(f"  Overall reward: {rewards[test_ts_id]:.6f}")
        
        # ---- diff-waiting-time breakdown ----
        wt_history = ts.reward_metrics_history.get("waiting_time", [])
        aggregated_wt = ts.get_aggregated_waiting_time()
        last_wt = ts.last_ts_waiting_time
        
        print(f"\n  [diff-waiting-time]")
        print(f"    waiting_time history samples: {len(wt_history)} -> {wt_history}")
        print(f"    aggregated_waiting_time: {aggregated_wt:.4f}")
        print(f"    last_ts_waiting_time (post-update): {last_wt:.4f}")
        # Recompute the diff that was actually used (last was updated already by compute_reward)
        # The actual last was the aggregated from PREVIOUS step
        max_wt_change = ts.max_veh * ts.sampling_interval_s
        if max_wt_change > 0:
            print(f"    max_waiting_change = max_veh * sampling_interval = {ts.max_veh:.1f} * {ts.sampling_interval_s} = {max_wt_change:.1f}")
        
        # ---- Check what _get_waiting_time_from_detectors returns right now ----
        instant_wt = ts._get_waiting_time_from_detectors()
        print(f"    _get_waiting_time_from_detectors() NOW = {instant_wt:.4f}")
        
        # Detailed per-detector jam info
        print(f"    Per-detector jam lengths:")
        for det_id in ts.detectors_e2:
            try:
                jam = ts.data_provider.get_detector_jam_length(det_id)
                veh_count = ts.data_provider.get_detector_vehicle_count(det_id)
                print(f"      {det_id}: jam_length={jam:.2f}m, veh_count={veh_count}")
            except Exception as e:
                print(f"      {det_id}: ERROR - {e}")
        
        # ---- Fallback: lane-based waiting time ----
        lane_wt = ts.get_accumulated_waiting_time_per_lane()
        print(f"    Lane-based waiting times: {[f'{w:.1f}' for w in lane_wt]}")
        print(f"    Lane-based total: {sum(lane_wt):.2f}")
        
        # ---- diff-departed-veh breakdown ----
        print(f"\n  [diff-departed-veh]")
        print(f"    initial_vehicles_this_cycle: {ts.initial_vehicles_this_cycle}")
        print(f"    departed_vehicles_this_cycle: {ts.departed_vehicles_this_cycle}")
        print(f"    _vehicles_at_cycle_start count: {len(ts._vehicles_at_cycle_start)}")
        print(f"    _vehicles_seen_this_cycle count: {len(ts._vehicles_seen_this_cycle)}")
        
        current_vehs = ts._get_current_vehicle_ids()
        print(f"    current vehicles in detectors: {len(current_vehs)}")
        
        # ---- halt-veh breakdown ----
        halt = ts.get_total_halting_veh_by_detectors()
        agg_halt = ts.get_aggregated_halting_vehicles()
        print(f"\n  [halt-veh-by-detectors]")
        print(f"    instant halting veh: {halt}")
        print(f"    aggregated halting veh: {agg_halt:.2f}")
        if ts.max_veh > 0:
            print(f"    ratio = {agg_halt/ts.max_veh:.4f}, reward = {-3.0 * min(1.0, agg_halt/ts.max_veh):.4f}")
        
        # ---- occupancy breakdown ----
        agg_occ = ts.get_aggregated_occupancy()
        print(f"\n  [occupancy]")
        print(f"    aggregated occupancy: {agg_occ:.4f}")
        print(f"    reward = {-agg_occ * 3.0:.4f}")
        
        if dones.get("__all__", False):
            print("\n*** SIMULATION ENDED ***")
            break
    
    sim.close()


if __name__ == "__main__":
    main()
