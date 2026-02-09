
import numpy as np

def verify_green_time_distribution():
    # Configuration from simulation.yml (USER UPDATED values)
    delta_time = 90
    yellow_time = 3
    min_green = 5
    num_phases = 4
    
    # TrafficSignal logic
    total_yellow_time = yellow_time * num_phases
    total_green_time = delta_time - total_yellow_time
    
    print(f"Metrics:")
    print(f"Cycle Time (delta_time): {delta_time}s")
    print(f"Total Yellow: {total_yellow_time}s")
    print(f"Total Green Available: {total_green_time}s")
    
    min_green_total = min_green * num_phases
    remaining_time = total_green_time - min_green_total
    
    print(f"Min Green Total (Required): {min_green_total}s")
    print(f"Flexible Time (Remaining): {remaining_time}s")
    
    if remaining_time <= 0:
        print("FAIL: Flexible time is <= 0. Agent has NO control.")
        return False
        
    print("PASS: Flexible time > 0. Agent can control phases.")
    
    # Test with two different actions
    
    # Action 1: Uniform
    action_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    green_uniform = min_green + (action_uniform * remaining_time)
    print(f"\nAction Uniform {action_uniform} -> Green Times: {green_uniform}")
    
    # Action 2: Skewed (Phase 0 preferred)
    action_skewed = np.array([0.7, 0.1, 0.1, 0.1])
    green_skewed = min_green + (action_skewed * remaining_time)
    print(f"Action Skewed  {action_skewed} -> Green Times: {green_skewed}")
    
    # Check variance
    if np.std(green_skewed) > 1.0:
        print("PASS: Significant variance observed in skewed action.")
        return True
    else:
        print("FAIL: Variance is too low despite valid remaining time.")
        return False

if __name__ == "__main__":
    verify_green_time_distribution()
