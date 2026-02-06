#!/usr/bin/env python3
"""
Diagnostic script to trace action flow from model logits to final green times.

This script helps identify WHERE uniformity is introduced:
1. Model logits → Are they differentiated or uniform?
2. Softmax output → Does temperature=0.3 help?
3. FRAP standardize → Does mapping lose differentiation?
4. Green time formula → min_green dominance?

Usage:
    python scripts/diagnose_action_flow.py
"""

import numpy as np
import torch
import torch.nn.functional as F

# Simulation parameters
MIN_GREEN = 5  # seconds
CYCLE_TIME = 90  # seconds
YELLOW_TIME = 3  # seconds per phase
NUM_ACTUAL_PHASES = 4
SOFTMAX_TEMPERATURE = 0.3

# Calculate derived values
total_yellow = YELLOW_TIME * NUM_ACTUAL_PHASES  # 12s
total_green = CYCLE_TIME - total_yellow  # 78s
min_green_total = MIN_GREEN * NUM_ACTUAL_PHASES  # 20s
remaining_time = total_green - min_green_total  # 58s

# FRAP mapping for grid4x4 (from intersection_config.json)
# actual_to_standard: {"0": 4, "1": 0, "2": 6, "3": 1}
ACTUAL_TO_STANDARD = {0: 4, 1: 0, 2: 6, 3: 1}

def analyze_scenario(name: str, model_logits: np.ndarray, action_mask: np.ndarray = None):
    """Trace action flow from model logits to green times."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    
    if action_mask is None:
        # Default mask for 4-phase intersection: phases 0,1,4,6 valid
        action_mask = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.float32)
    
    print(f"\n1. MODEL OUTPUT (raw logits):")
    print(f"   logits = {np.round(model_logits, 4)}")
    print(f"   logit range: [{model_logits.min():.4f}, {model_logits.max():.4f}]")
    print(f"   logit std: {model_logits.std():.4f}")
    
    # Step 2: Apply mask before softmax
    logits_tensor = torch.tensor(model_logits, dtype=torch.float32)
    mask_tensor = torch.tensor(action_mask, dtype=torch.float32)
    
    MASK_VALUE = -1e9
    logits_masked = logits_tensor + (1.0 - mask_tensor) * MASK_VALUE
    
    print(f"\n2. AFTER MASKING (invalid phases → -inf):")
    print(f"   masked_logits = {logits_masked.numpy()}")
    
    # Step 3: Apply softmax with temperature
    probs = F.softmax(logits_masked / SOFTMAX_TEMPERATURE, dim=-1).numpy()
    
    print(f"\n3. AFTER SOFTMAX (temperature={SOFTMAX_TEMPERATURE}):")
    print(f"   probs = {np.round(probs, 4)}")
    print(f"   valid phases sum: {probs[action_mask > 0].sum():.4f}")
    print(f"   valid phase variance: {probs[action_mask > 0].var():.6f}")
    
    # Step 4: FRAP standardize_action (map to actual phases)
    standard_ratios = probs.copy()
    actual_ratios = np.zeros(NUM_ACTUAL_PHASES)
    
    for actual_idx in range(NUM_ACTUAL_PHASES):
        std_idx = ACTUAL_TO_STANDARD.get(actual_idx, 0)
        actual_ratios[actual_idx] = standard_ratios[std_idx]
    
    # Normalize
    actual_ratios = actual_ratios / actual_ratios.sum() if actual_ratios.sum() > 0 else np.ones(NUM_ACTUAL_PHASES) / NUM_ACTUAL_PHASES
    
    print(f"\n4. AFTER FRAP MAPPING (8 → 4 phases):")
    print(f"   actual_ratios = {np.round(actual_ratios, 4)}")
    print(f"   actual ratio variance: {actual_ratios.var():.6f}")
    
    # Step 5: Compute green times
    green_times = MIN_GREEN + (actual_ratios * remaining_time)
    int_green_times = np.floor(green_times).astype(int)
    
    # Distribute remainder
    current_sum = np.sum(int_green_times)
    remainder = int(total_green - current_sum)
    if remainder > 0:
        fractional_parts = green_times - int_green_times
        indices = np.argsort(fractional_parts)[::-1]
        for i in range(remainder):
            idx = indices[i % len(indices)]
            int_green_times[idx] += 1
    
    print(f"\n5. FINAL GREEN TIMES:")
    for i, gt in enumerate(int_green_times):
        pct = gt / total_green * 100
        print(f"   Phase {i}: {gt}s ({pct:.1f}%)")
    
    print(f"\n   Green time range: [{int_green_times.min()}s, {int_green_times.max()}s]")
    print(f"   Green time std: {int_green_times.std():.2f}s")
    
    # Assessment
    print(f"\n6. ASSESSMENT:")
    gt_range = int_green_times.max() - int_green_times.min()
    if gt_range < 5:
        print(f"   ⚠️ GREEN TIMES ARE NEARLY UNIFORM (range={gt_range}s)")
    elif gt_range < 10:
        print(f"   ⚠️ GREEN TIMES SHOW LOW DIFFERENTIATION (range={gt_range}s)")
    else:
        print(f"   ✓ GREEN TIMES ARE DIFFERENTIATED (range={gt_range}s)")
    
    return int_green_times


def main():
    print("="*70)
    print("ACTION FLOW DIAGNOSTIC")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  min_green = {MIN_GREEN}s")
    print(f"  cycle_time = {CYCLE_TIME}s")
    print(f"  num_phases = {NUM_ACTUAL_PHASES}")
    print(f"  total_green = {total_green}s")
    print(f"  remaining_time = {remaining_time}s (after min_green)")
    print(f"  SOFTMAX_TEMPERATURE = {SOFTMAX_TEMPERATURE}")
    
    # Scenario 1: Uniform logits (untrained model)
    logits_uniform = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    analyze_scenario("Uniform Logits (untrained)", logits_uniform)
    
    # Scenario 2: Slightly differentiated logits
    logits_slight = np.array([0.5, 0.3, 0.0, 0.0, 0.8, 0.0, 0.4, 0.0])
    analyze_scenario("Slightly Differentiated", logits_slight)
    
    # Scenario 3: Well differentiated logits
    logits_diff = np.array([2.0, 0.5, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0])
    analyze_scenario("Well Differentiated", logits_diff)
    
    # Scenario 4: Strongly differentiated logits
    logits_strong = np.array([5.0, 1.0, 0.0, 0.0, 8.0, 0.0, 2.0, 0.0])
    analyze_scenario("Strongly Differentiated", logits_strong)
    
    # Scenario 5: What logits are needed for 40/20/25/15 split?
    print(f"\n{'='*70}")
    print("REVERSE ENGINEERING: What logits create 40%/20%/25%/15% split?")
    print(f"{'='*70}")
    
    target_ratios = np.array([0.40, 0.20, 0.25, 0.15])  # Actual phase ratios
    target_green = MIN_GREEN + (target_ratios * remaining_time)
    print(f"\nTarget green times: {np.round(target_green, 1)}")
    
    # These standard ratios would produce target actual ratios
    # actual[0]=std[4], actual[1]=std[0], actual[2]=std[6], actual[3]=std[1]
    std_ratios_needed = np.zeros(8)
    std_ratios_needed[4] = target_ratios[0]  # For actual[0]
    std_ratios_needed[0] = target_ratios[1]  # For actual[1]
    std_ratios_needed[6] = target_ratios[2]  # For actual[2]
    std_ratios_needed[1] = target_ratios[3]  # For actual[3]
    
    print(f"Standard ratios needed: {np.round(std_ratios_needed, 4)}")
    
    # Inverse softmax to get logits (with temp=0.3)
    # softmax(logits/T) = ratios → logits = T * log(ratios)
    # But we have mask, so only compute for valid phases
    logits_needed = np.zeros(8)
    for i, r in enumerate(std_ratios_needed):
        if r > 0:
            logits_needed[i] = SOFTMAX_TEMPERATURE * np.log(r / std_ratios_needed.max())
    
    print(f"Logits needed (approx): {np.round(logits_needed, 4)}")
    print(f"Logit range needed: {logits_needed[std_ratios_needed > 0].max() - logits_needed[std_ratios_needed > 0].min():.4f}")
    
    # Verify
    analyze_scenario("Verification of Target Ratios", logits_needed)
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
Key findings:
1. With SOFTMAX_TEMPERATURE=0.3, even small logit differences create 
   significant ratio differences.
   
2. For uniform output: logit std ≈ 0 (untrained model)

3. For differentiated output: need logit range of ~0.5+ (with temp=0.3)

4. If your model produces uniform outputs, check:
   - Model training status (entropy should decrease)
   - Observation variance (are inputs differentiated?)
   - Value function quality (vf_explained_var > 0.5)
""")


if __name__ == "__main__":
    main()
