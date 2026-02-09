
import sys
import os
import json
import numpy as np
from pathlib import Path

import importlib.util

# Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Load PhaseStandardizer directly from file to avoid triggering package __init__ which imports torch
file_path = str(Path(__file__).parent.parent.absolute() / "src/preprocessing/frap.py")
spec = importlib.util.spec_from_file_location("frap", file_path)
frap_module = importlib.util.module_from_spec(spec)
sys.modules["frap"] = frap_module
spec.loader.exec_module(frap_module)
PhaseStandardizer = frap_module.PhaseStandardizer

def verify_mask():
    """Verify that action mask only enables mapped phases."""
    
    # Load A0 config
    config_path = "network/grid4x4/intersection_config.json"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        full_config = json.load(f)
        
    a0_config = full_config["intersections"]["A0"]["phase_config"]
    
    # Create standardizer
    frap = PhaseStandardizer("A0")
    frap.load_config(a0_config)
    
    # Get mask
    mask = frap.get_phase_mask()
    print(f"Action Mask for A0: {mask}")
    
    # Check expected
    # A0 maps to 0, 1, 2, 3 (based on previous inspection)
    # So mask should be [1, 1, 1, 1, 0, 0, 0, 0]
    
    expected_mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    
    if np.array_equal(mask, expected_mask):
        print("PASS: Mask matches expected [1, 1, 1, 1, 0, 0, 0, 0]")
        return True
    else:
        print(f"FAIL: Expected {expected_mask}, got {mask}")
        return False

if __name__ == "__main__":
    success = verify_mask()
    sys.exit(0 if success else 1)
