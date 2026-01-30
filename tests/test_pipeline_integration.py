import unittest
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.environment.rllib_utils import SumoMultiAgentEnv
from src.models.mgmq_model import MGMQTorchModel
from src.config import load_model_config, get_env_config, get_mgmq_config
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.view_requirement import ViewRequirement

class TestPipelineIntegration(unittest.TestCase):
    """
    Test flow: Env -> Preprocessing -> Model -> Action -> Env
    Kiểm tra thông luồng dữ liệu từ đầu đến cuối.
    """

    @classmethod
    def setUpClass(cls):
        print(">>> SETUP: Loading Configs...")
        cls.config_path = os.path.join(project_root, "src/config/model_config.yml")
        cls.yaml_config = load_model_config(cls.config_path)
        
        # 1. Config Environment
        detector_path = os.path.join(project_root, "network/grid4x4/detector.add.xml")
        preprocessing_path = os.path.join(project_root, "network/grid4x4/intersection_config.json")
        
        cls.env_config = {
            "net_file": os.path.join(project_root, "network/grid4x4/grid4x4.net.xml"),
            "route_file": os.path.join(project_root, "network/grid4x4/grid4x4.rou.xml"),
            "additional_sumo_cmd": f"-a {detector_path}", # Add detectors
            "preprocessing_config": preprocessing_path,
            "use_gui": False,
            "num_seconds": 100, # Short run for testing
            "max_green": 60,
            "min_green": 5,
            "yellow_time": 3,
            "time_to_teleport": -1,
            "use_phase_standardizer": True,
            "virtual_display": None,
        }
        
        # 2. Config Model
        cls.mgmq_config = get_mgmq_config(cls.yaml_config)
        
    def test_end_to_end_flow(self):
        print("\n=== TEST: End-to-End Flow (Env -> Model -> Env) ===")
        
        # 1. Initialize Environment
        print("1. Initializing Environment...")
        env = SumoMultiAgentEnv(**self.env_config)
        obs, info = env.reset()
        
        ts_ids = list(obs.keys())
        first_ts = ts_ids[0]
        print(f"   ✓ Env initialized with {len(ts_ids)} agents.")
        print(f"   ✓ Sample Agent ID: {first_ts}")
        print(f"   ✓ Observation Structure Keys: {obs[first_ts].keys() if isinstance(obs[first_ts], dict) else 'Array'}")

        # 2. Initialize Model
        print("2. Initializing MGMQ Model...")
        # Mocking the obs_space and action_space based on env
        # For MultiAgentEnv, we grab specific agent's space
        if hasattr(env, "simulator") and hasattr(env.simulator, "traffic_signals"):
             action_space = env.simulator.traffic_signals[first_ts].action_space
             obs_space = env.simulator.traffic_signals[first_ts].observation_space
        else:
             # Fallback 
             obs_space = env.observation_space
             action_space = env.action_space
        
        print(f"   ✓ Observation Space: {obs_space}")
        print(f"   ✓ Action Space: {action_space} (Type: {type(action_space)})")

        num_outputs = 0
        if hasattr(action_space, 'n'):
             num_outputs = action_space.n
        else:
             print("   ! WARNING: Action space has no '.n' (not Discrete?). Using shape or arbitrary default.")
             if hasattr(action_space, 'shape'):
                 # For Box, usually we output mean/std, so 2 * shape[0]
                 # But here we assume we want discrete logic for testing
                 num_outputs = action_space.shape[0] 
        
        # Model config dict expected by RLlib
        model_config_rllib = {
            "custom_model_config": self.mgmq_config
        }
        
        model = MGMQTorchModel(
            obs_space, 
            action_space, 
            num_outputs,  # num_outputs
            model_config_rllib, 
            name="mgmq_test_model"
        )
        print("   ✓ Model initialized.")

        # 3. Data Preprocessing (Manual batching to simulate RLlib)
        print("3. Preprocessing (Simulating RLlib Batching)...")
        
        # RLlib passes inputs as a dictionary of tensors with a batch dimension
        # We take the single observation from 'reset' and add batch_size=1
        
        input_dict = {}
        
        # Handle dict observation (common in GNN/Complex models)
        if isinstance(obs[first_ts], dict):
            input_dict["obs"] = {}
            for key, value in obs[first_ts].items():
                # Convert to tensor and add batch dim
                tensor_val = torch.tensor(np.array([value]), dtype=torch.float32)
                input_dict["obs"][key] = tensor_val
        else:
            # Handle array observation
            obs_tensor = torch.tensor(np.array([obs[first_ts]]), dtype=torch.float32)
            input_dict["obs"] = obs_tensor
            input_dict["obs_flat"] = obs_tensor # RLlib provides flattened obs here
            
        print("   ✓ Input dict created successfully.")

        # 4. Model Forward Pass
        print("4. Running Model Forward Pass...")
        
        # Mock state (empty for non-recurrent/init, or zeros if recurrent)
        state = []
        if hasattr(model, "get_initial_state"):
            state = model.get_initial_state()
            # Convert to torch tensor with batch dim
            state = [torch.tensor(np.array([s]), dtype=torch.float32) for s in state]

        seq_lens = torch.tensor([1], dtype=torch.int32) # Sequence length 1 for single step
        
        try:
            model_out, new_state = model.forward(input_dict, state, seq_lens)
            print(f"   ✓ Model output shape (Logits): {model_out.shape}")
            self.assertEqual(model_out.shape[1], num_outputs, "Output dimension must match action space size")
            # Also check value function call
            val = model.value_function()
            print(f"   ✓ Value function output shape: {val.shape}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Model Forward failed with error: {e}")

        # 5. Action Selection
        print("5. Selecting Action from Logits...")
        # Simple argmax for testing
        action_idx = torch.argmax(model_out, dim=1).item()
        print(f"   ✓ Selected Action Index: {action_idx}")
        
        # 6. Apply Action to Environment
        print("6. Applying Action to Environment...")
        actions = {first_ts: action_idx}
        
        # Fill random actions for other agents to complete the step
        for ts_id in ts_ids:
            if ts_id != first_ts:
                actions[ts_id] = 0
                
        next_obs, rewards, dones, truncateds, infos = env.step(actions)
        
        print(f"   ✓ Environment stepped successfully.")
        print(f"   ✓ Reward received: {rewards.get(first_ts, 'N/A')}")
        
        # 7. Check if cycle repeats
        print("7. Verifying next observation...")
        self.assertIn(first_ts, next_obs, "Agent should have observation in next step")
        
        env.close()
        print("\n>>> SUCCESS: Pipeline Flow Verified!")

if __name__ == '__main__':
    unittest.main()
