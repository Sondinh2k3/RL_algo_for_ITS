#!/usr/bin/env python3
"""
Test Model Forward Pass.

This module tests the MGMQEncoder and MGMQTorchModel architectures
to ensure correct output shapes and data flow.

Key validations:
1. Encoder output dimensions match expected
2. Action mask propagation works correctly
3. Policy and value heads produce valid outputs
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMGMQEncoder:
    """Test the MGMQEncoder component."""
    
    @pytest.fixture
    def encoder(self):
        """Create a test encoder."""
        from src.models.mgmq_model import MGMQEncoder
        
        return MGMQEncoder(
            obs_dim=48,  # 12 lanes × 4 features
            num_agents=1,
            gat_hidden_dim=64,
            gat_output_dim=32,
            gat_num_heads=4,
            graphsage_hidden_dim=64,
            gru_hidden_dim=32,
            dropout=0.0,  # Disable dropout for testing
        )
        
    def test_encoder_output_shape(self, encoder):
        """Encoder should produce correct output dimensions."""
        batch_size = 4
        obs = torch.randn(batch_size, 48)
        
        joint_emb, intersection_emb, network_emb = encoder(obs)
        
        # joint_emb = intersection_emb + network_emb
        expected_joint_dim = encoder.joint_emb_dim
        assert joint_emb.shape == (batch_size, expected_joint_dim), \
            f"Expected joint_emb shape ({batch_size}, {expected_joint_dim}), got {joint_emb.shape}"
            
    def test_encoder_deterministic(self, encoder):
        """Encoder should be deterministic in eval mode."""
        encoder.eval()
        obs = torch.randn(2, 48)
        
        with torch.no_grad():
            out1, _, _ = encoder(obs)
            out2, _, _ = encoder(obs)
            
        assert torch.allclose(out1, out2), "Encoder output should be deterministic in eval mode"
        
    def test_encoder_gradient_flow(self, encoder):
        """Gradients should flow through encoder."""
        encoder.train()
        obs = torch.randn(2, 48, requires_grad=True)
        
        joint_emb, _, _ = encoder(obs)
        loss = joint_emb.sum()
        loss.backward()
        
        assert obs.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(obs.grad).any(), "Gradients should not be NaN"


class TestLocalMGMQEncoder:
    """Test the LocalMGMQEncoder component (neighbor-aware)."""
    
    @pytest.fixture
    def local_encoder(self):
        """Create a test local encoder."""
        from src.models.mgmq_model import LocalMGMQEncoder
        
        return LocalMGMQEncoder(
            obs_dim=48,
            max_neighbors=4,
            gat_hidden_dim=64,
            gat_output_dim=32,
            gat_num_heads=4,
            graphsage_hidden_dim=64,
            gru_hidden_dim=32,
            dropout=0.0,
        )
        
    def test_local_encoder_dict_input(self, local_encoder):
        """LocalEncoder should accept Dict observation."""
        batch_size = 4
        max_neighbors = 4
        
        obs_dict = {
            "self_features": torch.randn(batch_size, 48),
            "neighbor_features": torch.randn(batch_size, max_neighbors, 48),
            "neighbor_mask": torch.ones(batch_size, max_neighbors),
        }
        
        joint_emb = local_encoder(obs_dict)
        
        expected_dim = local_encoder.joint_emb_dim
        assert joint_emb.shape == (batch_size, expected_dim), \
            f"Expected shape ({batch_size}, {expected_dim}), got {joint_emb.shape}"
            
    def test_local_encoder_masked_neighbors(self, local_encoder):
        """LocalEncoder should handle masked (missing) neighbors."""
        batch_size = 2
        max_neighbors = 4
        
        obs_dict = {
            "self_features": torch.randn(batch_size, 48),
            "neighbor_features": torch.randn(batch_size, max_neighbors, 48),
            "neighbor_mask": torch.tensor([
                [1, 1, 0, 0],  # 2 valid neighbors
                [1, 0, 0, 0],  # 1 valid neighbor
            ], dtype=torch.float32),
        }
        
        joint_emb = local_encoder(obs_dict)
        
        assert not torch.isnan(joint_emb).any(), "Output should not contain NaN with masked neighbors"


class TestMGMQTorchModel:
    """Test the RLlib-compatible MGMQTorchModel wrapper."""
    
    def test_model_output_for_masked_softmax(self):
        """Model should output 2*action_dim for MaskedSoftmax distribution."""
        from gymnasium.spaces import Box, Dict
        from src.models.mgmq_model import MGMQTorchModel
        
        # Create observation and action spaces
        obs_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        action_space = Box(low=0, high=1, shape=(8,), dtype=np.float32)  # 8 phases
        
        model_config = {
            "custom_model_config": {
                "num_agents": 1,
                "gat_hidden_dim": 32,
                "gat_output_dim": 16,
                "gat_num_heads": 2,
                "graphsage_hidden_dim": 32,
                "gru_hidden_dim": 16,
                "use_masked_softmax": True,
            }
        }
        
        model = MGMQTorchModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=16,  # 2 * 8 for MaskedSoftmax
            model_config=model_config,
            name="test_model",
        )
        
        # Test forward pass
        batch_size = 4
        obs = torch.randn(batch_size, 48)
        
        # Set dummy action mask
        model._last_action_mask = torch.ones(batch_size, 8)
        
        output, state = model({"obs": obs})
        
        # Output should be [batch, 16] for MaskedSoftmax (logits + log_std)
        assert output.shape == (batch_size, 16), \
            f"Expected output shape ({batch_size}, 16), got {output.shape}"
            
    def test_value_function(self):
        """Model should produce a scalar value estimate."""
        from gymnasium.spaces import Box
        from src.models.mgmq_model import MGMQTorchModel
        
        obs_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        action_space = Box(low=0, high=1, shape=(8,), dtype=np.float32)
        
        model_config = {
            "custom_model_config": {
                "num_agents": 1,
                "gat_hidden_dim": 32,
                "gat_output_dim": 16,
                "gat_num_heads": 2,
                "use_masked_softmax": True,
            }
        }
        
        model = MGMQTorchModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=16,
            model_config=model_config,
            name="test_model",
        )
        
        batch_size = 4
        obs = torch.randn(batch_size, 48)
        model._last_action_mask = torch.ones(batch_size, 8)
        
        # Forward pass stores value internally
        model({"obs": obs})
        
        # Get value
        value = model.value_function()
        
        assert value.shape == (batch_size,), \
            f"Expected value shape ({batch_size},), got {value.shape}"
        assert not torch.isnan(value).any(), "Value should not contain NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
