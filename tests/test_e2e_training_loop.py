#!/usr/bin/env python3
"""
Test End-to-End Training Loop.

This module tests a minimal training iteration to verify:
1. Model weights update after training step
2. No NaN in gradients
3. Loss values are reasonable
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMinimalTrainingStep:
    """Test a minimal training step without RLlib."""
    
    @pytest.fixture
    def model_and_optimizer(self):
        """Create a model and optimizer for testing."""
        from src.models.mgmq_model import MGMQEncoder
        
        encoder = MGMQEncoder(
            obs_dim=48,
            num_agents=1,
            gat_hidden_dim=32,
            gat_output_dim=16,
            gat_num_heads=2,
            graphsage_hidden_dim=32,
            gru_hidden_dim=16,
            dropout=0.0,
        )
        
        # Add simple policy head
        policy_head = nn.Linear(encoder.output_dim, 8)
        
        model = nn.Sequential(encoder, nn.Flatten(start_dim=1))
        
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(policy_head.parameters()), lr=1e-3)
        
        return encoder, policy_head, optimizer
        
    def test_weights_update_after_step(self, model_and_optimizer):
        """Model weights should change after optimizer step."""
        encoder, policy_head, optimizer = model_and_optimizer
        
        # Save initial weights
        initial_weights = {
            name: param.clone() for name, param in encoder.named_parameters()
        }
        
        # Forward pass
        obs = torch.randn(4, 48)
        joint_emb, _, _ = encoder(obs)
        logits = policy_head(joint_emb)
        
        # Dummy loss
        target = torch.zeros(4, 8)
        target[:, 0] = 1.0  # Target: phase 0
        loss = nn.functional.cross_entropy(logits, target.argmax(dim=-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check weights changed
        weights_changed = False
        for name, param in encoder.named_parameters():
            if not torch.allclose(param, initial_weights[name], atol=1e-8):
                weights_changed = True
                break
                
        assert weights_changed, "Model weights should update after optimizer step"
        
    def test_no_nan_in_gradients(self, model_and_optimizer):
        """Gradients should not contain NaN."""
        encoder, policy_head, optimizer = model_and_optimizer
        
        obs = torch.randn(4, 48)
        joint_emb, _, _ = encoder(obs)
        logits = policy_head(joint_emb)
        
        loss = logits.sum()
        
        optimizer.zero_grad()
        loss.backward()
        
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in parameter: {name}"
                    
    def test_loss_is_finite(self, model_and_optimizer):
        """Loss should be a finite value."""
        encoder, policy_head, optimizer = model_and_optimizer
        
        obs = torch.randn(4, 48)
        joint_emb, _, _ = encoder(obs)
        logits = policy_head(joint_emb)
        
        target = torch.randint(0, 8, (4,))
        loss = nn.functional.cross_entropy(logits, target)
        
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"


class TestMaskedSoftmaxTraining:
    """Test training with MaskedSoftmax distribution."""
    
    def test_masked_softmax_backward(self):
        """MaskedSoftmax should support backward pass."""
        from src.models.masked_softmax_distribution import TorchMaskedSoftmax
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._last_action_mask = torch.ones(4, 8)
                self.linear = nn.Linear(48, 16)
                
        model = MockModel()
        model.train()
        
        obs = torch.randn(4, 48, requires_grad=True)
        inputs = model.linear(obs)
        
        dist = TorchMaskedSoftmax(inputs, model)
        sample = dist.sample()
        
        # Simulate policy loss
        loss = -dist.logp(sample).mean()
        
        loss.backward()
        
        assert obs.grad is not None, "Gradients should flow to observation"
        
    def test_entropy_gradient(self):
        """Entropy should have valid gradients for entropy bonus."""
        from src.models.masked_softmax_distribution import TorchMaskedSoftmax
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._last_action_mask = torch.ones(4, 8)
                self.linear = nn.Linear(48, 16)
                
        model = MockModel()
        model.train()
        
        obs = torch.randn(4, 48)
        inputs = model.linear(obs)
        inputs.retain_grad()
        
        dist = TorchMaskedSoftmax(inputs, model)
        entropy = dist.entropy()
        
        # Entropy bonus loss
        loss = -entropy.mean()
        loss.backward()
        
        assert inputs.grad is not None, "Entropy should have gradients"
        assert not torch.isnan(inputs.grad).any(), "Entropy gradients should not be NaN"


class TestValueFunctionTraining:
    """Test value function training."""
    
    def test_value_loss_and_gradient(self):
        """Value function should produce valid loss and gradients."""
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
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Forward pass
        obs = torch.randn(4, 48)
        model._last_action_mask = torch.ones(4, 8)
        
        _, _ = model({"obs": obs})
        value = model.value_function()
        
        # Value loss (MSE with target)
        target_value = torch.randn(4)
        value_loss = nn.functional.mse_loss(value, target_value)
        
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()
        
        assert torch.isfinite(value_loss), f"Value loss should be finite, got {value_loss.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
