#!/usr/bin/env python3
"""
Test Masked Softmax Distribution.

This module tests the MaskedSoftmax action distribution to ensure:
1. Masked phases get probability ≈ 0
2. Valid phases sum to 1
3. Entropy is calculated over valid phases only
4. Gradients flow correctly
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMaskedSoftmaxLogic:
    """Test the core masked softmax logic."""
    
    def test_masked_phases_get_zero_probability(self):
        """Masked phases should have probability ≈ 0 after softmax."""
        # Simulate raw logits
        logits = torch.randn(4, 8)  # batch=4, 8 phases
        
        # Action mask: only phases 0, 1, 4, 5 are valid
        action_mask = torch.tensor([
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
        ], dtype=torch.float32)
        
        # Apply mask
        MASK_VALUE = -1e9
        logits_masked = logits + (1.0 - action_mask) * MASK_VALUE
        
        # Softmax
        probs = F.softmax(logits_masked, dim=-1)
        
        # Check masked phases have near-zero probability
        for batch_idx in range(4):
            for phase_idx in [2, 3, 6, 7]:  # Masked phases
                assert probs[batch_idx, phase_idx] < 1e-6, \
                    f"Masked phase {phase_idx} should have prob ≈ 0, got {probs[batch_idx, phase_idx]}"
                    
    def test_valid_phases_sum_to_one(self):
        """Probabilities of valid phases should sum to 1."""
        logits = torch.randn(4, 8)
        action_mask = torch.tensor([
            [1, 1, 0, 0, 1, 1, 0, 0],
        ], dtype=torch.float32).expand(4, -1)
        
        MASK_VALUE = -1e9
        logits_masked = logits + (1.0 - action_mask) * MASK_VALUE
        probs = F.softmax(logits_masked, dim=-1)
        
        # Sum should be ≈ 1
        prob_sum = probs.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(4), atol=1e-5), \
            f"Probs should sum to 1, got {prob_sum}"


class TestEntropyCalculation:
    """Test entropy calculation over valid phases only."""
    
    def test_entropy_with_4_valid_phases(self):
        """Entropy with 4 uniform valid phases should be ln(4) ≈ 1.386."""
        # Create uniform logits
        logits = torch.zeros(1, 8)  # All equal → uniform after softmax
        
        # 4 valid phases
        action_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float32)
        
        MASK_VALUE = -1e9
        logits_masked = logits + (1.0 - action_mask) * MASK_VALUE
        probs = F.softmax(logits_masked, dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        
        # Calculate entropy: -sum(p * log(p)) for valid phases
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs * action_mask, dim=-1)
        
        expected_entropy = np.log(4)  # ≈ 1.386
        assert torch.allclose(entropy, torch.tensor([expected_entropy], dtype=torch.float32), atol=0.01), \
            f"Expected entropy ≈ {expected_entropy}, got {entropy.item()}"
            
    def test_entropy_with_8_valid_phases(self):
        """Entropy with 8 uniform valid phases should be ln(8) ≈ 2.08."""
        logits = torch.zeros(1, 8)
        action_mask = torch.ones(1, 8)  # All phases valid
        
        MASK_VALUE = -1e9
        logits_masked = logits + (1.0 - action_mask) * MASK_VALUE
        probs = F.softmax(logits_masked, dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs * action_mask, dim=-1)
        
        expected_entropy = np.log(8)  # ≈ 2.08
        assert torch.allclose(entropy, torch.tensor([expected_entropy], dtype=torch.float32), atol=0.01), \
            f"Expected entropy ≈ {expected_entropy}, got {entropy.item()}"
            
    def test_entropy_decreases_with_peaky_distribution(self):
        """Entropy should decrease when one phase dominates."""
        # Peaked logits: phase 0 has much higher logit
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        action_mask = torch.ones(1, 8)
        
        MASK_VALUE = -1e9
        logits_masked = logits + (1.0 - action_mask) * MASK_VALUE
        probs = F.softmax(logits_masked, dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs * action_mask, dim=-1)
        
        # Entropy should be much less than ln(8)
        assert entropy.item() < np.log(8) * 0.5, \
            f"Peaked distribution should have low entropy, got {entropy.item()}"


class TestTorchMaskedSoftmaxDistribution:
    """Test the actual TorchMaskedSoftmax distribution class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with action mask."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._last_action_mask = None
                
        return MockModel()
        
    def test_distribution_sample_shape(self, mock_model):
        """Sample should have correct shape."""
        from src.models.masked_softmax_distribution import TorchMaskedSoftmax
        
        batch_size = 4
        action_dim = 8
        
        # inputs = [logits, log_std]
        inputs = torch.randn(batch_size, 2 * action_dim)
        mock_model._last_action_mask = torch.ones(batch_size, action_dim)
        
        dist = TorchMaskedSoftmax(inputs, mock_model)
        sample = dist.sample()
        
        assert sample.shape == (batch_size, action_dim), \
            f"Expected sample shape ({batch_size}, {action_dim}), got {sample.shape}"
            
    def test_distribution_sample_is_simplex(self, mock_model):
        """Sampled action should be a valid simplex (sum=1, non-negative)."""
        from src.models.masked_softmax_distribution import TorchMaskedSoftmax
        
        inputs = torch.randn(4, 16)
        mock_model._last_action_mask = torch.ones(4, 8)
        
        dist = TorchMaskedSoftmax(inputs, mock_model)
        sample = dist.sample()
        
        # Check sum ≈ 1
        assert torch.allclose(sample.sum(dim=-1), torch.ones(4), atol=1e-4), \
            f"Sample should sum to 1, got {sample.sum(dim=-1)}"
            
        # Check non-negative
        assert (sample >= 0).all(), "Sample should be non-negative"
        
    def test_masked_phases_in_sample(self, mock_model):
        """Masked phases should have ≈ 0 in sample."""
        from src.models.masked_softmax_distribution import TorchMaskedSoftmax
        
        inputs = torch.randn(2, 16)
        mock_model._last_action_mask = torch.tensor([
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
        ], dtype=torch.float32)
        
        # Use deterministic sample for testing
        dist = TorchMaskedSoftmax(inputs, mock_model)
        sample = dist.deterministic_sample()
        
        for batch_idx in range(2):
            for phase_idx in [2, 3, 6, 7]:
                assert sample[batch_idx, phase_idx] < 1e-6, \
                    f"Masked phase {phase_idx} should be ≈ 0 in sample"


class TestGradientFlow:
    """Test that gradients flow correctly through the distribution."""
    
    def test_gradients_to_logits(self):
        """Gradients should flow from loss to logits."""
        from src.models.masked_softmax_distribution import TorchMaskedSoftmax
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._last_action_mask = torch.ones(2, 8)
                
        model = MockModel()
        model.train()
        
        inputs = torch.randn(2, 16, requires_grad=True)
        
        dist = TorchMaskedSoftmax(inputs, model)
        sample = dist.sample()
        
        # Simulate loss
        loss = sample.sum()
        loss.backward()
        
        assert inputs.grad is not None, "Gradients should flow to inputs"
        assert not torch.isnan(inputs.grad).any(), "Gradients should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
