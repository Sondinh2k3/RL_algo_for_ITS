"""
Dirichlet Action Distribution for RLlib PPO.

This module implements a custom Dirichlet distribution for continuous actions
that must satisfy the simplex constraint (sum to 1, all positive).

Why Dirichlet?
--------------
The standard Gaussian distribution used by PPO has issues for traffic signal
green time ratio allocation:

1. **Distribution Mismatch**: Gaussian can produce negative values and values
   that don't sum to 1. External clipping/normalization breaks PPO math.

2. **Scale Ambiguity (Gradient Saturation)**: If we normalize outputs externally,
   different raw outputs (e.g., [10,10] vs [1000,1000]) produce identical actions,
   but large values cause vanishing gradients due to Softmax saturation.

3. **Exploration Issues**: Gaussian std controls spread around mean, but for
   simplex-constrained actions, we need proper simplex exploration.

Dirichlet Solution:
------------------
- Model outputs: Raw logits → transformed to concentration parameters α
- α = softplus(logits) + 1 (ensures α > 1 for stable mode)
- Sample: action ~ Dirichlet(α) → automatically sum=1, all in (0,1)
- Higher α → more confident (peaked distribution)
- Lower α → more exploration (spread across simplex)

Usage
-----
>>> from src.models.dirichlet_distribution import register_dirichlet_distribution
>>> register_dirichlet_distribution()
>>> # In PPO config:
>>> config.training(model={"custom_action_dist": "dirichlet"})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict


# Concentration parameter bounds for numerical stability
# For Dirichlet, entropy can be negative (this is mathematically correct).
# PPO still works because relative entropy changes matter, not absolute values.
# Lower α → higher entropy (less peaked) → more exploration
# Higher α → lower entropy (more peaked) → deterministic actions
#
# IMPORTANT: With 4 phases, if all α_i = MAX, mode = [0.25, 0.25, 0.25, 0.25]
# To allow extreme ratios like [0.7, 0.1, 0.1, 0.1], we need MAX >> MIN
# With α = [20, 1, 1, 1], mode ≈ [0.86, 0.045, 0.045, 0.045]
CONCENTRATION_MIN = 0.5    # Avoid extreme corners (α < 0.5 causes very spread dist)
CONCENTRATION_MAX = 20.0   # INCREASED from 5.0 to allow more extreme ratios

# Entropy offset: Add this to make reported entropy non-negative for monitoring
# With CONCENTRATION_MAX=20 and 4 phases, minimum entropy ≈ -5.13
# Use offset of 6.0 to ensure positive entropy for all valid alpha values
ENTROPY_OFFSET = 6.0


class TorchDirichlet(TorchDistributionWrapper):
    """
    Dirichlet distribution for RLlib that outputs actions summing to 1.
    
    This distribution is ideal for:
    - Traffic signal green time ratios
    - Resource allocation problems  
    - Any action that must be a probability distribution (simplex constraint)
    
    The model outputs raw logits, which are transformed to concentration
    parameters α via softplus + offset. Samples are drawn from Dirichlet(α).
    
    Note on entropy:
    - Dirichlet entropy can be negative (mathematically correct)
    - We add ENTROPY_OFFSET to reported entropy for better monitoring
    - PPO entropy bonus still works correctly (relative changes matter)
    """
    
    @override(ActionDistribution)
    def __init__(
        self,
        inputs: TensorType,
        model: "TorchModelV2",
        *,
        action_space=None,
    ):
        """
        Initialize Dirichlet distribution from model outputs.
        
        Args:
            inputs: Raw logits from model [batch, action_dim]
            model: The RLlib model
            action_space: The action space (Box)
        """
        super().__init__(inputs, model)
        
        # Transform logits to concentration parameters
        # 
        # IMPORTANT: The transformation must allow model to express:
        #   - High α (>10) for dominant phases → peaked distribution
        #   - Low α (<1) for minor phases → spread distribution  
        #
        # Previous issue: softplus(logits) + 0.5 only gives range ~[0.6, 2.6]
        # for typical logits in [-2, 2], which is too narrow!
        #
        # NEW approach: Use exponential scaling for wider dynamic range
        # α = CONCENTRATION_MIN + (CONCENTRATION_MAX - CONCENTRATION_MIN) * sigmoid(logits)
        # This maps any logit to the full [MIN, MAX] range smoothly
        #
        # With sigmoid:
        #   logits = -4 → sigmoid ≈ 0.018 → α ≈ 0.85
        #   logits =  0 → sigmoid = 0.5   → α ≈ 10.25
        #   logits = +4 → sigmoid ≈ 0.982 → α ≈ 19.65
        #
        # This gives model full control over concentration range!
        self.concentration = CONCENTRATION_MIN + (CONCENTRATION_MAX - CONCENTRATION_MIN) * torch.sigmoid(inputs)
        
        # Create PyTorch Dirichlet distribution
        self.dist = torch.distributions.Dirichlet(self.concentration)
        
        # Required by RLlib for sampled_action_logp()
        self.last_sample = None
    
    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """
        Return the mode (most likely value) of the Dirichlet distribution.
        
        For Dirichlet with all α_i > 1, mode_i = (α_i - 1) / (sum(α) - K)
        For α_i <= 1, use mean as fallback.
        """
        # Check if all concentrations > 1
        all_gt_one = (self.concentration > 1.0).all(dim=-1, keepdim=True)
        
        # Mode for α > 1
        alpha_sum = self.concentration.sum(dim=-1, keepdim=True)
        K = self.concentration.shape[-1]
        mode = (self.concentration - 1.0) / (alpha_sum - K + 1e-8)
        
        # Mean as fallback
        mean = self.concentration / alpha_sum
        
        # Use mode where valid, mean otherwise
        result = torch.where(all_gt_one, mode, mean)
        
        # Ensure sum = 1 (numerical stability)
        result = result / result.sum(dim=-1, keepdim=True)
        
        # Store for sampled_action_logp()
        self.last_sample = result
        
        return result
    
    @override(ActionDistribution)
    def sample(self) -> TensorType:
        """
        Sample from the Dirichlet distribution.
        
        Uses reparameterized sampling for gradient flow.
        """
        # rsample() uses reparameterization trick
        sample = self.dist.rsample()
        
        # Clamp to avoid exact 0 or 1 (log issues)
        sample = torch.clamp(sample, min=1e-6, max=1.0 - 1e-6)
        
        # Re-normalize to ensure sum = 1
        sample = sample / sample.sum(dim=-1, keepdim=True)
        
        # Store for sampled_action_logp()
        self.last_sample = sample
        
        return sample
    
    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        """
        Compute log probability of given actions.
        
        Args:
            actions: Actions [batch, action_dim], must be valid simplex
        """
        # Ensure valid for Dirichlet
        actions = torch.clamp(actions, min=1e-6, max=1.0 - 1e-6)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        
        return self.dist.log_prob(actions)
    
    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        """
        Return log probability of the last sampled action.
        Required by RLlib's exploration strategies.
        """
        assert self.last_sample is not None, "Must call sample() first"
        return self.logp(self.last_sample)
    
    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        """
        Compute entropy of the Dirichlet distribution.
        
        Note: Dirichlet entropy can be negative (mathematically correct).
        We add ENTROPY_OFFSET to make it non-negative for better monitoring.
        This doesn't affect learning since only relative entropy changes matter.
        
        Higher entropy = more exploration (less peaked distribution).
        """
        # Add offset to make entropy non-negative for better interpretability
        return self.dist.entropy() + ENTROPY_OFFSET
    
    @override(ActionDistribution)
    def kl(self, other: "TorchDirichlet") -> TensorType:
        """Compute KL divergence KL(self || other)."""
        return torch.distributions.kl_divergence(self.dist, other.dist)
    
    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space, 
        model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        """
        Return required output size from the model.
        For Dirichlet: output_dim = action_dim (concentration params).
        """
        return int(np.prod(action_space.shape))


def register_dirichlet_distribution():
    """
    Register the Dirichlet distribution with RLlib's ModelCatalog.
    
    Call before creating RLlib config:
    ```python
    from src.models.dirichlet_distribution import register_dirichlet_distribution
    register_dirichlet_distribution()
    ```
    """
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_action_dist("dirichlet", TorchDirichlet)
    print("[Dirichlet] Registered 'dirichlet' action distribution with RLlib")
