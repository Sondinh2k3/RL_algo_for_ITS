"""
Masked Softmax + Gaussian Action Distribution for RLlib PPO.

This module implements a custom action distribution that:
1. Applies Action Masking BEFORE Softmax (not post-hoc)
2. Uses Gaussian noise for exploration
3. Outputs valid simplex actions (sum=1, all non-negative)

Why Masked Softmax + Gaussian instead of Dirichlet?
---------------------------------------------------
With Dirichlet, Action Masking happens AFTER sampling:
    - Dirichlet samples all 8 phases
    - Post-hoc masking zeros invalid phases  
    - PPO still learns to output values for invalid phases (wasted gradient)
    - Entropy calculation includes invalid phases (incorrect)

With Masked Softmax + Gaussian, Action Masking happens BEFORE Softmax:
    - Model outputs raw logits
    - Add Gaussian noise for exploration
    - Apply mask: logits_masked = logits + (1 - mask) * (-1e9)
    - Softmax: invalid phases get EXACTLY 0.0
    - Gradient only flows through valid phases
    - Entropy correctly measures uncertainty over valid phases only

Flow:
-----
    1. Actor Network outputs: logits [batch, 8]
    2. Model provides: action_mask [batch, 8] (from FRAP PhaseStandardizer)
    3. Add noise (training): logits_noisy = logits + std * N(0,1)
    4. Apply mask: logits_masked = logits_noisy + (1 - mask) * (-1e9)
    5. Softmax: action = softmax(logits_masked)
    
Result: Actions for masked phases are EXACTLY 0.0, sum of valid phases = 1.0

Usage:
------
>>> from src.models.masked_softmax_distribution import register_masked_softmax_distribution
>>> register_masked_softmax_distribution()
>>> # In PPO config:
>>> config.training(model={"custom_action_dist": "masked_softmax"})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, TYPE_CHECKING

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict

if TYPE_CHECKING:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


# Default exploration noise std
DEFAULT_NOISE_STD = 1.0

# Large negative value for masking (makes softmax output ~0)
MASK_VALUE = -1e9

# Number of standard phases
NUM_STANDARD_PHASES = 8

# Softmax temperature: lower = sharper output (more differentiated actions)
# Default 1.0 = standard softmax, 0.3 = much sharper differentiation
# This fixes the uniform action problem by amplifying differences in logits
SOFTMAX_TEMPERATURE = 0.3


class TorchMaskedSoftmax(TorchDistributionWrapper):
    """
    Masked Softmax + Gaussian Noise action distribution for RLlib.
    
    This distribution is designed for action spaces where:
    - Actions must sum to 1 (simplex constraint)
    - Some actions may be invalid and must be masked to exactly 0
    - Exploration is needed during training
    
    The model must store action_mask in self._last_action_mask before
    the distribution is created. This is done in MGMQTorchModel.forward().
    
    Architecture:
    - Model outputs: [logits, log_std] where each has shape [batch, action_dim]
    - Distribution applies mask, adds noise, then softmax
    - Sampling: softmax(masked_noisy_logits)
    - Log prob: computed using Gumbel-Softmax approximation
    
    Attributes:
        logits: Raw logits from model [batch, action_dim]
        log_std: Log standard deviation for noise [batch, action_dim]
        action_mask: Binary mask [batch, action_dim] (1=valid, 0=invalid)
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
        Initialize Masked Softmax distribution from model outputs.
        
        Args:
            inputs: Model outputs [batch, 2 * action_dim] = [logits, log_std]
            model: The RLlib model (must have _last_action_mask attribute)
            action_space: The action space (Box)
        """
        super().__init__(inputs, model)
        
        # Split inputs into logits and log_std
        action_dim = inputs.shape[-1] // 2
        self.logits = inputs[..., :action_dim]
        self.log_std = inputs[..., action_dim:]
        
        # Clamp log_std for stability
        self.log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        self.std = torch.exp(self.log_std)
        
        # Get action mask from model
        # Model must set self._last_action_mask in forward() before distribution is created
        if hasattr(model, '_last_action_mask') and model._last_action_mask is not None:
            self.action_mask = model._last_action_mask.to(self.logits.device)
            # Ensure same batch dimension
            if self.action_mask.dim() == 1:
                self.action_mask = self.action_mask.unsqueeze(0).expand(self.logits.size(0), -1)
        else:
            # Fallback: all phases valid (no masking)
            self.action_mask = torch.ones_like(self.logits)
        
        # Store for sampling
        self.last_sample = None
        self._deterministic = False
        
        # Store model reference for training mode check
        self._model = model
        
    def _apply_mask_and_softmax(self, logits: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Apply mask and softmax to logits.
        
        Args:
            logits: Raw logits [batch, action_dim]
            add_noise: Whether to add Gaussian noise (False for deterministic)
            
        Returns:
            Softmax probabilities [batch, action_dim] with masked phases = 0
        """
        # Step 1: Add Gaussian noise for exploration (training only)
        # Use model.training to check if in training mode (model is nn.Module)
        is_training = self._model.training if hasattr(self._model, 'training') else True
        if add_noise and is_training:
            noise = torch.randn_like(logits) * self.std
            logits_noisy = logits + noise
        else:
            logits_noisy = logits
            
        # Step 2: Apply mask (CRITICAL - this is the key difference from Dirichlet)
        # Masked phases get very large negative value -> softmax outputs ~0
        logits_masked = logits_noisy + (1.0 - self.action_mask) * MASK_VALUE
        
        # Step 3: Softmax normalization with temperature scaling
        # Temperature < 1.0 makes output sharper (more differentiated)
        # Result: valid phases sum to 1, masked phases = 0
        probs = F.softmax(logits_masked / SOFTMAX_TEMPERATURE, dim=-1)
        
        return probs
    
    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """
        Return the mode (argmax softmax) of the distribution.
        
        For deterministic action, we don't add noise and just take softmax.
        """
        self._deterministic = True
        
        # No noise for deterministic sampling
        probs = self._apply_mask_and_softmax(self.logits, add_noise=False)
        
        # Store for log_prob calculation
        self.last_sample = probs
        
        return probs
    
    @override(ActionDistribution)
    def sample(self) -> TensorType:
        """
        Sample from the distribution using reparameterization trick.
        
        Uses Gumbel-Softmax for differentiable sampling with masking.
        """
        self._deterministic = False
        
        # Sample with noise
        probs = self._apply_mask_and_softmax(self.logits, add_noise=True)
        
        # Clamp to avoid numerical issues
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        
        # Re-normalize to ensure sum = 1
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Store for log_prob calculation
        self.last_sample = probs
        
        return probs
    
    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        """
        Compute log probability of given actions.
        
        For masked softmax, we use a Categorical-like log probability
        but only over valid (unmasked) phases.
        
        Args:
            actions: Actions [batch, action_dim], should be valid simplex
        """
        # Ensure valid simplex
        actions = torch.clamp(actions, min=1e-8, max=1.0 - 1e-8)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        
        # Compute masked softmax probabilities (no noise for log_prob)
        probs = self._apply_mask_and_softmax(self.logits, add_noise=False)
        probs = torch.clamp(probs, min=1e-8)
        
        # Log probability using cross-entropy formulation
        # log p(a) = sum_i [ a_i * log(p_i) ] for valid phases only
        # This is equivalent to treating the action as a "soft" categorical
        log_probs = actions * torch.log(probs)
        
        # Sum only over valid (unmasked) phases
        log_probs = log_probs * self.action_mask
        log_prob = log_probs.sum(dim=-1)
        
        return log_prob
    
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
        Compute entropy of the distribution over VALID phases only.
        
        This is the key improvement over Dirichlet:
        - Dirichlet entropy includes invalid phases (incorrect)
        - Masked Softmax entropy only considers valid phases (correct)
        
        H = -sum_i [ p_i * log(p_i) ] for valid phases only
        """
        # Get probabilities (no noise for entropy calculation)
        probs = self._apply_mask_and_softmax(self.logits, add_noise=False)
        probs = torch.clamp(probs, min=1e-8)
        
        # Entropy: -sum(p * log(p)) for valid phases
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs * self.action_mask, dim=-1)
        
        return entropy
    
    @override(ActionDistribution)
    def kl(self, other: "TorchMaskedSoftmax") -> TensorType:
        """
        Compute KL divergence KL(self || other) over valid phases.
        
        KL = sum_i [ p_i * (log(p_i) - log(q_i)) ] for valid phases
        """
        p = self._apply_mask_and_softmax(self.logits, add_noise=False)
        q = other._apply_mask_and_softmax(other.logits, add_noise=False)
        
        p = torch.clamp(p, min=1e-8)
        q = torch.clamp(q, min=1e-8)
        
        # KL divergence over valid phases
        kl = torch.sum(p * (torch.log(p) - torch.log(q)) * self.action_mask, dim=-1)
        
        return kl
    
    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space, 
        model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        """
        Return required output size from the model.
        
        For Masked Softmax: output_dim = 2 * action_dim (logits + log_std)
        """
        action_dim = int(np.prod(action_space.shape))
        return 2 * action_dim


def register_masked_softmax_distribution():
    """
    Register the Masked Softmax distribution with RLlib's ModelCatalog.
    
    Call before creating RLlib config:
    ```python
    from src.models.masked_softmax_distribution import register_masked_softmax_distribution
    register_masked_softmax_distribution()
    ```
    """
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_action_dist("masked_softmax", TorchMaskedSoftmax)
    print("[MaskedSoftmax] Registered 'masked_softmax' action distribution with RLlib")
