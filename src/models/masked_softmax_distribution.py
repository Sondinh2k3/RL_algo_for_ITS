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
        Compute log-probability using exact marginalized Gaussian density.
        
        Since actions are generated by:
            z = mu + std * epsilon,  epsilon ~ N(0, I)
            a = softmax(z_masked / T)
        
        the softmax is shift-invariant (adding constant c to all z_i gives the
        same a). We marginalize over c analytically to get the exact density:
        
            log p(a) = Gaussian_term + Normalization + Jacobian
        
        where (all sums over valid phases, K = num valid phases):
            f_i = T * log(a_i) - mu_i
            S = sum(1/sigma_i^2),  M = sum(f_i / sigma_i^2)
            
            Gaussian_term  = -0.5 * sum(f_i^2/sigma_i^2) + M^2 / (2*S)
            Normalization   = -sum(log sigma_i) - (K-1)/2 * log(2π) - 0.5 * log(S)
            Jacobian        = -sum(log a_i) + (K-1) * log(T)
        
        This makes logp properly depend on std, enabling PPO to optimize
        the exploration noise level through the policy gradient.
        
        Args:
            actions: Actions [batch, action_dim], should be valid simplex
        """
        # Ensure valid simplex with numerical stability
        actions = torch.clamp(actions, min=1e-8, max=1.0 - 1e-8)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        
        valid_mask = self.action_mask
        num_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # f_i = T * log(a_i) - mu_i for valid phases
        f = (SOFTMAX_TEMPERATURE * torch.log(actions) - self.logits) * valid_mask
        
        # Precision (inverse variance) for valid phases only
        var = (self.std ** 2).clamp(min=1e-8)
        inv_var = valid_mask / var  # 1/sigma_i^2 for valid, 0 for masked
        
        # Marginalized Gaussian terms
        # S = sum(1/sigma_i^2), M = sum(f_i/sigma_i^2)
        S = inv_var.sum(dim=-1).clamp(min=1e-8)        # [batch]
        M = (f * inv_var).sum(dim=-1)                   # [batch]
        quad = (f * f * inv_var).sum(dim=-1)             # sum(f_i^2 / sigma_i^2)
        
        gaussian_logp = -0.5 * quad + 0.5 * M * M / S
        
        # Normalization constants
        log_sigma_sum = (torch.log(self.std.clamp(min=1e-8)) * valid_mask).sum(dim=-1)
        K_minus_1 = (num_valid.squeeze(-1) - 1).clamp(min=0)
        normalization = -log_sigma_sum - 0.5 * K_minus_1 * np.log(2 * np.pi) \
                        - 0.5 * torch.log(S)
        
        # Jacobian correction for softmax change-of-variables:
        # log|det(dz/da)| = -sum(log a_i) + (K-1) * log(T) for valid phases
        log_jac = -(torch.log(actions) * valid_mask).sum(dim=-1) \
                  + K_minus_1 * np.log(SOFTMAX_TEMPERATURE)
        
        log_prob = gaussian_logp + normalization + log_jac
        
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
        Compute entropy using the PRE-SOFTMAX Gaussian noise model.
        
        Since the actual exploration mechanism is Gaussian noise on logits:
            z_i = mu_i + std_i * eps_i,  eps_i ~ N(0,1)
        
        the entropy of the continuous action distribution must account for std.
        Using the Gaussian entropy for the noise distribution ensures that:
        - PPO entropy bonus directly affects std (exploration level)
        - Increasing entropy_coeff encourages larger std (more exploration)
        - Decreasing entropy_coeff allows std to shrink (exploitation)
        
        H = sum_i [ log(sigma_i) + 0.5 * log(2*pi*e) ] for valid phases
          = sum_i [ log(sigma_i) ] + K/2 * log(2*pi*e)
        
        This is the (K-1)-dimensional entropy (softmax removes 1 DOF),
        scaled to be comparable with the old categorical entropy range.
        
        Note: We subtract 1 DOF because softmax constrains sum=1,
        so only K-1 of K Gaussian noise dimensions are independent.
        """
        valid_mask = self.action_mask
        num_valid = valid_mask.sum(dim=-1).clamp(min=1)  # K
        
        # Gaussian entropy: sum(log(sigma_i)) for valid phases
        log_sigma_sum = (torch.log(self.std.clamp(min=1e-8)) * valid_mask).sum(dim=-1)
        
        # (K-1) independent dimensions (softmax removes 1 DOF)
        K_minus_1 = (num_valid - 1).clamp(min=0)
        
        # H = sum(log sigma_i) + (K-1)/2 * log(2*pi*e)
        entropy = log_sigma_sum + 0.5 * K_minus_1 * np.log(2.0 * np.pi * np.e)
        
        return entropy
    
    @override(ActionDistribution)
    def kl(self, other: "TorchMaskedSoftmax") -> TensorType:
        """
        Compute KL divergence KL(self || other) using Gaussian model.
        
        Since the action distribution is defined by Gaussian noise on logits:
            self:  z_i = mu_i + sigma_i * eps
            other: z_i = mu'_i + sigma'_i * eps
        
        The KL divergence between two K-dim Gaussians (reduced by 1 DOF for softmax) is:
            KL = sum_i [ log(sigma'_i/sigma_i) + (sigma_i^2 + (mu_i-mu'_i)^2)/(2*sigma'^2_i) - 0.5 ]
        
        This is summed over valid phases only.
        
        Using Gaussian KL ensures:
        - KL accounts for changes in both mean (logits) and std (exploration)
        - PPO's KL penalty correctly limits policy updates in both dimensions
        - Consistent with the logp() formulation
        """
        valid_mask = self.action_mask
        
        mu_self = self.logits * valid_mask
        mu_other = other.logits * valid_mask
        
        std_self = self.std.clamp(min=1e-8) * valid_mask + (1 - valid_mask) * 1.0  # avoid log(0)
        std_other = other.std.clamp(min=1e-8) * valid_mask + (1 - valid_mask) * 1.0
        
        var_other = std_other ** 2
        
        # KL per dimension: log(sigma'/sigma) + (sigma^2 + (mu-mu')^2)/(2*sigma'^2) - 0.5
        kl_per_dim = (
            torch.log(std_other / std_self)
            + (std_self ** 2 + (mu_self - mu_other) ** 2) / (2.0 * var_other)
            - 0.5
        ) * valid_mask
        
        kl = kl_per_dim.sum(dim=-1)
        
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
