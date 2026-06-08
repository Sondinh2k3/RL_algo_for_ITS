"""
Masked Multi-Categorical Action Distribution for Discrete Cycle Adjustment.

Implements a custom action distribution for MultiDiscrete action spaces
with action masking. Designed for the Discrete Cycle Adjustment approach
where each traffic phase gets an independent discrete action.

With 7 actions per phase: {-15s, -10s, -5s, 0s, +5s, +10s, +15s}
Keep action is the middle index (3).

Action Masking:
- Invalid phases (determined by FRAP PhaseStandardizer) are forced to select
  the "keep" action by setting all other logits to -inf.
- This ensures the model never wastes gradient on invalid phases.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, TYPE_CHECKING

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict

if TYPE_CHECKING:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


# Number of standard phases
NUM_STANDARD_PHASES = 8

# Large negative value for masking
MASK_VALUE = -1e9


class TorchMaskedMultiCategorical(TorchDistributionWrapper):
    """Masked Multi-Categorical distribution for MultiDiscrete action spaces.

    For each of the 8 standard phases, maintains an independent categorical
    distribution over N discrete actions. Invalid phases are masked
    to always select the "keep" (middle) action.

    The model must store action_mask in self._last_action_mask before
    the distribution is created.

    Model output: [batch, NUM_STANDARD_PHASES * num_actions_per_phase]
    """

    @override(ActionDistribution)
    def __init__(
        self,
        inputs: TensorType,
        model: "TorchModelV2",
        *,
        action_space=None,
    ):
        super().__init__(inputs, model)

        self.num_phases = NUM_STANDARD_PHASES

        # Infer num_actions from input size
        total_logits = inputs.shape[-1]
        self.num_actions = total_logits // self.num_phases
        self.keep_action_idx = self.num_actions // 2  # middle action = keep

        # Split inputs into per-phase logits: [batch, 8, N]
        batch_size = inputs.shape[0]
        self.all_logits = inputs.view(batch_size, self.num_phases, self.num_actions)

        # Get action mask from model
        if hasattr(model, '_last_action_mask') and model._last_action_mask is not None:
            self.action_mask = model._last_action_mask.to(inputs.device)
            if self.action_mask.dim() == 1:
                self.action_mask = self.action_mask.unsqueeze(0).expand(batch_size, -1)
        else:
            self.action_mask = torch.ones(batch_size, self.num_phases, device=inputs.device)

        # Apply mask: for invalid phases, force logits to select "keep"
        self.masked_logits = self._apply_mask(self.all_logits)

        # Compute per-phase log-probabilities
        self.log_probs = F.log_softmax(self.masked_logits, dim=-1)  # [B, 8, N]

        self.last_sample = None

    def _apply_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply action mask to logits.

        For invalid phases (mask=0), set logits to force "keep" action:
        all -inf except keep_action_idx = 0, so softmax gives 100% keep.
        """
        mask = self.action_mask.unsqueeze(-1)  # [B, 8, 1]

        # For invalid phases: replace logits with [-inf, ..., 0, ..., -inf]
        keep_only = torch.full_like(logits, MASK_VALUE)
        keep_only[..., self.keep_action_idx] = 0.0

        # Where mask=1 use original logits, where mask=0 use keep_only
        masked = logits * mask + keep_only * (1.0 - mask)
        return masked

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """Return the mode (argmax) of each phase's categorical distribution."""
        actions = torch.argmax(self.masked_logits, dim=-1)  # [B, 8]
        self.last_sample = actions
        return actions

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        """Sample from each phase's categorical distribution independently."""
        # Gumbel-max trick for differentiable sampling
        uniform = torch.rand_like(self.masked_logits).clamp(1e-8, 1.0 - 1e-8)
        gumbels = -torch.log(-torch.log(uniform))
        actions = torch.argmax(self.masked_logits + gumbels, dim=-1)  # [B, 8]
        self.last_sample = actions
        return actions

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        """Compute log-probability of given actions.

        Args:
            actions: [batch, 8] integer actions
        """
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        # Gather log-prob for each phase's chosen action
        log_p = self.log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, 8]

        # Sum log-probs across all phases (independent categoricals)
        total_log_p = log_p.sum(dim=-1)  # [B]
        return total_log_p

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        """Return log probability of the last sampled action."""
        assert self.last_sample is not None, "Must call sample() first"
        return self.logp(self.last_sample)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        """Compute entropy as sum of per-phase categorical entropies."""
        probs = torch.exp(self.log_probs)  # [B, 8, N]
        per_phase_entropy = -(probs * self.log_probs).sum(dim=-1)  # [B, 8]

        # Only count entropy for valid phases
        masked_entropy = per_phase_entropy * self.action_mask  # [B, 8]
        total_entropy = masked_entropy.sum(dim=-1)  # [B]
        return total_entropy

    @override(ActionDistribution)
    def kl(self, other: "TorchMaskedMultiCategorical") -> TensorType:
        """Compute KL divergence KL(self || other)."""
        probs_self = torch.exp(self.log_probs)  # [B, 8, N]
        log_probs_other = other.log_probs  # [B, 8, N]

        # KL per phase = sum(p * (log(p) - log(q)))
        per_phase_kl = (probs_self * (self.log_probs - log_probs_other)).sum(dim=-1)  # [B, 8]

        # Only count KL for valid phases
        masked_kl = per_phase_kl * self.action_mask  # [B, 8]
        total_kl = masked_kl.sum(dim=-1)  # [B]
        return total_kl

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space,
        model_config: ModelConfigDict,
    ) -> Union[int, np.ndarray]:
        """Return required model output size."""
        # Get num_discrete_actions from model config
        custom_cfg = model_config.get("custom_model_config", {})
        num_actions = custom_cfg.get("num_discrete_actions", 7)
        return NUM_STANDARD_PHASES * num_actions


def register_masked_multi_categorical():
    """Register the Masked Multi-Categorical distribution with RLlib."""
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_action_dist(
        "masked_multi_categorical", TorchMaskedMultiCategorical
    )
    print("[MaskedMultiCategorical] Registered 'masked_multi_categorical' action distribution")
