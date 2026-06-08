"""Masked Dirichlet action distribution for cycle-level continuous control.

Used with ``action_mode=cycle_level_continuous``. The policy outputs
concentration parameters ``alpha_i > 0`` for each of the 8 standard phases,
and samples a simplex vector ``pi ~ Dirichlet(alpha)`` that is mapped to
green-time ratios via ``TrafficSignal._get_green_time_from_ratio``.

Why Dirichlet (vs. Gaussian + softmax):
    * Simplex constraint ``sum(pi) = 1`` is enforced by construction; PPO
      never has to learn it.
    * Log-prob and entropy have closed forms, giving a lower-variance
      policy gradient than sampling through a softmax.
    * Concentration ``alpha`` is naturally interpretable: large alpha ->
      sharp decision, small alpha -> exploratory.

Masking strategy:
    We build the Dirichlet over **only the valid phases** (variable-dim per
    row is not friendly inside PyTorch's batched Dirichlet), so instead we
    fix alpha=1 (uniform) on invalid phases and zero those components of
    the sample afterwards, renormalising the valid ones. Entropy and KL
    contributions from the dummy invalid components are stable (alpha=1
    is a well-behaved uniform), so they add a constant bias that the PPO
    entropy term can absorb without destabilising training.
"""

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


NUM_STANDARD_PHASES = 8

# Bounds on the concentration parameter to keep training stable.
# alpha < 1 -> bimodal (mass at corners); alpha > 1 -> unimodal.
# We keep alpha >= ALPHA_MIN so the policy cannot collapse to a degenerate
# point mass, and <= ALPHA_MAX so entropy does not vanish.
ALPHA_MIN = 1.01
ALPHA_MAX = 50.0

# Invalid phases get alpha = 1 (uniform marginal). This keeps entropy /
# log-prob finite and well-defined; the corresponding components of the
# sample are zeroed and renormalised away before being returned.
ALPHA_INVALID = 1.0


class TorchMaskedDirichlet(TorchDistributionWrapper):
    """Dirichlet over NUM_STANDARD_PHASES simplex with action masking.

    Model output: ``[batch, NUM_STANDARD_PHASES]`` raw logits, passed
    through softplus to yield strictly positive concentrations. Invalid
    phases (``action_mask == 0``) are forced to alpha = 1 and zeroed out
    post-hoc from the sample, then the valid components are renormalised
    back onto the simplex.
    """

    @override(ActionDistribution)
    def __init__(
        self,
        inputs: TensorType,
        model,
        *,
        action_space=None,
    ):
        super().__init__(inputs, model)

        self.num_phases = NUM_STANDARD_PHASES

        # Raw logits -> strictly positive concentration via softplus + offset.
        alpha_raw = inputs.view(-1, self.num_phases)
        alpha = F.softplus(alpha_raw) + ALPHA_MIN
        alpha = torch.clamp(alpha, max=ALPHA_MAX)

        # Apply action mask: invalid phases get ALPHA_INVALID (uniform).
        if hasattr(model, "_last_action_mask") and model._last_action_mask is not None:
            mask = model._last_action_mask.to(inputs.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(alpha.shape[0], -1)
            alpha = alpha * mask + ALPHA_INVALID * (1.0 - mask)
            self.action_mask = mask
        else:
            self.action_mask = torch.ones_like(alpha)

        self.concentration = alpha
        self.dist = Dirichlet(self.concentration)
        self.last_sample = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_to_valid_simplex(self, sample: torch.Tensor) -> torch.Tensor:
        """Zero out invalid phases and renormalise onto the valid simplex."""
        masked = sample * self.action_mask
        total = masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return masked / total

    # ------------------------------------------------------------------
    # ActionDistribution API
    # ------------------------------------------------------------------

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        # Compute the mode of the *valid* Dirichlet marginal, not the full
        # 8-simplex one — otherwise invalid slots (alpha=1) break the
        # `all(alpha > 1)` check and we fall back to the mean, which flattens
        # the policy at evaluation time. Mode over valid components:
        #   mode_i = (alpha_i - 1) / (sum_valid(alpha) - K_valid)   if all valid alpha > 1
        #   else   mean_i = alpha_i / sum_valid(alpha)
        alpha = self.concentration
        mask = self.action_mask
        alpha_valid = alpha * mask  # invalid -> 0 (so they don't contribute to sum)
        k_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sum_valid = alpha_valid.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Check the "mode exists" condition only over valid slots.
        valid_gt1 = ((alpha > 1.0) | (mask == 0.0)).all(dim=-1, keepdim=True)
        denom_mode = (sum_valid - k_valid).clamp(min=1e-6)
        mode_if = (alpha_valid - mask) / denom_mode  # invalid slots are 0 after this
        mode_else = alpha_valid / sum_valid
        mode = torch.where(valid_gt1, mode_if, mode_else)

        # Invalid slots should be exactly 0; then renormalise valid slots.
        mode = mode * mask
        mode = mode / mode.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.last_sample = mode
        return mode

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        s = self.dist.rsample()
        s = self._project_to_valid_simplex(s)
        self.last_sample = s
        return s

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        # The Dirichlet is over the full 8-simplex, but the action coming back
        # from the env is a valid simplex — zero on masked slots and summing
        # to 1 on valid slots. To evaluate log_prob under the full distribution
        # we need a full-simplex point whose marginal over valid slots matches
        # the action exactly. We split a small eps-budget uniformly across
        # invalid slots and scale valid slots by (1 - total_eps) so the joint
        # stays on the simplex without distorting the valid components' ratios.
        a = actions.float()
        a = a * self.action_mask  # drop any noise on masked slots
        valid_count = self.action_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        invalid_count = (self.num_phases - valid_count).clamp(min=0.0)
        eps_per_invalid = 1e-6
        eps_total = eps_per_invalid * invalid_count
        scale = (1.0 - eps_total).clamp(min=1e-6)
        a_full = a * scale + (1.0 - self.action_mask) * eps_per_invalid
        a_full = a_full.clamp(min=1e-6)
        # Final renormalisation only corrects floating-point drift (<1e-6).
        a_full = a_full / a_full.sum(dim=-1, keepdim=True)
        return self.dist.log_prob(a_full)

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        assert self.last_sample is not None, "Must call sample() first"
        return self.logp(self.last_sample)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return self.dist.entropy()

    @override(ActionDistribution)
    def kl(self, other: "TorchMaskedDirichlet") -> TensorType:
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space,
        model_config: ModelConfigDict,
    ) -> Union[int, np.ndarray]:
        return NUM_STANDARD_PHASES


def register_masked_dirichlet():
    """Register the Masked Dirichlet distribution with RLlib."""
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_action_dist(
        "masked_dirichlet", TorchMaskedDirichlet
    )
    print("[MaskedDirichlet] Registered 'masked_dirichlet' action distribution")
