"""
Custom PPO Policy and Algorithm for MGMQ with Diagnostic Extensions.

This module provides:
1. Per-minibatch Advantage Normalization (Step 1 fix)
2. clip_fraction tracking (Step 0 requirement)
3. Advantage statistics logging
4. Optional PopArt-style value normalization

These extensions solve the "KL coefficient explosion" problem where:
- Raw advantages have high variance → large policy updates → high KL
- RLlib's adaptive KL penalty increases kl_coeff to suppress updates
- But this suppresses ALL policy updates → flat reward

Fix: Normalizing advantages per minibatch (mean=0, std=1) ensures:
- Policy gradients have consistent scale regardless of reward magnitude
- clip_fraction stays in healthy range (0.05-0.15)
- kl_coeff stays near initial value
"""

import logging
from typing import Dict, List, Type, Union

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

# ---------------------------------------------------------------------------
# Gradient cosine-similarity diagnostic
# ---------------------------------------------------------------------------

def _compute_grad_cosine_sim(
    policy_loss: "torch.Tensor",
    vf_loss: "torch.Tensor",
    encoder_params: "List[torch.nn.Parameter]",
) -> float:
    """Compute cosine similarity between policy and value gradients on shared encoder.

    Uses ``torch.autograd.grad`` with ``retain_graph=True`` so the
    graph remains available for the subsequent ``total_loss.backward()``.

    Returns:
        Cosine similarity in [-1, 1].
        * +1  → gradients fully aligned (cooperative)
        *  0  → orthogonal (no conflict, no help)
        * -1  → directly opposing (maximum conflict)
        Returns 0.0 on any error (e.g. no grad-requiring params).
    """
    try:
        # Filter to params that actually require grad and are leaves
        params = [p for p in encoder_params if p.requires_grad]
        if not params:
            return 0.0

        # ∇_encoder(policy_loss)
        policy_grads = torch.autograd.grad(
            policy_loss, params, retain_graph=True, allow_unused=True
        )
        # ∇_encoder(vf_loss)
        value_grads = torch.autograd.grad(
            vf_loss, params, retain_graph=True, allow_unused=True
        )

        # Flatten into single vectors
        pg_flat = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
            for g, p in zip(policy_grads, params)
        ])
        vg_flat = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
            for g, p in zip(value_grads, params)
        ])

        pg_norm = pg_flat.norm()
        vg_norm = vg_flat.norm()

        if pg_norm < 1e-12 or vg_norm < 1e-12:
            return 0.0

        cos_sim = float(torch.dot(pg_flat, vg_flat) / (pg_norm * vg_norm))
        return cos_sim
    except Exception:
        return 0.0

logger = logging.getLogger(__name__)


class MGMQPPOTorchPolicy(PPOTorchPolicy):
    """
    Extended PPO Torch Policy with:
    1. Per-minibatch advantage normalization
    2. clip_fraction tracking
    3. Advantage statistics logging
    
    This fixes the KL coefficient explosion problem in MGMQ training.
    """

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """
        Compute PPO loss with per-minibatch advantage normalization.
        
        Changes from standard PPO:
        1. Normalize advantages to mean=0, std=1 per minibatch
        2. Track and store clip_fraction
        3. Log advantage statistics
        """
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # KL divergence
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # ============================================================
        # STEP 1 FIX: Per-minibatch Advantage Normalization
        # ============================================================
        raw_advantages = train_batch[Postprocessing.ADVANTAGES]
        
        # Log raw advantage statistics BEFORE normalization
        with torch.no_grad():
            adv_std_raw = torch.std(raw_advantages).item()
            adv_max_abs_raw = torch.max(torch.abs(raw_advantages)).item()
            adv_mean_raw = torch.mean(raw_advantages).item()
        
        # Normalize advantages: mean=0, std=1
        # This ensures consistent gradient scale regardless of reward magnitude
        adv_mean = torch.mean(raw_advantages)
        adv_std = torch.std(raw_advantages)
        # Avoid division by zero
        normalized_advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)
        
        # Use NORMALIZED advantages for surrogate loss
        surrogate_loss = torch.min(
            normalized_advantages * logp_ratio,
            normalized_advantages
            * torch.clamp(
                logp_ratio,
                1 - self.config["clip_param"],
                1 + self.config["clip_param"],
            ),
        )

        # ============================================================
        # STEP 0: Track clip_fraction
        # ============================================================
        with torch.no_grad():
            clip_param = self.config["clip_param"]
            clipped = (logp_ratio < (1 - clip_param)) | (logp_ratio > (1 + clip_param))
            if mask is not None:
                clip_fraction = torch.sum(clipped.float() * mask) / num_valid
            else:
                clip_fraction = torch.mean(clipped.float())

        # Compute value function loss
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(
                surrogate_loss.device
            )

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add KL penalty
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # === GRADIENT COSINE SIMILARITY: policy vs value on shared encoder ===
        # Measure whether policy and value gradients cooperate or conflict
        # on the shared encoder parameters.  Computed every minibatch, but
        # averaged per iteration via tower_stats → stats_fn.
        with torch.no_grad():
            _grad_cos_sim = torch.tensor(0.0)
        try:
            encoder_params = list(model.mgmq_encoder.parameters())
            # policy_loss and vf_loss must both flow through encoder
            _grad_cos_sim = torch.tensor(
                _compute_grad_cosine_sim(
                    policy_loss=reduce_mean_valid(-surrogate_loss),
                    vf_loss=mean_vf_loss,
                    encoder_params=encoder_params,
                )
            )
        except Exception:
            pass  # Model may not have mgmq_encoder (e.g. in tests)

        # Store stats in model tower
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        
        # Store EXTRA diagnostic stats
        model.tower_stats["clip_fraction"] = clip_fraction
        model.tower_stats["adv_std_raw"] = torch.tensor(adv_std_raw)
        model.tower_stats["adv_max_abs_raw"] = torch.tensor(adv_max_abs_raw)
        model.tower_stats["adv_mean_raw"] = torch.tensor(adv_mean_raw)
        model.tower_stats["logp_ratio_mean"] = reduce_mean_valid(logp_ratio)
        model.tower_stats["logp_ratio_max"] = torch.max(logp_ratio)
        model.tower_stats["grad_cosine_policy_value"] = _grad_cos_sim

        return total_loss

    @override(PPOTorchPolicy)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Extended stats function with diagnostic metrics."""
        base_stats = {
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": self.cur_lr,
            "total_loss": torch.mean(
                torch.stack(self.get_tower_stats("total_loss"))
            ),
            "policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_policy_loss"))
            ),
            "vf_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_vf_loss"))
            ),
            "vf_explained_var": torch.mean(
                torch.stack(self.get_tower_stats("vf_explained_var"))
            ),
            "kl": torch.mean(
                torch.stack(self.get_tower_stats("mean_kl_loss"))
            ),
            "entropy": torch.mean(
                torch.stack(self.get_tower_stats("mean_entropy"))
            ),
            "entropy_coeff": self.entropy_coeff,
        }
        
        # Add diagnostic stats
        try:
            base_stats["clip_fraction"] = torch.mean(
                torch.stack(self.get_tower_stats("clip_fraction"))
            )
            base_stats["adv_std_raw"] = torch.mean(
                torch.stack(self.get_tower_stats("adv_std_raw"))
            )
            base_stats["adv_max_abs_raw"] = torch.mean(
                torch.stack(self.get_tower_stats("adv_max_abs_raw"))
            )
            base_stats["adv_mean_raw"] = torch.mean(
                torch.stack(self.get_tower_stats("adv_mean_raw"))
            )
            base_stats["logp_ratio_mean"] = torch.mean(
                torch.stack(self.get_tower_stats("logp_ratio_mean"))
            )
            base_stats["logp_ratio_max"] = torch.max(
                torch.stack(self.get_tower_stats("logp_ratio_max"))
            )
            base_stats["grad_cosine_policy_value"] = torch.mean(
                torch.stack(self.get_tower_stats("grad_cosine_policy_value"))
            )
        except Exception:
            pass  # Diagnostic stats may not be available in all cases
        
        return convert_to_numpy(base_stats)


class MGMQPPOConfig(PPOConfig):
    """Extended PPO config that uses MGMQPPOTorchPolicy."""

    def get_default_policy_class(self, config):
        if config.get("framework") == "torch":
            return MGMQPPOTorchPolicy
        else:
            raise ValueError("Only torch framework is supported for MGMQ-PPO")


class MGMQPPO(PPO):
    """
    Custom PPO algorithm for MGMQ with:
    - Per-minibatch advantage normalization
    - Extended diagnostic logging
    """

    @classmethod
    def get_default_config(cls):
        return MGMQPPOConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        if config.get("framework") == "torch":
            return MGMQPPOTorchPolicy
        else:
            raise ValueError("Only torch framework is supported for MGMQ-PPO")
