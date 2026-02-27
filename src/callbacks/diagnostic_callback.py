"""
Diagnostic Callback for GESA-MGMQ-PPO.

Implements the Step 0 measurement protocol:
- std(A) and max(|A|)  : Advantage stability
- clip_fraction         : Trust region health
- mean_entropy and KL   : Policy dynamics
- value_loss vs policy_loss magnitude
- MaskedSoftmax std stats (min/max/mean of noise std)
- Effective exploration metrics

This callback logs all metrics to both console and RLlib result dict.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import time

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID


class DiagnosticCallback(DefaultCallbacks):
    """
    Comprehensive diagnostic callback for MGMQ-PPO training.
    
    Logs all metrics required by the GESA-MGMQ-PPO diagnostic protocol:
    
    Step 0 Metrics:
    ├── Advantage Statistics
    │   ├── std(A)           : Standard deviation of advantages
    │   ├── max(|A|)         : Maximum absolute advantage
    │   ├── mean(A)          : Mean advantage (should be ~0 after normalization)
    │   └── skew(A)          : Skewness of advantages
    ├── Trust Region Health
    │   ├── clip_fraction    : Fraction of clipped policy ratios
    │   └── approx_kl        : Approximate KL divergence
    ├── Policy Dynamics
    │   ├── mean_entropy     : Mean policy entropy
    │   ├── kl_divergence    : KL divergence
    │   └── entropy_rate     : Rate of entropy change
    ├── Loss Magnitudes
    │   ├── policy_loss      : Policy loss magnitude
    │   ├── value_loss       : Value loss magnitude  
    │   └── vl_pl_ratio      : Ratio of value_loss to |policy_loss|
    ├── MaskedSoftmax Stats (if applicable)
    │   ├── min_std          : Minimum noise std
    │   ├── max_std          : Maximum noise std
    │   ├── mean_std         : Mean noise std
    │   └── logit_range      : Max - Min of logits
    └── Gradient Statistics
        ├── grad_norm        : Global gradient norm
        └── grad_max         : Maximum gradient element
    """
    
    def __init__(self, log_interval: int = 1, output_dir: str = None):
        super().__init__()
        self.log_interval = log_interval
        self.output_dir = Path(output_dir) if output_dir else None
        self._iteration = 0
        self._history = []
        self._last_entropy = None
        self._last_raw_reward = None
    
    def on_episode_step(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index=None,
        **kwargs,
    ) -> None:
        """Not needed for raw reward tracking anymore."""
        pass
    
    def on_episode_end(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index=None,
        **kwargs,
    ) -> None:
        """Store raw episode reward as custom metric by reading directly from env."""
        if episode is None or base_env is None or env_index is None:
            return
            
        try:
            # Access the actual underlying environment running in the worker
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped
                
            if hasattr(env, '_episode_raw_reward'):
                raw_sum = float(env._episode_raw_reward)
                episode.custom_metrics["raw_episode_reward"] = raw_sum
                
                # Per-agent calculation
                n_agents = len(env.ts_ids) if hasattr(env, 'ts_ids') else 16
                episode.custom_metrics["raw_reward_per_agent"] = raw_sum / max(n_agents, 1)
                
                # Reset env counter for next episode
                env._episode_raw_reward = 0.0
                
        except Exception as e:
            print(f"CALLBACK ERROR fetching raw reward: {e}")
        
    def on_learn_on_batch(
        self,
        *,
        policy: Policy,
        train_batch: SampleBatch,
        result: dict,
        **kwargs,
    ) -> None:
        """
        Called during policy.learn_on_batch(). 
        Compute advantage and clip fraction statistics.
        """
        # Extract advantages from the training batch
        if Postprocessing.ADVANTAGES in train_batch:
            advantages = train_batch[Postprocessing.ADVANTAGES]
            if isinstance(advantages, torch.Tensor):
                adv_np = advantages.detach().cpu().numpy()
            else:
                adv_np = np.asarray(advantages)
            
            # Advantage statistics
            adv_std = float(np.std(adv_np))
            adv_max_abs = float(np.max(np.abs(adv_np)))
            adv_mean = float(np.mean(adv_np))
            adv_min = float(np.min(adv_np))
            adv_max = float(np.max(adv_np))
            
            # Skewness
            if adv_std > 1e-8:
                adv_skew = float(np.mean(((adv_np - adv_mean) / adv_std) ** 3))
            else:
                adv_skew = 0.0
            
            result["diag/adv_std"] = adv_std
            result["diag/adv_max_abs"] = adv_max_abs
            result["diag/adv_mean"] = adv_mean
            result["diag/adv_min"] = adv_min
            result["diag/adv_max"] = adv_max
            result["diag/adv_skew"] = adv_skew
            
        # Compute clip fraction from action probabilities
        if (SampleBatch.ACTION_LOGP in train_batch and 
            "action_logp" in train_batch and
            SampleBatch.ACTION_DIST_INPUTS in train_batch):
            try:
                old_logp = train_batch[SampleBatch.ACTION_LOGP]
                if isinstance(old_logp, torch.Tensor):
                    old_logp = old_logp.detach()
                    
                # Get current log probs from policy
                with torch.no_grad():
                    # The train_batch should have both old and new action logps
                    # after the policy loss computation
                    pass  # Will be computed in on_train_result
            except Exception:
                pass
    
    def on_train_result(
        self,
        *,
        algorithm=None,
        result: dict,
        **kwargs,
    ) -> None:
        """
        Called after each training iteration.
        Aggregate and log all diagnostic metrics.
        """
        self._iteration += 1
        
        # Get vf_loss_coeff from algorithm config for effective ratio computation
        self._vf_loss_coeff = 1.0  # default
        if algorithm is not None:
            try:
                self._vf_loss_coeff = algorithm.config.get("vf_loss_coeff", 1.0)
            except Exception:
                pass
        
        # Extract metrics from RLlib result
        learner_stats = (
            result.get("info", {})
            .get("learner", {})
            .get("default_policy", {})
            .get("learner_stats", {})
        )
        
        # === CORE METRICS FROM RLLIB ===
        entropy = learner_stats.get("entropy", None)
        kl = learner_stats.get("kl", None)
        policy_loss = learner_stats.get("policy_loss", None)
        vf_loss = learner_stats.get("vf_loss", None)
        total_loss = learner_stats.get("total_loss", None)
        grad_norm = learner_stats.get("grad_gnorm", None)
        vf_explained_var = learner_stats.get("vf_explained_var", None)
        cur_kl_coeff = learner_stats.get("cur_kl_coeff", None)
        cur_lr = learner_stats.get("cur_lr", None)
        entropy_coeff = learner_stats.get("entropy_coeff", None)
        
        # === CUSTOM STATS FROM MGMQPPOTorchPolicy ===
        clip_fraction = learner_stats.get("clip_fraction", None)
        adv_std_raw = learner_stats.get("adv_std_raw", None)
        adv_max_abs_raw = learner_stats.get("adv_max_abs_raw", None)
        adv_mean_raw = learner_stats.get("adv_mean_raw", None)
        logp_ratio_mean = learner_stats.get("logp_ratio_mean", None)
        logp_ratio_max = learner_stats.get("logp_ratio_max", None)
        
        # === COMPUTE DERIVED METRICS ===
        metrics = {}
        
        # Value-to-policy loss ratio (both raw and effective)
        if vf_loss is not None and policy_loss is not None:
            pl_abs = abs(policy_loss) if abs(policy_loss) > 1e-8 else 1e-8
            raw_ratio = vf_loss / pl_abs
            effective_ratio = (self._vf_loss_coeff * vf_loss) / pl_abs
            metrics["diag/vl_pl_ratio"] = raw_ratio
            metrics["diag/vl_pl_ratio_effective"] = effective_ratio
            metrics["diag/vf_loss_coeff"] = self._vf_loss_coeff
            metrics["diag/vf_loss"] = vf_loss
            metrics["diag/policy_loss"] = policy_loss
        
        # Entropy rate of change
        if entropy is not None:
            if self._last_entropy is not None:
                metrics["diag/entropy_delta"] = entropy - self._last_entropy
            self._last_entropy = entropy
            metrics["diag/entropy"] = entropy
        
        # Custom advantage stats from MGMQPPOTorchPolicy
        if adv_std_raw is not None:
            metrics["diag/adv_std"] = float(adv_std_raw)
            metrics["diag/adv_max_abs"] = float(adv_max_abs_raw) if adv_max_abs_raw is not None else 0
            metrics["diag/adv_mean"] = float(adv_mean_raw) if adv_mean_raw is not None else 0
            if cur_lr is not None:
                metrics["diag/lr_x_adv_std"] = cur_lr * float(adv_std_raw)
        
        # Custom clip fraction from MGMQPPOTorchPolicy
        if clip_fraction is not None:
            metrics["diag/clip_fraction"] = float(clip_fraction)
        
        # Log probability ratio stats
        if logp_ratio_mean is not None:
            metrics["diag/logp_ratio_mean"] = float(logp_ratio_mean)
        if logp_ratio_max is not None:
            metrics["diag/logp_ratio_max"] = float(logp_ratio_max)
        
        # LR * std(A) already computed above from custom stats
        
        # KL coefficient tracking
        if cur_kl_coeff is not None:
            metrics["diag/kl_coeff"] = cur_kl_coeff
        if kl is not None:
            metrics["diag/kl"] = kl
            
        # Grad norm
        if grad_norm is not None:
            metrics["diag/grad_norm"] = grad_norm
            
        # VF explained variance
        if vf_explained_var is not None:
            metrics["diag/vf_explained_var"] = vf_explained_var
        
        # === TRY TO COMPUTE CLIP FRACTION AND MODEL-LEVEL STATS ===
        try:
            self._compute_model_diagnostics(algorithm, metrics)
        except Exception as e:
            metrics["diag/model_diag_error"] = str(e)
        
        # === REWARD STATS ===
        env_runners = result.get("env_runners", {})
        reward_mean = env_runners.get("episode_reward_mean", None)
        reward_max = env_runners.get("episode_reward_max", None)
        reward_min = env_runners.get("episode_reward_min", None)
        if reward_mean is not None:
            metrics["diag/reward_mean"] = reward_mean
            metrics["diag/reward_range"] = (reward_max or 0) - (reward_min or 0)
        
        # === RAW REWARD STATS (un-normalized, for monitoring policy quality) ===
        custom_metrics = env_runners.get("custom_metrics", {})
        raw_reward_mean = custom_metrics.get("raw_episode_reward_mean", None)
        raw_reward_per_agent = custom_metrics.get("raw_reward_per_agent_mean", None)
        if raw_reward_mean is not None:
            metrics["diag/raw_reward_mean"] = raw_reward_mean
        if raw_reward_per_agent is not None:
            metrics["diag/raw_reward_per_agent"] = raw_reward_per_agent
        
        # Track raw reward trend
        if raw_reward_mean is not None:
            if self._last_raw_reward is not None:
                metrics["diag/raw_reward_delta"] = raw_reward_mean - self._last_raw_reward
            self._last_raw_reward = raw_reward_mean
        
        # Store metrics in result
        result.update(metrics)
        
        # Store history
        record = {
            "iteration": self._iteration,
            "timestamp": time.time(),
        }
        record.update(metrics)
        self._history.append(record)
        
        # Log to console
        if self._iteration % self.log_interval == 0:
            self._print_diagnostic_report(metrics, result)
        
        # Save to file
        if self.output_dir:
            self._save_diagnostics()
    
    def _compute_model_diagnostics(self, algorithm, metrics: dict):
        """
        Compute clip fraction and model-level diagnostics by accessing the policy.
        """
        if algorithm is None:
            return
            
        try:
            policy = algorithm.get_policy("default_policy")
            if policy is None:
                return
                
            model = policy.model
            
            # === MASKED SOFTMAX SPECIFIC STATS ===
            if hasattr(model, 'use_masked_softmax') and model.use_masked_softmax:
                # Get log_std from the last policy output
                if hasattr(model, 'policy_out') and hasattr(model.policy_out, 'weight'):
                    with torch.no_grad():
                        # Access the policy head bias to infer typical log_std range
                        weight = model.policy_out.weight.data
                        bias = model.policy_out.bias.data if model.policy_out.bias is not None else None
                        
                        action_dim = model.action_dim
                        if bias is not None and len(bias) >= 2 * action_dim:
                            # log_std portion of the output bias
                            log_std_bias = bias[action_dim:].cpu().numpy()
                            std_from_bias = np.exp(log_std_bias)
                            metrics["diag/log_std_bias_mean"] = float(np.mean(log_std_bias))
                            metrics["diag/log_std_bias_min"] = float(np.min(log_std_bias))
                            metrics["diag/log_std_bias_max"] = float(np.max(log_std_bias))
                            metrics["diag/std_from_bias_mean"] = float(np.mean(std_from_bias))
            
            # === GRADIENT STATISTICS ===
            grad_norms_by_module = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_data = param.grad.data
                    grad_norm = float(grad_data.norm(2).item())
                    grad_max = float(grad_data.abs().max().item())
                    
                    # Categorize by module
                    module_name = name.split('.')[0]
                    if module_name not in grad_norms_by_module:
                        grad_norms_by_module[module_name] = []
                    grad_norms_by_module[module_name].append(grad_norm)
            
            for module_name, norms in grad_norms_by_module.items():
                metrics[f"diag/grad_{module_name}_mean"] = float(np.mean(norms))
                metrics[f"diag/grad_{module_name}_max"] = float(np.max(norms))
                
        except Exception as e:
            metrics["diag/model_access_error"] = str(e)[:100]
    
    def _print_diagnostic_report(self, metrics: dict, result: dict):
        """Print a formatted diagnostic report to console."""
        learner_stats = (
            result.get("info", {})
            .get("learner", {})
            .get("default_policy", {})
            .get("learner_stats", {})
        )
        
        print("\n" + "=" * 90)
        print(f"  GESA-MGMQ-PPO DIAGNOSTIC REPORT — Iteration {self._iteration}")
        print("=" * 90)
        
        # Reward (both normalized and raw)
        reward_mean = metrics.get("diag/reward_mean", "N/A")
        reward_range = metrics.get("diag/reward_range", "N/A")
        raw_reward = metrics.get("diag/raw_reward_mean", "N/A")
        raw_per_agent = metrics.get("diag/raw_reward_per_agent", "N/A")
        raw_delta = metrics.get("diag/raw_reward_delta", "N/A")
        print(f"\n  📊 REWARD")
        print(f"     normalized: mean={reward_mean:.2f}" if isinstance(reward_mean, float) else f"     normalized: mean={reward_mean}")
        print(f"     🎯 RAW (un-normalized): episode={raw_reward:.2f}" if isinstance(raw_reward, float) else f"     🎯 RAW (un-normalized): episode={raw_reward}")
        if isinstance(raw_per_agent, float):
            print(f"     🎯 RAW per-agent: {raw_per_agent:.2f}")
        if isinstance(raw_delta, float):
            trend = "↑" if raw_delta > 0 else "↓" if raw_delta < 0 else "→"
            print(f"     🎯 RAW trend: {trend} delta={raw_delta:+.2f}")
        print(f"     range={reward_range:.2f}" if isinstance(reward_range, float) else f"     range={reward_range}")
        
        # Advantage Statistics
        adv_std = metrics.get("diag/adv_std", "N/A")
        adv_max_abs = metrics.get("diag/adv_max_abs", "N/A")
        adv_mean = metrics.get("diag/adv_mean", "N/A")
        lr_x_adv = metrics.get("diag/lr_x_adv_std", "N/A")
        print(f"\n  📐 ADVANTAGE STABILITY")
        print(f"     std(A)={adv_std}" if not isinstance(adv_std, float) else f"     std(A)={adv_std:.4f}")
        print(f"     max(|A|)={adv_max_abs}" if not isinstance(adv_max_abs, float) else f"     max(|A|)={adv_max_abs:.4f}")
        print(f"     mean(A)={adv_mean}" if not isinstance(adv_mean, float) else f"     mean(A)={adv_mean:.6f}")
        print(f"     lr×std(A)={lr_x_adv}" if not isinstance(lr_x_adv, float) else f"     lr×std(A)={lr_x_adv:.6f}")
        if isinstance(lr_x_adv, float):
            if lr_x_adv > 0.01:
                print(f"     ⚠️  lr×std(A)={lr_x_adv:.6f} >> 0.01 → Advantage scale too large!")
            else:
                print(f"     ✅ lr×std(A) in safe range")
        
        # Trust Region
        kl = metrics.get("diag/kl", "N/A")
        kl_coeff = metrics.get("diag/kl_coeff", "N/A")
        clip_frac = metrics.get("diag/clip_fraction", "N/A")
        ratio_mean = metrics.get("diag/logp_ratio_mean", "N/A")
        ratio_max = metrics.get("diag/logp_ratio_max", "N/A")
        print(f"\n  🛡️  TRUST REGION")
        print(f"     kl_divergence={kl}" if not isinstance(kl, float) else f"     kl_divergence={kl:.6f}")
        print(f"     kl_coeff={kl_coeff}" if not isinstance(kl_coeff, float) else f"     kl_coeff={kl_coeff:.4f}")
        print(f"     clip_fraction={clip_frac}" if not isinstance(clip_frac, float) else f"     clip_fraction={clip_frac:.4f}")
        print(f"     logp_ratio: mean={ratio_mean}" if not isinstance(ratio_mean, float) else f"     logp_ratio: mean={ratio_mean:.4f}, max={ratio_max:.4f}" if isinstance(ratio_max, float) else f"     logp_ratio: mean={ratio_mean:.4f}")
        if isinstance(kl_coeff, float) and kl_coeff > 1.0:
            print(f"     ⚠️  kl_coeff={kl_coeff:.4f} > 1.0 → Policy updates heavily penalized!")
        
        # Policy Dynamics
        entropy = metrics.get("diag/entropy", "N/A")
        entropy_delta = metrics.get("diag/entropy_delta", "N/A")
        entropy_coeff = learner_stats.get("entropy_coeff", "N/A")
        print(f"\n  🔄 POLICY DYNAMICS")
        print(f"     entropy={entropy}" if not isinstance(entropy, float) else f"     entropy={entropy:.4f}")
        print(f"     entropy_delta={entropy_delta}" if not isinstance(entropy_delta, float) else f"     entropy_delta={entropy_delta:.6f}")
        print(f"     entropy_coeff={entropy_coeff}")
        
        # Loss Magnitudes
        vf_loss = metrics.get("diag/vf_loss", "N/A")
        pl = metrics.get("diag/policy_loss", "N/A")
        raw_ratio = metrics.get("diag/vl_pl_ratio", "N/A")
        eff_ratio = metrics.get("diag/vl_pl_ratio_effective", "N/A")
        vf_coeff = metrics.get("diag/vf_loss_coeff", "N/A")
        print(f"\n  📉 LOSS MAGNITUDES")
        print(f"     policy_loss={pl}" if not isinstance(pl, float) else f"     policy_loss={pl:.6f}")
        print(f"     value_loss={vf_loss}" if not isinstance(vf_loss, float) else f"     value_loss={vf_loss:.4f}")
        print(f"     vf_loss_coeff={vf_coeff}")
        print(f"     raw vl/|pl| ratio={raw_ratio}" if not isinstance(raw_ratio, float) else f"     raw vl/|pl| ratio={raw_ratio:.1f}")
        print(f"     effective vl/|pl| ratio={eff_ratio}" if not isinstance(eff_ratio, float) else f"     effective vl/|pl| ratio={eff_ratio:.1f} (= coeff × vf_loss / |pl|)")
        if isinstance(eff_ratio, float) and eff_ratio > 100:
            print(f"     ⚠️  Effective gradient ratio {eff_ratio:.0f}x → Policy updates suppressed, lower vf_loss_coeff")
        elif isinstance(eff_ratio, float) and eff_ratio > 50:
            print(f"     ⚠️  Effective gradient ratio {eff_ratio:.0f}x → Moderately critic-heavy")
        elif isinstance(eff_ratio, float):
            print(f"     ✅ Effective gradient ratio {eff_ratio:.0f}x → Healthy range")
        
        # MaskedSoftmax Stats
        log_std_mean = metrics.get("diag/log_std_bias_mean", None)
        if log_std_mean is not None:
            print(f"\n  🎯 MASKED SOFTMAX NOISE")
            print(f"     log_std_bias: mean={log_std_mean:.4f}, "
                  f"min={metrics.get('diag/log_std_bias_min', 'N/A'):.4f}, "
                  f"max={metrics.get('diag/log_std_bias_max', 'N/A'):.4f}")
            print(f"     std_from_bias_mean={metrics.get('diag/std_from_bias_mean', 'N/A'):.4f}")
        
        # Gradient Stats
        grad_norm = metrics.get("diag/grad_norm", "N/A")
        print(f"\n  🔬 GRADIENTS")
        print(f"     global_norm={grad_norm}" if not isinstance(grad_norm, float) else f"     global_norm={grad_norm:.4f}")
        # Per-module gradient stats
        for key, val in sorted(metrics.items()):
            if key.startswith("diag/grad_") and key != "diag/grad_norm":
                short_name = key.replace("diag/grad_", "")
                print(f"     {short_name}={val:.6f}" if isinstance(val, float) else f"     {short_name}={val}")
        
        # VF Explained Variance
        vf_ev = metrics.get("diag/vf_explained_var", "N/A")
        print(f"\n  📈 VALUE FUNCTION")
        print(f"     vf_explained_var={vf_ev}" if not isinstance(vf_ev, float) else f"     vf_explained_var={vf_ev:.4f}")
        
        # === TRIGGER ASSESSMENT ===
        print(f"\n  {'─' * 60}")
        print(f"  🔍 TRIGGER ASSESSMENT")
        triggers = self._assess_triggers(metrics)
        for trigger_name, (triggered, reason) in triggers.items():
            status = "🔴 TRIGGERED" if triggered else "🟢 OK"
            print(f"     {status} {trigger_name}: {reason}")
        
        print("=" * 90 + "\n")
    
    def _assess_triggers(self, metrics: dict) -> dict:
        """Assess trigger conditions for Steps 1-4."""
        triggers = {}
        
        # Step 1: Advantage & Scale
        lr_x_adv = metrics.get("diag/lr_x_adv_std", None)
        # Use EFFECTIVE ratio (accounts for vf_loss_coeff) for trigger assessment
        eff_ratio = metrics.get("diag/vl_pl_ratio_effective", None)
        kl_coeff = metrics.get("diag/kl_coeff", None)
        
        # Check if advantage scale is too large
        step1_triggered = False
        step1_reasons = []
        if isinstance(lr_x_adv, float) and lr_x_adv > 0.01:
            step1_triggered = True
            step1_reasons.append(f"lr×std(A)={lr_x_adv:.4f} >> 0.01")
        if isinstance(eff_ratio, float) and eff_ratio > 100:
            step1_triggered = True
            step1_reasons.append(f"effective vl/|pl|={eff_ratio:.0f} >> 100 (critic dominates gradient)")
        if isinstance(kl_coeff, float) and kl_coeff > 1.0:
            step1_triggered = True
            step1_reasons.append(f"kl_coeff={kl_coeff:.2f} > 1.0 (policy updates penalized)")
        triggers["Step 1 (Advantage & Scale)"] = (
            step1_triggered, 
            "; ".join(step1_reasons) if step1_reasons else "All metrics in safe range"
        )
        
        # Step 2: Dirichlet/Distribution Constraints
        # For MaskedSoftmax, check std bounds
        std_mean = metrics.get("diag/std_from_bias_mean", None)
        step2_triggered = False
        step2_reasons = []
        if isinstance(std_mean, float):
            if std_mean < 0.01:
                step2_triggered = True
                step2_reasons.append(f"std_mean={std_mean:.6f} < 0.01 → exploration collapsed")
            if std_mean > 10.0:
                step2_triggered = True
                step2_reasons.append(f"std_mean={std_mean:.4f} > 10.0 → noise explosion")
        triggers["Step 2 (Distribution Bounds)"] = (
            step2_triggered,
            "; ".join(step2_reasons) if step2_reasons else "Distribution within bounds"
        )
        
        # Step 3: GNN Architecture
        grad_norm = metrics.get("diag/grad_norm", None)
        step3_triggered = False
        step3_reasons = []
        if isinstance(grad_norm, float) and grad_norm > 50.0:
            step3_triggered = True
            step3_reasons.append(f"grad_norm={grad_norm:.2f} → gradient spikes")
        triggers["Step 3 (GNN Architecture)"] = (
            step3_triggered,
            "; ".join(step3_reasons) if step3_reasons else "Gradient norms stable"
        )
        
        # Step 4: Macro Stability
        reward_mean = metrics.get("diag/reward_mean", None)
        entropy = metrics.get("diag/entropy", None)
        step4_ready = False
        step4_reasons = []
        if isinstance(reward_mean, float):
            step4_reasons.append(f"reward={reward_mean:.2f}")
        if isinstance(entropy, float):
            step4_reasons.append(f"entropy={entropy:.4f}")
        # Step 4 is triggered when things are STABLE (positive signal)
        triggers["Step 4 (Macro Stability)"] = (
            step4_ready,
            "; ".join(step4_reasons) if step4_reasons else "Need more data"
        )
        
        return triggers
    
    def _save_diagnostics(self):
        """Save diagnostic history to JSON file."""
        if self.output_dir is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            outfile = self.output_dir / "diagnostic_history.json"
            
            # Convert numpy types to Python types for JSON serialization
            clean_history = []
            for record in self._history:
                clean = {}
                for k, v in record.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean[k] = v.tolist()
                    else:
                        clean[k] = v
                clean_history.append(clean)
            
            with open(outfile, 'w') as f:
                json.dump(clean_history, f, indent=2)
        except Exception as e:
            print(f"⚠ Failed to save diagnostics: {e}")
    
    def get_summary(self) -> dict:
        """Get a summary of all trigger assessments across all iterations."""
        if not self._history:
            return {"status": "no data"}
        
        last = self._history[-1]
        return {
            "iterations": len(self._history),
            "last_metrics": last,
            "triggers": self._assess_triggers(last),
        }
