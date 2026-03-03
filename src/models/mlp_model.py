"""
MLP Baseline Model for Traffic Signal Control.

This module provides a simple MLP model (no GNN) as a diagnostic baseline
to determine whether training issues are caused by the GNN architecture 
or by other components (PPO, environment, reward, etc.).

Architecture:
    obs → MLP Encoder → Policy Head (logits + log_std)
                      → Value Head (scalar)

The model uses the same action distribution (MaskedSoftmax) and observation 
space (Dict with features + action_mask) as the GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

# Number of standard phases for action masking
NUM_STANDARD_PHASES = 8

# Log std bounds for MaskedSoftmax
SOFTMAX_LOG_STD_MIN = -5.0
SOFTMAX_LOG_STD_MAX = 1.0  # Wider than GNN model to allow more exploration


class MLPEncoder(nn.Module):
    """
    Simple MLP Encoder for traffic signal observations.
    
    Replaces the GNN encoder (GAT + GraphSAGE + BiGRU) with a simple 
    feedforward network to test if GNN is causing training issues.
    
    Args:
        obs_dim: Observation dimension (e.g., 48 = 12 lanes * 4 features)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate (only used during training)
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.0,
    ):
        super(MLPEncoder, self).__init__()
        self.obs_dim = obs_dim
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self._output_dim = prev_dim
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            [batch, output_dim] encoded features
        """
        return self.network(obs)


class MLPTorchModel(TorchModelV2, nn.Module):
    """
    MLP Model wrapper for RLlib integration.
    
    A simple feedforward baseline that replaces GNN with MLP.
    Uses the same observation space (Dict with features + action_mask)
    and action distribution (MaskedSoftmax) as MGMQTorchModel.
    
    This allows direct A/B comparison: if MLP trains well but GNN doesn't,
    the issue is in the GNN architecture.
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        custom_config = model_config.get("custom_model_config", {})
        
        # Extract observation dimension from obs_space
        if hasattr(obs_space, 'original_space'):
            orig_space = obs_space.original_space
            if hasattr(orig_space, 'spaces') and 'features' in orig_space.spaces:
                obs_dim = int(np.prod(orig_space.spaces['features'].shape))
            else:
                obs_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, 'spaces') and 'features' in obs_space.spaces:
            obs_dim = int(np.prod(obs_space.spaces['features'].shape))
        else:
            obs_dim = int(np.prod(obs_space.shape))
        
        action_dim = int(np.prod(action_space.shape))
        self.action_dim = action_dim
        
        # Model configuration
        encoder_hidden_dims = custom_config.get("encoder_hidden_dims", [256, 128])
        policy_hidden_dims = custom_config.get("policy_hidden_dims", [128, 64])
        value_hidden_dims = custom_config.get("value_hidden_dims", [256, 128, 64])
        dropout = custom_config.get("dropout", 0.0)
        
        # Action distribution mode (always MaskedSoftmax for consistency)
        self.use_masked_softmax = custom_config.get("use_masked_softmax", True)
        self._last_action_mask = None
        
        # === Build Networks ===
        
        # Shared encoder (optional - can also have separate encoders)
        self.encoder = MLPEncoder(
            obs_dim=obs_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )
        
        encoder_dim = self.encoder.output_dim
        
        # Policy head
        policy_layers = []
        prev_dim = encoder_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Policy output: logits + log_std for MaskedSoftmax
        if self.use_masked_softmax:
            self.policy_out = nn.Linear(prev_dim, 2 * action_dim)
        else:
            self.policy_out = nn.Linear(prev_dim, num_outputs)
        
        # Value head (separate from policy - no shared encoder gradient conflict)
        # Build a completely independent value network from obs directly
        value_layers = []
        prev_dim = obs_dim  # Value network gets raw obs, not encoder output
        for hidden_dim in value_hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)
        
        # Internal state
        self._obs_for_value = None
        self._value = None
        
        self._init_weights()
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[MLPTorchModel] obs_dim={obs_dim}, action_dim={action_dim}, "
              f"encoder={encoder_hidden_dims}, policy={policy_hidden_dims}, "
              f"value={value_hidden_dims}, total_params={total_params:,}")
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in [self.encoder.network, self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        
        # Policy output: small gain for stable initial actions
        nn.init.orthogonal_(self.policy_out.weight, gain=0.01)
        nn.init.zeros_(self.policy_out.bias)
        
        # Value output: small gain to start predictions near 0
        nn.init.orthogonal_(self.value_out.weight, gain=0.1)
        nn.init.zeros_(self.value_out.bias)
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass for RLlib."""
        obs_flat = input_dict["obs_flat"].float()
        
        # Extract action_mask and features from flattened obs
        # RLlib flattens Dict keys alphabetically: "action_mask" before "features"
        if self.use_masked_softmax:
            action_mask = obs_flat[..., :NUM_STANDARD_PHASES]
            obs = obs_flat[..., NUM_STANDARD_PHASES:]
            self._last_action_mask = action_mask
        else:
            obs = obs_flat
            self._last_action_mask = None
        
        # Store raw obs for value network (separate from encoder)
        self._obs_for_value = obs
        
        # Encoder → Policy
        features = self.encoder(obs)
        policy_features = self.policy_net(features)
        policy_out = self.policy_out(policy_features)
        
        if self.use_masked_softmax:
            action_dim = self.action_dim
            logits = policy_out[..., :action_dim]
            log_std = policy_out[..., action_dim:]
            log_std = torch.clamp(log_std, SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            policy_out = torch.cat([logits, log_std], dim=-1)
        
        # Value network (completely separate - no gradient conflict)
        value_features = self.value_net(obs)
        self._value = self.value_out(value_features).squeeze(-1)
        
        return policy_out, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value function output."""
        assert self._value is not None, "Must call forward() first"
        return self._value
