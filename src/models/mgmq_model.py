"""MGMQ Model: GNN-based encoder with Actor-Critic heads for RLlib PPO.

Architecture:
  1. Split observation → lane features [48] + green-time ratios [8]
  2. GAT (Dual-Stream) for intersection embedding on lane features
  3. GraphSAGE + Bi-GRU for network/neighbour embedding
  4. Concatenate intersection_emb + network_emb + green_time → joint embedding
  5. Policy head → action logits
  6. Value head → state value
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from .gat_layer import (
    DualStreamGATLayer,
    get_lane_conflict_matrix,
    get_lane_cooperation_matrix,
)
from .graphsage_bigru import GraphSAGE_BiGRU, NeighborGraphSAGE_BiGRU

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LANES = 12
NUM_LANE_FEATURES = 4
LANE_OBS_DIM = NUM_LANES * NUM_LANE_FEATURES  # 48
GREEN_TIME_DIM = 8                              # 8 standard phases
TOTAL_OBS_DIM = LANE_OBS_DIM + GREEN_TIME_DIM  # 56
NUM_STANDARD_PHASES = 8

# Log-std bounds
SOFTMAX_LOG_STD_MIN = -5.0
SOFTMAX_LOG_STD_MAX = 0.5
LOG_STD_MIN = -20.0
LOG_STD_MAX = 0.5


# ---------------------------------------------------------------------------
# Network adjacency builder
# ---------------------------------------------------------------------------

def build_network_adjacency(
    ts_ids: list,
    net_file: str,
    directional: bool = True,
) -> torch.Tensor:
    """Build adjacency matrix from SUMO .net.xml.

    Returns [4, N, N] if directional, else [N, N].
    """
    import xml.etree.ElementTree as ET

    N = len(ts_ids)
    ts_set = set(ts_ids)
    ts_to_idx = {ts: i for i, ts in enumerate(ts_ids)}
    adj = torch.zeros(4, N, N) if directional else torch.eye(N)

    if not net_file:
        return adj

    try:
        tree = ET.parse(net_file)
        root = tree.getroot()

        coords = {}
        for j in root.findall("junction"):
            coords[j.get("id")] = (float(j.get("x", 0)), float(j.get("y", 0)))

        graph: Dict[str, set] = defaultdict(set)
        for edge in root.findall(".//edge"):
            if edge.get("id", "").startswith(":"):
                continue
            f, t = edge.get("from"), edge.get("to")
            if f and t:
                graph[f].add(t)
                graph[t].add(f)

        def _bfs_neighbours(start: str) -> set:
            found, visited, queue = set(), {start}, list(graph[start])
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                if cur in ts_set:
                    found.add(cur)
                else:
                    queue.extend(n for n in graph[cur] if n not in visited)
            return found

        def _direction(from_id: str, to_id: str) -> int:
            if from_id not in coords or to_id not in coords:
                return -1
            dx = coords[to_id][0] - coords[from_id][0]
            dy = coords[to_id][1] - coords[from_id][1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return -1
            angle = math.degrees(math.atan2(dy, dx)) % 360
            if 45 <= angle < 135:
                return 0   # N
            if 135 <= angle < 225:
                return 3   # W
            if 225 <= angle < 315:
                return 2   # S
            return 1       # E

        for ts_id in ts_ids:
            i = ts_to_idx[ts_id]
            for nb in _bfs_neighbours(ts_id):
                if nb in ts_to_idx:
                    j = ts_to_idx[nb]
                    if directional:
                        d = _direction(ts_id, nb)
                        if d >= 0:
                            adj[d, i, j] = 1.0
                    else:
                        adj[i, j] = 1
                        adj[j, i] = 1
    except Exception as e:
        print(f"Warning: adjacency build failed: {e}")

    return adj


# ---------------------------------------------------------------------------
# MGMQEncoder  (global graph)
# ---------------------------------------------------------------------------

class MGMQEncoder(nn.Module):
    """GAT + GraphSAGE_BiGRU encoder.

    Splits observation into lane features [48] (→ GAT) and green-time
    ratios [8] (concatenated after GAT).
    """

    def __init__(
        self,
        obs_dim: int = TOTAL_OBS_DIM,
        num_agents: int = 1,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 32,
        gat_num_heads: int = 4,
        graphsage_hidden_dim: int = 64,
        gru_hidden_dim: int = 32,
        dropout: float = 0.3,
        network_adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents

        # Lane / green-time split
        self.lane_obs_dim = LANE_OBS_DIM
        self.green_time_dim = GREEN_TIME_DIM
        self.lane_feature_dim = NUM_LANE_FEATURES  # 4

        # If obs_dim differs from expected 56, use a projection
        self._needs_input_proj = (obs_dim != TOTAL_OBS_DIM)
        if self._needs_input_proj:
            self.input_proj = nn.Linear(obs_dim, TOTAL_OBS_DIM)

        # GAT
        self.dual_stream_gat = DualStreamGATLayer(
            in_features=self.lane_feature_dim,
            hidden_dim=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.05,
        )
        gat_per_lane = self.dual_stream_gat.final_output_dim
        gat_total = NUM_LANES * gat_per_lane

        # GraphSAGE
        self.graphsage_bigru = GraphSAGE_BiGRU(
            in_features=gat_total,
            hidden_features=graphsage_hidden_dim,
            gru_hidden_size=gru_hidden_dim,
            dropout=dropout,
        )

        # Adjacency
        if network_adjacency is not None:
            if network_adjacency.dim() == 2:
                network_adjacency = network_adjacency.unsqueeze(0).expand(4, -1, -1).clone()
            self.register_buffer("network_adj", network_adjacency)
        else:
            self.register_buffer("network_adj", torch.ones(4, max(1, num_agents), max(1, num_agents)))

        # Lane adjacency (static 12×12)
        self.register_buffer("lane_adj_coop", get_lane_cooperation_matrix())
        self.register_buffer("lane_adj_conf", get_lane_conflict_matrix())

        # Green-time projection
        self.green_proj = nn.Sequential(
            nn.Linear(GREEN_TIME_DIM, gat_hidden_dim),
            nn.ReLU(),
        )

        # Output dims
        self.intersection_emb_dim = gat_total
        self.network_emb_dim = graphsage_hidden_dim
        self.green_emb_dim = gat_hidden_dim
        self.joint_emb_dim = self.intersection_emb_dim + self.network_emb_dim + self.green_emb_dim

    @property
    def output_dim(self) -> int:
        return self.joint_emb_dim

    def forward(self, obs: torch.Tensor, agent_idx: Optional[int] = None):
        B = obs.size(0)

        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            num_agents = 1
        else:
            num_agents = obs.size(1)

        obs_flat = obs.reshape(-1, self.obs_dim)

        # Project if needed
        if self._needs_input_proj:
            obs_flat = self.input_proj(obs_flat)

        # Split: lane [48] + green [8]
        lane_obs = obs_flat[:, :LANE_OBS_DIM]
        green_obs = obs_flat[:, LANE_OBS_DIM:LANE_OBS_DIM + GREEN_TIME_DIM]

        # GAT on lanes
        lane_feat = lane_obs.view(-1, NUM_LANES, self.lane_feature_dim)
        adj_coop = self.lane_adj_coop.unsqueeze(0).expand(lane_feat.size(0), -1, -1)
        adj_conf = self.lane_adj_conf.unsqueeze(0).expand(lane_feat.size(0), -1, -1)
        gat_out = self.dual_stream_gat(lane_feat, adj_coop, adj_conf)
        intersection_flat = gat_out.reshape(gat_out.size(0), -1)
        intersection_emb = intersection_flat.view(B, num_agents, -1)

        # GraphSAGE
        net_adj = (
            torch.ones(4, 1, 1, device=obs.device)
            if num_agents == 1
            else self.network_adj[:, :num_agents, :num_agents]
        )
        network_emb_seq = self.graphsage_bigru(intersection_emb, net_adj)

        # Select agent
        if agent_idx is not None and num_agents > 1:
            network_emb = network_emb_seq[:, agent_idx, :]
            agent_int_emb = intersection_emb[:, agent_idx, :]
        else:
            network_emb = network_emb_seq.mean(dim=1)
            agent_int_emb = intersection_emb.mean(dim=1)

        # Green-time embedding
        green_emb = self.green_proj(green_obs.view(B, num_agents, GREEN_TIME_DIM).mean(dim=1))

        return torch.cat([agent_int_emb, network_emb, green_emb], dim=-1), agent_int_emb, network_emb


# ---------------------------------------------------------------------------
# LocalMGMQEncoder  (star-graph / local GNN)
# ---------------------------------------------------------------------------

class LocalMGMQEncoder(nn.Module):
    """
    Local MGMQ Encoder with Spatial Neighbor Aggregation.

    Used when --use-local-gnn is enabled. Each agent aggregates information
    from its neighbors using BiGRU for SPATIAL aggregation.

    Architecture:
    1. GAT on lanes for self node
    2. GAT on lanes for each neighbor node
    3. NeighborGraphSAGE_BiGRU for SPATIAL aggregation over neighbors

    Args:
        obs_dim: Feature dimension (48 = 4 features * 12 detectors)
        max_neighbors: Maximum number of neighbors (K)
        gat_hidden_dim: GAT hidden dimension
        gat_output_dim: GAT output dimension per head
        gat_num_heads: Number of GAT attention heads
        graphsage_hidden_dim: GraphSAGE hidden dimension
        gru_hidden_dim: BiGRU hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        obs_dim: int = 56,
        max_neighbors: int = 4,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 32,
        gat_num_heads: int = 4,
        graphsage_hidden_dim: int = 64,
        gru_hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super(LocalMGMQEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.max_neighbors = max_neighbors
        self.num_lanes = NUM_LANES         # 12
        self.lane_obs_dim = LANE_OBS_DIM   # 48 (12 * 4)
        self.lane_feature_dim = NUM_LANE_FEATURES  # 4
        # Whether the observation carries green-time ratios after the lane block.
        self._has_green_features = obs_dim >= TOTAL_OBS_DIM

        # Dual-Stream GAT
        self.dual_stream_gat = DualStreamGATLayer(
            in_features=self.lane_feature_dim,
            hidden_dim=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.05,
        )

        # Static lane adjacency matrices
        self.register_buffer("lane_adj_coop", get_lane_cooperation_matrix())
        self.register_buffer("lane_adj_conf", get_lane_conflict_matrix())

        # GAT output: FLATTEN over 12 lanes (paper Eq.18)
        self.gat_per_lane_output = self.dual_stream_gat.final_output_dim
        self.gat_total_output = 12 * self.gat_per_lane_output

        # NeighborGraphSAGE_BiGRU for SPATIAL aggregation
        self.neighbor_aggregator = NeighborGraphSAGE_BiGRU(
            in_features=self.gat_total_output,
            hidden_features=graphsage_hidden_dim,
            gru_hidden_size=gru_hidden_dim,
            max_neighbors=max_neighbors,
            dropout=dropout,
        )

        # Self green-time projection. The critic needs the current phase state
        # to predict return; without it, value-function explained variance
        # collapses on multi-agent networks where reward is phase-conditioned.
        if self._has_green_features:
            self.green_proj = nn.Sequential(
                nn.Linear(GREEN_TIME_DIM, gat_hidden_dim),
                nn.ReLU(),
            )
            self.green_emb_dim = gat_hidden_dim
        else:
            self.green_proj = None
            self.green_emb_dim = 0

        # Output dimensions
        self.intersection_emb_dim = self.gat_total_output
        self.network_emb_dim = graphsage_hidden_dim
        self.joint_emb_dim = (
            self.intersection_emb_dim + self.network_emb_dim + self.green_emb_dim
        )

    @property
    def output_dim(self) -> int:
        return self.joint_emb_dim

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for Dict observation.

        Args:
            obs_dict: Dict with keys:
                - self_features: [B, obs_dim]      (lane [48] + optional green [8])
                - neighbor_features: [B, K, obs_dim]
                - neighbor_mask: [B, K]
                - neighbor_directions: [B, K] (optional, 0.0=N, 0.25=E, 0.5=S, 0.75=W)

        Returns:
            joint_emb: [B, joint_emb_dim]
        """
        self_feat = obs_dict["self_features"]          # [B, obs_dim]
        neighbor_feat = obs_dict["neighbor_features"]  # [B, K, obs_dim]
        mask = obs_dict["neighbor_mask"]               # [B, K]
        neighbor_dirs = obs_dict.get("neighbor_directions", None)

        B = self_feat.size(0)
        K = neighbor_feat.size(1)

        # 1. GAT on self lane features
        self_emb = self._run_gat(self_feat)

        # 2. GAT on neighbor lane features
        neighbor_feat_flat = neighbor_feat.reshape(B * K, -1)
        neighbor_emb_flat = self._run_gat(neighbor_feat_flat)
        neighbor_emb = neighbor_emb_flat.reshape(B, K, -1)

        # 3. Spatial neighbor aggregation
        network_emb = self.neighbor_aggregator(
            self_features=self_emb,
            neighbor_features=neighbor_emb,
            neighbor_mask=mask,
            neighbor_directions=neighbor_dirs,
        )

        # 4. Joint embedding (with self green-time if present in obs)
        parts = [self_emb, network_emb]
        if self.green_proj is not None:
            green = self_feat[:, self.lane_obs_dim:self.lane_obs_dim + GREEN_TIME_DIM]
            parts.append(self.green_proj(green))
        return torch.cat(parts, dim=-1)

    def _run_gat(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GAT to lane features and FLATTEN (paper Eq.18).

        x is [B, obs_dim] where obs_dim=56 (48 lane + 8 green) or 48 (lane only).
        Only the first lane_obs_dim (48) dims are used for GAT.
        """
        batch_size = x.size(0)
        lane_only = x[:, :self.lane_obs_dim]  # [B, 48]
        lane_feat = lane_only.view(batch_size, self.num_lanes, self.lane_feature_dim)
        adj_coop = self.lane_adj_coop.unsqueeze(0).expand(batch_size, -1, -1)
        adj_conf = self.lane_adj_conf.unsqueeze(0).expand(batch_size, -1, -1)
        gat_out = self.dual_stream_gat(lane_feat, adj_coop, adj_conf)
        return gat_out.reshape(batch_size, -1)


# ---------------------------------------------------------------------------
# Policy / Value head builder
# ---------------------------------------------------------------------------

def _build_head(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    layers: list = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
        prev = h
    return nn.Sequential(*layers)


def _init_model_weights(policy_net, value_net, policy_out, value_out):
    for module in [policy_net, value_net]:
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(policy_out.weight, gain=0.01)
    nn.init.zeros_(policy_out.bias)
    nn.init.orthogonal_(value_out.weight, gain=0.1)
    nn.init.zeros_(value_out.bias)


# ---------------------------------------------------------------------------
# MGMQTorchModel  (RLlib wrapper – global graph)
# ---------------------------------------------------------------------------

class MGMQTorchModel(TorchModelV2, nn.Module):
    """RLlib-compatible MGMQ model with global graph encoder and Actor-Critic heads."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})

        # Observation dim
        obs_dim = self._resolve_obs_dim(obs_space)
        self.action_dim = int(np.prod(action_space.shape)) if hasattr(action_space, "shape") else int(np.prod(action_space.nvec.shape)) * 3 if hasattr(action_space, "nvec") else 8

        # Distribution mode
        self.use_masked_softmax = cfg.get("use_masked_softmax", True)
        self.vf_share_coeff = cfg.get("vf_share_coeff", 1.0)
        self._last_action_mask = None

        # Action mode
        self.action_mode = cfg.get("action_mode", "ratio")

        # Encoder
        ts_ids = cfg.get("ts_ids")
        net_file = cfg.get("net_file")
        net_adj = build_network_adjacency(ts_ids, net_file, directional=True) if ts_ids else None
        self.mgmq_encoder = MGMQEncoder(
            obs_dim=obs_dim,
            num_agents=cfg.get("num_agents", 1),
            gat_hidden_dim=cfg.get("gat_hidden_dim", 64),
            gat_output_dim=cfg.get("gat_output_dim", 32),
            gat_num_heads=cfg.get("gat_num_heads", 4),
            graphsage_hidden_dim=cfg.get("graphsage_hidden_dim", 64),
            gru_hidden_dim=cfg.get("gru_hidden_dim", 32),
            dropout=cfg.get("dropout", 0.3),
            network_adjacency=net_adj,
        )

        emb_dim = self.mgmq_encoder.output_dim

        # Heads
        policy_hidden = cfg.get("policy_hidden_dims", [128, 64])
        value_hidden = cfg.get("value_hidden_dims", [128, 64])
        self.policy_net = _build_head(emb_dim, policy_hidden)
        self.value_net = _build_head(emb_dim, value_hidden)

        # Policy output size
        self.num_discrete_actions = cfg.get("num_discrete_actions", 7)
        if self.action_mode == "discrete_adjustment":
            policy_out_dim = NUM_STANDARD_PHASES * self.num_discrete_actions
        elif self.action_mode == "cycle_level_continuous":
            # Dirichlet concentration: one parameter per standard phase.
            policy_out_dim = NUM_STANDARD_PHASES
        else:
            # ratio mode: Gaussian (mean + log_std) over NUM_STANDARD_PHASES.
            policy_out_dim = 2 * self.action_dim

        self.policy_out = nn.Linear(policy_hidden[-1], policy_out_dim)
        self.value_out = nn.Linear(value_hidden[-1], 1)

        self._features = None
        self._value = None
        _init_model_weights(self.policy_net, self.value_net, self.policy_out, self.value_out)

    @staticmethod
    def _resolve_obs_dim(obs_space) -> int:
        if hasattr(obs_space, "original_space"):
            orig = obs_space.original_space
            if hasattr(orig, "spaces") and "features" in orig.spaces:
                return int(np.prod(orig.spaces["features"].shape))
        if hasattr(obs_space, "spaces") and "features" in obs_space.spaces:
            return int(np.prod(obs_space.spaces["features"].shape))
        return int(np.prod(obs_space.shape))

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_flat = input_dict["obs_flat"].float()

        # Extract action mask (alphabetically first in flattened Dict)
        if self.use_masked_softmax:
            action_mask = obs_flat[..., :NUM_STANDARD_PHASES]
            obs = obs_flat[..., NUM_STANDARD_PHASES:]
            self._last_action_mask = action_mask
        else:
            obs = obs_flat
            self._last_action_mask = None

        joint_emb, _, _ = self.mgmq_encoder(obs)

        self._features = joint_emb

        # Policy
        policy_features = self.policy_net(joint_emb)
        policy_out = self.policy_out(policy_features)

        if self.action_mode == "discrete_adjustment":
            # Apply mask: for invalid phases, force logits to strongly prefer "keep" (middle action)
            if self._last_action_mask is not None:
                mask = self._last_action_mask  # [B, 8]
                reshaped = policy_out.view(-1, NUM_STANDARD_PHASES, self.num_discrete_actions)
                # For invalid phases (mask=0), set huge negative for all actions except "keep"
                keep_idx = self.num_discrete_actions // 2  # middle action = keep
                invalid_mask = (1.0 - mask).unsqueeze(-1)  # [B, 8, 1]
                bias = torch.full_like(reshaped, -1e9) * invalid_mask
                bias[..., keep_idx] = 0.0  # "keep" action stays at 0
                reshaped = reshaped + bias
                policy_out = reshaped.view(-1, NUM_STANDARD_PHASES * self.num_discrete_actions)
        elif self.action_mode == "cycle_level_continuous":
            # Dirichlet: policy_out is the raw concentration logits [B, 8].
            # Masking is handled inside TorchMaskedDirichlet, which reads
            # self._last_action_mask set above.
            pass
        else:
            logits = policy_out[..., :self.action_dim]
            log_std = torch.clamp(policy_out[..., self.action_dim:], SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            if self._last_action_mask is not None:
                mask = self._last_action_mask  # [B, 8]
                logits = logits * mask
                log_std = log_std * mask
            policy_out = torch.cat([logits, log_std], dim=-1)

        # Value
        vi = self._apply_vf_isolation(joint_emb)
        self._value = self.value_out(self.value_net(vi)).squeeze(-1)

        return policy_out, state

    def _apply_vf_isolation(self, emb: torch.Tensor) -> torch.Tensor:
        if self.vf_share_coeff == 0.0:
            return emb.detach()
        if self.vf_share_coeff == 1.0:
            return emb
        return self.vf_share_coeff * emb + (1.0 - self.vf_share_coeff) * emb.detach()

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "Call forward() first"
        return self._value


# ---------------------------------------------------------------------------
# LocalMGMQTorchModel  (RLlib wrapper – local/star-graph)
# ---------------------------------------------------------------------------

class LocalMGMQTorchModel(TorchModelV2, nn.Module):
    """
    RLlib wrapper for LocalMGMQEncoder with Dict observation space.

    Designed for use with NeighborTemporalObservationFunction, which provides
    pre-packaged observations with neighbor features.

    Observation space expected:
        Dict({
            "self_features":      Box[T, feature_dim],
            "neighbor_features":  Box[K, T, feature_dim],
            "neighbor_mask":      Box[K],
            "neighbor_directions":Box[K],
            "action_mask":        Box[8],
        })
    Registered as "local_mgmq_model" in ModelCatalog.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})

        # Extract dimensions from Dict observation space
        if hasattr(obs_space, "spaces"):
            self_shape = obs_space.spaces["self_features"].shape
            neighbor_shape = obs_space.spaces["neighbor_features"].shape
        elif hasattr(obs_space, "original_space"):
            orig = obs_space.original_space
            self_shape = orig.spaces["self_features"].shape
            neighbor_shape = orig.spaces["neighbor_features"].shape
        else:
            feature_dim = custom_config.get("obs_dim", 48)
            K = custom_config.get("max_neighbors", 4)
            self_shape = (feature_dim,)
            neighbor_shape = (K, feature_dim)

        feature_dim = self_shape[0] if len(self_shape) == 1 else self_shape[-1]
        K = neighbor_shape[0]

        gat_hidden_dim = custom_config.get("gat_hidden_dim", 64)
        gat_output_dim = custom_config.get("gat_output_dim", 32)
        gat_num_heads = custom_config.get("gat_num_heads", 4)
        graphsage_hidden_dim = custom_config.get("graphsage_hidden_dim", 64)
        gru_hidden_dim = custom_config.get("gru_hidden_dim", 32)
        policy_hidden_dims = custom_config.get("policy_hidden_dims", [128, 64])
        value_hidden_dims = custom_config.get("value_hidden_dims", [128, 64])
        dropout = custom_config.get("dropout", 0.3)

        self.use_masked_softmax = custom_config.get("use_masked_softmax", True)
        self.action_dim = int(np.prod(action_space.shape))
        self.vf_share_coeff = custom_config.get("vf_share_coeff", 1.0)
        self._last_action_mask = None
        self.action_mode = custom_config.get("action_mode", "ratio")
        self.num_discrete_actions = custom_config.get("num_discrete_actions", 7)

        # LocalMGMQEncoder uses lane features [48], not full 56-dim obs
        self.mgmq_encoder = LocalMGMQEncoder(
            obs_dim=feature_dim,
            max_neighbors=K,
            gat_hidden_dim=gat_hidden_dim,
            gat_output_dim=gat_output_dim,
            gat_num_heads=gat_num_heads,
            graphsage_hidden_dim=graphsage_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            dropout=dropout,
        )

        joint_emb_dim = self.mgmq_encoder.output_dim

        # Policy head
        policy_layers: list = []
        prev_dim = joint_emb_dim
        for h in policy_hidden_dims:
            policy_layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU()])
            prev_dim = h
        self.policy_net = nn.Sequential(*policy_layers)

        if self.action_mode == "discrete_adjustment":
            self.policy_out = nn.Linear(prev_dim, NUM_STANDARD_PHASES * self.num_discrete_actions)
        elif self.action_mode == "cycle_level_continuous":
            # Dirichlet concentration parameters, one per standard phase.
            self.policy_out = nn.Linear(prev_dim, NUM_STANDARD_PHASES)
        else:
            self.policy_out = nn.Linear(prev_dim, 2 * self.action_dim)

        # Value head
        value_layers: list = []
        prev_dim = joint_emb_dim
        for h in value_hidden_dims:
            value_layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU()])
            prev_dim = h
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)

        self._features = None
        self._value = None
        _init_model_weights(self.policy_net, self.value_net, self.policy_out, self.value_out)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"]

        # Build obs_dict for LocalMGMQEncoder
        if isinstance(obs, dict):
            obs_dict = {
                "self_features": obs["self_features"].float(),
                "neighbor_features": obs["neighbor_features"].float(),
                "neighbor_mask": obs["neighbor_mask"].float(),
            }
            if "neighbor_directions" in obs:
                obs_dict["neighbor_directions"] = obs["neighbor_directions"].float()
            if self.use_masked_softmax and "action_mask" in obs:
                self._last_action_mask = obs["action_mask"].float()
            else:
                self._last_action_mask = None
        else:
            raise ValueError("LocalMGMQTorchModel expects Dict observation. "
                             "Ensure use_neighbor_obs=True in env_config.")

        # Handle T-dimension: self_features may be [B, T, 48]; take last step
        sf = obs_dict["self_features"]
        if sf.dim() == 3:
            obs_dict["self_features"] = sf[:, -1, :]
        nf = obs_dict["neighbor_features"]
        if nf.dim() == 4:
            obs_dict["neighbor_features"] = nf[:, :, -1, :]

        joint_emb = self.mgmq_encoder(obs_dict)
        self._features = joint_emb

        # Policy
        policy_features = self.policy_net(joint_emb)
        policy_out = self.policy_out(policy_features)

        if self.action_mode == "discrete_adjustment":
            if self._last_action_mask is not None:
                mask = self._last_action_mask
                reshaped = policy_out.view(-1, NUM_STANDARD_PHASES, self.num_discrete_actions)
                keep_idx = self.num_discrete_actions // 2
                invalid_mask = (1.0 - mask).unsqueeze(-1)
                bias = torch.full_like(reshaped, -1e9) * invalid_mask
                bias[..., keep_idx] = 0.0
                policy_out = (reshaped + bias).view(-1, NUM_STANDARD_PHASES * self.num_discrete_actions)
        elif self.action_mode == "cycle_level_continuous":
            # Raw concentration logits; masking happens inside the Dirichlet
            # distribution via self._last_action_mask.
            pass
        else:
            logits = policy_out[..., :self.action_dim]
            log_std = torch.clamp(policy_out[..., self.action_dim:], SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            if self._last_action_mask is not None:
                logits = logits * self._last_action_mask
                log_std = log_std * self._last_action_mask
            policy_out = torch.cat([logits, log_std], dim=-1)

        # Value
        if self.vf_share_coeff == 0.0:
            vi = joint_emb.detach()
        elif self.vf_share_coeff == 1.0:
            vi = joint_emb
        else:
            vi = self.vf_share_coeff * joint_emb + (1.0 - self.vf_share_coeff) * joint_emb.detach()
        self._value = self.value_out(self.value_net(vi)).squeeze(-1)

        return policy_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value is not None, "Must call forward() first"
        return self._value
