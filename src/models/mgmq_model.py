"""
MGMQ Model: Multi-Layer graph masking Q-Learning for PPO.

This module implements the complete MGMQ architecture as a custom model
for RLlib PPO algorithm. It combines:
1. GAT (Graph Attention Network) for intersection embedding
2. GraphSAGE + Bi-GRU for network embedding  
3. Joint embedding for policy and value networks

The model is designed for continuous action spaces in traffic signal control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from .gat_layer import GATLayer, MultiHeadGATLayer, DualStreamGATLayer, get_lane_conflict_matrix, get_lane_cooperation_matrix
from .graphsage_bigru import GraphSAGE_BiGRU, NeighborGraphSAGE_BiGRU


# Log std bounds to prevent entropy explosion
LOG_STD_MIN = -20.0  # Minimum log_std (very deterministic)
LOG_STD_MAX = 0.5    # FIXED: Reduced from 2.0 to 0.5 for faster entropy convergence

# Number of standard phases for action masking
NUM_STANDARD_PHASES = 8
                     # std = e^0.5 ≈ 1.65, which is reasonable for normalized actions

# Softmax temperature for action output (lower = more deterministic)
SOFTMAX_TEMPERATURE = 1.0

# For Softmax-based output, we use smaller std bounds since actions are already normalized [0,1]
SOFTMAX_LOG_STD_MIN = -5.0   # std = e^-5 ≈ 0.007 (very deterministic)
SOFTMAX_LOG_STD_MAX = -1.0   # std = e^-1 ≈ 0.37 (moderate exploration for normalized output)


def build_network_adjacency(
    ts_ids: list,
    net_file: str,
    directional: bool = True
) -> torch.Tensor:
    """
    Build adjacency matrix for the traffic network (controlled intersections only).
    
    This function uses the SUMO network file (.net.xml) to determine connectivity.
    Two controlled intersections are considered neighbors if they are connected
    directly or via a path of non-controlled junctions.
    
    Args:
        ts_ids: List of traffic signal IDs (controlled intersections only)
        net_file: Path to SUMO .net.xml file to parse connectivity
        directional: If True, return directional adjacency [4, N, N]
                     If False, return simple adjacency [N, N]
            
    Returns:
        If directional=True: Adjacency tensor of shape [4, N, N] where:
            Channel 0: North neighbors
            Channel 1: East neighbors  
            Channel 2: South neighbors
            Channel 3: West neighbors
        If directional=False: Adjacency matrix of shape [N, N]
    """
    import math
    
    N = len(ts_ids)
    ts_set = set(ts_ids)
    ts_to_idx = {ts: i for i, ts in enumerate(ts_ids)}
    
    if directional:
        adj = torch.zeros(4, N, N)  # [4, N, N] for 4 directions
    else:
        adj = torch.eye(N)  # Self-connections for non-directional
    
    if not net_file:
        print("Warning: net_file not provided. Returning identity/zero adjacency matrix.")
        return adj

    try:
        import xml.etree.ElementTree as ET
        from collections import defaultdict
        
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Get junction coordinates
        junction_coords = {}
        for junction in root.findall('junction'):
            junc_id = junction.get('id')
            x = float(junction.get('x', 0))
            y = float(junction.get('y', 0))
            junction_coords[junc_id] = (x, y)
        
        # Build graph of all junctions (controlled and non-controlled)
        graph = defaultdict(set)
        
        for edge in root.findall('.//edge'):
            # Skip internal edges
            if edge.get('id', '').startswith(':'):
                continue
                
            from_junction = edge.get('from')
            to_junction = edge.get('to')
            
            if from_junction and to_junction:
                graph[from_junction].add(to_junction)
                graph[to_junction].add(from_junction)  # Undirected
        
        def find_controlled_neighbors(start_ts: str) -> set:
            """BFS to find controlled intersections reachable without passing through other controlled ones."""
            neighbors = set()
            visited = {start_ts}
            queue = list(graph[start_ts])
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                if current in ts_set:
                    neighbors.add(current)
                else:
                    for next_junction in graph[current]:
                        if next_junction not in visited:
                            queue.append(next_junction)
            
            return neighbors
        
        def get_direction_index(from_id: str, to_id: str) -> int:
            """Get direction index (0=N, 1=E, 2=S, 3=W) from source to target."""
            if from_id not in junction_coords or to_id not in junction_coords:
                return -1
            
            x1, y1 = junction_coords[from_id]
            x2, y2 = junction_coords[to_id]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return -1
            
            angle = math.degrees(math.atan2(dy, dx))
            # Normalize to [0, 360)
            angle = angle % 360
            
            # Direction classification:
            # North: 45 to 135 degrees
            # West: 135 to 225 degrees
            # South: 225 to 315 degrees
            # East: 315 to 360 or 0 to 45 degrees
            if 45 <= angle < 135:
                return 0  # North
            elif 135 <= angle < 225:
                return 3  # West
            elif 225 <= angle < 315:
                return 2  # South
            else:
                return 1  # East
        
        # Build adjacency matrix
        for ts_id in ts_ids:
            controlled_neighbors = find_controlled_neighbors(ts_id)
            i = ts_to_idx[ts_id]
            
            for neighbor in controlled_neighbors:
                if neighbor in ts_to_idx:
                    j = ts_to_idx[neighbor]
                    
                    if directional:
                        # Get direction from ts_id to neighbor
                        dir_idx = get_direction_index(ts_id, neighbor)
                        if dir_idx >= 0:
                            adj[dir_idx, i, j] = 1.0
                    else:
                        adj[i, j] = 1
                        adj[j, i] = 1
                
        return adj
    except Exception as e:
        print(f"Error parsing net file '{net_file}': {e}")
        return adj


class MGMQEncoder(nn.Module):
    """
    MGMQ Encoder: GAT + GraphSAGE_BiGRU for feature extraction.
    
    This encoder follows the MGMQ architecture diagram:
    State -> GAT (Intersection Embedding) -> GraphSAGE_BiGRU (Network Embedding)
           -> Joint Embedding (concatenation)
    
    Args:
        obs_dim: Observation dimension per agent
        num_agents: Number of traffic signals/agents
        gat_hidden_dim: Hidden dimension for GAT
        gat_output_dim: Output dimension for GAT
        gat_num_heads: Number of GAT attention heads
        graphsage_hidden_dim: Hidden dimension for GraphSAGE
        gru_hidden_dim: Hidden dimension for Bi-GRU
        dropout: Dropout rate
        network_adjacency: Pre-computed directional adjacency matrix [4, N, N]
                          or simple adjacency [N, N] (will be expanded)
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_agents: int = 1,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 32,
        gat_num_heads: int = 4,
        graphsage_hidden_dim: int = 64,
        gru_hidden_dim: int = 32,
        dropout: float = 0.3,
        network_adjacency: Optional[torch.Tensor] = None,
    ):
        super(MGMQEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.gat_output_dim = gat_output_dim
        self.gat_num_heads = gat_num_heads
        
        # Assume features are organized by lane (12 lanes)
        self.num_lanes = 12
        if obs_dim % self.num_lanes == 0:
            self.lane_feature_dim = obs_dim // self.num_lanes
        else:
            print(f"Warning: Feature dim {obs_dim} not divisible by 12 lanes. Using projection.")
            self.lane_feature_dim = gat_hidden_dim
            
        # Layer 1: Dual-Stream GAT for intersection embedding (Lane-level)
        self.dual_stream_gat = DualStreamGATLayer(
            in_features=self.lane_feature_dim,
            hidden_dim=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.2
        )
        
        # Layer 2: GraphSAGE + Bi-GRU for network embedding
        # GAT outputs [12, gat_output_dim * gat_num_heads] per intersection
        # After Flatten: 12 * gat_output_dim * gat_num_heads
        gat_per_lane_output = gat_output_dim * gat_num_heads  # e.g., 32*4 = 128
        gat_total_output = self.num_lanes * gat_per_lane_output  # 12 * 128 = 1536
        
        self.graphsage_bigru = GraphSAGE_BiGRU(
            in_features=gat_total_output,
            hidden_features=graphsage_hidden_dim,
            gru_hidden_size=gru_hidden_dim,
            dropout=dropout
        )
        
        # Store network adjacency matrix (directional: [4, N, N] or simple: [N, N])
        if network_adjacency is not None:
            # If simple adjacency [N, N], expand to directional [4, N, N]
            if network_adjacency.dim() == 2:
                # Expand simple adjacency to all 4 directions
                N = network_adjacency.size(0)
                network_adjacency_4d = network_adjacency.unsqueeze(0).expand(4, -1, -1).clone()
                self.register_buffer('network_adj', network_adjacency_4d)
            else:
                self.register_buffer('network_adj', network_adjacency)
        else:
            # Default: fully connected for all directions
            N = max(1, num_agents)
            default_adj = torch.ones(4, N, N)  # [4, N, N]
            self.register_buffer('network_adj', default_adj)
            
        # Store Lane adjacency matrices separately (Static 12x12)
        lane_adj_coop = get_lane_cooperation_matrix()
        lane_adj_conf = get_lane_conflict_matrix()
        
        self.register_buffer('lane_adj_coop', lane_adj_coop)
        self.register_buffer('lane_adj_conf', lane_adj_conf)
        
        # Calculate output dimensions
        self.intersection_emb_dim = gat_total_output
        self.network_emb_dim = graphsage_hidden_dim
        self.joint_emb_dim = self.intersection_emb_dim + self.network_emb_dim
        
    @property
    def output_dim(self) -> int:
        """Return the joint embedding dimension."""
        return self.joint_emb_dim
    
    def set_adjacency_matrix(self, adj: torch.Tensor):
        """Update the network adjacency matrix."""
        self.network_adj = adj.to(self.network_adj.device)
        
    def forward(
        self, 
        obs: torch.Tensor,
        agent_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of MGMQ encoder.
        
        Args:
            obs: Observations tensor
                - Single agent: [batch, obs_dim]
                - Multi-agent: [batch, num_agents, obs_dim]
            agent_idx: Index of the agent (for single-agent mode)
            
        Returns:
            Tuple of (joint_embedding, intersection_embedding, network_embedding)
        """
        batch_size = obs.size(0)
        
        # Handle single vs multi-agent observations
        if obs.dim() == 2:
            # Single agent observation: [batch, obs_dim]
            obs = obs.unsqueeze(1)
            num_agents = 1
        else:
            # Multi-agent: [batch, num_agents, obs_dim]
            num_agents = obs.size(1)
            
        # --- Layer 1: Lane-level GAT (Intersection Embedding) ---
        
        # Flatten: [batch * num_agents, obs_dim]
        obs_flat = obs.reshape(-1, self.obs_dim)
        
        # Reshape to 12 lanes: [batch * num_agents, 12, lane_feature_dim]
        if self.obs_dim % 12 == 0:
            lane_features = obs_flat.view(-1, 12, self.obs_dim // 12)
        else:
            raise ValueError(f"obs_dim {self.obs_dim} must be divisible by 12 for GAT.")

        # Expand adj matrices
        lane_adj_coop_batch = self.lane_adj_coop.unsqueeze(0).expand(lane_features.size(0), -1, -1)
        lane_adj_conf_batch = self.lane_adj_conf.unsqueeze(0).expand(lane_features.size(0), -1, -1)
        
        # Run Dual-Stream GAT
        # gat_out: [batch * num_agents, 12, gat_output_dim * heads]
        gat_out = self.dual_stream_gat(lane_features, lane_adj_coop_batch, lane_adj_conf_batch)
        
        # FLATTEN over lanes to preserve spatial semantics (NOT Mean Pooling!)
        # Mean Pooling loses positional information (which lane is congested)
        # Flatten preserves: index 0-127 = Lane 0, index 128-255 = Lane 1, etc.
        # [batch * num_agents, 12 * gat_output_dim * heads] = [batch * num_agents, 1536]
        intersection_emb_flat = gat_out.flatten(start_dim=1)
        
        # Reshape back to [batch, num_agents, emb_dim]
        intersection_emb = intersection_emb_flat.view(batch_size, num_agents, -1)
        
        # --- Layer 2: Network-level GraphSAGE (Network Embedding) ---
        
        # Get network adjacency (directional: [4, N, N])
        if num_agents == 1:
            net_adj = torch.ones(4, 1, 1, device=obs.device)
        else:
            net_adj = self.network_adj[:, :num_agents, :num_agents]
        
        # GraphSAGE: Input [batch, N, features], adj [4, N, N]
        # Returns [batch, num_agents, hidden_features]
        network_emb_seq = self.graphsage_bigru(intersection_emb, net_adj)
        # Mean pooling over agents for network embedding
        network_emb = network_emb_seq.mean(dim=1)
        
        # Select intersection embedding for specific agent or use mean
        if agent_idx is not None and num_agents > 1:
            agent_intersection_emb = intersection_emb[:, agent_idx, :]
        else:
            agent_intersection_emb = intersection_emb.mean(dim=1)
        
        # Joint embedding: concatenate intersection and network embeddings
        joint_emb = torch.cat([agent_intersection_emb, network_emb], dim=-1)
        
        return joint_emb, agent_intersection_emb, network_emb


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
        obs_dim: int = 48,
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
        self.num_lanes = 12
        self.lane_feature_dim = obs_dim // self.num_lanes  # 48/12 = 4
        
        # Dual-Stream GAT
        self.dual_stream_gat = DualStreamGATLayer(
            in_features=self.lane_feature_dim,
            hidden_dim=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.2
        )
        
        # Static lane adjacency matrices
        self.register_buffer('lane_adj_coop', get_lane_cooperation_matrix())
        self.register_buffer('lane_adj_conf', get_lane_conflict_matrix())
        
        # GAT output dimension (after Flatten over 12 lanes)
        # Each lane outputs gat_output_dim * gat_num_heads features
        # Flatten: 12 * gat_output_dim * gat_num_heads = 12 * 128 = 1536
        self.gat_per_lane_output = gat_output_dim * gat_num_heads  # 128
        self.gat_total_output = self.num_lanes * self.gat_per_lane_output  # 1536
        
        # NeighborGraphSAGE_BiGRU for SPATIAL aggregation
        self.neighbor_aggregator = NeighborGraphSAGE_BiGRU(
            in_features=self.gat_total_output,
            hidden_features=graphsage_hidden_dim,
            gru_hidden_size=gru_hidden_dim,
            max_neighbors=max_neighbors,
            dropout=dropout
        )
        
        # Output dimensions
        self.intersection_emb_dim = self.gat_total_output
        self.network_emb_dim = graphsage_hidden_dim
        self.joint_emb_dim = self.intersection_emb_dim + self.network_emb_dim
        
    @property
    def output_dim(self) -> int:
        return self.joint_emb_dim
        
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for Dict observation.
        
        Args:
            obs_dict: Dict with keys:
                - self_features: [B, 48]
                - neighbor_features: [B, K, 48]
                - neighbor_mask: [B, K]
                
        Returns:
            joint_emb: [B, joint_emb_dim]
        """
        self_feat = obs_dict["self_features"]         # [B, 48]
        neighbor_feat = obs_dict["neighbor_features"] # [B, K, 48]
        mask = obs_dict["neighbor_mask"]              # [B, K]
        
        B = self_feat.size(0)
        K = neighbor_feat.size(1)
        
        # 1. GAT for self features
        self_emb = self._run_gat(self_feat)
        
        # 2. GAT for neighbor features
        neighbor_feat_flat = neighbor_feat.reshape(B * K, -1)
        neighbor_emb_flat = self._run_gat(neighbor_feat_flat)
        neighbor_emb = neighbor_emb_flat.reshape(B, K, -1)
        
        # 3. Spatial Neighbor Aggregation using BiGRU
        network_emb = self.neighbor_aggregator(
            self_features=self_emb,
            neighbor_features=neighbor_emb,
            neighbor_mask=mask
        )
        
        # 4. Joint Embedding
        joint_emb = torch.cat([self_emb, network_emb], dim=-1)
        
        return joint_emb
        
    def _run_gat(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GAT to features and FLATTEN to preserve spatial semantics.
        
        IMPORTANT: We use Flatten instead of Mean Pooling to preserve
        positional information. This allows the network to learn which
        specific lanes (e.g., North-Left at index 0-127) are congested.
        
        Args:
            x: [batch, 48] raw observation (12 lanes * 4 features)
            
        Returns:
            [batch, 1536] flattened GAT embedding (12 lanes * 128 features)
        """
        batch_size = x.size(0)
        lane_feat = x.view(batch_size, self.num_lanes, self.lane_feature_dim)
        
        adj_coop = self.lane_adj_coop.unsqueeze(0).expand(batch_size, -1, -1)
        adj_conf = self.lane_adj_conf.unsqueeze(0).expand(batch_size, -1, -1)
        
        # gat_out: [batch, 12, gat_per_lane_output]
        gat_out = self.dual_stream_gat(lane_feat, adj_coop, adj_conf)
        
        # FLATTEN instead of Mean Pooling to preserve spatial semantics
        # [batch, 12 * gat_per_lane_output] = [batch, 1536]
        gat_flattened = gat_out.flatten(start_dim=1)
        
        return gat_flattened


"""
Kiến trúc ban đầu, sử dụng đồ thị toàn cục để tính toán.
Chứa logic cốt lõi của GNN toàn cục. Nó lấy thông tin của tất cả các ngã tư cùng một lúc và 
dùng một ma trận kề (Adjacency matrix) khổng lồ để lan truyền thôg tin.
"""
class MGMQModel(nn.Module):
    """
    Complete MGMQ Model for PPO with Actor-Critic architecture.
    
    This model outputs both:
    - Policy (actor): Action distribution parameters
    - Value (critic): State value estimate
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension (continuous)
        num_agents: Number of agents
        gat_hidden_dim: GAT hidden dimension
        gat_output_dim: GAT output dimension per head
        gat_num_heads: Number of GAT attention heads
        graphsage_hidden_dim: GraphSAGE hidden dimension
        gru_hidden_dim: Bi-GRU hidden dimension
        policy_hidden_dims: Hidden dimensions for policy network
        value_hidden_dims: Hidden dimensions for value network
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int = 1,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 32,
        gat_num_heads: int = 4,
        graphsage_hidden_dim: int = 64,
        gru_hidden_dim: int = 32,
        policy_hidden_dims: List[int] = [128, 64],
        value_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
        adjacency_matrix: Optional[torch.Tensor] = None
    ):
        super(MGMQModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # MGMQ Encoder (shared between policy and value)
        self.encoder = MGMQEncoder(
            obs_dim=obs_dim,
            num_agents=num_agents,
            gat_hidden_dim=gat_hidden_dim,
            gat_output_dim=gat_output_dim,
            gat_num_heads=gat_num_heads,
            graphsage_hidden_dim=graphsage_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            dropout=dropout,
            network_adjacency=adjacency_matrix
        )
        
        joint_emb_dim = self.encoder.output_dim
        
        # Policy network (actor) - NO Dropout in RL!
        policy_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Policy output: mean for continuous actions
        self.policy_mean = nn.Linear(prev_dim, action_dim)
        # Log std as learnable parameter
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value network (critic) - NO Dropout in RL!
        value_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in value_hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights.
        
        NOTE: Value output uses smaller gain (0.1) to:
        - Start predictions near 0
        - Prevent large initial vf_loss
        - Help vf_explained_var stay positive
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller initialization for policy output (stable actions)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        # CRITICAL: Small gain for value output to prevent large initial predictions
        nn.init.orthogonal_(self.value_out.weight, gain=0.1)
        nn.init.zeros_(self.value_out.bias)
        
    def forward(
        self, 
        obs: torch.Tensor,
        agent_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observations [batch, obs_dim] or [batch, num_agents, obs_dim]
            agent_idx: Agent index for multi-agent
            
        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        # Get joint embedding from encoder
        joint_emb, _, _ = self.encoder(obs, agent_idx)
        
        # Policy network
        policy_features = self.policy_net(joint_emb)
        action_mean = self.policy_mean(policy_features)
        action_log_std = self.policy_log_std.expand_as(action_mean)
        
        # Value network
        value_features = self.value_net(joint_emb)
        value = self.value_out(value_features)
        
        return action_mean, action_log_std, value
    
    def get_action(
        self, 
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observations
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        action_mean, action_log_std, _ = self.forward(obs)
        
        if deterministic:
            return action_mean, torch.zeros(action_mean.size(0), device=obs.device)
        
        # Sample from Gaussian
        std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.rsample()  # Reparameterization trick
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (value, log_prob, entropy)
        """
        action_mean, action_log_std, value = self.forward(obs)
        
        std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, std)
        
        log_prob = normal.log_prob(actions).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return value.squeeze(-1), log_prob, entropy


"""
Là môt wrapper để mô hình này có thể chạy được với Thư viện RLlib,
Hạn chế; Khi RLlib chia nhỏ dữ liệu để huấn luyện(Batching), cấu trúc đồ thị toàn cục
bị phá vỡ, dẫn đến mô hình khó học được quan hệ giữa các nút giao hàng xóm.
"""
class MGMQTorchModel(TorchModelV2, nn.Module):
    """
    MGMQ Model wrapper for RLlib integration.
    
    This class wraps the MGMQModel to be compatible with RLlib's
    model API for use with PPO and other algorithms.
    
    Supports 3 action distribution modes:
    1. use_masked_softmax=True (RECOMMENDED): Masked Softmax + Gaussian
       - Action mask applied BEFORE softmax
       - Invalid phases get exactly 0.0
       - Gradient flows only through valid phases
    2. use_dirichlet=True: Dirichlet distribution (legacy)
    3. use_softmax_output=True: Softmax + Gaussian (no masking)
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
        
        # Get custom model config
        custom_config = model_config.get("custom_model_config", {})
        
        # Handle Dict observation space (with action_mask)
        # obs_space can be Dict or flattened version
        if hasattr(obs_space, 'original_space'):
            # Flattened Dict space - get original for dimensions
            orig_space = obs_space.original_space
            if hasattr(orig_space, 'spaces') and 'features' in orig_space.spaces:
                obs_dim = int(np.prod(orig_space.spaces['features'].shape))
            else:
                obs_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, 'spaces') and 'features' in obs_space.spaces:
            # Direct Dict space
            obs_dim = int(np.prod(obs_space.spaces['features'].shape))
        else:
            # Box space (legacy)
            obs_dim = int(np.prod(obs_space.shape))
            
        action_dim = int(np.prod(action_space.shape))
        
        # Model hyperparameters
        num_agents = custom_config.get("num_agents", 1)
        gat_hidden_dim = custom_config.get("gat_hidden_dim", 64)
        gat_output_dim = custom_config.get("gat_output_dim", 32)
        gat_num_heads = custom_config.get("gat_num_heads", 4)
        graphsage_hidden_dim = custom_config.get("graphsage_hidden_dim", 64)
        gru_hidden_dim = custom_config.get("gru_hidden_dim", 32)
        policy_hidden_dims = custom_config.get("policy_hidden_dims", [128, 64])
        value_hidden_dims = custom_config.get("value_hidden_dims", [128, 64])
        dropout = custom_config.get("dropout", 0.3)
        
        # === ACTION DISTRIBUTION OPTIONS ===
        # Option 1: use_masked_softmax=True (RECOMMENDED - NEW)
        #   - Model outputs logits + log_std → MaskedSoftmax distribution
        #   - Actions sampled via: logits + noise → apply mask → softmax
        #   - Invalid phases get exactly 0.0, gradient only for valid phases
        #   - Proper entropy calculation over valid phases only
        #
        # Option 2: use_dirichlet=True (LEGACY - has post-hoc masking issues)
        #   - Model outputs raw logits → Dirichlet concentration params
        #   - Actions automatically sum=1, all positive
        #   - BUT: masking applied post-hoc → gradient waste for invalid phases
        #
        # Option 3: use_softmax_output=True, use_dirichlet=False (LEGACY)
        #   - Model outputs Softmax(logits) as mean for Gaussian
        #   - Still has issues: Gaussian sampling can break sum=1
        #
        # Option 4: Both False (NOT RECOMMENDED - gradient death)
        self.use_masked_softmax = custom_config.get("use_masked_softmax", True)  # Default: NEW mode
        self.use_dirichlet = custom_config.get("use_dirichlet", False)  # Legacy fallback
        self.use_softmax_output = custom_config.get("use_softmax_output", False)  # Legacy fallback
        self.softmax_temperature = custom_config.get("softmax_temperature", SOFTMAX_TEMPERATURE)
        self.action_dim = int(np.prod(action_space.shape))  # Store for forward()
        
        # Store action_mask for MaskedSoftmax distribution to read
        self._last_action_mask = None
        
        # Build directional adjacency matrix if ts_ids provided
        ts_ids = custom_config.get("ts_ids", None)
        net_file = custom_config.get("net_file", None)
        if ts_ids is not None:
            # Build directional adjacency [4, N, N] for proper neighbor direction encoding
            network_adjacency = build_network_adjacency(ts_ids, net_file=net_file, directional=True)
        else:
            network_adjacency = None
        
        # Create MGMQ encoder
        self.mgmq_encoder = MGMQEncoder(
            obs_dim=obs_dim,
            num_agents=num_agents,
            gat_hidden_dim=gat_hidden_dim,
            gat_output_dim=gat_output_dim,
            gat_num_heads=gat_num_heads,
            graphsage_hidden_dim=graphsage_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            dropout=dropout,
            network_adjacency=network_adjacency
        )
        
        joint_emb_dim = self.mgmq_encoder.output_dim
        
        # Policy network
        # NOTE: No Dropout in RL policy/value networks!
        policy_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Output layer for policy
        # MaskedSoftmax: num_outputs = 2 * action_dim (logits + log_std)
        # Dirichlet: num_outputs = action_dim (concentration params only)
        # Gaussian: num_outputs = 2 * action_dim (mean + log_std)
        if self.use_masked_softmax:
            self.policy_out = nn.Linear(prev_dim, 2 * self.action_dim)  # logits + log_std
        elif self.use_dirichlet:
            self.policy_out = nn.Linear(prev_dim, self.action_dim)
        else:
            self.policy_out = nn.Linear(prev_dim, num_outputs)
        
        # Value network - NO Dropout!
        # Dropout in VF causes inconsistent value predictions,
        # leading to near-zero vf_explained_var and very high vf_loss.
        value_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in value_hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)
        
        # Store last features for value function
        self._features = None
        self._value = None
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights.
        
        NOTE: Value output uses smaller gain (0.1) to:
        - Start predictions near 0
        - Prevent large initial vf_loss  
        - Help vf_explained_var stay positive
        """
        for module in [self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        
        # Policy output: small gain for stable initial actions
        nn.init.orthogonal_(self.policy_out.weight, gain=0.01)
        nn.init.zeros_(self.policy_out.bias)
        
        # CRITICAL: Small gain for value output to prevent large initial predictions
        nn.init.orthogonal_(self.value_out.weight, gain=0.1)
        nn.init.zeros_(self.value_out.bias)
        
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass for RLlib.
        
        Args:
            input_dict: Dictionary with 'obs' key
            state: RNN state (unused)
            seq_lens: Sequence lengths (unused)
            
        Returns:
            Tuple of (policy_output, state)
            
        IMPORTANT: When use_softmax_output=True, the mean output is normalized via Softmax.
        This solves the "Scale Ambiguity & Vanishing Gradient" problem:
        - Raw logits (e.g., [2,3,5] vs [200,300,500]) produce the same action after external normalization
        - But gradients for large logits → 0 (vanishing gradient)
        - By applying Softmax INSIDE the model, gradient flows properly through the computation graph
        
        Data Flow:
        1. Raw Logits from policy_out
        2. Apply Softmax (with temperature) → mean is now in [0,1] and sums to 1
        3. Log_std is clamped to smaller range for normalized output
        
        MaskedSoftmax Mode (NEW):
        - Extract action_mask from flattened observation
        - Store in _last_action_mask for distribution to read
        - Output logits + log_std (distribution handles masking)
        """
        obs_flat = input_dict["obs_flat"].float()
        
        # === EXTRACT ACTION MASK FROM FLATTENED OBS ===
        # When obs_space is Dict({'features': [...], 'action_mask': [8]}),
        # RLlib flattens to: [features..., action_mask...]
        # The last NUM_STANDARD_PHASES (8) elements are the action_mask
        if self.use_masked_softmax:
            # Extract features (all except first 8) and action_mask (first 8)
            # CRITICAL FIX: RLlib/Gym flattens Dict keys alphabetically. 
            # "action_mask" comes BEFORE "features".
            action_mask = obs_flat[..., :NUM_STANDARD_PHASES]
            obs = obs_flat[..., NUM_STANDARD_PHASES:]
            # Store for MaskedSoftmax distribution to access
            self._last_action_mask = action_mask
        else:
            obs = obs_flat
            self._last_action_mask = None
        
        # Get joint embedding from MGMQ encoder
        joint_emb, _, _ = self.mgmq_encoder(obs)
        
        # Store for value function
        self._features = joint_emb
        
        # Policy output
        policy_features = self.policy_net(joint_emb)
        policy_out = self.policy_out(policy_features)
        
        if self.use_masked_softmax:
            # === MASKED SOFTMAX MODE (RECOMMENDED - NEW) ===
            # Output logits + log_std → MaskedSoftmax distribution handles:
            # 1. Add Gaussian noise to logits
            # 2. Apply mask: logits_masked = logits + (1-mask)*(-1e9)
            # 3. Softmax to get action probabilities
            # Invalid phases get exactly 0.0, gradient only for valid phases
            action_dim = self.action_dim
            logits = policy_out[..., :action_dim]
            log_std = policy_out[..., action_dim:]
            # Clamp log_std
            log_std = torch.clamp(log_std, SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            policy_out = torch.cat([logits, log_std], dim=-1)
        elif self.use_dirichlet:
            # === DIRICHLET MODE (LEGACY) ===
            # Output raw logits → Dirichlet distribution transforms them to concentration params
            # Actions sampled from Dirichlet automatically sum=1 and are all positive
            # BUT: masking applied post-hoc → gradient waste for invalid phases
            pass  # policy_out is already correct (action_dim outputs)
        elif self.use_softmax_output:
            # === LEGACY: Softmax + Gaussian mode ===
            # Split into mean and log_std
            action_dim = self.action_dim
            mean_logits = policy_out[..., :action_dim]
            log_std = policy_out[..., action_dim:]
            
            # Apply temperature-scaled Softmax
            mean = F.softmax(mean_logits / self.softmax_temperature, dim=-1)
            
            # For Softmax output, use tighter log_std bounds
            log_std = torch.clamp(log_std, SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            policy_out = torch.cat([mean, log_std], dim=-1)
        else:
            # Original behavior: raw mean output (NOT RECOMMENDED - gradient death)
            action_dim = self.action_dim
            mean = policy_out[..., :action_dim]
            log_std = torch.clamp(policy_out[..., action_dim:], LOG_STD_MIN, LOG_STD_MAX)
            policy_out = torch.cat([mean, log_std], dim=-1)
        
        # Compute and store value
        value_features = self.value_net(joint_emb)
        self._value = self.value_out(value_features).squeeze(-1)
        
        return policy_out, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value function output."""
        assert self._value is not None, "Must call forward() first"
        return self._value
    
    def get_encoder_output(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the joint embedding from the encoder."""
        joint_emb, _, _ = self.mgmq_encoder(obs)
        return joint_emb


class LocalMGMQTorchModel(TorchModelV2, nn.Module):
    """
    RLlib wrapper for LocalMGMQEncoder with Dict observation space.
    
    This model is designed for use with NeighborObservationFunction,
    which provides pre-packaged observations with neighbor features.
    
    Observation space expected:
        Dict({
            "self_features": Box[feature_dim],
            "neighbor_features": Box[K, feature_dim],
            "neighbor_mask": Box[K],
            "action_mask": Box[NUM_STANDARD_PHASES]  # For masked softmax
        })
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        """Initialize the local MGMQ model for RLlib."""
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        custom_config = model_config.get("custom_model_config", {})
        
        # Extract dimensions from Dict observation space
        if hasattr(obs_space, 'spaces'):
            # Dict space
            self_shape = obs_space.spaces["self_features"].shape  # (feature_dim,)
            neighbor_shape = obs_space.spaces["neighbor_features"].shape  # (K, feature_dim)
        elif hasattr(obs_space, 'original_space'):
            # Flattened Dict - get original
            orig = obs_space.original_space
            self_shape = orig.spaces["self_features"].shape
            neighbor_shape = orig.spaces["neighbor_features"].shape
        else:
            # Fallback defaults
            feature_dim = custom_config.get("obs_dim", 48)
            K = custom_config.get("max_neighbors", 4)
            self_shape = (feature_dim,)
            neighbor_shape = (K, feature_dim)
        
        feature_dim = self_shape[0] if len(self_shape) == 1 else self_shape[-1]
        K = neighbor_shape[0]
        
        # Model hyperparameters
        gat_hidden_dim = custom_config.get("gat_hidden_dim", 64)
        gat_output_dim = custom_config.get("gat_output_dim", 32)
        gat_num_heads = custom_config.get("gat_num_heads", 4)
        graphsage_hidden_dim = custom_config.get("graphsage_hidden_dim", 64)
        gru_hidden_dim = custom_config.get("gru_hidden_dim", 32)
        policy_hidden_dims = custom_config.get("policy_hidden_dims", [128, 64])
        value_hidden_dims = custom_config.get("value_hidden_dims", [128, 64])
        dropout = custom_config.get("dropout", 0.3)
        
        # Action distribution options (see MGMQTorchModel for detailed comments)
        self.use_masked_softmax = custom_config.get("use_masked_softmax", True)  # NEW default
        self.use_dirichlet = custom_config.get("use_dirichlet", False)  # Legacy
        self.use_softmax_output = custom_config.get("use_softmax_output", False)
        self.softmax_temperature = custom_config.get("softmax_temperature", SOFTMAX_TEMPERATURE)
        self.action_dim = int(np.prod(action_space.shape))
        
        # Store action_mask for MaskedSoftmax distribution to read
        self._last_action_mask = None
        
        # MGMQ Encoder
        self.mgmq_encoder = LocalMGMQEncoder(
            obs_dim=feature_dim,
            max_neighbors=K,
            gat_hidden_dim=gat_hidden_dim,
            gat_output_dim=gat_output_dim,
            gat_num_heads=gat_num_heads,
            graphsage_hidden_dim=graphsage_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            dropout=dropout
        )
        
        joint_emb_dim = self.mgmq_encoder.output_dim
        
        # Policy network
        # NOTE: No Dropout in RL policy/value networks!
        # Dropout causes different outputs for same input across forward passes,
        # which breaks PPO's value function consistency and advantage estimation.
        policy_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Output layer for policy
        # MaskedSoftmax: 2 * action_dim outputs (logits + log_std)
        # Dirichlet: action_dim outputs (concentration params only)
        # Gaussian: 2 * action_dim outputs (mean + log_std)
        if self.use_masked_softmax:
            self.policy_out = nn.Linear(prev_dim, 2 * self.action_dim)
        elif self.use_dirichlet:
            self.policy_out = nn.Linear(prev_dim, self.action_dim)
        else:
            self.policy_out = nn.Linear(prev_dim, num_outputs)
        
        # Value network - NO Dropout!
        # Dropout in VF causes inconsistent value predictions for the same state,
        # leading to near-zero or negative vf_explained_var and very high vf_loss.
        value_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in value_hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)
        
        self._features = None
        self._value = None
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights.
        
        NOTE: Value output uses smaller gain (0.1) to:
        - Start predictions near 0
        - Prevent large initial vf_loss
        - Help vf_explained_var stay positive
        """
        for module in [self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        
        # Policy output: small gain for stable initial actions
        nn.init.orthogonal_(self.policy_out.weight, gain=0.01)
        nn.init.zeros_(self.policy_out.bias)
        
        # CRITICAL: Small gain for value output to prevent large initial predictions
        nn.init.orthogonal_(self.value_out.weight, gain=0.1)
        nn.init.zeros_(self.value_out.bias)
        
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass for RLlib with Dict observation.
        
        Args:
            input_dict: Dictionary with 'obs' key containing Dict observation
            state: RNN state (unused)
            seq_lens: Sequence lengths (unused)
            
        Returns:
            Tuple of (policy_output, state)
            
        MaskedSoftmax Mode (NEW - RECOMMENDED):
        - Extract action_mask from obs dict
        - Store in _last_action_mask for distribution to read
        - Output logits + log_std (distribution handles masking)
        """
        obs = input_dict["obs"]
        
        # Convert to Dict format if needed
        if isinstance(obs, dict):
            # Already dict format
            obs_dict = {
                "self_features": obs["self_features"].float(),
                "neighbor_features": obs["neighbor_features"].float(),
                "neighbor_mask": obs["neighbor_mask"].float(),
            }
            # Extract action_mask for MaskedSoftmax distribution
            if self.use_masked_softmax and "action_mask" in obs:
                self._last_action_mask = obs["action_mask"].float()
            else:
                self._last_action_mask = None
        else:
            # Fallback: Should not happen with Dict obs space
            raise ValueError("Expected Dict observation, got tensor. Use NeighborTemporalObservationFunction.")
        
        # Get joint embedding from MGMQ encoder
        joint_emb = self.mgmq_encoder(obs_dict)
        
        # Store for value function
        self._features = joint_emb
        
        # Policy output
        policy_features = self.policy_net(joint_emb)
        policy_out = self.policy_out(policy_features)
        
        if self.use_masked_softmax:
            # === MASKED SOFTMAX MODE (RECOMMENDED - NEW) ===
            # Output logits + log_std → MaskedSoftmax distribution handles masking
            action_dim = self.action_dim
            logits = policy_out[..., :action_dim]
            log_std = policy_out[..., action_dim:]
            log_std = torch.clamp(log_std, SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            policy_out = torch.cat([logits, log_std], dim=-1)
        elif self.use_dirichlet:
            # === DIRICHLET MODE (LEGACY) ===
            # Output raw logits → Dirichlet distribution transforms them to concentration params
            pass  # policy_out is already correct (action_dim outputs)
        elif self.use_softmax_output:
            # === LEGACY: Softmax + Gaussian mode ===
            action_dim = self.action_dim
            mean_logits = policy_out[..., :action_dim]
            log_std = policy_out[..., action_dim:]
            
            mean = F.softmax(mean_logits / self.softmax_temperature, dim=-1)
            log_std = torch.clamp(log_std, SOFTMAX_LOG_STD_MIN, SOFTMAX_LOG_STD_MAX)
            policy_out = torch.cat([mean, log_std], dim=-1)
        else:
            # Original behavior: raw mean output (NOT RECOMMENDED)
            action_dim = self.action_dim
            mean = policy_out[..., :action_dim]
            log_std = torch.clamp(policy_out[..., action_dim:], LOG_STD_MIN, LOG_STD_MAX)
            policy_out = torch.cat([mean, log_std], dim=-1)
        
        # Compute and store value
        value_features = self.value_net(joint_emb)
        self._value = self.value_out(value_features).squeeze(-1)
        
        return policy_out, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value function output."""
        assert self._value is not None, "Must call forward() first"
        return self._value
    
    def get_encoder_output(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get the joint embedding from the encoder."""
        return self.mgmq_encoder(obs_dict)

