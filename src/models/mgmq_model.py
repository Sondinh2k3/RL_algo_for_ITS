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

from .gat_layer import GATLayer, MultiHeadGATLayer, get_lane_conflict_matrix, get_lane_cooperation_matrix
from .graphsage_bigru import GraphSAGE_BiGRU, TemporalGraphSAGE_BiGRU


# Log std bounds to prevent entropy explosion
LOG_STD_MIN = -20.0  # Minimum log_std (very deterministic)
LOG_STD_MAX = 0.5    # FIXED: Reduced from 2.0 to 0.5 for faster entropy convergence
                     # std = e^0.5 ≈ 1.65, which is reasonable for normalized actions


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
        obs_dim: Observation dimension per agent (flattened if temporal)
        num_agents: Number of traffic signals/agents
        gat_hidden_dim: Hidden dimension for GAT
        gat_output_dim: Output dimension for GAT
        gat_num_heads: Number of GAT attention heads
        graphsage_hidden_dim: Hidden dimension for GraphSAGE
        gru_hidden_dim: Hidden dimension for Bi-GRU
        dropout: Dropout rate
        network_adjacency: Pre-computed directional adjacency matrix [4, N, N]
                          or simple adjacency [N, N] (will be expanded)
        window_size: Size of observation history window (1 for non-temporal)
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
        window_size: int = 1
    ):
        super(MGMQEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.gat_output_dim = gat_output_dim
        self.gat_num_heads = gat_num_heads
        self.window_size = window_size
        
        # Calculate feature dimension per time step
        self.total_feature_dim = obs_dim // window_size
        
        # Assume features are organized by lane (12 lanes)
        # If not divisible by 12, we might need a different approach or projection
        self.num_lanes = 12
        if self.total_feature_dim % self.num_lanes == 0:
            self.lane_feature_dim = self.total_feature_dim // self.num_lanes
        else:
            # Fallback: Project entire observation to 12 * hidden
            self.lane_feature_dim = self.total_feature_dim # Treat as 1 node if not divisible? 
            # Actually, let's force projection to 12 nodes
            print(f"Warning: Feature dim {self.total_feature_dim} not divisible by 12 lanes. Using projection.")
            self.lane_feature_dim = gat_hidden_dim # Arbitrary, will be handled by input_proj
            
        # Input projection to match GAT input dimension
        # We project each lane's features to gat_hidden_dim
        if self.total_feature_dim % self.num_lanes == 0:
            self.lane_feature_dim = self.total_feature_dim // self.num_lanes
            self.input_proj = nn.Sequential(
                nn.Linear(self.lane_feature_dim, gat_hidden_dim),
                nn.LayerNorm(gat_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # Fallback: Project entire observation to 12 * hidden
            print(f"Warning: Feature dim {self.total_feature_dim} not divisible by {self.num_lanes} lanes. Using projection to {self.num_lanes} * {gat_hidden_dim}.")
            self.lane_feature_dim = self.total_feature_dim 
            self.input_proj = nn.Sequential(
                nn.Linear(self.total_feature_dim, self.num_lanes * gat_hidden_dim),
                nn.LayerNorm(self.num_lanes * gat_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Layer 1: Multi-head GAT for intersection embedding (Lane-level)
        # Split into Cooperation and Conflict GATs as per MGMQ paper
        self.gat_layer_coop = MultiHeadGATLayer(
            in_features=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.2,
            concat=True
        )
        
        self.gat_layer_conf = MultiHeadGATLayer(
            in_features=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.2,
            concat=True
        )
        
        # Layer 2: GraphSAGE + Bi-GRU for network embedding
        # Output dimension is doubled because we concat coop and conf outputs
        gat_total_output = (gat_output_dim * gat_num_heads) * 2
        
        if window_size > 1:
            # Use Temporal version if history is provided
            self.graphsage_bigru = TemporalGraphSAGE_BiGRU(
                in_features=gat_total_output,
                hidden_features=graphsage_hidden_dim,
                gru_hidden_size=gru_hidden_dim,
                history_length=window_size,
                dropout=dropout
            )
        else:
            # Use standard version
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
            # Reshape to [batch, 1, obs_dim]
            obs = obs.unsqueeze(1)
            num_agents = 1
        else:
            # Multi-agent: [batch, num_agents, obs_dim]
            num_agents = obs.size(1)
            
        # Reshape for temporal processing
        # obs: [batch, num_agents, window_size * total_feature_dim]
        # -> [batch, num_agents, window_size, total_feature_dim]
        obs_reshaped = obs.reshape(batch_size, num_agents, self.window_size, self.total_feature_dim)
        
        # --- Layer 1: Lane-level GAT (Intersection Embedding) ---
        
        # We need to process each agent and each time step
        # Flatten: [batch * num_agents * window_size, total_feature_dim]
        # NOTE: Use .reshape() instead of .view() to handle non-contiguous tensors
        obs_flat = obs_reshaped.reshape(-1, self.total_feature_dim)
        
        # Reshape to 12 lanes: [batch * ..., 12, lane_feature_dim]
        if self.total_feature_dim % 12 == 0:
            lane_features = obs_flat.view(-1, 12, self.total_feature_dim // 12)
            # Project: [batch * ..., 12, gat_hidden_dim]
            h_lanes = self.input_proj(lane_features)
        else:
            # Fallback if not divisible: Project whole vec to 12 * gat_hidden_dim
            # obs_flat: [batch * ..., total_feature_dim]
            # h_flat: [batch * ..., 12 * gat_hidden_dim]
            h_flat = self.input_proj(obs_flat)
            # Reshape to [batch * ..., 12, gat_hidden_dim]
            h_lanes = h_flat.view(-1, 12, self.gat_output_dim if hasattr(self, 'gat_output_dim') else 32) # Wait, gat_hidden_dim is not stored in self?
            # I need to check if gat_hidden_dim is stored. It is not.
            # But I can infer it from h_flat size.
            h_lanes = h_flat.view(h_flat.size(0), 12, -1)

        # Apply GAT on lanes
        # Expand adj matrices
        lane_adj_coop_batch = self.lane_adj_coop.unsqueeze(0).expand(h_lanes.size(0), -1, -1)
        lane_adj_conf_batch = self.lane_adj_conf.unsqueeze(0).expand(h_lanes.size(0), -1, -1)
        
        # Run separate GATs
        gat_out_coop = self.gat_layer_coop(h_lanes, lane_adj_coop_batch)
        gat_out_conf = self.gat_layer_conf(h_lanes, lane_adj_conf_batch)
        
        # Concatenate outputs: [batch * ..., 12, gat_total_output]
        gat_out = torch.cat([gat_out_coop, gat_out_conf], dim=-1)
        
        # Pooling (Mean) over lanes to get Intersection Embedding
        # [batch * ..., gat_output_dim * heads]
        intersection_emb_flat = gat_out.mean(dim=1)
        
        # Reshape back to [batch, window_size, num_agents, emb_dim]
        # Note: obs_reshaped was [batch, num_agents, window_size, ...]
        # So intersection_emb_flat corresponds to batch * num_agents * window_size
        intersection_emb_temporal = intersection_emb_flat.view(
            batch_size, num_agents, self.window_size, -1
        )
        
        # Permute to [batch, window_size, num_agents, emb_dim] for GraphSAGE compatibility if needed
        # But GraphSAGE expects [batch, T, N, feat]
        intersection_emb_temporal = intersection_emb_temporal.permute(0, 2, 1, 3)
        
        # Get current time step embedding
        # [batch, num_agents, emb_dim]
        intersection_emb = intersection_emb_temporal[:, -1, :, :]
        
        # --- Layer 2: Network-level GraphSAGE (Network Embedding) ---
        
        # Get network adjacency (directional: [4, N, N])
        if num_agents == 1:
            # Single agent: create dummy directional adjacency
            net_adj = torch.ones(4, 1, 1, device=obs.device)
        else:
            # self.network_adj is [4, N, N]
            net_adj = self.network_adj[:, :num_agents, :num_agents]  # [4, num_agents, num_agents]
        
        if self.window_size > 1:
            # Temporal GraphSAGE
            # Input: [batch, T, N, features], adj_directions: [4, N, N]
            network_emb = self.graphsage_bigru(intersection_emb_temporal, net_adj)
        else:
            # Standard GraphSAGE
            # Input: [batch, N, features], adj_directions: [4, N, N]
            # Output sequence: [batch, num_agents, graphsage_hidden_dim]
            network_emb_seq = self.graphsage_bigru(intersection_emb, net_adj, return_sequence=True)
            # Mean pooling over agents
            network_emb = network_emb_seq.mean(dim=1)
        
        # Select intersection embedding for specific agent or use mean
        if agent_idx is not None and num_agents > 1:
            # [batch, intersection_emb_dim]
            agent_intersection_emb = intersection_emb[:, agent_idx, :]
        else:
            # Mean pooling for single-agent or when no specific agent
            agent_intersection_emb = intersection_emb.mean(dim=1)
        
        # Joint embedding: concatenate intersection and network embeddings
        # [batch, joint_emb_dim]
        joint_emb = torch.cat([agent_intersection_emb, network_emb], dim=-1)
        
        return joint_emb, agent_intersection_emb, network_emb


class LocalTemporalMGMQEncoder(nn.Module):
    """
    Local Temporal MGMQ Encoder for Pre-packaged Observations.
    
    This encoder is designed for use with NeighborTemporalObservationFunction,
    which packages neighbor features directly in the observation. This makes it
    compatible with RLlib's batch shuffling since each sample is self-contained.
    
    Architecture:
    1. Time-distributed GAT on lanes (for self and each neighbor)
    2. Combine self + neighbors into local graph
    3. TemporalGraphSAGE_BiGRU on local star adjacency
    
    Args:
        obs_dim: Feature dimension per timestep (typically 48 = 4 features * 12 detectors)
        max_neighbors: Maximum number of neighbors (K)
        window_size: History length (T)
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
        window_size: int = 5,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 32,
        gat_num_heads: int = 4,
        graphsage_hidden_dim: int = 64,
        gru_hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super(LocalTemporalMGMQEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.max_neighbors = max_neighbors
        self.window_size = window_size
        self.num_lanes = 12
        
        # Calculate lane feature dimension
        self.lane_feature_dim = obs_dim // self.num_lanes  # 48/12 = 4
        
        # Input projection for lane features
        self.input_proj = nn.Sequential(
            nn.Linear(self.lane_feature_dim, gat_hidden_dim),
            nn.LayerNorm(gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers for lane-level processing
        self.gat_layer_coop = MultiHeadGATLayer(
            in_features=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.2,
            concat=True
        )
        
        self.gat_layer_conf = MultiHeadGATLayer(
            in_features=gat_hidden_dim,
            out_features=gat_output_dim,
            n_heads=gat_num_heads,
            dropout=dropout,
            alpha=0.2,
            concat=True
        )
        
        # Static lane adjacency matrices
        self.register_buffer('lane_adj_coop', get_lane_cooperation_matrix())
        self.register_buffer('lane_adj_conf', get_lane_conflict_matrix())
        
        # GAT output dimension
        self.gat_total_output = (gat_output_dim * gat_num_heads) * 2
        
        # Temporal GraphSAGE + BiGRU for local graph
        self.temporal_graphsage = TemporalGraphSAGE_BiGRU(
            in_features=self.gat_total_output,
            hidden_features=graphsage_hidden_dim,
            gru_hidden_size=gru_hidden_dim,
            history_length=window_size,
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
                - self_features: [B, T, 48]
                - neighbor_features: [B, K, T, 48]
                - neighbor_mask: [B, K]
                
        Returns:
            joint_emb: [B, joint_emb_dim]
        """
        self_feat = obs_dict["self_features"]        # [B, T, 48]
        neighbor_feat = obs_dict["neighbor_features"] # [B, K, T, 48]
        mask = obs_dict["neighbor_mask"]              # [B, K]
        
        B, T, feat_dim = self_feat.shape
        K = neighbor_feat.size(1)
        
        # ======== 1. Time-distributed GAT for self ========
        # [B, T, 48] -> [B, T, gat_dim]
        self_emb = self._run_gat_temporal(self_feat)
        
        # ======== 2. Time-distributed GAT for neighbors ========
        # [B, K, T, 48] -> [B*K, T, 48] -> [B*K, T, gat_dim] -> [B, K, T, gat_dim]
        neighbor_feat_flat = neighbor_feat.reshape(B * K, T, feat_dim)
        neighbor_emb_flat = self._run_gat_temporal(neighbor_feat_flat)
        neighbor_emb = neighbor_emb_flat.reshape(B, K, T, -1)
        
        # ======== 3. Combine into local graph ========
        # Stack: [B, T, 1+K, gat_dim]
        # self_emb: [B, T, gat_dim] -> [B, T, 1, gat_dim]
        # neighbor_emb: [B, K, T, gat_dim] -> [B, T, K, gat_dim]
        all_nodes = torch.cat([
            self_emb.unsqueeze(2),                    # [B, T, 1, gat_dim]
            neighbor_emb.permute(0, 2, 1, 3)          # [B, T, K, gat_dim]
        ], dim=2)  # -> [B, T, 1+K, gat_dim]
        
        # ======== 4. Build local star adjacency ========
        local_adj = self._build_star_adjacency(mask)  # [B, 4, 1+K, 1+K]
        
        # ======== 5. Temporal GraphSAGE + BiGRU ========
        # Input: [B, T, N, features] where N = 1+K
        network_emb = self.temporal_graphsage(all_nodes, local_adj)  # [B, hidden]
        
        # ======== 6. Joint Embedding ========
        # Intersection embedding: last timestep của self
        intersection_emb = self_emb[:, -1, :]  # [B, gat_dim]
        
        joint_emb = torch.cat([intersection_emb, network_emb], dim=-1)
        
        return joint_emb
        
    def _run_gat_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time-distributed GAT.
        
        Args:
            x: [B, T, 48] hoặc [B*K, T, 48]
        Returns:
            [B, T, gat_dim] hoặc [B*K, T, gat_dim]
        """
        batch_size, T, feat_dim = x.shape
        
        # Flatten B and T: [B*T, 48]
        x_flat = x.reshape(-1, feat_dim)
        
        # Reshape to lanes: [B*T, 12, 4]
        lane_feat = x_flat.view(-1, self.num_lanes, self.lane_feature_dim)
        
        # Project: [B*T, 12, gat_hidden]
        h = self.input_proj(lane_feat)
        
        # Expand lane adjacency
        adj_coop = self.lane_adj_coop.unsqueeze(0).expand(h.size(0), -1, -1)
        adj_conf = self.lane_adj_conf.unsqueeze(0).expand(h.size(0), -1, -1)
        
        # GAT forward
        out_coop = self.gat_layer_coop(h, adj_coop)  # [B*T, 12, heads*out]
        out_conf = self.gat_layer_conf(h, adj_conf)
        
        # Concat + pool over lanes
        gat_out = torch.cat([out_coop, out_conf], dim=-1)  # [B*T, 12, gat_dim]
        gat_pooled = gat_out.mean(dim=1)  # [B*T, gat_dim]
        
        # Reshape back: [B, T, gat_dim]
        return gat_pooled.view(batch_size, T, -1)
        
    def _build_star_adjacency(self, mask: torch.Tensor) -> torch.Tensor:
        """Build 1-hop star graph directional adjacency for local observation.
        
        Constructs a directional adjacency matrix for a star graph where:
        - Node 0: Self (central node)
        - Node 1: North neighbor (direction 0)
        - Node 2: East neighbor (direction 1)
        - Node 3: South neighbor (direction 2)
        - Node 4: West neighbor (direction 3)
        
        Edge Logic:
        - Self -> North neighbor: edge in direction 0 (North)
        - North neighbor -> Self: edge in direction 2 (South)
        (and similarly for other directions)
        
        Args:
            mask: [B, K] - Binary mask, 1 if neighbor valid, 0 otherwise
                  K should be 4 for standard 4-way intersection
        Returns:
            adj: [B, 4, 1+K, 1+K] - Directional adjacency matrices
                 Channel 0: North edges
                 Channel 1: East edges
                 Channel 2: South edges
                 Channel 3: West edges
        """
        B, K = mask.shape
        N = 1 + K  # Total nodes: Self + K neighbors
        
        # Create directional adjacency: [B, 4, N, N]
        adj = torch.zeros(B, 4, N, N, device=mask.device)
        
        # Build edges for each direction
        for direction in range(min(K, 4)):
            neighbor_idx = direction + 1  # Neighbor position in adj (1-indexed)
            
            # Edge: Self (idx 0) -> Neighbor in this direction
            # Apply mask so invalid neighbors have 0 weight
            adj[:, direction, 0, neighbor_idx] = mask[:, direction]
            
            # Edge: Neighbor -> Self (opposite direction)
            # If neighbor is to North (dir 0), self is to South (dir 2) from its view
            opposite_dir = (direction + 2) % 4
            adj[:, opposite_dir, neighbor_idx, 0] = mask[:, direction]
            
        return adj


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
        
        # Policy network (actor)
        policy_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Policy output: mean for continuous actions
        self.policy_mean = nn.Linear(prev_dim, action_dim)
        # Log std as learnable parameter
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value network (critic)
        value_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in value_hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller initialization for policy output
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value_out.weight, gain=1.0)
        
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
        
        # Observation and action dimensions
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
        
        # Output layer for policy (num_outputs = 2 * action_dim for mean and log_std)
        self.policy_out = nn.Linear(prev_dim, num_outputs)
        
        # Value network
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
        """Initialize weights."""
        for module in [self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.policy_out.weight, gain=0.01)
        nn.init.orthogonal_(self.value_out.weight, gain=1.0)
        
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
        """
        obs = input_dict["obs_flat"].float()
        
        # Get joint embedding from MGMQ encoder
        joint_emb, _, _ = self.mgmq_encoder(obs)
        
        # Store for value function
        self._features = joint_emb
        
        # Policy output
        policy_features = self.policy_net(joint_emb)
        policy_out = self.policy_out(policy_features)
        
        # Clamp log_std to prevent entropy explosion
        # policy_out contains [mean, log_std] - clamp the log_std part
        action_dim = policy_out.shape[-1] // 2
        mean = policy_out[..., :action_dim]
        log_std = policy_out[..., action_dim:]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
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


class LocalTemporalMGMQTorchModel(TorchModelV2, nn.Module):
    """
    RLlib wrapper for LocalTemporalMGMQEncoder with Dict observation space.
    
    This model is designed for use with NeighborTemporalObservationFunction,
    which provides pre-packaged observations with neighbor features.
    
    Observation space expected:
        Dict({
            "self_features": Box[T, feature_dim],
            "neighbor_features": Box[K, T, feature_dim],
            "neighbor_mask": Box[K]
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
        """Initialize the local temporal MGMQ model for RLlib."""
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        custom_config = model_config.get("custom_model_config", {})
        
        # Extract dimensions from Dict observation space
        # obs_space should be a Dict space
        if hasattr(obs_space, 'spaces'):
            # Dict space
            self_shape = obs_space.spaces["self_features"].shape  # (T, feature_dim)
            neighbor_shape = obs_space.spaces["neighbor_features"].shape  # (K, T, feature_dim)
        elif hasattr(obs_space, 'original_space'):
            # Flattened Dict - get original
            orig = obs_space.original_space
            self_shape = orig.spaces["self_features"].shape
            neighbor_shape = orig.spaces["neighbor_features"].shape
        else:
            # Fallback defaults
            T = custom_config.get("window_size", 5)
            feature_dim = custom_config.get("obs_dim", 48)
            K = custom_config.get("max_neighbors", 4)
            self_shape = (T, feature_dim)
            neighbor_shape = (K, T, feature_dim)
        
        T, feature_dim = self_shape
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
        
        # MGMQ Encoder
        self.mgmq_encoder = LocalTemporalMGMQEncoder(
            obs_dim=feature_dim,
            max_neighbors=K,
            window_size=T,
            gat_hidden_dim=gat_hidden_dim,
            gat_output_dim=gat_output_dim,
            gat_num_heads=gat_num_heads,
            graphsage_hidden_dim=graphsage_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            dropout=dropout
        )
        
        joint_emb_dim = self.mgmq_encoder.output_dim
        
        # Policy network
        policy_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.policy_net = nn.Sequential(*policy_layers)
        self.policy_out = nn.Linear(prev_dim, num_outputs)
        
        # Value network
        value_layers = []
        prev_dim = joint_emb_dim
        for hidden_dim in value_hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(prev_dim, 1)
        
        self._features = None
        self._value = None
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.policy_out.weight, gain=0.01)
        nn.init.orthogonal_(self.value_out.weight, gain=1.0)
        
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
        
        # Clamp log_std to prevent entropy explosion
        # policy_out contains [mean, log_std] - clamp the log_std part
        action_dim = policy_out.shape[-1] // 2
        mean = policy_out[..., :action_dim]
        log_std = policy_out[..., action_dim:]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
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

