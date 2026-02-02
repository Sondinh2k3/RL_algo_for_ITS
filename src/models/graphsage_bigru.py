"""
GraphSAGE + Bi-GRU Layer for Network Embedding.

This module implements the GraphSAGE aggregation combined with 
Bidirectional GRU for capturing SPATIAL features from neighbors
in the MGMQ architecture.

IMPORTANT NOTE:
    BiGRU in this module is used for SPATIAL aggregation (over directions or neighbors),
    NOT for temporal sequence processing.
    
    - DirectionalGraphSAGE: BiGRU aggregates over 4 SPATIAL directions (N, E, S, W)
    - NeighborGraphSAGE_BiGRU: BiGRU aggregates over K neighbors in star-graph topology

Reference: 
# - Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
# - MGMQ Paper: Multi-Layer graph masking Q-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalGraphSAGE(nn.Module):
    """
    Directional GraphSAGE with Bi-GRU Aggregation.
    
    This module implements the "Directional/Topology-aware" GraphSAGE mechanism:
    1. Projects node inputs into 5 directional vectors (Self, N, E, S, W).
    2. Aggregates neighbor information such that:
       - Input at North port comes from South outputs of neighbors.
       - Input at East port comes from West outputs of neighbors.
       - etc.
    3. Uses a Bi-Directional GRU to aggregate these directional inputs into a single embedding.
    
    IMPORTANT: BiGRU here processes a sequence of 4 SPATIAL DIRECTIONS (N, E, S, W),
               NOT a temporal sequence.
    
    Args:
        in_features: Input feature dimension.
        hidden_features: Dimension for directional vectors.
        out_features: Output dimension of the network embedding.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        out_features: int = 64,
        dropout: float = 0.5
    ):
        super(DirectionalGraphSAGE, self).__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        # 1. Directional Projections
        self.proj_self = nn.Linear(in_features, hidden_features)
        self.proj_north = nn.Linear(in_features, hidden_features)
        self.proj_east = nn.Linear(in_features, hidden_features)
        self.proj_south = nn.Linear(in_features, hidden_features)
        self.proj_west = nn.Linear(in_features, hidden_features)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 2. Bi-GRU Aggregator over 4 SPATIAL directions (N, E, S, W)
        self.bigru = nn.GRU(
            input_size=hidden_features,
            hidden_size=hidden_features,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Output projection: [h_self || G_k] -> out
        # G_k = BiGRU output for all 4 directions = 4 * (hidden_features * 2)
        # Total: hidden_features (self) + 4 * hidden_features * 2 (neighbors)
        self.output_linear = nn.Linear(hidden_features + 4 * hidden_features * 2, out_features)
        
        # LeakyReLU as per MGMQ specification
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, h: torch.Tensor, adj_directions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DirectionalGraphSAGE.
        
        Args:
            h: [Batch, N, In_Features] or [N, In_Features] - Node features
            adj_directions: [Batch, 4, N, N] or [4, N, N] - Directional adjacency matrices
        
        Returns:
            out: [Batch, N, Out_Features] or [N, Out_Features] - Node embeddings
        """
        # Handle batch dimension
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_nodes, _ = h.size()
        
        if adj_directions.dim() == 3:
            adj_directions = adj_directions.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Extract directional masks
        mask_north = adj_directions[:, 0, :, :]
        mask_east  = adj_directions[:, 1, :, :]
        mask_south = adj_directions[:, 2, :, :]
        mask_west  = adj_directions[:, 3, :, :]
        
        # Step 1: Directional Projections
        g_self = F.relu(self.proj_self(h))
        g_north = F.relu(self.proj_north(h))
        g_east = F.relu(self.proj_east(h))
        g_south = F.relu(self.proj_south(h))
        g_west = F.relu(self.proj_west(h))
        
        # Step 2: Topology-aware neighbor exchange
        in_north = torch.bmm(mask_north, g_south)
        in_east  = torch.bmm(mask_east, g_west)
        in_south = torch.bmm(mask_south, g_north)
        in_west  = torch.bmm(mask_west, g_east)
        
        # Step 3: Bi-GRU Aggregation over 4 SPATIAL directions
        # Input sequence S_v = [h_North, h_East, h_South, h_West] (clockwise from North)
        seq_tensor = torch.stack([in_north, in_east, in_south, in_west], dim=2)
        seq_flat = seq_tensor.view(batch_size * num_nodes, 4, self.hidden_features)
        
        # BiGRU processes sequence in both directions:
        # Forward: N -> E -> S -> W (upstream flow)
        # Backward: W -> S -> E -> N (downstream feedback)
        bigru_output, _ = self.bigru(seq_flat)  # [batch*nodes, 4, hidden*2]
        
        # G_k = Concat all BiGRU outputs for all 4 directions
        # Shape: [batch*nodes, 4 * hidden * 2]
        G_k = bigru_output.reshape(batch_size * num_nodes, -1)
        
        # Step 4: Combine self + neighbor context
        # z_raw = Concat(h_v, G_k)
        g_self_flat = g_self.view(batch_size * num_nodes, -1)
        z_raw = torch.cat([g_self_flat, G_k], dim=-1)
        
        # z_final = LeakyReLU(W * z_raw + b)
        out = self.output_linear(z_raw)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = out.view(batch_size, num_nodes, -1)
        
        if squeeze_output:
            out = out.squeeze(0)
            
        return out


class GraphSAGE_BiGRU(nn.Module):
    """
    Wrapper for DirectionalGraphSAGE with simplified interface.
    
    Args:
        in_features: Input feature dimension
        hidden_features: Output feature dimension  
        gru_hidden_size: Internal hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        gru_hidden_size: int = 32,
        dropout: float = 0.5
    ):
        super(GraphSAGE_BiGRU, self).__init__()
        
        self.layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=gru_hidden_size,
            out_features=hidden_features,
            dropout=dropout
        )
        self.hidden_features = hidden_features
        
    @property
    def output_dim(self) -> int:
        return self.hidden_features
    
    def forward(self, h: torch.Tensor, adj_directions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DirectionalGraphSAGE.
        
        Args:
            h: [Batch, N, In_Features] - Node features
            adj_directions: [Batch, 4, N, N] or [4, N, N] - Directional adjacency
            
        Returns:
            out: [Batch, N, Out_Features] - Node embeddings
        """
        return self.layer(h, adj_directions)


class NeighborGraphSAGE_BiGRU(nn.Module):
    """
    Neighbor-based GraphSAGE with Bi-GRU for Spatial Aggregation.
    
    Used when --use-local-gnn is enabled. BiGRU aggregates over K neighbors spatially.
    
    Architecture:
    1. Project self features: h_self = Linear(self_features)
    2. Project neighbor features: h_neighbors = Linear(neighbor_features)
    3. BiGRU aggregation over neighbors (spatial sequence)
    4. Combine: output = Linear([h_self || BiGRU_output])
    
    Args:
        in_features: Input feature dimension per node
        hidden_features: Output feature dimension
        gru_hidden_size: Hidden dimension for BiGRU
        max_neighbors: Maximum number of neighbors
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        gru_hidden_size: int = 32,
        max_neighbors: int = 4,
        dropout: float = 0.5
    ):
        super(NeighborGraphSAGE_BiGRU, self).__init__()
        
        self.hidden_features = hidden_features
        self.gru_hidden_size = gru_hidden_size
        
        # Self and neighbor projections
        self.self_proj = nn.Sequential(
            nn.Linear(in_features, gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.neighbor_proj = nn.Sequential(
            nn.Linear(in_features, gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bi-GRU for SPATIAL aggregation over neighbors
        self.spatial_bigru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        # Input: h_self (gru_hidden_size) + G_k (max_neighbors * gru_hidden_size * 2)
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden_size + max_neighbors * gru_hidden_size * 2, hidden_features),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        
        self.max_neighbors = max_neighbors
        
    def forward(
        self,
        self_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for neighbor-based spatial aggregation.
        
        Args:
            self_features: [Batch, In_Features] - Self node features
            neighbor_features: [Batch, MaxNeighbors, In_Features] - Neighbor features
            neighbor_mask: [Batch, MaxNeighbors] - Binary mask (1=valid, 0=padding)
            
        Returns:
            output: [Batch, Hidden_Features] - Aggregated embedding
        """
        # Project features
        h_self = self.self_proj(self_features)
        h_neighbors = self.neighbor_proj(neighbor_features)
        
        # Apply mask (zero out padded neighbors)
        mask_expanded = neighbor_mask.unsqueeze(-1).float()
        h_neighbors_masked = h_neighbors * mask_expanded
        
        # BiGRU Spatial Aggregation over K neighbors
        # Output contains forward and backward hidden states for all positions
        bigru_output, _ = self.spatial_bigru(h_neighbors_masked)  # [Batch, K, hidden*2]
        
        # G_k = Concat all BiGRU outputs for all K neighbors
        G_k = bigru_output.reshape(bigru_output.size(0), -1)  # [Batch, K * hidden * 2]
        
        # Combine: z_raw = Concat(h_self, G_k)
        z_raw = torch.cat([h_self, G_k], dim=-1)
        output = self.output_proj(z_raw)
        
        return output
    
    @property
    def output_dim(self) -> int:
        return self.hidden_features
