"""
GraphSAGE + Bi-GRU Layer for Network Embedding.

This module implements the GraphSAGE aggregation combined with 
Bidirectional GRU for capturing network-level temporal features
in the MGMQ architecture.

Reference: 
# - Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
# - MGMQ Paper: Multi-Layer graph masking Q-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List



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
        
        # 1. Projections (Shared Dense Layers for Directional Split)
        # Đoạn code này giúp cho mỗi node có 5 "bản sao" đặc trưng, mỗi bản dùng cho một hướng truyên thông tin khác nhau
        self.proj_self = nn.Linear(in_features, hidden_features)    # Chiếu đặc trưng node cho chính nó
        self.proj_north = nn.Linear(in_features, hidden_features)   # Chiếu đặc trưng node cho hướng Bắc
        self.proj_east = nn.Linear(in_features, hidden_features)    # Chiếu đặc trưng node cho hướng Đông
        self.proj_south = nn.Linear(in_features, hidden_features)   # Chiếu đặc trưng node cho hướng Nam
        self.proj_west = nn.Linear(in_features, hidden_features)    # Chiếu đặc trưng node cho hướng Tây
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 2. Bi-GRU Aggregator (Over directions: N, E, S, W, Self)
        # Input size: hidden_features. Hidden size: hidden_features.
        self.bigru = nn.GRU(
            input_size=hidden_features,
            hidden_size=hidden_features,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Final Output Linear
        # BiGRU (bidir) outputs 2 * hidden_features
        self.output_linear = nn.Linear(hidden_features * 2, out_features)
        
    def forward(self, h: torch.Tensor, adj_directions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DirectionalGraphSAGE.
        
        Args:
            h: [Batch, N, In_Features] or [N, In_Features] - Node features
            adj_directions: [Batch, 4, N, N] or [4, N, N] - Directional adjacency matrices
                           Channel 0: North neighbors mask
                           Channel 1: East neighbors mask
                           Channel 2: South neighbors mask
                           Channel 3: West neighbors mask
        
        Returns:
            out: [Batch, N, Out_Features] or [N, Out_Features] - Node embeddings
        """
        # === Input Validation ===
        assert h.dim() in [2, 3], f"Expected h to have 2 or 3 dims, got {h.dim()}"
        assert adj_directions.dim() in [3, 4], f"Expected adj_directions to have 3 or 4 dims, got {adj_directions.dim()}"
        
        # === Handle batch dimension for h ===
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_nodes, _ = h.size()
        
        # === Handle batch dimension for adj_directions ===
        # adj_directions should be [Batch, 4, N, N]
        if adj_directions.dim() == 3:
            # [4, N, N] -> [Batch, 4, N, N]
            adj_directions = adj_directions.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Validate adjacency shape
        assert adj_directions.size(1) == 4, f"Expected 4 directions, got {adj_directions.size(1)}"
        
        # === Extract directional masks ===
        # Each mask is [Batch, N, N]
        mask_north = adj_directions[:, 0, :, :]  # Neighbors to the North
        mask_east  = adj_directions[:, 1, :, :]  # Neighbors to the East
        mask_south = adj_directions[:, 2, :, :]  # Neighbors to the South
        mask_west  = adj_directions[:, 3, :, :]  # Neighbors to the West
        
        # === Step 1: Directional Projections ===
        # Each output: [Batch, N, Hidden]
        g_self = F.relu(self.proj_self(h))
        g_north = F.relu(self.proj_north(h))
        g_east = F.relu(self.proj_east(h))
        g_south = F.relu(self.proj_south(h))
        g_west = F.relu(self.proj_west(h))
        
        # === Step 2: Neighbor Exchange via Directional Adjacency ===
        # Logic: "My North input comes from South outputs of my North neighbors"
        # This is physically correct for traffic flow modeling
        in_north = torch.bmm(mask_north, g_south)  # North neighbors' South -> My North input
        in_east  = torch.bmm(mask_east, g_west)    # East neighbors' West -> My East input
        in_south = torch.bmm(mask_south, g_north)  # South neighbors' North -> My South input
        in_west  = torch.bmm(mask_west, g_east)    # West neighbors' East -> My West input
        
        # === Step 3: Bi-GRU Aggregation over Directions ===
        # Stack directions as sequence: [North, East, South, West, Self]
        # Shape: [Batch, N, 5, Hidden]
        seq_tensor = torch.stack([in_north, in_east, in_south, in_west, g_self], dim = 2)
        
        # Reshape for GRU: [Batch*N, 5, Hidden]
        seq_flat = seq_tensor.view(batch_size * num_nodes, 5, self.hidden_features)
        
        # Run BiGRU: output shape [Batch*N, 5, 2*Hidden]
        gru_out, _ = self.bigru(seq_flat)
        
        # Take the state at 'Self' position (last in sequence) as node embedding
        final_state = gru_out[:, -1, :]  # [Batch*N, 2*Hidden]
        
        # === Step 4: Output Projection ===
        # Reshape to [Batch, N, 2*Hidden]
        final_state = final_state.view(batch_size, num_nodes, -1)
        
        # Project to output dimension: [Batch, N, Out]
        out = self.output_linear(final_state)
        out = self.dropout_layer(out)
        
        if squeeze_output:
            out = out.squeeze(0)
            
        return out


class GraphSAGE_BiGRU(nn.Module):
    """
    Wrapper for DirectionalGraphSAGE to maintain API compatibility.
    
    This class provides a simple interface that wraps the DirectionalGraphSAGE
    implementation with additional parameters for backward compatibility.
    
    Args:
        in_features: Input feature dimension
        hidden_features: Output feature dimension  
        gru_hidden_size: Internal hidden dimension for directional vectors
        num_gru_layers: Unused, kept for compatibility
        dropout: Dropout rate
        aggregator_type: Unused, kept for compatibility
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        gru_hidden_size: int = 32,
        num_gru_layers: int = 1,
        dropout: float = 0.5,
        aggregator_type: str = 'mean'
    ):
        super(GraphSAGE_BiGRU, self).__init__()
        
        # Note: num_gru_layers and aggregator_type are kept for API compatibility
        # but not used in the directional implementation
        
        self.layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=gru_hidden_size,
            out_features=hidden_features,
            dropout=dropout
        )
        self.hidden_features = hidden_features
        
    @property
    def output_dim(self) -> int:
        """Return output feature dimension."""
        return self.hidden_features
    
    def forward(
        self, 
        h: torch.Tensor, 
        adj_directions: torch.Tensor, 
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through DirectionalGraphSAGE.
        
        Args:
            h: [Batch, N, In_Features] or [N, In_Features] - Node features
            adj_directions: [Batch, 4, N, N] or [4, N, N] - Directional adjacency
            return_sequence: Unused, kept for compatibility
            
        Returns:
            out: [Batch, N, Out_Features] or [N, Out_Features] - Node embeddings
        """
        return self.layer(h, adj_directions)


class TemporalGraphSAGE_BiGRU(nn.Module):
    """
    Temporal Extension of DirectionalGraphSAGE.
    
    This module processes temporal sequences of graph data:
    1. Spatial: DirectionalGraphSAGE at each time step
    2. Temporal: Bi-GRU aggregation over time history
    3. Output: Network-level embedding via mean pooling
    
    Args:
        in_features: Input feature dimension per node
        hidden_features: Output feature dimension
        gru_hidden_size: Hidden dimension for GRU layers
        history_length: Number of time steps (unused, kept for compatibility)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        gru_hidden_size: int = 32,
        history_length: int = 5,
        dropout: float = 0.5
    ):
        super(TemporalGraphSAGE_BiGRU, self).__init__()
        
        # Spatial Layer: DirectionalGraphSAGE for each timestep
        self.spatial_layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=gru_hidden_size,
            out_features=hidden_features,
            dropout=dropout
        )
        
        # Temporal Layer: Bi-GRU to aggregate across timesteps
        self.temporal_bigru = nn.GRU(
            input_size=hidden_features,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection: 2*gru_hidden -> hidden_features
        self.output_proj = nn.Linear(gru_hidden_size * 2, hidden_features)
        
    def forward(
        self, 
        h_history: torch.Tensor, 
        adj_directions: torch.Tensor, 
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for temporal graph data.
        
        Args:
            h_history: [Batch, T, N, In_Features] - Node features over T timesteps
            adj_directions: [Batch, 4, N, N] or [4, N, N] - Directional adjacency
            return_sequence: Unused, kept for compatibility
            
        Returns:
            network_emb: [Batch, Out_Features] - Network-level embedding
        """
        batch, T, N, feats = h_history.size()
        
        # === Step 1: Spatial Processing ===
        # Flatten batch and time: [Batch*T, N, Features]
        h_flat = h_history.reshape(batch * T, N, feats)
        
        # Expand adj_directions for all timesteps
        if adj_directions.dim() == 3:
            # [4, N, N] -> [Batch*T, 4, N, N]
            adj_exp = adj_directions.unsqueeze(0).expand(batch * T, -1, -1, -1)
        else:
            # [Batch, 4, N, N] -> [Batch*T, 4, N, N]
            adj_exp = adj_directions.repeat_interleave(T, dim=0)
        
        # Apply spatial GraphSAGE: [Batch*T, N, Hidden]
        spatial_out = self.spatial_layer(h_flat, adj_exp)
        
        # === Step 2: Temporal Processing ===
        # Reshape: [Batch, T, N, Hidden] -> [Batch, N, T, Hidden]
        spatial_out = spatial_out.view(batch, T, N, -1).permute(0, 2, 1, 3)
        
        # Flatten for temporal GRU: [Batch*N, T, Hidden]
        spatial_seq = spatial_out.reshape(batch * N, T, -1)
        
        # Apply temporal Bi-GRU
        t_out, _ = self.temporal_bigru(spatial_seq)
        
        # Take last timestep: [Batch*N, 2*GRU_Hidden]
        last_out = t_out[:, -1, :]
        
        # === Step 3: Output Projection ===
        # Project: [Batch*N, Hidden]
        out = self.output_proj(last_out)
        
        # Reshape: [Batch, N, Hidden]
        out = out.view(batch, N, -1)
        
        # === Step 4: Network-level Pooling ===
        # Mean pooling over nodes: [Batch, Hidden]
        network_emb = out.mean(dim=1)
        
        return network_emb
