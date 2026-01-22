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
        self.proj_self = nn.Linear(in_features, hidden_features)
        self.proj_north = nn.Linear(in_features, hidden_features)
        self.proj_east = nn.Linear(in_features, hidden_features)
        self.proj_south = nn.Linear(in_features, hidden_features)
        self.proj_west = nn.Linear(in_features, hidden_features)
        
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
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [Batch, N, In_Features]
            adj: [Batch, N, N]
        """
        # Ensure batch dim
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(h.size(0), -1, -1)

        batch_size, num_nodes, _ = h.size()
        
        # 1. Directional Projections
        # Each [Batch, N, Hidden]
        g_self = F.relu(self.proj_self(h))
        g_north = F.relu(self.proj_north(h))
        g_east = F.relu(self.proj_east(h))
        g_south = F.relu(self.proj_south(h))
        g_west = F.relu(self.proj_west(h))
        
        # 2. Neighbor Exchange (Sampling) via Adjacency
        # "My North input comes from Neighbors' South outputs"
        # Since adj is undirected/unknown, we sum contributions.
        # Ideally adj would be directional, but this approximates "South-bound flux affecting my North"
        in_north = torch.bmm(adj, g_south) # Neighbors' South -> My North
        in_east  = torch.bmm(adj, g_west)  # Neighbors' West -> My East
        in_south = torch.bmm(adj, g_north) # Neighbors' North -> My South
        in_west  = torch.bmm(adj, g_east)  # Neighbors' East -> My West
        
        # 3. Bi-GRU Aggregation
        # Sequence: [North, East, South, West, Self]
        # Stack: [Batch, N, 5, Hidden]
        seq_tensor = torch.stack([in_north, in_east, in_south, in_west, g_self], dim=2)
        
        # Flatten for GRU: [Batch*N, 5, Hidden]
        seq_flat = seq_tensor.view(batch_size * num_nodes, 5, self.hidden_features)
        
        # Run BiGRU
        # gru_out: [Batch*N, 5, 2*Hidden]
        gru_out, _ = self.bigru(seq_flat)
        
        # Take the state at the 'Self' position (index -1) as the integrated embedding
        final_state = gru_out[:, -1, :] # [Batch*N, 2*Hidden]
        
        # Reshape to [Batch, N, 2*Hidden]
        final_state = final_state.view(batch_size, num_nodes, -1)
        
        # Final projection
        # [Batch, N, Out]
        out = self.output_linear(final_state)
        out = self.dropout_layer(out)
        
        if squeeze_output:
            out = out.squeeze(0)
            
        return out


class GraphSAGE_BiGRU(nn.Module):
    """
    Wrapper for DirectionalGraphSAGE to maintain compatibility.
    Replaces the standard GraphSAGE + Temporal BiGRU with the Directional version.
    
    Args:
        in_features: Input feature dimension
        hidden_features: Output feature dimension
        gru_hidden_size: Internal hidden dimension for directional vectors
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        gru_hidden_size: int = 32,
        num_gru_layers: int = 1, # Unused
        dropout: float = 0.5,
        aggregator_type: str = 'mean' # Unused
    ):
        super(GraphSAGE_BiGRU, self).__init__()
        # Use the directional layer
        # Note: hidden_features is the output dimension here
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
    
    def forward(self, h, adj, return_sequence=False):
        return self.layer(h, adj)


class TemporalGraphSAGE_BiGRU(nn.Module):
    """
    Temporal Extension of DirectionalGraphSAGE.
    1. Spatial: Directional GraphSAGE at each time step.
    2. Temporal: Bi-GRU over time history.
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
        # Spatial Layer (Directional)
        self.spatial_layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=gru_hidden_size,
            out_features=hidden_features,
            dropout=dropout
        )
        
        # Temporal Layer (Bi-GRU over time)
        self.temporal_bigru = nn.GRU(
            input_size=hidden_features,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.output_proj = nn.Linear(gru_hidden_size * 2, hidden_features)
        
    def forward(self, h_history, adj, return_sequence=False):
        # h_history: [Batch, T, N, In]
        batch, T, N, feats = h_history.size()
        
        # 1. Spatial (Directional)
        h_flat = h_history.reshape(batch * T, N, feats)
        
        if adj.dim() == 2:
            adj_exp = adj.unsqueeze(0).expand(batch * T, -1, -1)
        else:
            adj_exp = adj.repeat_interleave(T, dim=0)
            
        # [Batch*T, N, Hidden]
        spatial_out = self.spatial_layer(h_flat, adj_exp)
        
        # Unfold time: [Batch, N, T, Hidden] (Permuted)
        spatial_out = spatial_out.view(batch, T, N, -1).permute(0, 2, 1, 3)
        # Flatten: [Batch*N, T, Hidden] for Temporal GRU
        spatial_seq = spatial_out.reshape(batch * N, T, -1)
        
        # 2. Temporal
        t_out, _ = self.temporal_bigru(spatial_seq)
        
        # Last step
        last_out = t_out[:, -1, :] # [Batch*N, 2*GRU_Hidden]
        
        # Project
        out = self.output_proj(last_out)
        
        # Reshape to [Batch, N, Out]
        out = out.view(batch, N, -1)
        
        # Pooling over nodes for Network Embedding
        network_emb = out.mean(dim=1)
        
        return network_emb
