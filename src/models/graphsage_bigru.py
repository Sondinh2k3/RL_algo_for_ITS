"""
GraphSAGE + Bi-GRU Layer for Network Embedding.

This module implements the GraphSAGE aggregation combined with 
Bidirectional GRU for capturing network-level temporal features
in the MGMQ architecture.

Reference: 
- Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
- MGMQ Paper: Multi-Layer graph masking Q-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer for neighborhood aggregation.
    
    GraphSAGE learns to aggregate feature information from a node's
    local neighborhood using mean/max/LSTM aggregators.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        aggregator_type: Type of aggregation ('mean', 'max', 'lstm')
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator_type: str = 'mean',
        dropout: float = 0.5,
        bias: bool = True
    ):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        
        # Linear transformation for self features
        self.self_linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Linear transformation for aggregated neighbor features
        self.neigh_linear = nn.Linear(in_features, out_features, bias=bias)
        
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(out_features)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.neigh_linear.weight)
        if self.self_linear.bias is not None:
            nn.init.zeros_(self.self_linear.bias)
        if self.neigh_linear.bias is not None:
            nn.init.zeros_(self.neigh_linear.bias)
    
    def _aggregate_neighbors(
        self, 
        h: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate neighbor features.
        
        Args:
            h: Node features [batch, N, in_features]
            adj: Adjacency matrix [batch, N, N] or [N, N]
            
        Returns:
            Aggregated neighbor features [batch, N, in_features]
        """
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(h.size(0), -1, -1)
            
        # Normalize adjacency by degree (exclude self-loops for neighbor aggregation)
        adj_no_self = adj.clone()
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj_no_self = adj_no_self - eye * adj_no_self  # Remove self-loops
        
        # Compute degree for normalization
        degree = adj_no_self.sum(dim=-1, keepdim=True).clamp(min=1)
        
        if self.aggregator_type == 'mean':
            # Mean aggregation
            neigh_agg = torch.bmm(adj_no_self, h) / degree
            
        elif self.aggregator_type == 'max':
            # Max aggregation (element-wise max over neighbors)
            batch_size, N, feat_dim = h.size()
            neigh_agg = torch.zeros_like(h)
            
            for b in range(batch_size):
                for i in range(N):
                    neighbors = adj_no_self[b, i].nonzero(as_tuple=True)[0]
                    if len(neighbors) > 0:
                        neigh_features = h[b, neighbors]
                        neigh_agg[b, i] = neigh_features.max(dim=0)[0]
                    else:
                        neigh_agg[b, i] = h[b, i]  # Use self if no neighbors
                        
        elif self.aggregator_type == 'lstm':
            # LSTM aggregation (order neighbors by some criterion)
            batch_size, N, feat_dim = h.size()
            neigh_agg = torch.zeros_like(h)
            
            for b in range(batch_size):
                for i in range(N):
                    neighbors = adj_no_self[b, i].nonzero(as_tuple=True)[0]
                    if len(neighbors) > 0:
                        neigh_features = h[b, neighbors].unsqueeze(0)  # [1, num_neighbors, feat]
                        _, (hn, _) = self.lstm(neigh_features)
                        neigh_agg[b, i] = hn.squeeze(0).squeeze(0)
                    else:
                        neigh_agg[b, i] = h[b, i]
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")
            
        return neigh_agg
    
    def forward(
        self, 
        h: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of GraphSAGE layer.
        
        Args:
            h: Node features [N, in_features] or [batch, N, in_features]
            adj: Adjacency matrix [N, N] or [batch, N, N]
            
        Returns:
            Updated node features [N, out_features] or [batch, N, out_features]
        """
        # Handle batch dimension
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Apply dropout to input
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Self transformation
        self_feat = self.self_linear(h)
        
        # Aggregate and transform neighbor features
        neigh_agg = self._aggregate_neighbors(h, adj)
        neigh_feat = self.neigh_linear(neigh_agg)
        
        # Combine self and neighbor features
        out = self_feat + neigh_feat
        
        # Apply layer normalization and activation
        out = self.layer_norm(out)
        out = F.relu(out)
        
        if squeeze_output:
            out = out.squeeze(0)
            
        return out


class GraphSAGE_BiGRU(nn.Module):
    """
    GraphSAGE + Bidirectional GRU for Network Embedding.
    
    This module combines GraphSAGE for spatial aggregation with
    Bi-GRU for temporal feature extraction, following the MGMQ architecture.
    
    The flow is:
    1. GraphSAGE aggregates neighborhood information (Spatial)
    2. Bi-GRU processes the sequence of intersection embeddings (Temporal)
    3. Output is the network-level embedding
    
    Args:
        in_features: Input feature dimension (from GAT output)
        hidden_features: Hidden dimension for GraphSAGE
        gru_hidden_size: Hidden size for Bi-GRU
        num_gru_layers: Number of GRU layers
        dropout: Dropout rate
        aggregator_type: GraphSAGE aggregator type
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
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gru_hidden_size = gru_hidden_size
        
        # GraphSAGE layer for spatial aggregation
        self.graphsage = GraphSAGELayer(
            in_features=in_features,
            out_features=hidden_features,
            aggregator_type=aggregator_type,
            dropout=dropout
        )
        
        # Bidirectional GRU for temporal/sequential processing
        # Note: This GRU is intended to run over TIME steps, not over NODES.
        # It should be used within TemporalGraphSAGE_BiGRU or with time-series input.
        self.bigru = nn.GRU(
            input_size=hidden_features,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(gru_hidden_size * 2, hidden_features)
        
    @property
    def output_dim(self) -> int:
        """Return the output dimension of this module."""
        return self.hidden_features
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of GraphSAGE (Spatial only).
        
        WARNING: This method only applies GraphSAGE. 
        For Temporal processing with Bi-GRU, use TemporalGraphSAGE_BiGRU.
        
        Args:
            h: Node features from GAT [batch, N, in_features] or [N, in_features]
            adj: Adjacency matrix [N, N] or [batch, N, N]
            return_sequence: If True, return full sequence; else return last state
            
        Returns:
            Network embedding: [batch, N, hidden_features]
        """
        # Handle batch dimension
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Apply GraphSAGE for spatial aggregation
        # Output: [batch, N, hidden_features]
        sage_out = self.graphsage(h, adj)
        
        # In non-temporal mode, we just return the spatial features
        # Bi-GRU is not applied here because there is no time dimension
        out = sage_out
        
        if squeeze_output:
            out = out.squeeze(0)
            
        return out


class TemporalGraphSAGE_BiGRU(nn.Module):
    """
    Temporal-aware GraphSAGE + Bi-GRU with history buffer.
    
    This version correctly implements Spatio-Temporal learning:
    1. Spatial: GraphSAGE aggregates neighbors at each time step.
    2. Temporal: Bi-GRU processes the history sequence of each node.
    
    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension
        gru_hidden_size: GRU hidden size
        history_length: Number of past time steps to consider
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
        self.history_length = history_length
        self.in_features = in_features
        
        # GraphSAGE layer for spatial aggregation (shared across time steps)
        self.graphsage = GraphSAGELayer(
            in_features=in_features,
            out_features=hidden_features,
            aggregator_type='mean',
            dropout=dropout
        )
        
        # Bi-GRU for temporal processing
        # Input: Sequence of spatial embeddings over time
        self.bigru = nn.GRU(
            input_size=hidden_features,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(gru_hidden_size * 2, hidden_features)
        
    @property
    def output_dim(self) -> int:
        return self.output_proj.out_features
        
    def forward(
        self,
        h_history: torch.Tensor,
        adj: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with temporal history.
        
        Args:
            h_history: Historical node features [batch, T, N, in_features]
                       where T is history_length
            adj: Adjacency matrix [N, N] or [batch, N, N]
            return_sequence: Whether to return full sequence (unused here)
            
        Returns:
            Network embedding: [batch, hidden_features] (Pooled over nodes)
        """
        batch_size, T, N, feat_dim = h_history.size()
        
        # 1. Spatial Aggregation (GraphSAGE) applied to each time step
        # Reshape to treat time steps as part of batch for parallel processing
        # [batch * T, N, feat_dim]
        # NOTE: Use .reshape() instead of .view() to handle non-contiguous tensors
        h_flat = h_history.reshape(-1, N, feat_dim)
        
        # Expand adjacency if needed
        if adj.dim() == 2:
            adj_expanded = adj.unsqueeze(0).expand(batch_size * T, -1, -1)
        else:
            # If adj is [batch, N, N], repeat for T
            adj_expanded = adj.repeat_interleave(T, dim=0)
            
        # Apply GraphSAGE
        # Output: [batch * T, N, hidden_features]
        spatial_emb_flat = self.graphsage(h_flat, adj_expanded)
        
        # Reshape back to separate time and batch
        # [batch, T, N, hidden_features]
        spatial_emb = spatial_emb_flat.view(batch_size, T, N, -1)
        
        # 2. Temporal Processing (Bi-GRU) applied to each node
        # We want to run GRU over T dimension for each node
        # Permute to: [batch * N, T, hidden_features]
        spatial_emb_per_node = spatial_emb.permute(0, 2, 1, 3).reshape(batch_size * N, T, -1)
        
        # Apply Bi-GRU
        # Output: [batch * N, T, gru_hidden * 2]
        gru_out, _ = self.bigru(spatial_emb_per_node)
        
        # Take the last time step output
        # [batch * N, gru_hidden * 2]
        last_gru_out = gru_out[:, -1, :]
        
        # Project to output dimension
        # [batch * N, hidden_features]
        node_temporal_emb = self.output_proj(last_gru_out)
        
        # Reshape back to [batch, N, hidden_features]
        node_temporal_emb = node_temporal_emb.view(batch_size, N, -1)
        
        # 3. Network Embedding (Pooling)
        # Mean pooling over all nodes to get single network vector
        # [batch, hidden_features]
        network_emb = node_temporal_emb.mean(dim=1)
        
        return network_emb
