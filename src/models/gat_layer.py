"""
Graph Attention Network (GAT) Layer for Intersection Embedding.

This module implements the GAT layer following the MGMQ architecture.
The GAT layer learns attention-weighted representations of **lanes (nodes)** within a single intersection by 
considering **conflicting and cooperating lanes** (edges) in the **intersection graph**.

Note:
- Nodes: 12 Standard Lanes (North-Left, North-Straight, North-Right, etc.)
- Edges: Defined by Conflict Matrix (FRAP logic) and Phase relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GATLayer(nn.Module):
    """
    Single-head Graph Attention Network Layer.
    
    This layer computes attention coefficients between nodes (lanes)
    and aggregates neighbor features using learned attention weights.
    
    Args:
        in_features: Number of input features per node (lane)
        out_features: Number of output features per node (lane)
        dropout: Dropout rate for attention coefficients
        alpha: Negative slope for LeakyReLU activation
        concat: If True, apply ELU activation to output
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable weight matrix W for feature transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        # a = [a_left || a_right] where || denotes concatenation
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(
        self, 
        h: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of GAT layer.
        
        Args:
            h: Node features tensor of shape [N, in_features] or [batch, N, in_features]
               (N=12 for standard intersection lanes)
            adj: Adjacency matrix of shape [N, N] or [batch, N, N]
            
        Returns:
            Updated node features of shape [N, out_features] or [batch, N, out_features]
        """
        # Handle batch dimension
        if h.dim() == 2:
            h = h.unsqueeze(0)
            adj = adj.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, N, _ = h.size()
        
        # Linear transformation: h' = h * W
        # Shape: [batch, N, out_features]
        Wh = torch.matmul(h, self.W) # [batch, N, out_features]. Do matmul chi nhan 2 chieu cuoi
        #===============================
        # Giải thích một chút cho bước trên:
        # Mục đích của phép nhân h*W là để biến đổi đặc trưng của các nút (lanes) từ không gian đặc trưng ban đầu (in_features)
        # sang không gian đặc trưng mới (out_features) thông qua ma trận trọng số W.
        # Cụ thể:
        # - h chứa thông tn ban đầu
        # - W là cách ta "nhìn" thông tin đó 
        # - hW tạo ra biểu diễn mới, hữu ích hơn cho việc học các mối quan hệ giữa các nút trong đồ thị.
        #===============================
        
        # Compute attention coefficients
        # e_ij = LeakyReLU(a^T * [Wh_i || Wh_j])
        
        # Self-attention coefficients for all pairs
        # Shape: [batch, N, 1, out_features] and [batch, 1, N, out_features]
        Wh1 = Wh.unsqueeze(2)  # [batch, N, 1, out_features]
        Wh2 = Wh.unsqueeze(1)  # [batch, 1, N, out_features]
        
        # Concatenate: [batch, N, N, 2*out_features]
        all_combinations = torch.cat([
            Wh1.repeat(1, 1, N, 1),
            Wh2.repeat(1, N, 1, 1)
        ], dim=-1)
        
        # Compute attention logits: [batch, N, N]
        e = self.leakyrelu(torch.matmul(all_combinations, self.a).squeeze(-1))
        
        # Mask attention coefficients for non-adjacent nodes
        # Use a large negative value for masked positions
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize attention coefficients with softmax
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # Apply attention to get updated features: h_out = attention * Wh
        # Shape: [batch, N, out_features]
        h_out = torch.bmm(attention, Wh)
        
        # Apply activation
        if self.concat:
            h_out = F.elu(h_out)
            
        if squeeze_output:
            h_out = h_out.squeeze(0)
            
        return h_out
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, dropout={self.dropout}'


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Network Layer.
    
    This layer applies multiple independent GAT attention heads and
    concatenates (or averages) their outputs.
    
    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node per head
        n_heads: Number of attention heads
        dropout: Dropout rate
        alpha: Negative slope for LeakyReLU
        concat: If True, concatenate heads; if False, average them
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(MultiHeadGATLayer, self).__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.out_features = out_features
        
        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            GATLayer(in_features, out_features, dropout, alpha, concat=True)
            for _ in range(n_heads)
        ])
        
    def forward(
        self, 
        h: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with multiple attention heads.
        
        Args:
            h: Node features [N, in_features] or [batch, N, in_features]
            adj: Adjacency matrix [N, N] or [batch, N, N]
            
        Returns:
            Multi-head attention output
            - If concat=True: [N, n_heads * out_features]
            - If concat=False: [N, out_features]
        """
        # Apply each attention head
        head_outputs = [head(h, adj) for head in self.attention_heads]
        
        if self.concat:
            # Concatenate head outputs along feature dimension
            return torch.cat(head_outputs, dim=-1)
        else:
            # Average head outputs
            return torch.mean(torch.stack(head_outputs), dim=0)
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of this layer."""
        if self.concat:
            return self.n_heads * self.out_features
        return self.out_features


def get_lane_conflict_matrix() -> torch.Tensor:
    """
    Returns a static 12x12 adjacency matrix representing lane conflicts.
    
    Nodes are ordered by direction (N, E, S, W) and turn (Left, Through, Right):
    0: NL, 1: NT, 2: NR
    3: EL, 4: ET, 5: ER
    6: SL, 7: ST, 8: SR
    9: WL, 10: WT, 11: WR
    
    A value of 1 indicates the lanes conflict (cannot move simultaneously).
    """
    # Initialize 12x12 matrix
    adj = torch.zeros((12, 12))
    
    # Define conflicts (undirected edges)
    # Format: (lane_idx_1, lane_idx_2)
    conflicts = [
        # Through vs Crossing Through
        (1, 4), (1, 10),  # NT vs ET, WT
        (7, 4), (7, 10),  # ST vs ET, WT
        
        # Through vs Crossing Left
        (1, 3), (1, 9),   # NT vs EL, WL
        (7, 3), (7, 9),   # ST vs EL, WL
        (4, 0), (4, 6),   # ET vs NL, SL
        (10, 0), (10, 6), # WT vs NL, SL
        
        # Left vs Opposing Through
        (0, 7),           # NL vs ST
        (6, 1),           # SL vs NT
        (3, 10),          # EL vs WT
        (9, 4),           # WL vs ET
        
        # Left vs Crossing Left (optional, depending on geometry, usually conflicting)
        (0, 3), (0, 9),   # NL vs EL, WL
        (6, 3), (6, 9),   # SL vs EL, WL
    ]
    
    # Fill matrix (symmetric)
    for i, j in conflicts:
        adj[i, j] = 1
        adj[j, i] = 1
        
    # Add self-loops
    adj = adj + torch.eye(12)
    
    return adj


def get_lane_cooperation_matrix() -> torch.Tensor:
    """
    Returns a static 12x12 adjacency matrix representing lane cooperation.
    Lanes are connected if they belong to the same standard signal phase.
    
    Nodes are ordered by direction (N, E, S, W) and turn (Left, Through, Right):
    0: NL, 1: NT, 2: NR
    3: EL, 4: ET, 5: ER
    6: SL, 7: ST, 8: SR
    9: WL, 10: WT, 11: WR
    """
    # Initialize 12x12 matrix
    adj = torch.zeros((12, 12))
    
    # Define phase groups (lanes that move together)
    # Standard 4-phase NEMA pattern
    phases = [
        [0, 6],             # Phase 1: NS Left (NL, SL)
        [1, 7, 2, 8],       # Phase 2: NS Through + Right (NT, ST, NR, SR)
        [3, 9],             # Phase 3: EW Left (EL, WL)
        [4, 10, 5, 11],     # Phase 4: EW Through + Right (ET, WT, ER, WR)
    ]
    
    # Connect all lanes within the same phase
    for phase_lanes in phases:
        for i in phase_lanes:
            for j in phase_lanes:
                if i != j:
                    adj[i, j] = 1
                    
    # Add self-loops
    adj = adj + torch.eye(12)
    
    return adj

