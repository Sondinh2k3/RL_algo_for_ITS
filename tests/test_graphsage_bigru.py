"""
Test suite cho các lớp GraphSAGE + Bi-GRU.

File này cung cấp các test cases để kiểm tra và trực quan hóa:
1. DirectionalGraphSAGE - GraphSAGE với BiGRU tổng hợp theo 4 hướng không gian
2. GraphSAGE_BiGRU - Wrapper đơn giản hóa cho DirectionalGraphSAGE
3. NeighborGraphSAGE_BiGRU - GraphSAGE cho topology star-graph

Chạy file này để thấy rõ đầu vào, đầu ra và đánh giá hoạt động của các lớp.

Usage:
    cd /home/sondinh2k3/Documents/ITS_VTS_Working/MGMQ_v8_oke
    source .venv/bin/activate && python tests/test_graphsage_bigru.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.models.graphsage_bigru import (
    DirectionalGraphSAGE,
    GraphSAGE_BiGRU,
    NeighborGraphSAGE_BiGRU
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def print_separator(title: str):
    """In một dòng phân cách với tiêu đề."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_tensor_info(name: str, tensor: torch.Tensor):
    """In thông tin chi tiết của tensor."""
    print(f"\n📊 {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Device: {tensor.device}")
    print(f"   Min: {tensor.min().item():.6f}")
    print(f"   Max: {tensor.max().item():.6f}")
    print(f"   Mean: {tensor.mean().item():.6f}")
    print(f"   Std: {tensor.std().item():.6f}")


def create_grid_adjacency(grid_size: int = 4) -> torch.Tensor:
    """
    Tạo directional adjacency matrix cho mạng lưới grid.
    
    Với grid 4x4, có 16 nodes được đánh số từ trái qua phải, trên xuống dưới:
    
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    
    Returns:
        adj_directions: [4, N, N] - 4 ma trận kề cho 4 hướng (N, E, S, W)
    """
    num_nodes = grid_size * grid_size
    
    # 4 directional matrices: [North, East, South, West]
    adj_north = torch.zeros(num_nodes, num_nodes)
    adj_east = torch.zeros(num_nodes, num_nodes)
    adj_south = torch.zeros(num_nodes, num_nodes)
    adj_west = torch.zeros(num_nodes, num_nodes)
    
    for i in range(grid_size):
        for j in range(grid_size):
            node_idx = i * grid_size + j
            
            # North neighbor (row - 1)
            if i > 0:
                neighbor_idx = (i - 1) * grid_size + j
                adj_north[node_idx, neighbor_idx] = 1.0
                
            # East neighbor (col + 1)
            if j < grid_size - 1:
                neighbor_idx = i * grid_size + (j + 1)
                adj_east[node_idx, neighbor_idx] = 1.0
                
            # South neighbor (row + 1)
            if i < grid_size - 1:
                neighbor_idx = (i + 1) * grid_size + j
                adj_south[node_idx, neighbor_idx] = 1.0
                
            # West neighbor (col - 1)
            if j > 0:
                neighbor_idx = i * grid_size + (j - 1)
                adj_west[node_idx, neighbor_idx] = 1.0
    
    adj_directions = torch.stack([adj_north, adj_east, adj_south, adj_west], dim=0)
    return adj_directions


def create_sample_intersection_embeddings(
    batch_size: int = 2, 
    num_intersections: int = 16, 
    embedding_dim: int = 32
) -> torch.Tensor:
    """
    Tạo embeddings mẫu cho các giao lộ (sau khi qua DualStreamGATLayer).
    
    Trong MGMQ pipeline:
    - GAT aggregates lane features per intersection -> [batch, n_intersections, n_lanes, gat_output_dim]
    - Lane pooling (mean) -> [batch, n_intersections, gat_output_dim]
    - GraphSAGE receives pooled intersection embeddings
    
    Returns:
        embeddings: [batch_size, num_intersections, embedding_dim]
    """
    torch.manual_seed(42)
    return torch.randn(batch_size, num_intersections, embedding_dim)


def visualize_grid_adjacency(adj_directions: torch.Tensor, grid_size: int = 4):
    """Trực quan hóa directional adjacency cho grid."""
    direction_names = ["North", "East", "South", "West"]
    
    print(f"\n📐 Directional Adjacency Matrices (Grid {grid_size}x{grid_size}):")
    
    for d, name in enumerate(direction_names):
        adj = adj_directions[d]
        num_connections = int(adj.sum().item())
        print(f"\n   {name}: {num_connections} connections")
        
        # Show sample connections
        connections = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] > 0:
                    connections.append(f"{i}→{j}")
                    if len(connections) >= 5:
                        break
            if len(connections) >= 5:
                break
        print(f"   Sample: {', '.join(connections)} ...")


# ==========================================
# TEST DirectionalGraphSAGE
# ==========================================

class TestDirectionalGraphSAGE:
    """Test lớp DirectionalGraphSAGE."""
    
    def test_forward_pass_basic(self):
        """Test forward pass cơ bản."""
        print_separator("TEST: DirectionalGraphSAGE Forward Pass")
        
        in_features = 32
        hidden_features = 16
        out_features = 64
        num_nodes = 16  # Grid 4x4
        batch_size = 2
        
        layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            dropout=0.0
        )
        layer.eval()
        
        print(f"\n📋 Cấu hình DirectionalGraphSAGE:")
        print(f"   in_features: {in_features}")
        print(f"   hidden_features: {hidden_features}")
        print(f"   out_features: {out_features}")
        print(f"   dropout: 0.0")
        
        # Create inputs
        h = create_sample_intersection_embeddings(batch_size, num_nodes, in_features)
        adj_directions = create_grid_adjacency(4)
        
        # Expand adj for batch
        adj_directions_batch = adj_directions.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        print("\n📥 INPUTS:")
        print_tensor_info("Node Features (h)", h)
        print_tensor_info("Directional Adjacency (adj_directions)", adj_directions_batch)
        
        visualize_grid_adjacency(adj_directions, grid_size=4)
        
        # Forward pass
        with torch.no_grad():
            output = layer(h, adj_directions_batch)
        
        print("\n📤 OUTPUT:")
        print_tensor_info("Node Embeddings", output)
        
        expected_shape = (batch_size, num_nodes, out_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"\n✅ Output shape đúng: {output.shape}")
        
    def test_forward_pass_2d_input(self):
        """Test forward pass với input 2D (không có batch dimension)."""
        print_separator("TEST: DirectionalGraphSAGE Forward Pass (2D Input)")
        
        in_features = 32
        hidden_features = 16
        out_features = 64
        num_nodes = 16
        
        layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            dropout=0.0
        )
        layer.eval()
        
        h = create_sample_intersection_embeddings(1, num_nodes, in_features).squeeze(0)
        adj_directions = create_grid_adjacency(4)
        
        print_tensor_info("Input Features (2D)", h)
        print_tensor_info("Directional Adjacency (3D)", adj_directions)
        
        with torch.no_grad():
            output = layer(h, adj_directions)
        
        print_tensor_info("Output Features", output)
        
        expected_shape = (num_nodes, out_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"\n✅ Output shape đúng: {output.shape}")
        
    def test_step_by_step_analysis(self):
        """Phân tích từng bước xử lý của DirectionalGraphSAGE."""
        print_separator("TEST: DirectionalGraphSAGE Step-by-Step Analysis")
        
        in_features = 32
        hidden_features = 16
        out_features = 64
        num_nodes = 16
        batch_size = 1
        
        layer = DirectionalGraphSAGE(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            dropout=0.0
        )
        layer.eval()
        
        h = create_sample_intersection_embeddings(batch_size, num_nodes, in_features)
        adj_directions = create_grid_adjacency(4).unsqueeze(0)
        
        print("\n" + "─" * 50)
        print("📌 STEP 0: Input")
        print("─" * 50)
        print_tensor_info("h (intersection embeddings)", h)
        print(f"   16 intersections in 4x4 grid, each with {in_features} features")
        
        with torch.no_grad():
            # Extract masks
            mask_north = adj_directions[:, 0, :, :]
            mask_east = adj_directions[:, 1, :, :]
            mask_south = adj_directions[:, 2, :, :]
            mask_west = adj_directions[:, 3, :, :]
            
            # Step 1: Directional Projections
            print("\n" + "─" * 50)
            print("📌 STEP 1: Directional Projections")
            print("─" * 50)
            print("   g_self = ReLU(Linear(h))")
            print("   g_north = ReLU(Linear(h))")
            print("   g_east = ReLU(Linear(h))")
            print("   g_south = ReLU(Linear(h))")
            print("   g_west = ReLU(Linear(h))")
            
            import torch.nn.functional as F
            g_self = F.relu(layer.proj_self(h))
            g_north = F.relu(layer.proj_north(h))
            g_east = F.relu(layer.proj_east(h))
            g_south = F.relu(layer.proj_south(h))
            g_west = F.relu(layer.proj_west(h))
            
            print_tensor_info("g_self", g_self)
            print_tensor_info("g_north (sample)", g_north)
            
            # Step 2: Topology-aware Neighbor Exchange
            print("\n" + "─" * 50)
            print("📌 STEP 2: Topology-aware Neighbor Exchange")
            print("─" * 50)
            print("   in_north = adj_north @ g_south (receive from south)")
            print("   in_east  = adj_east @ g_west  (receive from west)")
            print("   in_south = adj_south @ g_north (receive from north)")
            print("   in_west  = adj_west @ g_east  (receive from east)")
            
            in_north = torch.bmm(mask_north, g_south)
            in_east = torch.bmm(mask_east, g_west)
            in_south = torch.bmm(mask_south, g_north)
            in_west = torch.bmm(mask_west, g_east)
            
            print_tensor_info("in_north (from south neighbors)", in_north)
            print_tensor_info("in_east (from west neighbors)", in_east)
            
            # Step 3: BiGRU Aggregation
            print("\n" + "─" * 50)
            print("📌 STEP 3: BiGRU Spatial Aggregation (over 4 directions)")
            print("─" * 50)
            print("   S_v = [in_north, in_east, in_south, in_west]")
            print("   BiGRU processes sequence: N → E → S → W (forward)")
            print("                            W → S → E → N (backward)")
            
            seq_tensor = torch.stack([in_north, in_east, in_south, in_west], dim=2)
            print_tensor_info("seq_tensor (stacked directions)", seq_tensor)
            
            seq_flat = seq_tensor.view(batch_size * num_nodes, 4, hidden_features)
            print_tensor_info("seq_flat (for BiGRU)", seq_flat)
            
            bigru_output, hidden = layer.bigru(seq_flat)
            print_tensor_info("BiGRU output", bigru_output)
            
            G_k = bigru_output.reshape(batch_size * num_nodes, -1)
            print_tensor_info("G_k (flattened BiGRU)", G_k)
            
            # Step 4: Combine & Output
            print("\n" + "─" * 50)
            print("📌 STEP 4: Combine Self + Neighbor Context")
            print("─" * 50)
            print("   z_raw = Concat(g_self, G_k)")
            print("   output = LeakyReLU(Linear(z_raw))")
            
            g_self_flat = g_self.view(batch_size * num_nodes, -1)
            z_raw = torch.cat([g_self_flat, G_k], dim=-1)
            print_tensor_info("z_raw (concat)", z_raw)
            
            out = layer.output_linear(z_raw)
            out = layer.activation(out)
            out = out.view(batch_size, num_nodes, -1)
            print_tensor_info("final output", out)
        
        # Summary
        print("\n" + "─" * 50)
        print("📊 SUMMARY: Dimension Changes Through DirectionalGraphSAGE")
        print("─" * 50)
        print(f"   Input h:          [{batch_size}, {num_nodes}, {in_features}]")
        print(f"   g_* (projected):  [{batch_size}, {num_nodes}, {hidden_features}]")
        print(f"   in_* (exchanged): [{batch_size}, {num_nodes}, {hidden_features}]")
        print(f"   seq_tensor:       [{batch_size}, {num_nodes}, 4, {hidden_features}]")
        print(f"   BiGRU output:     [{batch_size * num_nodes}, 4, {hidden_features * 2}]")
        print(f"   G_k:              [{batch_size * num_nodes}, {4 * hidden_features * 2}]")
        print(f"   z_raw:            [{batch_size * num_nodes}, {hidden_features + 4 * hidden_features * 2}]")
        print(f"   Output:           [{batch_size}, {num_nodes}, {out_features}]")
        
    def test_gradient_flow(self):
        """Test gradient flow qua DirectionalGraphSAGE."""
        print_separator("TEST: DirectionalGraphSAGE Gradient Flow")
        
        layer = DirectionalGraphSAGE(
            in_features=32,
            hidden_features=16,
            out_features=64,
            dropout=0.0
        )
        layer.train()
        
        batch_size = 2
        num_nodes = 16
        
        h = create_sample_intersection_embeddings(batch_size, num_nodes, 32)
        h.requires_grad_(True)
        adj_directions = create_grid_adjacency(4).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        output = layer(h, adj_directions)
        loss = output.sum()
        loss.backward()
        
        print_tensor_info("Input Gradient (h.grad)", h.grad)
        
        print("\n📊 Parameter Gradients:")
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"   {name}: norm = {grad_norm:.6f}")
        
        assert h.grad is not None, "Input gradient is None"
        print("\n✅ Gradient flow hoạt động đúng")


# ==========================================
# TEST GraphSAGE_BiGRU (Wrapper)
# ==========================================

class TestGraphSAGE_BiGRU:
    """Test lớp GraphSAGE_BiGRU wrapper."""
    
    def test_forward_pass(self):
        """Test forward pass của wrapper."""
        print_separator("TEST: GraphSAGE_BiGRU Wrapper Forward Pass")
        
        in_features = 32
        hidden_features = 64
        gru_hidden_size = 16
        num_nodes = 16
        batch_size = 2
        
        layer = GraphSAGE_BiGRU(
            in_features=in_features,
            hidden_features=hidden_features,
            gru_hidden_size=gru_hidden_size,
            dropout=0.0
        )
        layer.eval()
        
        print(f"\n📋 Cấu hình GraphSAGE_BiGRU:")
        print(f"   in_features: {in_features}")
        print(f"   hidden_features (output): {hidden_features}")
        print(f"   gru_hidden_size: {gru_hidden_size}")
        print(f"   output_dim property: {layer.output_dim}")
        
        h = create_sample_intersection_embeddings(batch_size, num_nodes, in_features)
        adj_directions = create_grid_adjacency(4).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        print_tensor_info("Input Features", h)
        
        with torch.no_grad():
            output = layer(h, adj_directions)
        
        print_tensor_info("Output Features", output)
        
        expected_shape = (batch_size, num_nodes, hidden_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert layer.output_dim == hidden_features
        print(f"\n✅ Output shape đúng: {output.shape}")
        print(f"✅ output_dim property đúng: {layer.output_dim}")


# ==========================================
# TEST NeighborGraphSAGE_BiGRU
# ==========================================

class TestNeighborGraphSAGE_BiGRU:
    """Test lớp NeighborGraphSAGE_BiGRU."""
    
    def test_forward_pass(self):
        """Test forward pass cơ bản."""
        print_separator("TEST: NeighborGraphSAGE_BiGRU Forward Pass")
        
        in_features = 32
        hidden_features = 64
        gru_hidden_size = 16
        max_neighbors = 4
        batch_size = 8
        
        layer = NeighborGraphSAGE_BiGRU(
            in_features=in_features,
            hidden_features=hidden_features,
            gru_hidden_size=gru_hidden_size,
            max_neighbors=max_neighbors,
            dropout=0.0
        )
        layer.eval()
        
        print(f"\n📋 Cấu hình NeighborGraphSAGE_BiGRU:")
        print(f"   in_features: {in_features}")
        print(f"   hidden_features: {hidden_features}")
        print(f"   gru_hidden_size: {gru_hidden_size}")
        print(f"   max_neighbors: {max_neighbors}")
        print(f"   output_dim: {layer.output_dim}")
        
        # Create inputs
        torch.manual_seed(42)
        self_features = torch.randn(batch_size, in_features)
        neighbor_features = torch.randn(batch_size, max_neighbors, in_features)
        
        # Create neighbor mask (some nodes may have fewer neighbors)
        neighbor_mask = torch.ones(batch_size, max_neighbors)
        neighbor_mask[0, 3] = 0  # Node 0 has only 3 neighbors
        neighbor_mask[1, 2:] = 0  # Node 1 has only 2 neighbors
        
        print("\n📥 INPUTS:")
        print_tensor_info("Self Features", self_features)
        print_tensor_info("Neighbor Features", neighbor_features)
        print_tensor_info("Neighbor Mask", neighbor_mask)
        
        print(f"\n   Sample masks:")
        print(f"   Node 0: {neighbor_mask[0].tolist()} (3 neighbors)")
        print(f"   Node 1: {neighbor_mask[1].tolist()} (2 neighbors)")
        print(f"   Node 2: {neighbor_mask[2].tolist()} (4 neighbors)")
        
        with torch.no_grad():
            output = layer(self_features, neighbor_features, neighbor_mask)
        
        print("\n📤 OUTPUT:")
        print_tensor_info("Aggregated Embedding", output)
        
        expected_shape = (batch_size, hidden_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"\n✅ Output shape đúng: {output.shape}")
        
    def test_step_by_step_analysis(self):
        """Phân tích từng bước xử lý của NeighborGraphSAGE_BiGRU."""
        print_separator("TEST: NeighborGraphSAGE_BiGRU Step-by-Step Analysis")
        
        in_features = 32
        hidden_features = 64
        gru_hidden_size = 16
        max_neighbors = 4
        batch_size = 2
        
        layer = NeighborGraphSAGE_BiGRU(
            in_features=in_features,
            hidden_features=hidden_features,
            gru_hidden_size=gru_hidden_size,
            max_neighbors=max_neighbors,
            dropout=0.0
        )
        layer.eval()
        
        # Create inputs
        torch.manual_seed(42)
        self_features = torch.randn(batch_size, in_features)
        neighbor_features = torch.randn(batch_size, max_neighbors, in_features)
        neighbor_mask = torch.ones(batch_size, max_neighbors)
        
        print("\n" + "─" * 50)
        print("📌 STEP 0: Input")
        print("─" * 50)
        print_tensor_info("self_features", self_features)
        print_tensor_info("neighbor_features", neighbor_features)
        
        with torch.no_grad():
            # Step 1: Project features
            print("\n" + "─" * 50)
            print("📌 STEP 1: Feature Projection")
            print("─" * 50)
            print("   h_self = ReLU(Linear(self_features))")
            print("   h_neighbors = ReLU(Linear(neighbor_features))")
            
            h_self = layer.self_proj(self_features)
            h_neighbors = layer.neighbor_proj(neighbor_features)
            
            print_tensor_info("h_self (projected)", h_self)
            print_tensor_info("h_neighbors (projected)", h_neighbors)
            
            # Step 2: Apply mask
            print("\n" + "─" * 50)
            print("📌 STEP 2: Apply Neighbor Mask")
            print("─" * 50)
            print("   h_neighbors_masked = h_neighbors * mask")
            
            mask_expanded = neighbor_mask.unsqueeze(-1).float()
            h_neighbors_masked = h_neighbors * mask_expanded
            
            print_tensor_info("h_neighbors_masked", h_neighbors_masked)
            
            # Step 3: BiGRU Aggregation
            print("\n" + "─" * 50)
            print("📌 STEP 3: BiGRU Spatial Aggregation (over K neighbors)")
            print("─" * 50)
            print("   BiGRU processes sequence: neighbor_1 → neighbor_2 → ... → neighbor_K")
            
            bigru_output, _ = layer.spatial_bigru(h_neighbors_masked)
            print_tensor_info("BiGRU output", bigru_output)
            
            G_k = bigru_output.reshape(bigru_output.size(0), -1)
            print_tensor_info("G_k (flattened)", G_k)
            
            # Step 4: Combine & Output
            print("\n" + "─" * 50)
            print("📌 STEP 4: Combine Self + Aggregated Neighbors")
            print("─" * 50)
            print("   z_raw = Concat(h_self, G_k)")
            print("   output = LeakyReLU(Linear(z_raw))")
            
            z_raw = torch.cat([h_self, G_k], dim=-1)
            print_tensor_info("z_raw (concat)", z_raw)
            
            output = layer.output_proj(z_raw)
            print_tensor_info("final output", output)
        
        # Summary
        print("\n" + "─" * 50)
        print("📊 SUMMARY: Dimension Changes Through NeighborGraphSAGE_BiGRU")
        print("─" * 50)
        print(f"   self_features:       [{batch_size}, {in_features}]")
        print(f"   neighbor_features:   [{batch_size}, {max_neighbors}, {in_features}]")
        print(f"   h_self:              [{batch_size}, {gru_hidden_size}]")
        print(f"   h_neighbors:         [{batch_size}, {max_neighbors}, {gru_hidden_size}]")
        print(f"   BiGRU output:        [{batch_size}, {max_neighbors}, {gru_hidden_size * 2}]")
        print(f"   G_k:                 [{batch_size}, {max_neighbors * gru_hidden_size * 2}]")
        print(f"   z_raw:               [{batch_size}, {gru_hidden_size + max_neighbors * gru_hidden_size * 2}]")
        print(f"   Output:              [{batch_size}, {hidden_features}]")
        
    def test_gradient_flow(self):
        """Test gradient flow qua NeighborGraphSAGE_BiGRU."""
        print_separator("TEST: NeighborGraphSAGE_BiGRU Gradient Flow")
        
        layer = NeighborGraphSAGE_BiGRU(
            in_features=32,
            hidden_features=64,
            gru_hidden_size=16,
            max_neighbors=4,
            dropout=0.0
        )
        layer.train()
        
        batch_size = 4
        self_features = torch.randn(batch_size, 32, requires_grad=True)
        neighbor_features = torch.randn(batch_size, 4, 32, requires_grad=True)
        neighbor_mask = torch.ones(batch_size, 4)
        
        output = layer(self_features, neighbor_features, neighbor_mask)
        loss = output.sum()
        loss.backward()
        
        print_tensor_info("Self Features Gradient", self_features.grad)
        print_tensor_info("Neighbor Features Gradient", neighbor_features.grad)
        
        print("\n📊 Parameter Gradients:")
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"   {name}: norm = {grad_norm:.6f}")
        
        assert self_features.grad is not None
        assert neighbor_features.grad is not None
        print("\n✅ Gradient flow hoạt động đúng")


# ==========================================
# TEST INTEGRATION
# ==========================================

class TestIntegration:
    """Test tích hợp và production-like scenarios."""
    
    def test_mgmq_pipeline_simulation(self):
        """Mô phỏng pipeline MGMQ: GAT output -> Lane Pooling -> GraphSAGE."""
        print_separator("TEST: MGMQ Pipeline Simulation (GAT -> Pool -> GraphSAGE)")
        
        batch_size = 4
        n_intersections = 16  # Grid 4x4
        n_lanes = 12
        gat_output_dim = 32
        graphsage_output_dim = 64
        gru_hidden_size = 16
        
        print(f"\n📋 MGMQ Pipeline Configuration:")
        print(f"   batch_size: {batch_size}")
        print(f"   n_intersections: {n_intersections}")
        print(f"   n_lanes: {n_lanes}")
        print(f"   gat_output_dim: {gat_output_dim}")
        print(f"   graphsage_output_dim: {graphsage_output_dim}")
        
        # Simulate GAT output: [batch, n_intersections, n_lanes, gat_output_dim]
        torch.manual_seed(42)
        gat_output = torch.randn(batch_size, n_intersections, n_lanes, gat_output_dim)
        
        print("\n📌 STEP 1: GAT Output (lane embeddings per intersection)")
        print_tensor_info("GAT Output", gat_output)
        
        # Lane pooling: [batch, n_intersections, n_lanes, dim] -> [batch, n_intersections, dim]
        pooled_features = gat_output.mean(dim=2)
        
        print("\n📌 STEP 2: Lane Pooling (mean over lanes)")
        print_tensor_info("Pooled Features", pooled_features)
        
        # GraphSAGE
        graphsage = GraphSAGE_BiGRU(
            in_features=gat_output_dim,
            hidden_features=graphsage_output_dim,
            gru_hidden_size=gru_hidden_size,
            dropout=0.0
        )
        graphsage.eval()
        
        adj_directions = create_grid_adjacency(4).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        print("\n📌 STEP 3: GraphSAGE Network Embedding")
        
        with torch.no_grad():
            network_embedding = graphsage(pooled_features, adj_directions)
        
        print_tensor_info("Network Embedding", network_embedding)
        
        expected_shape = (batch_size, n_intersections, graphsage_output_dim)
        assert network_embedding.shape == expected_shape
        
        print("\n📊 Pipeline Summary:")
        print(f"   GAT Output:        {gat_output.shape}")
        print(f"   After Pooling:     {pooled_features.shape}")
        print(f"   Network Embedding: {network_embedding.shape}")
        print("\n✅ Pipeline simulation successful!")
        
    def test_model_statistics(self):
        """Test và hiển thị thống kê model."""
        print_separator("TEST: Model Statistics")
        
        # DirectionalGraphSAGE
        model1 = DirectionalGraphSAGE(
            in_features=32,
            hidden_features=16,
            out_features=64,
            dropout=0.3
        )
        
        # NeighborGraphSAGE_BiGRU
        model2 = NeighborGraphSAGE_BiGRU(
            in_features=32,
            hidden_features=64,
            gru_hidden_size=16,
            max_neighbors=4,
            dropout=0.3
        )
        
        def count_params(model):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable
        
        print("\n📊 DirectionalGraphSAGE:")
        total1, trainable1 = count_params(model1)
        print(f"   Total parameters: {total1:,}")
        print(f"   Trainable parameters: {trainable1:,}")
        
        print("\n📊 NeighborGraphSAGE_BiGRU:")
        total2, trainable2 = count_params(model2)
        print(f"   Total parameters: {total2:,}")
        print(f"   Trainable parameters: {trainable2:,}")
        
        # Layer breakdown for DirectionalGraphSAGE
        print("\n📊 DirectionalGraphSAGE Layer Breakdown:")
        for name, param in model1.named_parameters():
            print(f"   {name}: {param.numel():,} params")


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_all_tests():
    """Chạy tất cả các tests với output chi tiết."""
    print("\n" + "🔬" * 35)
    print(" COMPREHENSIVE GRAPHSAGE + BiGRU TEST SUITE")
    print("🔬" * 35)
    
    # Test DirectionalGraphSAGE
    ds_tests = TestDirectionalGraphSAGE()
    ds_tests.test_forward_pass_basic()
    ds_tests.test_forward_pass_2d_input()
    ds_tests.test_step_by_step_analysis()
    ds_tests.test_gradient_flow()
    
    # Test GraphSAGE_BiGRU Wrapper
    wrapper_tests = TestGraphSAGE_BiGRU()
    wrapper_tests.test_forward_pass()
    
    # Test NeighborGraphSAGE_BiGRU
    neighbor_tests = TestNeighborGraphSAGE_BiGRU()
    neighbor_tests.test_forward_pass()
    neighbor_tests.test_step_by_step_analysis()
    neighbor_tests.test_gradient_flow()
    
    # Integration Tests
    int_tests = TestIntegration()
    int_tests.test_mgmq_pipeline_simulation()
    int_tests.test_model_statistics()
    
    print("\n" + "=" * 70)
    print(" ✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GraphSAGE + BiGRU modules")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--pytest", action="store_true", help="Run with pytest")
    
    args = parser.parse_args()
    
    if args.pytest:
        if HAS_PYTEST:
            pytest.main([__file__, "-v", "-s"])
        else:
            print("pytest is not installed. Running tests directly...")
            run_all_tests()
    else:
        run_all_tests()
