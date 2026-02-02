"""
Test suite cho các lớp Graph Attention Network (GAT).

File này cung cấp các test cases để kiểm tra và trực quan hóa:
1. GATLayer - Single-head GAT
2. MultiHeadGATLayer - Multi-head GAT  
3. DualStreamGATLayer - Dual-stream GAT theo MGMQ specification
4. Adjacency matrices (conflict & cooperation)

Chạy file này để thấy rõ đầu vào, đầu ra và đánh giá hoạt động của các lớp GAT.

Usage:
    cd /home/sondinh2k3/Documents/ITS_VTS_Working/MGMQ_v8_oke
    python -m pytest tests/test_gat_layer.py -v -s
    
    Hoặc chạy trực tiếp:
    python tests/test_gat_layer.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
from typing import Dict, Tuple

from src.models.gat_layer import (
    GATLayer,
    MultiHeadGATLayer, 
    DualStreamGATLayer,
    get_lane_conflict_matrix,
    get_lane_cooperation_matrix
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


def visualize_adjacency_matrix(adj: torch.Tensor, name: str):
    """Trực quan hóa adjacency matrix."""
    lane_names = [
        "NL", "NT", "NR",  # North lanes
        "EL", "ET", "ER",  # East lanes
        "SL", "ST", "SR",  # South lanes
        "WL", "WT", "WR"   # West lanes
    ]
    
    print(f"\n📐 {name} (12 x 12):")
    print("    " + "  ".join(f"{n:>3}" for n in lane_names))
    print("    " + "-" * 48)
    
    adj_np = adj.numpy()
    for i, lane in enumerate(lane_names):
        row = [f"{int(adj_np[i, j]):>3}" for j in range(12)]
        print(f"{lane:>3} |" + "  ".join(row))


def create_sample_intersection_features(batch_size: int = 2, n_lanes: int = 12, n_features: int = 5) -> torch.Tensor:
    """
    Tạo dữ liệu mẫu mô phỏng đặc trưng của 12 làn xe tại một giao lộ.
    
    Mỗi làn có 5 đặc trưng:
    - queue_length: Số xe đang chờ (0-20)
    - density: Mật độ xe (0-1)
    - mean_speed: Tốc độ trung bình (0-15 m/s)
    - waiting_time: Thời gian chờ trung bình (0-60s)
    - phase_one_hot: One-hot encoding của pha hiện tại (0 hoặc 1)
    
    Returns:
        Tensor shape [batch_size, n_lanes, n_features]
    """
    torch.manual_seed(42)  # Để kết quả có thể tái lập
    
    features = torch.zeros(batch_size, n_lanes, n_features)
    
    for b in range(batch_size):
        for lane in range(n_lanes):
            # Simulated traffic features
            features[b, lane, 0] = torch.rand(1) * 20  # queue_length: 0-20
            features[b, lane, 1] = torch.rand(1)       # density: 0-1
            features[b, lane, 2] = torch.rand(1) * 15  # mean_speed: 0-15 m/s
            features[b, lane, 3] = torch.rand(1) * 60  # waiting_time: 0-60s
            features[b, lane, 4] = torch.randint(0, 2, (1,)).float()  # phase indicator
    
    return features


# ==========================================
# TEST ADJACENCY MATRICES
# ==========================================

class TestAdjacencyMatrices:
    """Test các ma trận kề (conflict và cooperation)."""
    
    def test_conflict_matrix_shape_and_symmetry(self):
        """Test shape và tính đối xứng của conflict matrix."""
        print_separator("TEST: Conflict Matrix Shape & Symmetry")
        
        adj = get_lane_conflict_matrix()
        
        print_tensor_info("Conflict Matrix", adj)
        
        # Kiểm tra shape
        assert adj.shape == (12, 12), f"Expected shape (12, 12), got {adj.shape}"
        print("✅ Shape đúng: (12, 12)")
        
        # Kiểm tra đối xứng
        is_symmetric = torch.allclose(adj, adj.T)
        assert is_symmetric, "Matrix is not symmetric"
        print("✅ Ma trận đối xứng")
        
        # Kiểm tra self-loops
        diagonal = torch.diag(adj)
        assert torch.all(diagonal == 1), "Self-loops missing"
        print("✅ Tất cả self-loops đều có")
        
        visualize_adjacency_matrix(adj, "Lane Conflict Matrix")
        
    def test_cooperation_matrix_shape_and_symmetry(self):
        """Test shape và tính đối xứng của cooperation matrix."""
        print_separator("TEST: Cooperation Matrix Shape & Symmetry")
        
        adj = get_lane_cooperation_matrix()
        
        print_tensor_info("Cooperation Matrix", adj)
        
        # Kiểm tra shape
        assert adj.shape == (12, 12), f"Expected shape (12, 12), got {adj.shape}"
        print("✅ Shape đúng: (12, 12)")
        
        # Kiểm tra đối xứng
        is_symmetric = torch.allclose(adj, adj.T)
        assert is_symmetric, "Matrix is not symmetric"
        print("✅ Ma trận đối xứng")
        
        visualize_adjacency_matrix(adj, "Lane Cooperation Matrix")
        
    def test_conflict_cooperation_complementary(self):
        """Test rằng conflict và cooperation matrices bổ sung cho nhau."""
        print_separator("TEST: Conflict & Cooperation Complementary")
        
        conflict = get_lane_conflict_matrix()
        cooperation = get_lane_cooperation_matrix()
        
        # Các cặp làn conflict KHÔNG nên cooperate (trừ self-loops)
        # Lấy các cặp conflict (không kể self-loops)
        conflict_pairs = []
        coop_pairs = []
        
        for i in range(12):
            for j in range(i+1, 12):
                if conflict[i, j] == 1:
                    conflict_pairs.append((i, j))
                if cooperation[i, j] == 1:
                    coop_pairs.append((i, j))
        
        print(f"\n📊 Số cặp conflict (không kể self-loops): {len(conflict_pairs)}")
        print(f"📊 Số cặp cooperation (không kể self-loops): {len(coop_pairs)}")
        
        # In một số conflict pairs
        print("\n🔴 Một số cặp CONFLICT:")
        lane_names = ["NL","NT","NR","EL","ET","ER","SL","ST","SR","WL","WT","WR"]
        for i, j in conflict_pairs[:5]:
            print(f"   {lane_names[i]} <-> {lane_names[j]}")
            
        print("\n🟢 Một số cặp COOPERATION:")
        for i, j in coop_pairs[:5]:
            print(f"   {lane_names[i]} <-> {lane_names[j]}")


# ==========================================
# TEST GATLayer (Single-Head)
# ==========================================

class TestGATLayer:
    """Test lớp GATLayer (single-head attention)."""
    
    def test_forward_pass_2d_input(self):
        """Test forward pass với input 2D [N, in_features]."""
        print_separator("TEST: GATLayer Forward Pass (2D Input)")
        
        in_features = 5
        out_features = 16
        n_lanes = 12
        
        # Tạo layer
        layer = GATLayer(in_features, out_features, dropout=0.0)
        layer.eval()
        
        print(f"\n📋 Cấu hình GATLayer:")
        print(f"   in_features: {in_features}")
        print(f"   out_features: {out_features}")
        print(f"   dropout: 0.0")
        
        # Tạo input
        x = create_sample_intersection_features(1, n_lanes, in_features).squeeze(0)
        adj = get_lane_conflict_matrix()
        
        print_tensor_info("Input Features (x)", x)
        print_tensor_info("Adjacency Matrix (adj)", adj)
        
        # Forward pass
        with torch.no_grad():
            output = layer(x, adj)
        
        print_tensor_info("Output Features", output)
        
        # Assertions
        assert output.shape == (n_lanes, out_features), f"Expected shape ({n_lanes}, {out_features}), got {output.shape}"
        print(f"\n✅ Output shape đúng: {output.shape}")
        
        # Kiểm tra gradient flow
        layer.train()
        x_grad = x.clone().requires_grad_(True)
        output_grad = layer(x_grad, adj)
        loss = output_grad.sum()
        loss.backward()
        
        assert x_grad.grad is not None, "Gradient không được tính"
        print("✅ Gradient flow hoạt động")
        
    def test_forward_pass_3d_input(self):
        """Test forward pass với input 3D [batch, N, in_features]."""
        print_separator("TEST: GATLayer Forward Pass (3D Input - Batched)")
        
        in_features = 5
        out_features = 16
        n_lanes = 12
        batch_size = 4
        
        layer = GATLayer(in_features, out_features, dropout=0.0)
        layer.eval()
        
        print(f"\n📋 Cấu hình:")
        print(f"   batch_size: {batch_size}")
        print(f"   n_lanes: {n_lanes}")
        print(f"   in_features: {in_features}")
        print(f"   out_features: {out_features}")
        
        # Tạo batched input
        x = create_sample_intersection_features(batch_size, n_lanes, in_features)
        adj = get_lane_conflict_matrix().unsqueeze(0).expand(batch_size, -1, -1)
        
        print_tensor_info("Input Features (batched)", x)
        print_tensor_info("Adjacency Matrix (batched)", adj)
        
        with torch.no_grad():
            output = layer(x, adj)
        
        print_tensor_info("Output Features (batched)", output)
        
        expected_shape = (batch_size, n_lanes, out_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"\n✅ Output shape đúng: {output.shape}")
        
    def test_attention_weights(self):
        """Test attention weights được tính toán đúng."""
        print_separator("TEST: GATLayer Attention Weights Analysis")
        
        in_features = 5
        out_features = 16
        n_lanes = 12
        
        layer = GATLayer(in_features, out_features, dropout=0.0)
        layer.eval()
        
        x = create_sample_intersection_features(1, n_lanes, in_features).squeeze(0)
        adj = get_lane_conflict_matrix()
        
        # Manually compute attention for analysis
        with torch.no_grad():
            x_unsq = x.unsqueeze(0)  # [1, N, in_features]
            adj_unsq = adj.unsqueeze(0)  # [1, N, N]
            
            Wh = torch.matmul(x_unsq, layer.W)  # [1, N, out_features]
            
            N = n_lanes
            Wh1 = Wh.unsqueeze(2)  # [1, N, 1, out_features]
            Wh2 = Wh.unsqueeze(1)  # [1, 1, N, out_features]
            
            all_combinations = torch.cat([
                Wh1.repeat(1, 1, N, 1),
                Wh2.repeat(1, N, 1, 1)
            ], dim=-1)  # [1, N, N, 2*out_features]
            
            e = layer.leakyrelu(torch.matmul(all_combinations, layer.a).squeeze(-1))
            
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_unsq > 0, e, zero_vec)
            attention = torch.nn.functional.softmax(attention, dim=-1)
            
        attention = attention.squeeze(0)  # [N, N]
        print_tensor_info("Attention Weights", attention)
        
        print("\n📊 Attention weights cho lane NT (index 1):")
        lane_names = ["NL","NT","NR","EL","ET","ER","SL","ST","SR","WL","WT","WR"]
        nt_attention = attention[1, :]
        for i, (name, att) in enumerate(zip(lane_names, nt_attention)):
            if adj[1, i] > 0:  # Chỉ in các connected lanes
                print(f"   {name}: {att.item():.4f}")


# ==========================================
# TEST MultiHeadGATLayer
# ==========================================

class TestMultiHeadGATLayer:
    """Test lớp MultiHeadGATLayer."""
    
    def test_multi_head_concat(self):
        """Test multi-head với concatenation."""
        print_separator("TEST: MultiHeadGATLayer (Concatenation)")
        
        in_features = 5
        out_features = 8
        n_heads = 4
        n_lanes = 12
        batch_size = 2
        
        layer = MultiHeadGATLayer(
            in_features=in_features,
            out_features=out_features,
            n_heads=n_heads,
            dropout=0.0,
            concat=True
        )
        layer.eval()
        
        print(f"\n📋 Cấu hình MultiHeadGATLayer:")
        print(f"   in_features: {in_features}")
        print(f"   out_features: {out_features}")
        print(f"   n_heads: {n_heads}")
        print(f"   concat: True")
        print(f"   Expected output_dim: {n_heads * out_features}")
        
        x = create_sample_intersection_features(batch_size, n_lanes, in_features)
        adj = get_lane_cooperation_matrix().unsqueeze(0).expand(batch_size, -1, -1)
        
        print_tensor_info("Input Features", x)
        
        with torch.no_grad():
            output = layer(x, adj)
        
        print_tensor_info("Output Features", output)
        
        expected_shape = (batch_size, n_lanes, n_heads * out_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert layer.output_dim == n_heads * out_features
        print(f"\n✅ Output shape đúng: {output.shape}")
        print(f"✅ output_dim property đúng: {layer.output_dim}")
        
    def test_multi_head_average(self):
        """Test multi-head với averaging."""
        print_separator("TEST: MultiHeadGATLayer (Averaging)")
        
        in_features = 5
        out_features = 16
        n_heads = 4
        n_lanes = 12
        
        layer = MultiHeadGATLayer(
            in_features=in_features,
            out_features=out_features,
            n_heads=n_heads,
            dropout=0.0,
            concat=False  # Average instead of concat
        )
        layer.eval()
        
        print(f"\n📋 Cấu hình MultiHeadGATLayer:")
        print(f"   in_features: {in_features}")
        print(f"   out_features: {out_features}")
        print(f"   n_heads: {n_heads}")
        print(f"   concat: False (averaging)")
        print(f"   Expected output_dim: {out_features}")
        
        x = create_sample_intersection_features(1, n_lanes, in_features).squeeze(0)
        adj = get_lane_cooperation_matrix()
        
        with torch.no_grad():
            output = layer(x, adj)
        
        print_tensor_info("Output Features", output)
        
        expected_shape = (n_lanes, out_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert layer.output_dim == out_features
        print(f"\n✅ Output shape đúng: {output.shape}")


# ==========================================
# TEST DualStreamGATLayer
# ==========================================

class TestDualStreamGATLayer:
    """Test lớp DualStreamGATLayer - Core của MGMQ."""
    
    def test_dual_stream_forward(self):
        """Test forward pass của DualStreamGATLayer."""
        print_separator("TEST: DualStreamGATLayer Forward Pass")
        
        in_features = 5
        hidden_dim = 32
        out_features = 8
        n_heads = 4
        n_lanes = 12
        batch_size = 2
        
        layer = DualStreamGATLayer(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=out_features,
            n_heads=n_heads,
            dropout=0.0
        )
        layer.eval()
        
        print(f"\n📋 Cấu hình DualStreamGATLayer:")
        print(f"   in_features (F): {in_features}")
        print(f"   hidden_dim (F'): {hidden_dim}")
        print(f"   out_features per head (D): {out_features}")
        print(f"   n_heads (K): {n_heads}")
        print(f"   Final output_dim: {layer.final_output_dim}")
        
        # Tạo inputs
        x = create_sample_intersection_features(batch_size, n_lanes, in_features)
        adj_same = get_lane_cooperation_matrix().unsqueeze(0).expand(batch_size, -1, -1)
        adj_diff = get_lane_conflict_matrix().unsqueeze(0).expand(batch_size, -1, -1)
        
        print("\n📥 INPUTS:")
        print_tensor_info("Raw Features (x)", x)
        print_tensor_info("Same-phase Adjacency (cooperation)", adj_same)
        print_tensor_info("Diff-phase Adjacency (conflict)", adj_diff)
        
        # Forward pass
        with torch.no_grad():
            output = layer(x, adj_same, adj_diff)
        
        print("\n📤 OUTPUT:")
        print_tensor_info("Final Embedding", output)
        
        expected_shape = (batch_size, n_lanes, n_heads * out_features)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"\n✅ Output shape đúng: {output.shape}")
        
        # Kiểm tra các sub-modules
        print("\n📊 Kiểm tra các sub-modules:")
        print(f"   input_proj output dim: {hidden_dim}")
        print(f"   gat_same output dim: {layer.gat_same.output_dim}")
        print(f"   gat_diff output dim: {layer.gat_diff.output_dim}")
        print(f"   final_proj input dim: {hidden_dim + 2 * (n_heads * out_features)}")
        print(f"   final_proj output dim: {layer.final_output_dim}")
        
    def test_dual_stream_gradient_flow(self):
        """Test gradient flow qua DualStreamGATLayer."""
        print_separator("TEST: DualStreamGATLayer Gradient Flow")
        
        layer = DualStreamGATLayer(
            in_features=5,
            hidden_dim=32,
            out_features=8,
            n_heads=4,
            dropout=0.0
        )
        layer.train()
        
        batch_size = 2
        n_lanes = 12
        
        x = create_sample_intersection_features(batch_size, n_lanes, 5)
        x.requires_grad_(True)
        
        adj_same = get_lane_cooperation_matrix().unsqueeze(0).expand(batch_size, -1, -1)
        adj_diff = get_lane_conflict_matrix().unsqueeze(0).expand(batch_size, -1, -1)
        
        output = layer(x, adj_same, adj_diff)
        loss = output.sum()
        loss.backward()
        
        print(f"\n📊 Gradient Analysis:")
        print_tensor_info("Input Gradient (x.grad)", x.grad)
        
        # Check parameter gradients
        print("\n📊 Parameter Gradients:")
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"   {name}: norm = {grad_norm:.6f}")
        
        assert x.grad is not None, "Input gradient is None"
        print("\n✅ Gradient flow hoạt động đúng")
        
    def test_dual_stream_step_by_step(self):
        """Test từng bước của DualStreamGATLayer để hiểu rõ data flow."""
        print_separator("TEST: DualStreamGATLayer Step-by-Step Analysis")
        
        in_features = 5
        hidden_dim = 32
        out_features = 8
        n_heads = 4
        n_lanes = 12
        batch_size = 1
        
        layer = DualStreamGATLayer(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=out_features,
            n_heads=n_heads,
            dropout=0.0
        )
        layer.eval()
        
        x = create_sample_intersection_features(batch_size, n_lanes, in_features)
        adj_same = get_lane_cooperation_matrix().unsqueeze(0)
        adj_diff = get_lane_conflict_matrix().unsqueeze(0)
        
        print("\n" + "─" * 50)
        print("📌 STEP 0: Raw Input")
        print("─" * 50)
        print_tensor_info("x (raw features)", x)
        print(f"\n   Mỗi lane có {in_features} features:")
        print("   [queue_length, density, mean_speed, waiting_time, phase_indicator]")
        
        # Sample lane features
        print(f"\n   Sample lane NT (index 1):")
        print(f"   {x[0, 1, :].tolist()}")
        
        with torch.no_grad():
            # Step 1: Linear Transformation
            print("\n" + "─" * 50)
            print("📌 STEP 1: Linear Transformation")
            print("─" * 50)
            print("   h = LeakyReLU(W_init * x + b_init)")
            
            h = layer.input_proj(x)
            print_tensor_info("h (projected features)", h)
            
            # Step 2: Same-phase Attention
            print("\n" + "─" * 50)
            print("📌 STEP 2: Same-phase Attention (Cooperation)")
            print("─" * 50)
            print("   h_same = MultiHeadGAT(h, adj_cooperation)")
            
            h_same = layer.gat_same(h, adj_same)
            print_tensor_info("h_same", h_same)
            
            # Step 3: Diff-phase Attention
            print("\n" + "─" * 50)
            print("📌 STEP 3: Diff-phase Attention (Conflict)")
            print("─" * 50)
            print("   h_diff = MultiHeadGAT(h, adj_conflict)")
            
            h_diff = layer.gat_diff(h, adj_diff)
            print_tensor_info("h_diff", h_diff)
            
            # Step 4: Concatenation & Final Projection
            print("\n" + "─" * 50)
            print("📌 STEP 4: Node State Update")
            print("─" * 50)
            print("   combined = [h || h_same || h_diff]")
            print("   h_out = ELU(W_final * combined + b_final)")
            
            combined = torch.cat([h, h_same, h_diff], dim=-1)
            print_tensor_info("combined", combined)
            
            h_out = layer.final_proj(combined)
            print_tensor_info("h_out (final embedding)", h_out)
        
        print("\n" + "─" * 50)
        print("📊 SUMMARY: Dimension Changes Through DualStreamGATLayer")
        print("─" * 50)
        print(f"   Input:     {x.shape} → ({batch_size}, {n_lanes}, {in_features})")
        print(f"   After Input Proj:  {h.shape} → ({batch_size}, {n_lanes}, {hidden_dim})")
        print(f"   h_same:    {h_same.shape} → ({batch_size}, {n_lanes}, {n_heads * out_features})")
        print(f"   h_diff:    {h_diff.shape} → ({batch_size}, {n_lanes}, {n_heads * out_features})")
        print(f"   Combined:  {combined.shape} → ({batch_size}, {n_lanes}, {hidden_dim + 2 * n_heads * out_features})")
        print(f"   Output:    {h_out.shape} → ({batch_size}, {n_lanes}, {layer.final_output_dim})")


# ==========================================
# TEST INTEGRATION
# ==========================================

class TestIntegration:
    """Test tích hợp và production-like scenarios."""
    
    def test_typical_mgmq_usage(self):
        """Test typical usage trong MGMQ architecture."""
        print_separator("TEST: Typical MGMQ Usage Scenario")
        
        # Configuration matching MGMQ defaults
        batch_size = 8
        n_intersections = 16  # Grid 4x4
        n_lanes = 12
        in_features = 5
        hidden_dim = 32
        out_features = 8
        n_heads = 4
        
        print(f"\n📋 MGMQ-typical Configuration:")
        print(f"   batch_size: {batch_size}")
        print(f"   n_intersections: {n_intersections}")
        print(f"   n_lanes: {n_lanes}")
        print(f"   in_features: {in_features}")
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   out_features: {out_features}")
        print(f"   n_heads: {n_heads}")
        
        # Create layer
        layer = DualStreamGATLayer(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=out_features,
            n_heads=n_heads,
            dropout=0.3
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in layer.parameters())
        trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Simulate processing all intersections
        all_embeddings = []
        
        adj_same = get_lane_cooperation_matrix()
        adj_diff = get_lane_conflict_matrix()
        
        layer.eval()
        with torch.no_grad():
            for ts_idx in range(n_intersections):
                # Features for one intersection (batched)
                x = create_sample_intersection_features(batch_size, n_lanes, in_features)
                
                adj_same_batch = adj_same.unsqueeze(0).expand(batch_size, -1, -1)
                adj_diff_batch = adj_diff.unsqueeze(0).expand(batch_size, -1, -1)
                
                embedding = layer(x, adj_same_batch, adj_diff_batch)
                all_embeddings.append(embedding)
        
        # Stack all embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)  # [batch, n_intersections, n_lanes, embedding_dim]
        
        print_tensor_info("All Intersection Embeddings", all_embeddings)
        
        expected_shape = (batch_size, n_intersections, n_lanes, n_heads * out_features)
        assert all_embeddings.shape == expected_shape, f"Expected {expected_shape}, got {all_embeddings.shape}"
        print(f"\n✅ Final shape đúng cho input GraphSAGE: {all_embeddings.shape}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_all_tests():
    """Chạy tất cả các tests với output chi tiết."""
    print("\n" + "🔬" * 35)
    print(" COMPREHENSIVE GAT LAYER TEST SUITE")
    print("🔬" * 35)
    
    # Test Adjacency Matrices
    adj_tests = TestAdjacencyMatrices()
    adj_tests.test_conflict_matrix_shape_and_symmetry()
    adj_tests.test_cooperation_matrix_shape_and_symmetry()
    adj_tests.test_conflict_cooperation_complementary()
    
    # Test GATLayer
    gat_tests = TestGATLayer()
    gat_tests.test_forward_pass_2d_input()
    gat_tests.test_forward_pass_3d_input()
    gat_tests.test_attention_weights()
    
    # Test MultiHeadGATLayer
    mh_tests = TestMultiHeadGATLayer()
    mh_tests.test_multi_head_concat()
    mh_tests.test_multi_head_average()
    
    # Test DualStreamGATLayer
    ds_tests = TestDualStreamGATLayer()
    ds_tests.test_dual_stream_forward()
    ds_tests.test_dual_stream_gradient_flow()
    ds_tests.test_dual_stream_step_by_step()
    
    # Integration Tests
    int_tests = TestIntegration()
    int_tests.test_typical_mgmq_usage()
    
    print("\n" + "=" * 70)
    print(" ✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GAT Layer modules")
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
