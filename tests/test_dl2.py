import torch
from src.models.gat_layer import MultiHeadGATLayer, get_lane_conflict_matrix

def print_tensor(name, tensor):
    print(f"{name}: shape={tensor.shape}\n{tensor}\n")

def test_multihead_gatlayer_step_by_step():
    # 1. Khởi tạo MultiHeadGATLayer
    in_features = 4
    out_features = 3
    n_heads = 2
    num_nodes = 5
    torch.manual_seed(42)
    gat = MultiHeadGATLayer(in_features, out_features, n_heads=n_heads, dropout=0.0, alpha=0.2, concat=True)

    # 2. Tạo dữ liệu đầu vào: node features và adjacency matrix
    h = torch.randn(num_nodes, in_features)  # [N, in_features]
    adj = torch.randint(0, 2, (num_nodes, num_nodes))  # [N, N], random 0/1 adjacency

    print_tensor("Input node features h", h)
    print_tensor("Adjacency matrix adj", adj)

    # 3. Chạy MultiHeadGATLayer
    out = gat(h, adj)
    print_tensor("Output of MultiHeadGATLayer", out)

    # 4. Kiểm tra shape đầu ra
    expected_shape = (num_nodes, n_heads * out_features)
    print(f"Expected output shape: {expected_shape}")
    assert out.shape == expected_shape, f"Output shape mismatch: {out.shape} != {expected_shape}"

    # 5. Nếu concat=False, kiểm tra shape trung bình
    gat_avg = MultiHeadGATLayer(in_features, out_features, n_heads=n_heads, dropout=0.0, alpha=0.2, concat=False)
    out_avg = gat_avg(h, adj)
    print_tensor("Output of MultiHeadGATLayer (concat=False, mean)", out_avg)
    expected_shape_avg = (num_nodes, out_features)
    print(f"Expected output shape (mean): {expected_shape_avg}")
    assert out_avg.shape == expected_shape_avg, f"Output shape mismatch: {out_avg.shape} != {expected_shape_avg}"

if __name__ == "__main__":
    test_multihead_gatlayer_step_by_step()