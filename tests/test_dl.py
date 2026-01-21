import torch
from src.models.gat_layer import GATLayer, get_lane_conflict_matrix

def print_tensor(name, tensor):
    print(f"{name}: shape={tensor.shape}\n{tensor}\n")

def test_gatlayer_attention_step_by_step():
    # 1. Khởi tạo GATLayer
    in_features = 4
    out_features = 3
    num_nodes = 5
    torch.manual_seed(42)
    gat = GATLayer(in_features, out_features, dropout=0.0, alpha=0.2, concat=True)

    # 2. Tạo dữ liệu đầu vào: node features và adjacency matrix
    h = torch.randn(num_nodes, in_features)  # [N, in_features]
    adj = torch.randint(0, 2, (num_nodes, num_nodes))  # [N, N], random 0/1 adjacency

    print_tensor("Input node features h", h)
    print_tensor("Adjacency matrix adj", adj)

    # 3. Thêm batch dimension (GATLayer sẽ tự xử lý, nhưng ta show rõ)
    h_batch = h.unsqueeze(0)  # [1, N, in_features]
    adj_batch = adj.unsqueeze(0)  # [1, N, N]

    print_tensor("h_batch", h_batch)
    print_tensor("Adjacency matrix adj_batch", adj_batch)


    # 4. Linear transformation: h' = h * W
    Wh = torch.matmul(h_batch, gat.W)  # [1, N, out_features]
    print_tensor("After linear transformation Wh", Wh)

    # 5. Tính attention logits cho từng cặp node
    Wh1 = Wh.unsqueeze(2)  # [1, N, 1, out_features]
    Wh2 = Wh.unsqueeze(1)  # [1, 1, N, out_features]
    all_combinations = torch.cat([Wh1.repeat(1, 1, num_nodes, 1), Wh2.repeat(1, num_nodes, 1, 1)], dim=-1)
    print_tensor("All combinations [Wh_i || Wh_j]", all_combinations[0])  # [N, N, 2*out_features]

    e = gat.leakyrelu(torch.matmul(all_combinations, gat.a).squeeze(-1))  # [1, N, N]
    print_tensor("Raw attention logits e", e[0])

    # 6. Mask attention theo adjacency
    zero_vec = -9e15 * torch.ones_like(e)
    attention = torch.where(adj_batch > 0, e, zero_vec)
    print_tensor("Masked attention logits", attention[0])

    # 7. Chuẩn hóa attention bằng softmax
    attention_softmax = torch.softmax(attention, dim=-1)
    print_tensor("Normalized attention (softmax)", attention_softmax[0])

    # 8. Áp dụng attention để tổng hợp đặc trưng mới
    h_out = torch.bmm(attention_softmax, Wh)
    print_tensor("Output features h_out (before activation)", h_out[0])

    # 9. Kích hoạt ELU nếu concat=True
    h_out_activated = torch.nn.functional.elu(h_out)
    print_tensor("Output features h_out (after ELU)", h_out_activated[0])

if __name__ == "__main__":
    test_gatlayer_attention_step_by_step()