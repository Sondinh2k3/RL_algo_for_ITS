# Luồng Thuật toán MGMQ (Multi-layer Graph Mask Q-Learning)

Tài liệu này mô tả quy trình từng bước từ việc thu thập dữ liệu tại các làn đường đến khi đưa ra quyết định chuyển pha đèn dựa trên mô hình MGMQ.

## 1. Thu thập Quan sát (Lane-level Observation)
Lõi của MGMQ nằm ở việc xử lý dữ liệu ở cấp độ làn đường. Mỗi làn đường $i$ cung cấp một vector trạng thái 5 chiều $S_i^t$:
- **WVN**: Số lượng xe đang dừng chờ.
- **RVN**: Tổng số lượng xe (đang chạy + dừng).
- **VWT**: Thời gian chờ tích lũy của toàn bộ xe trên làn.
- **FVT**: Thời gian chờ của xe đầu tiên trong hàng đợi.
- **P**: Trạng thái đèn hiện tại của làn (0: Đỏ, 1: Xanh).

## 2. Các bước xử lý Đặc trưng (Embedding Pipeline)

Quy trình biến đổi thông tin diễn ra qua 2 giai đoạn chính:

### Bước A: Nhúng đặc trưng Nút giao (Intersection Embedding)
1. **Phân tách Đồ thị**: Tách các làn đường thành hai nhóm cạnh: $E1$ (các làn cùng pha) và $E2$ (các làn khác pha).
2. **GAT Layer**: Áp dụng Graph Attention Network với Multi-head Attention để tính toán sự ảnh hưởng giữa các làn.
3. **Tổng hợp**: Kết quả là một vector nhúng $g_k$ duy nhất đại diện cho toàn bộ trạng thái của nút giao $k$.

### Bước B: Nhúng đặc trưng Mạng lưới (Network Embedding)
1. **Phối hợp Hướng**: Sử dụng bốn loại nút ngõ ra đại diện cho các hướng N, E, S, W.
2. **Bi-GRU Aggregator**: Sử dụng bộ tổng hợp GRU hai chiều thay cho GraphSAGE truyền thống để trích xuất đặc trưng vị trí từ các nút giao lân cận.
3. **Hợp nhất**: Kết hợp thông tin từ chính nút giao và thông tin từ hàng xóm thu được qua Bi-GRU tạo thành vector $G_k$.

## 3. Ra quyết định và Huấn luyện (DQN & Action Masking)
- **Hợp nhất cuối**: Nối hai vector ($g_k$ và $G_k$) thành vector trạng thái tổng hợp $z_k$.
- **Ước lượng Q-value**: $z_k$ được đưa vào mạng DDQN (Double DQN) để tính giá trị Q cho tất cả các pha đèn.
- **Action Masking**:
    - Kiểm tra danh sách các pha hợp lệ tại nút giao hiện tại.
    - Áp dụng mặt nạ: gán $Q(s, a) = -\infty$ cho các hành động không thuộc danh sách hợp lệ.
- **Lựa chọn**: Sử dụng $\epsilon$-greedy để chọn pha đèn tối ưu nhất hoặc thực hiện khám phá.

---

## 4. Sơ đồ thực thi

```mermaid
graph LR
    subgraph Input
      L[Lane Features: WVN, RVN, VWT, FVT, P]
    end
    
    subgraph Layer1: Intersection
      GAT[GAT with Multi-head Attention]
      E1E2[E1/E2 Subgraph Split]
      GAT --> IE[Intersection Embedding g_k]
      E1E2 --> GAT
    end
    
    subgraph Layer2: Network
      SAGE[Improved GraphSAGE]
      BIGRU[Bi-GRU Aggregator]
      SAGE --> NE[Network Embedding G_k]
      BIGRU --> SAGE
    end
    
    IE -- concat -- Z[Joint Embedding z_k]
    NE -- concat -- Z
    
    Z --> DDQN[Double DQN Head]
    DDQN --> MASK[Action Masking: -inf]
    MASK --> Action[Optimal Signal Phase]
```
