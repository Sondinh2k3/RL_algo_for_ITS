# Tài liệu Thuật toán MGMQ (Local Temporal Version)

## 1. Tổng quan (Overview)

**MGMQ (Multi-Layer Graph Masking Q-Learning)** phiên bản **Local Temporal** là một kiến trúc Deep Reinforcement Learning tiên tiến được thiết kế để điều khiển đèn tín hiệu giao thông trong môi trường Multi-Agent.

Khác với các phương pháp GNN truyền thống dựa trên đồ thị toàn cầu (Global Graph), MGMQ sử dụng kiến trúc **Local Spatio-Temporal Graph** (Đồ thị Không gian - Thời gian Cục bộ). Mỗi Agent (giao lộ) tự xây dựng một đồ thị hình sao (Star Graph) gồm chính nó và các hàng xóm trực tiếp, cho phép nó hoạt động độc lập mà vẫn phối hợp hiệu quả. Điều này giải quyết triệt để vấn đề huấn luyện phân tán (Decentralized Training) và batch shuffling trong thư viện RLlib.

---

## 2. Luồng Dữ liệu (Algorithm Flow)

Dưới đây là sơ đồ luồng dữ liệu chi tiết từ khi nhận quan sát đến khi ra quyết định (Action):

```mermaid
graph TD
    subgraph "1. Spatio-Temporal Observation"
        Obs[Input Dict] --> SelfFeat[Self Features\n(T, 48)]
        Obs --> NeighborFeat[Neighbor Features\n(K, T, 48)]
        Obs --> Mask[Neighbor Mask\n(K)]
    end

    subgraph "2. Intersection Embedding (Time-Distributed GAT)"
        SelfFeat --> GAT_Self[GAT Encoder]
        NeighborFeat --> GAT_Neighbor[GAT Encoder]
        
        GAT_Self -- lane-level attention --> SelfEmb[Self Emb\n(T, gat_dim)]
        GAT_Neighbor -- lane-level attention --> NeighborEmb[Neighbor Emb\n(K, T, gat_dim)]
    end

    subgraph "3. Local Graph Construction"
        SelfEmb --> Node0[Central Node]
        NeighborEmb --> Node1_K[Neighbor Nodes]
        Mask --> Adj[Star Adjacency Matrix\n(1+K, 1+K)]
        
        Node0 & Node1_K --> LocalGraph[Local Star Graph\n(Nodes: 1+K, Time: T)]
    end

    subgraph "4. Spatio-Temporal Aggregation"
        LocalGraph & Adj --> GraphSAGE[GraphSAGE\n(Spatial Aggregation)]
        GraphSAGE -- for each timestep --> AggregatedSeq[Aggregated Sequence\n(T, hidden)]
        AggregatedSeq --> BiGRU[Bi-GRU\n(Temporal Processing)]
        BiGRU --> NetworkEmb[Network Embedding\n(hidden_dim)]
    end

    subgraph "5. Joint Embedding & Output"
        SelfEmb -- last timestep --> InterEmb[Intersection Context]
        NetworkEmb --> NetContext[Network Context]
        
        InterEmb & NetContext --> Concat[Concatenate] --> JointEmb[Joint Embedding]
        
        JointEmb --> Actor[Policy Head\n(Action Probabilities)]
        JointEmb --> Critic[Value Head\n(State Value)]
    end
```

---

## 3. Chi tiết Thành phần (Component Details)

### 3.1. Spatio-Temporal Observation (`NeighborTemporalObservationFunction`)
Thay vì chỉ quan sát trạng thái hiện tại, mỗi Agent thu thập một "gói" dữ liệu bao gồm lịch sử của chính nó và các hàng xóm.
*   **Self Features `[T, 48]`**: Lịch sử $T$ bước thời gian của 48 đặc trưng (Density, Queue, Occupancy, Speed trên 12 làn).
*   **Neighbor Features `[K, T, 48]`**: Lịch sử của $K$ hàng xóm gần nhất (mặc định $K=4$).
*   **Neighbor Mask `[K]`**: Đánh dấu hàng xóm nào tồn tại (1) hoặc không (0).

### 3.2. Intersection Encoder (Time-Distributed GAT)
Xử lý thông tin chi tiết tại cấp độ làn đường (Lane-level) cho mỗi bước thời gian.
*   **Module**: `MultiHeadGATLayer` (chia sẻ trọng số giữa Self và Neighbors).
*   **Input**: `[batch, T, 12_lanes, 4_features]`.
*   **Mechanics**: Sử dụng 2 đồ thị con (Cooperation & Conflict) để tính toán Attention giữa các làn xe.
*   **Output**: Một vector embedding đại diện cho trạng thái giao thông của một giao lộ tại một thời điểm $t$.

### 3.3. Spatio-Temporal Aggregator (Directional GraphSAGE + BiGRU)
Đây là "bộ não" xử lý không gian và thời gian, giải quyết bài toán tầm nhìn cục bộ và bảo toàn thông tin hướng của dòng giao thông.

1.  **Directional Projection (Phân tách hướng)**:
    *   Vector trạng thái của nút giao (từ GAT) được chiếu thành 5 vector thành phần: **Self, North, East, South, West**.
    *   Điều này giúp mô hình phân biệt được thông tin đến từ các hướng khác nhau.

2.  **Topology-Aware Sampling (Ghép cặp luồng)**:
    *   Thay vì tổng hợp vô hướng, hệ thống ghép cặp vector dựa trên dòng chảy vật lý:
        *   Input Cửa Bắc $\leftarrow$ Output Cửa Nam của hàng xóm.
        *   Input Cửa Đông $\leftarrow$ Output Cửa Tây của hàng xóm.
        *   (Tương tự cho South và West).
    *   Giúp nút giao biết được áp lực giao thông đang đổ về từ hướng nào.

3.  **Bi-GRU Aggregation (Tổng hợp Không gian - Thời gian)**:
    *   **Spatial Aggregation**: Tại mỗi bước thời gian, chuỗi vector 5 hướng `[N, E, S, W, Self]` được đưa qua Bi-GRU hai chiều để tổng hợp ngữ cảnh không gian topo.
    *   **Temporal Aggregation**: Chuỗi các vector kết quả theo thời gian $t=1...T$ tiếp tục được xử lý để nắm bắt xu hướng (ví dụ: tắc nghẽn đang lan truyền từ hướng Bắc xuống).

### 3.4. Action Space & Execution
*   **Action**: Continuous vector (softmax ratio), đại diện cho **tỷ lệ thời gian xanh** phân bổ cho các pha.
*   **Execution**:
    1.  Model xuất ra `ratio` (ví dụ: `[0.4, 0.3, 0.3]`).
    2.  Chuẩn hóa tổng bằng 1.0.
    3.  Nhân với `total_green_time` chu kỳ để ra thời gian thực (giây).
    4.  Apply vào SUMO theo thứ tự pha cố định (được chuẩn hóa bởi `PhaseStandardizer`).

---

## 4. Cơ chế Phần thưởng (Reward Function)

Hàm mục tiêu được thiết kế gồm 2 thành phần chính để cân bằng giữa giảm ùn tắc và tăng thông lượng:

### 4.1. Penalty for Halting (`halt-veh-by-detectors`)
*   **Mục tiêu**: Phạt nặng khi có xe dừng chờ (ùn tắc).
*   **Công thức**: `Reward = -3.0 * (Total_Halting / Max_Capacity)`
*   **Dải giá trị**: `[-3.0, 0.0]`
    *   `0.0`: Không có xe dừng (Lý tưởng).
    *   `-3.0`: Tắc cứng toàn bộ.

### 4.2. Outflow Efficiency (`diff-departed-veh`)
*   **Mục tiêu**: Khuyến khích xả xe ra khỏi giao lộ (Thông lượng).
*   **Công thức**: `Reward = (Departed_Vehicles / Initial_Vehicles) * 3.0`
*   **Dải giá trị**: `[0.0, 3.0]`
    *   `3.0`: Giải tỏa được 100% số xe đang có (Hiệu quả cao).
    *   `0.0`: Không giải tỏa được xe nào.

**Tổng Reward**: `Total = 0.5 * Penalty + 0.5 * Efficiency` (Range `[-1.5, 1.5]`).

---

## 5. Cấu hình (Configuration)

Các tham số quan trọng trong `src/config/model_config.yml`:

| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `use_local_gnn` | `False` | Kích hoạt chế độ Local Temporal GNN |
| `max_neighbors` | `4` | Số lượng hàng xóm tối đa trong quan sát |
| `window_size` | `4` | Độ dài cửa sổ lịch sử (T) |
| `gat_hidden_dim` | `256` | Kích thước ẩn của GAT |
| `graphsage_hidden_dim` | `256` | Kích thước ẩn của GraphSAGE |
| `gru_hidden_dim` | `128` | Kích thước ẩn của Bi-GRU |
