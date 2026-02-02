# Tài liệu Thuật toán MGMQ (Directional GraphSAGE Version)

## 1. Tổng quan (Overview)

**MGMQ (Multi-Layer Graph Masking Q-Learning)** là một kiến trúc Deep Reinforcement Learning tiên tiến được thiết kế để điều khiển đèn tín hiệu giao thông trong môi trường Multi-Agent.

Kiến trúc chính sử dụng **Directional GraphSAGE** với **Bi-GRU** để tổng hợp thông tin theo hướng không gian (N, E, S, W), cho phép mô hình hiểu được luồng giao thông đang đến từ hướng nào. Mỗi Agent (giao lộ) xử lý thông tin từ các hàng xóm theo 4 hướng, giúp phối hợp điều khiển hiệu quả.

> [!IMPORTANT]
> **BiGRU trong MGMQ được dùng để tổng hợp thông tin theo 4 HƯỚNG KHÔNG GIAN (North, East, South, West), KHÔNG PHẢI để xử lý chuỗi thời gian.** Cấu hình (`history_length: 1`) không sử dụng temporal features.

---

## 2. Luồng Dữ liệu (Algorithm Flow)

Dưới đây là sơ đồ luồng dữ liệu chi tiết từ khi nhận quan sát đến khi ra quyết định (Action):

```mermaid
graph TD
    subgraph "1. Observation (Current Timestep)"
        Obs[Input: obs] --> Reshape[Reshape to Lanes<br/>(12 lanes, 4 features)]
    end

    subgraph "2. Intersection Embedding (GAT Layer)"
        Reshape --> GAT[Dual-Stream GAT]
        GAT -- cooperation & conflict attention --> IntersectionEmb[Intersection Embedding<br/>(num_agents, gat_dim)]
    end

    subgraph "3. Network-level GraphSAGE"
        IntersectionEmb --> DirProj[Directional Projections<br/>(Self, N, E, S, W)]
        DirProj --> NeighborExchange[Topology-Aware<br/>Neighbor Exchange]
    end

    subgraph "4. Directional Aggregation (Bi-GRU over 4 Directions)"
        NeighborExchange --> DirSeq[Direction Sequence<br/>(in_N, in_E, in_S, in_W)]
        DirSeq --> BiGRU[Bi-GRU<br/>(4 directions as sequence)]
        BiGRU --> NeighborContext[Neighbor Context]
        NeighborContext --> Combine[Combine with Self]
        Combine --> NetworkEmb[Network Embedding<br/>(hidden_dim)]
    end

    subgraph "5. Joint Embedding & Output"
        IntersectionEmb -- agent embedding --> AgentEmb[Agent Intersection Emb]
        NetworkEmb --> NetContext[Network Context]
        
        AgentEmb & NetContext --> Concat[Concatenate] --> JointEmb[Joint Embedding]
        
        JointEmb --> Actor[Policy Head<br/>(Dirichlet Distribution)]
        JointEmb --> Critic[Value Head<br/>(State Value)]
    end
```

---

## 3. Chi tiết Thành phần (Component Details)

### 3.1. Observation (`StandardIntersectionObservationFunction`)

Với cấu hình hiện tại (`history_length: 1`), mỗi Agent quan sát trạng thái **tại thời điểm hiện tại**:

| Component | Shape | Mô tả |
|-----------|-------|-------|
| `observation` | `[48]` | 48 đặc trưng (Density, Queue, Occupancy, Speed × 12 làn) |

**48 features** được tính từ 12 làn đường (3 làn × 4 hướng N/E/S/W), mỗi làn có 4 metrics:
- **Density**: Mật độ giao thông chuẩn hóa `[0,1]`
- **Queue**: Độ dài hàng đợi chuẩn hóa `[0,1]`
- **Occupancy**: Độ chiếm dụng chuẩn hóa `[0,1]`
- **Speed**: Tốc độ trung bình chuẩn hóa `[0,1]`

> [!NOTE]
> Nếu kích hoạt `local_gnn.enabled: true`, sẽ sử dụng `NeighborTemporalObservationFunction` với Dict observation bao gồm `self_features`, `neighbor_features`, và `neighbor_mask`. Khi đó, `NeighborGraphSAGE_BiGRU` sẽ được dùng để tổng hợp thông tin từ hàng xóm theo **không gian** (không phải thời gian).

### 3.2. Intersection Encoder (`DualStreamGATLayer`)

Xử lý thông tin chi tiết tại cấp độ làn đường (Lane-level) cho mỗi bước thời gian.

**Module**: `DualStreamGATLayer` trong `src/models/gat_layer.py`

**Quy trình 4 bước:**
1. **Linear Transformation**: Input → Latent space (`in_features` → `hidden_dim`)
2. **Dual-Stream Attention**: 
   - **Same-phase Attention** (Cooperation Matrix): Các làn cùng pha đèn
   - **Diff-phase Attention** (Conflict Matrix): Các làn xung đột
3. **Multi-head Aggregation**: Kết hợp output từ nhiều attention heads
4. **Output Projection**: Tạo embedding cuối cùng cho mỗi giao lộ

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `gat_hidden_dim` | 128 | Dimension ẩn của GAT |
| `gat_output_dim` | 64 | Dimension output mỗi head |
| `gat_num_heads` | 4 | Số attention heads |

**Input**: `[batch, T, 12_lanes, 4_features]`  
**Output**: `[batch, T, gat_output_dim × gat_num_heads]` = `[batch, T, 256]`

### 3.3. Directional Aggregator (`DirectionalGraphSAGE` hay `GraphSAGE_BiGRU`)

Đây là "bộ não" xử lý thông tin không gian theo hướng (Directional/Spatial Aggregation), giải quyết bài toán tầm nhìn cục bộ và bảo toàn thông tin hướng của dòng giao thông.

> [!IMPORTANT]
> **BiGRU trong DirectionalGraphSAGE được dùng để tổng hợp thông tin theo 4 HƯỚNG KHÔNG GIAN (N, E, S, W), KHÔNG PHẢI để xử lý thông tin thời gian.**

**1. Directional Projection (Phân tách hướng)**:
- Vector trạng thái của nút giao (từ GAT) được chiếu thành 5 vector thành phần: **Self, North, East, South, West**.
- Mỗi hướng có một linear projection riêng biệt (`proj_self`, `proj_north`, `proj_east`, `proj_south`, `proj_west`).
- Điều này giúp mô hình phân biệt được thông tin đến từ các hướng khác nhau.

**2. Topology-Aware Neighbor Exchange (Ghép cặp luồng)**:
- Thay vì tổng hợp vô hướng, hệ thống ghép cặp vector dựa trên dòng chảy vật lý:
  - `in_north` ← `g_south` của hàng xóm phía Bắc (xe từ Bắc đi vào = xe ra từ cửa Nam của hàng xóm Bắc)
  - `in_east` ← `g_west` của hàng xóm phía Đông
  - `in_south` ← `g_north` của hàng xóm phía Nam
  - `in_west` ← `g_east` của hàng xóm phía Tây
- Giúp nút giao biết được áp lực giao thông đang đổ về từ hướng nào.

**3. Bi-GRU Directional Aggregation (Tổng hợp theo hướng)**:
- Chuỗi 4 vector hướng `[in_north, in_east, in_south, in_west]` được đưa vào Bi-GRU.
- BiGRU xử lý chuỗi này như một **sequence có thứ tự theo không gian** (N→E→S→W), không phải theo thời gian.
- **Forward GRU**: Học pattern từ North → East → South → West
- **Backward GRU**: Học pattern từ West → South → East → North
- Kết hợp 2 hidden states cuối để tạo **Neighbor Context** cho mỗi node.

**4. Output Combination**:
- Kết hợp `g_self` (thông tin tự thân) với `h_neighbors` (neighbor context từ BiGRU)
- Đưa qua `output_proj` để tạo embedding cuối cùng cho mỗi node

```python
# Code minh họa trong DirectionalGraphSAGE.forward():
seq_tensor = torch.stack([in_north, in_east, in_south, in_west], dim=2)  # [B, N, 4, H]
# BiGRU xử lý sequence 4 phần tử = 4 hướng, KHÔNG PHẢI T timesteps
_, h_n = self.bigru(seq_flat)  # h_n: [2, B*N, H] - 2 directions của BiGRU
```

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `graphsage_hidden_dim` | 128 | Dimension ẩn của GraphSAGE projections |
| `gru_hidden_dim` | 64 | Dimension ẩn của Bi-GRU (xử lý 4 hướng) |

### 3.4. Temporal Extension (Optional) - `TemporalGraphSAGE_BiGRU`

Khi `history_length > 1`, hệ thống có thể sử dụng `TemporalGraphSAGE_BiGRU` để kết hợp cả xử lý không gian và thời gian. Module này có **2 BiGRU riêng biệt**:

**1. Spatial BiGRU** (trong `DirectionalGraphSAGE`):
- Xử lý sequence 4 hướng `[N, E, S, W]` **tại mỗi timestep**
- Output: `[Batch, T, N, Hidden]`

**2. Temporal BiGRU** (`temporal_bigru`):
- Xử lý sequence T timesteps **sau khi đã aggregation không gian**
- Input: `[Batch*N, T, Hidden]`
- Output: Lấy hidden state cuối cùng

```python
# TemporalGraphSAGE_BiGRU.forward():
# Step 1: Spatial Processing (cho mỗi timestep)
spatial_out = self.spatial_layer(h_flat, adj_exp)  # DirectionalGraphSAGE

# Step 2: Temporal Processing (sau khi có spatial embeddings)
t_out, _ = self.temporal_bigru(spatial_seq)  # BiGRU over T timesteps
```

> [!NOTE]
> **Cấu hình hiện tại** (`history_length: 1`) **không sử dụng Temporal BiGRU**. Chỉ có Directional BiGRU được dùng để tổng hợp thông tin từ 4 hướng.

### 3.5. Action Space & Execution

**Action**: **Dirichlet Distribution** output, đại diện cho **tỷ lệ thời gian xanh** phân bổ cho các pha.

> [!IMPORTANT]
> Sử dụng **Dirichlet Distribution** thay vì Gaussian để đảm bảo:
> - Output tự động nằm trong simplex (tổng = 1.0)
> - Không cần softmax normalize
> - Tránh vấn đề Scale Ambiguity & Vanishing Gradient

**Execution Flow:**
1. Model xuất ra `ratio` từ Dirichlet (ví dụ: `[0.4, 0.3, 0.2, 0.1]`)
2. `PhaseStandardizer.standardize_action()` chuyển đổi từ 4 pha chuẩn → pha thực tế
3. `TrafficSignal._get_green_time_from_ratio()`:
   - Áp dụng Action Masking cho các pha không tồn tại
   - Phân bổ `min_green` cho mỗi pha
   - Phân phối thời gian còn lại theo tỷ lệ
4. Apply vào SUMO qua `data_provider.set_traffic_light_phase()`

---

## 4. Cơ chế Phần thưởng (Reward Function)

Hệ thống reward được cấu hình trong `model_config.yml` với các hàm sau:

### 4.1. Penalty for Halting (`halt-veh-by-detectors`)

**Mục tiêu**: Phạt khi có xe dừng chờ (ùn tắc).

**Công thức**: 
$$R_{halt} = -3.0 \times \frac{\text{Aggregated Halting Vehicles}}{\text{Max Capacity}}$$

**Dải giá trị**: `[-3.0, 0.0]`
- `0.0`: Không có xe dừng (Lý tưởng)
- `-3.0`: Tắc cứng toàn bộ capacity

### 4.2. Outflow Efficiency (`diff-departed-veh`)

**Mục tiêu**: Khuyến khích xả xe ra khỏi giao lộ (Thông lượng).

**Công thức**:
$$R_{depart} = \frac{\text{Departed Vehicles This Cycle}}{\text{Initial Vehicles This Cycle}} \times 3.0$$

**Dải giá trị**: `[0.0, 3.0]`
- `3.0`: Giải tỏa được 100% số xe (Hiệu quả cao)
- `0.0`: Không giải tỏa được xe nào

> [!NOTE]
> Có threshold `MIN_VEHICLES_THRESHOLD = 1.0` để tránh spurious rewards khi lưu lượng thấp.

### 4.3. Occupancy Penalty (`occupancy`)

**Mục tiêu**: Phạt độ chiếm dụng cao trên các detector.

**Công thức**:
$$R_{occ} = -3.0 \times \text{Aggregated Occupancy}$$

**Dải giá trị**: `[-3.0, 0.0]`

### 4.4. Các hàm reward khác (có sẵn nhưng disabled mặc định)

| Reward Function | Công thức | Range | Mục tiêu |
|-----------------|-----------|-------|----------|
| `average-speed` | `(avg_speed × 6.0) - 3.0` | `[-3, 3]` | Tối đa hóa tốc độ |
| `queue` | `-3.0 × (queued / max_veh)` | `[-3, 0]` | Giảm hàng đợi |
| `pressure` | `-3.0 × pressure` | `[-3, 3]` | Cân bằng lưu lượng vào/ra |
| `diff-waiting-time` | Chênh lệch waiting time | `[-3, 3]` | Giảm thời gian chờ |
| `teleport-penalty` | `-3.0 × teleport_ratio` | `[-3, 0]` | Phạt xe bị teleport do kẹt quá lâu |

**Tổng Reward**: `Total = Σ(weight_i × reward_i)` với weights mặc định bằng nhau.

---

## 5. Cơ chế Sampling & Aggregation

### 5.1. Detector History Sampling

Dữ liệu được thu thập định kỳ mỗi `sampling_interval_s = 10s`:

```
Cycle (delta_time = 90s)
|--- sample 1 (t=10s) ---|--- sample 2 (t=20s) ---|--- ... ---|--- sample 9 (t=90s) ---|
```

Mỗi sample lưu 4 metrics: `density`, `queue`, `occupancy`, `speed` cho mỗi detector.

### 5.2. Aggregation Functions

Khi tính reward cuối cycle, các mẫu được aggregate:

| Metric | Aggregation | Lý do |
|--------|-------------|-------|
| Halting Vehicles | **Mean** | Phản ánh trạng thái trung bình |
| Queue Length | **Mean** | Ổn định hơn max |
| Average Speed | **Mean** | Tốc độ đại diện |
| Waiting Time | **Sum** | Tích lũy thời gian chờ |
| Occupancy | **Mean** | Độ bão hòa trung bình |

---

## 6. Cấu hình (Configuration)

Các tham số quan trọng trong `src/config/model_config.yml`:

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `history_length` | `1` | Độ dài cửa sổ lịch sử (T). **T=1 → Không dùng Temporal BiGRU** |
| `local_gnn.enabled` | `false` | Kích hoạt chế độ Local GNN (Star Graph với neighbors) |
| `local_gnn.max_neighbors` | `4` | Số lượng hàng xóm tối đa (K) |
| `local_gnn.obs_dim` | `48` | 4 features × 12 detectors |
| `mgmq.gat.hidden_dim` | `128` | Kích thước ẩn của GAT |
| `mgmq.gat.output_dim` | `64` | Kích thước output mỗi head |
| `mgmq.gat.num_heads` | `4` | Số attention heads |
| `mgmq.graphsage.hidden_dim` | `128` | Kích thước ẩn của GraphSAGE projections |
| `mgmq.gru.hidden_dim` | `64` | Kích thước ẩn của Bi-GRU (xử lý 4 hướng) |
| `mgmq.dropout` | `0.3` | Dropout rate |

### 6.1. Cấu hình hiện tại

Với `history_length: 1` và `local_gnn.enabled: false`:
- Sử dụng **MGMQEncoder** với **DirectionalGraphSAGE** (GraphSAGE_BiGRU)
- BiGRU chỉ xử lý **4 hướng không gian** (N, E, S, W)
- **Không có** temporal processing

### 6.2. Kích hoạt Temporal Features (Optional)

Để sử dụng cả spatial và temporal processing:
```yaml
mgmq:
  history_length: 5  # T > 1 để kích hoạt TemporalGraphSAGE_BiGRU
```

Khi `history_length > 1`:
- Model sẽ sử dụng **TemporalGraphSAGE_BiGRU**
- Có **2 BiGRU**: một cho spatial (4 hướng), một cho temporal (T timesteps)

---

## 7. File Structure

```
src/
├── models/
│   ├── mgmq_model.py          # MGMQEncoder, LocalTemporalMGMQEncoder, MGMQTorchModel
│   ├── gat_layer.py           # DualStreamGATLayer, MultiHeadGATLayer
│   ├── graphsage_bigru.py     # DirectionalGraphSAGE, TemporalGraphSAGE_BiGRU
│   └── dirichlet_distribution.py  # Dirichlet action distribution
├── environment/
│   └── drl_algo/
│       ├── traffic_signal.py  # TrafficSignal (reward, observation, action execution)
│       └── observations.py    # NeighborTemporalObservationFunction
└── preprocessing/
    ├── standardizer.py        # IntersectionStandardizer (GPI)
    └── frap.py                # PhaseStandardizer (FRAP)
```
