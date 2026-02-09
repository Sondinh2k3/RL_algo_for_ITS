# Mô-đun MGMQ: Tích chập Chú ý Đồ thị cấp Làn đường (Lane-level GAT)

Dựa trên tài liệu: *"A large-scale traffic signal control algorithm based on multi-layer graph deep reinforcement learning.pdf"* (Tao Wang et al.)

Trong mô hình MGMQ, tầng xử lý nút giao (Intersection layer) sử dụng thuật toán **Graph Attention Network (GAT)** để mô hình hóa sự tương tác động giữa các làn đường đi vào.

## 1. Cấu trúc Đồ thị Nút giao
Khác với các phương pháp coi cả nút giao là một thực thể, MGMQ chia nhỏ nút giao thành các thực thể làn đường:
- **Nút (Nodes)**: Mỗi làn đường là một nút, mang vector trạng thái $S_i^t$ (WVN, RVN, VWT, FVT, P).
- **Cạnh (Edges)**: Được chia thành 2 đồ thị con (subgraphs) dựa trên quan hệ pha đèn:
    - **Sub-graph $g_1$ (Cạnh $E_1$)**: Liên kết giữa các làn đường nằm trong cùng một nhóm pha đèn (cùng Xanh hoặc cùng Đỏ).
    - **Sub-graph $g_2$ (Cạnh $E_2$)**: Liên kết giữa các làn đường thuộc các nhóm pha đèn khác nhau (có xung đột hoặc không cùng trạng thái).

## 2. Cơ chế Xử lý: Multi-head Mask Attention
Mô hình thực hiện tính toán song song trên cả hai đồ thị con $g_1$ và $g_2$:

### Trọng số Chú ý (Attention Score)
Hệ số chú ý $\alpha_{i,j}$ giữa làn $i$ và $j$ được tính toán bằng cách trích xuất mức độ quan trọng:
- Sử dụng hàm **LeakyReLU** (negative slope = 0.05) thay cho ReLU thông thường để tránh hiện tượng "chết" các nơ-ron.
- **Mask Attention**: Chỉ tập trung vào tập hợp các hàng xóm trực tiếp liên kết với làn đường đó trong đồ thị, loại bỏ nhiễu từ các làn không liên quan.

### Tổng hợp Đặc trưng
Sử dụng nhiều đầu chú ý (K-heads) để nắm bắt các khía cạnh ảnh hưởng khác nhau (ví dụ: một đầu chú ý đến hàng đợi, một đầu chú ý đến thời gian chờ).

$$h'_i = \sigma \left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in N_i} \alpha_{ij}^k W^k h_j \right)$$

## 3. Cập nhật và Hợp nhất
Sau khi có các vector đặc trưng từ hai đồ thị con ($h_i^1$ và $h_i^2$), mô hình thực hiện:
1. **Update**: Cập nhật trạng thái mới cho mỗi nút (làn đường).
2. **Concatenate**: Nối tất cả các vector của mọi làn đường trong nút giao lại thành một vector nhúng nút giao (Intersection Embedding) $g_k$.

Cơ chế này giúp MGMQ nhận diện chính xác luồng giao thông nào đang chịu áp lực lớn nhất và mối quan hệ ràng buộc giữa các luồng đó để đưa ra quyết định pha đèn hợp lý nhất.
