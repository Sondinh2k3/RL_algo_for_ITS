# Tổng quan Kiến trúc MGMQ (Nguồn tài liệu gốc RG)

Dựa trên tài liệu: *"A large-scale traffic signal control algorithm based on multi-layer graph deep reinforcement learning.pdf"* (Tao Wang et al.)

**MGMQ (Multi-layer Graph Mask Q-Learning)** là một thuật toán học tăng cường chiều sâu (DRL) tiên tiến, được thiết kế đặc biệt để điều khiển đèn tín hiệu giao thông trong mạng lưới đô thị quy mô lớn với khả năng chuyển đổi (transferability) mạnh mẽ.

## 1. Triết lý Thiết kế: Đồ thị Đa tầng (Multi-layer Graph)
MGMQ chia môi trường giao thông thành hai tầng đồ thị riêng biệt để trích xuất đầy đủ đặc trưng hình học và không gian:

1.  **Tầng Nút giao (Intersection-layer Graph - Lower)**:
    *   **Nút**: Các làn đường ngõ vào (entry lanes).
    *   **Cạnh**: Mối quan hệ giữa các làn (cùng nhóm đèn hoặc khác nhóm đèn).
    *   **Mục tiêu**: Trích xuất thông tin cấu trúc nội bộ của từng nút giao.
2.  **Tầng Mạng lưới (Traffic Network-layer Graph - Upper)**:
    *   **Nút**: Toàn bộ nút giao và các làn đường ngõ ra định hướng.
    *   **Cạnh**: Các liên kết vật lý (đường nối) giữa các nút giao.
    *   **Mục tiêu**: Phối hợp thông tin giữa các tác nhân lân cận để tối ưu hóa lưu lượng khu vực.

## 2. Các thành phần công nghệ lõi

### a. GAT với Multi-head Attention (Cấp độ Làn)
Sử dụng tại tầng nút giao để tính toán sự ảnh hưởng động giữa các làn đường. Cơ chế **Mask Attention** đảm bảo chỉ tính toán trên các làn có kết nối thực tế, giúp tác nhân thích ứng với các cấu trúc nút giao đa dạng (T-junction, Crossroad).

### b. Improved GraphSAGE với Bi-GRU (Cấp độ Mạng lưới)
Cải tiến thuật toán GraphSAGE truyền thống bằng cách thay thế bộ tổng hợp (aggregator) thông thường thành **GRU hai chiều (Bi-GRU)**. Điều này cho phép mô hình nắm bắt được thông tin vị trí và hướng (N, E, S, W) của các nút giao lân cận một cách chính xác hơn.

### c. Double DQN (DDQN) & Action Masking
*   **Học tập**: Sử dụng thuật toán DDQN để ước lượng giá trị Q ổn định, kết hợp với cơ chế chia sẻ tham số (parameter sharing) giữa tất cả các tác nhân.
*   **Action Masking**: Kỹ thuật then chốt cho phép MGMQ hoạt động trên bất kỳ tổ chức pha đèn nào. Các hành động không hợp lệ được gán giá trị $-\infty$ để ép tác nhân luôn chọn các pha đèn khả thi về mặt vật lý.

## 3. Khả năng Chuyển đổi Không cần Huấn luyện (Zero-shot Transfer)
Nhờ vào việc nhúng đặc trưng ở cấp độ làn đường và cơ chế Masking, MGMQ có thể được huấn luyện trên các mạng lưới giả lập và áp dụng trực tiếp vào các mạng lưới thực tế phức tạp mà không cần tái huấn luyện, ưu việt hơn hẳn các thuật toán MARL truyền thống.
