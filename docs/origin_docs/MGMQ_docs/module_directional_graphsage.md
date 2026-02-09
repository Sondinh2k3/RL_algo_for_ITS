# Mô-đun MGMQ: Hợp tác Đồ thị định hướng (Improved GraphSAGE)

Dựa trên tài liệu: *"A large-scale traffic signal control algorithm based on multi-layer graph deep reinforcement learning.pdf"* (Tao Wang et al.)

Trong kiến trúc MGMQ, tầng mạng lưới (Traffic network layer) sử dụng thuật toán **GraphSAGE cải tiến** để thực hiện sự phối hợp giữa các nút giao lân cận.

## 1. Hạt nhân GraphSAGE cải tiến
Thuật toán GraphSAGE truyền thống thường sử dụng các bộ tổng hợp trung bình (Mean) hoặc tổng (Sum), làm mất đi thông tin về cấu trúc không gian và hướng. MGMQ giải quyết vấn đề này bằng cách:

- **Bổ sung Nút làn ngõ ra (Exit Lane Nodes)**: Đồ thị mạng lưới không chỉ bao gồm các nút giao mà còn bao gồm các nút đại diện cho các hướng xuất phát (N, E, S, W).
- **Bộ tổng hợp Bi-GRU (Bidirectional GRU Aggregator)**: Thay thế các hàm tổng hợp thông thường bằng mạng nơ-ron hồi quy GRU hai chiều.
    - **Lý do**: Bi-GRU có khả năng nắm bắt mối quan hệ tuần tự giữa các đầu vào tiền nhiệm và kế nhiệm, cực kỳ phù hợp để mã hóa thông tin vị trí của bốn hướng (Bắc, Đông, Nam, Tây).

## 2. Quy trình Tổng hợp Không gian
Quy trình trích xuất đặc trưng mạng lưới tại nút giao $k$:

1.  **Lấy mẫu hàng xóm (Neighbor Sampling)**: Chỉ thu thập các nút hàng xóm có dòng xe đổ về nút giao hiện tại (Self node).
2.  **Mã hóa Hướng**: Các vector nhúng từ các hướng khác nhau được đưa vào Bi-GRU theo thứ tự không gian.
3.  **Tính toán Bi-GRU**:
    - **Forward GRU**: Xử lý thông tin ảnh hưởng theo chiều thuận.
    - **Backward GRU**: Xử lý thông tin ảnh hưởng theo chiều ngược.
4.  **Hợp nhất**: Kết quả đầu ra của Bi-GRU được nối lại để tạo thành vector nhúng mạng lưới $G_k$, đại diện cho ảnh hưởng tổng thể của các nút giao lân cận.

## 3. Ưu điểm của Bi-GRU Aggregator
- **Bảo toàn tính hướng**: Nhận biết được áp lực giao thông đến từ hướng nào (ví dụ: áp lực từ phía Bắc khác với phía Tây).
- **Khả năng mở rộng**: Mô hình có thể thích ứng với các mạng lưới giao thông có kích thước và hình dáng khác nhau nhờ cơ chế lấy mẫu và tổng hợp linh hoạt.
- **Phối hợp Zero-shot**: Việc mã hóa hướng giúp tác nhân hiểu được "ngữ cảnh không gian", từ đó có thể ra quyết định đúng ngay cả ở các khu vực mới mà nó chưa từng được huấn luyện.
